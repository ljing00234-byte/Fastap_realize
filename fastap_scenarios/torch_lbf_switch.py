"""Device-resident LBF rollouts with within-episode changes of the teammate group."""

import torch
from torch.nn import functional as F

from .torch_foraging import TorchForaging
from .torch_lbf_ego import _EgoSpec
from .torch_pool_ego import TorchPoolEgo


class _OpenForaging(TorchForaging):
    """Use one extra, off-board cell for the optional fourth player's absence."""

    def __init__(self, batch_size, device, seed):
        super().__init__(batch_size, device, seed, n_agents=4)
        self.absent_cell = self._coordinates.size(0)
        self._coordinates = F.pad(self._coordinates, (0, 0, 0, 1), value=-1)
        self._neighbors = F.pad(self._neighbors, (0, 0, 0, 1))
        self._valid_move = F.pad(self._valid_move, (0, 0, 0, 1), value=False)
        self._local = F.pad(self._local, (0, 0, 0, 1, 0, 1), value=-1)
        self._distance = F.pad(self._distance, (0, 1, 0, 1), value=self.absent_cell)

    def load_open_board(self, board, present, fourth_levels):
        positions = torch.as_tensor(board['positions'], device=self.device)
        levels = torch.as_tensor(board['levels'], device=self.device)
        if positions.ndim != 3 or positions.size(1) not in (3, 4):
            raise ValueError("Expected three or four physical players in the initial snapshot")
        if positions.size(1) == 3:
            torch._assert_async((~present).all(), "A three-player snapshot needs one initial teammate")
            positions = F.pad(positions, (0, 0, 0, 1))
            levels = F.pad(levels, (0, 1))
        if fourth_levels is None:
            latent = torch.randint(1, 4, (self.batch_size,), device=self.device, generator=self.generator)
            fourth_levels = torch.where(levels[:, 3] > 0, levels[:, 3], latent)
        self.fourth_levels = torch.as_tensor(fourth_levels, dtype=torch.long, device=self.device).clone()
        if self.fourth_levels.shape != (self.batch_size,):
            raise ValueError("Expected one fourth-player level per board")
        torch._assert_async(((self.fourth_levels >= 1) & (self.fourth_levels <= 3)).all(),
                            "Expected fourth-player levels in 1..3")
        torch._assert_async(((~present) | (levels[:, 3] == self.fourth_levels)).all(),
                            "The starting fourth player's level must match its stored level")
        # TorchForaging encodes row and column as row*6 + column. (6, 0)
        # selects the off-board entry, which is observed and reported as (-1,-1,0).
        positions = positions.clone()
        levels = levels.clone()
        positions[:, 3] = torch.where(present[:, None], positions[:, 3], positions.new_tensor([6, 0]))
        levels[:, 3] = torch.where(present, self.fourth_levels, 0)
        self.load_snapshot(**{**board, 'positions': positions, 'levels': levels})

    def set_fourth_player(self, present, spawn_ranks):
        if spawn_ranks is None:
            spawn_ranks = self._random(self.batch_size, self.absent_cell)
        if (spawn_ranks.shape != self._field.shape or spawn_ranks.device != self.device
                or not spawn_ranks.is_floating_point()):
            raise ValueError("Provide floating spawn ranks [batch,36] on the world device")
        occupied = (self._field > 0).scatter(1, self._cells[:, :3], True)
        arrivals = present & (self._cells[:, 3] == self.absent_cell)
        new_cells = spawn_ranks.masked_fill(occupied, torch.inf).argmin(1)
        self._cells[:, 3] = torch.where(arrivals, new_cells,
                                        torch.where(present, self._cells[:, 3], self.absent_cell))
        self.levels[:, 3] = torch.where(present, self.fourth_levels, 0)
        self._update_status()


class TorchSwitchingLBFEgo(_EgoSpec, TorchPoolEgo):
    """Two persistent egos; a row's ordered group is [policy, policy or -1].

    Load a paired initial snapshot, then apply a roster before reading the
    observations for its first action. Egos' histories are owned by the caller.
    """

    def __init__(self, batch_size, device, *, teammates, seed=0):
        super().__init__(_OpenForaging(batch_size, device, seed), teammates, seed + 1)
        self._initial_hidden = torch.cat([mate.model.init_hidden() for mate in self.teammates])

    def _ids(self, ids):
        ids = torch.as_tensor(ids, device=self.device)
        if ids.dtype != torch.long or ids.shape != (self.batch_size, 2):
            raise ValueError("Expected two integer group entries per board")
        first, second = ids.unbind(1)
        valid = ((first >= 0) & (first < len(self.teammates)) & (second >= -1)
                 & (second < len(self.teammates)) & (first != second))
        torch._assert_async(valid.all(), "Invalid group: expected distinct policy IDs, second may be -1")
        return ids

    def reset(self, layout="random", *, teammate_ids=None):
        if layout != "random":
            raise ValueError("Switching evaluation uses random starts or explicit paired snapshots")
        if teammate_ids is None:
            size = (self.batch_size,)
            first = torch.randint(len(self.teammates), size, generator=self.generator, device=self.device)
            second = torch.randint(len(self.teammates) - 1, size, generator=self.generator, device=self.device)
            pair = torch.randint(2, size, generator=self.generator, device=self.device).bool()
            teammate_ids = torch.stack((first, torch.where(pair, second + (second >= first), -1)), dim=1)
        ids = self._ids(teammate_ids)
        self.game.reset(layout)
        board = self.game.snapshot()
        # With one teammate all three starting levels count towards food difficulty.
        food_level = torch.where(ids[:, 1] >= 0, board['levels'].sort(1).values[:, :3].sum(1),
                                 board['levels'][:, :3].sum(1))
        board['field'] = (board['field'] > 0).long() * food_level[:, None, None]
        board['initial_food_total'] = board['field'].sum((1, 2))
        self.load_snapshot(teammate_ids=ids, fourth_levels=board['levels'][:, 3], **board)

    def load_snapshot(self, *, teammate_ids, fourth_levels=None, **board):
        ids = self._ids(teammate_ids)
        self.game.load_open_board(board, ids[:, 1] >= 0, fourth_levels)
        self.teammate_ids = ids.clone()
        self._flat_ids = self.teammate_ids.clamp_min(0).flatten()
        self.sample_counts.scatter_add_(0, self._flat_ids, (ids >= 0).long().flatten())
        self._hidden = self._initial_hidden[self._flat_ids]
        self._previous = self._hidden.new_zeros(self._flat_ids.numel(), self.n_actions)
        self._all_obs = None
        self.last_joint_actions = None

    def apply_teammates(self, teammate_ids, spawn_ranks=None):
        """Redraw a group before its next decision, preserving finished rows.

        An identical ordered group retains both controllers. On a group change
        both frozen policies start fresh, matching OpenTeamEnv._sample_group.
        The optional fourth body appears on an unoccupied square or leaves;
        an already present body keeps its position even when its policy changes.
        """
        ids = self._ids(teammate_ids)
        active = ~(self.terminated | self.truncated)
        ids = torch.where(active[:, None], ids, self.teammate_ids)
        changed = (ids != self.teammate_ids).any(1)
        self.game.set_fourth_player(ids[:, 1] >= 0, spawn_ranks)
        self.teammate_ids = ids
        self._flat_ids = ids.clamp_min(0).flatten()
        changed_slots = changed[:, None].expand(-1, 2).reshape(-1, 1)
        self._hidden = torch.where(changed_slots, self._initial_hidden[self._flat_ids], self._hidden)
        self._previous = torch.where(changed_slots, 0, self._previous)
        self._all_obs = None
        return changed
