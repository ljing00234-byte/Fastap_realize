"""Batched CN with two persistent egos and one or two changing teammate policies."""

import torch
from torch.nn import functional as F

from .torch_cooperative_navigation import TorchCooperativeNavigation
from .torch_pool_ego import TorchPoolEgo


class TorchSwitchingCNEgo(TorchPoolEgo):
    scene_key = TorchCooperativeNavigation.scene_key
    n_actions = TorchCooperativeNavigation.n_actions
    obs_shape = TorchCooperativeNavigation.obs_shape
    state_shape = TorchCooperativeNavigation.state_shape
    episode_limit = TorchCooperativeNavigation.episode_limit

    def __init__(self, batch_size, device, *, teammates, seed=0):
        game = TorchCooperativeNavigation(batch_size, device, seed, n_agents=4)
        super().__init__(game, teammates, seed + 1)
        self._initial_hidden = torch.cat([teammate.model.init_hidden() for teammate in self.teammates])

    def _ids(self, ids):
        ids = torch.as_tensor(ids, device=self.device)
        if ids.dtype != torch.long or ids.shape != (self.batch_size, 2):
            raise ValueError("Expected two ordered integer teammate IDs per CN map, with -1 for an absent slot")
        first, second = ids.unbind(1)
        valid = ((first >= 0) & (first < len(self.teammates)) & (second >= -1) &
                 (second < len(self.teammates)) & (first != second))
        torch._assert_async(valid.all(), "Invalid CN group IDs")
        return ids

    def reset(self, layout="random", *, teammate_ids=None):
        if teammate_ids is None:
            first = torch.randint(len(self.teammates), (self.batch_size,), generator=self.generator, device=self.device)
            second = torch.randint(len(self.teammates) - 1, first.shape, generator=self.generator, device=self.device)
            both = torch.randint(2, first.shape, generator=self.generator, device=self.device).bool()
            teammate_ids = torch.stack((first, torch.where(both, second + (second >= first), -1)), dim=1)
        self.game.reset(layout)
        self.load_snapshot(positions=self.game.positions, velocity=self.game.velocity,
                           landmarks=self.game.landmarks, teammate_ids=teammate_ids)

    def load_snapshot(self, *, positions, velocity, landmarks, teammate_ids):
        """Load a paired start. The fourth slot, when present, requires its own position."""
        ids = self._ids(teammate_ids)
        positions = torch.as_tensor(positions, dtype=torch.float64, device=self.device)
        velocity = torch.as_tensor(velocity, dtype=torch.float64, device=self.device)
        if positions.ndim != 3 or positions.shape != velocity.shape:
            raise ValueError("Expected matching batches of CN positions and velocities")
        if positions.shape[1] == 3:
            torch._assert_async((ids[:, 1] == -1).all(), "Missing starting position of the fourth CN agent")
            positions = F.pad(positions, (0, 0, 0, 1))
            velocity = F.pad(velocity, (0, 0, 0, 1))
        self.game.load_snapshot(positions, velocity, landmarks)
        present = F.pad(ids >= 0, (2, 0), value=True)
        self.game.present = present
        self.game.positions *= present.unsqueeze(-1)
        self.game.velocity *= present.unsqueeze(-1)
        self.game.available[:, 3, 1:] = present[:, 3, None]
        self.teammate_ids = ids.clone()
        self._flat_ids = self.teammate_ids.clamp_min(0).flatten()
        self._hidden = self._initial_hidden[self._flat_ids]
        self._previous = self._hidden.new_zeros(self._flat_ids.numel(), self.n_actions)
        self.sample_counts.scatter_add_(0, self._flat_ids, (ids >= 0).long().flatten())
        self._all_obs = None
        self.last_joint_actions = None

    def apply_teammates(self, teammate_ids, spawn_positions=None):
        """Install the next group before ego observations; return actual ordered-group changes.

        Bodies in occupied slots keep their state. An arrival starts at a fresh
        uniform position with zero velocity, as in the scalar CN environment.
        """
        ids = self._ids(teammate_ids)
        active = ~(self.terminated | self.truncated)
        ids = torch.where(active[:, None], ids, self.teammate_ids)
        changed = (ids != self.teammate_ids).any(1)
        present = ids[:, 1] >= 0
        arrives = present & ~self.game.present[:, 3]
        if spawn_positions is None:
            spawn_positions = torch.rand((self.batch_size, 2), dtype=torch.float64, device=self.device,
                                         generator=self.game.rng) * 2 - 1
        else:
            spawn_positions = torch.as_tensor(spawn_positions, dtype=torch.float64, device=self.device)
            if spawn_positions.shape != (self.batch_size, 2):
                raise ValueError("Expected one potential joining position per CN map")
        position = torch.where(arrives[:, None], spawn_positions, self.game.positions[:, 3])
        self.game.positions[:, 3] = torch.where(present[:, None], position, 0.)
        self.game.velocity[:, 3] *= (present & ~arrives)[:, None]
        self.game.present[:, 3] = present
        self.game.available[:, 3, 1:] = present[:, None]
        self.teammate_ids = ids
        self._flat_ids = ids.clamp_min(0).flatten()
        # OpenTeamEnv restarts both teammate controllers when the ordered group changes.
        reset = changed[:, None].expand(-1, 2).reshape(-1, 1)
        self._hidden = torch.where(reset, self._initial_hidden[self._flat_ids], self._hidden)
        self._previous = torch.where(reset, 0., self._previous)
        self._all_obs = None
        return changed
