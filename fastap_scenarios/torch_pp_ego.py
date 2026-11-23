"""Two controllable PP predators paired with one independently controlled predator."""

import torch

from .torch_pool_ego import TorchPoolEgo
from .torch_predator_prey import TorchPredatorPrey


class TorchPPEgo(TorchPoolEgo):
    scene_key = TorchPredatorPrey.scene_key
    n_actions = TorchPredatorPrey.n_actions
    obs_shape = TorchPredatorPrey.obs_shape
    state_shape = TorchPredatorPrey.state_shape
    episode_limit = TorchPredatorPrey.episode_limit

    def __init__(self, batch_size, device, *, teammates, seed=0, teammate_seed=None):
        game = TorchPredatorPrey(batch_size, device, seed)
        super().__init__(game, teammates, seed + 1 if teammate_seed is None else teammate_seed)
        self._initial_hidden = torch.cat([teammate.model.init_hidden() for teammate in self.teammates])

    def _begin(self, teammate_ids):
        super()._begin(teammate_ids)
        self._hidden = self._initial_hidden[self._flat_ids]

    def apply_teammates(self, teammate_ids):
        """Apply a policy redraw before the next decision and return which rows changed.

        The third predator keeps its position and velocity. A different policy
        starts with its own initial memory and an empty previous action. Redrawing
        the current policy keeps its memory, as in the scalar OpenTeamEnv.
        """
        ids = torch.as_tensor(teammate_ids, device=self.device)
        if ids.shape != (self.batch_size,) or ids.dtype != torch.long:
            raise ValueError("Expected one integer teammate policy index per PP game")
        torch._assert_async(((ids >= 0) & (ids < len(self.teammates))).all(), "Teammate ID outside PP pool")
        changed = (ids != self.teammate_ids) & ~(self.terminated | self.truncated)
        self.teammate_ids = torch.where(changed, ids, self.teammate_ids)
        self._flat_ids = self.teammate_ids
        self._hidden = torch.where(changed[:, None], self._initial_hidden[self._flat_ids], self._hidden)
        self._previous = torch.where(changed[:, None], 0, self._previous)
        return changed

    @property
    def roster_counts(self):
        return self.sample_counts.sum().reshape(1)
