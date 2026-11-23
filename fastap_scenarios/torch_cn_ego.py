"""Two controllable navigators paired with one frozen teammate for the full episode."""

from .torch_cooperative_navigation import TorchCooperativeNavigation
from .torch_pool_ego import TorchPoolEgo


class TorchCNEgo(TorchPoolEgo):
    scene_key = TorchCooperativeNavigation.scene_key
    n_actions = TorchCooperativeNavigation.n_actions
    obs_shape = TorchCooperativeNavigation.obs_shape
    state_shape = TorchCooperativeNavigation.state_shape
    episode_limit = TorchCooperativeNavigation.episode_limit

    def __init__(self, batch_size, device, *, teammates, seed=0, teammate_seed=None):
        game = TorchCooperativeNavigation(batch_size, device, seed)
        super().__init__(game, teammates, seed + 1 if teammate_seed is None else teammate_seed)

    @property
    def roster_counts(self):
        return self.sample_counts.sum().reshape(1)
