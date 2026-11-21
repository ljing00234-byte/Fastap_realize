"""Two controllable PP predators paired with one frozen predator for the full episode."""

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

    @property
    def roster_counts(self):
        return self.sample_counts.sum().reshape(1)
