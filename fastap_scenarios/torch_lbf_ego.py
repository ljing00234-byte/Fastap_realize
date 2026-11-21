"""Two controllable LBF roles and one or two frozen teammates fixed per episode."""

import torch

from .torch_foraging import TorchForaging
from .torch_pool_ego import TorchPoolEgo


class _EgoSpec:
    scene_key = "lbf"
    control = "ego"
    n_agents = 2
    n_actions = TorchForaging.n_actions
    obs_shape = TorchForaging.obs_shape
    state_shape = TorchForaging.state_shape
    episode_limit = TorchForaging.episode_limit


class TorchLBFEgo(_EgoSpec, TorchPoolEgo):
    def __init__(self, batch_size, device, *, teammates, seed=0, teammate_seed=None, n_teammates=1):
        if type(n_teammates) is not int or n_teammates not in (1, 2):
            raise ValueError("Expected one or two frozen LBF teammates")
        game = TorchForaging(batch_size, device, seed, n_agents=2 + n_teammates)
        super().__init__(game, teammates, seed + 1 if teammate_seed is None else teammate_seed)


class TorchMixedLBFEgo(_EgoSpec):
    """Stratify each rollout equally across one- and two-teammate games."""

    def __init__(self, batch_size, device, *, teammates, seed=0, teammate_seed=None):
        if batch_size < 2 or batch_size % 2:
            raise ValueError("Mixed-roster LBF requires a positive, even training batch")
        self.batch_size = batch_size
        if teammate_seed is None:
            teammate_seed = seed + 1
        self.worlds = tuple(
            TorchLBFEgo(batch_size // 2, device, teammates=teammates, n_teammates=count,
                        seed=seed + offset, teammate_seed=teammate_seed + offset)
            for count, offset in ((1, 0), (2, 104729))
        )
        self.device = self.worlds[0].device
        self.roster_counts = torch.zeros(2, dtype=torch.long, device=self.device)

    @property
    def sample_counts(self):
        return self.worlds[0].sample_counts + self.worlds[1].sample_counts

    def reset(self, layout="random"):
        for world in self.worlds:
            world.reset(layout=layout)
        self.roster_counts += self.batch_size // 2

    @property
    def terminated(self):
        return torch.cat([w.terminated for w in self.worlds])

    @property
    def truncated(self):
        return torch.cat([w.truncated for w in self.worlds])

    def observations(self):
        return torch.cat([w.observations() for w in self.worlds])

    def state(self):
        return torch.cat([w.state() for w in self.worlds])

    def avail_actions(self):
        return torch.cat([w.avail_actions() for w in self.worlds])

    def step(self, actions):
        if actions.shape != (self.batch_size, 2):
            raise ValueError("Expected two ego actions per mixed-roster game")
        steps = [w.step(chunk) for w, chunk in zip(self.worlds, actions.chunk(2))]
        return tuple(torch.cat(values) for values in zip(*steps))
