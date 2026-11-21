"""Batched two-role ego control with fixed, independently selected recurrent teammates."""

import torch
from torch.nn import functional as F


class TorchPoolEgo:
    """Expose the first two roles to the learner; run each other role with its frozen policy."""

    n_agents = 2
    control = "ego"

    def __init__(self, game, teammates, teammate_seed):
        self.game = game
        self.batch_size = game.batch_size
        self.n_teammates = game.n_agents - self.n_agents
        self.teammates = tuple(teammates)
        if len(self.teammates) < self.n_teammates:
            raise ValueError("Too few distinct teammates for the requested roster")
        self.spec = self.teammates[0].spec
        if ((self.spec.scene_key, self.spec.control, self.spec.n_agents,
             self.spec.obs_shape, self.spec.n_actions)
                != (game.scene_key, "team", 3, game.obs_shape, game.n_actions)):
            raise ValueError(f"Expected frozen three-role {game.scene_key} teammates")
        if any(t.spec != self.spec for t in self.teammates):
            raise ValueError("The frozen pool must have matching agent specifications")
        self.generator = torch.Generator(device=game.device).manual_seed(teammate_seed)
        self.device = self.generator.device
        for teammate in self.teammates:
            if any(p.device != self.device or p.requires_grad for p in teammate.model.parameters()):
                raise ValueError("Expected frozen teammate weights on the game device")
        n_slots = self.batch_size * self.n_teammates
        self._rows = torch.arange(n_slots, device=self.device)
        # Each external slot uses the policy exported for original team role 2.
        self._role = torch.eye(self.spec.n_agents, device=self.device)[2].expand(n_slots, -1)
        self.sample_counts = torch.zeros(len(self.teammates), dtype=torch.long, device=self.device)

    def _begin(self, teammate_ids):
        if teammate_ids is None:
            ids = torch.randint(len(self.teammates), (self.batch_size,), device=self.device,
                                generator=self.generator)
            if self.n_teammates == 2:
                second = torch.randint(len(self.teammates) - 1, (self.batch_size,), device=self.device,
                                       generator=self.generator)
                ids = torch.stack((ids, second + (second >= ids)), dim=1)
        else:
            ids = torch.as_tensor(teammate_ids, device=self.device)
            shape = (self.batch_size,) if self.n_teammates == 1 else (self.batch_size, 2)
            if (ids.dtype != torch.long or ids.shape != shape
                    or (ids < 0).any() or (ids >= len(self.teammates)).any()):
                raise ValueError(f"Expected valid teammate indices of shape {shape}")
            if self.n_teammates == 2 and (ids[:, 0] == ids[:, 1]).any():
                raise ValueError("Choose distinct teammates for each two-teammate game")
            ids = ids.clone()
        self.teammate_ids = ids
        self._flat_ids = ids.flatten()
        self.sample_counts.scatter_add_(0, self._flat_ids, torch.ones_like(self._flat_ids))
        self._hidden = self.teammates[0].model.init_hidden().expand(ids.numel(), -1)
        self._previous = self._hidden.new_zeros(ids.numel(), self.n_actions)
        self._all_obs = None
        self.last_joint_actions = None

    def reset(self, layout="random", *, teammate_ids=None):
        self.game.reset(layout=layout)
        self._begin(teammate_ids)

    def load_snapshot(self, *, teammate_ids=None, **board):
        """Start an episode on the supplied board with fresh teammate memory."""
        self.game.load_snapshot(**board)
        self._begin(teammate_ids)

    @property
    def terminated(self):
        return self.game.terminated

    @property
    def truncated(self):
        return self.game.truncated

    def state(self):
        return self.game.state()

    def _full_observations(self):
        if self._all_obs is None:
            self._all_obs = self.game.observations()
        return self._all_obs

    def observations(self):
        return self._full_observations()[:, :self.n_agents]

    def avail_actions(self):
        return self.game.avail_actions()[:, :self.n_agents]

    @torch.no_grad()
    def step(self, actions):
        if actions.shape != (self.batch_size, self.n_agents) or actions.dtype != torch.long:
            raise ValueError("Expected two ego action indices per map")
        features = [self._full_observations()[:, 2:].reshape(-1, self.obs_shape)]
        if self.spec.obs_last_action:
            features.append(self._previous)
        if self.spec.obs_agent_id:
            features.append(self._role)
        inputs = torch.cat(features, dim=-1)
        # Compute all candidates, then gather on device. Each row has its own
        # hidden state and keeps only its assigned policy's next state.
        candidates = [teammate.model(inputs, self._hidden) for teammate in self.teammates]
        values, hidden = zip(*candidates)
        values = torch.stack(values)[self._flat_ids, self._rows]
        hidden = torch.stack(hidden)[self._flat_ids, self._rows]
        active = ~(self.terminated | self.truncated)
        active = active[:, None].expand(-1, self.n_teammates).reshape(-1)
        available = self.game.avail_actions()[:, 2:].reshape(-1, self.n_actions)
        action = values.masked_fill(~available, -torch.inf).argmax(-1)
        self._hidden = torch.where(active[:, None], hidden, self._hidden)
        self._previous = torch.where(active[:, None], F.one_hot(action, self.n_actions).float(),
                                     self._previous)
        self.last_joint_actions = torch.cat((actions, action.reshape(self.batch_size, self.n_teammates)), dim=1)
        self._all_obs = None
        return self.game.step(self.last_joint_actions)
