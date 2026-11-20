"""EPyMARL interface for two controlled agents and a separately selected teammate group."""

from dataclasses import dataclass
from operator import index
from typing import Callable, Tuple

import numpy as np

from .world import ForagingWorld


# Each factory returns a fresh policy for one slot. A recurrent policy can keep its
# hidden state in that callable. Inputs are only the slot's local observation and mask.
Policy = Callable[[np.ndarray, np.ndarray, np.random.Generator], int]


@dataclass(frozen=True)
class TeammateGroup:
    name: str
    policies: Tuple[Callable[[], Policy], ...]

    def __post_init__(self):
        if not self.name or len(self.policies) not in (1, 2):
            raise ValueError("Each named teammate group must contain one or two policy factories")
        if not all(callable(factory) for factory in self.policies):
            raise TypeError("Teammate policies must be provided as no-argument factories")


def _random_legal(obs, avail, rng):
    return int(rng.choice(np.flatnonzero(avail)))


def _load_if_possible(obs, avail, rng):
    return 5 if avail[5] else _random_legal(obs, avail, rng)


# Scripted fixtures for checking the task before trained teammate policies exist.
DEMO_GROUPS = (
    TeammateGroup("random_solo", (lambda: _random_legal,)),
    TeammateGroup("load_pair", (lambda: _load_if_possible, lambda: _load_if_possible)),
)


class _EpisodeStream:
    def __init__(self, seed_sequence):
        world, group, wait, policy = seed_sequence.spawn(4)
        self.world = ForagingWorld(seed=int(world.generate_state(1)[0]))
        self.group_rng = np.random.default_rng(group)
        self.wait_rng = np.random.default_rng(wait)
        self.policy_rng = np.random.default_rng(policy)


class FastapLBFEnv:
    n_agents = 2

    def __init__(self, seed=0, switch_interval=(5, 8), teammate_groups=None):
        self.groups = tuple(DEMO_GROUPS if teammate_groups is None else teammate_groups)
        if not self.groups or len({g.name for g in self.groups}) != len(self.groups):
            raise ValueError("Provide at least one teammate group, with distinct group names")
        if switch_interval is not None:
            low, high = map(index, switch_interval)
            if not 1 <= low <= high:
                raise ValueError("switch_interval must be None or positive inclusive bounds [low, high]")
            switch_interval = low, high
        self.switch_interval = switch_interval
        self._streams = [_EpisodeStream(seq) for seq in np.random.SeedSequence(seed).spawn(2)]
        self._next_test_mode = False
        self._stream = self._streams[0]
        self.episode_limit = self.world.episode_limit
        self._done = True
        self._policies = ()
        self.group_name = None
        self.events = []

    @property
    def world(self):
        return self._stream.world

    @property
    def n_teammates(self):
        return len(self._policies)

    def set_test_mode(self, test_mode):
        """Set the phase for the next reset. A running episode keeps its schedule."""
        self._next_test_mode = bool(test_mode)

    def _sample_group(self, step):
        # Algorithm 3 resamples from the whole pool, including the current group.
        group = self.groups[self._stream.group_rng.integers(len(self.groups))]
        self._policies = tuple(factory() for factory in group.policies)
        self.group_name = group.name
        self.events.append(dict(step=step, group=group.name, n_teammates=self.n_teammates))

    def _next_switch(self, step):
        if self._test_mode and self.switch_interval is not None:
            low, high = self.switch_interval
            return step + int(self._stream.wait_rng.integers(low, high + 1))
        return None

    def reset(self):
        self._test_mode = self._next_test_mode
        self._stream = self._streams[int(self._test_mode)]
        self.events = []
        self._sample_group(step=0)
        self.world.reset(n_teammates=self.n_teammates)
        self._switch_step = self._next_switch(0)
        self._done = False
        return self.get_obs(), self.get_state()

    def step(self, actions):
        if self._done:
            raise RuntimeError("Call reset() before stepping a new episode")
        controlled = [index(action) for action in actions]
        if len(controlled) != self.n_agents or any(a < 0 or a >= self.world.n_actions for a in controlled):
            raise ValueError("Provide one action ID in [0, 5] for each of the two controlled agents")
        obs, masks = self.world.observations(), self.world.avail_actions()
        joint_actions = controlled.copy()
        for slot, policy in enumerate(self._policies, start=self.n_agents):
            action = index(policy(obs[slot], masks[slot], self._stream.policy_rng))
            if not 0 <= action < self.world.n_actions or not masks[slot, action]:
                raise ValueError(f"Teammate policy in slot {slot} returned an unavailable action: {action}")
            joint_actions.append(action)
        reward, self._done, info = self.world.step(joint_actions)
        if not self._done and self.world.current_step == self._switch_step:
            # Apply a group change before the controller reads its next observation.
            self._sample_group(self.world.current_step)
            self.world.set_teammate_count(self.n_teammates)
            self._switch_step = self._next_switch(self.world.current_step)
        if self._done:
            info = dict(info, teammate_resamples=len(self.events) - 1,
                        teammate_group_changes=sum(prev["group"] != cur["group"]
                                                   for prev, cur in zip(self.events, self.events[1:])))
        return reward, self._done, info

    def get_obs(self):
        return self.world.observations()[: self.n_agents]

    def get_obs_agent(self, agent_id):
        if not 0 <= agent_id < self.n_agents:
            raise IndexError(agent_id)
        return self.get_obs()[agent_id]

    def get_obs_size(self):
        return self.world.obs_shape

    def get_state(self):
        return self.world.state()

    def get_state_size(self):
        return self.world.state_shape

    def get_avail_actions(self):
        return self.world.avail_actions()[: self.n_agents]

    def get_avail_agent_actions(self, agent_id):
        if not 0 <= agent_id < self.n_agents:
            raise IndexError(agent_id)
        return self.get_avail_actions()[agent_id]

    def get_total_actions(self):
        return self.world.n_actions

    def get_env_info(self):
        return dict(n_agents=self.n_agents, n_actions=self.get_total_actions(),
                    obs_shape=self.get_obs_size(), state_shape=self.get_state_size(),
                    episode_limit=self.episode_limit)

    def close(self):
        for stream in self._streams:
            stream.world.close()
