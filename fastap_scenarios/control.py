"""Choose who supplies actions; the underlying game computes motion and rewards."""

from operator import index

import numpy as np


class _EpisodeStream:
    def __init__(self, seed_sequence, world_factory):
        world, group, wait, policy = seed_sequence.spawn(4)
        self.world = world_factory(seed=int(world.generate_state(1)[0]))
        self.group_rng = np.random.default_rng(group)
        self.wait_rng = np.random.default_rng(wait)
        self.policy_rng = np.random.default_rng(policy)


class _AgentView:
    """EPyMARL reads the first n_agents slots. Each phase has independent randomness."""

    def __init__(self, world_factory, seed):
        self._streams = [_EpisodeStream(seq, world_factory) for seq in np.random.SeedSequence(seed).spawn(2)]
        self._next_test_mode = False
        self._stream = self._streams[0]
        self.episode_limit = self.world.episode_limit
        self.last_joint_actions = ()

    @property
    def world(self):
        return self._stream.world

    def set_test_mode(self, test_mode):
        """Set the phase for the next reset. A running episode keeps its schedule."""
        self._next_test_mode = bool(test_mode)

    def _begin_episode(self):
        self._test_mode = self._next_test_mode
        self._stream = self._streams[int(self._test_mode)]
        self.last_joint_actions = ()

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


class TeamEnv(_AgentView):
    """Fixed roster; every cooperative slot is driven by the caller."""

    def __init__(self, world_factory, seed, n_teammates):
        super().__init__(world_factory, seed)
        self.n_teammates = n_teammates
        self.n_agents = 2 + n_teammates

    def reset(self):
        self._begin_episode()
        self.world.reset(n_teammates=self.n_teammates)
        return self.get_obs(), self.get_state()

    def step(self, actions):
        joint_actions = tuple(map(index, actions))
        result = self.world.step(joint_actions)
        self.last_joint_actions = joint_actions
        return result


class OpenTeamEnv(_AgentView):
    """Two ego slots and an independently sampled group of teammate policies."""

    n_agents = 2

    def __init__(self, world_factory, seed, switch_interval, teammate_groups):
        self.groups = tuple(teammate_groups)
        if not self.groups or len({g.name for g in self.groups}) != len(self.groups):
            raise ValueError("Provide at least one teammate group, with distinct group names")
        self.set_switch_interval(switch_interval)
        super().__init__(world_factory, seed)
        self._done = True
        self._policies = ()
        self.group_index = None
        self.group_name = None
        self.events = []

    def set_switch_interval(self, switch_interval):
        if switch_interval is not None:
            low, high = map(index, switch_interval)
            if not 1 <= low <= high:
                raise ValueError("switch_interval must be None or positive inclusive bounds [low, high]")
            switch_interval = low, high
        self.switch_interval = switch_interval

    @property
    def n_teammates(self):
        return len(self._policies)

    def _sample_group(self, step, group_index=None):
        # Algorithm 3 resamples from the whole pool, including the current group.
        if group_index is None:
            group_index = self._stream.group_rng.integers(len(self.groups))
        group_index = index(group_index)
        if not 0 <= group_index < len(self.groups):
            raise ValueError("Group index outside the available teammate pool")
        group = self.groups[group_index]
        # A recurrent teammate that stays in the same group keeps its trajectory.
        if step == 0 or group.name != self.group_name:
            self._policies = tuple(factory() for factory in group.policies)
        self.group_index = group_index
        self.group_name = group.name
        self.events.append(dict(step=step, group=group.name, n_teammates=self.n_teammates))

    def _next_switch(self, step):
        if self._test_mode and self.switch_interval is not None:
            low, high = self.switch_interval
            return step + int(self._stream.wait_rng.integers(low, high + 1))
        return None

    def reset(self, *, group_index=None):
        self._begin_episode()
        self.events = []
        self.last_teammate_obs = None
        self._sample_group(step=0, group_index=group_index)
        self.world.reset(n_teammates=self.n_teammates)
        self._switch_step = self._next_switch(0)
        self._done = False
        return self.get_obs(), self.get_state()

    def step(self, actions):
        if self._done:
            raise RuntimeError("Call reset() before stepping a new episode")
        controlled = [index(action) for action in actions]
        if len(controlled) != self.n_agents or any(a < 0 or a >= self.world.n_actions for a in controlled):
            raise ValueError(f"Provide two action IDs in [0, {self.world.n_actions - 1}]")
        obs, masks = self.world.observations(), self.world.avail_actions()
        joint_actions = controlled.copy()
        for slot, policy in enumerate(self._policies, start=self.n_agents):
            action = index(policy(obs[slot], masks[slot], self._stream.policy_rng))
            if not 0 <= action < self.world.n_actions or not masks[slot, action]:
                raise ValueError(f"Teammate policy in slot {slot} returned an unavailable action: {action}")
            joint_actions.append(action)
        reward, self._done, info = self.world.step(joint_actions)
        self.last_joint_actions = tuple(joint_actions)
        self.last_teammate_obs = obs[self.n_agents:]
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
