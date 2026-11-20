"""Fastap cooperative navigation (appendix B.2 of the UAI supplement)."""

from itertools import combinations
from operator import index

import numpy as np

from .particle import Particle, ParticleWorld, advance, overlaps

AGENT_RADIUS = 0.15
AGENT_ACCELERATION = 5.0
DEFAULT_COLLISION_PENALTY = 1.0


class CooperativeNavigationWorld(ParticleWorld):
    """Two controlled agents, one or two teammates, and four fixed landmarks.

    Observation: own velocity/position, four relative landmark positions,
    three other players' (relative x, relative y, present) triples in slot order.
    State: four (x, y, vx, vy, present) slots followed by four landmark positions.
    An absent fourth player uses zeros; its presence bit distinguishes overlap.
    """

    obs_shape = 21
    state_shape = 28

    def __init__(self, seed=None, episode_limit=25, collision_penalty=DEFAULT_COLLISION_PENALTY):
        super().__init__(seed=seed, episode_limit=episode_limit)
        if not np.isfinite(collision_penalty) or collision_penalty < 0:
            raise ValueError("collision_penalty must be finite and nonnegative")
        self.collision_penalty = float(collision_penalty)
        self.landmarks = []
        self.agents = []

    @property
    def n_teammates(self):
        return len(self.agents) - 2

    @staticmethod
    def _count(value):
        count = index(value)
        if isinstance(value, (bool, np.bool_)) or count not in (1, 2):
            raise ValueError("Cooperative navigation requires one or two teammates")
        return count

    def _new_agent(self):
        return Particle(self.rng.uniform(-1, 1, 2), radius=AGENT_RADIUS, acceleration=AGENT_ACCELERATION)

    def reset(self, n_teammates=2):
        count = self._count(n_teammates)
        self.landmarks = [Particle(self.rng.uniform(-1, 1, 2), radius=0.05,
                                   movable=False, collidable=False) for _ in range(4)]
        self.agents = [self._new_agent() for _ in range(2 + count)]
        self._begin_episode()

    def set_teammate_count(self, n_teammates):
        self._require_active()
        count = self._count(n_teammates)
        if count > self.n_teammates:
            self.agents.append(self._new_agent())
        elif count < self.n_teammates:
            self.agents.pop()

    def observations(self):
        observations = np.zeros((len(self.agents), self.obs_shape), dtype=np.float32)
        for i, agent in enumerate(self.agents):
            row = observations[i]
            row[:2] = agent.velocity
            row[2:4] = agent.position
            row[4:12] = np.concatenate([landmark.position - agent.position
                                        for landmark in self.landmarks])
            others = [other for j, other in enumerate(self.agents) if j != i]
            peer_features = row[12:].reshape(3, 3)
            for peer, other in zip(peer_features, others):
                peer[:2] = other.position - agent.position
                peer[2] = 1
        return observations

    def state(self):
        slots = np.zeros((4, 5), dtype=np.float32)
        for row, agent in zip(slots, self.agents):
            row[:2] = agent.position
            row[2:4] = agent.velocity
            row[4] = 1
        return np.concatenate([slots.ravel()] + [p.position for p in self.landmarks]).astype(np.float32)

    def step(self, joint_actions):
        self._require_active()
        actions = self._check_actions(joint_actions, len(self.agents))
        # Landmarks do not collide and never move.
        advance(self.agents, actions)
        distances = [min(np.linalg.norm(target.position - agent.position) for agent in self.agents)
                     for target in self.landmarks]
        collisions = sum(overlaps(a, b) for a, b in combinations(self.agents, 2))
        distance_sum = float(sum(distances))
        reward = -distance_sum - self.collision_penalty * collisions
        done, info = self._finish_step()
        info.update(landmark_distance_sum=distance_sum, agent_collisions=int(collisions))
        return float(reward), done, info
