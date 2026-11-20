"""Fastap predator–prey (§8.2), with MPE simple_tag kinematics and observations."""

from operator import index

import numpy as np

from .particle import MOVES, Particle, ParticleWorld, advance, overlaps


def evade_predators(observation, available, rng):
    """Default fixed prey policy. Favor distance from predators and room to move.

    This policy is a reproduction choice. It uses only the prey's own 14-element
    simple_tag observation, as would a replacement learned or scripted policy.
    """
    velocity = observation[:2]
    position = observation[2:4]
    obstacles = observation[4:8].reshape(2, 2)
    predators = observation[8:14].reshape(3, 2)
    # Score a short look ahead, in coordinates relative to the prey.
    offsets = 0.20 * np.asarray(MOVES) + 0.15 * velocity
    predator_clearance = np.linalg.norm(offsets[:, None] - predators, axis=2).min(axis=1)
    obstacle_distance = np.linalg.norm(offsets[:, None] - obstacles, axis=2).min(axis=1)
    obstacle_cost = 2 * np.maximum(0, 0.28 - obstacle_distance)
    edge_cost = 2 * np.maximum(0, np.abs(position + offsets) - 0.9).sum(axis=1)
    return int(np.argmax(np.where(available, predator_clearance - obstacle_cost - edge_cost, -np.inf)))


class PredatorPreyWorld(ParticleWorld):
    """Three predators (two permanent egos, one teammate), one prey, two obstacles.

    External joint actions contain the three predators in slot order. The prey
    policy is called separately with (observation, action mask, world RNG).
    Supply a stateless prey_policy directly, or a zero-argument
    prey_policy_factory to construct fresh policy state on every reset.
    """

    obs_shape = 16
    state_shape = 20
    n_teammates = 1

    def __init__(self, seed=None, episode_limit=25, prey_policy=None, prey_policy_factory=None):
        super().__init__(seed=seed, episode_limit=episode_limit)
        if prey_policy is not None and prey_policy_factory is not None:
            raise ValueError("Choose either prey_policy or prey_policy_factory")
        self.prey_policy = evade_predators if prey_policy is None else prey_policy
        if not callable(self.prey_policy):
            raise TypeError("prey_policy must be callable")
        if prey_policy_factory is not None and not callable(prey_policy_factory):
            raise TypeError("prey_policy_factory must be callable")
        self.prey_policy_factory = prey_policy_factory
        self.last_prey_action = None

    @staticmethod
    def _check_count(count):
        if isinstance(count, (bool, np.bool_)) or index(count) != 1:
            raise ValueError("Predator–prey requires exactly one teammate predator")

    def reset(self, n_teammates=1):
        self._check_count(n_teammates)
        if self.prey_policy_factory is not None:
            policy = self.prey_policy_factory()
            if not callable(policy):
                raise TypeError("prey_policy_factory must return a callable policy")
            self.prey_policy = policy
        self.last_prey_action = None
        self.players = [
            Particle(position=self.rng.uniform(-1, 1, size=2), radius=0.075,
                     acceleration=3.0, max_speed=1.0)
            for _ in range(3)
        ]
        self.prey = Particle(position=self.rng.uniform(-1, 1, size=2), radius=0.05,
                             acceleration=4.0, max_speed=1.3)
        self.obstacles = [
            Particle(position=self.rng.uniform(-0.9, 0.9, size=2), radius=0.2, movable=False)
            for _ in range(2)
        ]
        self._begin_episode()

    def set_teammate_count(self, n_teammates):
        self._require_active()
        self._check_count(n_teammates)

    def _observe(self, observer):
        parts = [observer.velocity, observer.position]
        parts.extend(o.position - observer.position for o in self.obstacles)
        parts.extend(p.position - observer.position for p in self.players if p is not observer)
        if observer is not self.prey:
            parts.extend((self.prey.position - observer.position, self.prey.velocity))
        return np.concatenate(parts).astype(np.float32)

    def observations(self):
        return np.stack([self._observe(predator) for predator in self.players])

    def prey_observation(self):
        return self._observe(self.prey)

    def state(self):
        """Global (x, y, vx, vy) per predator and prey, then both obstacle positions."""
        values = []
        for mover in [*self.players, self.prey]:
            values.extend((mover.position, mover.velocity))
        values.extend(o.position for o in self.obstacles)
        return np.concatenate(values).astype(np.float32)

    def step(self, joint_actions):
        self._require_active()
        actions = self._check_actions(joint_actions, len(self.players))
        available = np.ones(self.n_actions, dtype=np.int32)
        prey_action = self.prey_policy(self.prey_observation(), available, self.rng)
        (prey_action,) = self._check_actions([prey_action], 1)
        advance([*self.players, self.prey, *self.obstacles], [*actions, prey_action, 0, 0])
        self.last_prey_action = prey_action
        contacts = sum(overlaps(predator, self.prey) for predator in self.players)
        team_reward = float(10 * contacts)
        done, info = self._finish_step()
        info.update(contacts=int(contacts), prey_reward=-team_reward)
        return team_reward, done, info
