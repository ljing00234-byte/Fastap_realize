"""Headless two-dimensional MPE dynamics for the Fastap particle scenarios.

Unit-mass integration and soft contacts use the constants in OpenAI MPE's
`multiagent/core.py`. Integer actions follow its discrete-input convention.
The initial sampling range is not a physical wall.
"""

from dataclasses import dataclass, field
from itertools import combinations
from operator import index
from typing import Optional

import numpy as np


# stay, left, right, down, up
MOVES = np.array([(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)], dtype=float)
_DT = 0.1
_DAMPING = 0.25
_CONTACT_FORCE = 100.0
_CONTACT_MARGIN = 0.001


@dataclass(eq=False)
class Particle:
    position: np.ndarray
    radius: float
    acceleration: float = 5.0
    max_speed: Optional[float] = None
    movable: bool = True
    collidable: bool = True
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))

    def __post_init__(self):
        self.position = np.array(self.position, dtype=float, copy=True)
        self.velocity = np.array(self.velocity, dtype=float, copy=True)


def overlaps(a, b):
    """Return whether two different particles overlap, excluding tangency."""
    return a is not b and np.linalg.norm(a.position - b.position) < a.radius + b.radius


def advance(bodies, actions):
    """Apply all controls/contacts simultaneously, then integrate one MPE tick."""
    if len(actions) != len(bodies):
        raise ValueError("Expected one action per body, including stationary objects")
    forces = [body.acceleration * MOVES[action] for body, action in zip(bodies, actions)]
    for a, b in combinations(range(len(bodies)), 2):
        first, second = bodies[a], bodies[b]
        if not (first.collidable and second.collidable and (first.movable or second.movable)):
            continue
        displacement = first.position - second.position
        distance = np.linalg.norm(displacement)
        overlap = first.radius + second.radius - distance
        penetration = _CONTACT_MARGIN * np.logaddexp(0, overlap / _CONTACT_MARGIN)
        # A deterministic normal keeps an exact coincidence finite and separable.
        normal = displacement / distance if distance else np.array([1.0, 0.0])
        contact = _CONTACT_FORCE * penetration * normal
        forces[a] += contact
        forces[b] -= contact

    for body, force in zip(bodies, forces):
        if body.movable:
            body.velocity = (1 - _DAMPING) * body.velocity + _DT * force
            speed = np.linalg.norm(body.velocity)
            if body.max_speed is not None and speed > body.max_speed:
                body.velocity *= body.max_speed / speed
            body.position += _DT * body.velocity


class ParticleWorld:
    n_actions = len(MOVES)

    def __init__(self, seed=None, episode_limit=25):
        self.episode_limit = index(episode_limit)
        if isinstance(episode_limit, (bool, np.bool_)) or self.episode_limit < 1:
            raise ValueError("episode_limit must be a positive integer")
        self.rng = np.random.default_rng(seed)
        self.current_step = 0
        self._active = False

    def _begin_episode(self):
        self.current_step = 0
        self._active = True

    def _require_active(self):
        if not self._active:
            raise RuntimeError("Reset the world before taking a step or changing the roster")

    def _check_actions(self, actions, n_players):
        if len(actions) != n_players:
            raise ValueError(f"Expected {n_players} actions, one for each active player")
        result = [index(action) for action in actions]
        if any(action < 0 or action >= self.n_actions for action in result):
            raise ValueError(f"Action IDs must be in [0, {self.n_actions - 1}]")
        return result

    def _finish_step(self):
        self.current_step += 1
        done = self.current_step >= self.episode_limit
        self._active = not done
        return done, {"episode_limit": done}

    def avail_actions(self):
        return np.ones((2 + self.n_teammates, self.n_actions), dtype=np.int32)

    def close(self):
        self._active = False
