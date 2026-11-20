"""Game selection, independent of a learner or an optimizer."""

from dataclasses import dataclass
from functools import partial
from operator import index
from typing import Callable, Tuple

import numpy as np

from .control import OpenTeamEnv, TeamEnv
from .cooperative_navigation import CooperativeNavigationWorld
from .foraging import ForagingWorld
from .game import GameWorld
from .policies import CN_DEMO, LBF_DEMO, PP_DEMO, TeammateGroup
from .predator_prey import PredatorPreyWorld


@dataclass(frozen=True)
class Scenario:
    world: Callable[..., GameWorld]
    teammate_counts: Tuple[int, ...]
    demo_groups: Tuple[TeammateGroup, ...]

    def check_count(self, count):
        if isinstance(count, (bool, np.bool_)) or index(count) not in self.teammate_counts:
            raise ValueError(f"Supported teammate counts: {self.teammate_counts}; received {count!r}")
        return index(count)


SCENARIOS = {
    "lbf": Scenario(ForagingWorld, (1, 2), LBF_DEMO),
    "pp": Scenario(PredatorPreyWorld, (1,), PP_DEMO),
    "cn": Scenario(CooperativeNavigationWorld, (1, 2), CN_DEMO),
}


def _scenario(key):
    try:
        return SCENARIOS[key]
    except KeyError:
        raise ValueError(f"Unknown game {key!r}. Choose from {', '.join(SCENARIOS)}") from None


def make_world(key, *, seed=0, **world_options) -> GameWorld:
    """A standalone game receiving one joint action for the cooperating team."""
    return _scenario(key).world(seed=seed, **world_options)


def make_env(key, *, control="ego", seed=0, world_options=None, n_teammates=None,
             teammate_groups=None, switch_interval=None):
    """Adapt the same game to external control of two egos or the entire team.

    team: fixed n_teammates (default one), no policy injection or switching.
    ego: injected groups or rule-based fixtures; optional evaluation resampling.
    """
    scene = _scenario(key)
    world = partial(scene.world, **(world_options or {}))
    if n_teammates is not None:
        n_teammates = scene.check_count(n_teammates)
    if control == "team":
        if teammate_groups is not None or switch_interval is not None:
            raise ValueError("Teammate policies and switch intervals require control='ego'")
        return TeamEnv(world, seed, 1 if n_teammates is None else n_teammates)
    if control != "ego":
        raise ValueError("control must be 'ego' or 'team'")
    if teammate_groups is None:
        groups = tuple(g for g in scene.demo_groups
                       if n_teammates is None or len(g.policies) == n_teammates)
    else:
        groups = tuple(teammate_groups)
    for group in groups:
        count = scene.check_count(len(group.policies))
        if n_teammates is not None and count != n_teammates:
            raise ValueError(f"Group {group.name!r} must contain {n_teammates} teammates")
    return OpenTeamEnv(world, seed, switch_interval, groups)
