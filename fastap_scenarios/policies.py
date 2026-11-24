"""Local action interface shared by scripted and learned teammate policies."""

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np


# obs, valid action mask, episode RNG -> discrete action. A fresh policy per
# slot can own its own recurrent state even if several slots share weights.
Policy = Callable[[np.ndarray, np.ndarray, np.random.Generator], int]


@dataclass(frozen=True)
class TeammateGroup:
    name: str
    policies: Tuple[Callable[[], Policy], ...]
    # Original role-0/1 partners, when the group has a jointly trained LBF roster.
    probe_egos: Tuple[Callable[[], Policy], ...] = ()

    def __post_init__(self):
        if not self.name or not self.policies:
            raise ValueError("A teammate group needs a name and at least one policy factory")
        if not all(callable(factory) for factory in self.policies):
            raise TypeError("Teammate policies must be provided as no-argument factories")
        if self.probe_egos and len(self.probe_egos) != 2:
            raise ValueError("Paired probes need both original ego policies")
        if not all(callable(factory) for factory in self.probe_egos):
            raise TypeError("Probe ego policies must be provided as no-argument factories")


def random_legal(obs, avail, rng):
    return int(rng.choice(np.flatnonzero(avail)))


def stay(obs, avail, rng):
    return 0


def load_if_possible(obs, avail, rng):
    return 5 if avail[5] else random_legal(obs, avail, rng)


RANDOM_SOLO = TeammateGroup("random_solo", (lambda: random_legal,))
LBF_DEMO = (RANDOM_SOLO, TeammateGroup("load_pair", (lambda: load_if_possible,) * 2))
PP_DEMO = (RANDOM_SOLO, TeammateGroup("stationary_solo", (lambda: stay,)))
CN_DEMO = (RANDOM_SOLO, TeammateGroup("random_pair", (lambda: random_legal,) * 2))
