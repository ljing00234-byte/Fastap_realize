"""Fastap's LBF, predator–prey and cooperative navigation games."""

from .game import GameWorld
from .policies import TeammateGroup
from .registry import SCENARIOS, make_env, make_world

__all__ = ["GameWorld", "SCENARIOS", "TeammateGroup", "make_env", "make_world"]
