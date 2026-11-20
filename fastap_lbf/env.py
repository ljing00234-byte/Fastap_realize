"""Compatibility entry point for the original LBF experiments."""

from fastap_scenarios.control import OpenTeamEnv
from fastap_scenarios.foraging import ForagingWorld
from fastap_scenarios.policies import LBF_DEMO as DEMO_GROUPS, Policy, TeammateGroup
from fastap_scenarios.registry import SCENARIOS


class FastapLBFEnv(OpenTeamEnv):
    def __init__(self, seed=0, switch_interval=(5, 8), teammate_groups=None):
        groups = DEMO_GROUPS if teammate_groups is None else tuple(teammate_groups)
        for group in groups:
            SCENARIOS["lbf"].check_count(len(group.policies))
        super().__init__(ForagingWorld, seed, switch_interval, groups)
