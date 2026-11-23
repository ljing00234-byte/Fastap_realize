"""Paper Fastap rollouts on the existing games and their raw team rewards (App. B.2)."""

from functools import partial
import hashlib
import json
from pathlib import Path

import numpy as np

from .control import OpenTeamEnv
from .policies import TeammateGroup
from .registry import SCENARIOS


def verified_pool_members(manifest_path, scene):
    """Return (manifest ID, verified checkpoint directory) in publication order."""
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if scene not in SCENARIOS or (manifest.get("format"), manifest.get("scene"), manifest.get("role")) != (
            f"{scene}-teammate-pool-v1", scene, 2):
        raise ValueError("Incompatible teammate pool for this scene")
    members = {}
    for member in manifest["members"]:
        name = member["id"]
        path = (manifest_path.parent / member["path"]).resolve()
        if not isinstance(name, str) or not name or name in members:
            raise ValueError("Invalid or duplicate member ID")
        if manifest_path.parent not in path.parents or path in members.values():
            raise ValueError("Invalid or duplicate member path")
        if hashlib.sha256((path / "agent.th").read_bytes()).hexdigest() != member["sha256"]:
            raise ValueError(f"Frozen actor checksum mismatch for {name}")
        members[name] = path
    if not members:
        raise ValueError("The pool does not have a published member yet")
    return list(members.items())


def load_fastap_groups(scene, manifest_path):
    """Use the published solo-role weights. Pair adjacent members cyclically in LBF/CN.

    Each active slot constructs a fresh recurrent policy with its original role ID.
    The stable group order alternates singles and pairs for incremental introduction.
    """
    from fastap_agents import load_frozen_agent

    members = verified_pool_members(manifest_path, scene)
    scenario = SCENARIOS[scene]
    policies = []
    for name, path in members:
        agent = load_frozen_agent(path, device="cpu")
        spec = agent.spec
        if (spec.scene_key, spec.control, spec.n_agents, spec.obs_shape, spec.n_actions) != (
                scene, "team", 3, scenario.world.obs_shape, scenario.world.n_actions):
            raise ValueError(f"Incompatible solo teammate checkpoint in {path}")
        policies.append((name, agent.policy_factory(2)))
    groups = []
    for i, (name, factory) in enumerate(policies):
        groups.append(TeammateGroup(name, (factory,)))
        if 2 in scenario.teammate_counts and len(policies) > 1:
            other, other_factory = policies[(i + 1) % len(policies)]
            groups.append(TeammateGroup(f"{name}+{other}", (factory, other_factory)))
    if len({group.name for group in groups}) != len(groups):
        raise ValueError("Member IDs produce ambiguous group names")
    return groups


class FastapRolloutEnv(OpenTeamEnv):
    """Sample one trajectory for a named group, with central targets on each transition.

    reset/observe return ego observations, global state and ego legal actions. The
    rollout collector can keep group identity and central targets out of ego inference.
    Use separate instances for training and evaluation to keep their RNGs independent.
    """

    def __init__(self, scene, groups, *, seed=0, world_options=None):
        scenario = SCENARIOS[scene]
        groups = tuple(groups)
        for group in groups:
            scenario.check_count(len(group.policies))
        super().__init__(partial(scenario.world, **(world_options or {})), seed, None, groups)
        self.max_teammates = max(scenario.teammate_counts)
        self.n_actions = self.world.n_actions
        self.obs_shape = self.world.obs_shape
        self.state_shape = self.world.state_shape
        # This adapter selects the switch schedule per episode. Paired stationary and
        # switching rollouts from equally seeded instances start on identical maps.
        self.set_test_mode(True)

    def observe(self):
        return self.get_obs(), self.get_state(), self.get_avail_actions()

    def add_group(self, group):
        """Make a newly trained group's policies available at the next reset."""
        if not self._done:
            raise RuntimeError("Introduce new teammate groups between training episodes")
        if not 1 <= len(group.policies) <= self.max_teammates:
            raise ValueError("Unsupported generated group size")
        if any(existing.name == group.name for existing in self.groups):
            raise ValueError(f"Duplicate group name: {group.name}")
        self.groups += (group,)

    def reset(self, group_index=None, *, switch_interval=None):
        """Select a group for Algorithm 2 or enable within-episode redraws for Algorithm 3."""
        self.set_switch_interval(switch_interval)
        obs, state = super().reset(group_index=group_index)
        self.last_teammate_actions = None
        return obs, state, self.get_avail_actions()

    def step(self, ego_actions):
        group_index = self.group_index
        reward, done, info = super().step(ego_actions)
        # A redraw on this boundary may already have updated the world or changed
        # its roster. Use the stored inputs and actions of the policies that acted.
        actions = self.last_joint_actions[self.n_agents:]
        count = len(actions)
        teammate_obs = np.zeros((self.max_teammates, self.obs_shape), dtype=np.float32)
        teammate_obs[:count] = self.last_teammate_obs
        teammate_actions = np.full(self.max_teammates, -1, dtype=np.int64)
        teammate_actions[:count] = actions
        self.last_teammate_actions = teammate_actions
        return reward, done, dict(info, group_index=group_index, teammate_obs=teammate_obs,
                                  teammate_actions=teammate_actions)
