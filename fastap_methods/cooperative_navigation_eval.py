"""Test a single frozen navigation teammate through the normal two-ego environment."""

import numpy as np

from fastap_agents import load_frozen_agent
from fastap_scenarios import TeammateGroup, make_env
from fastap_scenarios.policies import random_legal
from fastap_scenarios.torch_cooperative_navigation import TorchCooperativeNavigation
from .particle_eval import paired_return_difference, validation_returns


def check_cn_adapter(checkpoint, seeds):
    """Replay the exported policy via both public controllers on identical starts."""
    seeds = tuple(seeds)
    if not seeds:
        raise ValueError("Provide at least one start for adapter comparison")
    frozen = load_frozen_agent(checkpoint, device="cpu")
    spec = frozen.spec
    if (spec.scene_key, spec.control, spec.n_agents) != ("cn", "team", 3):
        raise ValueError("Expected a single CN teammate trained in a three-player team")
    group = TeammateGroup("T1", (frozen.policy_factory(2),))
    checked = 0
    for seed in seeds:
        ego = make_env("cn", control="ego", seed=seed, teammate_groups=[group])
        joint = make_env("cn", control="team", seed=seed, n_teammates=1)
        try:
            for env in (ego, joint):
                env.set_test_mode(True)
                env.reset()
            if ego.n_agents != 2 or joint.n_agents != spec.n_agents:
                raise ValueError("Unexpected number of externally controlled agents")
            policies = [frozen.policy_factory(i)() for i in range(spec.n_agents)]
            rng = np.random.default_rng(seed)
            for step in range(ego.episode_limit):
                obs, masks = joint.get_obs(), joint.get_avail_actions()
                np.testing.assert_array_equal(ego.get_obs(), obs[:2])
                np.testing.assert_array_equal(ego.get_avail_actions(), masks[:2])
                actions = [p(o, m, rng) for p, o, m in zip(policies, obs, masks)]
                r1, d1, _ = ego.step(actions[:2])
                r2, d2, _ = joint.step(actions)
                expected_done = step + 1 == ego.episode_limit
                if ego.last_joint_actions != tuple(actions) or r1 != r2 or d1 != expected_done or d2 != expected_done:
                    raise ValueError("CN controllers disagreed on action, reward or timeout")
                np.testing.assert_array_equal(ego.get_state(), joint.get_state())
                checked += 1
        finally:
            ego.close()
            joint.close()
    return dict(passed=True, seeds=list(seeds), matched_transitions=checked, external_ego_slots=2,
                frozen_teammate_slot=2, obs_size=spec.obs_shape, state_size=TorchCooperativeNavigation.state_shape,
                actions_per_agent=spec.n_actions)


def validate_cn(actor, spec, *, n_maps, seed, device):
    world = TorchCooperativeNavigation(n_maps, device=device, seed=seed)
    values = validation_returns(actor, spec, world)
    return dict(n_maps=n_maps, map_batch_seed=seed, trained=dict(mean_return=float(values.mean())))


def _rollout(seed, ego, group):
    env = make_env("cn", control="ego", seed=seed, n_teammates=1, teammate_groups=[group])
    try:
        env.set_test_mode(True)
        obs, _ = env.reset()
        policies = [ego.policy_factory(i)() for i in range(env.n_agents)]
        rng = np.random.default_rng(seed)
        total_return = distance_sum = 0.
        collisions = 0
        for step in range(env.episode_limit):
            actions = [p(o, m, rng) for p, o, m in zip(policies, obs, env.get_avail_actions())]
            reward, done, info = env.step(actions)
            if done != (step + 1 == env.episode_limit) or len(env.last_joint_actions) != 3:
                raise ValueError("Unexpected CN episode length or roster")
            total_return += reward
            distance_sum += info["landmark_distance_sum"]
            collisions += info["agent_collisions"]
            obs = env.get_obs()
        if not np.isclose(total_return, -distance_sum - env.world.collision_penalty * collisions):
            raise ValueError("CN reward disagrees with distance and collision totals")
        return total_return, distance_sum / env.episode_limit, collisions
    finally:
        env.close()


def evaluate_cn_cases(cases, seeds):
    scores, returns = {}, {}
    for case, (partners, teammate, random) in cases.items():
        ego = load_frozen_agent(partners, device="cpu")
        if (ego.spec.scene_key, ego.spec.control, ego.spec.n_agents) != ("cn", "team", 3):
            raise ValueError("Expected a three-player CN team")
        if random:
            factory = lambda: random_legal
        else:
            mate = load_frozen_agent(teammate, device="cpu")
            if mate.spec != ego.spec:
                raise ValueError("CN partner and teammate specifications must match")
            factory = mate.policy_factory(2)
        group = TeammateGroup("T1", (factory,))
        data = np.array([_rollout(seed, ego, group) for seed in seeds])
        returns[case] = data[:, 0].tolist()
        scores[case] = dict(mean_return=float(data[:, 0].mean()),
                            mean_landmark_distance_sum_per_step=float(data[:, 1].mean()),
                            mean_pair_collisions_per_episode=float(data[:, 2].mean()))
    paired = {k: paired_return_difference(returns["trained"], v) for k, v in returns.items() if k != "trained"}
    return dict(scores=scores, episode_returns=returns, paired=paired,
                evidence_for_teammate=all(paired[k]["bootstrap_95_ci"][0] > 0 for k in ("initial_T1", "random_T1")),
                evidence_for_team=paired["initial_team"]["bootstrap_95_ci"][0] > 0)
