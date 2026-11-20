"""Paired LBF evaluation for a frozen three-player team's single teammate."""

from operator import index
from pathlib import Path

import numpy as np

from fastap_agents import load_frozen_agent
from fastap_scenarios import make_world
from fastap_scenarios.policies import random_legal


def _episode_return(seed, factories):
    world = make_world("lbf", seed=seed)
    try:
        world.reset(n_teammates=1)
        policies = [factory() for factory in factories]
        # Changing one role's action sampling cannot advance another role's RNG.
        streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)]
        total = 0.0
        done = False
        while not done:
            actions = [act(obs, mask, rng) for act, obs, mask, rng
                       in zip(policies, world.observations(), world.avail_actions(), streams)]
            reward, done, _ = world.step(actions)
            total += reward
        return float(total)
    finally:
        world.close()


def evaluate_pair(ego_checkpoint, teammate_checkpoint, seeds, *, random_teammate=False):
    """Evaluate slots 0/1 with fixed slot 2 on the supplied independent maps.

    ego_checkpoint may be the original three-player team or a trained pair of
    egos. A random teammate uses the same legal action mask as the frozen role.
    Success means collecting at least one food. The standard error is estimated
    across maps, and is undefined (None) with a single map.
    """
    seeds = [index(seed) for seed in seeds]
    if not seeds or min(seeds) < 0 or len(set(seeds)) != len(seeds):
        raise ValueError("Supply distinct, nonnegative seeds for at least one evaluation map")
    ego = load_frozen_agent(ego_checkpoint, device="cpu")
    if ego.spec.scene_key != "lbf" or (ego.spec.control, ego.spec.n_agents) not in {("team", 3), ("ego", 2)}:
        raise ValueError("The ego checkpoint must be from a three-player LBF team or two LBF egos")
    factories = [ego.policy_factory(0), ego.policy_factory(1)]
    if random_teammate:
        factories.append(lambda: random_legal)
    else:
        teammate = (ego if Path(ego_checkpoint).resolve() == Path(teammate_checkpoint).resolve()
                    else load_frozen_agent(teammate_checkpoint, device="cpu"))
        if (teammate.spec.scene_key, teammate.spec.control, teammate.spec.n_agents) != ("lbf", "team", 3):
            raise ValueError("The frozen teammate must come from slot 2 of a three-player LBF team")
        factories.append(teammate.policy_factory(2))

    returns = [_episode_return(seed, factories) for seed in seeds]
    n = len(returns)
    return {
        "n_episodes": n,
        "seeds": seeds,
        "episode_returns": returns,
        "mean_return": float(np.mean(returns)),
        "return_se": float(np.std(returns, ddof=1) / np.sqrt(n)) if n > 1 else None,
        "success_rate": sum(value > 0 for value in returns) / n,
    }
