"""Select one PP hunter on validation starts; assess its contribution on paired scalar games."""

import numpy as np

from fastap_agents import load_frozen_agent
from .particle_eval import paired_return_difference, validation_returns
from fastap_scenarios.particle import overlaps
from fastap_scenarios.policies import random_legal
from fastap_scenarios.predator_prey import PredatorPreyWorld
from fastap_scenarios.torch_predator_prey import TorchPredatorPrey


def validate_pp(actor, spec, *, n_maps, seed, device):
    world = TorchPredatorPrey(n_maps, device=device, seed=seed)
    values = validation_returns(actor, spec, world)
    return dict(n_maps=n_maps, map_batch_seed=seed,
                trained=dict(mean_return=float(values.mean()), any_tag_rate=float((values > 0).mean())))


def rollout_contacts(seed, factories):
    """Count each predator's contacts in one scalar episode from the specified start."""
    game = PredatorPreyWorld(seed=seed)
    try:
        game.reset()
        policies = [factory() for factory in factories]
        rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)]
        totals = np.zeros(3, dtype=np.int64)
        done = False
        while not done:
            actions = [p(o, mask, rng) for p, o, mask, rng in
                       zip(policies, game.observations(), game.avail_actions(), rngs)]
            reward, done, _ = game.step(actions)
            tagged = [overlaps(body, game.prey) for body in game.players]
            if reward != 10 * sum(tagged):
                raise ValueError("Measured tags disagree with the environment reward")
            totals += tagged
        return totals
    finally:
        game.close()


def evaluate_pp_cases(cases, seeds):
    reports, returns, per_role = {}, {}, {}
    for case, (partners, teammate, random) in cases.items():
        ego = load_frozen_agent(partners, device="cpu")
        if (ego.spec.scene_key, ego.spec.control, ego.spec.n_agents) != ("pp", "team", 3):
            raise ValueError("Expected a three-predator PP self-play checkpoint")
        factories = [ego.policy_factory(0), ego.policy_factory(1)]
        if random:
            factories.append(lambda: random_legal)
        else:
            mate = load_frozen_agent(teammate, device="cpu")
            if mate.spec != ego.spec:
                raise ValueError("The PP teammate must have the same actor specification as its partners")
            factories.append(mate.policy_factory(2))
        counts = np.stack([rollout_contacts(seed, factories) for seed in seeds])
        totals = counts.sum(axis=1)
        returns[case] = (10 * totals).tolist()
        per_role[case] = counts.tolist()
        reports[case] = dict(mean_return=float(10 * totals.mean()), any_tag=int((totals > 0).sum()),
                             mean_contacts=float(totals.mean()), teammate_mean_contacts=float(counts[:, 2].mean()),
                             multiple_predators_tagged=int(((counts > 0).sum(1) >= 2).sum()))
    comparisons = {other: paired_return_difference(returns["trained"], values)
                   for other, values in returns.items() if other != "trained"}
    return dict(scores=reports, paired=comparisons, episode_returns=returns, per_role_contacts=per_role,
                evidence_for_teammate=all(comparisons[k]["bootstrap_95_ci"][0] > 0
                                          for k in ("random_T1", "initial_T1")),
                evidence_for_team=comparisons["initial_team"]["bootstrap_95_ci"][0] > 0)
