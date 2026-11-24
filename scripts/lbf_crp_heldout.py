"""Paired, unseen-map trajectory and cross-play evaluation for LBF CRP partitions."""

from collections import Counter
from contextlib import closing
from pathlib import Path

import numpy as np
import torch

from fastap_agents import load_frozen_agent
from fastap_methods.fastap_training import _batch, _episode
from fastap_scenarios.fastap_rollout import FastapRolloutEnv, load_fastap_groups, verified_pool_members
from scripts.plot_lbf_crp import aggregate_crossplay, project_embeddings


class _FrozenEgoPair:
    """Adapt the two original QMIX slot policies to the shared rollout collector."""

    def __init__(self, actor, rng):
        self.actor = actor
        self.rng = rng

    def parameters(self):
        return self.actor.model.parameters()

    def act(self, obs, previous, available, hidden):
        if hidden is None:
            hidden = [self.actor.policy_factory(i)() for i in (0, 1)]
        # Each slot policy carries its own previous action and recurrent state.
        actions = [policy(obs[0, i].numpy(), available[0, i].numpy(), self.rng)
                   for i, policy in enumerate(hidden)]
        return torch.tensor([actions]), hidden


def collect_heldout(pool_path, training_probes, *, map_seeds=range(978000, 978016), ego_seed_start=338000):
    """Collect both probes on identical starts; reject overlap with training initial states.

    training_probes contains (states, actions, slots, filled) as stored by TeammateCRP.
    """
    map_seeds = list(map_seeds)
    if not map_seeds or len(set(map_seeds)) != len(map_seeds):
        raise ValueError("Expected distinct held-out map seeds")
    members = verified_pool_members(Path(pool_path), 'lbf')
    names = [name for name, _ in members]
    groups = [g for g in load_fastap_groups('lbf', pool_path) if len(g.policies) == 1]
    if names != [f'T{i}' for i in range(1, 9)] or names != [g.name for g in groups]:
        raise ValueError("Expected the T1–T8 single-teammate policies in publication order")
    actor = load_frozen_agent(members[0][1], device='cpu')
    seen = {probe[0][0].detach().cpu().numpy().tobytes() for probe in training_probes}
    if not seen:
        raise ValueError("Supply training probes to check held-out map independence")
    trials = {policy: [] for policy in ('random_legal', 'fixed_T1_ego')}
    for map_index, map_seed in enumerate(map_seeds):
        starts = set()
        for i in range(len(groups)):
            for policy, episodes in trials.items():
                rng = np.random.default_rng(ego_seed_start + map_index)
                ego = _FrozenEgoPair(actor, rng) if policy == 'fixed_T1_ego' else None
                with closing(FastapRolloutEnv('lbf', groups, seed=map_seed)) as env:
                    ep = _episode(env, i, ego, rng)
                starts.add(ep['state'][0].numpy().tobytes())
                episodes.append({key: ep[key] for key in
                                 ('state', 'teammate_actions', 'teammate_mask', 'filled', 'reward')})
        if len(starts) != 1 or not starts.isdisjoint(seen):
            raise ValueError(f"Unpaired or repeated train/evaluation start for map seed {map_seed}")
        seen.update(starts)
    return dict(map_seeds=map_seeds, ego_seed_start=ego_seed_start, members=names,
                trials={policy: _batch(episodes) for policy, episodes in trials.items()})


def trajectory_silhouette(points, labels):
    """Mean Euclidean silhouette over trajectories, with zero for singleton samples."""
    points, labels = np.asarray(points, dtype=np.float64), np.asarray(labels)
    if len(points) != len(labels) or points.ndim != 2 or not np.isfinite(points).all():
        raise ValueError("Expected one finite vector per trajectory label")
    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    if not 1 < len(counts) < len(points):
        return None
    values = []
    for start in range(0, len(points), 32):
        own = inverse[start:start + 32]
        distances = np.linalg.norm(points[start:start + 32, None, :] - points[None, :, :], axis=2)
        sums = np.column_stack([distances[:, inverse == j].sum(axis=1) for j in range(len(counts))])
        rows = np.arange(len(own))
        within = sums[rows, own] / np.maximum(counts[own] - 1, 1)
        means = sums / counts
        means[rows, own] = np.inf
        between = means.min(axis=1)
        scale = np.maximum(within, between)
        scores = np.divide(between - within, scale, out=np.zeros_like(scale), where=scale > 0)
        scores[counts[own] == 1] = 0.
        values.extend(scores)
    return float(np.mean(values))


@torch.no_grad()
def evaluate_heldout(model, assignments, heldout, crossplay, *, include_rows=False):
    """Use each member's irrevocable online label and the variant's frozen final encoder."""
    names = heldout['members']
    if set(names) != set(assignments) or crossplay['ids'] != names:
        raise ValueError("Assignments, probes and directed cross-play must share the same members")
    if (crossplay['rows'], crossplay['columns']) != ('original trained partners in slots 0/1',
                                                    'evaluated frozen teammate in slot 2'):
        raise ValueError("Incorrect directed cross-play axes")
    sizes = Counter(assignments.values())
    summary = dict(num_clusters=len(sizes), member_singleton_clusters=sum(n == 1 for n in sizes.values()),
                   crossplay=aggregate_crossplay(crossplay, assignments)['summaries'], heldout={})
    names_by_row = names * len(heldout['map_seeds'])
    seeds_by_row = np.repeat(heldout['map_seeds'], len(names))
    labels = [assignments[name] for name in names_by_row]
    training = model.training
    device = next(model.parameters()).device
    model.eval()
    try:
        for policy, batch in heldout['trials'].items():
            parts = []
            for start in range(0, len(labels), 32):
                sample = [batch[key][start:start + 32].to(device) for key in
                          ('state', 'teammate_actions', 'teammate_mask', 'filled')]
                parts.append(model.encode(*sample).cpu().double().numpy())
            points = np.concatenate(parts)
            if points.shape != (len(labels), 16) or not np.isfinite(points).all():
                raise ValueError("Expected one finite 16D behavior embedding per held-out trajectory")
            silhouette = trajectory_silhouette(points, labels)
            rows = [dict(teammate=name, cluster=label, episode=int(seed), embedding=point.tolist())
                    for name, label, seed, point in zip(names_by_row, labels, seeds_by_row, points)]
            if np.all(points == points[0]):
                pca_score, pca_variance = silhouette, None
            else:
                pca = project_embeddings(rows, assignments)
                pca_score = trajectory_silhouette([row['xyz'] for row in pca['rows']], labels)
                pca_variance = sum(pca['explained_variance_ratio'])
            returns = batch['reward'].sum(1).double().numpy()
            summary['heldout'][policy] = dict(silhouette_16d=silhouette, silhouette_pca3=pca_score,
                                              pca3_variance_fraction=pca_variance, episodes=len(returns),
                                              mean_return=float(returns.mean()), any_food=int((returns > 0).sum()),
                                              all_food=int(np.isclose(returns, 1.).sum()))
            if include_rows:
                summary['heldout'][policy]['rows'] = rows
    finally:
        model.train(training)
    return summary
