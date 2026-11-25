"""Blind behavior-feature and known-policy controls for the labeled LBF rule benchmark."""

import argparse
from itertools import combinations
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastap_methods.lbf_probe_starts import pre_load_moves
from scripts.audit_lbf_crp_order import compare_partitions
from scripts.evaluate_lbf_crp_prediction import digest
from scripts.lbf_rule_probes import RULES, action_probabilities


def behavior_events(obs, available, action):
    """Return successes and opportunities for two decisions in the observed local state."""
    load_allowed = bool(available[5])
    successes = np.array([int(load_allowed and action == 5), 0])
    opportunities = np.array([int(load_allowed), 0])
    closer = pre_load_moves(obs, available)
    if closer.any():
        opportunities[1] = 1
        successes[1] = int(1 <= action <= 4 and closer[action-1])
    return successes, opportunities


def average_linkage(features, known_k):
    """Merge the closest cluster pair using mean pairwise Euclidean distance."""
    if not 1 <= known_k <= len(features):
        raise ValueError('Cluster count is outside the number of agents')
    distance = np.linalg.norm(features[:, None] - features[None, :], axis=-1)
    clusters = [(i,) for i in range(len(features))]
    while len(clusters) > known_k:
        _, left, right = min((distance[np.ix_(a, b)].mean(), a, b)
                             for a, b in combinations(clusters, 2))
        clusters.remove(left)
        clusters.remove(right)
        clusters.append(tuple(sorted((*left, *right))))
        clusters.sort()
    return clusters


def measure_split(data, sidecar):
    """Only the known-policy arm sees the candidate rules. Neither arm receives labels."""
    members = data['members']
    seeds = data['map_seeds']
    batch = data['batch']
    present = batch['filled'].bool().unsqueeze(-1) & batch['teammate_mask'].bool()
    if not torch.equal(present.sum(-1), batch['filled'].long()):
        raise ValueError('Expected one observed teammate for every real step')
    actions = batch['teammate_actions'].masked_fill(~present, 0).sum(-1).numpy()
    valid = batch['filled'].numpy().astype(bool)
    n_rows, horizon = valid.shape
    if n_rows != len(members) * len(seeds) or not valid[:, 0].all():
        raise ValueError('Expected a map-major grid of nonempty probes')
    starts = batch['state'][:, 0].reshape(len(seeds), len(members), -1)
    if not torch.equal(starts, starts[:, :1].expand_as(starts)):
        raise ValueError('Initial states must match for all members on each map')
    obs = np.asarray(sidecar['obs'])[:, :horizon]
    masks = np.asarray(sidecar['available'])[:, :horizon].astype(bool)
    if obs.shape != (*valid.shape, 21) or masks.shape != (*valid.shape, 6):
        raise ValueError('The sidecar must contain local obs and legal actions for the observed teammate')
    counts = np.zeros((n_rows, 2, 2), dtype=int)
    likelihoods = np.zeros((n_rows, len(RULES)), dtype=np.float64)
    tv_sum = np.zeros((n_rows, len(RULES), len(RULES)), dtype=np.float64)
    different = np.zeros_like(tv_sum, dtype=int)
    policy_cache = {}
    for row, step in np.argwhere(valid):
        local, available, action = obs[row, step], masks[row, step], actions[row, step]
        if not 0 <= action < 6 or not available[action]:
            raise ValueError('Published action must be legal at this decision')
        key = (local.tobytes(), available.tobytes())
        if key not in policy_cache:
            probs = np.asarray([action_probabilities(rule, local, available) for rule in RULES], dtype=float)
            if (probs.shape != (len(RULES), 6) or not np.isfinite(probs).all() or (probs < 0).any()
                    or not np.allclose(probs.sum(1), 1) or (probs[:, ~available] != 0).any()):
                raise ValueError('Invalid categorical probabilities for the available actions')
            log_p = np.full(probs.shape, -np.inf)
            np.log(probs, where=probs > 0, out=log_p)
            tv = np.abs(probs[:, None] - probs[None, :]).sum(-1) / 2
            policy_cache[key] = log_p, tv
        log_p, tv = policy_cache[key]
        likelihoods[row] += log_p[:, action]
        tv_sum[row] += tv
        different[row] += tv > 1e-12
        successes, opportunities = behavior_events(local, available, action)
        counts[row, 0] += successes
        counts[row, 1] += opportunities
    fields = dict(counts=counts, likelihoods=likelihoods, tv_sum=tv_sum, different=different,
                  steps=valid.sum(1))
    return {k: x.reshape(len(seeds), len(members), *x.shape[1:]) for k, x in fields.items()}


def controls(measured, members, maps):
    counts = measured['counts'][maps].sum(0)
    # Symmetric Jeffreys pseudocounts give a neutral estimate in an unobserved context.
    rates = (counts[:, 0] + .5) / (counts[:, 1] + 1)
    clusters = average_linkage(rates, len(RULES))
    labels = {members[j]: f'C{i + 1}' for i, block in enumerate(clusters) for j in block}
    scores = measured['likelihoods'][maps].sum(0)
    oracle = {}
    for member, row in zip(members, scores):
        best = row.max()
        if not np.isfinite(best):
            raise ValueError(f'No generating rule supports every observed action of {member}')
        # An exact tie gives an unidentified type; do not break ties using member names.
        winners = np.isclose(row, best, rtol=0, atol=1e-10)
        oracle[member] = dict(best_candidates=[r for r, pick in zip(RULES, winners) if pick],
                              log_likelihood=[float(x) if np.isfinite(x) else None for x in row])
    decisions = measured['steps'][maps].sum()
    return dict(blind=dict(assignments=labels, features=rates.tolist(), successes=counts[:, 0].tolist(),
                           opportunities=counts[:, 1].tolist()),
                known_policy=dict(members=oracle, decisions=int(decisions),
                                  mean_pairwise_total_variation=(measured['tv_sum'][maps].sum((0, 1))/decisions).tolist(),
                                  decisions_with_distinct_action_distribution=measured['different'][maps].sum((0, 1)).tolist()))


def reveal(result, labels, members):
    if set(members) != set(labels) or set(labels.values()) != set(RULES):
        raise ValueError('Truth of the three rule families must cover all and only blind members')
    pairs = list(combinations(members, 2))
    blind = result['blind']
    blind['partition_vs_truth'] = compare_partitions(blind['assignments'], labels, pairs)
    oracle = result['known_policy']
    unique_correct = [name for name in members if oracle['members'][name]['best_candidates'] == [labels[name]]]
    oracle['unique_correct'] = len(unique_correct)
    oracle['members_with_unique_correct_rule'] = unique_correct


def run(directory):
    sidecar = torch.load(directory / 'observations.pt', map_location='cpu', weights_only=True)
    results = {}
    members = None
    for split in ('train', 'query'):
        data = torch.load(directory / f'{split}.pt', map_location='cpu', weights_only=True)
        if members is None:
            members = data['members']
        if data['members'] != members:
            raise ValueError('Split members must have the same order')
        seeds = data['map_seeds']
        measured = measure_split(data, sidecar[split])
        results[split] = dict(map_seeds=seeds, **controls(measured, members, slice(None)))
        if split == 'query':
            results[split]['eight_map_blocks'] = [
                dict(map_seeds=seeds[start:start+8], **controls(measured, members, slice(start, start+8)))
                for start in range(0, len(seeds)-7, 8)]
    # Reveal labels only after all feature estimates, groupings, and oracle scores are fixed.
    truth = json.loads((directory / 'truth.json').read_text())
    labels = truth['labels']
    source_sha256 = {n: digest(directory/n) for n in ('train.pt', 'query.pt', 'observations.pt')}
    if any(truth['files_sha256'][k] != sha for k, sha in source_sha256.items()):
        raise ValueError('The trajectory data or observation sidecar differs from the generated dataset')
    source_sha256['truth.json'] = digest(directory/'truth.json')
    for value in results.values():
        reveal(value, labels, members)
        for block in value.get('eight_map_blocks', []):
            reveal(block, labels, members)
    report = dict(members=members, feature_order=['P(LOAD | legal LOAD)',
                  'P(move reduces nearest visible food L1 distance | no legal LOAD, both approaching '
                  'and nonapproaching cardinal moves are available)'],
                  blind_method='Two prespecified domain features with Jeffreys Beta(0.5,0.5) smoothing. '
                               'Offline average-linkage clustering in unscaled Euclidean space with supplied K=3. '
                               'Fit each split separately, without identity labels or rewards.',
                  known_policy_method='Provided three generator distributions in advance. Evaluate exact observed '
                                      'actions given the same local observation and legality mask. Sum conditional '
                                      'log likelihoods by anonymous member, uniform rule prior. null represents '
                                      'zero likelihood. Multiple maximizers remain unresolved.',
                  rule_order=list(RULES), results=results, source_sha256=source_sha256)
    target = directory/'positive_control.json'
    with TemporaryDirectory(dir=directory) as temp:
        staged = Path(temp)/target.name
        staged.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        try:
            os.link(staged, target)
        except FileExistsError:
            if target.read_bytes() != staged.read_bytes():
                raise
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    if ROOT/'outputs' not in directory.parents or os.getenv('FASTAP_ROOT') != str(ROOT):
        parser.error('Source scripts/epymarl-env.sh and select a project output directory')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.set_num_threads(1)
    report = run(directory)
    for split, row in report['results'].items():
        print(split, row['blind']['partition_vs_truth']['ari'], row['known_policy']['unique_correct'])
