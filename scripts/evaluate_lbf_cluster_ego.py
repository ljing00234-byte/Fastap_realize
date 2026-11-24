"""Test whether a known teammate cluster helps select two frozen LBF egos.

collect rolls out every candidate on identical boards with every teammate, once
per split. score chooses policies using the calibration split and evaluates the
resulting rules on the untouched test split, without fitting any actor weights.
"""

import argparse
from contextlib import closing
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastap_agents import load_frozen_agent
from fastap_methods.lbf_pool_ego_eval import _stationary_returns
from fastap_methods.particle_eval import paired_return_difference
from fastap_scenarios.fastap_rollout import FastapRolloutEnv, load_fastap_groups, verified_pool_members
from fastap_scenarios.torch_lbf_ego import TorchLBFEgo


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ego_candidates(runs):
    records, actors = [], []
    for directory in runs:
        status = read(directory / 'status.json')
        cfg = read(directory / 'config.json')
        if status['phase'] != 'complete' or cfg['scene'] != 'lbf' or cfg['control'] != 'ego':
            raise ValueError(f'Expected a completed LBF ego run: {directory}')
        checkpoint = directory / status['selected_checkpoint']
        actor = load_frozen_agent(checkpoint, device='cpu')
        if (actor.spec.n_agents, actor.spec.control, actor.spec.scene_key) != (2, 'ego', 'lbf'):
            raise ValueError(f'Expected a two-role LBF ego checkpoint: {checkpoint}')
        records.append(dict(run=str(directory.relative_to(ROOT)), seed=cfg['seed'],
                            checkpoint=status['selected_checkpoint'], sha256=sha(checkpoint / 'agent.th'),
                            training_steps=status['training_steps'], training_pool=cfg['teammate_pool'],
                            training_teammates=read(directory / 'protocol.json')['pool_members']))
        actors.append(actor)
    if len({rec['sha256'] for rec in records}) != len(records):
        raise ValueError('Each candidate must be a different ego policy')
    return records, actors


@torch.inference_mode()
def rollout(first_seed, count, actors, teammates, groups):
    # Use the public test-mode reset, including its child RNG stream. Calling
    # ForagingWorld(seed).reset(1) directly would give a different board.
    fields, positions, levels, hashes = [], [], [], []
    for seed in range(first_seed, first_seed + count):
        with closing(FastapRolloutEnv('lbf', groups, seed=seed)) as env:
            _, state, _ = env.reset(0)
            fields.append(env.world.field.copy())
            positions.append([p.position for p in env.world.players])
            levels.append([p.level for p in env.world.players])
            hashes.append(hashlib.sha256(np.asarray(state, dtype='<f4').tobytes()).hexdigest())
    boards = dict(field=np.stack(fields), positions=np.array(positions), levels=np.array(levels))
    returns = np.empty((len(actors), len(teammates), count), dtype=np.float32)
    for column, teammate in enumerate(teammates):
        world = TorchLBFEgo(count, 'cpu', teammates=[teammate])
        teammate_ids = torch.zeros(count, dtype=torch.long)
        for row, ego in enumerate(actors):
            world.load_snapshot(**boards, teammate_ids=teammate_ids)
            returns[row, column] = _stationary_returns(ego.model, ego.spec, world).numpy()
    return returns, hashes


def collect(args):
    torch.set_num_threads(1)
    members = verified_pool_members(args.pool, 'lbf')
    teammates = [load_frozen_agent(path, device='cpu') for _, path in members]
    groups = [g for g in load_fastap_groups('lbf', args.pool) if len(g.policies) == 1]
    candidates, actors = ego_candidates([directory.resolve() for directory in args.ego_runs])
    seeds = dict(calibration=(args.calibration_seed, args.calibration_maps),
                 test=(args.test_seed, args.test_maps))
    if (any(first < 0 or count < 1 for first, count in seeds.values()) or
            max(args.calibration_seed, args.test_seed) <
            min(args.calibration_seed + args.calibration_maps, args.test_seed + args.test_maps)):
        raise ValueError('Use disjoint, nonempty calibration and test seed ranges')
    meta = dict(pool_manifest_sha256=sha(args.pool), members=[name for name, _ in members],
                teammate_sha256={name: sha(path / 'agent.th') for name, path in members}, egos=candidates,
                game='LBF, one stationary frozen teammate in slot 2; two greedy egos in slots 0/1',
                board_generator='FastapRolloutEnv(lbf, singles, seed).reset(0); identical boards for every ego and teammate',
                reward='Fraction of initial food collected per episode', splits={})
    arrays = {}
    for split, (first, count) in seeds.items():
        arrays[split], hashes = rollout(first, count, actors, teammates, groups)
        meta['splits'][split] = dict(first_seed=first, count=count, initial_state_sha256=hashes)
    if set(meta['splits']['calibration']['initial_state_sha256']) & set(meta['splits']['test']['initial_state_sha256']):
        raise ValueError('Calibration and test share an initial state')
    args.returns.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=args.returns.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            np.savez_compressed(stream, metadata=json.dumps(meta), **arrays)
            stream.flush()
            os.link(temporary, args.returns)
        finally:
            temporary.unlink(missing_ok=True)
    print(f'Saved {sum(a.size for a in arrays.values())} paired ego episodes to {args.returns}')


def score_partition(calibration, test, labels):
    """All arrays are [ego, teammate, map]. Returns are paired by whole map."""
    labels = np.asarray(labels)
    members = np.arange(len(labels))
    if (calibration.ndim != 3 or test.ndim != 3 or calibration.shape[:2] != test.shape[:2]
            or calibration.shape[1] != len(labels) or len(labels) < 2):
        raise ValueError('Return arrays and partition must cover the same teammates and egos')
    mean_calibration = calibration.astype(np.float64).mean(axis=2)
    by_cluster = {c: int(mean_calibration[:, labels == c].mean(axis=1).argmax()) for c in np.unique(labels)}
    selected = np.array([by_cluster[c] for c in labels])
    global_ego = int(mean_calibration.mean(axis=1).argmax())
    matched = test[selected, members].astype(np.float64)
    global_scores = test[global_ego].astype(np.float64)
    # Every member receives each other member's label once. A shift can leave
    # the label unchanged for members of the same cluster.
    shifts = list(range(1, len(labels)))
    shuffled = np.stack([test[np.roll(selected, shift), members] for shift in shifts]).mean(axis=0, dtype=np.float64)
    compared = dict(correct_cluster=matched, shuffled_cluster=shuffled, no_cluster=global_scores)
    per_map = {key: scores.mean(axis=0) for key, scores in compared.items()}
    return dict(cluster_ego_indices=by_cluster, global_ego_index=global_ego,
                fraction_shuffled_labels_changed=float(np.mean([np.roll(labels,k) != labels for k in shifts])),
                fraction_shuffled_ego_choices_changed=float(np.mean([np.roll(selected,k) != selected for k in shifts])),
                mean_return={key: float(values.mean()) for key, values in per_map.items()},
                mean_by_teammate={key: values.mean(axis=1).tolist() for key, values in compared.items()},
                correct_minus_global=paired_return_difference(per_map['correct_cluster'], per_map['no_cluster']),
                correct_minus_shuffled=paired_return_difference(per_map['correct_cluster'], per_map['shuffled_cluster']),
                size_matched_partitions=position_among_partitions(mean_calibration, test, labels,
                                                                  per_map['correct_cluster']))


def equal_size_partitions(members, sizes):
    """Enumerate unlabeled partitions once by placing the smallest remaining member first."""
    if not members:
        yield ()
        return
    for size in sorted(set(sizes)):
        remaining_sizes = list(sizes)
        remaining_sizes.remove(size)
        for others in combinations(members[1:], size - 1):
            block = (members[0], *others)
            remaining = tuple(member for member in members if member not in block)
            for tail in equal_size_partitions(remaining, remaining_sizes):
                yield (block, *tail)


def position_among_partitions(calibration_means, test, labels, observed):
    """Refit the same selection rule for every partition with the observed cluster sizes."""
    members = tuple(range(len(labels)))
    sizes = sorted(np.unique(labels, return_counts=True)[1].tolist())
    selected = np.empty(len(members), dtype=int)
    selections = np.zeros(calibration_means.shape, dtype=int)
    test_means = test.mean(axis=2, dtype=np.float64)
    null_means = []
    best_by_block = {}
    for partition in equal_size_partitions(members, sizes):
        for block in partition:
            if block not in best_by_block:
                best_by_block[block] = calibration_means[:, block].mean(axis=1).argmax()
            selected[list(block)] = best_by_block[block]
        selections[selected, members] += 1
        null_means.append(test_means[selected, members].mean())
    count = len(null_means)
    random_map_means = np.einsum('etm,et->m', test, selections, dtype=np.float64) / (count * len(members))
    distances = np.asarray(null_means) - observed.mean()
    ties = np.isclose(distances, 0., rtol=0., atol=1e-12)
    less = int(np.count_nonzero((distances < 0) & ~ties))
    equal = int(ties.sum())
    return dict(cluster_sizes=sizes, unique_unlabeled_partitions=count, includes_observed_partition=True,
                mean_random_partition_return=float(random_map_means.mean()),
                correct_minus_random=paired_return_difference(observed, random_map_means),
                partitions_below=less, partitions_tied=equal, partitions_above=count-less-equal,
                descriptive_midrank_percentile=100. * (less + 0.5 * equal) / count)


def prediction_map_indices(path, meta):
    """Verify the actual initial state of each prediction query, for every partner trial."""
    data = torch.load(path, map_location='cpu', weights_only=True)
    if data['members'] != meta['members'] or data['pool_sha256'] != meta['pool_manifest_sha256']:
        raise ValueError('Prediction and ego evaluation have different teammate pools')
    split = meta['splits']['test']
    indices = np.asarray(data['map_seeds']['query']) - split['first_seed']
    if np.any(indices < 0) or np.any(indices >= split['count']) or not len(indices):
        raise ValueError('Prediction test maps are not covered by ego evaluation')
    for trial in data['trials'].values():
        states = trial['query']['state'][:, 0].reshape(len(indices), len(meta['members']), -1)
        for i, by_member in zip(indices, states):
            expected = split['initial_state_sha256'][i]
            for state in by_member:
                if hashlib.sha256(np.asarray(state, dtype='<f4').tobytes()).hexdigest() != expected:
                    raise ValueError('Prediction and ego evaluation initial states differ')
    return indices


def score(args):
    with np.load(args.returns, allow_pickle=False) as data:
        meta = json.loads(data['metadata'].item())
        calibration, test = data['calibration'], data['test']
    prediction_maps = prediction_map_indices(args.prediction_heldout, meta) if args.prediction_heldout else None
    reports = []
    for path in args.clustering:
        cluster = read(path)
        assignments = cluster['assignments']
        if (set(assignments) != set(meta['members'])
                or cluster['pool_manifest_sha256'] != meta['pool_manifest_sha256']):
            raise ValueError(f'Partition does not match the evaluated pool: {path}')
        labels = [assignments[name] for name in meta['members']]
        report = score_partition(calibration, test, labels)
        choices = report.pop('cluster_ego_indices')
        report['cluster_ego_training_seeds'] = {k: meta['egos'][v]['seed'] for k, v in choices.items()}
        report['global_ego_training_seed'] = meta['egos'][report.pop('global_ego_index')]['seed']
        reports.append(dict(clustering=str(path.resolve().relative_to(ROOT)), clustering_sha256=sha(path),
                            assignments=assignments, **report))
        if prediction_maps is not None:
            matched = score_partition(calibration, test[:, :, prediction_maps], labels)
            reports[-1]['matched_prediction_B_maps'] = {k: matched[k] for k in (
                'mean_return', 'correct_minus_global', 'correct_minus_shuffled', 'size_matched_partitions')}
    output = dict(protocol=meta, returns_sha256=sha(args.returns),
                  interpretation='Policy selection with known training-time cluster labels for the same teammates. '
                                 'No test rewards used to choose policies; ego weights remain frozen.',
                  cluster_control='Seven cyclic shifts of the eight teammate identities at evaluation; '
                                  'mean over shifts. Repeated labels retain their correct cluster.',
                  partition_control='Enumerate every unlabeled partition with the same cluster sizes. '
                                    'Fit the best ego per block on the same calibration maps, then test. '
                                    'Average random partitions uniformly on each test map. Descriptive '
                                    'percentile gives ties half credit; it is not a significance test.',
                  confidence_intervals='10,000 paired bootstrap resamples of whole test maps (seed 49711), '
                                       'keeping all teammates and conditions for a sampled map together.',
                  results=reports)
    if prediction_maps is not None:
        output['matched_prediction_query'] = dict(heldout_sha256=sha(args.prediction_heldout),
                                                  count=len(prediction_maps), all_initial_states_verified=True,
                                                  test_indices=prediction_maps.tolist())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode='w', dir=args.output.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(output, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.link(temporary, args.output)
        finally:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    collect_parser = modes.add_parser('collect')
    collect_parser.add_argument('pool', type=Path)
    collect_parser.add_argument('returns', type=Path)
    collect_parser.add_argument('--ego-runs', type=Path, nargs='+',
                                default=[ROOT / 'outputs' / f'lbf_pool_ego_qmix_{i:02d}' for i in range(1, 6)])
    collect_parser.add_argument('--calibration-seed', type=int, default=6491000)
    collect_parser.add_argument('--test-seed', type=int, default=6720000)
    collect_parser.add_argument('--calibration-maps', type=int, default=256)
    collect_parser.add_argument('--test-maps', type=int, default=1024)
    score_parser = modes.add_parser('score')
    score_parser.add_argument('returns', type=Path)
    score_parser.add_argument('output', type=Path)
    score_parser.add_argument('clustering', type=Path, nargs='+')
    score_parser.add_argument('--prediction-heldout', type=Path, help='Check matching initial states on prediction maps')
    args = parser.parse_args()
    path = (args.returns if args.mode == 'collect' else args.output).resolve()
    if ROOT / 'outputs' not in path.parents or os.environ.get('FASTAP_ROOT') != str(ROOT):
        parser.error('Source scripts/epymarl-env.sh and write results inside project outputs/')
    if path.exists():
        parser.error(f'Output already exists: {path}')
    if args.mode == 'collect':
        collect(args)
    else:
        score(args)


if __name__ == '__main__':
    main()
