"""Compare categorical CRP trained on random or pre-load starts on both held-out map sets."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastap_methods.lbf_probe_starts import pre_load_moves
from scripts.evaluate_lbf_crp_prediction import digest, load
from scripts.evaluate_lbf_rule_prediction import check_fit
from scripts.validate_lbf_rule_clusters import frozen_partitions, partition_score, publish


def coverage(data, sidecar, labels):
    """Count real action opportunities separately for the opening and for the entire episode."""
    members = data['members']
    batch = data['batch']
    opportunities = np.zeros(len(members), dtype=int)
    opening = np.zeros_like(opportunities)
    approaches = np.zeros_like(opportunities)
    opening_approaches = np.zeros_like(opportunities)
    for row, step in batch['filled'].nonzero().tolist():
        index = row % len(members)
        closer = pre_load_moves(sidecar['obs'][row, step].numpy(), sidecar['available'][row, step].numpy())
        if closer.any():
            action = batch['teammate_actions'][row, step, batch['teammate_mask'][row, step]].item()
            took = int(1 <= action <= 4 and closer[action-1])
            opportunities[index] += 1
            approaches[index] += took
            if step == 0:
                opening[index] += 1
                opening_approaches[index] += took
    valid = batch['filled'].reshape(len(data['map_seeds']), len(members), -1)
    result = {}
    for family in ['all', *sorted(set(labels.values()))]:
        sel = [i for i, member in enumerate(members) if family == 'all' or labels[member] == family]
        result[family] = dict(episodes=len(data['map_seeds']) * len(sel), actions=int(valid[:, sel].sum()),
                              opening_opportunities=int(opening[sel].sum()),
                              opening_approaches=int(opening_approaches[sel].sum()),
                              opportunities=int(opportunities[sel].sum()), approaches=int(approaches[sel].sum()))
    return result


def scores(assignments, labels):
    similar = {k: v for k, v in labels.items() if v in ('BASELINE', 'SOCIAL1')}
    hard = partition_score({k: assignments[k] for k in similar}, similar)
    return dict(all=partition_score(assignments, labels), baseline_social1=hard)


def compare(reference, diagnostic, seeds):
    dirs = dict(shared_random=reference, shared_preload=diagnostic)
    train, query, metadata, hashes = {}, {}, {}, {}
    seen = set()
    for mode, directory in dirs.items():
        metadata[mode] = json.loads((directory/'truth.json').read_text())
        hashes[mode] = {name: digest(directory/name) for name in ('train.pt', 'query.pt', 'observations.pt')}
        if any(metadata[mode]['files_sha256'][k] != h for k, h in hashes[mode].items()):
            raise ValueError('Dataset fingerprint differs from recorded generation')
        train[mode], query[mode] = load(directory/'train.pt'), load(directory/'query.pt')
        for name, data in [('train', train[mode]), ('query', query[mode])]:
            maps = data['map_seeds']
            split = metadata[mode]['splits'][name]
            if maps != split['map_seeds'] or split.get('selection', {}).get('mode', 'shared_random') != mode:
                raise ValueError(f'The {name} trajectory file must match the {mode} collection protocol')
            initial = data['batch']['state'][:, 0].reshape(len(maps), len(data['members']), -1)
            if not torch.equal(initial, initial[:, :1].expand_as(initial)):
                raise ValueError('Every teammate must start on exactly the same boards')
            actual = [hashlib.sha256(s.numpy().tobytes()).hexdigest() for s in initial[:, 0]]
            if (actual != split['initial_state_sha256'] or
                    len(set(actual)) != len(actual) or not seen.isdisjoint(actual)):
                raise ValueError('Unmatched, duplicate, or overlapping map layouts across train/query datasets')
            seen.update(actual)
            if data['members'] != train['shared_random']['members']:
                raise ValueError('Anonymous rosters must match')
    fixed = ('labels', 'pool_sha256', 'fixed_partner_sha256', 'policy_rng', 'assignment_seed')
    if any(metadata['shared_random'][k] != metadata['shared_preload'][k] for k in fixed):
        raise ValueError('Rule, replica or partner setup differs between datasets')
    runs = []
    for seed in seeds:
        protocols = []
        for mode, directory in dirs.items():
            ckpt, fit = check_fit(directory, seed, 'categorical', train[mode])
            if ckpt['crp'].get('masked_action_weight', 0.) != 0 or fit['train_sha256'] != hashes[mode]['train.pt']:
                raise ValueError('Expected an unmasked fit on the matching train split')
            protocols.append(fit['protocol'])
            path = directory/'categorical'/f'seed{seed}'/'crp.pt'
            item = dict(seed=seed, train_mode=mode, checkpoint_sha256=digest(path),
                        online=fit['assignments'], heldout={})
            for query_mode, held in query.items():
                frozen = frozen_partitions(path, held, fit['protocol']['probe_episodes'])
                item['heldout'][query_mode] = frozen['blocks']
            runs.append(item)
        if protocols[0] != protocols[1]:
            raise ValueError('The two checkpoints must use the same categorical training settings')

    labels = metadata['shared_random']['labels']
    for item in runs:
        item['online_scores'] = scores(item['online'], labels)
        for blocks in item['heldout'].values():
            for block in blocks:
                block['scores'] = scores(block['assignments'], labels)
    opportunities = {}
    summaries = {}
    for mode, directory in dirs.items():
        local = load(directory/'observations.pt')
        opportunities[mode] = {split: coverage(data, local[split], labels)
                               for split, data in [('train', train[mode]), ('query', query[mode])]}
        summaries[mode] = {}
        for query_mode in dirs:
            blocks = [block['scores'] for run in runs if run['train_mode'] == mode
                      for block in run['heldout'][query_mode]]
            summaries[mode][query_mode] = dict(
                blocks=len(blocks), mean_ari=float(np.mean([b['all']['ari'] for b in blocks])),
                exact_recoveries=sum(b['all']['exact_recovery'] for b in blocks),
                mean_baseline_social1_ari=float(np.mean([b['baseline_social1']['ari'] for b in blocks])))
    return dict(seeds=seeds, labels=labels, data_sha256=hashes, coverage=opportunities,
                summary_by_train_then_query=summaries, results=runs,
                method='Original unmasked categorical CRP. Frozen weights; reset clusters and assign the same '
                       'nine blind members in the same order for each block of eight shared held-out maps. '
                       'Evaluate both fits on both map sets. ARI of one is perfect rule recovery.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('diagnostic', type=Path)
    parser.add_argument('--seeds', type=int, nargs='+', default=[7, 17, 27])
    args = parser.parse_args()
    dirs = (args.reference.resolve(), args.diagnostic.resolve())
    if (os.getenv('FASTAP_ROOT') != str(ROOT) or dirs[0] == dirs[1] or
            any(ROOT/'outputs' not in d.parents for d in dirs) or len(set(args.seeds)) != len(args.seeds)):
        parser.error('Use two distinct project output directories, unique model seeds and project-local caches')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.set_num_threads(1)
    target = dirs[1]/'start_comparison.json'
    publish(target, compare(*dirs, args.seeds))
    print(target)
