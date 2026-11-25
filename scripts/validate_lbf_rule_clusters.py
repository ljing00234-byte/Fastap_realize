"""Blindly fit the production CRP on rule teammates, then reveal labels to score partitions."""

import argparse
from collections import Counter
from itertools import combinations
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml

from fastap_methods.fastap_crp import TeammateCRP
from fastap_methods.lbf_crp_analysis import _candidate_scores
from scripts.audit_lbf_crp_order import compare_partitions
from scripts.evaluate_lbf_cluster_ego import equal_size_partitions
from scripts.evaluate_lbf_crp_prediction import FIELDS, digest, load


def member_batch(data, member, first, count):
    """Read only states, observed teammate actions, and their masks, in map-major layout."""
    names = data['members']
    maps = len(data['map_seeds'])
    idx = names.index(member)
    if len(names) != len(set(names)) or count < 1 or first < 0 or first + count > maps:
        raise ValueError("Invalid blind roster or map block")
    rows = slice(first * len(names) + idx, (first + count) * len(names), len(names))
    batch = data['batch']
    if any(len(batch[field]) != maps * len(names) for field in FIELDS):
        raise ValueError("Expected a complete map by member grid")
    return {field: batch[field][rows] for field in FIELDS}


def fit(directory, decoder, seed):
    # This phase only opens the anonymous training split; the rule key is read in score().
    dataset = directory / 'train.pt'
    data = load(dataset)
    config = yaml.safe_load((ROOT / 'configs/methods/fastap.yaml').read_text())
    cfg = {**config, **config['scenes']['lbf']}
    order = data['members']
    if len(data['map_seeds']) != cfg['probe_episodes']:
        raise ValueError("The blind dataset must match the configured admission probe budget")
    model_args = dict(state_dim=data['batch']['state'].shape[-1], n_actions=6,
                      n_teammates=data['batch']['teammate_actions'].shape[-1])
    torch.manual_seed(seed)
    crp = TeammateCRP(**model_args, alpha=cfg['alpha'], lr=cfg['lr'], decoder=decoder)
    admissions = []
    for name in order:
        counts = list(crp.counts)
        start = len(crp._probes)
        batch = member_batch(data, name, 0, cfg['probe_episodes'])
        label = crp.add_group(name, batch, model_steps=cfg['crp_model_steps'])
        admissions.append(dict(teammate=name, cluster=f'C{label+1}', probe_slice=[start, len(crp._probes)],
                               candidates=_candidate_scores(counts, crp.alpha, crp.last_assignment_log_scores.tolist())))
    protocol = dict(decoder=decoder, seed=seed, order=order, alpha=crp.alpha, minibatch_size=crp.minibatch_size,
                    lr=cfg['lr'], crp_model_steps=cfg['crp_model_steps'], probe_episodes=cfg['probe_episodes'])
    summary = dict(train_sha256=digest(dataset), protocol=protocol, admissions=admissions,
                   assignments={n: f'C{k+1}' for n, k in crp.assignments.items()})
    output = directory / decoder / f'seed{seed}'
    output.parent.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as stage:
        temp = Path(stage)
        (temp / 'clustering.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
        torch.save(dict(model_args=model_args, crp=crp.state_dict()), temp / 'crp.pt')
        temp.rename(output)


@torch.inference_mode()
def frozen_partitions(checkpoint_path, query, bag_size):
    """Freeze the actual behavior model and start an empty CRP for each block of new maps."""
    checkpoint = load(checkpoint_path)
    params = checkpoint['crp']
    crp = TeammateCRP(**checkpoint['model_args'], alpha=params['alpha'], decoder=params.get('decoder', 'paper'),
                      minibatch_size=params['minibatch_size'])
    crp.model.load_state_dict(params['model'])
    crp.model.eval()
    maps = query['map_seeds']
    if not maps or len(maps) % bag_size:
        raise ValueError("Each disjoint evaluation block must contain a complete admission budget")
    blocks = []
    for start in range(0, len(maps), bag_size):
        crp.counts, crp.centers = [], []
        assignments = {}
        for name in query['members']:
            probes = crp._unpack(member_batch(query, name, start, bag_size))
            label, center = crp._assign(probes)
            if label == len(crp.counts):
                crp.counts.append(0)
                crp.centers.append(center)
            crp.counts[label] += 1
            crp.centers[label] = center
            assignments[name] = f'C{label+1}'
        blocks.append(dict(map_seeds=maps[start:start+bag_size], assignments=assignments))
    return dict(checkpoint_sha256=digest(checkpoint_path), blocks=blocks)


def transfer(directory, legacy, seeds):
    query = load(directory/'query.pt')
    train = load(directory/'train.pt')
    if train['members'] != query['members'] or set(train['map_seeds']) & set(query['map_seeds']):
        raise ValueError("Use the same blind member order and disjoint train/query maps")
    n_maps = len(train['map_seeds'])
    runs = []
    for seed in seeds:
        for decoder in ('categorical', 'paper'):
            run = directory/decoder/f'seed{seed}'
            online = json.loads((run/'clustering.json').read_text())
            if online['protocol']['order'] != query['members'] or online['train_sha256'] != digest(directory/'train.pt'):
                raise ValueError("Blind checkpoint belongs to a different training set or order")
            frozen = frozen_partitions(run/'crp.pt', query, n_maps)
            runs.append(dict(source='rule_probes', decoder=decoder, seed=seed,
                             online=online['assignments'], **frozen))
        # Also test the categorical weights that were already trained on QMIX T1–T8,
        # without allowing them to learn from any rule-teammate trajectories.
        frozen = frozen_partitions(legacy/'categorical'/f'seed{seed}'/'crp.pt', query, n_maps)
        runs.append(dict(source='frozen_T1_T8', decoder='categorical', seed=seed, **frozen))
    return dict(train_sha256=digest(directory/'train.pt'), query_sha256=digest(directory/'query.pt'),
                order=query['members'], fit_maps=train['map_seeds'], results=runs)


def partition_score(assignments, truth):
    names = tuple(sorted(truth))
    if set(names) != set(assignments):
        raise ValueError("Blind predictions must cover precisely the identities in the scoring key")
    pairs = list(combinations(names, 2))
    agreement = compare_partitions(assignments, truth, pairs)
    same_true = sum(truth[a] == truth[b] for a, b in pairs)
    same_pred = sum(assignments[a] == assignments[b] for a, b in pairs)
    # Under permutation of the hidden labels all distinct partitions of the
    # observed true sizes are equiprobable. Conditioning on the predicted
    # grouping, the ARI is monotone in the number of correct same-class pairs.
    null_total = at_least = 0
    for blocks in equal_size_partitions(names, list(Counter(truth.values()).values())):
        overlap = sum(assignments[a] == assignments[b] for block in blocks for a, b in combinations(block, 2))
        at_least += overlap >= agreement['same_in_both']
        null_total += 1
    return dict(clusters=len(set(assignments.values())), ari=agreement['ari'],
                exact_recovery=not agreement['changed_pairs'], true_same_pairs=same_true,
                predicted_same_pairs=same_pred, correct_same_pairs=agreement['same_in_both'],
                exact_permutation_p=at_least/null_total, null_partition_count=null_total)


def score(directory):
    blind = json.loads((directory/'blind_partitions.json').read_text())
    truth_path = directory/'truth.json'
    truth = json.loads(truth_path.read_text())['labels']
    report = dict(blind_sha256=digest(directory/'blind_partitions.json'), truth_sha256=digest(truth_path),
                  labels=truth, criterion='Exact rule recovery compares unordered partitions. ARI=1 is exact, '
                  'and its expected value is zero under independent random labels of fixed sizes. '
                  'Query evaluation freezes weights and resets seats for each disjoint block of eight maps.',
                  results=[])
    for run in blind['results']:
        item = {key: run[key] for key in ('source','decoder','seed')}
        if 'online' in run:
            item['online'] = partition_score(run['online'], truth)
        item['heldout'] = [partition_score(b['assignments'], truth) for b in run['blocks']]
        report['results'].append(item)
    return report


def publish(path, data):
    with TemporaryDirectory(dir=path.parent) as staging:
        temp = Path(staging)/path.name
        temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
        os.link(temp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('fit','transfer','score'))
    parser.add_argument('directory', type=Path)
    parser.add_argument('--decoder', choices=('paper','categorical'), default='categorical')
    parser.add_argument('--seeds', type=int, nargs='+', default=[7,17,27])
    parser.add_argument('--legacy', type=Path, default=ROOT/'outputs/fastap_lbf_formal_pool_01/crp_categorical_v1')
    args = parser.parse_args()
    base = args.directory.resolve()
    if os.getenv('FASTAP_ROOT') != str(ROOT) or ROOT/'outputs' not in base.parents:
        parser.error('Source scripts/epymarl-env.sh and use a project output directory')
    if min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Use distinct, nonnegative model seeds')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.set_num_threads(1)
    if args.phase == 'fit':
        for seed in args.seeds:
            fit(base, args.decoder, seed)
    elif args.phase == 'transfer':
        publish(base/'blind_partitions.json', transfer(base, args.legacy.resolve(), args.seeds))
    else:
        publish(base/'partition_scores.json', score(base))


if __name__ == '__main__':
    main()
