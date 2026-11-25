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


def fit(directory, decoder, seed, *, data_dir=None, masked_action_weight=0., masked_action_span=3):
    # This phase only opens the anonymous training split; the rule key is read in score().
    output = directory / decoder / f'seed{seed}'
    if output.exists():
        raise FileExistsError(output)
    dataset = (data_dir or directory) / 'train.pt'
    data = load(dataset)
    config = yaml.safe_load((ROOT / 'configs/methods/fastap.yaml').read_text())
    cfg = {**config, **config['scenes']['lbf']}
    order = data['members']
    if len(data['map_seeds']) != cfg['probe_episodes']:
        raise ValueError("The blind dataset must match the configured admission probe budget")
    model_args = dict(state_dim=data['batch']['state'].shape[-1], n_actions=6,
                      n_teammates=data['batch']['teammate_actions'].shape[-1])
    torch.manual_seed(seed)
    crp = TeammateCRP(**model_args, alpha=cfg['alpha'], lr=cfg['lr'], decoder=decoder,
                      masked_action_weight=masked_action_weight, masked_action_span=masked_action_span,
                      mask_seed=seed)
    admissions = []
    for name in order:
        counts = list(crp.counts)
        start = len(crp._probes)
        batch = member_batch(data, name, 0, cfg['probe_episodes'])
        label = crp.add_group(name, batch, model_steps=cfg['crp_model_steps'])
        admissions.append(dict(teammate=name, cluster=f'C{label+1}', probe_slice=[start, len(crp._probes)],
                               candidates=_candidate_scores(counts, crp.alpha, crp.last_assignment_log_scores.tolist())))
        if masked_action_weight:
            admissions[-1].update(last_reconstruction_loss=crp.last_reconstruction_loss,
                                  last_masked_action_loss=crp.last_masked_action_loss)
    protocol = dict(decoder=decoder, seed=seed, order=order, alpha=crp.alpha, minibatch_size=crp.minibatch_size,
                    lr=cfg['lr'], crp_model_steps=cfg['crp_model_steps'], probe_episodes=cfg['probe_episodes'])
    if masked_action_weight:
        protocol.update(masked_action_weight=masked_action_weight, masked_action_span=masked_action_span,
                        mask_seed=seed)
    summary = dict(train_sha256=digest(dataset), protocol=protocol, admissions=admissions,
                   assignments={n: f'C{k+1}' for n, k in crp.assignments.items()})
    output.parent.mkdir(parents=True, exist_ok=True)
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


def transfer(directory, legacy, seeds, *, data_dir=None, paired_control=None):
    data_dir = data_dir or directory
    query = load(data_dir/'query.pt')
    train = load(data_dir/'train.pt')
    train_sha = digest(data_dir/'train.pt')
    if train['members'] != query['members'] or set(train['map_seeds']) & set(query['map_seeds']):
        raise ValueError("Use the same blind member order and disjoint train/query maps")
    n_maps = len(train['map_seeds'])
    runs = []
    for seed in seeds:
        if paired_control is None:
            comparisons = [(directory, decoder, 'rule_probes') for decoder in ('categorical', 'paper')]
        else:
            comparisons = [(paired_control, 'categorical', 'unmasked_control'),
                           (directory, 'categorical', 'masked_auxiliary')]
        control_protocol = None
        for parent, decoder, source in comparisons:
            run = parent/decoder/f'seed{seed}'
            online = json.loads((run/'clustering.json').read_text())
            protocol = online['protocol']
            if (protocol['order'] != query['members'] or online['train_sha256'] != train_sha or
                    protocol['seed'] != seed or protocol['decoder'] != decoder):
                raise ValueError("Blind checkpoint belongs to a different training set, order, seed or decoder")
            if source == 'unmasked_control':
                if protocol.get('masked_action_weight', 0) != 0:
                    raise ValueError("Expected an unmasked categorical control")
                control_protocol = protocol
            if source == 'masked_auxiliary':
                if (protocol.get('masked_action_weight', 0) <= 0 or
                        any(protocol[k] != control_protocol[k] for k in
                            ('minibatch_size', 'lr', 'alpha', 'crp_model_steps', 'probe_episodes'))):
                    raise ValueError("Compare the masked objective at identical CRP and training settings")
            frozen = frozen_partitions(run/'crp.pt', query, n_maps)
            runs.append(dict(source=source, decoder=decoder, seed=seed,
                             online=online['assignments'], **frozen))
        # Also test the categorical weights that were already trained on QMIX T1–T8,
        # without allowing them to learn from any rule-teammate trajectories.
        if paired_control is None:
            frozen = frozen_partitions(legacy/'categorical'/f'seed{seed}'/'crp.pt', query, n_maps)
            runs.append(dict(source='frozen_T1_T8', decoder='categorical', seed=seed, **frozen))
    return dict(train_sha256=train_sha, query_sha256=digest(data_dir/'query.pt'),
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


def score_partition_on_maps(assignments, truth, similar):
    metrics = partition_score(assignments, truth)
    if similar:
        restricted = partition_score({n: assignments[n] for n in similar}, similar)
        n = len(similar)
        metrics['similar_rule_pairs'] = dict(
            ari=restricted['ari'], families=sorted(set(similar.values())),
            same_family_joined=restricted['correct_same_pairs'], same_family_total=restricted['true_same_pairs'],
            different_family_joined=restricted['predicted_same_pairs'] - restricted['correct_same_pairs'],
            different_family_total=n*(n-1)//2 - restricted['true_same_pairs'])
    return metrics


def score(directory, data_dir=None):
    blind = json.loads((directory/'blind_partitions.json').read_text())
    truth_path = (data_dir or directory)/'truth.json'
    key = json.loads(truth_path.read_text())
    if (blind['train_sha256'] != key['files_sha256']['train.pt'] or
            blind['query_sha256'] != key['files_sha256']['query.pt']):
        raise ValueError("Scoring key does not describe the blind trajectory splits")
    truth = key['labels']
    # The two food-loading rules were the unresolved pair in the prior blind test.
    similar = {n: r for n, r in truth.items() if r in ('BASELINE', 'SOCIAL1')}
    if len(set(similar.values())) < 2:
        similar = {}
    report = dict(blind_sha256=digest(directory/'blind_partitions.json'), truth_sha256=digest(truth_path),
                  labels=truth, criterion='Exact rule recovery compares unordered partitions. ARI=1 is exact, '
                  'and its expected value is zero under independent random labels of fixed sizes. '
                  'Query evaluation freezes weights and resets seats for each disjoint block of eight maps.',
                  results=[])
    for run in blind['results']:
        item = {key: run[key] for key in ('source','decoder','seed')}
        if 'online' in run:
            item['online'] = score_partition_on_maps(run['online'], truth, similar)
        item['heldout'] = [score_partition_on_maps(b['assignments'], truth, similar) for b in run['blocks']]
        report['results'].append(item)
    controls = {r['seed']: r for r in report['results'] if r['source'] == 'unmasked_control'}
    if controls:
        paired = []
        for run in report['results']:
            if run['source'] != 'masked_auxiliary':
                continue
            prev = controls[run['seed']]
            paired.append(dict(seed=run['seed'], online_ari_delta=run['online']['ari']-prev['online']['ari'],
                               heldout_ari_deltas=[after['ari'] - before['ari']
                                                   for before, after in zip(prev['heldout'], run['heldout'])]))
        report['masked_minus_unmasked'] = paired
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
    parser.add_argument('--data-dir', type=Path, help='use existing blind trajectory splits in another directory')
    parser.add_argument('--paired-control', type=Path, help='compare categorical checkpoints from this directory '
                        'against the masked categorical fit in directory (transfer phase)')
    parser.add_argument('--masked-action-weight', type=float, default=0., help='auxiliary fit loss coefficient')
    parser.add_argument('--masked-action-span', type=int, default=3, help='number of consecutive timesteps to hide')
    parser.add_argument('--legacy', type=Path, default=ROOT/'outputs/fastap_lbf_formal_pool_01/crp_categorical_v1')
    args = parser.parse_args()
    base = args.directory.resolve()
    data_dir = args.data_dir.resolve() if args.data_dir else base
    paired_control = args.paired_control.resolve() if args.paired_control else None
    paths = [base, data_dir]
    if paired_control:
        paths.append(paired_control)
    if os.getenv('FASTAP_ROOT') != str(ROOT) or any(ROOT/'outputs' not in d.parents for d in paths):
        parser.error('Source scripts/epymarl-env.sh and use a project output directory')
    if paired_control and (args.phase != 'transfer' or paired_control == base):
        parser.error('--paired-control compares a separate checkpoint directory during transfer')
    if min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Use distinct, nonnegative model seeds')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.set_num_threads(1)
    if args.phase == 'fit':
        for seed in args.seeds:
            fit(base, args.decoder, seed, data_dir=data_dir, masked_action_weight=args.masked_action_weight,
                masked_action_span=args.masked_action_span)
    elif args.phase == 'transfer':
        publish(base/'blind_partitions.json', transfer(base, args.legacy.resolve(), args.seeds,
                                                       data_dir=data_dir, paired_control=paired_control))
    else:
        publish(base/'partition_scores.json', score(base, data_dir))


if __name__ == '__main__':
    main()
