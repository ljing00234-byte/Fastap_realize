"""Evaluate whether a blind CRP fit transfers action predictions across maps and rule replicas."""

import argparse
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastap_methods.fastap_crp import BehaviorModel
from scripts.evaluate_lbf_crp_prediction import FIELDS, digest, fit_state_only, load, summarize


@torch.inference_mode()
def predict_by_rule(model, state_only, support, query, labels):
    """Compare each query with every support identity on its paired, different opening."""
    same = torch.tensor([[a == b for b in labels] for a in labels])
    n = len(labels)
    identity = torch.eye(n, dtype=torch.bool)
    controls = {'wrong': ~identity, 'same_rule_other_id': same & ~identity, 'wrong_rule': ~same}
    if not all(mask.any(1).all() for mask in controls.values()):
        raise ValueError('Require two or more rule families and at least two identities per family')
    n_rows = len(query['state'])
    if n_rows != len(support['state']) or n_rows == 0 or n_rows % n:
        raise ValueError('Require complete matched support and query map blocks')
    model.eval()
    state_only.eval()
    result = {k: [] for k in ('correct', 'state_only', *controls)}
    for start in range(0, n_rows, n):
        rows = slice(start, start+n)
        context = model.encode(*(support[key][rows] for key in FIELDS))
        targets = [query[key][rows] for key in FIELDS]
        # Row i is query member i; column j uses only member j's A trajectory.
        pair_scores = torch.stack([model.episode_action_nll(*targets, vector.expand(n, -1))
                                   for vector in context], dim=1).double()
        result['correct'].append(pair_scores.diag())
        result['state_only'].append(state_only.episode_action_nll(*targets, torch.zeros_like(context)))
        for name, mask in controls.items():
            result[name].append(pair_scores.masked_fill(~mask, 0).sum(1) / mask.sum(1))
    return {k: torch.stack(values).double().numpy() for k, values in result.items()}


def check_splits(splits, labels):
    members = splits['train']['members']
    if not members or len(set(members)) != len(members) or set(labels) != set(members):
        raise ValueError('Anonymous IDs must match scoring labels one-to-one')
    seen_states, seen_seeds = set(), set()
    for split in ('train', 'support', 'query'):
        part = splits[split]
        seeds = part['map_seeds']
        batch = part['batch']
        rows = len(seeds) * len(members)
        if (part['members'] != members or not seeds or len(set(seeds)) != len(seeds)
                or seen_seeds.intersection(seeds) or any(batch[k].size(0) != rows for k in FIELDS)):
            raise ValueError(f'Unexpected IDs, seeds or rows in {split}')
        seen_seeds.update(seeds)
        for initial in batch['state'][:, 0].split(len(members)):
            key = initial[0].contiguous().numpy().tobytes()
            if not torch.equal(initial, initial[0].expand_as(initial)) or key in seen_states:
                raise ValueError(f'Unpaired or repeated map in {split}')
            seen_states.add(key)
    if len(splits['support']['map_seeds']) != len(splits['query']['map_seeds']):
        raise ValueError('Use one independent support map per query map')


def check_fit(directory, seed, decoder, train):
    path = directory / decoder / f'seed{seed}'
    checkpoint = load(path / 'crp.pt')
    cfg = json.loads((path / 'clustering.json').read_text())
    protocol = cfg['protocol']
    if (protocol['decoder'], protocol['seed']) != (decoder, seed):
        raise ValueError(f'Wrong decoder or seed in {path}')
    records = cfg['admissions']
    order = [r['teammate'] for r in records]
    if order != protocol['order'] or sorted(order) != sorted(train['members']):
        raise ValueError('Checkpoint arrival order differs from the blind dataset')
    probes = checkpoint['crp']['probes']
    maps = len(train['map_seeds'])
    if len(probes) != maps * len(order):
        raise ValueError('Checkpoint has a different number of training trajectories')
    for j, record in enumerate(records):
        member = train['members'].index(order[j])
        if list(record['probe_slice']) != [j*maps, (j+1)*maps]:
            raise ValueError('Unexpected per-member probe ordering')
        for i in range(maps):
            row = i * len(order) + member
            batch = train['batch']
            length = int(batch['filled'][row].sum())
            for field, actual in zip(FIELDS, probes[j*maps+i]):
                expected = batch[field][row, :length+int(field == 'state')]
                if not torch.equal(expected, actual):
                    raise ValueError('The fitted checkpoint does not match the blind training trajectories')
    return checkpoint, cfg


def evaluate(directory, seeds):
    splits = {k: load(directory / f'{k}.pt') for k in ('train', 'support', 'query')}
    labels = json.loads((directory / 'truth.json').read_text())['labels']
    check_splits(splits, labels)
    members = splits['train']['members']
    map_seeds = splits['query']['map_seeds']
    n_maps = len(map_seeds)
    bootstrap_seed = 87543
    samples = np.random.default_rng(bootstrap_seed).integers(n_maps, size=(10000, n_maps))
    results = []
    for seed in seeds:
        categorical, reference = check_fit(directory, seed, 'categorical', splits['train'])
        state_only = fit_state_only(categorical['model_args'], reference['protocol'], reference['admissions'],
                                    categorical['crp']['probes'], seed)
        for decoder in ('categorical', 'paper'):
            if decoder == 'categorical':
                ckpt, cfg = categorical, reference
            else:
                ckpt, cfg = check_fit(directory, seed, decoder, splits['train'])
                keys = ('order', 'crp_model_steps', 'minibatch_size', 'lr')
                if any(reference['protocol'][k] != cfg['protocol'][k] for k in keys):
                    raise ValueError('Train protocol mismatch between categorical and paper fits')
            model = BehaviorModel(**ckpt['model_args'], decoder=decoder)
            model.load_state_dict(ckpt['crp']['model'])
            scores = predict_by_rule(model, state_only, splits['support']['batch'], splits['query']['batch'],
                                     [labels[n] for n in members])
            row = summarize(scores, samples, map_seeds, members)
            gains = row['gain']
            for source in ('correct', 'same_rule_other_id'):
                difference = (scores['wrong_rule']-scores[source]).mean(1)
                bounds = np.quantile(difference[samples].mean(1), [.025, .975])
                gains[f'wrong_rule_minus_{source}'] = dict(mean_nats=float(difference.mean()),
                                                           paired_95_ci=bounds.tolist())
            path = directory / decoder / f'seed{seed}' / 'crp.pt'
            results.append(dict(decoder=decoder, seed=seed, checkpoint_sha256=digest(path), **row))
    return dict(criterion='For every categorical seed the lower paired 95% bounds for other-ID minus correct '
                          'and state-only minus correct NLL must both exceed zero.',
                primary_criterion_passed=all(r['both_lower_bounds_positive'] for r in results
                                             if r['decoder'] == 'categorical'),
                units='Nats per observed action; average within each query episode, then across episodes.',
                conditions={'correct': 'Same identity on A.', 'wrong': 'Every other identity on A, equally.',
                            'same_rule_other_id': 'Every other identity of the same rule on A, equally.',
                            'wrong_rule': 'Every identity from a different rule on A, equally.',
                            'state_only': 'Independently trained categorical state-history GRU with zero context, '
                                          'same initial decoder and minibatches as the categorical fit.'},
                rule_labels_used_for='Post-fit evaluation masks only. No query action is encoded.',
                confidence='Pointwise paired bootstrap of whole A/B map pairs with all identities held together.',
                bootstrap_seed=bootstrap_seed, bootstrap_resamples=len(samples),
                split_hashes={p: digest(directory / p) for p in ('train.pt', 'support.pt', 'query.pt', 'truth.json')},
                seeds=seeds, members=members, support_seeds=splits['support']['map_seeds'], results=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--seeds', type=int, nargs='+', default=[7, 17, 27])
    args = parser.parse_args()
    directory = args.directory.resolve()
    if (os.getenv('FASTAP_ROOT') != str(ROOT) or ROOT / 'outputs' not in directory.parents
            or len(set(args.seeds)) != len(args.seeds)):
        parser.error('Source scripts/epymarl-env.sh and use distinct seeds and a project output directory')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.set_num_threads(1)
    target = directory / 'rule_prediction.json'
    if target.exists():
        raise FileExistsError(target)
    report = evaluate(directory, args.seeds)
    with TemporaryDirectory(dir=directory) as staging:
        temp = Path(staging) / target.name
        temp.write_text(json.dumps(report, allow_nan=False, indent=2)+'\n')
        os.link(temp, target)
    print(f'Cross-map prediction passed: {report["primary_criterion_passed"]}', flush=True)


if __name__ == '__main__':
    main()
