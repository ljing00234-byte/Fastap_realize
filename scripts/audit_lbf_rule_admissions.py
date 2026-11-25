"""Replay categorical admissions on blinded LBF teammates and attribute errors after scoring."""

import argparse
from copy import deepcopy
from itertools import combinations
import json
from math import log
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastap_methods.fastap_crp import TeammateCRP
from scripts.audit_lbf_crp_order import compare_partitions
from scripts.evaluate_lbf_crp_prediction import FIELDS, digest, load
from scripts.evaluate_lbf_rule_prediction import check_fit, check_splits
from scripts.validate_lbf_rule_clusters import member_batch

SELECTED = ROOT / 'outputs/lbf_rule_informative_starts_v1'
ORDINARY = ROOT / 'outputs/lbf_rule_blind_v1'
MODES = ('historical', 'current_encoder')
HARD = ('BASELINE', 'SOCIAL1')


def make_crp(args, protocol, mode):
    return TeammateCRP(**args, alpha=protocol['alpha'], lr=protocol['lr'], decoder='categorical',
                       minibatch_size=protocol['minibatch_size'], center_mode=mode)


def training_snapshots(checkpoint, fit, train):
    """Only use the archived anonymous training actions to reproduce the weights at each admission."""
    cfg = fit['protocol']
    torch.manual_seed(cfg['seed'])
    model = make_crp(checkpoint['model_args'], cfg, 'historical')
    snapshots = []
    for expected in fit['admissions']:
        name = expected['teammate']
        batch = member_batch(train, name, 0, cfg['probe_episodes'])
        label = model.add_group(name, batch, model_steps=cfg['crp_model_steps'])
        reference = [row['log_joint'] + log(len(snapshots) + cfg['alpha']) for row in expected['candidates']]
        if (f'C{label+1}' != expected['cluster'] or not np.allclose(
                model.last_assignment_log_scores.tolist(), reference, rtol=0., atol=1.e-7)):
            raise RuntimeError(f'Historical assignments or scores differ at {name}')
        snapshots.append(deepcopy(model.model.state_dict()))
    if any(not torch.equal(v, checkpoint['crp']['model'][k]) for k, v in snapshots[-1].items()):
        raise RuntimeError('Replayed final parameters differ from the archived model')
    return snapshots


def pair_errors(assignments, labels):
    pairs = list(combinations(labels, 2))
    truth = {p for p in pairs if labels[p[0]] == labels[p[1]]}
    joined = {p for p in pairs if assignments[p[0]] == assignments[p[1]]}
    return dict(ari=compare_partitions(assignments, labels, pairs)['ari'],
                split=len(truth - joined), same=len(truth),
                merged=len(joined - truth), different=len(pairs) - len(truth),
                exact=truth == joined)


def decision_errors(admissions, labels):
    """After blind scoring, compare a mistaken winner to the best-scoring pair-error minimizer."""
    from fastap_methods.lbf_admission_trace import ACTION_NAMES, contrast
    cases = []
    for event in admissions:
        name, previous, trace = event['member'], event['previous'], event['trace']
        candidates = trace['candidates']
        costs = {c['cluster']: sum((old == c['cluster']) != (labels[name] == labels[n])
                                   for n, old in previous.items()) for c in candidates}
        winner = trace['winner']
        if costs[winner] == min(costs.values()):
            continue
        best = max((c for c in candidates if costs[c['cluster']] == min(costs.values())),
                   key=lambda c: c['log_score'])
        alt = best['cluster']
        gap = contrast(trace, winner, alt)
        left = next(j for j, c in enumerate(candidates) if c['cluster'] == winner)
        right = candidates.index(best)
        supporting = []
        for i in gap['strongest_for']:
            row = trace['rows'][i]
            supporting.append(dict(map_seed=row['map_seed'], step_zero_based=row['step'],
                                   action=ACTION_NAMES[row['action']], opportunity=row['opportunity'],
                                   evidence_margin=row['logp'][left] - row['logp'][right]))
        same = [n for n, c in previous.items() if labels[n] == labels[name] and c != winner]
        different = [n for n, c in previous.items() if labels[n] != labels[name] and c == winner]
        cases.append(dict(member=name, rule=labels[name], winner=winner, alternative=alt,
                          winner_is_new=winner == candidates[-1]['cluster'], split_from=same,
                          merged_with=different, incremental_pair_errors=costs, prior_margin=gap['prior_margin'],
                          evidence_margin=gap['evidence_margin'], total_margin=gap['total_margin'],
                          by_action=gap['by_action'], by_opportunity=gap['by_opportunity'], strongest=supporting))
    return cases


def summarize(rows, labels):
    """Pair counts accumulate within each partition; compare seeds on identical evidence."""
    results = {}
    families = dict(all=labels, baseline_social1={k: v for k, v in labels.items() if v in HARD})
    for family, truth in families.items():
        scores = [pair_errors(row['assignments'], truth) for row in rows]
        summary = {k: sum(s[k] for s in scores) for k in ('split', 'same', 'merged', 'different', 'exact')}
        summary['mean_ari'] = float(np.mean([s['ari'] for s in scores]))
        summary['partitions'] = len(scores)
        pairs = list(combinations(truth, 2))
        stability = [compare_partitions(a['assignments'], b['assignments'], pairs)['ari']
                     for a, b in combinations(rows, 2) if a['seed'] != b['seed'] and a['block'] == b['block']]
        summary['between_seed_ari'] = float(np.mean(stability)) if stability else None
        results[family] = summary
    return results


@torch.inference_mode()
def replay(checkpoint, cfg, snapshots, data, start, mode, trace=False):
    """Blindly assign held-out groups using successive training-only snapshots, or the fixed last one."""
    crp = make_crp(checkpoint['model_args'], cfg, mode)
    history = {}
    admissions = []
    size = cfg['probe_episodes']
    maps = data['map_seeds'][start:start+size]
    for name, state in zip(cfg['order'], snapshots):
        crp.model.load_state_dict(state)
        group = crp._unpack(member_batch(data, name, start, size))
        counts = crp.counts.copy()
        label = crp.admit_probes(name, group)
        if trace:
            from fastap_methods.lbf_admission_trace import trace_admission
            details = trace_admission(crp.model, group, counts, crp.last_centers_before, crp.alpha,
                                      minibatch_size=crp.minibatch_size, map_seeds=maps,
                                      expected_scores=crp.last_assignment_log_scores,
                                      candidate_centers=crp.last_candidate_centers)
            admissions.append(dict(member=name, previous=dict(history), trace=details))
        history[name] = f'C{label+1}'
    return dict(seed=cfg['seed'], block=start//size, map_seeds=maps, assignments=history), admissions


def classify_vectors(support, query, rules, bag_size, allowed):
    """Compare mean distances to other identities of each rule on independent support maps."""
    refs = support.mean(0)
    bags = query.reshape(-1, bag_size, len(rules), query.shape[-1]).mean(1)
    eligible = [i for i, rule in enumerate(rules) if rule in allowed]
    families = sorted(set(allowed))
    counts = {a: {b: 0 for b in families} for a in families}
    ties = 0
    for bag in bags:
        distances = torch.cdist(bag.double(), refs.double())
        for i in eligible:
            loss = [distances[i, [j for j in eligible if rules[j] == family and j != i]].mean().item()
                    for family in families]
            if not np.isfinite(loss).all():
                raise ValueError('Each rule needs a different support identity')
            if loss.count(min(loss)) != 1:
                ties += 1
                continue
            counts[rules[i]][families[int(np.argmin(loss))]] += 1
    total = len(bags) * len(eligible)
    return dict(correct=sum(counts[r][r] for r in families), total=total, ties=ties, confusion=counts)


@torch.inference_mode()
def encoder_probe(checkpoint, cfg, splits, labels):
    model = make_crp(checkpoint['model_args'], cfg, 'historical').model.eval()
    model.load_state_dict(checkpoint['crp']['model'])
    encoded = []
    for split in ('support', 'query'):
        data = splits[split]
        batch = data['batch']
        if len(data['map_seeds']) % cfg['probe_episodes']:
            raise ValueError('Use complete query/support bags')
        vectors = [model.encode(*(batch[k][i:i+32] for k in FIELDS))
                   for i in range(0, len(batch['filled']), 32)]
        encoded.append(torch.cat(vectors).reshape(len(data['map_seeds']), len(labels), -1))
    rules = [labels[name] for name in cfg['order']]
    return {family: classify_vectors(*encoded, rules, cfg['probe_episodes'], included)
            for family, included in [('all', sorted(set(rules))), ('baseline_social1', HARD)]}


def validate_data(selected, ordinary):
    directories = dict(selected=selected, ordinary=ordinary)
    splits, metadata, hashes = {}, {}, {}
    signatures = set()
    for domain, directory in directories.items():
        metadata[domain] = json.loads((directory/'truth.json').read_text())
        truth = metadata[domain]
        hashes[domain] = {key: digest(directory/f'{key}.pt') for key in ('train', 'support', 'query')}
        if any(value != truth['files_sha256'][f'{key}.pt'] for key, value in hashes[domain].items()):
            raise ValueError('Rule dataset differs from its archived manifest')
        splits[domain] = {key: load(directory/f'{key}.pt') for key in hashes[domain]}
        check_splits(splits[domain], truth['labels'])
        for data in splits[domain].values():
            initial = data['batch']['state'][::len(data['members']), 0]
            states = {s.contiguous().numpy().tobytes() for s in initial}
            if signatures & states:
                raise ValueError('Opening reused across evaluation splits or domains')
            signatures.update(states)
    if any(metadata['selected'][k] != metadata['ordinary'][k]
           for k in ('labels', 'policy_rng', 'fixed_partner_sha256', 'assignment_seed')):
        raise ValueError('Different rule replicas or partners across datasets')
    return splits, metadata['selected']['labels'], hashes


def run(staging, selected, ordinary, seeds):
    splits, labels, hashes = validate_data(selected, ordinary)
    source_report = selected/'start_comparison.json'
    archived = json.loads(source_report.read_text())
    if (archived['data_sha256']['shared_preload']['train.pt'] != hashes['selected']['train']
            or archived['data_sha256']['shared_random']['query.pt'] != hashes['ordinary']['query']
            or archived['data_sha256']['shared_preload']['query.pt'] != hashes['selected']['query']):
        raise ValueError('Reference evaluation used different trajectories')
    runs = []
    embeddings = []
    integrity = []
    examples = []
    for seed in seeds:
        train = splits['selected']['train']
        checkpoint, fit = check_fit(selected, seed, 'categorical', train)
        cfg = fit['protocol']
        size = cfg['probe_episodes']
        snapshots = training_snapshots(checkpoint, fit, train)
        prior = next(row for row in archived['results'] if row['seed'] == seed and row['train_mode'] == 'shared_preload')
        if prior['checkpoint_sha256'] != digest(selected/'categorical'/f'seed{seed}'/'crp.pt'):
            raise ValueError('Archived evaluation used different fitted weights')
        checked = 0
        for domain, data in [('train', train), ('selected', splits['selected']['query']),
                             ('ordinary', splits['ordinary']['query'])]:
            if data['members'] != cfg['order']:
                raise ValueError('Arrival order differs across evaluations')
            for start in range(0, len(data['map_seeds']), size):
                for mode in MODES:
                    trace = seed == seeds[0] and domain == 'selected' and start == 0
                    row, admissions = replay(checkpoint, cfg, snapshots, data, start, mode, trace=trace)
                    runs.append(dict(domain=domain, center_mode=mode, **row))
                    if trace:
                        examples.append(dict(seed=seed, mode=mode, admissions=admissions))
                    if domain == 'train':
                        if mode == 'historical' and row['assignments'] != fit['assignments']:
                            raise RuntimeError('Historical path does not reproduce the archived online partition')
                    else:
                        # The previous benchmark froze a single final encoder for every admission.
                        # Without drift both center protocols must yield its archived partition.
                        final, _ = replay(checkpoint, cfg, [snapshots[-1]] * len(snapshots), data, start, mode)
                        query_key = 'shared_preload' if domain == 'selected' else 'shared_random'
                        expected = prior['heldout'][query_key][start//size]
                        if any(final[k] != expected[k] for k in ('assignments', 'map_seeds')):
                            raise RuntimeError('Frozen-encoder control differs from the archived partition')
                        checked += 1
        embeddings.append(dict(seed=seed, **{domain: encoder_probe(checkpoint, cfg, part, labels)
                                            for domain, part in splits.items()}))
        integrity.append(dict(seed=seed, online_scores_replayed=len(snapshots),
                              final_weight_tensors_verified=len(snapshots[-1]), frozen_partitions_replayed=checked))
    summary = {domain: {mode: summarize([row for row in runs if row['domain'] == domain and
                                         row['center_mode'] == mode], labels) for mode in MODES}
               for domain in ('train', 'selected', 'ordinary')}
    traces_file = f'seed_{seeds[0]}_selected_block_0.json'
    (staging/traces_file).write_text(json.dumps(dict(labels=labels, results=examples), allow_nan=False)+'\n')
    report = dict(seeds=seeds, labels=labels, dataset_sha256=hashes, reference_sha256=digest(source_report),
                  design='Replay fitted training-only model snapshots in admission order. On each disjoint '
                         'eight-map test block start a new CRP, replacing only its weights by the next snapshot '
                         'before each arrival. No optimizer sees a held-out trajectory. Identical weights, '
                         'CRP priors, score formula, admitted member order and evidence in both arms.',
                  encoder_test='Post-fit supervised readout. Query = mean of eight whole A->A embeddings. '
                               'For each rule average Euclidean distance to its other support identities '
                               '(32 separate maps per identity, query ID always excluded). Choose nearest rule. '
                               'Ties count as errors. Supports, maps and IDs are held out from the query score; '
                               'these policy IDs also supplied unlabeled trajectories during original training.',
                  checkpoint_checks=integrity, encoder_readout=embeddings, summary=summary, partitions=runs,
                  action_traces=traces_file,
                  first_selected_block_errors={row['mode']: decision_errors(row['admissions'], labels)
                                               for row in examples})
    (staging/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--selected', type=Path, default=SELECTED)
    parser.add_argument('--ordinary', type=Path, default=ORDINARY)
    parser.add_argument('--seeds', type=int, nargs='+', default=[7, 17, 27])
    args = parser.parse_args()
    target, selected, ordinary = (p.resolve() for p in (args.directory, args.selected, args.ordinary))
    if (os.getenv('FASTAP_ROOT') != str(ROOT) or
            any(ROOT/'outputs' not in p.parents for p in (target, selected, ordinary)) or
            target.exists() or len(set(args.seeds)) != len(args.seeds)):
        parser.error('Source the project environment and use a new project output directory and unique seeds')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.set_num_threads(1)
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=target.parent, prefix='.admission-audit-') as staging:
        run(Path(staging), selected, ordinary, args.seeds)
        Path(staging).rename(target)
    print(target/'report.json')


if __name__ == '__main__':
    main()
