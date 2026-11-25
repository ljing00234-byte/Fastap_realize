"""Replay per-action evidence for the formal LBF pool under two CRP center protocols."""

import argparse
from itertools import combinations
import json
from math import log
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastap_methods.fastap_crp import BehaviorModel
from fastap_methods.lbf_admission_trace import ACTION_NAMES, trace_admission
from scripts.audit_lbf_crp_order import compare_partitions
from scripts.evaluate_lbf_crp_prediction import digest, load


def replay(directory, metadata, checkpoint, destination):
    cfg = metadata['protocol']
    model = BehaviorModel(**checkpoint['model_args'], decoder='categorical').eval()
    start_plan = cfg.get('shared_start_plan')
    maps = start_plan['map_seeds'] if start_plan else None
    comparisons = []
    with destination.open('x') as stream:
        for row, snap in zip(metadata['admissions'], checkpoint['admissions']):
            model.load_state_dict(snap['model'])
            name = snap['teammate']
            start, end = snap['probe_slice']
            counts = snap['counts_before']
            if name != row['teammate'] or row['probe_slice'] != [start, end]:
                raise ValueError(f'Inconsistent admission snapshot for {name}')
            expected = [c['log_joint'] + log(sum(counts) + cfg['alpha']) for c in row['candidates']]
            trace = trace_admission(model, checkpoint['crp']['probes'][start:end], counts,
                                    snap['centers_before'], cfg['alpha'], minibatch_size=cfg['minibatch_size'],
                                    expected_scores=expected, map_seeds=maps)
            if trace['winner'] != row['cluster'] or trace['winner'] != metadata['assignments'][name]:
                raise ValueError(f'Actual admission of {name} did not reproduce')
            stream.write(json.dumps(dict(teammate=name, **trace), allow_nan=False) + '\n')
            diff = trace['new_vs_existing']
            if diff:
                comparisons.append(dict(teammate=name, chosen=trace['winner'], new=diff['lhs'],
                                        best_existing=diff['rhs'], prior_margin=diff['prior_margin'],
                                        total_margin=diff['total_margin'], by_action=diff['by_action'],
                                        by_opportunity=diff['by_opportunity']))
    return dict(clusters=metadata['clusters'], assignments=metadata['assignments'], new_vs_existing=comparisons,
                checkpoint_sha256=digest(directory/'crp.pt'), action_trace=destination.name)


def audit(historical, current, destination, seeds):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    protocols = {'historical': historical, 'current_encoder': current}
    runs = {mode: {} for mode in protocols}
    with TemporaryDirectory(dir=destination.parent) as temp:
        stage = Path(temp)
        for seed in seeds:
            for mode, base in protocols.items():
                run = base / f'seed{seed}'
                meta = json.loads((run/'clustering.json').read_text())
                cfg = dict(meta['protocol'])
                if (cfg.pop('center_mode', 'historical') != mode or cfg['decoder'] != 'categorical'
                        or cfg['seed'] != seed or cfg['order'] != [f'T{i}' for i in range(1, 9)]):
                    raise ValueError('Expected matching categorical runs on ordered T1–T8')
                checkpoint = load(run/'crp.pt')
                if len(checkpoint['admissions']) != len(meta['admissions']) or len(meta['admissions']) != 8:
                    raise ValueError('Expected eight admission snapshots')
                if mode == 'historical':
                    original, original_cfg, original_pool = checkpoint, cfg, meta['pool_manifest_sha256']
                else:
                    if original_cfg != cfg or original_pool != meta['pool_manifest_sha256']:
                        raise ValueError('Probe or training protocol differs between paired runs')
                    # Both experiments must present the same evidence to the same fitted model at every stage.
                    old_probes, new_probes = original['crp']['probes'], checkpoint['crp']['probes']
                    if len(old_probes) != len(new_probes) or any(
                            len(a) != len(b) or any(not torch.equal(x, y) for x, y in zip(a, b))
                            for a, b in zip(old_probes, new_probes)):
                        raise ValueError('Probe tensors differ between center modes')
                    for a, b in zip(original['admissions'], checkpoint['admissions']):
                        if (a['teammate'] != b['teammate'] or a['probe_slice'] != b['probe_slice']
                                or a['model'].keys() != b['model'].keys()
                                or any(not torch.equal(v, b['model'][k]) for k, v in a['model'].items())):
                            raise ValueError('Fitted models or admission order differ between center modes')
                path = stage / f'{mode}_seed{seed}.jsonl'
                runs[mode][seed] = replay(run, meta, checkpoint, path)
        pairs = list(combinations(original_cfg['order'], 2))
        stability = {}
        for mode, by_seed in runs.items():
            scores = [dict(seeds=[a, b], **compare_partitions(by_seed[a]['assignments'],
                                                              by_seed[b]['assignments'], pairs))
                      for a, b in combinations(seeds, 2)]
            stability[mode] = dict(seed_pairs=scores, mean_ari=sum(x['ari'] for x in scores) / len(scores))
        report = dict(action_names=ACTION_NAMES, index_base=0, pool_sha256=original_pool, runs=runs,
                      cross_seed_stability=stability, checked='Identical compact probe tensors and all '
                      'post-fit model parameters at all eight admissions within each seed pair.')
        (stage/'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        stage.rename(destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('historical', type=Path)
    parser.add_argument('current', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--seeds', type=int, nargs='+', default=[7, 17, 27])
    args = parser.parse_args()
    paths = [p.resolve() for p in (args.historical, args.current, args.destination)]
    if (os.getenv('FASTAP_ROOT') != str(ROOT) or any(ROOT/'outputs' not in p.parents for p in paths)
            or len(set(paths)) != 3 or len(set(args.seeds)) != len(args.seeds) or len(args.seeds) < 2):
        parser.error('Use separate project output directories, project-local caches and at least two distinct seeds')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.set_num_threads(1)
    audit(*paths, args.seeds)
    print(paths[-1]/'comparison.json')
