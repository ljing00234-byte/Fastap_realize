"""Measure the effect of changing only group arrival order in the formal LBF CRP run."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
import json
from multiprocessing import get_context
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from fastap_methods.fastap_crp import TeammateCRP, _collate
from scripts.audit_lbf_crp import FORMAL, file_hash
from scripts.plot_lbf_crp import aggregate_crossplay


def compare_partitions(first, second, pairs):
    # ARI from the pair-count contingency table, independent of the names of the clusters.
    a = {p for p in pairs if first[p[0]] == first[p[1]]}
    b = {p for p in pairs if second[p[0]] == second[p[1]]}
    n = len(pairs)
    denom = (len(a) + len(b)) * n - 2 * len(a) * len(b)
    ari = 2 * (n * len(a & b) - len(a) * len(b)) / denom if denom else 1.
    changed = a ^ b
    return dict(ari=ari, changed_pairs=[list(p) for p in pairs if p in changed],
                same_in_both=len(a & b), separate_in_both=n-len(a | b))


def fit_seed(seed):
    torch.set_num_threads(1)
    protocol_file = json.loads((FORMAL/'clustering.json').read_text())
    config = protocol_file['protocol']
    saved = torch.load(FORMAL/'crp.pt', weights_only=True, map_location='cpu')
    source = {event['teammate']: event for event in protocol_file['admissions']}
    names = config['order']
    if names != list(source) or len(source) != 8 or len(saved['admissions']) != len(names):
        raise ValueError('The source run must contain the eight admissions in formal run order')
    keys = ('state', 'teammate_actions', 'teammate_mask', 'filled')
    batches = {}
    for name, event in source.items():
        left, right = event['probe_slice']
        batches[name] = dict(zip(keys, _collate(saved['crp']['probes'][left:right], 'cpu')))

    orders = {'formal': names, 'reversed': names[::-1], 'rotate_four': names[4:] + names[:4]}
    runs = []
    for order, roster in orders.items():
        torch.manual_seed(seed)
        crp = TeammateCRP(**saved['model_args'], alpha=config['alpha'], lr=config['lr'],
                          minibatch_size=config['minibatch_size'], device='cpu')
        verify = order == 'formal' and seed == config['seed']
        for i, name in enumerate(roster):
            label = crp.add_group(name, batches[name], model_steps=config['crp_model_steps'])
            if verify:
                snapshot = saved['admissions'][i]
                match = all(torch.equal(v, snapshot['model'][k]) for k, v in crp.model.state_dict().items())
                if not match or f'C{label+1}' != source[name]['cluster']:
                    raise RuntimeError(f'Formal model or assignment mismatch at {name}')
        assignments = {name: f'C{crp.assignments[name]+1}' for name in names}
        runs.append(dict(seed=seed, order=order, arrival=roster, assignments=assignments,
                         clusters=len(crp.counts), verified_formal_snapshots=len(names) if verify else 0))
    return runs


def run(output):
    seeds = (7, 17, 27)
    source = json.loads((FORMAL/'clustering.json').read_text())
    pool = ROOT / source['pool_manifest']
    if file_hash(pool) != source['pool_manifest_sha256']:
        raise ValueError('Formal pool manifest changed')
    with ProcessPoolExecutor(max_workers=3, mp_context=get_context('spawn')) as workers:
        results = [row for group in workers.map(fit_seed, seeds) for row in group]

    config = source['protocol']
    names = config['order']
    crossplay_path = pool.with_name('crossplay.json')
    crossplay = json.loads(crossplay_path.read_text())
    if crossplay['ids'] != names:
        raise ValueError('Unexpected IDs/order in cross-play')
    pairs = list(combinations(names, 2))
    for row in results:
        summaries = aggregate_crossplay(crossplay, row['assignments'])['summaries']
        row['within_distinct'] = summaries['within_cluster_other_members']
        row['across'] = summaries['across_clusters']
    comparisons = []
    for seed in seeds:
        for a, b in combinations([r for r in results if r['seed'] == seed], 2):
            comparisons.append(dict(seed=seed, first=a['order'], second=b['order'],
                                    **compare_partitions(a['assignments'], b['assignments'], pairs)))
    coassign = [{'pair': [a, b], 'count': sum(r['assignments'][a] == r['assignments'][b] for r in results)}
                for a, b in pairs]
    report = dict(protocol='Reuse each teammate’s formal 8 probes. Reset the model and optimizer per run. '
                           'The arrival order changes both what the encoder has seen and the available CRP clusters.',
                  formal_weights_sha256=file_hash(FORMAL/'crp.pt'), pool_manifest_sha256=file_hash(pool),
                  crossplay_sha256=file_hash(crossplay_path), settings=config, seeds=seeds, results=results,
                  order_comparisons_at_fixed_seed=comparisons, coassignment_counts_over_nine_runs=coassign)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, nargs='?',
                        default=FORMAL.parent/'crp_diagnostics/crossplay_admission_audit/order.json')
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if os.getenv('FASTAP_ROOT') != str(ROOT) or ROOT/'outputs' not in output.parents or output.exists():
        parser.error('Source scripts/epymarl-env.sh and use a new output file under project outputs/')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    output.parent.mkdir(exist_ok=True, parents=True)
    run(output)
