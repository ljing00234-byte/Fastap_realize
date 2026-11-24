"""Check alignment of the formal CRP partition with directed teammate swaps on paired LBF starts."""

import argparse
from itertools import combinations
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

DEFAULT_MATRIX = ROOT/'outputs/fastap_lbf_formal_pool_01/crp_diagnostics/crossplay_admission_audit/matrix.json'


def board_keys(first_seed, n_maps):
    from fastap_methods.pool_crossplay import _initial_boards

    maps = _initial_boards(first_seed, n_maps)
    return {tuple(np.asarray(maps[k][i]).tobytes() for k in ('field', 'positions', 'levels'))
            for i in range(n_maps)}


def symmetric_pairs(data):
    ids = data['ids']
    returns = np.array([[c['mean_return'] for c in row] for row in data['crossplay']])
    disagreement = np.array(data['action_disagreement'])
    if disagreement.shape != returns.shape or not np.allclose(disagreement, disagreement.T):
        raise ValueError('Invalid common-history action disagreement matrix')
    pairs = list(combinations(range(len(ids)), 2))
    return returns, pairs, np.array([(returns[a,b] + returns[b,a]) / 2 for a,b in pairs]), np.array(
        [disagreement[a,b] for a,b in pairs])


def same_member_pairs(partition, pairs):
    return np.array([any(a in group and b in group for group in partition) for a, b in pairs])


def describe(partition, pairs, swaps, disagreement):
    within = same_member_pairs(partition, pairs)
    x, y = swaps[within].mean(), swaps[~within].mean()
    return dict(within_unordered_pairs=int(within.sum()), across_unordered_pairs=int((~within).sum()),
                within_return=float(x), across_return=float(y), return_gap=float(x-y),
                within_action_disagreement=float(disagreement[within].mean()),
                across_action_disagreement=float(disagreement[~within].mean()))


def same_sizes_partitions(n):
    # The formal partition has one triple, one pair, and three unlabeled singletons.
    members = set(range(n))
    for triple in combinations(range(n), 3):
        remaining = members.difference(triple)
        for pair in combinations(sorted(remaining), 2):
            yield [triple, pair, *[(i,) for i in sorted(remaining.difference(pair))]]


def average_ranks(x):
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - .5*(counts-1))[inverse]


def main(output):
    from fastap_methods.pool_crossplay import evaluate_pool
    from fastap_scenarios.fastap_rollout import verified_pool_members
    from scripts.audit_lbf_crp import FORMAL, file_hash

    pool = FORMAL.parent
    source = json.loads((pool/'pool.json').read_text())
    clusters = json.loads((FORMAL/'clustering.json').read_text())
    if file_hash(pool/'pool.json') != clusters['pool_manifest_sha256']:
        raise ValueError('The formal pool manifest has changed')
    ids = [member for member, _ in verified_pool_members(pool/'pool.json', 'lbf')]
    original = json.loads((pool/'crossplay.json').read_text())
    if original['ids'] != ids or original['n_maps'] != 256 or original['first_map_seed'] != 870000:
        raise ValueError('Expected the original 8-member, 256-map formal cross-play matrix')
    partition = [tuple(ids.index(member) for member in group) for group in clusters['clusters'].values()]
    if sorted(map(len, partition)) != [1,1,1,2,3] or sorted(i for g in partition for i in g) != list(range(8)):
        raise ValueError('Expected the complete formal seed-7 partition')
    partitions = list(same_sizes_partitions(len(ids)))
    _, pairs, swaps, disagreement = symmetric_pairs(original)
    scores = np.array([describe(p, pairs, swaps, disagreement)['return_gap'] for p in partitions])
    # Select using only the original means; retain this partition on every subsequent map.
    reference = partitions[scores.argmax()]
    actual_gap = describe(partition, pairs, swaps, disagreement)['return_gap']
    worse, better = int((scores < actual_gap).sum()), int((scores > actual_gap).sum())

    first_seed, n_maps = 987000, 256
    original_starts = board_keys(original['first_map_seed'], original['n_maps'])
    fresh_starts = board_keys(first_seed, n_maps)
    if (len(fresh_starts) != n_maps or len(original_starts) != original['n_maps']
            or not original_starts.isdisjoint(fresh_starts)):
        raise ValueError('Repeated full initial LBF state within or across map samples')
    new = evaluate_pool(pool, source['members'], first_seed, n_maps)
    if new['ids'] != ids or (new['rows'], new['columns']) != (original['rows'], original['columns']):
        raise ValueError('Cross-play axes differ between samples')
    results = {}
    for label, matrix in [('original', original), ('new_maps', new)]:
        returns, pairs, swaps, disagreement = symmetric_pairs(matrix)
        rows = [dict(a=ids[a], b=ids[b], both_directions=[float(returns[a,b]),float(returns[b,a])],
                     avg=float(v), action_disagreement=float(d),
                     same_crp=clusters['assignments'][ids[a]] == clusters['assignments'][ids[b]])
                for (a,b),v,d in zip(pairs,swaps,disagreement)]
        results[label] = dict(matrix=matrix, original_partner_return=float(returns.diagonal().mean()),
                              swapped_member_return=float(swaps.mean()),
                              crp=describe(partition, pairs, swaps, disagreement),
                              reference=describe(reference, pairs, swaps, disagreement),
                              spearman_return_vs_disagreement=float(np.corrcoef(average_ranks(swaps),
                                                                                average_ranks(disagreement))[0,1]),
                              unordered_swaps=sorted(rows,key=lambda row:-row['avg']))
    report = dict(pool_manifest_sha256=file_hash(pool/'pool.json'), original_matrix_sha256=file_hash(pool/'crossplay.json'),
                  crp_assignments=clusters['assignments'], reference=[[ids[i] for i in g] for g in reference],
                  selection='Reference maximizes within-minus-across return over all partitions of the formal group '
                            'sizes on original maps only. New maps test this frozen choice. Diagonal is excluded.',
                  action_disagreement='Every frozen role-2 policy is queried on the same histories generated by '
                                      'the T1 team. Ranks compare 28 unordered pairs; pairs share policies.',
                  random_partition_comparison=dict(total=len(partitions), better=better, worse=worse,
                                                   tied=len(partitions)-better-worse, actual_rank=better+1,
                                                   mean_return_gap=float(scores.mean()),
                                                   one_sided_fraction_at_least_formal=(len(partitions)-worse)/len(partitions)),
                  results=results)
    with output.open('x',encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')


def compatibility_groups(symmetric_return):
    """Average-link the old-map mutual returns and cut at three groups."""
    groups = [(i,) for i in range(len(symmetric_return))]
    while len(groups) > 3:
        i, j = max(combinations(range(len(groups)), 2), key=lambda pair:
                   symmetric_return[np.ix_(groups[pair[0]], groups[pair[1]])].mean())
        merged = tuple(sorted((*groups[i], *groups[j])))
        groups = [g for k, g in enumerate(groups) if k not in (i, j)] + [merged]
    return sorted(groups)


def render_plot(matrix_file):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    audit = json.loads(matrix_file.read_text())
    old, new = [audit['results'][key]['matrix'] for key in ('original', 'new_maps')]
    ids, assignments = old['ids'], audit['crp_assignments']
    if new['ids'] != ids:
        raise ValueError('Cross-play axes differ between samples')
    raw_old = symmetric_pairs(old)[0]
    raw_new = symmetric_pairs(new)[0]
    compatibility = compatibility_groups((raw_old + raw_old.T) / 2)
    formal = [tuple(i for i, name in enumerate(ids) if assignments[name] == cluster)
              for cluster in sorted(set(assignments.values()))]
    mutual = np.ma.array((raw_new + raw_new.T) / 2, mask=np.eye(len(ids), dtype=bool))
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('#e8ebee')
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.9))
    fig.subplots_adjust(left=.05, right=.93, top=.84, bottom=.17, wspace=.22)
    for panel, (ax, groups, title) in enumerate(zip(
            axes, [compatibility, formal], ['Compatibility groups from old maps', 'Formal CRP groups'])):
        order = [i for group in groups for i in group]
        data = mutual[np.ix_(order, order)]
        im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
        labels = [ids[i] for i in order]
        ax.set_xticks(range(len(ids)), labels, fontsize=10)
        row_labels = labels if panel == 0 else [f'{name}  {assignments[name]}' for name in labels]
        ax.set_yticks(range(len(ids)), row_labels, fontsize=10)
        ax.tick_params(length=0)
        ax.set_title(title, pad=10, fontsize=13)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for boundary in np.cumsum([len(g) for g in groups])[:-1] - .5:
            ax.axvline(boundary, color='white', lw=2)
            ax.axhline(boundary, color='white', lw=2)
        for row, i in enumerate(order):
            for col, j in enumerate(order):
                if i == j:
                    ax.text(col, row, '–', ha='center', va='center', color='#747e87')
                    continue
                value = float(data[row, col])
                color = '#10171c' if value > .51 else 'white'
                ax.text(col, row, f'{value:.2f}', ha='center', va='center', color=color, fontsize=10)
                if panel and assignments[ids[i]] == assignments[ids[j]]:
                    ax.add_patch(Rectangle((col-.44, row-.44), .88, .88, fill=False,
                                           edgecolor='#ff9150', linewidth=2.2))
    fig.suptitle(f'LBF teammate interchangeability on {new["n_maps"]} new maps',
                 fontsize=18, fontweight='semibold', y=.96)
    cax = fig.add_axes([.937, .20, .012, .61])
    colorbar = fig.colorbar(im, cax=cax)
    colorbar.set_label('Mean return, both swap directions')
    colorbar.outline.set_visible(False)
    fig.text(.5, .065, 'Orange outlines mark swaps within a formal CRP cluster.    Dash = original partners.',
             ha='center', fontsize=10, color='#37414b')
    fig.savefig(matrix_file.with_name('compatibility_audit.png'), dpi=200, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, nargs='?', default=DEFAULT_MATRIX)
    parser.add_argument('--plot', action='store_true', help='Render existing matrix JSON without resampling.')
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if os.environ.get('FASTAP_ROOT') != str(ROOT) or ROOT/'outputs' not in output.parents:
        parser.error('Source scripts/epymarl-env.sh and select a file under project outputs/')
    if args.plot:
        render_plot(output)
    else:
        if output.exists():
            parser.error('Choose a fresh output file when collecting new data')
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        output.parent.mkdir(parents=True, exist_ok=True)
        main(output)
