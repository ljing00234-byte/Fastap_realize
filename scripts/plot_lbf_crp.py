"""Plot the LBF Figure 6 reproduction from measured cross-play and CRP trajectory embeddings."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
COLORS = ('#176aa1', '#d66118', '#278050', '#a64e8d', '#8a6720', '#2b8c90', '#7255a2', '#bb4848')
MARKERS = ('o', '^', 's', 'D', 'P', 'X', 'v', 'h')


def aggregate_crossplay(crossplay, assignments):
    """Equal weight per ordered member pair; rows supply the two original partners."""
    ids = crossplay['ids']
    if not ids or len(set(ids)) != len(ids) or set(ids) != set(assignments):
        raise ValueError("Cross-play and CRP must contain the same distinct teammate IDs")
    scores = np.asarray([[cell['mean_return'] for cell in row] for row in crossplay['crossplay']], dtype=np.float64)
    if scores.shape != (len(ids), len(ids)) or not np.isfinite(scores).all():
        raise ValueError("Expected a finite square matrix of measured member returns")
    labels = np.array([assignments[name] for name in ids])
    clusters = list(dict.fromkeys(labels.tolist()))
    means = []
    counts = []
    for source in clusters:
        mean_row, count_row = [], []
        for target in clusters:
            block = scores[np.ix_(labels == source, labels == target)]
            mean_row.append(float(block.mean()))
            count_row.append(int(block.size))
        means.append(mean_row)
        counts.append(count_row)
    same_cluster = labels[:, None] == labels[None, :]
    original = np.eye(len(ids), dtype=bool)
    summaries = {}
    for name, mask in [('original_partners', original), ('within_cluster_including_self', same_cluster),
                       ('within_cluster_other_members', same_cluster & ~original),
                       ('across_clusters', ~same_cluster)]:
        values = scores[mask]
        summaries[name] = dict(pairings=int(values.size), mean_return=float(values.mean()) if values.size else None)
    return dict(cluster_ids=clusters, members={c: [name for name in ids if assignments[name] == c] for c in clusters},
                rows='cluster of the original partners in slots 0 and 1',
                columns='cluster of the frozen teammate in slot 2',
                weighting='Equal weight for each ordered member pairing, including self-pairings on the diagonal.',
                metric='Raw team return averaged over the same maps for each member pairing.',
                maps_per_pair=crossplay['n_maps'], first_map_seed=crossplay['first_map_seed'],
                pairings=counts, mean_return=means, summaries=summaries)


def project_embeddings(rows, assignments):
    """Joint centered PCA of final-encoder trajectories, using the first three right singular vectors."""
    if not rows or {r['teammate'] for r in rows} != set(assignments):
        raise ValueError('Missing trajectory embeddings for a clustered teammate')
    if any(r['cluster'] != assignments[r['teammate']] for r in rows):
        raise ValueError('A trajectory has a cluster label inconsistent with the CRP assignment')
    if len({(r['teammate'], r['episode']) for r in rows}) != len(rows):
        raise ValueError('Duplicate probe episode within a teammate')
    vectors = np.asarray([r['embedding'] for r in rows], dtype=np.float64)
    if vectors.ndim != 2 or min(vectors.shape) < 3 or not np.isfinite(vectors).all():
        raise ValueError('3D PCA needs at least three finite vectors of dimension three or greater')
    center = vectors.mean(axis=0)
    centered = vectors - center
    _, singular, components = np.linalg.svd(centered, full_matrices=False)
    basis = components[:3].copy()
    # Fix the arbitrary axis signs for reproducible coordinates.
    for axis in basis:
        if axis[np.argmax(np.abs(axis))] < 0:
            axis *= -1
    power = singular ** 2
    if power.sum() == 0:
        raise ValueError('All trajectory embeddings coincide; PCA explained variance is undefined')
    coordinates = centered @ basis.T
    return dict(method='PCA on centered final-encoder trajectory embeddings, without feature scaling.',
                center=center.tolist(), components=basis.tolist(), explained_variance_ratio=(power[:3]/power.sum()).tolist(),
                rows=[dict(teammate=r['teammate'], cluster=r['cluster'], episode=r['episode'], xyz=xyz.tolist())
                      for r, xyz in zip(rows, coordinates)])


def _heatmap(ax, matrix, labels, title, *, counts=None):
    image = ax.imshow(matrix, cmap='viridis', vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.tick_params(length=0, pad=5)
    ax.set_xlabel('换入的冻结队友' if counts is None else '换入队友所属簇')
    ax.set_ylabel('两名原搭档的训练队伍' if counts is None else '两名原搭档的训练队伍所属簇')
    ax.set_title(title, fontsize=12, pad=11)
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            label = f'{value:.3f}'
            if counts is not None:
                label += f'\n{counts[i][j]} 对'
            ax.text(j, i, label, ha='center', va='center', fontsize=8 if len(labels) > 5 else 10,
                    color='white' if value < .52 else '#141820', fontweight='bold' if i == j else 'normal')
    return image


def _draw_pca(ax, pca, assignments, cluster_ids):
    from matplotlib.ticker import MaxNLocator

    handles = []
    for i, (teammate, cluster) in enumerate(assignments.items()):
        points = np.array([row['xyz'] for row in pca['rows'] if row['teammate'] == teammate])
        color = COLORS[cluster_ids.index(cluster) % len(COLORS)]
        handles.append(ax.scatter(*points.T, c=color, marker=MARKERS[i % len(MARKERS)], s=37,
                                  edgecolor='#202a31', linewidth=.55, alpha=.90, depthshade=False,
                                  label=f'{teammate} · {cluster}'))
    for i, axis in enumerate((ax.xaxis, ax.yaxis, ax.zaxis)):
        axis.set_label_text(f'PC{i+1} ({pca["explained_variance_ratio"][i]:.1%})', fontsize=9)
        axis.set_major_locator(MaxNLocator(4))
        axis.pane.set_facecolor('#f0f2f4')
    ax.view_init(elev=23, azim=-58)
    ax.tick_params(labelsize=8, pad=1)
    return handles


def _embedding_figure(plt, path, pca, assignments, cluster_ids, title, note):
    fig = plt.figure(figsize=(7.2, 6.8))
    ax = fig.add_axes([.06,.20,.85,.70], projection='3d')
    handles = _draw_pca(ax, pca, assignments, cluster_ids)
    fig.suptitle(title, fontsize=15)
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5,.075), ncol=4, frameon=False,
               columnspacing=1.3, handletextpad=.3)
    fig.text(.5,.04, note, ha='center', va='center', fontsize=9)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def render_plots(directory, source, cluster_data, pca, assignments):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Noto Sans CJK JP', 'DejaVu Sans'],
                         'axes.unicode_minus': False, 'font.size': 10, 'axes.labelpad': 6})
    cluster_ids = cluster_data['cluster_ids']
    matrix, counts = cluster_data['mean_return'], cluster_data['pairings']
    n_maps = cluster_data['maps_per_pair']
    original = [[cell['mean_return'] for cell in row] for row in source['crossplay']]
    member_text = '     '.join(f'{c}  {", ".join(ids)}' for c, ids in cluster_data['members'].items())
    variance = sum(pca['explained_variance_ratio'])
    note = f'每点为一条训练轨迹，颜色为入簇标签，形状为队友。前三轴合计解释 {variance:.1%} 方差。'
    pair_note = f'每格等权平均所列有向组合的回报。每对使用相同的 {n_maps} 张地图，簇内含原配。'

    fig = plt.figure(figsize=(16.8, 6.1))
    grid = fig.add_gridspec(1, 3, left=.055, right=.98, bottom=.23, top=.83, wspace=.25,
                           width_ratios=[1, 1.05, 1])
    left = fig.add_subplot(grid[0])
    image = _heatmap(left, original, source['ids'], '队友互换')
    middle = fig.add_subplot(grid[1], projection='3d')
    handles = _draw_pca(middle, pca, assignments, cluster_ids)
    middle.set_title('行为嵌入 · 训练轨迹 PCA', fontsize=12, pad=12)
    right = fig.add_subplot(grid[2])
    _heatmap(right, matrix, cluster_ids, '按 CRP 簇汇总的交叉评估', counts=counts)
    fig.suptitle('LBF 正式队友的行为分簇与交叉评估', y=.963, fontsize=16)
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5,.075), ncol=4, frameon=False,
               fontsize=8, handletextpad=.2, columnspacing=.7)
    cb = fig.colorbar(image, cax=fig.add_axes([.105,.12,.15,.018]), orientation='horizontal')
    cb.set_label('平均团队回报', fontsize=9)
    fig.text(.81,.12, f'每个组合共用 {n_maps} 张地图，对角线包括原配。', ha='center', fontsize=9)
    fig.text(.5,.035,'每点为一条训练轨迹，颜色为入簇标签，形状为队友。', ha='center', fontsize=9)
    fig.savefig(directory/'figure6.png', dpi=180, bbox_inches='tight', pad_inches=.18)
    plt.close(fig)

    _embedding_figure(plt, directory/'behavior_embeddings.png', pca, assignments, cluster_ids,
                      'LBF 队友的行为嵌入 · 训练轨迹', note)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    fig.subplots_adjust(left=.13, right=.87, top=.87, bottom=.21)
    image = _heatmap(ax, matrix, cluster_ids, 'LBF 队友簇之间的交叉评估', counts=counts)
    cb = fig.colorbar(image, ax=ax, shrink=.88, pad=.035)
    cb.set_label('平均团队回报')
    fig.text(.49,.098,member_text, ha='center', fontsize=9)
    fig.text(.49,.04,pair_note, ha='center', fontsize=9)
    fig.savefig(directory/'cluster_crossplay.png', dpi=180, bbox_inches='tight', pad_inches=.16)
    plt.close(fig)


def plot_run(directory, crossplay_path, heldout_embeddings=None):
    directory, crossplay_path = directory.resolve(), crossplay_path.resolve()
    if ROOT/'outputs' not in directory.parents:
        raise ValueError('Use a clustering run under project outputs/')
    if heldout_embeddings is not None:
        heldout_embeddings = heldout_embeddings.resolve()
        if ROOT/'outputs' not in heldout_embeddings.parents:
            raise ValueError('Use held-out embeddings under project outputs/')
    run = json.loads((directory/'clustering.json').read_text())
    manifest_path = (ROOT / run['pool_manifest']).resolve()
    if crossplay_path != manifest_path.with_name('crossplay.json'):
        raise ValueError('Choose the cross-play matrix from the same pool as the CRP run')
    manifest_bytes = manifest_path.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != run['pool_manifest_sha256']:
        raise ValueError('The source pool manifest changed after clustering')
    crossplay_bytes = crossplay_path.read_bytes()
    source = json.loads(crossplay_bytes)
    if (source['rows'], source['columns']) != ('original trained partners in slots 0/1',
                                               'evaluated frozen teammate in slot 2'):
        raise ValueError('Cross-play axes do not match the original-partners by frozen-teammate protocol')
    members = json.loads(manifest_bytes)['members']
    if source['ids'] != [m['id'] for m in members]:
        raise ValueError('Cross-play teammate IDs/order differ from the source pool')
    if run['member_sha256'] != {m['id']: m['sha256'] for m in members}:
        raise ValueError('Clustered actor hashes differ from those in the source pool')
    if set(run['assignments']) != set(source['ids']):
        raise ValueError('Clustered teammate IDs differ from the source pool')
    assignments = {name: run['assignments'][name] for name in source['ids']}
    rows = json.loads((directory/'embeddings.json').read_text())['rows']
    cluster_data = aggregate_crossplay(source, assignments)
    cluster_data['source_crossplay'] = str(crossplay_path.relative_to(ROOT))
    cluster_data['source_crossplay_sha256'] = hashlib.sha256(crossplay_bytes).hexdigest()
    pca = project_embeddings(rows, assignments)
    render_plots(directory, source, cluster_data, pca, assignments)
    for name, obj in [('cluster_crossplay.json', cluster_data), ('pca.json', pca)]:
        (directory/name).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    if heldout_embeddings is not None:
        import matplotlib.pyplot as plt

        heldout = json.loads(heldout_embeddings.read_text())
        if heldout['checkpoint_sha256'] != hashlib.sha256((directory/'crp.pt').read_bytes()).hexdigest():
            raise ValueError('Held-out embeddings come from a different model')
        heldout_pca = project_embeddings(heldout['rows'], assignments)
        silhouette = heldout['silhouette_16d']
        silhouette_text = '未定义' if silhouette is None else f'{silhouette:.3f}'
        note = (f'{len(heldout["map_seeds"])} 张留出地图，每张 8 名队友共用 T1 的两名搭档。颜色沿用训练入簇标签，形状为队友。\n'
                f'PCA 在留出轨迹上拟合，前三轴解释 {sum(heldout_pca["explained_variance_ratio"]):.1%} 方差。'
                f'原始 16 维轮廓系数 {silhouette_text}。')
        _embedding_figure(plt, heldout_embeddings.with_suffix('.png'), heldout_pca, assignments,
                          cluster_data['cluster_ids'], 'LBF 队友的行为嵌入 · 留出轨迹', note)


def export_heldout_embeddings(directory, heldout_path):
    """Encode fixed-T1 query trajectories and retain the member's original admission label."""
    import sys
    sys.path.insert(0, str(ROOT))
    import torch
    from fastap_methods.fastap_crp import BehaviorModel
    from scripts.lbf_crp_heldout import evaluate_heldout

    directory, heldout_path = directory.resolve(), heldout_path.resolve()
    if not all(ROOT/'outputs' in path.parents for path in (directory, heldout_path)):
        raise ValueError('Use the clustering run and held-out data under project outputs/')
    run = json.loads((directory/'clustering.json').read_text())
    checkpoint = directory/'crp.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    trials = torch.load(heldout_path, map_location='cpu', weights_only=True)
    order = run['protocol']['order']
    if {k: f'C{v+1}' for k, v in saved['crp']['assignments'].items()} != run['assignments']:
        raise ValueError('Training assignments differ from the frozen encoder checkpoint')
    manifest = ROOT / run['pool_manifest']
    if (trials['pool_sha256'] != run['pool_manifest_sha256'] or
            hashlib.sha256(manifest.read_bytes()).hexdigest() != trials['pool_sha256'] or
            trials['members'] != order or trials['row_order'] != 'map_index * n_members + member_index'):
        raise ValueError('Query trajectory identities or pool mismatch')
    maps = trials['map_seeds']['query']
    batch = trials['trials']['fixed_T1']['query']
    starts = batch['state'][:, 0].reshape(len(maps), len(order), -1)
    unique_starts = {row[0].numpy().tobytes() for row in starts}
    training_starts = {probe[0][0].numpy().tobytes() for probe in saved['crp']['probes']}
    if (not torch.equal(starts, starts[:, :1].expand_as(starts)) or len(unique_starts) != len(maps)
            or not training_starts.isdisjoint(unique_starts)):
        raise ValueError('Each query map must be shared across teammates and disjoint from training maps')
    decoder = saved['crp'].get('decoder', 'paper')
    model = BehaviorModel(**saved['model_args'], decoder=decoder)
    model.load_state_dict(saved['crp']['model'])
    crossplay = json.loads(manifest.with_name('crossplay.json').read_text())
    query = dict(members=order, map_seeds=maps, trials={'fixed_T1': batch})
    result = evaluate_heldout(model, run['assignments'], query, crossplay, include_rows=True)['heldout']['fixed_T1']
    result.update(map_seeds=maps, members=order, label_source='training CRP assignments; no readmission',
                  partner='fixed_T1', split='query', decoder=decoder, seed=run['protocol']['seed'],
                  heldout_sha256=hashlib.sha256(heldout_path.read_bytes()).hexdigest(),
                  checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    target = heldout_path.with_name(f'heldout_embeddings_{decoder}_seed{result["seed"]}.json')
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path, help='directory containing clustering.json and embeddings.json')
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument('--crossplay', type=Path, help='plot the crossplay.json of the source teammate pool')
    operation.add_argument('--export-query', type=Path, metavar='HELDOUT.PT',
                           help='use the project PyTorch environment to export fixed-T1 query embeddings')
    parser.add_argument('--heldout-embeddings', type=Path,
                        help='also draw this run’s exported fixed-T1 query embeddings alongside their JSON')
    args = parser.parse_args()
    if args.export_query:
        if args.heldout_embeddings:
            parser.error('--heldout-embeddings is used with --crossplay')
        print(export_heldout_embeddings(args.run_dir, args.export_query))
    else:
        plot_run(args.run_dir, args.crossplay, args.heldout_embeddings)
