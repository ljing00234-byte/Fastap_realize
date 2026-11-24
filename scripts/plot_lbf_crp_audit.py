"""Plot held-out trajectory embeddings and paired CRP audit metrics."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.plot_lbf_crp import COLORS, project_embeddings


PROBE = 'fixed_T1_ego'
SEED_STYLE = [('#146c94', 'o'), ('#b85423', '^'), ('#477b54', 's')]


def plot_clouds(directory, report, runs, clouds):
    pca_seed = report['seed_list'][0]
    variants = list(report['variants'])
    nrows = (len(variants) + 2) // 3
    fig = plt.figure(figsize=(13, 3.8 * nrows + 1.7))
    fig.subplots_adjust(left=.045, right=.96, top=1 - 1.23 / fig.get_figheight(),
                        bottom=.8 / fig.get_figheight(), wspace=.12, hspace=.3)
    for i, variant in enumerate(variants):
        record = runs[variant, pca_seed]
        members = record['assignments']
        rows = clouds[variant][PROBE]
        expected = {(name, seed) for name in members for seed in report['heldout_map_seeds']}
        if len(rows) != len(expected) or {(r['teammate'], r['episode']) for r in rows} != expected:
            raise ValueError(f'Incomplete held-out trajectories for {variant}')
        if any(r['cluster'] != members[r['teammate']] for r in rows):
            raise ValueError(f'Unexpected trajectory label in {variant}')
        ax = fig.add_subplot(nrows, 3, i + 1, projection='3d')
        emb = np.array([r['embedding'] for r in rows], dtype=float)
        if np.all(emb == emb[0]):
            xyz = np.zeros((len(emb), 3))
            variance = 'Identical embeddings'
        else:
            pca = project_embeddings(rows, members)
            xyz = np.array([r['xyz'] for r in pca['rows']])
            variance = f'PC1–3 explain {sum(pca["explained_variance_ratio"]):.1%}'
        labels = np.array([r['cluster'] for r in rows])
        for label in dict.fromkeys(members.values()):
            points = xyz[labels == label]
            ax.scatter(*points.T, c=COLORS[int(label[1:]) - 1], s=13, alpha=.73,
                       linewidth=.22, edgecolor='white', depthshade=False)
        # Keep an equal data scale on all three axes in each independently fitted space.
        centers = (xyz.min(0) + xyz.max(0)) / 2
        radius = np.ptp(xyz, axis=0).max() * .53 or 1.
        for axis, center, name in zip((ax.xaxis, ax.yaxis, ax.zaxis), centers, ('PC1', 'PC2', 'PC3')):
            axis.set_view_interval(center - radius, center + radius, ignore=True)
            axis.set_major_locator(MaxNLocator(3))
            axis.set_label_text(name, fontsize=8)
            axis.labelpad = -1
        ax.set_box_aspect((1, 1, 1))
        ax.tick_params(labelsize=7, pad=0)
        ax.view_init(elev=23, azim=-58)
        ax.set_title(f'{variant}\n{record["num_clusters"]} clusters  ·  {variance}', fontsize=10)

    fig.suptitle('LBF teammate behavior on held-out maps', fontsize=17, y=.986)
    fig.text(.5, 1 - .57 / fig.get_figheight(),
             f'Seed {pca_seed}  ·  fixed T1 ego pair  ·  {len(expected)} trajectories per panel  ·  '
             'PCA fitted independently to each model', ha='center', fontsize=10)
    k = max(runs[v, pca_seed]['num_clusters'] for v in variants)
    color_handles = [Line2D([], [], linestyle='none', marker='o', color=COLORS[j], label=f'C{j+1}')
                     for j in range(k)]
    fig.legend(handles=color_handles, ncol=k, loc='lower center', bbox_to_anchor=(.5, .30 / fig.get_figheight()),
               frameon=False, columnspacing=1.2, handletextpad=.2)
    fig.text(.5, .14 / fig.get_figheight(),
             'Colors follow online cluster creation order within each model. Cluster IDs have no correspondence across panels.',
             ha='center', fontsize=9)
    fig.savefig(directory / 'audit_heldout_pca.png', dpi=180)
    plt.close(fig)


def values(record):
    crossplay = record['crossplay']
    within = crossplay['within_cluster_other_members']['mean_return']
    across = crossplay['across_clusters']['mean_return']
    delta = None if within is None or across is None else within - across
    return (record['updates_above_99pct'] / record['updates'], record['heldout'][PROBE]['silhouette_16d'], delta)


def plot_metrics(directory, report, runs):
    variants, seeds = list(report['variants']), report['seed_list']
    if len(seeds) > len(SEED_STYLE):
        raise ValueError('Expected at most three audit seeds')
    height = max(4.7, len(variants) * .48 + 2.9)
    fig, axes = plt.subplots(1, 3, figsize=(14, height), sharey=True)
    fig.subplots_adjust(left=.195, right=.95, top=1 - 1.33 / height, bottom=1.12 / height, wspace=.25)
    offsets = (np.arange(len(seeds)) - (len(seeds) - 1) / 2) * .19
    handles = []
    for seed, offset, (color, marker) in zip(seeds, offsets, SEED_STYLE):
        handles.append(Line2D([], [], linestyle='none', color=color, marker=marker, label=f'Seed {seed}'))
        for y, variant in enumerate(variants):
            for ax, value in zip(axes, values(runs[variant, seed])):
                if value is None:
                    ax.text(1.02, y + offset, '×', color=color, fontsize=11, ha='left', va='center',
                            transform=ax.get_yaxis_transform())
                else:
                    ax.scatter(value, y + offset, s=34, c=color, marker=marker, zorder=3)
    for j, (ax, title, label) in enumerate(zip(axes,
            ['Single-trajectory dominance', 'Held-out separation', 'Teammate interchangeability'],
            ['Updates with top weight > 99%', '16D silhouette', 'Within − across mean return'])):
        ax.set_title(title, fontsize=12, loc='left')
        ax.set_xlabel(label, fontsize=10)
        ax.spines[['top','right','left']].set_visible(False)
        ax.tick_params(axis='y', length=0)
        ax.set_axisbelow(True)
        ax.grid(axis='x', color='#d9dfe2', linewidth=.65)
        for y in range(1, len(variants), 2):
            ax.axhspan(y - .49, y + .49, color='#f1f4f5', zorder=-2)
        if j == 0:
            ax.set_xlim(-.042, 1.042)
            ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        else:
            ax.axvline(0, color='#596575', lw=.9, zorder=1)
            ax.margins(x=.12)
            ax.xaxis.set_major_locator(MaxNLocator(5))
    axes[0].set_yticks(range(len(variants)), variants, fontsize=10)
    axes[0].set_ylim(len(variants) - .43, -.5)
    fig.suptitle('LBF CRP audit across model seeds', fontsize=17, y=.97)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, 1 - .54 / height),
               ncol=len(handles), frameon=False)
    fig.text(.195, .43 / height, 'Silhouette uses all held-out trajectories with the fixed T1 ego pair. '
             'Return differences use directed member swaps and exclude original partners.', fontsize=9)
    fig.text(.195, .18 / height, '× at the right edge marks an undefined metric for that seed. '
             'Each dot represents one model fit.', fontsize=9)
    fig.savefig(directory / 'audit_metrics.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, help='experiment directory produced by audit_lbf_crp.py evaluate')
    directory = parser.parse_args().directory
    report = json.loads((directory / 'report.json').read_text())
    clouds = json.loads((directory / 'heldout_embeddings.json').read_text())
    runs = {(r['variant'], r['seed']): r for r in report['results']}
    expected = {(v, s) for v in report['variants'] for s in report['seed_list']}
    if not expected or set(runs) != expected or len(runs) != len(report['results']):
        raise ValueError('Expected a complete grid of model variants and seeds')
    plot_clouds(directory, report, runs, clouds)
    plot_metrics(directory, report, runs)


if __name__ == '__main__':
    main()
