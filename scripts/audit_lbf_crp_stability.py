"""Replay online LBF admissions and perturb one held-out trajectory per eight-episode group."""

import argparse
from itertools import combinations
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from fastap_methods.fastap_crp import TeammateCRP, _collate
from scripts.audit_lbf_crp import file_hash
from scripts.audit_lbf_crp_order import compare_partitions


def refreshed_centers(member_vectors, labels):
    """One vote per previously admitted group, in the current encoder's coordinates."""
    if len(member_vectors) != len(labels):
        raise ValueError("Each prior member needs a vector and a cluster label")
    return [torch.stack([v for v, label in zip(member_vectors, labels) if label == cluster]).mean(0)
            for cluster in range(max(labels, default=-1) + 1)]


def replacement_bags(n_maps, size):
    """Each map is removed once and supplied once, always for the same teammate."""
    if size < 1 or n_maps < 2 * size or n_maps % size:
        raise ValueError("Need two or more complete, disjoint bags for the replacement design")
    for start in range(0, n_maps, size):
        baseline = list(range(start, start + size))
        yield baseline, [(idx, (original + size) % n_maps) for idx, original in enumerate(baseline)]


def assign(crp, group, counts, centers):
    crp.counts, crp.centers = counts, centers
    label, _ = crp._assign(group)
    return label


@torch.inference_mode()
def audit_run(run, heldout, reference):
    meta = json.loads((run / 'clustering.json').read_text())
    ckpt = torch.load(run / 'crp.pt', map_location='cpu', weights_only=True)
    cfg = meta['protocol']
    probes = ckpt['crp']['probes']
    if (len(probes) != len(reference['crp']['probes']) or
            any(not torch.equal(a, b) for group, ref in zip(probes, reference['crp']['probes'])
                for a, b in zip(group, ref))):
        raise ValueError("The fitted checkpoint did not use the reference training trajectories")
    names = heldout['members']
    if (cfg['order'] != names or len(ckpt['admissions']) != len(names)
            or heldout['pool_sha256'] != meta['pool_manifest_sha256']):
        raise ValueError("Admission order and held-out teammate IDs differ")
    q = heldout['trials']['fixed_T1']['query']
    seeds = heldout['map_seeds']['query']
    crp = TeammateCRP(**ckpt['model_args'], alpha=cfg['alpha'], decoder=cfg['decoder'])
    held = crp._unpack(q)
    if len(held) != len(names) * len(seeds):
        raise ValueError("Expected one rollout per teammate on every held-out map")
    labels, alternative, groups, stages = [], [], [], []
    for i, (name, snapshot) in enumerate(zip(names, ckpt['admissions'])):
        start, end = snapshot['probe_slice']
        if snapshot['teammate'] != name or end - start != cfg['probe_episodes']:
            raise ValueError("Mismatched admission snapshot or probe count")
        current = probes[start:end]
        crp.model.load_state_dict(snapshot['model'])
        crp.model.eval()
        counts = snapshot['counts_before']
        stored = snapshot['centers_before']
        recorded = assign(crp, current, counts, stored)
        if f'C{recorded+1}' != meta['assignments'][name] or counts != [labels.count(k) for k in range(len(stored))]:
            raise ValueError("Historical MAP or group counts did not reproduce")
        vectors = [crp.model.encode(*_collate(g, 'cpu')).mean(0) for g in groups]
        refreshed = refreshed_centers(vectors, labels)
        one_stage_label = assign(crp, current, counts, refreshed)
        # Fit and replay sampling are independent of cluster labels. These same model
        # snapshots are valid for the full alternative online path with refreshed centers.
        new_centers = refreshed_centers(vectors, alternative)
        new_counts = [alternative.count(k) for k in range(len(new_centers))]
        alternative.append(assign(crp, current, new_counts, new_centers))

        observations = []
        if i:  # T1 has only the forced new-cluster candidate.
            member_query = held[i::len(names)]
            for baseline, swaps in replacement_bags(len(seeds), end - start):
                rows = [member_query[k] for k in baseline]
                results = {}
                for mode, centers in [('stored', stored), ('refreshed', refreshed)]:
                    original = assign(crp, rows, counts, centers)
                    choices = []
                    for slot, incoming in swaps:
                        replaced = rows.copy()
                        replaced[slot] = member_query[incoming]
                        choices.append(assign(crp, replaced, counts, centers))
                    results[mode] = dict(original=original, replaced=choices)
                observations.append(dict(map_seeds=[seeds[k] for k in baseline],
                                         replacement_seeds=[seeds[k] for _, k in swaps], **results))
        stages.append(dict(teammate=name, counts_before=counts, training_label=recorded,
                           training_label_refreshed_same_history=one_stage_label, heldout=observations))
        labels.append(recorded)
        groups.append(current)
    bags = [bag for stage in stages for bag in stage['heldout']]
    summary = dict(eligible_admissions=len(stages) - 1, baseline_bags=len(bags))
    for mode in ['stored', 'refreshed']:
        paired = [(b[mode]['original'], x) for b in bags for x in b[mode]['replaced']]
        summary[mode] = dict(replacements=len(paired), changed=sum(a != b for a, b in paired))
    summary['stored_vs_refreshed_changed_bags'] = sum(b['stored']['original'] != b['refreshed']['original']
                                                       for b in bags)
    return dict(decoder=cfg['decoder'], seed=cfg['seed'], assignments=meta['assignments'],
                refreshed_online_assignments={n: f'C{label+1}' for n, label in zip(names, alternative)},
                summary=summary, stages=stages, checkpoint_sha256=file_hash(run / 'crp.pt'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--reference', type=Path, default=ROOT/'outputs/fastap_lbf_formal_pool_01/crp_own_original_seed7')
    parser.add_argument('--seeds', nargs='+', type=int, default=[7, 17, 27])
    args = parser.parse_args()
    base = args.directory.resolve()
    if (os.getenv('FASTAP_ROOT') != str(ROOT) or ROOT/'outputs' not in base.parents
            or ROOT/'outputs' not in args.reference.resolve().parents):
        parser.error('Source scripts/epymarl-env.sh and choose the experiment directory under outputs')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.set_num_threads(1)
    heldout_path = base / 'heldout.pt'
    held = torch.load(heldout_path, map_location='cpu', weights_only=True)
    if held['training_checkpoint_sha256'] != file_hash(args.reference/'crp.pt'):
        raise ValueError("Query trajectories were checked against a different training reference")
    ref = torch.load(args.reference / 'crp.pt', map_location='cpu', weights_only=True)
    results = [audit_run(base / decoder / f'seed{seed}', held, ref)
               for decoder in ('paper', 'categorical') for seed in args.seeds]
    pairwise = []
    pairs = list(combinations(held['members'], 2))
    for a, b in combinations(results, 2):
        if a['decoder'] == b['decoder']:
            pairwise.append(dict(decoder=a['decoder'], seeds=[a['seed'], b['seed']],
                                 **compare_partitions(a['assignments'], b['assignments'], pairs)))
    report = dict(heldout_sha256=file_hash(heldout_path), reference_sha256=file_hash(args.reference/'crp.pt'),
                  scoring='At each historical post-fit snapshot: preserve prior counts; recompute the arriving '
                          'vector and candidate centers on each eight-episode bag before applying MAP. '
                          'Compare stored centers to re-encoded prior groups with the same historical labels. '
                          'Report a separate end-to-end refreshed-center online partition on training probes.',
                  map_seeds=held['map_seeds']['query'], partner='fixed_T1', snapshots_replayed=len(results)*8,
                  stage_label_indexing='zero-based in stages (0 means C1)', results=results,
                  between_seed_agreement=pairwise)
    path = base / 'admission_stability.json'
    with TemporaryDirectory(dir=base) as staging:
        temp = Path(staging) / path.name
        temp.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        try:
            os.link(temp, path)
        except FileExistsError:
            if temp.read_bytes() != path.read_bytes():
                raise
    for item in results:
        print(item['decoder'], item['seed'], item['summary'])


if __name__ == '__main__':
    main()
