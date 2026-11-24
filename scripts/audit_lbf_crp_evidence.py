"""Decompose saved LBF CRP admission scores and compare them with measured teammate swaps."""

import argparse
import json
import math
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.nn import functional as F

from fastap_methods import fastap_crp
from fastap_methods.fastap_crp import BehaviorModel, _collate
from scripts.audit_lbf_crp import FORMAL, file_hash


def episode_scores(model, sample, center):
    """Split each original trajectory score into log action preference and log density mass."""
    states, actions, slots, filled = sample
    v = center.expand(len(states), -1)
    original = model.trajectory_log_prob(*sample, v).double()
    history = torch.cat((states[:, :-1], v[:, None].expand(-1, actions.size(1), -1)), dim=-1)
    decoded, _ = model.gru(history)
    mean, logits = model.gaussian(decoded).chunk(2, dim=-1)
    scale = (F.softplus(logits) + 1e-4).double().reshape(*actions.shape, model.n_actions)
    mean = mean.double().reshape_as(scale)
    active = slots & filled.unsqueeze(-1)

    # On each active slot, log N(e_k) = log N(0) + (mu_k - 1/2) / sigma_k^2.
    vertex_bias = (mean - .5) / scale.square()
    log_z = (-.5 * (mean / scale).square() - scale.log() - .5 * math.log(2 * math.pi)).sum(-1)
    log_z += vertex_bias.logsumexp(-1)
    observed = actions.masked_fill(~active, 0)
    log_relative = vertex_bias.log_softmax(-1).gather(-1, observed.unsqueeze(-1)).squeeze(-1)
    preference = log_relative.masked_fill(~active, 0).sum((1, 2))
    mass = log_z.masked_fill(~active, 0).sum((1, 2))
    # Official scores reduce float32 per-feature log densities. The algebraic split uses float64.
    torch.testing.assert_close(preference + mass, original, atol=1e-3, rtol=1e-6)
    return original.numpy(), preference.numpy(), mass.numpy()


@torch.no_grad()
def audit(run):
    source = json.loads((run / 'clustering.json').read_bytes())
    pool_path = ROOT / source['pool_manifest']
    if file_hash(pool_path) != source['pool_manifest_sha256']:
        raise ValueError('The source teammate pool changed')
    path = pool_path.with_name('crossplay.json')
    crossplay = json.loads(path.read_bytes())
    ids, protocol = crossplay['ids'], source['protocol']
    if (ids != protocol['order'] or crossplay['rows'] != 'original trained partners in slots 0/1'
            or crossplay['columns'] != 'evaluated frozen teammate in slot 2'):
        raise ValueError('Unexpected member IDs or cross-play direction')
    matrix = np.array([[cell['mean_return'] for cell in row] for row in crossplay['crossplay']])
    if matrix.shape != (len(ids), len(ids)) or not np.isfinite(matrix).all():
        raise ValueError('Invalid source cross-play matrix')

    saved = torch.load(run / 'crp.pt', map_location='cpu', weights_only=True)
    model = BehaviorModel(**saved['model_args']).eval()
    alpha = protocol['alpha']
    clusters, result = {}, []
    if len(saved['admissions']) != len(source['admissions']) or len(ids) != len(source['admissions']):
        raise ValueError('Incomplete admission records')
    for i, (event, snap) in enumerate(zip(source['admissions'], saved['admissions'])):
        name = ids[i]
        if name != event['teammate'] or name != snap['teammate'] or event['probe_slice'] != snap['probe_slice']:
            raise ValueError('Checkpoint and report refer to different admissions')
        model.load_state_dict(snap['model'])
        counts = snap['counts_before']
        if counts != [len(members) for members in clusters.values()]:
            raise ValueError('Prior counts differ from historical assignments')
        start, stop = snap['probe_slice']
        group = saved['crp']['probes'][start:stop]
        sample = _collate(group, 'cpu')
        vector = model.encode(*sample).mean(0)
        centers = [(n * v + vector) / (n + 1) for n, v in zip(counts, snap['centers_before'])]
        centers.append(vector)
        log_prior = np.log([*counts, alpha]) - math.log(sum(counts) + alpha)
        raw, preference, mass = (np.stack(t) for t in zip(*(episode_scores(model, sample, c) for c in centers)))
        full = raw.sum(1) + log_prior
        labels = [f'C{k+1}' for k in range(len(centers))]
        actual = int(full.argmax())
        if (labels[actual] != event['cluster'] or len(event['candidates']) != len(centers)
                or [c['cluster'] for c in event['candidates']] != labels):
            raise ValueError('Replayed candidate set or MAP differs from the formal record')
        recorded = [c['log_joint'] for c in event['candidates']]
        np.testing.assert_allclose(full, recorded, atol=1e-9, rtol=0.)

        candidates = []
        for k, label in enumerate(labels):
            prior_members = clusters.get(label, [])
            swaps = {}
            for member in prior_members:
                j = ids.index(member)
                swaps[member] = dict(arriving_egos_with_old_teammate=matrix[i, j].item(),
                                     old_egos_with_arriving_teammate=matrix[j, i].item(),
                                     bidirectional_mean=float((matrix[i, j] + matrix[j, i]) / 2))
            delta = raw[actual] - raw[k]
            candidates.append(dict(
                cluster=label, existing_members=list(prior_members), crossplay_against_members=swaps,
                crossplay_mean=float(np.mean([v['bidirectional_mean'] for v in swaps.values()])) if swaps else None,
                log_prior=log_prior[k].item(), log_joint=full[k].item(),
                margin_winner_minus_candidate=dict(
                    total=float(full[actual] - full[k]), log_prior=float(log_prior[actual] - log_prior[k]),
                    observed_action_preference=float((preference[actual] - preference[k]).sum()),
                    log_density_mass=float((mass[actual] - mass[k]).sum()), per_episode_log_evidence=delta.tolist(),
                    episodes_preferring_winner=int((delta > 0).sum()), episodes_preferring_candidate=int((delta < 0).sum()))))
        loo = (full[:, None] - raw).argmax(axis=0)
        result.append(dict(teammate=name, selected=labels[actual], episode_count=len(group),
                           max_abs_decomposition_roundoff=float(np.abs(raw - preference - mass).max()),
                           winner_if_omitting_each_episode=[labels[k] for k in loo],
                           candidates=candidates))
        clusters.setdefault(labels[actual], []).append(name)
    return dict(formal_run=str(run.relative_to(ROOT)), checkpoint_sha256=file_hash(run/'crp.pt'),
                clustering_sha256=file_hash(run/'clustering.json'), pool_sha256=source['pool_manifest_sha256'],
                crossplay_sha256=file_hash(path), maps_per_pair=crossplay['n_maps'],
                decomposition='For each candidate and recorded state, Z is the sum of its Gaussian densities over '
                              'all six nominal one-hot actions per active slot. Log density of the action taken = '
                              'log of its share of Z + log Z. The MAP uses the complete original log density.',
                omission='Drop one episode from the accumulated log evidence only. Keep its admission-time fitted '
                         'model, all candidate centers, counts and the other seven likelihoods fixed.',
                admissions=result)


class _ObservedCRP(fastap_crp.TeammateCRP):
    """Observe the production optimizer's actual draws without consuming any random numbers."""

    def _fit(self, probes, steps):
        ids = {id(probe): i for i, probe in enumerate(probes)}
        self.updates = []

        def record_batch(batch, device):
            self.drawn = [ids[id(probe)] for probe in batch]
            return _collate(batch, device)

        with patch.object(fastap_crp, '_collate', record_batch):
            super()._fit(probes, steps)

    def _model_loss(self, log_likelihood):
        weights = log_likelihood.detach().double().softmax(0)
        winner = weights.argmax().item()
        self.updates.append(dict(sampled_probe_ids=self.drawn, weights=weights.tolist(),
                                 top_probe_id=self.drawn[winner], top_weight=weights[winner].item()))
        return super()._model_loss(log_likelihood)


def exposure(rows, identities):
    n = len(identities)
    draws = np.zeros(n, dtype=int)
    total = np.zeros(n)
    wins = np.zeros(n, dtype=int)
    for row in rows:
        ids = row['sampled_probe_ids']
        if len(set(ids)) != len(ids):
            raise ValueError('Unexpected repeated draw within a minibatch')
        draws[ids] += 1
        total[ids] += row['weights']
        wins[row['top_probe_id']] += 1
    np.testing.assert_allclose(total.sum(), len(rows), rtol=1e-13)
    members = np.array([item['teammate'] for item in identities])
    by_member = {}
    for name in dict.fromkeys(members.tolist()):
        mask = members == name
        by_member[name] = dict(draws=int(draws[mask].sum()), wins=int(wins[mask].sum()),
                               summed_weight=float(total[mask].sum()),
                               weight_share=float(total[mask].sum()/len(rows)))
    probes = [dict(**item, draws=int(draws[j]), wins=int(wins[j]), summed_weight=float(total[j]))
              for j, item in enumerate(identities)]
    return dict(updates=len(rows), updates_above_99pct=sum(r['top_weight'] > .99 for r in rows),
                distinct_sampled_probes=int((draws > 0).sum()), distinct_winning_probes=int((wins > 0).sum()),
                highest_weight_probe=probes[int(total.argmax())],
                by_member=by_member, by_probe=probes)


def training_coverage(run):
    source = json.loads((run/'clustering.json').read_bytes())
    cfg = source['protocol']
    saved = torch.load(run/'crp.pt', map_location='cpu', weights_only=True)
    torch.manual_seed(cfg['seed'])
    crp = _ObservedCRP(**saved['model_args'], alpha=cfg['alpha'], lr=cfg['lr'],
                       minibatch_size=cfg['minibatch_size'])
    stages, identities, history = [], [], []
    if len(saved['admissions']) != len(source['admissions']):
        raise ValueError('Incomplete admission snapshots')
    for event, snap in zip(source['admissions'], saved['admissions']):
        name = event['teammate']
        start, stop = snap['probe_slice']
        if snap['teammate'] != name or event['probe_slice'] != [start, stop] or start != len(identities):
            raise ValueError('Inconsistent episode provenance')
        probes = saved['crp']['probes'][start:stop]
        identities.extend(dict(probe_id=i, teammate=name, episode=i-start+1, length=len(probe[1]))
                          for i, probe in enumerate(probes, start))
        batch = dict(zip(('state', 'teammate_actions', 'teammate_mask', 'filled'), _collate(probes, 'cpu')))
        label = crp.add_group(name, batch, model_steps=cfg['crp_model_steps'])
        if event['cluster'] != f'C{label+1}' or len(crp.updates) != cfg['crp_model_steps']:
            raise ValueError('Formal MAP or optimizer step count mismatch')
        if any(not torch.equal(v, snap['model'][k]) for k, v in crp.model.state_dict().items()):
            raise ValueError(f'Model tensors differ from the original snapshot at {name}')
        stages.append(dict(arriving_member=name, exposure=exposure(crp.updates, identities), steps=crp.updates))
        history.extend(crp.updates)
    return dict(formal_run=str(run.relative_to(ROOT)), checkpoint_sha256=file_hash(run/'crp.pt'),
                seed=cfg['seed'], bitwise_verified_snapshots=len(stages), probe_identity=identities,
                measure='For each actual draw, softmax of the original episode log likelihood coefficients in Eq. 4 '
                        'was computed in float64 from detached float32 scores. The loss, optimizer and draws are '
                        'unchanged. Each update contributes one unit of total weight. These are loss weights, '
                        'not norms of parameter gradients. Probe IDs are zero based, episode numbers one based.',
                overall=exposure(history, identities), stages=stages)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', nargs='?', type=Path)
    parser.add_argument('--training-coverage', action='store_true',
                        help='replay the original 512 training updates and trace trajectory identities')
    args = parser.parse_args()
    default_name = 'training_coverage.json' if args.training_coverage else 'admission_evidence.json'
    output = (ROOT / (args.output or FORMAL.parent/'crp_diagnostics/crossplay_admission_audit'/default_name)).resolve()
    if os.getenv('FASTAP_ROOT') != str(ROOT) or ROOT / 'outputs' not in output.parents:
        parser.error('Source scripts/epymarl-env.sh and choose an output file under project outputs/')
    if output.exists():
        parser.error('Choose a new output file')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.set_num_threads(1)
    report = training_coverage(FORMAL) if args.training_coverage else audit(FORMAL)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as out:
        json.dump(report, out, indent=2, allow_nan=False)
        out.write('\n')
