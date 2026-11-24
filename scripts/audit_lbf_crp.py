"""Replay the LBF CRP training stream and evaluate controlled behavior-model ablations on new maps."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from fastap_methods.fastap_crp import BehaviorModel, TeammateCRP, _collate
from scripts.lbf_crp_heldout import collect_heldout, evaluate_heldout, trajectory_silhouette
from scripts.plot_lbf_crp import project_embeddings

VARIANTS = {
    'paper': 'Formal implementation of Eqs. 1–4.',
    'mean_nll': 'Average the negative trajectory log likelihoods instead of Eq. 4.',
    'discrete_eq4': 'Normalize Gaussian densities over the six one-hot actions; retain Eq. 4.',
    'discrete_nll': 'Normalized discrete action probabilities and mean trajectory NLL.',
    'scaled_state': 'Divide food, coordinates and levels by their known ranges in both networks.',
    'bounded_sigma': 'Clamp Gaussian standard deviations to a minimum of 0.5; retain Eqs. 1–4.',
    'refresh_centers': 'Re-encode previously admitted groups after each model fit, keeping their labels.',
    'joint_obs': 'Use the concatenated observations of the three present agents as the global state.',
    'joint_obs_discrete_nll': 'Concatenated observations, normalized action probabilities and mean NLL.',
}
DEFAULT_VARIANTS = tuple(VARIANTS)
TRANSFER = 'joint_obs_discrete_nll_transfer'
VARIANTS[TRANSFER] = ('Exploratory cross-episode prediction: use a different episode of the same teammate '
                      'for the behavior context during training, on top of joint_obs_discrete_nll.')
FORMAL = ROOT / 'outputs/fastap_lbf_formal_pool_01/crp_seed7'


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def transform(states, variant):
    if variant.startswith('joint_obs'):
        from scripts.lbf_crp_observations import joint_observations
        return joint_observations(states)
    if variant == 'scaled_state':
        return states / states.new_tensor([9] * 36 + [5, 5, 3] * 4)
    return states


class AuditBehaviorModel(BehaviorModel):
    variant = 'paper'

    def _score_actions(self, mean, scale, target, active):
        if 'discrete' in self.variant:
            # In log N(e_k; mu, diag(sigma^2)), only (mu_k-.5)/sigma_k^2
            # depends on which one-hot vertex e_k is evaluated.
            logits = ((mean - .5) / scale.square()).reshape_as(target)
            scores = (logits.log_softmax(-1) * target).sum(-1)
            return scores.masked_fill(~active, 0).sum((1, 2))
        if self.variant == 'bounded_sigma':
            scale = scale.clamp_min(.5)
        return super()._score_actions(mean, scale, target, active)


class AuditCRP(TeammateCRP):
    model_type = AuditBehaviorModel

    def __init__(self, variant, **kwargs):
        super().__init__(**kwargs)
        self.model.variant = variant
        self.max_weight = []
        self.effective_batch_fraction = []
        self.group_slices = []
        if variant == TRANSFER:
            self.peer_rng = np.random.default_rng(torch.initial_seed())
            self.verified_pairs = 0

    def _fit(self, probes, steps):
        if self.model.variant == TRANSFER:
            from scripts.audit_lbf_crp_transfer import fit_cross_episode
            self.verified_pairs += fit_cross_episode(self, probes, steps, self.peer_rng)
        else:
            super()._fit(probes, steps)

    def _model_loss(self, log_likelihood):
        if 'nll' in self.model.variant:
            weights = torch.full_like(log_likelihood, 1. / len(log_likelihood), dtype=torch.float64)
            loss = -log_likelihood.mean()
        else:
            weights = log_likelihood.detach().double().softmax(0)
            loss = super()._model_loss(log_likelihood)
        self.max_weight.append(weights.max().item())
        self.effective_batch_fraction.append(1. / (len(weights) * weights.square().sum().item()))
        return loss

    @torch.no_grad()
    def _assign(self, group):
        if self.model.variant == 'refresh_centers' and self.centers:
            self.model.eval()
            by_cluster = [[] for _ in self.centers]
            for name, rows in zip(self.assignments, self.group_slices):
                probes = self._probes[rows]
                v = self.model.encode(*_collate(probes, self.device)).mean(0)
                by_cluster[self.assignments[name]].append(v)
            self.centers = [torch.stack(vs).mean(0) for vs in by_cluster]
        start = len(self._probes)
        self.group_slices.append(slice(start, start + len(group)))
        return super()._assign(group)


def fit(source, saved, run, variant, seed):
    protocol = source['protocol']
    torch.manual_seed(seed)
    probes = [(transform(s, variant), a, m, f) for s, a, m, f in saved['crp']['probes']]
    args = dict(saved['model_args'], state_dim=probes[0][0].size(-1))
    crp = AuditCRP(variant, **args, alpha=protocol['alpha'], lr=protocol['lr'],
                   minibatch_size=protocol['minibatch_size'])
    verify = variant == 'paper' and seed == protocol['seed']
    for event, snapshot in zip(source['admissions'], saved['admissions']):
        left, right = event['probe_slice']
        fields = _collate(probes[left:right], 'cpu')
        group = dict(zip(('state', 'teammate_actions', 'teammate_mask', 'filled'), fields))
        label = crp.add_group(event['teammate'], group, model_steps=protocol['crp_model_steps'])
        if verify and (event['cluster'] != f'C{label+1}' or
                       any(not torch.equal(t, snapshot['model'][k]) for k, t in crp.model.state_dict().items())):
            raise RuntimeError('The paper/seed7 control did not reproduce a formal admission')
    assignments = {name: f'C{label+1}' for name, label in crp.assignments.items()}
    return dict(variant=variant, seed=seed, assignments=assignments, model_args=args,
                bitwise_verified_formal_snapshots=verify, model=crp.model.state_dict(),
                max_weight=crp.max_weight, effective_batch_fraction=crp.effective_batch_fraction,
                formal_checkpoint_sha256=file_hash(run/'crp.pt'),
                **({'verified_cross_episode_pairs': crp.verified_pairs} if variant == TRANSFER else {}))


@torch.no_grad()
def first_batch_scores(source, saved):
    """Isolate the time product while holding each full-episode context constant."""
    torch.manual_seed(source['protocol']['seed'])
    model = AuditBehaviorModel(**saved['model_args'])
    lo, hi = source['admissions'][0]['probe_slice']
    states, actions, slots, valid = _collate(saved['crp']['probes'][lo:hi], 'cpu')
    context = model.encode(states, actions, slots, valid)
    horizons = sorted({min(steps, actions.size(1)) for steps in (1, 5, 10, actions.size(1))})
    maximum = {}
    for t in horizons:
        ll = model.trajectory_log_prob(states[:, :t+1], actions[:, :t], slots[:, :t],
                                      valid[:, :t], context)
        maximum[t] = ll.double().softmax(0).max().item()
    gaps = {}
    for variant in ('paper', 'discrete_eq4'):
        model.variant = variant
        scores = model.trajectory_log_prob(states, actions, slots, valid, context).double()
        if len(scores) >= 2:
            first, second = scores.topk(2).values.tolist()
            gaps[variant] = first - second
    return dict(seed=source['protocol']['seed'], teammate=source['admissions'][0]['teammate'],
                trajectories=len(context), frozen_full_episode_context=True,
                top_weight_by_scored_steps=maximum, full_episode_top_two_log_score_gaps=gaps)


@torch.no_grad()
def fitted_probe_scores(model, probes, admissions, assignments, variant):
    model.eval()
    rows = []
    for event in admissions:
        name = event['teammate']
        left, right = event['probe_slice']
        states, acts, slots, filled = _collate(probes[left:right], 'cpu')
        vectors = model.encode(transform(states, variant), acts, slots, filled)
        rows.extend(dict(teammate=name, episode=i, cluster=assignments[name], embedding=v.tolist())
                    for i, v in enumerate(vectors, 1))
    labels = [r['cluster'] for r in rows]
    vectors = np.array([r['embedding'] for r in rows])
    score = trajectory_silhouette(vectors, labels)
    if np.all(vectors == vectors[0]):
        return dict(silhouette_16d=score, silhouette_pca3=score)
    projection = project_embeddings(rows, assignments)
    return dict(silhouette_16d=score,
                silhouette_pca3=trajectory_silhouette([r['xyz'] for r in projection['rows']], labels))


def evaluate_all(run, directory, seed_list, variants):
    saved = torch.load(run/'crp.pt', map_location='cpu', weights_only=True)
    source = json.loads((run/'clustering.json').read_text())
    pool = ROOT / source['pool_manifest']
    crossplay_path = pool.with_name('crossplay.json')
    pool_digest = file_hash(pool)
    if pool_digest != source['pool_manifest_sha256']:
        raise ValueError('The frozen source pool manifest has changed')
    crossplay = json.loads(crossplay_path.read_text())
    held = collect_heldout(pool, saved['crp']['probes'], map_seeds=range(981000, 981032), ego_seed_start=418000)
    results, clouds = [], {}
    digest = file_hash(run/'crp.pt')
    for variant in variants:
        data = dict(held, trials={p: dict(b, state=transform(b['state'], variant))
                                  for p, b in held['trials'].items()})
        for seed in seed_list:
            check = torch.load(directory / f'{variant}-{seed}.pt', map_location='cpu', weights_only=True)
            if (check['variant'], check['seed'], check['formal_checkpoint_sha256']) != (variant, seed, digest):
                raise ValueError('Wrong checkpoint for this evaluation')
            model = AuditBehaviorModel(**check['model_args'])
            model.variant = variant
            model.load_state_dict(check['model'])
            metrics = evaluate_heldout(model, check['assignments'], data, crossplay, include_rows=seed == seed_list[0])
            metrics['fitted_probes'] = fitted_probe_scores(model, saved['crp']['probes'], source['admissions'],
                                                          check['assignments'], variant)
            if seed == seed_list[0]:
                clouds[variant] = {p: m.pop('rows') for p, m in metrics['heldout'].items()}
            results.append(dict(variant=variant, seed=seed, assignments=check['assignments'],
                                **{k: check[k] for k in ('verified_cross_episode_pairs',) if k in check},
                                updates=len(check['max_weight']),
                                updates_above_99pct=sum(w > .99 for w in check['max_weight']),
                                median_effective_batch_fraction=float(np.median(check['effective_batch_fraction'])),
                                bitwise_verified_formal_snapshots=check['bitwise_verified_formal_snapshots'], **metrics))
    report = dict(formal_checkpoint_sha256=digest, pool_manifest_sha256=pool_digest,
                  source_crossplay_sha256=file_hash(crossplay_path), variants={v: VARIANTS[v] for v in variants},
                  seed_list=seed_list, training=source['protocol'], heldout_map_seeds=held['map_seeds'],
                  heldout_ego_seed_start=held['ego_seed_start'], results=results)
    report['first_batch_before_training'] = first_batch_scores(source, saved)
    write_json(directory/'heldout_embeddings.json', clouds)
    write_json(directory/'report.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['fit', 'evaluate'])
    parser.add_argument('directory', type=Path)
    parser.add_argument('--run', type=Path, default=FORMAL)
    parser.add_argument('--seeds', type=int, nargs='+', default=[7, 17, 27])
    parser.add_argument('--variants', nargs='+', choices=list(VARIANTS), default=DEFAULT_VARIANTS)
    a = parser.parse_args()
    run, directory = (ROOT/a.run).resolve(), (ROOT/a.directory).resolve()
    if (os.getenv('FASTAP_ROOT') != str(ROOT) or ROOT/'outputs' not in directory.parents
            or ROOT not in run.parents or directory == run):
        parser.error('Source scripts/epymarl-env.sh and use a separate project output directory')
    if len(set(a.seeds)) != len(a.seeds) or len(set(a.variants)) != len(a.variants):
        parser.error('Use distinct seeds and distinct variants')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.set_num_threads(1)
    directory.mkdir(parents=True, exist_ok=True)
    if a.phase == 'evaluate':
        evaluate_all(run, directory, a.seeds, a.variants)
        return
    source = json.loads((run/'clustering.json').read_text())
    saved = torch.load(run/'crp.pt', map_location='cpu', weights_only=True)
    for seed in a.seeds:
        for variant in a.variants:
            out = directory / f'{variant}-{seed}.pt'
            if out.exists():
                raise FileExistsError(out)
            data = fit(source, saved, run, variant, seed)
            with TemporaryDirectory(dir=directory) as tmp:
                checkpoint = Path(tmp)/'model.pt'
                torch.save(data, checkpoint)
                checkpoint.rename(out)
            print(f'{variant} seed={seed}: {len(set(data["assignments"].values()))} clusters', flush=True)


if __name__ == '__main__':
    main()
