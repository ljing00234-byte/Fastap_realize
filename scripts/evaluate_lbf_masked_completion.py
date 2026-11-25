"""Measure full and masked same-episode action reconstruction on held-out LBF maps."""

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
from scripts.evaluate_lbf_crp_prediction import FIELDS, digest, load
from scripts.evaluate_lbf_rule_prediction import check_fit
from scripts.lbf_rule_positive_control import behavior_events


def evaluation_windows(filled, span):
    """A single centered window for each scored timestep; shift it inward at boundaries."""
    lengths = filled.sum(1, keepdim=True)
    if span < 1 or (lengths == 0).any() or (filled[:, 1:] & ~filled[:, :-1]).any():
        raise ValueError("Require nonempty valid timestep prefixes and a positive span")
    # On short episodes keep one observed action available, when possible.
    widths = (lengths - 1).clamp(min=1, max=span)
    time = torch.arange(filled.size(1))
    starts = torch.minimum((time - widths // 2).clamp_min(0), lengths - widths)
    for start in range(filled.size(1)):
        scored = (starts == start) & filled
        if scored.any():
            hidden = (time >= start) & (time < start + widths) & filled
            yield scored, hidden & scored.any(1, keepdim=True)


def approach_choices(filled, sidecar):
    """Actions that approach visible food at genuine pre-load choice points, from state alone."""
    choices = torch.zeros((*filled.shape, 6), dtype=torch.bool)
    for row, step in filled.nonzero().tolist():
        obs = sidecar['obs'][row, step].numpy()
        available = sidecar['available'][row, step].numpy()
        if behavior_events(obs, available, 0)[1][1]:
            for action in np.flatnonzero(available[1:5]) + 1:
                choices[row, step, action] = bool(behavior_events(obs, available, action)[0][1])
    return choices


@torch.inference_mode()
def evaluate_episodes(model, batch, choices, span):
    """Return one loss per active decision in map-major order."""
    model.eval()
    full, completion, binary = (torch.zeros_like(batch['filled'], dtype=torch.double) for _ in range(3))
    for start in range(0, full.size(0), 32):
        rows = slice(start, start + 32)
        states, actions, slots, filled = (batch[k][rows] for k in FIELDS)
        active = slots & filled.unsqueeze(-1)
        if not torch.equal(active.sum(-1), filled.long()):
            raise ValueError("Expected exactly one present teammate per valid LBF step")
        target = actions.masked_fill(~active, 0).sum(-1)
        action_slot = active.long().argmax(-1)[..., None, None].expand(-1, -1, 1, model.n_actions)
        context = model.encode(states, actions, slots, filled)
        full[rows] = -model.action_log_probs(states, actions, slots, filled, context).sum(-1).double()
        for scored, hidden in evaluation_windows(filled, span):
            context = model.encode(states, actions, slots, filled, action_mask=hidden.unsqueeze(-1) & active)
            logits = model.action_logits(states, context).gather(2, action_slot).squeeze(2)[scored]
            log_probs = logits.double().log_softmax(-1)
            taken = target[scored].unsqueeze(-1)
            completion[rows][scored] = -log_probs.gather(1, taken).squeeze(1)
            closer = choices[rows][scored]
            # Binary event on the same unrestricted six-action probability distribution.
            consistent = closer == closer.gather(1, taken)
            binary[rows][scored] = -log_probs.masked_fill(~consistent, -torch.inf).logsumexp(-1)
    return dict(full=full, completion=completion, binary=binary)


def compare(before, after, selected, n_maps, bootstrap):
    """Every eligible episode contributes its mean; resample entire maps with all teammates."""
    counts = selected.sum(1)
    active = counts > 0
    if not active.any():
        return dict(decisions=0, episodes=0, maps=0)
    original = before.masked_fill(~selected, 0).sum(1) / counts.clamp_min(1)
    auxiliary = after.masked_fill(~selected, 0).sum(1) / counts.clamp_min(1)
    gains = (original - auxiliary).reshape(n_maps, -1).sum(1).numpy()
    map_counts = active.reshape(n_maps, -1).sum(1).numpy()
    draw_counts = map_counts[bootstrap].sum(1)
    draws = gains[bootstrap].sum(1)[draw_counts > 0] / draw_counts[draw_counts > 0]
    return dict(decisions=int(counts.sum()), episodes=int(active.sum()), maps=int((map_counts > 0).sum()),
                old_mean_nll=float(original[active].mean()), new_mean_nll=float(auxiliary[active].mean()),
                gain_nats=float(gains.sum()/map_counts.sum()), paired_95_ci=np.quantile(draws, [.025, .975]).tolist())


def evaluate(reference, updated, seeds, span):
    truth = json.loads((reference/'truth.json').read_text())
    hashes = {name: digest(reference/name) for name in ('train.pt', 'query.pt', 'observations.pt')}
    if any(truth['files_sha256'][name] != fingerprint for name, fingerprint in hashes.items()):
        raise ValueError("Rule benchmark trajectories or observations changed")
    query = load(reference/'query.pt')
    train = load(reference/'train.pt')
    batch = query['batch']
    filled = batch['filled']
    n_maps = len(query['map_seeds'])
    rules = np.tile([truth['labels'][name] for name in query['members']], n_maps)
    if (len(rules) != len(filled) or len(set(query['map_seeds'])) != n_maps
            or train['members'] != query['members'] or set(train['map_seeds']) & set(query['map_seeds'])):
        raise ValueError("Expected matched members on held-out query maps")
    choices = approach_choices(filled, load(reference/'observations.pt')['query'])
    approaching = choices.any(-1)
    masks = dict(all=filled, approach_both=approaching & torch.tensor(np.isin(rules, ['BASELINE','SOCIAL1']))[:, None],
                 approach_baseline=approaching & torch.tensor(rules == 'BASELINE')[:, None],
                 approach_social1=approaching & torch.tensor(rules == 'SOCIAL1')[:, None])
    bootstrap = np.random.default_rng(87543).integers(n_maps, size=(10000, n_maps))
    results = []
    for seed in seeds:
        evaluations, fingerprints, configs = [], [], []
        for directory in (reference, updated):
            ckpt, config = check_fit(directory, seed, 'categorical', train)
            weight = ckpt['crp'].get('masked_action_weight', 0.)
            if (weight != config['protocol'].get('masked_action_weight', 0.) or
                    (directory == reference and weight != 0.) or (directory == updated and not weight > 0.)):
                raise ValueError("Expected an unmasked reference and an auxiliary-trained comparison")
            model = BehaviorModel(**ckpt['model_args'], decoder='categorical')
            model.load_state_dict(ckpt['crp']['model'])
            evaluations.append(evaluate_episodes(model, batch, choices, span))
            fingerprints.append(digest(directory/'categorical'/f'seed{seed}'/'crp.pt'))
            configs.append(config['protocol'])
        for key in ('lr', 'order', 'alpha', 'minibatch_size', 'crp_model_steps', 'probe_episodes'):
            if configs[0][key] != configs[1][key]:
                raise ValueError(f"Fitted controls differ on {key}")
        old, new = evaluations
        metrics = {}
        for name, mask in masks.items():
            metrics[name] = {kind: compare(old[kind], new[kind], mask, n_maps, bootstrap)
                             for kind in (('full', 'completion') if name == 'all' else ('completion', 'binary'))}
        results.append(dict(seed=seed, old_checkpoint_sha256=fingerprints[0], new_checkpoint_sha256=fingerprints[1],
                            training_protocol=configs[1], metrics=metrics))
    return dict(results=results, query_map_seeds=query['map_seeds'], members=query['members'],
                dataset_sha256=hashes, span=span,
                masking="For each valid action mask a centered same-episode span of real actions. Shift the window "
                        "inside episode boundaries; keep one visible step when possible. Score the center target once. "
                        "Full states including later transitions and the terminal state remain visible to the encoder.",
                approach="Before LOAD is legal, at least one visible food, and approaching and nonapproaching "
                         "legal cardinal moves exist. Binary loss predicts whether the observed action moves "
                         "toward nearest visible food, from the unmodified six-way probabilities.",
                weighting="Equal eligible episodes after within-episode mean NLL. Positive old minus new favors the "
                          "auxiliary task. Paired bootstrap of 10,000 whole query-map resamples, seed 87543.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('updated', type=Path)
    parser.add_argument('--seeds', type=int, nargs='+', default=[7, 17, 27])
    parser.add_argument('--span', type=int, default=3)
    args = parser.parse_args()
    original, updated = args.reference.resolve(), args.updated.resolve()
    if (os.getenv('FASTAP_ROOT') != str(ROOT) or original == updated or
            any(ROOT/'outputs' not in path.parents for path in (original, updated))):
        parser.error("Source scripts/epymarl-env.sh and select distinct project output directories")
    if args.span < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Use positive mask length and distinct training seeds")
    torch.set_num_threads(1)
    target = updated/'same_episode_completion.json'
    result = evaluate(original, updated, args.seeds, args.span)
    with TemporaryDirectory(dir=updated) as staging:
        temp = Path(staging)/target.name
        temp.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        os.link(temp, target)
    print(target)
