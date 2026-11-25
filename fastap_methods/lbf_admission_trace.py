"""Per-decision evidence for categorical CRP admissions on LBF trajectories."""

from math import fsum, isfinite, log

import torch

from fastap_scenarios.torch_foraging import TorchForaging
from .fastap_crp import _collate
from .lbf_probe_starts import pre_load_moves


ACTION_NAMES = ('NONE', 'NORTH', 'SOUTH', 'WEST', 'EAST', 'LOAD')


def _opportunities(states, slots, actions):
    """Reconstruct what each acting teammate could see and do before its action."""
    states = states.detach().cpu()
    players = states[:, 36:].reshape(-1, 4, 3)
    n_players = (players[:, :, 2] > 0).sum(1)
    if ((n_players < 3) | (n_players > 4) | (slots + 2 >= n_players)).any():
        raise ValueError("Each observed LBF teammate must be present in its board state")
    result = [None] * len(states)
    for size in n_players.unique().tolist():
        rows = (n_players == size).nonzero().flatten()
        board = TorchForaging(len(rows), device='cpu', n_agents=size)
        board.load_snapshot(states[rows, :36].reshape(-1, 6, 6),
                            players[rows, :size, :2], players[rows, :size, 2])
        indices = torch.arange(len(rows))
        roles = slots[rows] + 2
        observations = board.observations()[indices, roles].numpy()
        masks = board.avail_actions()[indices, roles].numpy()
        for row, obs, available in zip(rows.tolist(), observations, masks):
            action = int(actions[row])
            if not available[action]:
                raise ValueError("Recorded teammate chose an unavailable action on the reconstructed LBF board")
            if available[5]:
                result[row] = 'load_available'
            else:
                closer = pre_load_moves(obs, available)
                if closer.any():
                    result[row] = ('preload_approach' if 1 <= action <= 4 and closer[action - 1]
                                   else 'preload_other')
                else:
                    result[row] = 'other'
    return result


def contrast(trace, lhs, rhs):
    """Explain log score(lhs) minus log score(rhs) in episode, timestep and slot order.

    The cumulative curve holds full-trajectory candidate vectors fixed while adding
    each decision's evidence. Its final discrepancy records the trainer's rounding.
    """
    candidates = trace['candidates']
    labels = [c['cluster'] for c in candidates]
    left, right = labels.index(lhs), labels.index(rhs)
    a, b = candidates[left], candidates[right]
    prior = a['log_prior'] - b['log_prior']
    official = a['log_score'] - b['log_score']
    rows = trace['rows']
    terms = [row['logp'][left] - row['logp'][right] for row in rows]
    by_action, by_opportunity = {}, {}
    per_episode = [0.] * trace['episodes']
    running = prior
    cumulative = []
    for row, term in zip(rows, terms):
        for bucket, key in ((by_action, ACTION_NAMES[row['action']]), (by_opportunity, row['opportunity'])):
            item = bucket.setdefault(key, dict(count=0, margin=0.))
            item['count'] += 1
            item['margin'] += term
        per_episode[row['episode']] += term
        running += term
        cumulative.append(running)
    order = sorted(range(len(terms)), key=terms.__getitem__)
    evidence = fsum(terms)
    return dict(lhs=lhs, rhs=rhs, prior_margin=prior, evidence_margin=evidence, total_margin=official,
                reduction_roundoff=official - (prior + evidence), per_episode=per_episode,
                by_action=by_action, by_opportunity=by_opportunity, cumulative_margin=cumulative,
                strongest_for=[i for i in reversed(order) if terms[i] > 0][:5],
                strongest_against=[i for i in order if terms[i] < 0][:5])


@torch.no_grad()
def trace_admission(model, group, counts, centers, alpha, *, minibatch_size=32, expected_scores=None,
                    map_seeds=None, candidate_centers=None):
    """Replay one admission using its model at assignment time and exact pre-admission centers.

    Pass actual mixed candidate centers when available. Otherwise reconstruct them as
    the CRP does from the supplied counts, old centers and current member's trajectories.
    expected_scores is the unnormalized last_assignment_log_scores from the actual CRP.
    """
    if model.decoder != 'categorical' or model.n_actions != len(ACTION_NAMES):
        raise ValueError("Expected the LBF six-action categorical decoder")
    if (not group or len(counts) != len(centers) or any(n <= 0 for n in counts)
            or not isfinite(alpha) or alpha <= 0 or minibatch_size < 1):
        raise ValueError("Invalid CRP counts, centers, prior or probes")
    if map_seeds is not None and len(map_seeds) != len(group):
        raise ValueError("Expected one map seed per episode")
    if model.training:
        raise ValueError("Use the behavior model in evaluation mode, as at CRP assignment")
    device = next(model.parameters()).device
    batches = [_collate(group[start:start + minibatch_size], device)
               for start in range(0, len(group), minibatch_size)]
    if candidate_centers is None:
        vector = sum(model.encode(*batch).sum(0) for batch in batches) / len(group)
        candidate_centers = [(n * center.to(device) + vector) / (n + 1) for n, center in zip(counts, centers)]
        candidate_centers.append(vector)
    if len(candidate_centers) != len(counts) + 1:
        raise ValueError("Expected one candidate for each existing cluster and one new cluster")
    log_weights = torch.tensor([*counts, alpha], device=device, dtype=torch.float64).log()
    scores = log_weights.clone()
    rows = []
    offset = 0
    for batch in batches:
        states, actions, slots, filled = batch
        active = slots & filled.unsqueeze(-1)
        logp = []
        for i, candidate in enumerate(candidate_centers):
            terms = model.action_log_probs(*batch, candidate.to(device).expand(len(states), -1))
            # Match the trainer's float32 episode reduction before the float64 group reduction.
            scores[i] += terms.sum((1, 2)).double().sum()
            logp.append(terms[active])
        episode, step, slot = active.nonzero(as_tuple=True)
        moves = actions[active].cpu()
        contexts = _opportunities(states[episode, step], slot.cpu(), moves)
        values = torch.stack(logp, dim=1).tolist()
        for (e, t, s), move, context, probs in zip(active.nonzero().tolist(), moves.tolist(), contexts, values):
            row = dict(episode=offset + e, step=t, slot=s, role=s + 2, action=move,
                       opportunity=context, logp=probs)
            if map_seeds is not None:
                row['map_seed'] = map_seeds[offset + e]
            rows.append(row)
        offset += len(states)
    if not scores.isfinite().all():
        raise FloatingPointError("Nonfinite admission evidence")
    if expected_scores is not None:
        expected = torch.as_tensor(expected_scores, dtype=torch.float64, device=device)
        torch.testing.assert_close(scores, expected, rtol=0., atol=1e-6)
        if int(scores.argmax()) != int(expected.argmax()):
            raise ValueError("Replayed MAP assignment differs from the recorded admission")
    score_values = scores.tolist()
    labels = [f'C{i + 1}' for i in range(len(scores))]
    ranking = sorted(range(len(scores)), key=score_values.__getitem__, reverse=True)
    candidates = [dict(cluster=label, is_new=i == len(counts), log_score=score,
                       log_evidence=score - weight, log_prior=weight - log(sum(counts) + alpha))
                  for i, (label, score, weight) in enumerate(zip(labels, score_values, log_weights.tolist()))]
    trace = dict(winner=labels[ranking[0]], episodes=len(group), candidates=candidates, rows=rows,
                 new_vs_existing=None, winner_vs_runner_up=None)
    if counts:
        best_existing = next(i for i in ranking if i < len(counts))
        trace['new_vs_existing'] = contrast(trace, labels[-1], labels[best_existing])
        trace['winner_vs_runner_up'] = contrast(trace, labels[ranking[0]], labels[ranking[1]])
    return trace
