"""Measure whether the frozen LBF admission models distinguish action choices on the same history."""

import argparse
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.nn import functional as F

from fastap_agents import load_frozen_agent
from fastap_agents.batched import select_actions
from fastap_methods.fastap_crp import BehaviorModel, _collate
from fastap_scenarios.fastap_rollout import verified_pool_members
from fastap_scenarios.torch_foraging import TorchForaging
from scripts.audit_lbf_crp import FORMAL, file_hash
from scripts.lbf_crp_heldout import collect_heldout


def frozen_candidates(saved, event, cluster_names):
    """Reconstruct the posterior candidate centers from the original eight training probes."""
    snap = next(s for s in saved['admissions'] if s['teammate'] == event['teammate'])
    if snap['probe_slice'] != event['probe_slice']:
        raise ValueError('Different admission probes in checkpoint and log')
    start, stop = snap['probe_slice']
    if stop - start != 8:
        raise ValueError('Expected eight formal probes')
    sample = _collate(saved['crp']['probes'][start:stop], 'cpu')
    model = BehaviorModel(**saved['model_args']).eval()
    model.load_state_dict(snap['model'])
    vector = model.encode(*sample).mean(0)
    all_centers = [(n * c + vector) / (n + 1) for n, c in zip(snap['counts_before'], snap['centers_before'])]
    all_centers.append(vector)
    centers, priors, formal_delta = [], [], []
    for label in cluster_names:
        index = next(i for i, c in enumerate(event['candidates']) if c['cluster'] == label)
        record = event['candidates'][index]
        center = all_centers[index]
        prior = record['log_prior']
        score = model.trajectory_log_prob(*sample, center.expand(stop - start, -1)).double().sum().item() + prior
        if not math.isclose(score, record['log_joint'], abs_tol=1e-9, rel_tol=0.):
            raise ValueError(f'Original admission score mismatch for {label}')
        centers.append(center)
        priors.append(prior)
        formal_delta.append(score)
    return model, centers, priors, formal_delta[0] - formal_delta[1]


def teacher_forced_actions(batch, actors):
    """Each policy sees the real role-2 local observation and its *observed* previous action."""
    states, filled = batch['state'][:, :-1], batch['filled']
    real = states[filled]
    players = real[:, 36:].reshape(-1, 4, 3)
    if not torch.all(players[:, 3] == real.new_tensor([-1, -1, 0])):
        raise ValueError('Expected exactly three real players')
    world = TorchForaging(len(real), 'cpu', n_agents=3)
    world.load_snapshot(real[:, :36].reshape(-1, 6, 6), players[:, :3, :2], players[:, :3, 2])
    obs = states.new_zeros((*filled.shape, 3, world.obs_shape))
    legal = torch.zeros((*filled.shape, 3, world.n_actions), dtype=torch.bool)
    obs[filled], legal[filled] = world.observations(), world.avail_actions()
    legal[..., 0] |= ~filled.unsqueeze(-1)
    actual = batch['teammate_actions'][:, :, 0]
    slots = batch['teammate_mask']
    if not torch.equal(slots[:, :, 0], filled) or slots[:, :, 1].any():
        raise ValueError('Unexpected active teammate slots')
    # The other two network rows have independent recurrent states and are never scored.
    previous = states.new_zeros((*filled.shape, 3, world.n_actions))
    previous[:, 1:, 2] = F.one_hot(actual[:, :-1].clamp_min(0), world.n_actions).float()
    results, validation = {}, {}
    for source_index, (name, actor) in enumerate(actors.items()):
        hidden, actions = None, []
        for t in range(filled.size(1)):
            chosen, hidden = select_actions(actor.model, actor.spec, obs[:, t], legal[:, t], hidden, previous[:, t])
            actions.append(chosen[:, 2])
        actions = torch.stack(actions, dim=1)
        if not legal[:, :, 2].gather(-1, actions.unsqueeze(-1))[filled].all():
            raise ValueError('Predicted an illegal action')
        source_rows = slice(source_index, None, len(actors))
        active = filled[source_rows]
        mismatches = (actions[source_rows] != actual[source_rows]) & active
        validation[name] = dict(valid_steps=int(active.sum()), mismatches=int(mismatches.sum()))
        if mismatches.any():
            raise ValueError(f"Original actor replay failed for {name}: {validation[name]}")
        results[name] = actions
    return results, validation


def evaluate_blocks(batch, predicted, names, seeds, ego, model, centers, priors, policies, candidates):
    records, action_dist, map_dist = [], [], []
    for i, source in enumerate(names):
        for m in range(0, len(seeds), 8):
            rows = torch.arange(m, m + 8) * len(names) + i
            states, slots, filled = (batch[key][rows] for key in ('state', 'teammate_mask', 'filled'))
            embeddings, margins = [], []
            for policy in policies:
                actions = batch['teammate_actions'][rows].clone()
                actions[:, :, 0] = torch.where(filled, predicted[policy][rows], actions[:, :, 0])
                sample = states, actions, slots, filled
                by_candidate = [model.trajectory_log_prob(*sample, c.expand(8, -1)).double().sum().item() + p
                                for c, p in zip(centers, priors)]
                margins.append(by_candidate[0] - by_candidate[1])
                v = model.encode(*sample)
                embeddings.append(v)
                # Fixed source, ego and action policy on two different held-out maps.
                map_dist.extend((v[::2] - v[1::2]).norm(dim=-1).tolist())
            action_dist.extend((embeddings[0] - embeddings[1]).norm(dim=-1).tolist())
            changed = (predicted[policies[0]][rows] != predicted[policies[1]][rows]) & filled
            records.append(dict(ego=ego, source=source, map_seeds=seeds[m:m + 8],
                                active_steps=int(filled.sum()), different_actions=int(changed.sum()),
                                left_minus_right_log_joint=dict(zip(policies, margins)),
                                winner={p: candidates[int(delta < 0)] for p, delta in zip(policies, margins)}))
    return records, action_dist, map_dist


def summarize(rows, policies, candidates):
    left, right = candidates
    table = {a: {b: 0 for b in candidates} for a in candidates}
    for row in rows:
        a, b = (row['winner'][p] for p in policies)
        table[a][b] += 1
    return dict(groups=len(rows), source_steps=sum(row['active_steps'] for row in rows),
                different_actions=sum(row['different_actions'] for row in rows),
                groups_with_different_actions=sum(row['different_actions'] > 0 for row in rows),
                winner_changes=table[left][right] + table[right][left], winner_transition=table)


@torch.no_grad()
def run(output):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    source = json.loads((FORMAL / 'clustering.json').read_text())
    saved = torch.load(FORMAL / 'crp.pt', map_location='cpu', weights_only=True)
    pool = ROOT / source['pool_manifest']
    if file_hash(pool) != source['pool_manifest_sha256']:
        raise ValueError('Formal pool has changed')
    members = verified_pool_members(pool, 'lbf')
    if {name: file_hash(path / 'agent.th') for name, path in members} != source['member_sha256']:
        raise ValueError('Formal teammate weights have changed')
    held = collect_heldout(pool, saved['crp']['probes'], map_seeds=range(991000, 991032), ego_seed_start=491000)
    actors = {name: load_frozen_agent(path, device='cpu') for name, path in members}
    names = held['members']
    if names != list(actors):
        raise ValueError('Unexpected source order')
    predicted, validation = {}, {}
    for ego, batch in held['trials'].items():
        predicted[ego], validation[ego] = teacher_forced_actions(batch, actors)
    comparisons = {}
    for stage, policies, candidates in [('T3', ['T1', 'T2'], ['C1', 'C2']), ('T8', ['T7', 'T8'], ['C4', 'C5'])]:
        event = next(e for e in source['admissions'] if e['teammate'] == stage)
        model, centers, priors, formal_margin = frozen_candidates(saved, event, candidates)
        blocks, action_dist, map_dist = [], [], []
        for ego, batch in held['trials'].items():
            records, actions, maps = evaluate_blocks(batch, predicted[ego], names, held['map_seeds'], ego, model,
                                                      centers, priors, policies, candidates)
            blocks.extend(records)
            action_dist.extend(actions)
            map_dist.extend(maps)
        comparisons[stage] = dict(action_policies=policies, candidates=candidates, log_priors=priors,
                                  formal_eight_probe_left_minus_right_log_joint=formal_margin,
                                  summary=summarize(blocks, policies, candidates),
                                  by_ego={ego: summarize([r for r in blocks if r['ego'] == ego], policies, candidates)
                                          for ego in held['trials']},
                                  encoder_16d_euclidean=dict(
                                      median_same_history_action_policy_swap=float(np.median(action_dist)),
                                      median_same_policy_different_map=float(np.median(map_dist)),
                                      same_history_pairs=len(action_dist), different_map_pairs=len(map_dist)),
                                  blocks=blocks)
    result = dict(checkpoint_sha256=file_hash(FORMAL / 'crp.pt'), pool_manifest_sha256=file_hash(pool),
                  map_seeds=held['map_seeds'], ego_seed_start=held['ego_seed_start'], sources=names,
                  protocol=('Per map, run each real source once with each ego. Teacher-force each queried frozen '
                            'policy on the same observations and the true source role-2 previous actions. Score the '
                            'two replacement action sequences on that frozen state history using the two original '
                            'formal candidate vectors. Each eight-map, single-source, single-ego group receives one '
                            'formal log prior per candidate. Transition table rows are the first queried policy\'s '
                            'winner and columns are the second policy\'s winner. Groups reuse the same 32 maps. '
                            'Encoder distances are a separate readout, with adjacent map pairs for the comparator.'),
                  source_action_replay=validation, comparisons=comparisons)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: v['summary'] for k, v in comparisons.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, nargs='?',
                        default=FORMAL.parent/'crp_diagnostics/crossplay_admission_audit/shared_histories.json')
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    local_cache = all(os.getenv(key) and ROOT in Path(os.environ[key]).resolve().parents
                      for key in ('TMPDIR', 'XDG_CACHE_HOME'))
    if (os.getenv('FASTAP_ROOT') != str(ROOT) or not local_cache
            or ROOT/'outputs' not in output.parents or output.exists()):
        parser.error('Source scripts/epymarl-env.sh and choose a new output file under project outputs/')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    output.parent.mkdir(exist_ok=True, parents=True)
    run(output)
