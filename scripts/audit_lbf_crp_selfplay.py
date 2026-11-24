"""Train the LBF behavior encoder with each frozen teammate's own two QMIX partners."""

import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from fastap_agents import load_frozen_agent
from fastap_methods.fastap_training import _batch, _episode
from fastap_scenarios.fastap_rollout import FastapRolloutEnv, load_fastap_groups, verified_pool_members
from scripts.audit_lbf_crp import AuditCRP, FORMAL, file_hash
from scripts.lbf_crp_heldout import _FrozenEgoPair, collect_heldout, evaluate_heldout


def selfplay_episode(env, index, actors, rng):
    ep = _episode(env, index, _FrozenEgoPair(actors[index], rng), rng)
    return {key: ep[key] for key in ('state', 'teammate_actions', 'teammate_mask', 'filled', 'reward')}


def run(output):
    source = json.loads((FORMAL / 'clustering.json').read_bytes())
    config = source['protocol']
    pool = ROOT / source['pool_manifest']
    if file_hash(pool) != source['pool_manifest_sha256']:
        raise ValueError('Source teammate pool changed')
    groups = [g for g in load_fastap_groups('lbf', pool) if len(g.policies) == 1]
    members = verified_pool_members(pool, 'lbf')
    names = [name for name, _ in members]
    if names != config['order'] or names != [g.name for g in groups]:
        raise ValueError('Role ordering differs between the probe and frozen ego policies')
    actors = [load_frozen_agent(path, device='cpu') for _, path in members]
    saved = torch.load(FORMAL / 'crp.pt', map_location='cpu', weights_only=True)
    probes = saved['crp']['probes']
    training = []
    rng = np.random.default_rng(config['probe_ego_seed'])
    with closing(FastapRolloutEnv('lbf', groups, seed=config['probe_env_seed'])) as env:
        for i, event in enumerate(source['admissions']):
            start, stop = event['probe_slice']
            episodes = [selfplay_episode(env, i, actors, rng) for _ in range(start, stop)]
            if any(not torch.equal(e['state'][0], probe[0][0]) for e, probe in zip(episodes, probes[start:stop])):
                raise ValueError(f'Training map mismatch for {names[i]}')
            training.append(_batch(episodes))

    held = collect_heldout(pool, probes, map_seeds=range(981000, 981032), ego_seed_start=418000)
    del held['trials']['random_legal']
    fixed = held['trials']['fixed_T1_ego']
    own = []
    for j, seed in enumerate(held['map_seeds']):
        for i in range(len(members)):
            with closing(FastapRolloutEnv('lbf', groups, seed=seed)) as env:
                ep = selfplay_episode(env, i, actors, np.random.default_rng(held['ego_seed_start'] + j))
            if not torch.equal(ep['state'][0], fixed['state'][len(own), 0]):
                raise ValueError('Original partners and fixed T1 probes started on different held-out maps')
            own.append(ep)
    held['trials']['own_original_ego'] = _batch(own)
    crossplay_path = pool.with_name('crossplay.json')
    crossplay = json.loads(crossplay_path.read_bytes())
    results = []
    for seed in (7, 17, 27):
        torch.manual_seed(seed)
        crp = AuditCRP('paper', **saved['model_args'], alpha=config['alpha'], lr=config['lr'],
                       minibatch_size=config['minibatch_size'])
        for name, batch in zip(names, training):
            crp.add_group(name, batch, model_steps=config['crp_model_steps'])
        labels = {name: f'C{k+1}' for name, k in crp.assignments.items()}
        results.append(dict(seed=seed, assignments=labels, updates=len(crp.max_weight),
                            updates_above_99pct=sum(w > .99 for w in crp.max_weight),
                            **evaluate_heldout(crp.model, labels, held, crossplay)))
    returns = torch.cat([b['reward'].sum(1) for b in training]).double().numpy()
    report = dict(training_ego='For Tj use the two original greedy QMIX partners of Tj throughout the episode.',
                  method='Formal behavior encoder, decoder, Eq. 4 and online CRP assignment.',
                  interpretation='On own-partner trials all three policies vary across teams. On fixed-T1 trials '
                                 'only the frozen role 2 teammate varies.',
                  matched_formal_training_starts=len(returns), probe_env_seed=config['probe_env_seed'],
                  probe_ego_seed=config['probe_ego_seed'], episodes_per_teammate=config['probe_episodes'],
                  updates_per_teammate=config['crp_model_steps'], alpha=config['alpha'], lr=config['lr'],
                  minibatch_size=config['minibatch_size'], order=names, heldout_map_seeds=held['map_seeds'],
                  heldout_ego_seed_start=held['ego_seed_start'], matched_heldout_starts=len(own),
                  source_checkpoint_sha256=file_hash(FORMAL / 'crp.pt'), pool_manifest_sha256=file_hash(pool),
                  crossplay_sha256=file_hash(crossplay_path),
                  training_returns=dict(mean_return=float(returns.mean()), any_food=int((returns > 0).sum()),
                                        all_food=int(np.isclose(returns, 1.).sum())), results=results)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', nargs='?', type=Path, default=FORMAL.parent/'crp_diagnostics/figure6_audit/selfplay.json')
    target = (ROOT / parser.parse_args().output).resolve()
    if os.getenv('FASTAP_ROOT') != str(ROOT) or ROOT/'outputs' not in target.parents or target.exists():
        parser.error('Source scripts/epymarl-env.sh and use a new output file under project outputs/')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.set_num_threads(1)
    target.parent.mkdir(parents=True, exist_ok=True)
    run(target)
