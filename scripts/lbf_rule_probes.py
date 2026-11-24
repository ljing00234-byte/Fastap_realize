"""Collect blinded trajectories of published LBF heuristic families on shared maps.

The goal rules follow HRI-EU's LBF experiment definitions. Each rule observes the
local food/player triples and legal actions of one slot in our 6x6 environment.
"""

import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastap_methods.fastap_training import _batch, _episode
from fastap_scenarios.fastap_rollout import FastapRolloutEnv, load_fastap_groups
from fastap_scenarios.policies import TeammateGroup


SOURCE = 'https://github.com/HRI-EU/lb-foraging#available-agent-heuristics'
RULES = ('BASELINE', 'EGOISTIC', 'SOCIAL1')
MEMBERS = [f'B{i:02}' for i in range(1, 10)]
SPLITS = {'train': (6830000, 8), 'support': (6840000, 32), 'query': (6850000, 32)}
DELTAS = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
FIELDS = ('state', 'teammate_actions', 'teammate_mask', 'filled', 'reward')
DEFAULT_POOL = ROOT / 'outputs/fastap_lbf_formal_pool_01/pool.json'


def action_probabilities(rule, obs, avail):
    """Return probabilities over NONE, NORTH, SOUTH, WEST, EAST, LOAD."""
    if rule not in RULES:
        raise ValueError(f'Unknown LBF rule: {rule}')
    moves = np.flatnonzero(avail[1:5]) + 1
    explore = np.zeros(6)
    if len(moves):
        explore[moves] = 1 / len(moves)
    else:
        explore[0] = 1
    load = np.array([0., 0., 0., 0., 0., 1.])
    if rule == 'BASELINE':
        return load if avail[5] else explore

    triples = np.asarray(obs).reshape(7, 3)
    foods = triples[:3]
    foods = foods[foods[:, 2] > 0]
    players = triples[3:]
    own = players[0]
    if rule == 'EGOISTIC':
        foods = foods[foods[:, 2] <= own[2]]
        reference = own[:2]
    else:
        reference = players[players[:, 2] > 0, :2].mean(axis=0)
    if not len(foods):
        return explore
    squared = ((foods[:, :2] - reference) ** 2).sum(axis=1)
    candidates = foods[squared == squared.min(), :2]
    total = np.zeros(6)
    for target in candidates:
        distance = np.abs(own[:2] - target).sum()
        if distance == 1 and avail[5]:
            total += load
            continue
        remaining = np.abs(own[:2] + DELTAS[moves - 1] - target).sum(axis=1)
        closer = moves[remaining < distance]
        if len(closer):
            total[closer] += 1 / len(closer)
        else:
            total += explore
    return total / len(candidates)


def collect_episode(rule, member_index, map_seed, ego_factories):
    # Give replicas independent policy randomness while resetting identical boards
    # and identical recurrent original T1 partners for each member.
    rng = np.random.default_rng(np.random.SeedSequence([74105, map_seed, member_index]))
    masks = []

    def act(obs, avail, _env_rng):
        probs = action_probabilities(rule, obs, avail)
        action = int(rng.choice(6, p=probs))
        masks.append(np.asarray(avail, dtype=bool).copy())
        return action

    group = TeammateGroup(MEMBERS[member_index], (lambda: act,))
    with closing(FastapRolloutEnv('lbf', [group], seed=map_seed)) as env:
        episode = _episode(env, 0, None, np.random.default_rng(map_seed),
                           probe_egos=tuple(factory() for factory in ego_factories))
    active = episode['filled']
    if active.sum().item() != len(masks) or not torch.equal(episode['teammate_mask'][:, 0], active):
        raise RuntimeError("Recorded rule steps do not match the environment transitions")
    available = torch.zeros(len(active), 6, dtype=torch.bool)
    available[active] = torch.from_numpy(np.stack(masks))
    sidecar = dict(obs=episode['teammate_obs'][:, 0].clone(), available=available)
    return {key: episode[key] for key in FIELDS}, sidecar


def collect(directory, pool):
    if ROOT/'outputs' not in directory.parents or ROOT/'outputs' not in pool.parents:
        raise ValueError('Use project output paths for the pool and data')
    if os.environ.get('FASTAP_ROOT') != str(ROOT):
        raise ValueError('Source scripts/epymarl-env.sh for project-local caches')
    directory.mkdir(parents=True, exist_ok=True)
    filenames = ['train.pt', 'support.pt', 'query.pt', 'observations.pt', 'truth.json']
    if any((directory / name).exists() for name in filenames):
        raise FileExistsError(f'Dataset files already exist in {directory}')
    torch.set_num_threads(1)
    roster = np.random.default_rng(53621).permutation(np.repeat(RULES, 3)).tolist()
    first = next(g for g in load_fastap_groups('lbf', pool) if g.name == 'T1')
    if len(first.probe_egos) != 2:
        raise ValueError('Two verified T1 partners are required')
    seen = set()
    local = {}
    splits = {}
    with TemporaryDirectory(dir=directory) as temp:
        stage = Path(temp)
        for split, (start, count) in SPLITS.items():
            episodes, local_rows, initial_hashes = [], [], []
            seeds = list(range(start, start + count))
            for seed in seeds:
                initial = set()
                for index, rule in enumerate(roster):
                    episode, sidecar = collect_episode(rule, index, seed, first.probe_egos)
                    initial.add(hashlib.sha256(episode['state'][0].numpy().tobytes()).hexdigest())
                    episodes.append(episode)
                    local_rows.append(sidecar)
                if len(initial) != 1 or not initial.isdisjoint(seen):
                    raise RuntimeError(f'Duplicate or unpaired map at seed {seed}')
                initial_hashes.append(next(iter(initial)))
                seen.update(initial)
            batch = _batch(episodes)
            torch.save(dict(members=MEMBERS, map_seeds=seeds, batch=batch), stage/f'{split}.pt')
            local[split] = _batch(local_rows)
            splits[split] = dict(map_seeds=seeds, initial_state_sha256=initial_hashes,
                                  episodes=len(episodes), decisions=int(batch['filled'].sum()),
                                  distinct_initial_states=len(initial_hashes))
        torch.save(local, stage/'observations.pt')
        digest = {name: hashlib.sha256((stage/name).read_bytes()).hexdigest() for name in filenames[:-1]}
        metadata = dict(
            labels=dict(zip(MEMBERS, roster)), source=SOURCE,
            rules={'BASELINE': 'Walk at random, load when adjacent to food.',
                   'EGOISTIC': 'Target the nearest visible food collectible at the agent’s own level.',
                   'SOCIAL1': 'Target the food nearest the Euclidean center of visible players, regardless of level.'},
            adaptation='All three rules act from slot 2 local sight-one observations. Explore via uniform legal '
                       'cardinal moves when no eligible target or no goal-reducing move exists; wait if blocked. '
                       'Use uniform ties over equally near targets and legal Manhattan-distance-reducing moves. '
                       'Food levels in this task exceed every individual agent level; EGOISTIC always explores. '
                       'These are local goal-rule adaptations, without the source implementation’s optional memory or patience.',
            replicas='Three independently sampled copies of each exact rule. These are nine anonymous policy '
                     'streams, not nine independently learned policies. Every map × member gets its own RNG.',
            assignment_seed=53621, policy_rng='SeedSequence([74105, map_seed, zero_based_member_index])',
            row_order='map_index * 9 + member_index', fixed_partners='T1 greedy roles 0 and 1',
            pool=str(pool.relative_to(ROOT)), pool_sha256=hashlib.sha256(pool.read_bytes()).hexdigest(),
            fixed_partner_sha256=json.loads(pool.read_text())['members'][0]['sha256'],
            splits=splits, files_sha256=digest)
        (stage/'truth.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
        for name in filenames:
            os.link(stage/name, directory/name)
    return splits


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--pool', type=Path, default=DEFAULT_POOL)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    summaries = collect(args.directory.resolve(), args.pool.resolve())
    for split, summary in summaries.items():
        print(f'{split}: {summary["episodes"]} episodes, {summary["decisions"]} valid actions')
