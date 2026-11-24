import json
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch
import yaml

from fastap_agents import FastapPolicy
from fastap_methods import TRAINERS
from fastap_methods.fastap_training import _episode, collect_probe_episodes, evaluate_fastap
from fastap_scenarios import SCENARIOS, TeammateGroup
from fastap_scenarios.fastap_rollout import FastapRolloutEnv
from fastap_scenarios.policies import stay

ROOT = Path(__file__).resolve().parents[1]


class FastapTrainingTests(unittest.TestCase):
    def test_probe_egos_keep_their_roles_and_start_fresh_per_episode(self):
        def first_move():
            acted = False

            def act(obs, available, rng):
                nonlocal acted
                choice = 0 if acted else int(np.flatnonzero(available[1:5])[0]) + 1
                acted = True
                return choice

            return act

        group = TeammateGroup('solo', (lambda: stay,), probe_egos=(lambda: stay, first_move))
        with closing(FastapRolloutEnv('lbf', [group], seed=19)) as env:
            probes = collect_probe_episodes(env, 0, np.random.default_rng(32), 2)
        self.assertEqual(len(probes), 2)
        for ep in probes:
            self.assertFalse(ep['actions'][:, 0].any())
            self.assertGreater(ep['actions'][0, 1].item(), 0)
            self.assertFalse(ep['actions'][1:, 1].any())
            self.assertTrue(ep['filled'].all())
            self.assertEqual(ep['reward'].sum().item(), 0.)

    def test_random_probe_mode_replays_the_previous_episode_collector(self):
        solo = TeammateGroup('solo', (lambda: stay,))
        trained = TeammateGroup('trained', (lambda: stay,), probe_egos=(lambda: stay, lambda: stay))
        with closing(FastapRolloutEnv('lbf', [solo], seed=19)) as env:
            rng = np.random.default_rng(32)
            expected = [_episode(env, 0, None, rng) for _ in range(2)]
        for group, use_original in ((solo, True), (trained, False)):
            with self.subTest(group=group.name), closing(FastapRolloutEnv('lbf', [group], seed=19)) as env:
                actual = collect_probe_episodes(env, 0, np.random.default_rng(32), 2,
                                                use_original_egos=use_original)
                self.assertEqual(len(actual), len(expected))
                for ep, old in zip(actual, expected):
                    for field in old:
                        self.assertTrue(torch.equal(ep[field], old[field]), field)

    def test_alternating_admissions_and_decentralized_checkpoint(self):
        scene = yaml.safe_load((ROOT / 'configs/scenarios/cn.yaml').read_text())
        scene['env_args']['world_options']['episode_limit'] = 3
        agent = yaml.safe_load((ROOT / 'configs/agents/fastap.yaml').read_text())
        method = yaml.safe_load((ROOT / 'configs/methods/fastap.yaml').read_text())
        method.update(t_max=9, batch_size=1, buffer_size=4, group_interval_steps=6,
                      probe_episodes=1, crp_model_steps=1, test_nepisode=2, switch_interval=[1, 1])
        groups = SCENARIOS['cn'].demo_groups
        temp_root = ROOT / '.tmp/epymarl'
        temp_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=temp_root) as temp:
            path = Path(temp)

            def generate(i):
                if i == 1:
                    earlier = [json.loads(line) for line in (path / 'metrics.jsonl').read_text().splitlines()]
                    self.assertEqual([m['step'] for m in earlier if m['event'] == 'episode'], [3, 6])
                return groups[i]

            result = TRAINERS['fastap'](scene_config=scene, agent_config=agent, method_config=method,
                                       run_dir=path, seed=8, device='cpu', max_groups=2, group_factory=generate)
            self.assertEqual(result, json.loads((path / 'summary.json').read_text()))
            self.assertEqual((result['training_steps'], result['probe_steps'], result['encoder_updates']), (9, 6, 3))
            self.assertEqual(result['optimizer_updates'], result['q_updates'] + result['encoder_updates'])
            admissions = [json.loads(line) for line in (path / 'metrics.jsonl').read_text().splitlines()
                          if '"admit"' in line]
            self.assertEqual([(a['step'], a['group']) for a in admissions], [(0, groups[0].name), (6, groups[1].name)])
            saved = path / result['checkpoint']
            training = torch.load(saved / 'training.pt', weights_only=True, map_location='cpu')
            self.assertEqual(set(training['crp']['assignments']), {g.name for g in groups})
            policy = FastapPolicy.load(saved)
            actual = evaluate_fastap(policy, 'cn', groups, episodes=2, seed=1_000_008,
                                     switch_interval=[1, 1], world_options=scene['env_args']['world_options'])
            self.assertEqual(actual, result['evaluation'])
            self.assertEqual(actual['stationary']['mean_group_changes'], 0)
            self.assertEqual(actual['switching']['mean_resamples'], 2)
