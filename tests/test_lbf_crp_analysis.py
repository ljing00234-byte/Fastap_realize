import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from scripts.plot_lbf_crp import aggregate_crossplay, project_embeddings


class LbfCrpAnalysisTests(unittest.TestCase):
    def test_paired_replacements_and_current_encoder_centers(self):
        from scripts.audit_lbf_crp_stability import refreshed_centers, replacement_bags

        removed, added = [], []
        for base, swaps in replacement_bags(32, 8):
            self.assertEqual([slot for slot, _ in swaps], list(range(8)))
            for slot, incoming in swaps:
                self.assertNotIn(incoming, base)
                removed.append(base[slot])
                added.append(incoming)
        self.assertEqual(sorted(added), sorted(removed))
        self.assertEqual(sorted(removed), list(range(32)))
        vectors = [torch.tensor([v]) for v in [1., 7., 5.]]
        centers = refreshed_centers(vectors, [0, 1, 0])
        self.assertEqual([v.item() for v in centers], [3., 7.])

    def test_standalone_crp_keeps_failed_probes_and_replays_the_same_start_in_both_modes(self):
        from fastap_agents import AgentSpec
        from fastap_methods.lbf_crp_analysis import ROOT, run_crp
        from fastap_scenarios.fastap_rollout import load_fastap_groups

        spec = AgentSpec("lbf", "team", 3, 21, 6, 8, True, True, True)
        temp_root = ROOT / '.tmp/epymarl'
        temp_root.mkdir(exist_ok=True, parents=True)
        with TemporaryDirectory(dir=temp_root) as temp:
            base = Path(temp)
            with torch.random.fork_rng(devices=[]):
                model = spec.build_network()
            weights = {key: torch.zeros_like(value) for key, value in model.state_dict().items()}
            members = []
            for i in range(1, 9):
                checkpoint = base / f'T{i}'
                checkpoint.mkdir()
                spec.save(checkpoint)
                torch.save(weights, checkpoint / 'agent.th')
                digest = hashlib.sha256((checkpoint / 'agent.th').read_bytes()).hexdigest()
                members.append(dict(id=checkpoint.name, path=checkpoint.name, sha256=digest))
            pool = base / 'pool.json'
            pool.write_text(json.dumps(dict(format='lbf-teammate-pool-v1', scene='lbf', role=2,
                                            target_size=8, members=members)))
            groups = load_fastap_groups('lbf', pool)
            for single, mixed in zip(groups[::2], groups[1::2]):
                actors = [factory() for factory in (*single.probe_egos, *single.policies)]
                self.assertEqual([actor.role.argmax().item() for actor in actors], [0, 1, 2])
                self.assertTrue(all(actor.model is actors[0].model for actor in actors))
                self.assertFalse(mixed.probe_egos)

            config = dict(scenes={'lbf': {'alpha': .5, 'groups_per_round': 4}},
                          probe_episodes=1, crp_model_steps=1, lr=.0005)
            own_dir, random_dir = base / 'own', base / 'random'
            summary = run_crp(pool, own_dir, method_config=config)
            total = summary['probe_outcomes']['total']
            self.assertEqual(summary['protocol']['probe_ego'], 'own_original')
            self.assertEqual((total['episodes'], total['any_food'], total['all_food']), (8, 0, 0))
            for member in members:
                self.assertEqual(summary['probe_outcomes']['per_group'][member['id']]['episode_returns'], [0.])
            legacy = run_crp(pool, random_dir, method_config=config, probe_ego='random_legal')
            self.assertEqual(legacy['protocol']['probe_ego'], 'random_legal')
            own, random = [torch.load(directory / 'crp.pt', weights_only=True, map_location='cpu')['crp']['probes']
                           for directory in (own_dir, random_dir)]
            self.assertEqual(len(own), len(random))
            self.assertEqual(len(own), 8)
            for own_probe, random_probe in zip(own, random):
                states, random_states = own_probe[0], random_probe[0]
                self.assertTrue(torch.equal(states[0], random_states[0]))
                self.assertTrue(torch.equal(states, states[0].expand_as(states)))
                self.assertFalse(torch.equal(random_states, random_states[0].expand_as(random_states)))

    def test_joint_observations_use_real_slots_and_preserve_padding(self):
        from contextlib import closing
        from fastap_scenarios.foraging import ForagingWorld
        from scripts.lbf_crp_observations import joint_observations

        with closing(ForagingWorld(seed=7)) as world:
            world.reset(n_teammates=1)
            state = torch.from_numpy(world.state())
            actual = torch.from_numpy(world.observations()).flatten()
        result = joint_observations(torch.stack([state, torch.zeros_like(state)]))
        torch.testing.assert_close(result[0], actual)
        self.assertFalse(result[1].any())

    def test_vertex_probability_is_gaussian_density_normalized_over_actions(self):
        from scripts.audit_lbf_crp import AuditBehaviorModel
        model = AuditBehaviorModel(3, 3, 2)
        model.variant = 'discrete_eq4'
        mean = torch.tensor([.2, -.7, .5, .4, .6, -.3]).reshape(1, 1, 6)
        scale = torch.tensor([.8, .5, .4, .2, .7, .3]).reshape_as(mean)
        target = torch.tensor([1., 0., 0., 0., 1., 0.]).reshape(1, 1, 2, 3)
        active = torch.tensor([[[True, False]]])
        dist = torch.distributions.Normal(mean.flatten()[:3], scale.flatten()[:3])
        log_density = dist.log_prob(torch.eye(3)).sum(-1)
        expected = log_density.log_softmax(0)[0]
        torch.testing.assert_close(model._score_actions(mean, scale, target, active)[0], expected)

    def test_eq4_and_mean_nll_weight_each_trajectory_as_reported(self):
        from scripts.audit_lbf_crp import AuditCRP
        for variant in ['paper', 'mean_nll']:
            crp = AuditCRP(variant, state_dim=3, n_actions=3, n_teammates=1, alpha=.5)
            ll = torch.tensor([-40., -21., -20.], requires_grad=True)
            crp._model_loss(ll).backward()
            expected = ll.detach().softmax(0) if variant == 'paper' else torch.ones(3) / 3
            torch.testing.assert_close(-ll.grad, expected)
            self.assertAlmostEqual(crp.max_weight[0], expected.max().item(), places=6)

    @torch.no_grad()
    def test_admission_evidence_counts_only_observed_actions(self):
        from fastap_methods.fastap_crp import BehaviorModel
        from scripts.audit_lbf_crp_evidence import episode_scores

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(74)
            model = BehaviorModel(state_dim=3, n_actions=3, n_teammates=2).eval()
            states = torch.randn(1, 4, 3)
            center = torch.randn(model.width)
        actions = torch.tensor([[[0, 1], [2, -1], [-1, -1]]])
        slots = torch.tensor([[[True, True], [True, False], [False, False]]])
        filled = torch.tensor([[True, True, False]])
        score, preference, mass = episode_scores(model, (states, actions, slots, filled), center)

        # Enumerate the Gaussian score of every possible action at each observed slot.
        expected_preference, expected_mass = 0., 0.
        for _, step, slot in slots.nonzero().tolist():
            one_slot = torch.zeros_like(slots)
            one_slot[0, step, slot] = True
            scores = []
            for action in range(model.n_actions):
                candidate = actions.clone()
                candidate[0, step, slot] = action
                scores.append(model.trajectory_log_prob(states, candidate, one_slot, filled, center[None]))
            log_mass = torch.cat(scores).logsumexp(0).item()
            expected_mass += log_mass
            expected_preference += scores[actions[0, step, slot].item()].item() - log_mass
        np.testing.assert_allclose(preference, [expected_preference], atol=1e-5)
        np.testing.assert_allclose(mass, [expected_mass], atol=1e-5)
        np.testing.assert_allclose(score, preference + mass, atol=1e-5)

    def test_directed_blocks_and_within_cluster_swaps(self):
        # The two C1 members are nonadjacent in the original matrix.
        matrix = [[1., .1, .2], [.3, 1., .9], [.6, .7, 1.]]
        crossplay = dict(ids=['T1', 'T2', 'T3'], n_maps=256, first_map_seed=870000,
                         crossplay=[[dict(mean_return=x) for x in row] for row in matrix])
        assignments = dict(T1='C1', T2='C2', T3='C1')
        result = aggregate_crossplay(crossplay, assignments)
        np.testing.assert_allclose(result['mean_return'], [[.7, .4], [.6, 1.]])
        self.assertEqual(result['pairings'], [[4, 2], [2, 1]])
        swaps = result['summaries']['within_cluster_other_members']
        self.assertEqual(swaps['pairings'], 2)
        self.assertAlmostEqual(swaps['mean_return'], .4)
        for assignments, empty in [
                (dict(T1='C1', T2='C2', T3='C3'), 'within_cluster_other_members'),
                (dict(T1='C1', T2='C1', T3='C1'), 'across_clusters')]:
            summary = aggregate_crossplay(crossplay, assignments)['summaries']
            self.assertEqual(summary[empty], dict(pairings=0, mean_return=None))

    def test_common_pca_preserves_trajectory_labels(self):
        rows = [
            dict(teammate=t, cluster=c, episode=i, embedding=x)
            for t, c, i, x in [('T1', 'C1', 1, [2, 1, 4]), ('T2', 'C2', 1, [5, 3, 4]),
                               ('T1', 'C1', 2, [3, 2, 5]), ('T2', 'C2', 2, [4, 4, 1])]
        ]
        result = project_embeddings(rows, dict(T1='C1', T2='C2'))
        coords = np.array([row['xyz'] for row in result['rows']])
        original = np.array([row['embedding'] for row in rows])
        np.testing.assert_allclose(coords.mean(0), 0, atol=1e-12)
        np.testing.assert_allclose(coords @ result['components'] + result['center'], original)
        self.assertAlmostEqual(sum(result['explained_variance_ratio']), 1.)
        self.assertEqual([(r['teammate'], r['episode']) for r in result['rows']],
                         [(r['teammate'], r['episode']) for r in rows])


if __name__ == '__main__':
    unittest.main()
