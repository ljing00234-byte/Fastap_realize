import unittest

import numpy as np
import torch

from scripts.plot_lbf_crp import aggregate_crossplay, project_embeddings


class LbfCrpAnalysisTests(unittest.TestCase):
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
