from contextlib import closing
import math
import unittest

import numpy as np
import torch

from fastap_methods.fastap_crp import TeammateCRP
from fastap_methods.lbf_admission_trace import contrast, trace_admission
from fastap_methods.lbf_probe_starts import pre_load_moves
from fastap_scenarios.fastap_rollout import FastapRolloutEnv
from fastap_scenarios.policies import LBF_DEMO


def recorded_probe(size, length, seed):
    with closing(FastapRolloutEnv('lbf', LBF_DEMO, seed=seed)) as env:
        _, state, _ = env.reset(size - 1)
        states, acts, opportunities = [state.copy()], [], []
        for _ in range(length):
            observations, masks = env.world.observations(), env.world.avail_actions()
            _, done, info = env.step(np.zeros(2, dtype=int))
            taken = info['teammate_actions']
            acts.append(taken)
            states.append(env.get_state().copy())
            for slot in range(size):
                obs, avail, a = observations[slot + 2], masks[slot + 2], taken[slot]
                closer = pre_load_moves(obs, avail)
                if avail[5]:
                    opportunities.append('load_available')
                elif closer.any():
                    toward = 1 <= a <= 4 and closer[a - 1]
                    opportunities.append('preload_approach' if toward else 'preload_other')
                else:
                    opportunities.append('other')
            if done:
                break
    actions = torch.from_numpy(np.stack(acts))
    return ((torch.from_numpy(np.stack(states)), actions, actions >= 0, torch.ones(len(actions), dtype=torch.bool)),
            opportunities)


class LbfAdmissionTraceTests(unittest.TestCase):
    def test_trace_reduces_to_actual_candidate_scores_for_mixed_rosters_and_ragged_batches(self):
        group, expected_opportunities = zip(*(recorded_probe(*spec) for spec in ((1, 4, 15), (2, 5, 21),
                                                                                 (1, 3, 8))))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            crp = TeammateCRP(48, 6, 2, alpha=.5, minibatch_size=2, decoder='categorical')
            counts = [2, 1]
            centers = [torch.randn(16) for _ in counts]
        crp.counts, crp.centers = counts, centers
        label, _ = crp._assign(group)
        trace = trace_admission(crp.model, group, counts, centers, crp.alpha, minibatch_size=2,
                                map_seeds=[15, 21, 8], expected_scores=crp.last_assignment_log_scores)
        self.assertEqual(trace['winner'], f'C{label + 1}')
        self.assertEqual([row['opportunity'] for row in trace['rows']], sum((p for p in expected_opportunities), []))
        self.assertEqual([row['map_seed'] for row in trace['rows']], [15]*4 + [21]*10 + [8]*3)
        self.assertEqual([row['role'] for row in trace['rows']], [2]*4 + [2, 3]*5 + [2]*3)
        for diff in (trace['new_vs_existing'], trace['winner_vs_runner_up'], contrast(trace, 'C1', 'C2')):
            margin = diff['total_margin']
            self.assertAlmostEqual(margin, diff['cumulative_margin'][-1] + diff['reduction_roundoff'])
            self.assertAlmostEqual(margin, diff['prior_margin'] + sum(diff['per_episode']), places=4)
            for bucket in ('by_action', 'by_opportunity'):
                self.assertEqual(sum(v['count'] for v in diff[bucket].values()), len(trace['rows']))
                self.assertAlmostEqual(sum(v['margin'] for v in diff[bucket].values()), diff['evidence_margin'])
            self.assertLess(abs(diff['reduction_roundoff']), 1e-4)
        self.assertAlmostEqual(contrast(trace, 'C3', 'C1')['prior_margin'], math.log(.5 / 2.))
        with self.assertRaises(AssertionError):
            trace_admission(crp.model, group, counts, centers, crp.alpha, minibatch_size=2,
                            expected_scores=crp.last_assignment_log_scores + .01)

    def test_forced_first_admission_and_supplied_candidate_center(self):
        probe, _ = recorded_probe(1, 2, 11)
        crp = TeammateCRP(48, 6, 2, alpha=.5, decoder='categorical')
        _, vector = crp._assign([probe])
        trace = trace_admission(crp.model, [probe], [], [], .5, candidate_centers=[vector],
                                expected_scores=crp.last_assignment_log_scores)
        self.assertEqual(trace['winner'], 'C1')
        self.assertIsNone(trace['new_vs_existing'])
        self.assertIsNone(trace['winner_vs_runner_up'])
        self.assertEqual(trace['candidates'][0]['log_prior'], 0.)
