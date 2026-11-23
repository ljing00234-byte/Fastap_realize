import unittest

import torch

from fastap_agents import AgentSpec, FrozenAgent
from fastap_methods.lbf_ego_switch_eval import HORIZON, _boards, _draws, evaluate_ego_switch
from fastap_methods.lbf_pool_ego_eval import ROSTERS, evaluate_pool_ego
from fastap_scenarios.torch_lbf_ego import TorchLBFEgo


class GreedyAction(torch.nn.Module):
    def __init__(self, action):
        super().__init__()
        self.greedy = torch.nn.Parameter(torch.eye(6)[action])

    def init_hidden(self):
        return self.greedy.new_zeros(1, 1)

    def forward(self, inputs, hidden):
        return self.greedy.expand(len(inputs), -1), hidden + 1


def policy(control, action):
    spec = AgentSpec(scene_key="lbf", control=control, n_agents=2 if control == "ego" else 3,
                     obs_shape=TorchLBFEgo.obs_shape, n_actions=6, hidden_dim=1, use_rnn=True,
                     obs_last_action=True, obs_agent_id=True)
    return FrozenAgent(spec, GreedyAction(action))


class SwitchEvaluationTests(unittest.TestCase):
    def test_waits_include_both_endpoints_and_can_redraw_the_same_group(self):
        tape = torch.zeros(1, 3, 40)
        tape[0, 1, 0] = .9999
        tape[0, 2, 3] = .9999
        times, ids, ranks = _draws(tape, 8, (5, 8))
        self.assertEqual(times.tolist(), [[5, 13, 18]])
        self.assertEqual(ids.tolist(), [[[0, -1], [0, -1], [0, 1]]])
        self.assertEqual(tuple(ranks.shape), (1, 3, 36))

    def test_paired_stationary_agrees_with_original_evaluator_and_no_late_events(self):
        first, n_maps = 1030, 3
        starts, _ = _boards(first, n_maps, "cpu")
        single, _ = _boards(first + 1, 1, "cpu")
        for roster in range(2):
            for k, values in starts[roster].items():
                torch.testing.assert_close(values[1], single[roster][k][0])
        boards = dict(first_seed=first, n_maps=n_maps, snapshots=dict(zip(ROSTERS, starts)))
        ego, mates = policy("ego", 5), {"walk": policy("team", 1), "load": policy("team", 5)}
        original = evaluate_pool_ego(ego.model, ego.spec, mates, boards)
        report = evaluate_ego_switch(ego, mates, first_seed=first, n_maps=n_maps, schedule_seed=99,
                                     interval=(HORIZON, HORIZON), chunk_size=4)
        self.assertEqual(report["stationary"], report["switching"])
        for metric in ("mean_return", "any_food_rate", "all_food_rate"):
            self.assertAlmostEqual(report["stationary"][metric], original[metric])
            self.assertEqual(report["paired"][metric]["bootstrap_95_ci"], [0., 0.])
        self.assertEqual(report["events"]["redraws"], 0)

    def test_schedule_and_outcomes_are_invariant_to_batching(self):
        ego, mates = policy("ego", 2), {"A": policy("team", 1), "B": policy("team", 0)}
        args = dict(first_seed=90, n_maps=3, schedule_seed=150, interval=(5, 8))
        a = evaluate_ego_switch(ego, mates, chunk_size=2, **args)
        b = evaluate_ego_switch(ego, mates, chunk_size=8, **args)
        self.assertEqual(a, b)
        self.assertGreater(a["events"]["one_to_two"], 0)
        self.assertGreater(a["events"]["two_to_one"], 0)
        self.assertGreater(a["events"]["same_group_redraws"], 0)
