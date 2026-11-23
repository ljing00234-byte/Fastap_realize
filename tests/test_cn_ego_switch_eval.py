import unittest
from unittest.mock import patch

import torch

from fastap_agents import AgentSpec, FrozenAgent
from fastap_methods.cn_ego_switch_eval import evaluate_ego_switch, make_switch_schedule
from fastap_methods.cn_pool_ego_eval import evaluate_pool_ego, make_evaluation_boards


class CNEgoSwitchEvaluationTests(unittest.TestCase):
    def setUp(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(717)
            fields = dict(scene_key="cn", obs_shape=21, n_actions=5, hidden_dim=8, use_rnn=True,
                          obs_last_action=True, obs_agent_id=True)
            ego = AgentSpec(control="ego", n_agents=2, **fields)
            mate = AgentSpec(control="team", n_agents=3, **fields)
            self.ego = FrozenAgent(ego, ego.build_network())
            self.pool = {k: FrozenAgent(mate, mate.build_network()) for k in ("A", "B", "C")}
        self.boards = make_evaluation_boards(381, 12, device="cpu")

    def test_schedule_preserves_initial_partner_and_map_seed_across_panel_sizes(self):
        ids, coords, due = make_switch_schedule(self.boards, self.pool, schedule_seed=551)
        self.assertTrue((ids[..., :5, 1] == -1).all())
        for j in range(len(self.pool)):
            self.assertTrue((ids[j, :, :5, 0] == j).all())
        self.assertEqual(set(due.long().argmax(-1).flatten().tolist()), {5, 6, 7, 8})
        self.assertTrue((ids[..., 0] != ids[..., 1]).all())
        self.assertTrue((coords.abs() <= 1).all())
        single = make_switch_schedule(make_evaluation_boards(388, 1, device="cpu"), self.pool, schedule_seed=551)
        for panel, row in zip((ids, coords, due), single):
            torch.testing.assert_close(panel[:, 7], row[:, 0])

    def test_no_events_match_heldout_and_paired_redraws_change_roster(self):
        static = evaluate_pool_ego(self.ego.model, self.ego.spec, self.pool, self.boards, return_device_tensors=True)
        with patch("fastap_methods.cn_ego_switch_eval.evaluate_pool_ego", side_effect=AssertionError("extra rollout")):
            same = evaluate_ego_switch(self.ego, self.pool, self.boards, schedule_seed=61, interval=(25, 25),
                                       stationary=static)
        for arm in ("policy_switch", "open_switch"):
            torch.testing.assert_close(same[arm]["episode_returns"], static["episode_returns"], rtol=1e-6, atol=1e-6)
            self.assertEqual(same["events"][arm]["redraws"], 0)
        result = evaluate_ego_switch(self.ego, self.pool, self.boards, schedule_seed=61, stationary=static)
        events = result["events"]
        self.assertEqual(events["policy_switch"]["redraws"], events["open_switch"]["redraws"])
        self.assertEqual(events["policy_switch"]["two_teammate_steps"], 0)
        self.assertGreater(events["open_switch"]["one_to_two"], 0)
        self.assertGreater(events["open_switch"]["two_to_one"], 0)
        for arm in ("policy_switch", "open_switch"):
            e = events[arm]
            self.assertEqual(e["redraws"], e["group_changes"] + e["same_group_redraws"])
            s = result[arm]
            self.assertEqual(s["n_episodes"], 36)
            self.assertAlmostEqual(s["mean_return"], -25*s["mean_landmark_distance_sum_per_step"]
                                   - s["mean_pair_collisions_per_episode"], places=5)
        delta = result["open_switch"]["mean_return"] - result["policy_switch"]["mean_return"]
        self.assertAlmostEqual(result["paired"]["open_minus_policy"]["mean_difference"], delta)
