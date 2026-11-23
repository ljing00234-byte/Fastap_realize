import unittest

import torch

from fastap_agents import AgentSpec, FrozenAgent
from fastap_methods.pp_ego_switch_eval import evaluate_ego_switch, make_switch_schedule
from fastap_methods.pp_pool_ego_eval import evaluate_pool_ego, make_evaluation_boards


def build_policies():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(141)
        fields = dict(scene_key="pp", obs_shape=16, n_actions=5, hidden_dim=8, use_rnn=True,
                      obs_last_action=True, obs_agent_id=True)
        ego = AgentSpec(control="ego", n_agents=2, **fields)
        mate = AgentSpec(control="team", n_agents=3, **fields)
        return FrozenAgent(ego, ego.build_network()), {name: FrozenAgent(mate, mate.build_network())
                                                       for name in ("A", "B")}


class PPSwitchEvaluationTests(unittest.TestCase):
    def test_schedule_boundary_same_policy_redraw_and_per_seed_reproducibility(self):
        boards = make_evaluation_boards(42, 16, device="cpu")
        schedule = make_switch_schedule(boards, ["A", "B"], schedule_seed=97)
        ids, due = schedule["policy_ids"], schedule["redraws"]
        self.assertTrue((ids[0, :, :5] == 0).all())
        self.assertTrue((ids[1, :, :5] == 1).all())
        first_events = due.long().argmax(-1)
        self.assertEqual(set(first_events.flatten().tolist()), {5, 6, 7, 8})
        for schedule_row in due.flatten(0, 1):
            event_steps = torch.where(schedule_row)[0]
            self.assertTrue(((event_steps.diff() >= 5) & (event_steps.diff() <= 8)).all())
        changes = ids.diff(dim=-1, prepend=ids[..., :1]) != 0
        self.assertFalse((changes & ~due).any())
        self.assertTrue((~changes & due).any())
        one_map = make_switch_schedule(make_evaluation_boards(48, 1, device="cpu"), ["A", "B"], schedule_seed=97)
        torch.testing.assert_close(one_map["policy_ids"][:, 0], ids[:, 6])
        torch.testing.assert_close(one_map["redraws"][:, 0], due[:, 6])

    def test_stationary_paired_scores_and_control_without_midgame_redraw(self):
        ego, mates = build_policies()
        boards = make_evaluation_boards(301, 8, device="cpu")
        ref = evaluate_pool_ego(ego.model, ego.spec, mates, boards, return_device_tensors=True)
        for interval in ((25, 25), (5, 5)):
            schedule = make_switch_schedule(boards, mates, schedule_seed=120, interval=interval)
            results = evaluate_ego_switch(ego, mates, boards, schedule)
            static, switched = results["stationary"], results["switching"]
            torch.testing.assert_close(static["episode_returns"], ref["episode_returns"])
            for key in ("mean_return", "any_tag_rate"):
                self.assertAlmostEqual(results["paired"][key]["mean_difference"], switched[key] - static[key])
            self.assertAlmostEqual(sum(switched["mean_contacts_by_role"]) * 10, switched["mean_return"])
            self.assertEqual(results["events"]["redraws"], int(schedule["redraws"].sum()))
            self.assertEqual(results["events"]["redraws"], results["events"]["group_changes"]
                             + results["events"]["same_group_redraws"])
            if interval[0] == 25:
                torch.testing.assert_close(static["episode_returns"], switched["episode_returns"])
                self.assertEqual(static["per_teammate"], switched["per_teammate"])
                self.assertEqual(results["events"]["group_changes"], 0)
                self.assertEqual(results["paired"]["mean_return"]["bootstrap_95_ci"], [0., 0.])
