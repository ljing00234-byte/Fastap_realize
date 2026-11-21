import unittest

import numpy as np
import torch

from fastap_agents import AgentSpec, FrozenAgent
from fastap_methods.pp_pool_ego_eval import (BOARD_GENERATOR, BOOTSTRAP_RESAMPLES, compare_ego_returns,
                                              evaluate_pool_ego, make_evaluation_boards, serialize_scores)
from fastap_methods.predator_prey_eval import rollout_contacts


class PursueOrWait(torch.nn.Module):
    def __init__(self, pursue):
        super().__init__()
        self.pursue = pursue
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def init_hidden(self):
        return self.bias.new_zeros(1, 1)

    def forward(self, inputs, hidden):
        scores = inputs.new_zeros((len(inputs), 5))
        if self.pursue:
            x, y = inputs[:, 12:14].unbind(1)
            scores[:, 1:] = torch.stack((-x, x, -y, y), dim=1)
        else:
            scores[:, 0] = 1
        return scores, hidden


class PPEgoEvaluationTests(unittest.TestCase):
    def test_stationary_returns_and_role_contacts_equal_scalar_rollouts(self):
        first, n_maps = 15, 8
        boards = make_evaluation_boards(first, n_maps)
        fields = dict(scene_key="pp", obs_shape=16, n_actions=5, hidden_dim=1, use_rnn=True,
                      obs_last_action=True, obs_agent_id=True)
        ego = FrozenAgent(AgentSpec(control="ego", n_agents=2, **fields), PursueOrWait(True))
        teammate_spec = AgentSpec(control="team", n_agents=3, **fields)
        mates = {name: FrozenAgent(teammate_spec, PursueOrWait(pursue))
                 for name, pursue in (("chase", True), ("wait", False))}
        ego.model.train()
        scored = evaluate_pool_ego(ego.model, ego.spec, mates, boards)
        self.assertTrue(ego.model.training)
        self.assertEqual(scored["n_episodes"], n_maps * len(mates))
        for name, mate in mates.items():
            factories = [ego.policy_factory(0), ego.policy_factory(1), mate.policy_factory(2)]
            contacts = np.stack([rollout_contacts(seed, factories) for seed in range(first, first + n_maps)])
            np.testing.assert_array_equal(10 * contacts.sum(1), scored["episode_returns"][name])
            np.testing.assert_array_equal(contacts.mean(0), scored["per_teammate"][name]["mean_contacts_by_role"])
        self.assertGreater(max(scored["episode_returns"]["chase"]), 10)
        self.assertNotEqual(scored["episode_returns"]["chase"], scored["episode_returns"]["wait"])

        device_boards = make_evaluation_boards(first, n_maps, device="cpu")
        device_scores = evaluate_pool_ego(ego.model, ego.spec, mates, device_boards, return_device_tensors=True)
        self.assertEqual(device_scores["board_generator"], BOARD_GENERATOR)
        for field in ("board_device", "policy_device", "scoring_device"):
            self.assertEqual(device_scores[field], "cpu")
        self.assertEqual(device_scores["episode_returns"].shape, (2, n_maps))
        saved = serialize_scores(device_scores)
        np.testing.assert_array_equal(saved["episode_returns"]["chase"], device_scores["episode_returns"][0].numpy())
        self.assertEqual(saved["mean_return"], device_scores["episode_returns"].mean().item())
        paired = compare_ego_returns(device_scores, device_scores, device="cpu")
        self.assertEqual(paired["bootstrap_95_ci"], [0, 0])
        self.assertEqual(paired["bootstrap_device"], "cpu")
        self.assertEqual(paired["bootstrap_resamples"], BOOTSTRAP_RESAMPLES)
        with self.assertRaisesRegex(ValueError, "requested evaluation device"):
            compare_ego_returns(device_scores, device_scores, device="cuda:0")
        with self.assertRaisesRegex(ValueError, "original return tensors"):
            compare_ego_returns(saved, saved, device="cuda:0")

    def test_device_maps_follow_pp_reset_law_and_are_seed_addressable(self):
        boards = make_evaluation_boards(110, 9, device="cpu")
        single = make_evaluation_boards(113, 1, device="cpu")
        self.assertTrue(torch.equal(boards["positions"][3], single["positions"][0]))
        self.assertTrue(torch.all(boards["positions"][:, :4].abs() <= 1))
        self.assertTrue(torch.all(boards["positions"][:, 4:].abs() <= 0.9))
        self.assertFalse(boards["velocity"].any())
        self.assertEqual(boards["positions"].dtype, torch.float64)
        expected = 2 * torch.rand((6, 2), dtype=torch.float64, generator=torch.Generator().manual_seed(113)) - 1
        expected[4:] *= .9
        self.assertTrue(torch.equal(single["positions"][0], expected))

    def test_bootstrap_preserves_partner_pairings_within_maps(self):
        ids = dict(first_map_seed=7, n_maps_per_teammate=2, pool_order=["A", "B"])
        trained = dict(**ids, episode_returns={"A": [20, 0], "B": [0, 20]})
        control = dict(**ids, episode_returns={"B": [0, 0], "A": [0, 0]})
        self.assertEqual(compare_ego_returns(trained, control), dict(mean_difference=10, bootstrap_95_ci=[10, 10]))
        tensor_trained = dict(trained, episode_returns=torch.tensor([[20., 0.], [0., 20.]]))
        tensor_control = dict(control, episode_returns=torch.zeros(2, 2))
        paired = compare_ego_returns(tensor_trained, tensor_control, device="cpu")
        self.assertEqual(paired["mean_difference"], 10)
        self.assertEqual(paired["bootstrap_95_ci"], [10, 10])
        with self.assertRaises(ValueError):
            compare_ego_returns(trained, dict(control, first_map_seed=8))
