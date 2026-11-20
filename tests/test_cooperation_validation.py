import unittest

import numpy as np
import torch

from fastap_agents import AgentSpec
from fastap_methods.cooperation_validation import _summarize_pairs, evaluate_cooperation
from fastap_scenarios.foraging import ForagingWorld
from fastap_scenarios.torch_foraging import TorchForaging


class CooperationValidationTests(unittest.TestCase):
    def test_paired_gate_rejects_a_tie_and_distinguishes_full_from_partial_success(self):
        one_food = np.tile([1, 1, 0, 1 / 3], (20, 1))
        tied = _summarize_pairs(one_food, one_food, "adjacent")
        self.assertFalse(tied["gate_passed"])
        self.assertEqual(tied["paired_one_sided_p"], 1)
        self.assertEqual(tied["trained"]["success_rate"], 1)
        self.assertEqual(tied["trained"]["full_completion_rate"], 0)
        lower_return = one_food.copy()
        lower_return[:10] = 0
        passed = _summarize_pairs(one_food, lower_return, "approach")
        self.assertTrue(passed["gate_passed"])
        self.assertEqual(passed["trained_only_successes"], 10)
        self.assertEqual(passed["random_only_successes"], 0)
        self.assertEqual(passed["paired_one_sided_p"], 1 / 1024)

    def test_adjacent_positive_control_and_reproducible_heldout_evaluation(self):
        spec = AgentSpec("lbf", "team", 3, 21, 6, 64, True, True, True)
        model = spec.build_network()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.fc2.bias[5] = 1  # Always LOAD when legal.
        kwargs = dict(n_maps=32, device="cpu")
        validation = evaluate_cooperation(model, spec, "adjacent", **kwargs)
        self.assertTrue(model.training)
        self.assertTrue(validation["gate_passed"])
        self.assertEqual(validation["trained"]["first_step_success_rate"], 1)
        self.assertEqual(validation["trained"]["full_completion_rate"], 0)
        self.assertAlmostEqual(validation["trained"]["mean_return"], 1 / 3)
        self.assertGreater(validation["paired_difference"]["first_step_success_rate"], 0)
        heldout = evaluate_cooperation(model, spec, "adjacent", split="heldout", **kwargs)
        self.assertNotEqual(validation["map_batch_seed"], heldout["map_batch_seed"])
        self.assertEqual(heldout, evaluate_cooperation(model, spec, "adjacent", split="heldout", **kwargs))
        alternate = evaluate_cooperation(model, spec, "approach", split="heldout", split_seed=7123, **kwargs)
        self.assertEqual((alternate["split_seed"], alternate["map_batch_seed"],
                          alternate["random_teammate_seed"]), (7123, 7125, 7126))

    def test_random_replacement_matches_scalar_environment_on_identical_snapshots(self):
        spec = AgentSpec("lbf", "team", 3, 21, 6, 64, True, True, True)
        model = spec.build_network()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.fc2.bias[5] = 1
        n_maps = 8
        report = evaluate_cooperation(model, spec, "adjacent", n_maps=n_maps, device="cpu")
        starts = TorchForaging(n_maps, device="cpu", seed=report["map_batch_seed"])
        starts.reset("adjacent")
        snapshot = starts.snapshot()
        rng = torch.Generator().manual_seed(report["random_teammate_seed"])
        scores = torch.stack([torch.rand(n_maps, 6, generator=rng)
                              for _ in range(starts.episode_limit)]).numpy()
        records = []
        for i in range(n_maps):
            world = ForagingWorld(seed=i)
            try:
                world.reset(n_teammates=1)
                world.field[:] = snapshot["field"][i].numpy()
                for slot, player in enumerate(world.players):
                    player.position = tuple(snapshot["positions"][i, slot].tolist())
                    player.level = int(snapshot["levels"][i, slot])
                world.initial_food_total = int(snapshot["initial_food_total"][i])
                world._env._gen_valid_moves()
                total = 0.
                for t in range(world.episode_limit):
                    legal = world.avail_actions().astype(bool)
                    actions = [5 if row[5] else 0 for row in legal]
                    actions[2] = int(np.where(legal[2], scores[t, i], -1).argmax())
                    reward, done, _ = world.step(actions)
                    if t == 0:
                        first = reward > 0
                    total += reward
                    if done:
                        break
                records.append([first, total > 0, not world.field.any(), total])
            finally:
                world.close()
        means = np.mean(records, axis=0)
        for name, expected in zip(("first_step_success_rate", "success_rate",
                                   "full_completion_rate", "mean_return"), means):
            self.assertAlmostEqual(report["random_teammate"][name], expected, places=6)
