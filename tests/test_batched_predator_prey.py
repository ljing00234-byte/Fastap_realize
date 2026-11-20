from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch
import yaml

from fastap_agents import load_frozen_agent
from fastap_methods.batched_qmix import BatchedQMIX
from fastap_methods.predator_prey_eval import paired_return_difference
from fastap_methods.single_teammate import independent_test
from fastap_scenarios.torch_predator_prey import TorchPredatorPrey, check_parity

ROOT = Path(__file__).resolve().parents[1]


class BatchedPredatorPreyTests(unittest.TestCase):
    def test_standard_prey_and_physics_match_scalar_world(self):
        self.assertTrue(check_parity("cpu")["passed"])

    def test_multiple_hunters_score_separately(self):
        world = TorchPredatorPrey(1, "cpu")
        world.load_snapshot([[[.14, 0], [-.14, 0], [0, .14], [0, 0], [-.8, .8], [.8, -.8]]],
                            [[[-.8, 0], [.8, 0], [0, -.8], [0, 0]]])
        reward, terminal, truncated = world.step(torch.zeros((1, 3), dtype=torch.long))
        self.assertEqual(reward.item(), 30)
        self.assertTrue(world.last_contacts.all())
        self.assertFalse((terminal | truncated).any())

    def test_trains_exports_and_reloads_third_predator(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        cfg = yaml.safe_load((ROOT / "configs/methods/qmix.yaml").read_text())
        cfg.update(buffer_size=4, batch_size=2)
        actor = yaml.safe_load((ROOT / "configs/agents/drqn.yaml").read_text())
        learner = BatchedQMIX(actor, cfg, device="cpu", seed=4, world_type=TorchPredatorPrey)
        with TemporaryDirectory(dir=ROOT/".tmp") as tmp:
            run = Path(tmp)
            learner.save_checkpoint(run/"checkpoints/0")
            batch = learner.train_batch(TorchPredatorPrey(2, "cpu", seed=6), "random")
            self.assertEqual(batch["optimizer_updates"], 1)
            chosen = learner.save_checkpoint(run/"checkpoints/50")
            report = independent_test(run, chosen, range(3), scene="pp")
            self.assertEqual((report["scene"], report["role"], report["map_seeds"]), ("pp", 2, [0, 1, 2]))
            self.assertEqual(load_frozen_agent(run/"T1").spec.scene_key, "pp")
            for case, contacts in report["per_role_contacts"].items():
                np.testing.assert_array_equal(np.asarray(contacts).sum(1)*10, report["episode_returns"][case])

    def test_paired_confidence_bounds(self):
        a = [10, 30, 20, 50]
        self.assertEqual(paired_return_difference(a, a)["bootstrap_95_ci"], [0, 0])
        self.assertEqual(paired_return_difference(a, [0, 20, 10, 40])["bootstrap_95_ci"], [10, 10])
        with self.assertRaises(ValueError):
            paired_return_difference(a, [0])
