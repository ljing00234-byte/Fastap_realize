import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch
import yaml

from fastap_agents import FrozenAgent
from fastap_methods import train_qmix
from fastap_scenarios import TeammateGroup


ROOT = Path(__file__).resolve().parents[1]


class BaselineIntegrationTests(unittest.TestCase):
    def test_trained_third_slot_stays_frozen_during_ego_training(self):
        previous_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous_threads)
        torch.set_num_threads(1)
        scene = yaml.safe_load((ROOT / "configs/scenarios/lbf.yaml").read_text())
        agent = yaml.safe_load((ROOT / "configs/agents/drqn.yaml").read_text())
        method = yaml.safe_load((ROOT / "configs/methods/qmix.yaml").read_text())
        method.update(t_max=50, batch_size=2, buffer_size=4, test_interval=50,
                      test_nepisode=1, save_model_interval=50, learner_log_interval=1)

        def train(control, directory, groups=None):
            config = dict(scene, env_args=dict(scene["env_args"], control=control, n_teammates=1))
            return train_qmix(scene_config=config, agent_config=agent, method_config=method,
                              run_dir=directory, seed=7, device="cpu", teammate_groups=groups)

        tmp_root = ROOT / ".tmp/epymarl"
        tmp_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=tmp_root) as temp:
            team_dir = Path(temp) / "team"
            team = train("team", team_dir)
            frozen = FrozenAgent.load(team_dir / team["checkpoint"])
            self.assertEqual(frozen.spec.n_agents, 3)
            self.assertTrue(frozen.spec.obs_agent_id)
            self.assertFalse(any(parameter.requires_grad for parameter in frozen.model.parameters()))
            weights = {key: value.clone() for key, value in frozen.model.state_dict().items()}

            # Observe the actual inputs while the two ego agents learn with slot 2.
            role_ids = set()

            def record_role(module, inputs):
                vector = inputs[0].reshape(-1, inputs[0].shape[-1])[0]
                role_ids.add(tuple(vector[-3:].tolist()))

            hook = frozen.model.register_forward_pre_hook(record_role)
            try:
                group = TeammateGroup("third_slot", (frozen.policy_factory(2),))
                ego = train("ego", Path(temp) / "ego", [group])
            finally:
                hook.remove()

            self.assertEqual(role_ids, {(0.0, 0.0, 1.0)})
            for key, value in frozen.model.state_dict().items():
                self.assertTrue(torch.equal(weights[key], value), key)
            for result in (team, ego):
                self.assertGreater(result["optimizer_updates"], 0)
                self.assertGreater(result["weight_change"], 0.0)
                self.assertTrue(math.isfinite(result["final_td_loss"]))
                self.assertGreaterEqual(result["eval_return_mean"], 0.0)
                self.assertLessEqual(result["eval_return_mean"], 1.0)
