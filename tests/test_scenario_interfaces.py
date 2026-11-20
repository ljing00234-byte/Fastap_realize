import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch
import yaml

from fastap_scenarios import SCENARIOS, TeammateGroup, make_env, make_world


ROOT = Path(__file__).resolve().parents[1]


class ScenarioInterfaceTests(unittest.TestCase):
    def test_full_team_replays_raw_world_transitions(self):
        for key, scene in SCENARIOS.items():
            for count in scene.teammate_counts:
                with self.subTest(key=key, count=count):
                    env = make_env(key, control="team", seed=27, n_teammates=count)
                    reference = make_env(key, control="team", seed=27, n_teammates=count)
                    env.reset()
                    reference.reset()
                    try:
                        self.assertEqual(env.n_agents, count + 2)
                        for _ in range(env.episode_limit):
                            actions = [int(np.flatnonzero(mask)[-1]) for mask in env.get_avail_actions()]
                            transition = env.step(actions)
                            self.assertEqual(transition, reference.world.step(actions))
                            self.assertEqual(env.last_joint_actions, tuple(actions))
                            np.testing.assert_array_equal(env.get_state(), reference.world.state())
                            np.testing.assert_array_equal(env.get_obs(), reference.world.observations())
                            if transition[1]:
                                break
                    finally:
                        env.close()
                        reference.close()

    def test_teammate_policy_receives_own_view_and_reaches_correct_slot(self):
        for key in SCENARIOS:
            for count in SCENARIOS[key].teammate_counts:
                with self.subTest(key=key, count=count):
                    views = []

                    def act(obs, mask, rng):
                        views.append(obs.copy())
                        return int(np.flatnonzero(mask)[-1])

                    group = TeammateGroup("fixture", (lambda: act,) * count)
                    env = make_env(key, teammate_groups=[group], switch_interval=(1, 1), seed=5)
                    try:
                        env.set_test_mode(True)
                        env.reset()
                        before = env.world.observations()
                        env.step([0, 0])
                        np.testing.assert_array_equal(np.stack(views), before[2:])
                        self.assertEqual(env.last_joint_actions[:2], (0, 0))
                        self.assertEqual(len(env.last_joint_actions), 2 + count)
                        self.assertEqual([e["step"] for e in env.events], [0, 1])
                        self.assertEqual(env.world.current_step, 1)
                    finally:
                        env.close()

    def test_pp_rejects_a_second_teammate_at_construction(self):
        illegal_group = TeammateGroup("two", (lambda: None,) * 2)
        with self.assertRaises(ValueError):
            make_env("pp", teammate_groups=[illegal_group])
        with self.assertRaises(ValueError):
            make_env("pp", control="team", n_teammates=2)
        with self.assertRaises(ValueError):
            make_world("pp").reset(n_teammates=2)

    def test_configs_run_on_epymarl_episode_runner_with_both_control_modes(self):
        # Exercise the real episode/batch path with deterministic legal actions.
        # No learner, optimizer, checkpoint or GPU is needed for this check.
        sys.path.insert(0, str(ROOT / "third_party/epymarl/src"))
        self.addCleanup(sys.path.pop, 0)
        from fastap_scenarios.epymarl import register_epymarl
        from runners import REGISTRY as runners
        register_epymarl()
        for key in SCENARIOS:
            config = yaml.safe_load((ROOT / f"configs/scenarios/{key}.yaml").read_text())
            for mode in ("ego", "team"):
                with self.subTest(key=key, mode=mode):
                    env_args = dict(config["env_args"], seed=9, control=mode)
                    args = SimpleNamespace(env=config["env"], env_args=env_args, batch_size_run=1,
                                           device="cpu", render=False, test_nepisode=1)
                    runner = runners[config["runner"]](args, Mock())
                    try:
                        info = runner.get_env_info()
                        scheme = {
                            "state": {"vshape": info["state_shape"]},
                            "obs": {"vshape": info["obs_shape"], "group": "agents"},
                            "avail_actions": {"vshape": info["n_actions"], "group": "agents"},
                            "actions": {"vshape": 1, "group": "agents", "dtype": torch.long},
                            "reward": {"vshape": 1}, "terminated": {"vshape": 1},
                        }
                        controller = Mock()
                        controller.select_actions.side_effect = (
                            lambda batch, t_ep, **kwargs: batch["avail_actions"][:, t_ep].argmax(dim=-1))
                        runner.setup(scheme, {"agents": info["n_agents"]}, {}, controller)
                        episode = runner.run(test_mode=True)
                        self.assertEqual(runner.t, info["episode_limit"])
                        self.assertEqual(episode["obs"].shape[2], 2 if mode == "ego" else 3)
                        self.assertTrue(torch.isfinite(episode["reward"]).all())
                    finally:
                        runner.close_env()
