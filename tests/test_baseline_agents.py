import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from fastap_agents import (AgentBackend, AgentSpec, get_agent_backend, load_frozen_agent,
                           register_agent_backend, registry)


ROOT = Path(__file__).resolve().parents[1]


class FrozenAgentTests(unittest.TestCase):
    def test_exported_slots_match_basic_mac_across_an_episode(self):
        sys.path.insert(0, str(ROOT / "third_party/epymarl/src"))
        self.addCleanup(sys.path.pop, 0)
        from components.episode_buffer import EpisodeBatch
        from controllers.basic_controller import BasicMAC

        info = dict(obs_shape=3, n_agents=4, n_actions=5)
        steps = 4
        batch_scheme = {
            "obs": {"vshape": 3, "group": "agents"},
            "avail_actions": {"vshape": 5, "group": "agents"},
            "actions_onehot": {"vshape": (5,), "group": "agents"},
        }
        tmp = ROOT / ".tmp" / "epymarl"
        tmp.mkdir(parents=True, exist_ok=True)
        # The feed-forward setting covers checkpoints from the original smoke runs.
        for recurrent, prev_action, agent_id in [(True, True, True), (False, False, True),
                                                  (True, True, False)]:
            with self.subTest(recurrent=recurrent, prev_action=prev_action, agent_id=agent_id):
                config = dict(agent="rnn", mac="basic_mac", agent_output_type="q",
                              use_rnn=recurrent, obs_last_action=prev_action, obs_agent_id=agent_id,
                              hidden_dim=8, action_selector="epsilon_greedy", evaluation_epsilon=0.,
                              epsilon_start=0., epsilon_finish=0., epsilon_anneal_time=10)
                spec = AgentSpec.from_epymarl(info, config, scene_key="cn", control="team")
                args = SimpleNamespace(**info, **config)
                with torch.random.fork_rng(devices=[]), tempfile.TemporaryDirectory(dir=tmp) as path:
                    torch.manual_seed(17)
                    mac = BasicMAC(batch_scheme, {"agents": 4}, args)
                    mac.save_models(path)
                    spec.save(path)
                    rng_state = torch.get_rng_state()
                    network = load_frozen_agent(path)
                    self.assertTrue(torch.equal(torch.get_rng_state(), rng_state))
                    self.assertEqual(network.spec, spec)
                    self.assertFalse(network.model.training)
                    self.assertTrue(all(not p.requires_grad for p in network.model.parameters()))
                    policies = [network.policy_factory(i)() for i in range(4)]

                    batch = EpisodeBatch(batch_scheme, {"agents": 4}, 1, steps)
                    batch["obs"].normal_()
                    mac.init_hidden(1)
                    for t in range(steps):
                        mask = batch["avail_actions"][0, t]
                        mask.fill_(1)
                        mask[:, t] = 0
                        with torch.no_grad():
                            q = mac.forward(batch, t)[0]
                        greedy = q.masked_fill(mask == 0, -torch.inf).argmax(-1)
                        actions = [policy(batch["obs"][0, t, i], mask[i], None)
                                   for i, policy in enumerate(policies)]
                        self.assertEqual(actions, greedy.tolist())
                        torch.testing.assert_close(torch.cat([p.hidden for p in policies]),
                                                   mac.hidden_states.reshape(4, -1))
                        batch["actions_onehot"][0, t].scatter_(1, greedy[:, None], 1)
                    fresh = network.policy_factory(2)()
                    self.assertEqual(fresh.hidden.count_nonzero().item(), 0)
                    self.assertEqual(fresh.last_action.count_nonzero().item(), 0)
                    with self.assertRaises(ValueError):
                        network.policy_factory(4)
                    with self.assertRaises(ValueError):
                        fresh([0, 0, 0], [0] * 5, None)

    def test_register_independently_parameterized_roles(self):
        # A second EPyMARL architecture stores each role's Q weights separately.
        # Its adapter selects the matching subnetwork after loading the checkpoint.
        from fastap_agents import FrozenAgent

        class SeparateSpec(AgentSpec):
            FORMAT = "test-separate-q-v1"

        def build_spec(info, cfg, *, scene_key, control):
            return SeparateSpec(scene_key=scene_key, control=control, n_agents=info["n_agents"],
                                obs_shape=info["obs_shape"], n_actions=info["n_actions"],
                                obs_last_action=False, obs_agent_id=False, use_rnn=False,
                                hidden_dim=cfg["hidden_dim"])

        def load_roles(path, *, device):
            spec = SeparateSpec.load(path)
            weights = torch.load(Path(path) / "agent.th", map_location="cpu", weights_only=True)
            actors = []
            for slot in range(spec.n_agents):
                prefix = f"agents.{slot}."
                network = spec.build_network()
                network.load_state_dict({k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)})
                actors.append(FrozenAgent(spec, network.to(device)))
            return SimpleNamespace(spec=spec, policy_factory=lambda slot: actors[slot].policy_factory(slot))

        cfg = dict(agent="rnn_ns", mac="basic_mac", agent_output_type="q", use_rnn=False, hidden_dim=4,
                   obs_last_action=False, obs_agent_id=False, action_selector="epsilon_greedy",
                   epsilon_start=0., epsilon_finish=0., epsilon_anneal_time=1)
        info = dict(obs_shape=3, n_agents=2, n_actions=3)
        backend = AgentBackend(SeparateSpec.FORMAT, get_agent_backend("rnn").build_controller,
                               build_spec, load_roles)
        tmp = ROOT / ".tmp" / "epymarl"
        tmp.mkdir(parents=True, exist_ok=True)
        with patch.dict(registry._BACKENDS), tempfile.TemporaryDirectory(dir=tmp) as directory:
            register_agent_backend(cfg["agent"], backend)
            selected = get_agent_backend(cfg["agent"])
            mac = selected.build_controller({"obs": {"vshape": 3}}, {"agents": 2},
                                            SimpleNamespace(**cfg, **info))
            with torch.no_grad():
                for slot, network in enumerate(mac.agent.agents):
                    network.fc2.weight.zero_()
                    network.fc2.bias.zero_()
                    network.fc2.bias[slot] = 1
            mac.save_models(directory)
            selected.build_spec(info, cfg, scene_key="test", control="ego").save(directory)
            actor = load_frozen_agent(directory)
            actions = [actor.policy_factory(i)()([0] * 3, [1] * 3, None) for i in range(2)]
            self.assertEqual(actions, [0, 1])
