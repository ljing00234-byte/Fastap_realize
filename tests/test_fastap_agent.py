import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

from fastap_agents import FastapAgent, FastapAgentSpec, FastapPolicy


class FastapAgentTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(32)
        self.spec = FastapAgentSpec(obs_dim=3, state_dim=5, n_actions=4, n_agents=2,
                                    max_teammates=2, local_dim=4, global_dim=6, hidden_dim=8)
        self.agent = FastapAgent(self.spec)
        self.obs = torch.randn(3, 5, 2, 3)
        self.state = torch.randn(3, 5, 5)
        self.previous = F.one_hot(torch.randint(4, (3, 5, 2)), 4).float()
        self.previous[:, 0].zero_()

    def test_local_sequences_are_causal_and_each_role_runs_in_isolation(self):
        joint = self.agent.local_sequence(self.obs, self.previous, sample=False)
        single = self.agent.policy.roles[0](self.obs[:, :, 0], self.previous[:, :, 0], sample=False)
        torch.testing.assert_close(single.q_values, joint.q_values[:, :, 0])
        changed = self.obs.clone()
        changed[:, 3:] += 87
        changed[:, :, 1] -= 48
        other = self.agent.local_sequence(changed, self.previous, sample=False)
        torch.testing.assert_close(joint.context[:, :3, 0], other.context[:, :3, 0])
        torch.testing.assert_close(joint.q_values[:, :3, 0], other.q_values[:, :3, 0])

        hidden = None
        masks = torch.ones_like(self.previous, dtype=torch.bool)
        masks[..., 2] = False
        for t in range(self.obs.shape[1]):
            actions, hidden = self.agent.act(self.obs[:, t], self.previous[:, t], masks[:, t],
                                             hidden, sample=False)
            expected = joint.q_values[:, t].masked_fill(~masks[:, t], -torch.inf).argmax(-1)
            torch.testing.assert_close(actions, expected)
        torch.testing.assert_close(hidden.encoder, joint.hidden.encoder)
        torch.testing.assert_close(hidden.q_network, joint.hidden.q_network)
        masks[0, 0, 0] = False
        with self.assertRaises(ValueError):
            self.agent.act(self.obs[:, 0], self.previous[:, 0], masks[:, 0])

    def test_context_and_auxiliary_paths_preserve_gradients(self):
        local = self.agent.local_sequence(self.obs, self.previous)
        q_params = {id(p) for p in self.agent.q_parameters()}
        self.assertEqual(q_params, {id(p) for name, p in self.agent.named_parameters()
                                    if name.startswith("policy.roles.") and ".q_" in name})
        z = self.agent.global_sequence(self.state, self.previous)
        conditional, obs, logits = self.agent.auxiliary(local.context, local.history, z)
        self.assertEqual(z.shape, (3, 5, 6))
        self.assertEqual(obs.shape, (3, 5, 2, 2, 3))
        self.assertEqual(logits.shape, (3, 5, 2, 2, 4))
        self.assertEqual(conditional.batch_shape, local.context.shape)
        self.assertTrue((local.std > 0).all())
        self.assertFalse(torch.equal(local.context, local.mean))
        loss = (local.q_values.square().mean() - conditional.log_prob(local.context).mean()
                + obs.square().mean() + logits.square().mean())
        loss.backward()
        for name, parameter in self.agent.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        changed = self.state.clone()
        changed[:, 2:] += 30
        torch.testing.assert_close(self.agent.global_sequence(self.state, self.previous, sample=False)[:, :2],
                                   self.agent.global_sequence(changed, self.previous, sample=False)[:, :2])

    def test_local_checkpoint_runs_without_central_components(self):
        tmp = Path(__file__).resolve().parents[1] / ".tmp"
        tmp.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp) as directory:
            path = self.agent.policy.save(directory)
            payload = torch.load(path, weights_only=True)
            self.assertTrue(all(name.startswith("roles.") for name in payload["weights"]))
            rng_before = torch.get_rng_state()
            policy = FastapPolicy.load(directory)
            self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
            self.assertFalse(policy.training)
            self.assertFalse(any(parameter.requires_grad for parameter in policy.parameters()))
            reference = self.agent.local_sequence(self.obs, self.previous, sample=False)
            restored = policy(self.obs, self.previous, sample=False)
            torch.testing.assert_close(restored.q_values, reference.q_values)
            torch.testing.assert_close(restored.context, reference.context)
