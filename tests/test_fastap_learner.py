import copy
import math
import unittest
from unittest.mock import patch

import torch

from fastap_agents.fastap import FastapAgent, FastapAgentSpec
from fastap_methods.fastap_crp import BehaviorModel, TeammateCRP
from fastap_methods.fastap_learner import FastapLearner, _context_loss


def sample():
    # Episode 1 ends a step early; its second teammate slot is absent throughout.
    valid = torch.tensor([[True, True, False], [True, True, True]])
    slots = valid.unsqueeze(-1).expand(-1, -1, 2).clone()
    slots[0, :, 1] = False
    teammate_actions = torch.randint(0, 3, slots.shape).masked_fill(~slots, -1)
    return dict(obs=torch.randn(2, 4, 2, 4), state=torch.randn(2, 4, 5),
                actions=torch.randint(0, 3, (2, 3, 2)), available=torch.ones(2, 4, 2, 3, dtype=torch.bool),
                reward=torch.randn(2, 3), filled=valid, terminated=torch.zeros_like(valid),
                truncated=torch.tensor([[False, True, False], [False, False, True]]),
                teammate_actions=teammate_actions, teammate_obs=torch.randn(2, 3, 2, 4),
                teammate_mask=slots, cluster=torch.tensor([0, 1]))


def learner(**overrides):
    spec = FastapAgentSpec(obs_dim=4, state_dim=5, n_agents=2, n_actions=3,
                           max_teammates=2, local_dim=2, global_dim=3, hidden_dim=8)
    args = dict(alpha_gce=1, alpha_lce=1, alpha_rec=.1)
    return FastapLearner(FastapAgent(spec), **{**args, **overrides})


class FastapLearnerTests(unittest.TestCase):
    def test_context_kernel_gradient_and_moving_average(self):
        # Single encoder, two clusters, one timestep each; no compactness term at eta=0.
        ctx = torch.tensor([[[[0.0]]], [[[1.0]]]], requires_grad=True)
        labels, filled = torch.arange(2), torch.ones(2, 1, dtype=torch.bool)
        loss, centers = _context_loss(ctx, labels, filled, {}, 0, 1)
        self.assertAlmostEqual(loss.item(), -math.log(1 - math.exp(-2)), places=6)
        loss.backward()
        self.assertGreater(ctx.grad[0].item(), 0.)
        self.assertLess(ctx.grad[1].item(), 0.)
        self.assertFalse(centers[1].requires_grad)
        _, updated = _context_loss(ctx + 2, labels, filled, centers, .01, 1)
        self.assertAlmostEqual(updated[1].item(), 2.98, places=5)

    def test_joint_update_masks_and_target_interval(self):
        torch.manual_seed(5)
        first = learner(target_interval=2)
        second = copy.deepcopy(first)
        data = sample()
        changed = copy.deepcopy(data)
        changed['reward'][0, 2] = -1e8
        changed['state'][0, 3] = 1e3
        changed['obs'][0, 3] = 1e3
        changed['actions'][0, 2] = -1
        changed['teammate_obs'][~data['teammate_mask']] = 1e3
        changed['teammate_actions'][~data['teammate_mask']] = 430
        torch.manual_seed(27)
        result = first.update(data)
        torch.manual_seed(27)
        other = second.update(changed)
        self.assertEqual(result, other)
        self.assertAlmostEqual(result['loss'], result['td'] + result['gce'] + result['lce']
                               - .001 * result['mi_bound'] + .1 * result['rec'], places=4)
        for p, q in zip(first.params, second.params):
            torch.testing.assert_close(p, q)
        for network in (first.agent.policy, first.agent.global_encoder, first.agent.variational,
                        first.agent.action_decoders, first.agent.obs_decoders, first.mixer):
            self.assertTrue(any(first.optimizer.state[p]['exp_avg'].abs().sum() > 0 for p in network.parameters()))
        self.assertTrue(any(not torch.equal(target, live) for target, live in
                            zip(first.target_agent.parameters(), first.agent.parameters())))
        first.update(data)
        for target, live in zip(first.target_agent.parameters(), first.agent.parameters()):
            torch.testing.assert_close(target, live)
        restored = learner(target_interval=2)
        restored.load_state_dict(copy.deepcopy(first.state_dict()))
        self.assertEqual(restored.updates, 2)
        self.assertEqual(restored.q_updates, 4)
        torch.manual_seed(90)
        a = first.update(data)
        torch.manual_seed(90)
        b = restored.update(data)
        self.assertEqual(a, b)

    def test_td_receives_raw_rewards_and_respects_timeout(self):
        data = sample()
        # A constant target mixer returns 2 irrespective of sampled contexts or available actions.
        data['reward'] = torch.tensor([[4., 2., 0.], [0., -3., 1.]])
        data['terminated'][1, 0] = True
        data['filled'][1, 1:] = False
        data['available'][:, :, :, 1:] = False
        for bootstrap_time_limit in (True, False):
            model = learner(bootstrap_at_time_limit=bootstrap_time_limit)
            with torch.no_grad():
                for network in (model.mixer, model.target_mixer):
                    for p in network.parameters():
                        p.zero_()
                model.target_mixer.V[-1].bias.fill_(2.)
            # Hold the injected constant mixers fixed through the per-cluster Q steps.
            model.optimizer.param_groups[0]['lr'] = 0.
            result = model.update(data)
            targets = [4 + .99 * 2, 2 + .99 * 2 * bootstrap_time_limit, 0.]
            self.assertAlmostEqual(result['td'], sum(x*x for x in targets)/3, places=5)

    def test_reconstruction_uses_joint_likelihood_of_present_teammates(self):
        model = learner()
        data = sample()
        data['teammate_obs'].zero_()
        with torch.no_grad():
            for heads in (model.agent.obs_decoders, model.agent.action_decoders):
                for parameter in heads.parameters():
                    parameter.zero_()
        # For every ego, a uniform action distribution costs log(3) for the
        # single-member cluster and 2*log(3) for the two-member cluster.
        self.assertAlmostEqual(model.update(data)['rec'], 2 * 3 * math.log(3), places=5)

    def test_q_steps_each_cluster_before_encoder_sweep(self):
        model = learner(target_interval=3)
        data = sample()
        expected_q = {id(p) for p in model.q_params}
        expected_encoder = {id(p) for p in model.encoder_params}
        phases = []

        def record(optimizer, args, kwargs):
            phases.append({id(p) for p in model.params if p.grad is not None})

        with model.optimizer.register_step_post_hook(record):
            model.update(data)
            self.assertEqual(phases, [expected_q, expected_q, expected_encoder])
            phases.clear()
            data['cluster'].zero_()
            model.update(data)
            self.assertEqual(phases, [expected_q, expected_encoder])
        self.assertEqual((model.q_updates, model.updates), (3, 2))
        # Target frequency is in completed encoder sweeps, irrespective of cluster count.
        self.assertTrue(any(not torch.equal(t, p) for t, p in zip(
            model.target_mixer.parameters(), model.mixer.parameters())))


class FastapCRPTests(unittest.TestCase):
    def test_categorical_training_weights_episodes_equally_and_admission_sums_evidence(self):
        torch.manual_seed(24)
        crp = TeammateCRP(5, 6, 2, decoder='categorical', alpha=.5)
        model = crp.model
        with torch.no_grad():
            model.categorical.weight.zero_()
            model.categorical.bias.copy_(torch.arange(12).float().reshape(6, 2).T.flatten() / 2)
        states = torch.randn(2, 4, 5)
        actions = torch.tensor([[[0, 50], [99, 27], [41, 18]], [[5, 4], [2, 1], [4, 3]]])
        slots = torch.ones_like(actions, dtype=torch.bool)
        slots[0, 0, 1] = False
        filled = torch.tensor([[True, False, False], [True, True, True]])
        fields = states, actions, slots, filled
        context = model.encode(*fields)
        self.assertEqual(model.action_logits(states, context).shape, (2, 3, 2, 6))
        log_p = model.categorical.bias.reshape(2, 6).log_softmax(-1)
        short = log_p[0, 0]
        long = (log_p[0, [5, 2, 4]] + log_p[1, [4, 1, 3]]).sum()
        expected_loss = -(short + long / 6) / 2
        torch.testing.assert_close(model.trajectory_log_prob(*fields, context), torch.stack((short, long)))
        torch.testing.assert_close(model.episode_action_nll(*fields, context), -torch.stack((short, long / 6)))
        observed_loss = crp.behavior_loss(*fields, context)
        torch.testing.assert_close(observed_loss, expected_loss)
        torch.testing.assert_close(torch.autograd.grad(observed_loss, model.categorical.bias)[0],
                                   torch.autograd.grad(expected_loss, model.categorical.bias)[0])
        batch = dict(zip(('state', 'teammate_actions', 'teammate_mask', 'filled'), fields))
        crp.add_group('one', batch)
        self.assertAlmostEqual(crp.last_model_loss, expected_loss.item(), places=5)
        restored = TeammateCRP(5, 6, 2, decoder='categorical', alpha=1.)
        restored.load_state_dict(crp.state_dict())
        torch.testing.assert_close(restored.model.action_logits(states, context), model.action_logits(states, context))
        with self.assertRaisesRegex(ValueError, 'decoder'):
            TeammateCRP(5, 6, 2, alpha=.5).load_state_dict(crp.state_dict())

    def test_paper_action_predictions_match_normalized_vertex_densities(self):
        torch.manual_seed(17)
        model = BehaviorModel(5, 6, 2)
        states = torch.randn(3, 4, 5)
        context = torch.randn(3, 16)
        history = torch.cat((states[:, :-1], context[:, None].expand(-1, 3, -1)), -1)
        decoded, _ = model.gru(history)
        mean, raw_scale = model.gaussian(decoded).reshape(3, 3, 2, 2, 6).unbind(2)
        scale = torch.nn.functional.softplus(raw_scale) + 1e-4
        density = torch.distributions.Normal(mean.unsqueeze(-2), scale.unsqueeze(-2))
        scores = density.log_prob(torch.eye(6)).sum(-1)
        torch.testing.assert_close(model.action_logits(states, context).log_softmax(-1), scores.log_softmax(-1))

    def test_padding_and_absence_mask_for_encoder_and_decoder(self):
        torch.manual_seed(21)
        model = BehaviorModel(5, 3, 2).eval()
        data = sample()
        states, acts, slots, valid = [data[key] for key in ('state', 'teammate_actions', 'teammate_mask', 'filled')]
        polluted = [value.clone() for value in (states, acts, slots, valid)]
        polluted[0][0, 3] = 3400
        polluted[1][~slots] = 99
        before = model.encode(states, acts, slots, valid)
        after = model.encode(*polluted)
        torch.testing.assert_close(before, after)
        torch.testing.assert_close(model.trajectory_log_prob(states, acts, slots, valid, before),
                                   model.trajectory_log_prob(*polluted, after))

    def test_posterior_map_uses_updated_candidate_mean_and_counts(self):
        crp = TeammateCRP(5, 3, 2, alpha=.5)
        crp.counts = [2]
        crp.centers = [torch.full((16,), 3.)]
        group = crp._unpack(sample())
        captured = []

        def log_prob(state, action, slot, filled, center):
            captured.append(center[0, 0].item())
            return center.new_full((center.size(0),), 0.)

        vector = torch.zeros(2, 16)
        with patch.object(crp.model, 'encode', return_value=vector), patch.object(
                crp.model, 'trajectory_log_prob', side_effect=log_prob):
            label, center = crp._assign(group)
        self.assertEqual(captured, [2.0, 0.0])
        self.assertEqual(label, 0)
        torch.testing.assert_close(center, torch.full((16,), 2.))
        with patch.object(crp.model, 'encode', return_value=vector), patch.object(
                crp.model, 'trajectory_log_prob', side_effect=lambda *args: -args[-1][:, 0]):
            label, _ = crp._assign(group)
        self.assertEqual(label, 1)

    def test_model_objective_training_and_checkpoint(self):
        torch.manual_seed(6)
        crp = TeammateCRP(5, 3, 2, alpha=.5)
        data = sample()
        from fastap_methods.fastap_crp import _collate
        fields = _collate(crp._unpack(data), 'cpu')
        with torch.no_grad():
            log_p = crp.model.trajectory_log_prob(*fields, crp.model.encode(*fields))
            expected_loss = math.log(2) - torch.logsumexp(log_p, 0).item()
        self.assertEqual(crp.add_group('one', data), 0)
        self.assertAlmostEqual(crp.last_model_loss, expected_loss, places=4)
        self.assertGreater(crp.model.token.weight.grad.abs().sum().item(), 0)
        self.assertGreater(crp.model.gaussian.weight.grad.abs().sum().item(), 0)
        with self.assertRaisesRegex(ValueError, 'already assigned'):
            crp.add_group('one', data)
        restored = TeammateCRP(5, 3, 2, alpha=1)
        legacy = copy.deepcopy(crp.state_dict())
        del legacy['decoder']
        restored.load_state_dict(legacy)
        self.assertEqual(restored.add_group('two', data), crp.add_group('two', data))
        self.assertEqual(sum(restored.counts), 2)
        self.assertEqual(restored.alpha, .5)
        self.assertTrue(all(t.device.type == 'cpu' for probe in restored._probes for t in probe))


if __name__ == '__main__':
    unittest.main()
