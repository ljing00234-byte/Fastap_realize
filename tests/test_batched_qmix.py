import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path

import torch
from torch.nn import functional as F
import yaml

from fastap_agents import load_frozen_agent
from fastap_agents.batched import agent_inputs
from fastap_methods.batched_qmix import BatchedQMIX, EpisodeReplay, select_actions
from fastap_methods.qmix import PROJECT
from fastap_scenarios.torch_foraging import TorchForaging


class BatchedQMIXTests(unittest.TestCase):
    def make_learner(self, **overrides):
        agent = yaml.safe_load((PROJECT / 'configs/agents/drqn.yaml').read_text())
        method = yaml.safe_load((PROJECT / 'configs/methods/qmix.yaml').read_text())
        method.update(batch_size=3, buffer_size=7, target_update_interval_or_tau=2)
        method.update(overrides)
        return BatchedQMIX(agent, method, device='cpu', seed=29)

    def test_device_native_initialization_and_epsilon_schedule(self):
        before = torch.get_rng_state()
        original = self.make_learner()
        native = self.make_learner(initialize_on_device=True)
        self.assertTrue(torch.equal(torch.get_rng_state(), before))
        for first, second in zip(original.params, native.params):
            torch.testing.assert_close(first, second)
        worlds = [TorchForaging(3, 'cpu', seed=4) for _ in range(2)]
        for _ in range(2):
            result = [learner.train_batch(world, 'random') for learner, world in zip((original, native), worlds)]
            self.assertEqual(result[0]['epsilon'], result[1]['epsilon'])
            self.assertEqual(result[0]['loss'], result[1]['loss'])
        for field in original.replay.data:
            torch.testing.assert_close(original.replay.data[field][:6], native.replay.data[field][:6])

    def test_reward_standardisation_matches_epymarl_without_changing_replay(self):
        learner = self.make_learner(standardise_rewards=True)
        raw_learner = self.make_learner()
        from components.standarize_stream import RunningMeanStd
        reference = RunningMeanStd(shape=(1,), device='cpu')
        episodes = learner._collect(TorchForaging(3, 'cpu', seed=81), 'random', 0)
        # The sampled EPyMARL batch retains shorter episodes' padding up to the longest.
        episodes['filled'][:] = False
        episodes['filled'][0, :3] = True
        episodes['filled'][1, :7] = True
        episodes['filled'][2, :2] = True
        episodes['reward'].zero_()
        episodes['reward'][0, 1] = 1/3
        episodes['reward'][1, 5] = 2/3
        episodes['reward'][2, 0] = 1
        learner.replay.insert(episodes)
        sample = learner.replay.sample(3, learner.replay_rng)
        raw = sample['reward'].clone()
        reference.update(raw[:, :7].unsqueeze(-1))
        expected = (raw - reference.mean) / reference.var.sqrt()
        expected_loss = raw_learner._update({**sample, 'reward': expected})
        actual_loss = learner._update(sample)
        torch.testing.assert_close(actual_loss, expected_loss)
        torch.testing.assert_close(learner.reward_stats.mean, reference.mean)
        torch.testing.assert_close(learner.reward_stats.var, reference.var)
        torch.testing.assert_close(sample['reward'], raw)
        torch.testing.assert_close(learner.replay.data['reward'][:3], episodes['reward'])

    def test_sequence_matches_independent_rollout_and_shares_gradients(self):
        learner = self.make_learner()
        obs = torch.randn(2, 5, 3, 21)
        previous = torch.zeros(2, 5, 3, 6)
        inputs = agent_inputs(learner.spec, obs, previous)
        output = learner.online(inputs)
        hidden = learner.actor.init_hidden().expand(6, -1)
        manual = []
        for x in inputs.unbind(1):
            qs, hidden = learner.actor(x.reshape(6, -1), hidden)
            manual.append(qs.reshape(2, 3, -1))
        expected = torch.stack(manual, 1)
        torch.testing.assert_close(output, expected)
        for result in (output, expected):
            learner.actor.zero_grad()
            result.square().sum().backward()
            grad = learner.actor.rnn.weight_ih.grad
            if result is output:
                seq_grad = grad.clone()
            else:
                torch.testing.assert_close(seq_grad, grad)

    def test_update_matches_epymarl_on_the_same_batch(self):
        learner = self.make_learner()
        from controllers.basic_controller import BasicMAC
        from learners.q_learner import QLearner

        class Batch(dict):
            pass

        sample = learner._collect(TorchForaging(3, 'cpu', seed=3), 'adjacent', 0.6)
        batch = Batch(obs=sample['obs'], state=sample['state'], avail_actions=sample['available'])
        batch.batch_size, batch.max_seq_length = batch['obs'].shape[:2]
        batch.device = 'cpu'
        actions = F.pad(sample['actions'], (0, 0, 0, 1))
        batch['actions'] = actions.unsqueeze(-1)
        batch['actions_onehot'] = F.one_hot(actions, 6).float()
        for key in ('reward', 'terminated', 'filled'):
            batch[key] = F.pad(sample[key], (0, 1)).unsqueeze(-1)
        actor_config = yaml.safe_load((PROJECT / 'configs/agents/drqn.yaml').read_text())
        args = SimpleNamespace(**vars(learner.cfg), **actor_config, n_actions=6,
                               learner_log_interval=2000, use_cuda=False)
        scheme = {'obs': {'vshape': 21}, 'actions_onehot': {'vshape': (6,)}}
        mac = BasicMAC(scheme, {'agents': 3}, args)
        mac.agent.load_state_dict(learner.actor.state_dict())
        stats = {}
        logger = SimpleNamespace(log_stat=lambda name, value, step: stats.update({name: value}))
        reference = QLearner(mac, scheme, logger, args)
        reference.mixer.load_state_dict(learner.mixer.state_dict())
        reference.target_mixer.load_state_dict(learner.target_mixer.state_dict())
        reference.train(batch, t_env=75, episode_num=3)
        loss = learner._update(sample)
        torch.testing.assert_close(loss, torch.tensor(stats['loss']))
        for serial, vector in zip(reference.params, learner.params):
            torch.testing.assert_close(serial.grad, vector.grad, rtol=5e-5, atol=3e-6)

    def test_replay_wrap_and_distinct_sample(self):
        replay = EpisodeReplay(5)
        generator = torch.Generator().manual_seed(3)
        replay.insert({'obs': torch.arange(3)})
        replay.insert({'obs': torch.arange(3, 6)})
        self.assertEqual(set(replay.sample(5, generator)['obs'].tolist()), set(range(1, 6)))

    def test_training_updates_clock_device_and_deployable_checkpoint(self):
        learner = self.make_learner()
        world = TorchForaging(2, 'cpu', seed=1)
        initial = learner.actor.fc1.weight.detach().clone()
        first = learner.train_batch(world, 'adjacent')
        second = learner.train_batch(world, 'adjacent')
        self.assertEqual(first['optimizer_updates'], 0)
        self.assertEqual(second['optimizer_updates'], 2)
        self.assertEqual(second['training_episodes'], 4)
        self.assertEqual(second['training_steps'], first['batch_steps'] + second['batch_steps'])
        self.assertTrue(torch.isfinite(torch.tensor(second['loss'])))
        self.assertFalse(torch.equal(initial, learner.actor.fc1.weight))
        for live, target in zip(learner.actor.parameters(), learner.target_actor.parameters()):
            torch.testing.assert_close(live, target)
        self.assertFalse(any(p.grad is not None for p in learner.target_actor.parameters()))

        world.reset(layout='adjacent')
        obs, mask = world.observations(), world.avail_actions()
        last = torch.zeros_like(mask, dtype=torch.float)
        with tempfile.TemporaryDirectory(dir=PROJECT / '.tmp') as tmp:
            path = learner.save_checkpoint(Path(tmp) / 'first')
            frozen = load_frozen_agent(path)
            greedy, _ = select_actions(frozen.model, frozen.spec, obs, mask, None, last)
            expected, _ = select_actions(learner.actor, learner.spec, obs, mask, None, last)
            self.assertTrue(torch.equal(greedy, expected))
            self.assertTrue(mask.gather(-1, greedy.unsqueeze(-1)).all())
        learner.clear_replay()
        self.assertEqual(learner.train_batch(world, 'approach')['optimizer_updates'], 2)

    def test_fill_mask_keeps_terminal_transition_and_bootstraps_truncation(self):
        # Force lane zero to terminate at step 1 and lane one to time out at 25.
        class MixedEndWorld(TorchForaging):
            def step(self, actions):
                reward, terminated, truncated = super().step(actions)
                self.terminated[0] = True
                return reward, self.terminated, truncated

        learner = self.make_learner()
        sample = learner._collect(MixedEndWorld(2, 'cpu', seed=2), 'random', 1.0)
        self.assertEqual(sample['filled'][0].sum(), 1)
        self.assertTrue(sample['terminated'][0, 0])
        self.assertEqual(sample['filled'][1].sum(), 25)
        self.assertFalse(sample['terminated'][1, -1])
        self.assertEqual(sample['obs'].size(1), 26)
        self.assertTrue(sample['available'][:, :-1].gather(-1, sample['actions'].unsqueeze(-1)).all())


if __name__ == '__main__':
    unittest.main()
