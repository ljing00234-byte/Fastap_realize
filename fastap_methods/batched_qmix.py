"""Device-resident, fixed-horizon QMIX for batched cooperative games."""

from copy import deepcopy
from itertools import chain
from math import isfinite
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from fastap_agents import get_agent_backend
from fastap_agents.batched import agent_inputs, select_actions, SequenceQ
from fastap_scenarios.torch_foraging import TorchForaging
from .qmix import _checkpoint


class EpisodeReplay:
    """Ring of complete episodes on one device, uniformly sampled without replacement."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.data = {}
        self.position = self.size = 0

    def insert(self, episodes):
        n = episodes['obs'].shape[0]
        if n > self.capacity:
            raise ValueError("The replay must fit at least one rollout batch")
        if not self.data:
            self.data = {k: v.new_empty((self.capacity, *v.shape[1:])) for k, v in episodes.items()}
        count = min(self.capacity - self.position, n)
        for name, value in episodes.items():
            buffer = self.data[name]
            buffer[self.position:self.position + count].copy_(value[:count])
            if count < n:
                buffer[:n - count].copy_(value[count:])
        self.position = (self.position + n) % self.capacity
        self.size = min(self.capacity, self.size + n)

    def sample(self, n, generator):
        if not 0 < n <= self.size:
            raise ValueError("Not enough episodes to sample")
        indices = torch.randperm(self.size, device=generator.device, generator=generator)[:n]
        return {name: value.index_select(0, indices) for name, value in self.data.items()}


class BatchedQMIX:
    """One shared team policy; train_batch collects and optimizes one batch of games."""

    def __init__(self, agent_config, method_config, *, device, seed, world_type=TorchForaging):
        from fastap_agents.local_q import _add_epymarl_src
        _add_epymarl_src()
        from components.standarize_stream import RunningMeanStd
        from modules.mixers.qmix import QMixer

        self.device = torch.device(device)
        if self.device.type == 'cuda':
            if self.device.index not in (None, 0):
                raise ValueError("Use the monitored logical cuda:0 device")
            from scripts.gpu2_monitor import require_monitored_cuda
            require_monitored_cuda()
        info = {key: getattr(world_type, key) for key in ("n_agents", "n_actions", "state_shape", "obs_shape")}
        cfg = self.cfg = SimpleNamespace(**method_config)
        cfg.bootstrap_at_time_limit = getattr(cfg, 'bootstrap_at_time_limit', True)
        if (cfg.name, cfg.learner, cfg.mixer, cfg.action_selector) != (
                'qmix', 'q_learner', 'qmix', 'epsilon_greedy'):
            raise ValueError("This trainer requires the epsilon-greedy QMIX learner")
        if cfg.standardise_returns or not cfg.double_q:
            raise ValueError("Batched QMIX requires Double Q and unstandardised returns")
        self.reward_stats = RunningMeanStd(shape=(1,), device=self.device) if cfg.standardise_rewards else None
        if not (1 <= cfg.batch_size <= cfg.buffer_size and cfg.target_update_interval_or_tau >= 1
                and cfg.epsilon_anneal_time > 0 and 0 <= cfg.epsilon_finish <= cfg.epsilon_start <= 1):
            raise ValueError("Invalid QMIX replay, target or exploration settings")
        backend = get_agent_backend(agent_config['agent'])
        self.spec = backend.build_spec(info, agent_config, scene_key=world_type.scene_key,
                                       control=getattr(world_type, 'control', 'team'))
        cfg.n_agents = self.spec.n_agents
        cfg.state_shape = info['state_shape']
        # Historical runs used CPU-seeded initial weights. Device-native runs
        # draw them on the same GPU used for environment and learner tensors.
        cfg.initialize_on_device = getattr(cfg, 'initialize_on_device', False)
        init_device = self.device if cfg.initialize_on_device else torch.device('cpu')
        with torch.random.fork_rng(devices=[init_device] if init_device.type == 'cuda' else []):
            if init_device.type == 'cuda':
                torch.cuda.manual_seed(seed)
            else:
                torch.random.default_generator.manual_seed(seed)
            with init_device:
                self.actor = self.spec.build_network().to(self.device)
                self.mixer = QMixer(cfg).to(self.device)
        self.online = SequenceQ(self.actor)
        self.target_actor = deepcopy(self.actor)
        self.target = SequenceQ(self.target_actor)
        self.target_mixer = deepcopy(self.mixer)
        self.target.requires_grad_(False)
        self.target_mixer.requires_grad_(False)
        self.params = list(chain(self.actor.parameters(), self.mixer.parameters()))
        # Fused CUDA Adam also keeps its update counter on device.
        self.optimizer = torch.optim.Adam(self.params, lr=cfg.lr, fused=self.device.type == 'cuda')
        self.action_rng = torch.Generator(device=self.device).manual_seed(seed + 1)
        self.replay_rng = torch.Generator(device=self.device).manual_seed(seed + 2)
        self.steps = self.episodes = self.updates = 0
        self.clear_replay()

    def clear_replay(self):
        self.replay = EpisodeReplay(self.cfg.buffer_size)

    def save_models(self, directory):
        directory = Path(directory)
        torch.save(self.actor.state_dict(), directory / 'agent.th')
        torch.save(self.mixer.state_dict(), directory / 'mixer.th')
        torch.save(self.optimizer.state_dict(), directory / 'opt.th')
        if self.reward_stats is not None:
            torch.save(dict(mean=self.reward_stats.mean, var=self.reward_stats.var, count=self.reward_stats.count),
                       directory / 'reward_stats.th')

    def save_checkpoint(self, directory):
        directory = Path(directory)
        directory.parent.mkdir(parents=True, exist_ok=True)
        return _checkpoint(self, self.spec, directory.parent, directory.name)

    @torch.no_grad()
    def _collect(self, world, layout, epsilon):
        world.reset(layout=layout)
        trajectory = {k: [] for k in ('obs', 'state', 'available', 'actions', 'reward',
                                      'terminated', 'truncated', 'filled')}
        available = world.avail_actions()
        last = torch.zeros_like(available, dtype=torch.float32)
        hidden = None
        for _ in range(world.episode_limit):
            obs = world.observations()
            trajectory['obs'].append(obs)
            trajectory['state'].append(world.state())
            trajectory['available'].append(available)
            actions, hidden = select_actions(self.actor, self.spec, obs, available, hidden, last,
                                             epsilon=epsilon, generator=self.action_rng)
            active = ~(world.terminated | world.truncated)
            reward, terminated, truncated = world.step(actions)
            trajectory['actions'].append(actions)
            trajectory['reward'].append(reward)
            trajectory['terminated'].append(terminated.clone())
            trajectory['truncated'].append(truncated.clone())
            trajectory['filled'].append(active)
            last = F.one_hot(actions, self.spec.n_actions).float()
            available = world.avail_actions()
        trajectory['obs'].append(world.observations())
        trajectory['state'].append(world.state())
        trajectory['available'].append(available)
        return {name: torch.stack(values, dim=1) for name, values in trajectory.items()}

    def _td_rewards(self, sample):
        rewards = sample['reward']
        if self.reward_stats is None:
            return rewards
        # EPyMARL trims the sampled batch to its longest episode, then updates
        # moments over *all* entries in that rectangle, including shorter games' zero padding.
        # Compute the same moments with a device mask to avoid a host read of the length.
        columns = sample['filled'].any(dim=0)
        count = (columns.sum() * rewards.size(0)).to(torch.float64)
        mean = ((rewards * columns).sum().reshape(1) / count).to(rewards.dtype)
        variance = (((rewards - mean).square() * columns).sum().reshape(1) / (count - 1)).to(rewards.dtype)
        self.reward_stats.update_from_moments(mean, variance, count)
        return (rewards - self.reward_stats.mean) / self.reward_stats.var.sqrt()

    def _update(self, sample):
        actions = sample['actions']
        previous = F.pad(F.one_hot(actions, self.spec.n_actions).float(), (0, 0, 0, 0, 1, 0))
        inputs = agent_inputs(self.spec, sample['obs'], previous)
        q = self.online(inputs)
        chosen = q[:, :-1].gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        values = self.mixer(chosen, sample['state'][:, :-1]).squeeze(-1)
        with torch.no_grad():
            next_actions = q[:, 1:].detach().masked_fill(~sample['available'][:, 1:], -torch.inf).argmax(-1)
            next_q = self.target(inputs)[:, 1:].gather(-1, next_actions.unsqueeze(-1)).squeeze(-1)
            next_values = self.target_mixer(next_q, sample['state'][:, 1:]).squeeze(-1)
            end = sample['terminated']
            if not self.cfg.bootstrap_at_time_limit:
                end = end | sample['truncated']
            targets = self._td_rewards(sample) + self.cfg.gamma * (~end) * next_values
        valid = sample['filled']
        loss = ((values - targets).square() * valid).sum() / valid.sum()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, self.cfg.grad_norm_clip)
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.cfg.target_update_interval_or_tau == 0:
            self.target_actor.load_state_dict(self.actor.state_dict())
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        return loss.detach()

    def train_batch(self, world, layout):
        """Read aggregated scalars on the host only once per batch of completed games."""
        if self.cfg.initialize_on_device:
            steps = torch.full((), self.steps, dtype=torch.float64, device=self.device)
            fraction = (steps / self.cfg.epsilon_anneal_time).clamp_max(1.0)
        else:
            fraction = min(self.steps / self.cfg.epsilon_anneal_time, 1.0)
        epsilon = self.cfg.epsilon_start - (self.cfg.epsilon_start - self.cfg.epsilon_finish) * fraction
        start = perf_counter()
        cuda = self.device.type == 'cuda'
        if cuda:
            rollout_start = torch.cuda.Event(enable_timing=True)
            rollout_end = torch.cuda.Event(enable_timing=True)
            update_end = torch.cuda.Event(enable_timing=True)
            rollout_start.record()
        episodes = self._collect(world, layout, epsilon)
        batch_size = episodes['actions'].shape[0]
        if cuda:
            rollout_end.record()
        collected = perf_counter()
        # Match the serial learner's first update on episode batch_size, then one
        # per newly collected episode (also when this batch crosses warm-up).
        missing = max(0, self.cfg.batch_size - 1 - self.replay.size)
        n_updates = max(0, batch_size - missing)
        self.replay.insert(episodes)
        loss = None
        for _ in range(n_updates):
            sample = self.replay.sample(self.cfg.batch_size, self.replay_rng)
            loss = self._update(sample)
        if cuda:
            update_end.record()
        statistics = [episodes['filled'].sum(), episodes['reward'].sum(1).mean()]
        if loss is not None:
            statistics.append(loss)
        if self.cfg.initialize_on_device:
            statistics.append(epsilon)
        numbers = torch.stack(statistics).detach().cpu().tolist()
        if self.cfg.initialize_on_device:
            epsilon = numbers.pop()
        if not all(isfinite(value) for value in numbers):
            raise FloatingPointError(f'Nonfinite batch statistics at {self.steps} training steps')
        batch_steps = int(numbers[0])
        self.steps += batch_steps
        self.episodes += batch_size
        elapsed = perf_counter() - start
        return dict(training_steps=self.steps, training_episodes=self.episodes, optimizer_updates=self.updates,
                    batch_steps=batch_steps, mean_return=numbers[1], epsilon=epsilon,
                    loss=numbers[2] if loss is not None else None, elapsed_seconds=elapsed,
                    rollout_seconds=(rollout_start.elapsed_time(rollout_end) / 1000
                                     if cuda else collected - start),
                    update_seconds=(rollout_end.elapsed_time(update_end) / 1000
                                    if cuda else elapsed - (collected - start)))
