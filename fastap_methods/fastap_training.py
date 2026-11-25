"""Fastap's alternating cluster admissions and coordination training (Algorithms 1–3)."""

from collections import deque
from contextlib import closing
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch.nn import functional as F

from fastap_agents import FastapAgent, FastapAgentSpec
from .fastap_crp import TeammateCRP
from .fastap_learner import FastapLearner

ROOT = Path(__file__).resolve().parents[1]
SMOKE_SETTINGS = dict(t_max=200, batch_size=1, buffer_size=32, test_nepisode=2, test_interval=200,
                      save_model_interval=200, group_interval_steps=100, probe_episodes=2, crp_model_steps=2)


@torch.no_grad()
def _episode(env, group, policy, rng, *, epsilon=0., switch_interval=None, probe_egos=()):
    """Store CPU tensors, with one extra observation/state for the final TD target."""
    obs, state, available = env.reset(group, switch_interval=switch_interval)
    t, n, o, a, p = env.episode_limit, env.n_agents, env.obs_shape, env.n_actions, env.max_teammates
    data = dict(obs=np.zeros((t + 1, n, o), np.float32),
                state=np.zeros((t + 1, env.state_shape), np.float32),
                available=np.zeros((t + 1, n, a), bool), actions=np.zeros((t, n), np.int64),
                reward=np.zeros(t, np.float32), terminated=np.zeros(t, bool),
                truncated=np.zeros(t, bool), filled=np.zeros(t, bool),
                teammate_obs=np.zeros((t, p, o), np.float32),
                teammate_actions=np.full((t, p), -1, np.int64))
    data['available'][..., 0] = True
    hidden = None
    if policy is not None:
        device = next(policy.parameters()).device
        previous = torch.zeros(1, n, a, device=device)
    for step in range(t):
        data['obs'][step], data['state'][step], data['available'][step] = obs, state, available
        if probe_egos:
            actions = np.array([act(local, mask, rng) for act, local, mask in zip(probe_egos, obs, available)])
        elif policy is None:
            actions = np.array([rng.choice(np.flatnonzero(mask)) for mask in available])
        else:
            local = torch.as_tensor(data['obs'][step], device=device).unsqueeze(0)
            mask = torch.as_tensor(data['available'][step], device=device).unsqueeze(0)
            greedy, hidden = policy.act(local, previous, mask, hidden)
            actions = greedy[0].cpu().numpy().copy()
            if epsilon:
                for slot in np.flatnonzero(rng.random(n) < epsilon):
                    actions[slot] = rng.choice(np.flatnonzero(available[slot]))
            previous = F.one_hot(torch.as_tensor(actions, device=device), a).float().unsqueeze(0)
        reward, done, info = env.step(actions)
        data['actions'][step] = actions
        data['reward'][step] = reward
        data['filled'][step] = True
        data['truncated'][step] = done and info['episode_limit']
        data['terminated'][step] = done and not info['episode_limit']
        data['teammate_obs'][step] = info['teammate_obs']
        data['teammate_actions'][step] = info['teammate_actions']
        obs, state, available = env.observe()
        if done:
            data['obs'][step + 1], data['state'][step + 1], data['available'][step + 1] = obs, state, available
            data['teammate_mask'] = data['teammate_actions'] >= 0
            return {key: torch.from_numpy(value) for key, value in data.items()}
    raise RuntimeError("Scenario exceeded its episode limit")


def collect_probe_episodes(env, group_index, rng, count, *, use_original_egos=True):
    """Roll out a frozen group with fresh recurrent partners in each stationary episode."""
    if type(count) is not int or count < 1:
        raise ValueError("Probe count must be a positive integer")
    factories = env.groups[group_index].probe_egos if use_original_egos else ()
    episodes = []
    for _ in range(count):
        egos = tuple(factory() for factory in factories)
        episodes.append(_episode(env, group_index, None, rng, probe_egos=egos))
    return episodes


def _batch(episodes):
    return {name: torch.stack([ep[name] for ep in episodes]) for name in episodes[0]}


class _ClusterReplay:
    """Bound the total number of stored episodes and sample equally from each cluster."""

    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.batch_size = batch_size
        self.buffers = {}

    def admit(self, cluster):
        if cluster in self.buffers:
            return
        limit = self.capacity // (len(self.buffers) + 1)
        if limit < self.batch_size:
            raise ValueError("buffer_size must fit batch_size episodes per active cluster")
        self.buffers[cluster] = ()
        self.buffers = {k: deque(v, maxlen=limit) for k, v in self.buffers.items()}

    def sample(self, rng):
        if any(len(v) < self.batch_size for v in self.buffers.values()):
            return None
        episodes = [buffer[i] for buffer in self.buffers.values()
                    for i in rng.choice(len(buffer), size=self.batch_size, replace=False)]
        return _batch(episodes)


def evaluate_fastap(policy, scene, groups, *, episodes, seed, switch_interval, world_options=None):
    """Greedy local execution on paired initial maps and groups, with fresh episode memories."""
    from fastap_scenarios.fastap_rollout import FastapRolloutEnv

    if episodes < 1:
        raise ValueError("Evaluation needs at least one episode")
    scores = {}
    device = next(policy.parameters()).device
    with torch.random.fork_rng(devices=[0] if device.type == "cuda" else []):
        for label, interval in (("stationary", None), ("switching", switch_interval)):
            returns, changes, redraws = [], [], []
            for i in range(episodes):
                # A roster change can consume world random numbers; reseed each map separately.
                episode_seed = seed + i
                rng = np.random.default_rng(episode_seed)
                torch.random.default_generator.manual_seed(episode_seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed(episode_seed)
                with closing(FastapRolloutEnv(scene, groups, seed=episode_seed, world_options=world_options)) as env:
                    data = _episode(env, int(rng.integers(len(groups))), policy, rng, switch_interval=interval)
                    returns.append(data['reward'].sum().item())
                    redraws.append(len(env.events) - 1)
                    changes.append(sum(a['group'] != b['group'] for a, b in zip(env.events, env.events[1:])))
            scores[label] = dict(mean_return=float(np.mean(returns)), episodes=episodes,
                                 mean_group_changes=float(np.mean(changes)), mean_resamples=float(np.mean(redraws)))
    return scores


def _checkpoint(base, step, agent, learner, crp):
    destination = base / str(step)
    with TemporaryDirectory(dir=base, prefix=".writing-") as staging:
        agent.policy.save(staging)
        torch.save(dict(step=step, learner=learner.state_dict(), crp=crp.state_dict()),
                   Path(staging) / "training.pt")
        Path(staging).rename(destination)
    return str(destination.relative_to(base.parent))


def _fit(scene, env, probe_env, agent, cfg, seed, run_dir, log, world_options, group_factory, max_groups):
    replay = _ClusterReplay(cfg['buffer_size'], cfg['batch_size'])
    learner = FastapLearner(agent, **{k: cfg[k] for k in (
        'alpha_gce', 'alpha_lce', 'alpha_rec', 'alpha_mi', 'kappa', 'eta', 'gamma', 'lr',
        'target_interval', 'bootstrap_at_time_limit', 'grad_norm_clip')})
    device = next(agent.parameters()).device
    crp = TeammateCRP(env.state_shape, env.n_actions, env.max_teammates,
                      alpha=cfg['alpha'], device=device, lr=cfg['lr'], decoder=cfg.get('crp_decoder', 'paper'),
                      center_mode=cfg.get('crp_center_mode', 'historical'))
    rng = np.random.default_rng(seed)
    probe_rng = np.random.default_rng(seed + 1)
    start_mode = cfg.get('probe_starts', 'independent')
    start_plans = {}
    if start_mode != 'independent':
        from .lbf_probe_collection import collect_lbf_probes_on_starts
        from .lbf_probe_starts import select_probe_starts
    members = {}
    step = episodes = next_group = probe_steps = last_eval = last_save = 0
    next_round = 0
    losses = {}

    def record(**event):
        log.write(json.dumps(dict(step=step, **event), allow_nan=False) + "\n")
        log.flush()

    while step < cfg['t_max']:
        if step >= next_round and next_group < max_groups:
            for idx in range(next_group, min(max_groups, next_group + cfg['groups_per_round'])):
                if idx == len(env.groups):
                    group = group_factory(idx)
                    env.add_group(group)
                    probe_env.add_group(group)
                group = env.groups[idx]
                name = group.name
                if start_mode == 'independent':
                    probes = collect_probe_episodes(probe_env, idx, probe_rng, cfg['probe_episodes'])
                else:
                    count = len(group.policies)
                    if count not in start_plans:
                        start_plans[count] = select_probe_starts(seed + 1, cfg['probe_episodes'],
                                                                 mode=start_mode, n_teammates=count)
                    probes = collect_lbf_probes_on_starts(group, probe_rng, start_plans[count])
                probe_steps += sum(int(ep['filled'].sum()) for ep in probes)
                cluster = crp.add_group(name, _batch(probes), model_steps=cfg['crp_model_steps'])
                members.setdefault(cluster, []).append(idx)
                replay.admit(cluster)
                returns = [ep['reward'].sum().item() for ep in probes]
                food_counts = {'probe_episodes_with_food': sum(r > 0 for r in returns)} if scene == 'lbf' else {}
                record(event="admit", group=name, cluster=cluster, probe_steps=probe_steps,
                       probe_episodes=len(probes), probe_return_mean=float(np.mean(returns)),
                       probe_ego_policy='original_partners' if group.probe_egos else 'random_legal',
                       model_loss=crp.last_model_loss, **food_counts)
                next_group = idx + 1
            next_round = step + cfg['group_interval_steps']
        # Warm each new cluster's replay, then sample clusters and their members uniformly.
        cold = [c for c, buf in replay.buffers.items() if len(buf) < cfg['batch_size']]
        cluster = int(rng.choice(cold or list(members)))
        group = int(rng.choice(members[cluster]))
        ratio = min(1., step / cfg['epsilon_anneal_time'])
        epsilon = cfg['epsilon_start'] + ratio * (cfg['epsilon_finish'] - cfg['epsilon_start'])
        ep = _episode(env, group, agent.policy, rng, epsilon=epsilon)
        ep['cluster'] = torch.tensor(cluster)
        step += int(ep['filled'].sum())
        episodes += 1
        replay.buffers[cluster].append(ep)
        batch = replay.sample(rng)
        if batch is not None:
            losses = learner.update(batch)
        record(event="episode", group=env.groups[group].name, cluster=cluster, epsilon=epsilon,
               episode_return=ep['reward'].sum().item(), q_updates=learner.q_updates, encoder_updates=learner.updates,
               losses=losses if batch is not None else {})
        final = step >= cfg['t_max']
        if final or step - last_eval >= cfg['test_interval']:
            evaluation = evaluate_fastap(agent.policy, scene, env.groups, episodes=cfg['test_nepisode'],
                                         seed=seed + 1_000_000, switch_interval=cfg['switch_interval'],
                                         world_options=world_options)
            record(event="evaluation", **evaluation)
            last_eval = step
        if final or step - last_save >= cfg['save_model_interval']:
            checkpoint = _checkpoint(run_dir / "checkpoints", step, agent, learner, crp)
            last_save = step
    generated = getattr(group_factory, 'records', [])
    return dict(method="fastap", scene=scene, seed=seed, training_steps=step, training_episodes=episodes,
                probe_steps=probe_steps, optimizer_updates=learner.q_updates + learner.updates,
                q_updates=learner.q_updates, encoder_updates=learner.updates, last_losses=losses, checkpoint=checkpoint,
                clusters={str(k): [env.groups[i].name for i in v] for k, v in members.items()},
                generated_groups=generated, teammate_training_steps=sum(g['training_steps'] for g in generated),
                evaluation=evaluation, **({'probe_start_plans': {str(k): v for k, v in start_plans.items()}}
                                         if start_plans else {}))


def train_fastap(*, scene_config, agent_config, method_config, run_dir, seed=7, device="cpu",
                 teammate_groups=None, group_factory=None, max_groups=None):
    """Use supplied groups or call group_factory(index) when Algorithm 1 admits a new group."""
    from fastap_scenarios.fastap_rollout import FastapRolloutEnv
    from fastap_scenarios.registry import SCENARIOS
    options = scene_config['env_args']
    scene = options['key']
    if options.get('control', 'ego') != 'ego' or agent_config.get('agent') != 'fastap':
        raise ValueError("Fastap requires ego control and a Fastap agent configuration")
    if group_factory is not None:
        if teammate_groups is not None or type(max_groups) is not int or max_groups < 1:
            raise ValueError("Supply a generator with positive max_groups or a fixed teammate pool")
    cfg = {**method_config, **method_config['scenes'][scene]}
    start_mode = cfg.get('probe_starts', 'independent')
    if start_mode not in ('independent', 'shared_random', 'shared_preload') or (
            scene != 'lbf' and start_mode != 'independent'):
        raise ValueError("Shared probe starts require LBF and either shared_random or shared_preload")
    interval = options.get('switch_interval') or cfg['switch_interval']
    cfg['switch_interval'] = tuple(interval)
    if len(interval) != 2 or not 1 <= interval[0] <= interval[1]:
        raise ValueError("switch_interval requires positive inclusive bounds")
    for key in ('t_max', 'batch_size', 'buffer_size', 'test_nepisode', 'test_interval', 'save_model_interval',
                'group_interval_steps', 'groups_per_round', 'probe_episodes', 'crp_model_steps', 'epsilon_anneal_time'):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not 0 <= cfg['epsilon_finish'] <= cfg['epsilon_start'] <= 1:
        raise ValueError("Invalid epsilon schedule")
    if str(device) not in ('cpu', 'cuda', 'cuda:0'):
        raise ValueError("Use cpu or monitored cuda:0")
    cuda = str(device) != 'cpu'
    if cuda:
        from scripts.gpu2_monitor import require_monitored_cuda
        require_monitored_cuda()
    run_dir = Path(run_dir).resolve()
    if ROOT not in run_dir.parents:
        raise ValueError("Keep training outputs inside the project")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'checkpoints').mkdir()
    world_options = options.get('world_options', {})
    if group_factory is None:
        groups = tuple(SCENARIOS[scene].demo_groups if teammate_groups is None else teammate_groups)
        if options.get('n_teammates') is not None:
            groups = tuple(g for g in groups if len(g.policies) == options['n_teammates'])
        max_groups = len(groups)
    else:
        groups = (group_factory(0),)
    with closing(FastapRolloutEnv(scene, groups, seed=seed, world_options=world_options)) as env, \
            closing(FastapRolloutEnv(scene, groups, seed=seed + 1, world_options=world_options)) as probe_env, \
            torch.random.fork_rng(devices=[0] if cuda else []), (run_dir / "metrics.jsonl").open("x") as log:
        torch.random.default_generator.manual_seed(seed)
        if cuda:
            torch.cuda.manual_seed(seed)
        spec = FastapAgentSpec(env.obs_shape, env.state_shape, env.n_actions, env.n_agents, env.max_teammates,
                               **agent_config['context_dims'][scene], hidden_dim=agent_config['hidden_dim'])
        summary = _fit(scene, env, probe_env, FastapAgent(spec).to(device), cfg, seed, run_dir, log,
                       world_options, group_factory, max_groups)
    (run_dir / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary
