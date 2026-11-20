"""Vanilla QMIX using the pinned EPyMARL controller, learner, replay, and runner."""

import json
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch
import yaml

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "third_party" / "epymarl" / "src"


class _Metrics:
    def __init__(self, stream):
        self.stream = stream
        self.latest = {}

    def log_stat(self, key, value, t):
        value = float(value)
        if not math.isfinite(value):
            raise FloatingPointError(f"Nonfinite {key} at training step {t}: {value}")
        self.latest[key] = value
        self.stream.write(json.dumps({"step": int(t), "metric": key, "value": value}) + "\n")


def _settings(scene, agent, method, device, seed, teammates):
    shared_keys = agent.keys() & method.keys()
    if shared_keys:
        raise ValueError(f"Agent and method configs both define: {', '.join(sorted(shared_keys))}")
    config = {}
    for name in ("default.yaml", "algs/qmix.yaml"):
        with (SRC / "config" / name).open() as source:
            config.update(yaml.safe_load(source))
    # Training targets and reported returns use the rewards produced by the game.
    config.update(standardise_rewards=False, standardise_returns=False)
    config.update(method)
    config.update(agent)
    if (config["name"], config["learner"], config["mixer"], config["agent_output_type"]) != (
            "qmix", "q_learner", "qmix", "q"):
        raise ValueError("Vanilla QMIX requires a local Q actor, q_learner and qmix mixer")
    if config["standardise_rewards"] or config["standardise_returns"]:
        raise ValueError("This baseline uses the game's raw rewards and returns")
    if not 1 <= config["batch_size"] <= config["buffer_size"]:
        raise ValueError("Expected 1 <= batch_size <= buffer_size")
    for name in ("t_max", "test_interval", "test_nepisode", "save_model_interval"):
        if config[name] < 1:
            raise ValueError(f"{name} must be positive")
    if config["batch_size_run"] != 1:
        raise ValueError("The episode runner collects one full episode at a time")
    config.update(env=scene["env"], runner=scene["runner"], env_args=dict(scene["env_args"], seed=seed))
    if teammates is not None:
        config["env_args"]["teammate_groups"] = tuple(teammates)
    if str(device) not in ("cpu", "cuda", "cuda:0"):
        raise ValueError("Use cpu or the project's bound logical cuda:0")
    config.update(seed=seed, use_cuda=str(device) != "cpu", device=str(device))
    if config["use_cuda"]:
        from scripts.gpu2_monitor import require_monitored_cuda
        require_monitored_cuda()
    return config


def _checkpoint(learner, spec, base, step):
    # Publish the actor metadata, actor/mixer weights and optimizer together.
    destination = base / str(step)
    with TemporaryDirectory(dir=base, prefix=".writing-") as staging:
        learner.save_models(staging)
        spec.save(staging)
        Path(staging).rename(destination)
    return destination


def _fit(runner, args, metrics, checkpoints):
    from components.episode_buffer import ReplayBuffer
    from components.transforms import OneHot
    from fastap_agents import get_agent_backend
    from learners import REGISTRY as learners

    info = runner.get_env_info()
    args.n_agents, args.n_actions, args.state_shape = info["n_agents"], info["n_actions"], info["state_shape"]
    scheme = {
        "obs": {"vshape": info["obs_shape"], "group": "agents"},
        "state": {"vshape": args.state_shape},
        "actions": {"vshape": (1,), "group": "agents", "dtype": torch.long},
        "avail_actions": {"vshape": args.n_actions, "group": "agents", "dtype": torch.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": torch.uint8},
    }
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}
    replay = ReplayBuffer(scheme, groups, args.buffer_size, info["episode_limit"] + 1,
                          preprocess=preprocess, device="cpu" if args.buffer_cpu_only else args.device)
    backend = get_agent_backend(args.agent)
    mac = backend.build_controller(replay.scheme, groups, args)
    runner.setup(scheme, groups, preprocess, mac)
    learner = learners[args.learner](mac, replay.scheme, metrics, args)
    if args.use_cuda:
        learner.cuda()
    control = args.env_args.get("control", "ego")
    spec = backend.build_spec(info, vars(args), scene_key=args.env_args["key"], control=control)
    starting_weights = [p.detach().cpu().clone() for p in mac.parameters()]
    checkpoint = _checkpoint(learner, spec, checkpoints, 0)
    episodes = 0
    last_test = last_save = 0
    while runner.t_env < args.t_max:
        with torch.no_grad():
            episode = runner.run(test_mode=False)
        episodes += 1
        replay.insert_episode_batch(episode)
        step = runner.t_env
        final_episode = step >= args.t_max
        if replay.can_sample(args.batch_size):
            sample = replay.sample(args.batch_size)
            sample = sample[:, :sample.max_t_filled()]
            if sample.device != args.device:
                sample.to(args.device)
            if final_episode:
                learner.log_stats_t = -args.learner_log_interval - 1
            learner.train(sample, step, episodes)
        if final_episode or step - last_test >= args.test_interval:
            # The epsilon-greedy selector draws random numbers even at epsilon=0.
            # An evaluation block must not alter exploration in subsequent training.
            with torch.random.fork_rng(devices=[0] if args.use_cuda else []), torch.no_grad():
                for _ in range(args.test_nepisode):
                    runner.run(test_mode=True)
            last_test = step
            metrics.stream.flush()
        if final_episode or step - last_save >= args.save_model_interval:
            checkpoint = _checkpoint(learner, spec, checkpoints, step)
            last_save = step
    weight_change = max((p.detach().cpu() - start).abs().max().item()
                        for p, start in zip(mac.parameters(), starting_weights))
    return {
        "method": args.name, "scene": args.env_args["key"], "control": control,
        "n_agents": args.n_agents, "seed": args.seed, "training_steps": runner.t_env,
        "training_episodes": episodes, "optimizer_updates": learner.training_steps,
        "weight_change": weight_change, "final_td_loss": metrics.latest.get("loss"),
        "eval_return_mean": metrics.latest["test_return_mean"],
        "checkpoint": str(checkpoint.relative_to(checkpoints.parent)),
    }


def train_qmix(*, scene_config, agent_config, method_config, run_dir, seed=7,
               device="cpu", teammate_groups=None):
    """Train from scratch, logging raw game rewards; stop at an episode boundary.

    Each config is a flat EPyMARL-compatible mapping except for scene.env_args.
    Optional teammate_groups supplies policy factories to an ego-controlled game.
    The configured agent backend builds its controller and checkpoint metadata.
    """
    if not (SRC / "config" / "default.yaml").is_file():
        raise FileNotFoundError(f"Set up the pinned EPyMARL checkout at {SRC.parent}")
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from fastap_scenarios.epymarl import register_epymarl
    from runners import REGISTRY as runners
    register_epymarl()
    config = _settings(scene_config, agent_config, method_config, device, seed, teammate_groups)
    args = SimpleNamespace(**config)
    run_dir = Path(run_dir).resolve()
    if PROJECT not in run_dir.parents:
        raise ValueError("Run outputs must stay inside the project")
    run_dir.mkdir(parents=True, exist_ok=True)
    if any((run_dir / artifact).exists() for artifact in ("summary.json", "metrics.jsonl", "checkpoints")):
        raise FileExistsError(f"Existing training artifacts in {run_dir}; use a fresh run directory")
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir()
    # The upstream replay sampler uses NumPy's process-global RandomState.
    numpy_state = np.random.get_state()
    try:
        np.random.seed(seed)
        with torch.random.fork_rng(devices=[0] if args.use_cuda else []), (run_dir / "metrics.jsonl").open("x") as log:
            torch.random.default_generator.manual_seed(seed)
            if args.use_cuda:
                torch.cuda.manual_seed(seed)
            metrics = _Metrics(log)
            runner = runners[args.runner](args, metrics)
            try:
                summary = _fit(runner, args, metrics, checkpoints)
            finally:
                runner.close_env()
    finally:
        np.random.set_state(numpy_state)
    report = run_dir / "summary.json"
    report.with_suffix(".tmp").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    report.with_suffix(".tmp").replace(report)
    return summary
