"""Train Fastap with frozen or online-generated teammates and evaluate stationary and switching teams."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

from gpu2_monitor import BASELINE_PEAK, OUTPUTS, ROOT, require_monitored_cuda, run_monitored
from run_baseline import load_yaml, project_path


PYTHON = ROOT / "envs/epymarl/bin/python"


def check_storage(args):
    if os.environ.get("FASTAP_ROOT") != str(ROOT):
        raise ValueError("Source scripts/epymarl-env.sh before starting Fastap")
    if not PYTHON.is_file() or ROOT not in PYTHON.resolve().parents:
        raise ValueError("Expected the project envs/epymarl interpreter")
    for key in ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "PYTHONPYCACHEPREFIX", "TORCH_HOME",
                "TORCH_EXTENSIONS_DIR", "CUDA_CACHE_PATH", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
        value = os.environ.get(key)
        if not value or not Path(value).is_absolute() or ROOT not in Path(value).resolve().parents:
            raise ValueError(f"Expected a project-local {key} from scripts/epymarl-env.sh")
    if OUTPUTS not in args.run_dir.parents:
        raise ValueError("Choose a run directory under project outputs/")
    for path in (args.teammate_pool, args.agent_config, args.method_config):
        if path is not None and (ROOT not in path.parents or not path.is_file()):
            raise ValueError(f"Expected a project-local input file: {path}")
    if args.run_dir.exists():
        allowed = {"monitor.json"} if args.worker else set()
        if not args.run_dir.is_dir() or any(p.name not in allowed for p in args.run_dir.iterdir()):
            raise ValueError(f"Run directory must be fresh: {args.run_dir}")
    existing = next(path for path in (args.run_dir, *args.run_dir.parents) if path.exists())
    free = shutil.disk_usage(existing).free
    if free == 0:
        raise ValueError(f"No free space for {args.run_dir}")
    return free


def train(args, free_bytes):
    import torch
    from fastap_methods.fastap_training import SMOKE_SETTINGS, train_fastap
    from fastap_scenarios.fastap_rollout import load_fastap_groups

    cuda = args.device == "cuda"
    uuid = require_monitored_cuda() if cuda else None
    device = "cuda:0" if cuda else "cpu"
    scene = load_yaml(ROOT / f"configs/scenarios/{args.scene}.yaml")
    agent = load_yaml(args.agent_config)
    method = load_yaml(args.method_config)
    if method.get("name") != "fastap":
        raise ValueError("--method-config must specify name: fastap")
    overrides = dict(SMOKE_SETTINGS) if args.smoke else {}
    if args.steps is not None:
        overrides["t_max"] = args.steps
    if args.crp_decoder is not None:
        overrides["crp_decoder"] = args.crp_decoder
    if args.crp_center_mode is not None:
        overrides["crp_center_mode"] = args.crp_center_mode
    if args.probe_starts is not None:
        overrides["probe_starts"] = args.probe_starts
    method.update(overrides)
    # Command-line overrides take precedence over scene-specific settings as well.
    specific = method["scenes"][args.scene]
    specific.update({key: value for key, value in overrides.items() if key in specific})
    torch.set_num_threads(1)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.teammate_source == "pool":
        pool_hash = hashlib.sha256(args.teammate_pool.read_bytes()).hexdigest()
        groups = load_fastap_groups(args.scene, args.teammate_pool)
        if hashlib.sha256(args.teammate_pool.read_bytes()).hexdigest() != pool_hash:
            raise ValueError("The teammate pool changed while its actors were loading")
        population_args = dict(teammate_groups=groups)
        source_config = dict(teammate_pool=str(args.teammate_pool.relative_to(ROOT)),
                             teammate_pool_sha256=pool_hash, teammate_groups=[group.name for group in groups])
    else:
        from fastap_methods.fastap_population import OnlineTeammates

        factory = OnlineTeammates(scene, args.run_dir, seed=args.seed, device=device,
                                  steps_per_group=args.teammate_training_steps)
        population_args = dict(teammate_groups=None, group_factory=factory, max_groups=args.online_groups)
        source_config = dict(online_groups=args.online_groups, teammate_training_steps=args.teammate_training_steps)
    config = dict(scene=scene, agent=agent, method=method, seed=args.seed, smoke=args.smoke,
                  device=device, gpu_uuid=uuid, free_disk_bytes_before_training=free_bytes,
                  agent_config=str(args.agent_config.relative_to(ROOT)),
                  method_config=str(args.method_config.relative_to(ROOT)), teammate_source=args.teammate_source,
                  **source_config)
    with (args.run_dir / "config.json").open("x", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2, allow_nan=False)
        stream.write("\n")
    summary = train_fastap(scene_config=scene, agent_config=agent, method_config=method,
                           run_dir=args.run_dir, seed=args.seed, device=device, **population_args)
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=project_path, help="fresh directory under project outputs/")
    parser.add_argument("--scene", required=True, choices=("lbf", "pp", "cn"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda", help="cuda uses monitored physical GPU 2")
    parser.add_argument("--smoke", action="store_true", help="200-step run with smaller training batches")
    parser.add_argument("--steps", type=int, help="override the ego training step budget")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--crp-decoder", choices=("paper", "categorical"), help="override the CRP behavior decoder")
    parser.add_argument("--crp-center-mode", choices=("historical", "current_encoder"),
                        help="how to compute existing cluster centers for CRP admission")
    parser.add_argument("--probe-starts", choices=("independent", "shared_random", "shared_preload"),
                        help="LBF initial maps for collecting clustering trajectories")
    parser.add_argument("--teammate-source", choices=("pool", "online"), default="pool")
    parser.add_argument("--teammate-pool", type=project_path)
    parser.add_argument("--online-groups", type=int, default=8, help="maximum number of groups to train online")
    parser.add_argument("--teammate-training-steps", type=int, help="QMIX steps per online group (2M, smoke 125)")
    parser.add_argument("--agent-config", type=project_path, default=ROOT / "configs/agents/fastap.yaml")
    parser.add_argument("--method-config", type=project_path, default=ROOT / "configs/methods/fastap.yaml")
    parser.add_argument("--expected-peak-mib", type=int, default=BASELINE_PEAK, help="reserve this GPU process peak")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.probe_starts is not None and args.scene != "lbf":
        parser.error("--probe-starts applies to LBF")
    if args.teammate_source == "online" and args.teammate_pool is not None:
        parser.error("--teammate-pool requires --teammate-source pool")
    if args.teammate_source == "pool" and args.teammate_pool is None:
        args.teammate_pool = project_path(f"outputs/{args.scene}_teammate_pool_01/pool.json")
    if args.teammate_training_steps is None:
        args.teammate_training_steps = 125 if args.smoke else 2_000_000
    if min(args.online_groups, args.teammate_training_steps) < 1:
        parser.error("Online group count and teammate training steps must be positive")
    if args.seed < 0 or (args.steps is not None and args.steps < 1) or args.expected_peak_mib < 1:
        parser.error("Use a nonnegative seed, positive step budget and positive estimated GPU peak")
    if args.worker and args.device != "cuda":
        parser.error("--worker is reserved for the GPU 2 monitor")
    try:
        free_bytes = check_storage(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.device == "cuda" and not args.worker:
        worker_args = [str(Path(__file__).resolve()), "--worker", "--expected-peak-mib",
                       str(args.expected_peak_mib), *sys.argv[1:]]
        run_monitored(args.run_dir, worker_args, args.expected_peak_mib)
        return
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        if Path(sys.prefix).resolve() != PYTHON.parent.parent:
            os.execv(str(PYTHON), [str(PYTHON), "-B", str(Path(__file__).resolve()), *sys.argv[1:]])
    sys.path.insert(0, str(ROOT))
    print(f"Fastap {args.scene} on {args.device}, output {args.run_dir}, free {free_bytes // (1024**3)} GiB",
          flush=True)
    train(args, free_bytes)


if __name__ == "__main__":
    main()
