"""Run under the project's GPU budget monitor, using a fresh directory in outputs/."""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import torch
import yaml


PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "third_party" / "epymarl" / "src"
ENV_KEY = "lbforaging:Foraging-8x8-2p-3f-v1"


def gpu_uuid():
    uuid = subprocess.check_output(
        ["nvidia-smi", "-i", "2", "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    ).strip()
    if not uuid.startswith("GPU-") or os.environ.get("CUDA_VISIBLE_DEVICES") != uuid:
        raise RuntimeError("Bind CUDA_VISIBLE_DEVICES to the full UUID of physical GPU 2.")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("The experiment requires GPU 2 as logical cuda:0.")
    return uuid


def experiment_config(run_dir):
    config = {}
    for path in ["default.yaml", "envs/gymma.yaml", "algs/qmix.yaml"]:
        with (SRC / "config" / path).open() as stream:
            config.update(yaml.safe_load(stream))
    config["env_args"].update(key=ENV_KEY, time_limit=25)
    config.update(
        seed=7, use_cuda=True, buffer_cpu_only=True, batch_size=2, buffer_size=8,
        t_max=224, test_nepisode=2, test_interval=50, log_interval=1,
        runner_log_interval=50, learner_log_interval=1, save_model=True,
        save_model_interval=50, local_results_path=str(run_dir),
    )
    return config


def verify_run(run_dir, info):
    for name in ("loss", "grad_norm", "test_return_mean"):
        if not info.get(name) or not all(math.isfinite(value) for value in info[name]):
            raise RuntimeError(f"Missing or nonfinite {name}: {info.get(name)}")
    checkpoint_files = sorted(
        (run_dir / "models").glob("*/*/agent.th"),
        key=lambda path: int(path.parent.name),
    )
    if len(checkpoint_files) < 2:
        raise RuntimeError("Expected an initial and a trained agent checkpoint.")
    first = torch.load(checkpoint_files[0], map_location="cpu")
    last = torch.load(checkpoint_files[-1], map_location="cpu")
    deltas = [(last[key] - initial).abs().max().item() for key, initial in first.items()]
    if not all(math.isfinite(delta) for delta in deltas) or max(deltas) <= 0:
        raise RuntimeError("Agent checkpoints contain nonfinite or unchanged parameters.")
    return {
        "training_steps_in_env": int(info["episode_T"][-1]),
        "training_episodes": int(info["episode"][-1]),
        "optimizer_updates": len(info["loss"]),
        "final_td_loss": float(info["loss"][-1]),
        "last_test_step": int(info["test_return_mean_T"][-1]),
        "last_test_return_mean": float(info["test_return_mean"][-1]),
        "max_agent_weight_change": max(deltas),
        "compared_checkpoints": [str(path.relative_to(run_dir)) for path in
                                 (checkpoint_files[0], checkpoint_files[-1])],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    run_dir = parser.parse_args().run_dir.resolve()
    if PROJECT / "outputs" not in run_dir.parents:
        parser.error("Choose a dedicated run directory inside project outputs/.")
    run_dir.mkdir(parents=True, exist_ok=True)
    if any((run_dir / name).exists() for name in ("models", "sacred", "summary.json")):
        parser.error("This run directory already contains experiment output.")
    uuid = gpu_uuid()
    torch.set_num_threads(1)
    sys.path.insert(0, str(SRC))
    from main import ex
    from sacred.observers import FileStorageObserver

    config = experiment_config(run_dir)
    ex.add_config(config)
    ex.observers.append(FileStorageObserver.create(str(run_dir / "sacred")))
    run = ex.run()
    if run.status != "COMPLETED":
        raise RuntimeError(f"Sacred reported {run.status}.")
    summary = verify_run(run_dir, run.info)
    peak = torch.cuda.max_memory_allocated()
    if peak <= 0:
        raise RuntimeError("PyTorch did not record any GPU tensor allocations.")
    summary.update(
        gpu_uuid=uuid, logical_device="cuda:0", torch_peak_allocated_bytes=peak,
        seed=config["seed"], env_key=ENV_KEY, use_rnn=config["use_rnn"],
    )
    report = json.dumps(summary, indent=2, allow_nan=False)
    (run_dir / "summary.json").write_text(report + "\n")
    print(report, flush=True)


if __name__ == "__main__":
    main()
