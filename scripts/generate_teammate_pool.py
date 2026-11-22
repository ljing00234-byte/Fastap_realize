"""Build a fixed cohort of independent QMIX teammates and evaluate cross-play."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import sleep

from gpu2_monitor import ROOT, write_json


BASE_RUNS = {"lbf": "outputs/lbf_single_T1_horizon25_01", "pp": "outputs/pp_single_T1_horizon25_01",
             "cn": "outputs/cn_single_T1_horizon25_01"}
EVIDENCE = {"lbf": ("evidence_for_any_food", "evidence_for_full_completion"),
            "pp": ("evidence_for_teammate", "evidence_for_team"),
            "cn": ("evidence_for_teammate", "evidence_for_team")}
CROSSPLAY_FIRST_SEED = {"lbf": 870000, "pp": 1510000, "cn": 2010000}


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train_member(pool, name, seed, cfg):
    runs = pool / "runs"
    runs.mkdir(exist_ok=True)
    run = runs / name
    log = runs / (name + ".log")
    attempt = 1
    while run.exists() and any(run.iterdir()):
        monitor_file = run / "monitor.json"
        monitor = read(monitor_file) if monitor_file.is_file() else {}
        if monitor.get("returncode") == 0 and read(run / "status.json")["phase"] == "complete":
            return run
        if monitor.get("returncode") is None:
            raise RuntimeError(f"Check the unfinished GPU monitor before resuming {run}")
        attempt += 1
        run = runs / f"{name}-attempt-{attempt}"
    command = [sys.executable, "-B", str(ROOT / "scripts/train_lbf_cooperation.py"), str(run),
               "--scene", cfg["scene"], "--single-teammate", "--num-envs", str(cfg["num_envs"]),
               "--eval-maps", str(cfg["eval_maps"]),
               "--total-steps", str(cfg["total_step_budget"]), "--heldout-seed", str(cfg["heldout_seed"]),
               "--seed", str(seed)]
    # Every member follows its reference T1, including runs saved before this flag existed.
    bootstrap = cfg["method"].get("bootstrap_at_time_limit", True)
    command.append("--bootstrap-at-time-limit" if bootstrap else "--no-bootstrap-at-time-limit")
    while True:
        with log.open("a+") as stream:
            stream.seek(0, 2)
            offset = stream.tell()
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode == 0:
                return run
            stream.seek(offset)
            detail = stream.read()
        if "RuntimeError: Insufficient headroom:" in detail and not run.exists():
            print(f"{name} waiting for GPU 2 headroom", flush=True)
            sleep(60)
            continue
        raise RuntimeError(f"{name} exited with {result.returncode}. See {log}")


def publish(pool, name, run, seed, base_cfg):
    cfg, status, report = (read(run / f"{key}.json") for key in ("config", "status", "heldout"))
    if (not cfg.get("single_teammate") or cfg["seed"] != seed or status["phase"] != "complete"
            or status["training_steps"] < cfg["total_step_budget"] or report["role"] != 2):
        raise ValueError(f"Incomplete or incompatible source run {run}")
    # Runs predating the explicit time-limit flag used the EPyMARL default.
    if {"bootstrap_at_time_limit": True, **cfg["method"]} != {"bootstrap_at_time_limit": True, **base_cfg["method"]}:
        raise ValueError(f"QMIX settings differ from the reference run in {run}")
    for key in ("scene", "agent", "total_step_budget", "num_envs", "eval_maps", "heldout_seed"):
        if cfg[key] != base_cfg[key]:
            raise ValueError(f"{key} differs from the reference run in {run}")
    source = run / "T1"
    meta = read(source / "role.json")
    chosen = run / status["selected_checkpoint"]
    sha = digest(source / "agent.th")
    if (meta["slot"] != 2 or meta["sha256"] != sha or report["checkpoint"] != status["selected_checkpoint"]
            or digest(chosen / "agent.th") != sha or (chosen/"agent.json").read_bytes() != (source/"agent.json").read_bytes()):
        raise ValueError(f"Frozen actor differs from its evaluated checkpoint in {run}")
    members = pool / "members"
    members.mkdir(exist_ok=True)
    member = members / name
    if not member.exists():
        with TemporaryDirectory(dir=members) as tmp:
            staging = Path(tmp) / name
            staging.mkdir()
            for file in ("agent.th", "agent.json", "role.json"):
                shutil.copyfile(source / file, staging / file)
            staging.rename(member)
    if any(digest(member / f) != digest(source / f) for f in ("agent.th", "agent.json", "role.json")):
        raise ValueError(f"Pool member {name} does not match its evaluated source")
    return dict(id=name, path=str(member.relative_to(pool)), training_seed=seed, sha256=sha,
                source_run=str(run.relative_to(ROOT)), selected_checkpoint=status["selected_checkpoint"],
                training_steps=status["training_steps"], test_maps=report["maps"],
                scores=report["scores"], **{key: report[key] for key in EVIDENCE[cfg["scene"]]})


def generate(pool, base, size):
    cfg = read(base / "config.json")
    seeds = [cfg["seed"] + 101*i for i in range(size)]
    scene = cfg["scene"]
    protocol = dict(scene=scene, target_size=size, teammate_slot=2, base_run=str(base.relative_to(ROOT)),
                    training_seeds=seeds, per_member_steps=cfg["total_step_budget"], eval_maps=cfg["eval_maps"],
                    num_envs=cfg["num_envs"], member_test_seed=cfg["heldout_seed"],
                    crossplay_first_seed=CROSSPLAY_FIRST_SEED[scene], crossplay_maps=256)
    protocol["bootstrap_at_time_limit"] = cfg["method"].get("bootstrap_at_time_limit", True)
    if scene == "cn":
        reference_protocol = read(base / "protocol.json")
        for key in ("episode_limit", "n_egos", "n_teammates", "n_landmarks", "collision_penalty",
                    "primary_endpoint", "secondary_endpoint"):
            protocol[key] = reference_protocol[key]
    previous = pool / "protocol.json"
    if previous.is_file():
        recorded = read(previous)
        if scene == "lbf":
            recorded.setdefault("bootstrap_at_time_limit", True)
        if recorded != protocol:
            raise ValueError("Existing pool has a different protocol")
    else:
        write_json(previous, protocol)
    manifest_path = pool / "pool.json"
    manifest = (read(manifest_path) if manifest_path.is_file() else
                dict(format=f"{scene}-teammate-pool-v1", scene=scene, role=2, target_size=size, members=[]))
    current = None
    try:
        for number, seed in enumerate(seeds, start=1):
            current = f"T{number}"
            write_json(pool / "status.json", dict(phase="generating", current=current,
                                                  finished=len(manifest["members"]), target=size))
            print(f"Starting {current}, seed {seed}", flush=True)
            run = base if number == 1 else train_member(pool, current, seed, cfg)
            if scene == "cn" and read(run / "protocol.json") != {**reference_protocol, "training_seed": seed}:
                raise ValueError(f"CN training or evaluation protocol differs from the reference in {run}")
            entry = publish(pool, current, run, seed, cfg)
            if number <= len(manifest["members"]):
                if entry != manifest["members"][number - 1]:
                    raise ValueError(f"Existing pool entry differs from verified source for {current}")
                continue
            if entry["sha256"] in {m["sha256"] for m in manifest["members"]}:
                raise ValueError(f"Duplicate actor weights for {current}")
            manifest["members"].append(entry)
            write_json(manifest_path, manifest)
            print(f"Published {current}, mean return {entry['scores']['trained']['mean_return']:.3f}", flush=True)
        from fastap_methods.pool_crossplay import evaluate_pool
        write_json(pool / "status.json", dict(phase="crossplay", finished=size, target=size))
        write_json(pool / "crossplay.json", evaluate_pool(pool, manifest["members"],
                                                         protocol["crossplay_first_seed"], protocol["crossplay_maps"]))
        counts = {key.removeprefix("evidence_for_") + "_confirmed": sum(m[key] for m in manifest["members"])
                  for key in EVIDENCE[scene]}
        write_json(pool / "status.json", dict(phase="complete", finished=size, target=size, **counts))
    except BaseException as exc:
        write_json(pool / "status.json", dict(phase="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                                              current=current, finished=len(manifest["members"]), error=repr(exc)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--scene", choices=BASE_RUNS, default="lbf")
    parser.add_argument("--base-run", type=Path)
    parser.add_argument("--size", type=int, default=8)
    args = parser.parse_args()
    pool = (ROOT / args.run_dir).resolve()
    if args.size < 2 or ROOT / "outputs" not in pool.parents:
        parser.error("Choose separate project output directories and at least two members")
    if os.environ.get("FASTAP_ROOT") != str(ROOT):
        parser.error("Source scripts/epymarl-env.sh before launch")
    pool.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT))
    with (pool / ".pool.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        existing = pool / "protocol.json"
        base_path = args.base_run or (read(existing)["base_run"] if existing.is_file() else BASE_RUNS[args.scene])
        base = (ROOT / base_path).resolve()
        if ROOT / "outputs" not in base.parents or base == pool or base in pool.parents or pool in base.parents:
            parser.error("Choose separate project output directories for the pool and its reference run")
        if not (base / "config.json").is_file():
            parser.error(f"Train the reference T1 in {base} first, or supply its directory with --base-run")
        if read(base / "config.json")["scene"] != args.scene:
            parser.error("The reference run must match --scene")
        if (pool / "status.json").is_file() and read(pool / "status.json")["phase"] == "complete":
            if read(pool / "protocol.json")["target_size"] != args.size or read(pool / "protocol.json")["base_run"] != str(base.relative_to(ROOT)):
                parser.error("Completed pool belongs to a different experiment")
            print(f"Completed pool: {pool / 'pool.json'}", flush=True)
            return
        generate(pool, base, args.size)


if __name__ == "__main__":
    main()
