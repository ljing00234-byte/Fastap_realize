"""Shared GPU 2 launch lock, memory budget and live monitoring (MiB)."""

import csv
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
EXPECTED_PEAK = 2048
BASELINE_PEAK = 4096
RESERVE = 1024


def now():
    return datetime.now(timezone.utc).isoformat()


def query(kind, fields, *, timeout=8):
    output = subprocess.check_output(
        ["nvidia-smi", "-i", "2", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
        text=True, timeout=timeout,
    )
    return list(csv.reader(output.splitlines(), skipinitialspace=True))


def require_monitored_cuda():
    """Check the launcher's UUID binding before allocating any training tensors."""
    (row,) = query("gpu", "index,uuid")
    index, uuid = row
    if (index != "2" or not uuid.startswith("GPU-") or
            uuid != os.environ.get("FASTAP_MONITORED_GPU_UUID") or
            uuid != os.environ.get("CUDA_VISIBLE_DEVICES")):
        raise RuntimeError("Launch CUDA training through the project's GPU 2 monitor")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected physical GPU 2 as the only visible CUDA device")
    return uuid


def snapshot(expected_uuid=None):
    (row,) = query("gpu", "index,uuid,name,memory.total,memory.free")
    index, uuid, name, total, free = row
    if index != "2" or not uuid.startswith("GPU-") or (expected_uuid and uuid != expected_uuid):
        raise RuntimeError(f"Unexpected GPU 2 identity: {row}")
    processes = {}
    for pid, executable, used in query("compute-apps", "pid,process_name,used_gpu_memory"):
        processes[int(pid)] = {"executable": executable, "used_mib": int(used)}
    return dict(time=now(), physical_index=int(index), uuid=uuid, name=name, total_mib=int(total),
                free_mib=int(free), compute_processes=processes)


def belongs_to_project(pid, executable):
    # Check full command arguments only in memory; never log them.
    proc = Path("/proc") / str(pid)
    if executable.startswith(str(ROOT) + "/"):
        return True
    try:
        if proc.stat().st_uid != os.getuid():
            return False
        cwd = (proc / "cwd").readlink()
        if cwd == ROOT or ROOT in cwd.parents:
            return True
        return (str(ROOT) + "/").encode() in (proc / "cmdline").read_bytes()
    except FileNotFoundError as exc:
        # NVML may briefly list an exited process after its /proc entry is gone.
        # Accept its disappearance only when the device itself confirms it.
        deadline = time.monotonic() + 2.0
        while (remaining := deadline - time.monotonic()) > 0:
            rows = query("compute-apps", "pid", timeout=remaining)
            if all(int(pid_text) != pid for (pid_text,) in rows):
                return False
            time.sleep(min(0.2, remaining))
        raise RuntimeError(f"Cannot establish ownership of GPU process {pid}") from exc
    except PermissionError as exc:
        raise RuntimeError(f"Cannot establish ownership of GPU process {pid}") from exc


def project_processes(state, own_pid=None):
    return {
        pid: info["used_mib"] for pid, info in state["compute_processes"].items()
        if pid == own_pid or belongs_to_project(pid, info["executable"])
    }


def capacity(free, project_used, previous_cap):
    # Add project usage back before measuring external pressure; subtract it only once.
    cap = (free + project_used) // 2
    if previous_cap is not None:
        cap = min(cap, previous_cap)
    return cap, min(free, cap - project_used)


def pending_allocations(processes):
    """Reserve the declared peak even if another worker has only initialized CUDA."""
    reserved = 0
    smoke_script = str(ROOT / "scripts/qmix_lbf_smoke.py").encode()
    baseline_script = str(ROOT / "scripts/run_baseline.py").encode()
    cooperation_script = str(ROOT / "scripts/train_lbf_cooperation.py").encode()
    pool_ego_script = str(ROOT / "scripts/train_lbf_pool_ego.py").encode()
    shared_pool_ego_script = str(ROOT / "scripts/train_pool_ego.py").encode()
    lbf_switch_script = str(ROOT / "scripts/evaluate_lbf_ego_switch.py").encode()
    for pid, used in processes.items():
        try:
            args = (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
        except FileNotFoundError:
            continue
        if any(script in args for script in (baseline_script, cooperation_script, pool_ego_script,
                                             shared_pool_ego_script, lbf_switch_script)):
            flag = b"--expected-peak-mib"
            peak = int(args[args.index(flag) + 1]) if flag in args else BASELINE_PEAK
        elif smoke_script in args:
            peak = EXPECTED_PEAK
        else:
            continue
        reserved += max(0, peak - used)
    return reserved


def write_json(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def stop(child):
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        child.wait()
        return
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def launch(run_dir, lock, worker_args, expected_peak):
    python = ROOT / "envs/epymarl/bin/python"
    if not python.is_file() or os.environ.get("FASTAP_ROOT") != str(ROOT):
        raise RuntimeError("Set up envs/epymarl and source scripts/epymarl-env.sh first")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"Run directory must be empty: {run_dir}")
    state = snapshot()
    project = project_processes(state)
    used_before = sum(project.values())
    ledger = OUTPUTS / "gpu2_budget.json"
    if ledger.exists():
        budget = json.loads(ledger.read_text())
        if budget["uuid"] != state["uuid"]:
            raise RuntimeError("Existing budget refers to another GPU")
    else:
        if project:
            raise RuntimeError("An active project GPU job exists without a recorded shared budget")
        budget = {"uuid": state["uuid"], "initial_snapshot": state}
    cap, headroom = capacity(state["free_mib"], used_before, budget.get("budget_mib"))
    pending = pending_allocations(project)
    headroom -= pending
    if headroom < expected_peak + RESERVE:
        raise RuntimeError(f"Insufficient headroom: {headroom} MiB (project uses {used_before}/{cap} MiB)")
    run_dir.mkdir(parents=True, exist_ok=True)
    budget.update(budget_mib=cap, last_launch_check=state["time"])
    write_json(ledger, budget)
    report = {"initial_snapshot": state, "budget_mib": cap, "estimated_peak_mib": expected_peak,
              "initial_project_processes_mib": project, "initial_headroom_mib": headroom,
              "initial_pending_smoke_allocations_mib": pending,
              "reserve_mib": RESERVE, "sampled_process_peak_mib": 0,
              "sampled_project_peak_mib": used_before, "samples": 0}
    write_json(run_dir / "monitor.json", report)
    cmd = [str(python), *worker_args]
    child = subprocess.Popen(cmd, cwd=ROOT, start_new_session=True,
                             env={**os.environ, "CUDA_VISIBLE_DEVICES": state["uuid"],
                                  "FASTAP_MONITORED_GPU_UUID": state["uuid"]})
    report["pid"] = child.pid
    try:
        while child.poll() is None:
            current = snapshot(state["uuid"])
            processes = project_processes(current, child.pid)
            used = processes.get(child.pid, 0)
            total_used = sum(processes.values())
            if used and not lock.closed:
                lock.close()
            report["samples"] += 1
            report["sampled_process_peak_mib"] = max(report["sampled_process_peak_mib"], used)
            report["sampled_project_peak_mib"] = max(report["sampled_project_peak_mib"], total_used)
            cap = min(cap, json.loads(ledger.read_text())["budget_mib"])
            if total_used > cap or current["free_mib"] < RESERVE:
                raise RuntimeError(f"GPU limit reached: project {total_used}/{cap} MiB, free {current['free_mib']} MiB")
            time.sleep(0.25)
    except BaseException as exc:
        report["error"] = str(exc) or type(exc).__name__
        stop(child)
        raise
    finally:
        report.update(finished_at=now(), returncode=child.returncode)
        write_json(run_dir / "monitor.json", report)
    if child.returncode:
        raise SystemExit(child.returncode)


def run_monitored(run_dir, worker_args, expected_peak):
    run_dir = Path(run_dir).resolve()
    if OUTPUTS not in run_dir.parents:
        raise ValueError("run_dir must be inside project outputs/")
    if expected_peak <= 0:
        raise ValueError("Estimated GPU peak must be positive")
    OUTPUTS.mkdir(exist_ok=True)
    with (OUTPUTS / ".gpu2.lock").open("a") as lock:
        # Serialize starts until this PID becomes visible or exits. A subsequent
        # launch can then count any of this run's expected peak still pending.
        fcntl.flock(lock, fcntl.LOCK_EX)
        signal.signal(signal.SIGTERM, interrupt)
        launch(run_dir, lock, worker_args, expected_peak)


def interrupt(signum, frame):
    raise KeyboardInterrupt(f"Received signal {signum}")
