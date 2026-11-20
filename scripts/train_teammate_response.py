"""Train one LBF teammate, freeze it, then train a joint response for both egos."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def train(directory, control, steps, seed, teammate=None):
    command = [sys.executable, "-B", str(ROOT / "scripts/run_baseline.py"), str(directory),
               "--scene", "lbf", "--control", control, "--teammates", "1",
               "--steps", str(steps), "--seed", str(seed)]
    if teammate is not None:
        command.extend(["--teammate-checkpoint", str(teammate)])
    with (directory.parent / f"{directory.name}.log").open("x") as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    summary = json.loads((directory / "summary.json").read_text())
    if summary["optimizer_updates"] <= 0 or summary["weight_change"] <= 0:
        raise RuntimeError(f"No learned policy in {directory}")


def select_checkpoint(directory, seeds, teammate=None):
    from fastap_methods.response_eval import evaluate_pair

    candidates = sorted((p for p in (directory / "checkpoints").iterdir()
                         if p.is_dir() and p.name.isdigit() and int(p.name) > 0),
                        key=lambda p: int(p.name))
    scores = []
    for path in candidates:
        result = evaluate_pair(path, path if teammate is None else teammate, seeds)
        scores.append({"checkpoint": path.name, "mean_return": result["mean_return"],
                       "success_rate": result["success_rate"]})
    # Prefer the later snapshot when validation returns tie.
    best = max(scores, key=lambda item: (item["mean_return"], int(item["checkpoint"])))
    write_json(directory / "selection.json", {"map_seeds": list(seeds), "scores": scores,
                                             "selected": best["checkpoint"]})
    return directory / "checkpoints" / best["checkpoint"]


def policy_digest(directory):
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ("agent.json", "agent.th")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--team-steps", type=int, default=200000)
    parser.add_argument("--response-steps", type=int, default=200000)
    parser.add_argument("--team-seed", type=int, default=101)
    parser.add_argument("--response-seed", type=int, default=202)
    args = parser.parse_args()
    directory = (ROOT / args.run_dir).resolve()
    if ROOT / "outputs" not in directory.parents:
        parser.error("Choose a new experiment directory inside outputs/")
    if min(args.team_steps, args.response_steps) < 1:
        parser.error("Both training budgets must be positive")
    sys.path.insert(0, str(ROOT))
    import torch
    from fastap_methods.response_eval import evaluate_pair

    torch.set_num_threads(1)
    directory.mkdir(parents=True, exist_ok=False)
    team, response = directory / "team", directory / "response"
    validation_seeds = range(10000, 10064)
    test_seeds = range(20000, 20128)
    status = dict(scene="lbf", teammate_slot=2, ego_slots=[0, 1],
                  team_steps=args.team_steps, response_steps=args.response_steps,
                  team_seed=args.team_seed, response_seed=args.response_seed)

    def phase(name, **updates):
        status.update(phase=name, **updates)
        write_json(directory / "status.json", status)
        print(name, flush=True)

    try:
        phase("training_team")
        train(team, "team", args.team_steps, args.team_seed)
        phase("selecting_teammate")
        teammate = select_checkpoint(team, validation_seeds)
        digest = policy_digest(teammate)
        phase("training_best_response", teammate=str(teammate.relative_to(directory)),
              teammate_sha256=digest)
        train(response, "ego", args.response_steps, args.response_seed, teammate)
        if policy_digest(teammate) != digest:
            raise RuntimeError("The frozen teammate checkpoint changed during response training")
        phase("selecting_response")
        best_response = select_checkpoint(response, validation_seeds, teammate)
        phase("evaluating", best_response=str(best_response.relative_to(directory)))
        report = {"map_seeds": list(test_seeds),
                  "original_partners": evaluate_pair(teammate, teammate, test_seeds),
                  "learned_response": evaluate_pair(best_response, teammate, test_seeds),
                  "original_partners_random_teammate": evaluate_pair(
                      teammate, teammate, test_seeds, random_teammate=True)}
        report["response_minus_original_mean"] = (
            report["learned_response"]["mean_return"] - report["original_partners"]["mean_return"])
        write_json(directory / "heldout.json", report)
        phase("complete")
    except Exception as exc:
        phase("failed", failed_phase=status.get("phase"), error=str(exc))
        raise


if __name__ == "__main__":
    main()
