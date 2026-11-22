"""Compare mixed-partner and T1-only PP ego training on paired held-out starts."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

from gpu2_monitor import OUTPUTS, ROOT, require_monitored_cuda, run_monitored, write_json


PARTNERS = [f"T{i}" for i in range(1, 9)]
DEVICE = "cuda:0"


def read_run(directory, training_partners, uuid):
    import torch
    from fastap_methods.pp_pool_ego_eval import BOARD_GENERATOR

    config, protocol, status, heldout = [
        json.loads((directory / f"{name}.json").read_text())
        for name in ("config", "protocol", "status", "heldout")
    ]
    fixed = training_partners == ["T1"]
    if (config.pop("training_teammate", None) != ("T1" if fixed else None)
            or protocol.pop("training_pool_order", PARTNERS) != training_partners):
        raise ValueError(f"Unexpected training partners in {directory}")
    # The sampling description also legitimately differs between the two runs.
    protocol.pop("teammate_sampling")
    budget = config["method"]["t_max"]
    if ((config["scene"], config["control"], config["method"]["name"], config["gpu_uuid"]) != ("pp", "ego", "qmix", uuid)
            or config["method"]["bootstrap_at_time_limit"] or budget != 2_000_000
            or (status["phase"], status["training_steps"], protocol["total_step_budget"]) != ("complete", budget, budget)
            or protocol["episode_limit"] != 25 or protocol["controlled_slots"] != [0, 1]):
        raise ValueError(f"Expected a complete 2M-step finite-horizon PP ego run in {directory}")
    counts = status["sampling_counts"]
    if ({name for name, count in counts.items() if count > 0} != set(training_partners)
            or sum(counts.values()) != status["training_episodes"]
            or status["training_episodes"] * protocol["episode_limit"] != budget):
        raise ValueError(f"Partner exposure differs from the recorded training protocol in {directory}")
    score = heldout["scores"]["final"]
    maps = protocol["heldout_map_seeds_per_roster"]
    if (status["final_checkpoint"] != f"checkpoints/{budget}" or heldout["primary_checkpoint"] != status["final_checkpoint"]
            or heldout["first_seed"] != score["first_map_seed"] or score["first_map_seed"] != protocol["heldout_first_seed"]
            or heldout["shared_map_seeds_per_roster"] != maps or score["n_maps_per_teammate"] != maps
            or score["n_episodes"] != heldout["episodes_per_policy"] or heldout["episodes_per_policy"] != maps * len(PARTNERS)
            or score["pool_order"] != PARTNERS or [m["id"] for m in protocol["pool_members"]] != PARTNERS
            or score["board_generator"] != BOARD_GENERATOR or score["board_device"] != DEVICE):
        raise ValueError(f"Unexpected held-out checkpoint, teammates or maps in {directory}")
    returns = torch.tensor([score["episode_returns"][name] for name in PARTNERS], dtype=torch.float64, device=DEVICE)
    if returns.shape != (len(PARTNERS), maps) or returns.mean().item() != score["mean_return"]:
        raise ValueError(f"Episode returns disagree with the held-out report in {directory}")
    score["episode_returns"] = returns
    return config, protocol, status, score


def compare(mixed_dir, fixed_dir):
    import torch
    from fastap_methods.pp_pool_ego_eval import compare_ego_returns

    uuid = require_monitored_cuda()
    mixed = read_run(mixed_dir, PARTNERS, uuid)
    fixed = read_run(fixed_dir, ["T1"], uuid)
    if mixed[:2] != fixed[:2] or mixed[2]["optimizer_updates"] != fixed[2]["optimizer_updates"]:
        raise ValueError("Run settings or update counts differ beyond the choice of training partner")
    for file in ("agent.th", "mixer.th"):
        a, b = [torch.load(d / "checkpoints/0" / file, map_location=DEVICE, weights_only=True)
                for d in (mixed_dir, fixed_dir)]
        if a.keys() != b.keys() or any(not torch.equal(a[key], b[key]) for key in a):
            raise ValueError(f"Initial weights differ in {file}")

    groups = {}
    for label, names in (("T1", PARTNERS[:1]), ("T2-T8", PARTNERS[1:]), ("T1-T8", PARTNERS)):
        scores = []
        stats = {}
        for training, run in (("mixed", mixed), ("fixed_T1", fixed)):
            score = run[3]
            returns = score["episode_returns"][[PARTNERS.index(name) for name in names]]
            scores.append({**score, "episode_returns": returns, "pool_order": names})
            stats[training] = {"mean_return": returns.mean().item(), "any_tag_rate": (returns > 0).double().mean().item()}
        groups[label] = dict(partners=names, episodes=int(returns.numel()), **stats,
                             fixed_minus_mixed=compare_ego_returns(scores[1], scores[0], device=DEVICE))
    return dict(mixed_run=str(mixed_dir.relative_to(ROOT)), fixed_run=str(fixed_dir.relative_to(ROOT)),
                training_seed=mixed[0]["seed"], training_steps=mixed[2]["training_steps"],
                training_episodes=mixed[2]["training_episodes"], optimizer_updates=mixed[2]["optimizer_updates"],
                mixed_training_counts=mixed[2]["sampling_counts"], fixed_training_counts=fixed[2]["sampling_counts"],
                first_map_seed=mixed[3]["first_map_seed"], maps_per_partner=mixed[3]["n_maps_per_teammate"],
                groups=groups, per_partner={name: {"mixed": mixed[3]["per_teammate"][name],
                                                  "fixed_T1": fixed[3]["per_teammate"][name]} for name in PARTNERS},
                device=DEVICE, gpu_uuid=uuid,
                uncertainty="Paired bootstrap across map seeds. All partners on a map are kept together. "
                            "Intervals are conditional on the trained policies and their training seed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mixed_run", type=Path)
    parser.add_argument("fixed_run", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    mixed, fixed, output = [(ROOT / path).resolve() for path in (args.mixed_run, args.fixed_run, args.output_dir)]
    if any(OUTPUTS not in p.parents for p in (mixed, fixed, output)) or len({mixed, fixed, output}) != 3:
        parser.error("Choose distinct project output directories")
    if not args.worker:
        for key in ("TMPDIR", "XDG_CACHE_HOME", "TORCH_HOME", "CUDA_CACHE_PATH"):
            if ROOT not in Path(os.environ.get(key, "/")).resolve().parents:
                parser.error("Source scripts/epymarl-env.sh before launching")
        print(f"Output {output}; disk free {shutil.disk_usage(ROOT).free // (1024 ** 3)} GiB", flush=True)
        run_monitored(output, [str(Path(__file__).resolve()), str(mixed), str(fixed), str(output), "--worker"], 1536)
        return
    sys.path.insert(0, str(ROOT))
    report = compare(mixed, fixed)
    write_json(output / "comparison.json", report)
    print(json.dumps(report["groups"], indent=2), flush=True)


if __name__ == "__main__":
    main()
