"""Compose a game, local agent and training method; run them on monitored GPU 2."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

from gpu2_monitor import BASELINE_PEAK, ROOT, run_monitored


def project_path(value):
    """Use the same path base for the interactive launcher and its worker."""
    return (ROOT / value).resolve()


def load_yaml(path):
    import yaml

    with path.open(encoding="utf-8") as source:
        return yaml.safe_load(source)


def pool_paths(manifest_path, scene):
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("format"), manifest.get("scene"), manifest.get("role")) != (
            f"{scene}-teammate-pool-v1", scene, 2) or scene not in ("lbf", "pp"):
        raise ValueError("Incompatible teammate pool for this scene")
    paths = []
    for member in manifest["members"]:
        path = (manifest_path.parent / member["path"]).resolve()
        if manifest_path.parent not in path.parents or path in paths:
            raise ValueError("Invalid or duplicate member path")
        if hashlib.sha256((path / "agent.th").read_bytes()).hexdigest() != member["sha256"]:
            raise ValueError(f"Frozen actor checksum mismatch for {member['id']}")
        paths.append(path)
    if not paths:
        raise ValueError("The pool does not have a published member yet")
    return paths


def frozen_groups(paths, scene):
    from fastap_agents import load_frozen_agent
    from fastap_scenarios import SCENARIOS, TeammateGroup, make_world

    key = scene["env_args"]["key"]
    game = SCENARIOS[key]
    world = make_world(key, **scene["env_args"].get("world_options", {}))
    try:
        n_actions, obs_shape = world.n_actions, world.obs_shape
    finally:
        world.close()
    groups = []
    for path in paths:
        agent = load_frozen_agent(path)
        spec = agent.spec
        if spec.scene_key != key or spec.control != "team":
            raise ValueError(f"{path} must contain an all-team checkpoint from {key}")
        count = game.check_count(spec.n_agents - 2)
        if spec.n_actions != n_actions or spec.obs_shape != obs_shape:
            raise ValueError(f"Incompatible observation or action dimensions in {path}")
        name = str(path.relative_to(ROOT))
        factories = tuple(agent.policy_factory(slot) for slot in range(2, 2 + count))
        groups.append(TeammateGroup(name, factories))
    return groups


def train(args):
    sys.path.insert(0, str(ROOT))
    import torch
    from fastap_methods import TRAINERS
    from gpu2_monitor import require_monitored_cuda

    uuid = require_monitored_cuda()
    torch.set_num_threads(1)
    scene = load_yaml(ROOT / "configs/scenarios" / f"{args.scene}.yaml")
    agent = load_yaml(args.agent_config)
    method = load_yaml(args.method_config)
    if args.control:
        scene["env_args"]["control"] = args.control
    if args.teammates is not None:
        scene["env_args"]["n_teammates"] = args.teammates
    if args.switch_interval:
        scene["env_args"]["switch_interval"] = args.switch_interval
    if args.smoke:
        method.update(t_max=125, batch_size=2, buffer_size=8, test_nepisode=2,
                      test_interval=75, save_model_interval=125, learner_log_interval=1)
    if args.steps is not None:
        method["t_max"] = args.steps
    trainer = TRAINERS[method["name"]]
    paths = pool_paths(args.teammate_pool, args.scene) if args.teammate_pool else args.teammate_checkpoint
    groups = frozen_groups(paths, scene) if paths else None
    config = dict(scene=scene, agent=agent, method=method, seed=args.seed, smoke=args.smoke,
                  teammate_checkpoints=[str(p.relative_to(ROOT)) for p in paths],
                  gpu_uuid=uuid)
    with (args.run_dir / "config.json").open("x", encoding="utf-8") as out:
        json.dump(config, out, ensure_ascii=False, indent=2)
        out.write("\n")
    summary = trainer(scene_config=scene, agent_config=agent, method_config=method,
                      run_dir=args.run_dir, seed=args.seed, device="cuda:0", teammate_groups=groups)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=project_path, help="fresh directory under project outputs/")
    parser.add_argument("--scene", required=True, choices=sorted(p.stem for p in (ROOT / "configs/scenarios").glob("*.yaml")))
    parser.add_argument("--agent-config", type=project_path, default=ROOT / "configs/agents/drqn.yaml")
    parser.add_argument("--method-config", type=project_path, default=ROOT / "configs/methods/qmix.yaml")
    parser.add_argument("--control", choices=("team", "ego"), help="which cooperative roles to train")
    parser.add_argument("--teammates", type=int, help="restrict to a supported team size")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--teammate-checkpoint", action="append", default=[], type=project_path,
                        help="frozen all-team checkpoint directory; repeat to build a pool")
    source.add_argument("--teammate-pool", type=project_path, help="pool.json with frozen teammate policies for this scene")
    parser.add_argument("--switch-interval", type=int, nargs=2, metavar=("MIN", "MAX"),
                        help="enable teammate resampling during ego evaluation")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--smoke", action="store_true", help="125-step integration run with small batches")
    parser.add_argument("--steps", type=int, help="override the training timestep budget")
    parser.add_argument("--expected-peak-mib", type=int, default=BASELINE_PEAK,
                        help="reserve expected process GPU memory inside the shared budget")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.control == "team" and (args.teammate_checkpoint or args.teammate_pool or args.switch_interval):
        parser.error("External teammate checkpoints and switching require ego control")
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be positive")
    if args.switch_interval and not 1 <= args.switch_interval[0] <= args.switch_interval[1]:
        parser.error("Expected 1 <= MIN <= MAX")
    if any(ROOT not in p.parents for p in args.teammate_checkpoint):
        parser.error("Keep teammate checkpoints inside the project")
    if args.teammate_pool and ROOT not in args.teammate_pool.parents:
        parser.error("Keep the pool manifest inside the project")
    if args.worker:
        train(args)
        return

    # Make the reservation visible to concurrent monitors in a canonical form.
    worker = [str(Path(__file__).resolve()), "--worker", "--expected-peak-mib",
              str(args.expected_peak_mib), *sys.argv[1:]]
    run_monitored(args.run_dir, worker, args.expected_peak_mib)


if __name__ == "__main__":
    main()
