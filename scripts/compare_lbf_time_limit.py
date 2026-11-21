"""Compare two complete LBF ego runs differing only in the 25-step TD target."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compare_lbf_ego_seeds import read_run, ROSTERS


def compare(baseline, finite):
    import torch
    from fastap_methods.lbf_pool_ego_eval import compare_ego_returns

    signatures, runs, scores = [], {}, {}
    for label, directory, expected in (("bootstrap", baseline, True), ("finite", finite, False)):
        (config, protocol), steps, run = read_run(directory)
        if config["method"].pop("bootstrap_at_time_limit") is not expected or protocol["episode_limit"] != 25:
            raise ValueError(f"Unexpected timeout treatment or horizon in {directory}")
        signatures.append((config, protocol, run["seed"]))
        runs[label] = run
        scores[label] = json.loads((directory / "heldout.json").read_text())["scores"]
    if signatures[0] != signatures[1]:
        raise ValueError("Both runs must use the same seed, settings, teammate pool and evaluation protocol")
    for name in ("agent.th", "mixer.th"):
        a, b = [torch.load(d / "checkpoints/0" / name, weights_only=True, map_location="cpu")
                for d in (baseline, finite)]
        if a.keys() != b.keys() or any(not torch.equal(a[key], b[key]) for key in a):
            raise ValueError(f"Different initial weights in {name}")
    if scores["bootstrap"]["initial"] != scores["finite"]["initial"]:
        raise ValueError("Initial policies did not give identical held-out results")

    paired = compare_ego_returns(scores["finite"]["final"], scores["bootstrap"]["final"])
    interval_scores = {}
    for label, run in runs.items():
        at = {step: row["mean_return"] for step, row in zip(steps, run["validation"])}
        interval_scores[label] = at[2_000_000] - at[1_000_000]
    return dict(training_seed=runs["finite"]["seed"], runs=runs, nominal_steps=steps,
                validation_gain_1m_to_2m=interval_scores, heldout_finite_minus_bootstrap=paired,
                heldout_map_seeds=protocol["heldout_map_seeds_per_roster"],
                heldout_episodes_per_policy=protocol["heldout_episodes_per_policy"],
                uncertainty="95% paired bootstrap interval over map seeds, keeping both rosters "
                            "and all partners together. Conditional on this pair of trained policies.")


def plot(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    fonts = set(font_manager.get_font_names())
    plt.rcParams.update({"font.family": [f for f in ("Noto Sans CJK SC", "Noto Sans CJK JP") if f in fonts]
                         + ["DejaVu Sans"], "axes.spines.top": False, "axes.spines.right": False})
    fig, (curve, effect) = plt.subplots(1, 2, figsize=(10.8, 4.5), gridspec_kw={"width_ratios": [1.5, 1]})
    fig.subplots_adjust(left=.08, right=.97, bottom=.24, top=.76, wspace=.37)
    fig.suptitle(f"LBF QMIX · 25 步超时处理对照（种子 {report['training_seed']}）",
                 x=.08, ha="left", y=.96, fontsize=17, fontweight="bold")
    for key, label, color, style in (("bootstrap", "超时保留后续价值", "#28617a", "--"),
                                     ("finite", "超时截断后续价值", "#b04a27", "-")):
        rows = report["runs"][key]["validation"]
        curve.plot([r["training_steps"] / 10_000 for r in rows], [r["mean_return"] for r in rows],
                   label=label, color=color, lw=2.2, ls=style)
    curve.set(xlabel="训练环境步数（万）", ylabel="平均团队回报", xticks=[0, 50, 100, 150, 200])
    curve.set_ylim(bottom=0)
    curve.set_title("验证学习曲线", loc="left")
    curve.grid(axis="y", alpha=.2)
    curve.legend(frameon=False, loc="lower right")

    paired = report["heldout_finite_minus_bootstrap"]
    effect.axvline(0, color="#839099", lw=1, ls="--")
    for y, row in enumerate([paired, *[paired["per_roster"][r] for r in ROSTERS]]):
        lower, upper = row["bootstrap_95_ci"]
        effect.hlines(y, lower, upper, lw=2.2, color="#b04a27")
        effect.plot(row["mean_difference"], y, "o", color="#b04a27")
    effect.set(yticks=[0, 1, 2], yticklabels=["全体", "单队友", "双队友"], ylim=(2.5, -.5),
               xlabel="平均回报差（截断减保留）")
    effect.set_title("最终独立测试", loc="left")
    fig.text(.08, .05, f"右图为 {report['heldout_map_seeds']:,} 个地图种子的配对差及 95% 区间，"
             f"每个策略评估 {report['heldout_episodes_per_policy']:,} 局。", color="#495961", fontsize=10)
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    analysis = commands.add_parser("compare", help="Analyze matched runs with the training Python environment")
    analysis.add_argument("baseline", type=Path)
    analysis.add_argument("finite", type=Path)
    analysis.add_argument("output", type=Path, help="Output JSON file")
    draw = commands.add_parser("plot", help="Render a comparison JSON with the matplotlib environment")
    draw.add_argument("report", type=Path)
    draw.add_argument("output", type=Path, help="Output figure path")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "compare":
        report = compare(args.baseline, args.finite)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    else:
        plot(json.loads(args.report.read_text()), args.output)
