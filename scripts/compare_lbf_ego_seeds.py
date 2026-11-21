# /// script
# requires-python = ">=3.9"
# dependencies = ["matplotlib==3.9.4", "numpy>=1.23"]
# ///
"""Compare five completed stationary LBF QMIX ego runs to Fastap Figure 3(a) and Table 1.

Source scripts/epymarl-env.sh and use Python with matplotlib (tested with 3.9.4).
The paper-curve JSON contains ordered {steps, mean_return} coordinates from Figure 3(a).
"""

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[1]
TARGET = 2_000_000
PAPER_PDF = "https://proceedings.mlr.press/v216/zhang23a/zhang23a.pdf"
PAPER_TABLE = {"mean_return": 0.569, "std": 0.033}
ROSTERS = ("one_teammate", "two_teammates")


def read_run(directory):
    def read(filename):
        return json.loads((directory / filename).read_text())

    cfg, protocol, status = read("config.json"), read("protocol.json"), read("status.json")
    seed = cfg.pop("seed")
    if (cfg["scene"], cfg["control"], cfg["method"]["name"]) != ("lbf", "ego", "qmix"):
        raise ValueError(f"{directory} is not a LBF QMIX ego run")
    if (status["phase"] != "complete" or protocol["total_step_budget"] != TARGET or
            cfg["method"]["t_max"] != TARGET or protocol.pop("training_seed") != seed):
        raise ValueError(f"Incomplete run, wrong budget or inconsistent seed in {directory}")
    # Compare how the training RNG streams derive from the seed, in addition to all fixed settings.
    for name in ("training_board_seed", "training_teammate_seed"):
        protocol[name] -= seed

    interval = protocol["validation_interval"]
    schedule = list(range(interval, TARGET, interval)) + [TARGET]
    max_batch_steps = protocol["episode_limit"] * protocol["num_envs"]
    members = [m["id"] for m in protocol["pool_members"]]
    with (directory / "metrics.jsonl").open() as log:
        validations = [row for row in map(json.loads, log) if row["kind"] == "validation"]
    if len(validations) != len(schedule):
        raise ValueError(f"Missing or extra validation checkpoints in {directory}")
    trajectory = []
    for nominal, row in zip(schedule, validations):
        actual = row["training_steps"]
        if not nominal <= actual < nominal + max_batch_steps:
            raise ValueError(f"Unexpected validation step {actual} for {nominal} in {directory}")
        if (row["first_map_seed"], row["n_maps_per_teammate"], row["n_episodes"], row["pool_order"]) != (
                protocol["validation_first_seed"], protocol["validation_map_seeds_per_roster"],
                protocol["validation_episodes"], members):
            raise ValueError(f"Inconsistent validation maps, episodes or pool in {directory}")
        trajectory.append(dict(training_steps=actual, mean_return=row["mean_return"]))

    heldout = read("heldout.json")
    final = heldout["scores"]["final"]
    actual = trajectory[-1]["training_steps"]
    if (heldout["primary_checkpoint"] != f"checkpoints/{actual}" or
            status["final_checkpoint"] != heldout["primary_checkpoint"] or status["training_steps"] != actual):
        raise ValueError(f"Held-out score does not refer to the final validation checkpoint in {directory}")
    if (final["first_map_seed"], final["n_maps_per_teammate"], final["n_episodes"], final["pool_order"]) != (
            protocol["heldout_first_seed"], protocol["heldout_map_seeds_per_roster"],
            protocol["heldout_episodes_per_policy"], members):
        raise ValueError(f"Inconsistent held-out maps, episodes or pool in {directory}")
    run = dict(seed=seed, run_directory=str(directory), validation=trajectory,
               training_steps=actual, training_episodes=status["training_episodes"],
               optimizer_updates=status["optimizer_updates"], final_checkpoint=heldout["primary_checkpoint"],
               heldout={"mean_return": final["mean_return"],
                        **{roster: final["per_roster"][roster]["mean_return"] for roster in ROSTERS}})
    return (cfg, protocol), schedule, run


def summarize(directories, paper_path):
    runs = []
    for directory in directories:
        signature, schedule, run = read_run(directory)
        if runs and signature != reference:
            raise ValueError(f"Configurations or protocols differ beyond the training seed in {directory}")
        reference = signature
        runs.append(run)
    if len(runs) != 5 or len({r["seed"] for r in runs}) != 5:
        raise ValueError("Provide five distinct completed training seeds")
    runs.sort(key=lambda r: r["seed"])
    paper = json.loads(paper_path.read_text())
    paper_steps = [p["steps"] for p in paper]
    paper_values = [p["mean_return"] for p in paper]
    if (not paper_steps or paper_steps[0] != 0 or paper_steps[-1] != TARGET or
            not np.all(np.diff(paper_steps) > 0) or not np.isfinite(paper_values).all()):
        raise ValueError(f"Expected an ordered Figure 3(a) curve spanning 0 to {TARGET:,} in {paper_path}")

    checkpoints = []
    for i, nominal in enumerate(schedule):
        by_seed = {str(run["seed"]): run["validation"][i]["mean_return"] for run in runs}
        checkpoints.append(dict(nominal_steps=nominal, validation_by_seed=by_seed,
                                mean_return=mean(by_seed.values()), sample_std=stdev(by_seed.values()),
                                paper_figure_approx=float(np.interp(nominal, paper_steps, paper_values))))
    heldout = {}
    for metric in ("mean_return", *ROSTERS):
        values = [run["heldout"][metric] for run in runs]
        heldout[metric] = dict(mean=mean(values), sample_std=stdev(values))
    protocol = reference[1]
    return dict(runs=runs, checkpoints=checkpoints, final_heldout=heldout,
                validation_episodes=protocol["validation_episodes"],
                heldout_episodes_per_seed=protocol["heldout_episodes_per_policy"],
                alignment="First completed parallel rollout at or after each scheduled validation; "
                          "exact evaluation steps are recorded per run. No smoothing.",
                dispersion="Sample standard deviation of the five training-seed means (ddof=1).",
                paper=dict(figure_3a_url=PAPER_PDF + "#page=7", table_1_url=PAPER_PDF + "#page=8",
                           figure_3a_digitized=paper, table_1_stationary_qmix=PAPER_TABLE,
                           note="Figure values are approximations; the paper does not specify the "
                                "number of training seeds or the aggregation unit of the Table 1 SD."))


def draw(report, path):
    cjk_fonts = ["Noto Sans CJK SC", "Noto Sans CJK JP", "Microsoft YaHei", "PingFang SC",
                 "WenQuanYi Zen Hei", "Droid Sans Fallback"]
    installed_fonts = set(font_manager.get_font_names())
    plt.rcParams.update({"font.family": [name for name in cjk_fonts if name in installed_fonts] + ["DejaVu Sans"],
                         "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#b7c1c9", "axes.labelcolor": "#364855",
                         "xtick.color": "#485966", "ytick.color": "#485966"})
    fig, (individual, aggregate) = plt.subplots(1, 2, figsize=(12.8, 6.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=.072, right=.978, top=.80, bottom=.25, wspace=.10)
    fig.suptitle("固定队友下的 LBF QMIX 学习曲线", x=.072, y=.963, ha="left",
                 fontsize=22, fontweight="bold", color="#182e40")
    interval = report["checkpoints"][0]["nominal_steps"] / 10_000
    fig.text(.072, .885, f"每 {interval:g} 万步取首个完成并行采样的检查点，按同一进度比较五个策略。",
             color="#526572", fontsize=11)
    seed_colors = ["#017c8e", "#be5c24", "#805b9a", "#50814c", "#9c4f69"]
    for run, color in zip(report["runs"], seed_colors):
        rows = run["validation"]
        individual.plot([r["training_steps"] / 10_000 for r in rows], [r["mean_return"] for r in rows],
                        color=color, lw=1.6, alpha=.94, label=str(run["seed"]))
    individual.set_title("各训练种子的验证回报", loc="left", pad=12, fontweight="bold", fontsize=14)
    individual.legend(loc="lower right", ncols=3, frameon=False, fontsize=9, columnspacing=.9, handlelength=1.5)
    individual.set_ylabel("平均团队回报")

    curve = report["checkpoints"]
    x = [p["nominal_steps"] / 10_000 for p in curve]
    central = np.array([p["mean_return"] for p in curve])
    sd = np.array([p["sample_std"] for p in curve])
    band = aggregate.fill_between(x, central - sd, central + sd, color="#8ab4c9", alpha=.44, linewidth=0)
    local_line, = aggregate.plot(x, central, lw=2.8, color="#154968")
    paper = report["paper"]["figure_3a_digitized"]
    paper_line, = aggregate.plot([r["steps"] / 10_000 for r in paper], [r["mean_return"] for r in paper],
                                 lw=2.1, ls=(0, (4, 2)), color="#a14d27")
    aggregate.set_title("五次训练均值与论文图 3(a)", loc="left", pad=12, fontweight="bold", fontsize=14)
    aggregate.legend([local_line, band, paper_line], ["本地均值", "跨种子 ±1 样本标准差", "论文曲线（估读）"],
                     loc="lower right", frameon=False, fontsize=9)
    for ax in (individual, aggregate):
        ax.grid(axis="y", color="#e2e7eb", lw=.8)
        ax.set_axisbelow(True)
        ax.set(xlabel="训练环境步数（万）", xticks=[0, 40, 80, 120, 160, 200], xlim=(0, 201))
        ax.set_ylim(bottom=0)
    heldout = report["final_heldout"]["mean_return"]
    table = report["paper"]["table_1_stationary_qmix"]
    fig.text(.072, .12, f"200 万步独立测试     本地 {heldout['mean']:.3f} ± {heldout['sample_std']:.3f}"
             f"     |     论文表 1（固定队友） {table['mean_return']:.3f} ± {table['std']:.3f}",
             fontsize=12, color="#182e40", fontweight="bold")
    fig.text(.072, .063, f"每个种子每次验证 {report['validation_episodes']:,} 局，"
             f"最终独立测试 {report['heldout_episodes_per_seed']:,} 局。论文未注明表中标准差的统计单位。",
             fontsize=10, color="#526572")
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs=5, type=Path, metavar="RUN_DIR")
    parser.add_argument("--paper-curve", type=Path, default=ROOT / ".tmp/epymarl/lbf_fig3a_qmix_digitized.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/lbf_pool_ego_qmix_five_seeds")
    args = parser.parse_args()
    report = summarize(args.run_dirs, args.paper_curve)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    draw(report, args.output_dir / "learning-curves.png")
    (args.output_dir / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved {args.output_dir / 'learning-curves.png'} and {args.output_dir / 'comparison.json'}")


if __name__ == "__main__":
    main()
