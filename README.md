# Fastap_realize

本仓库用于复现论文 *Fast Teammate Adaptation in the Presence of Sudden Policy Change*。

## 游戏场景搭建模块

本模块按论文搭建了协作采食（LBF）、追捕（PP）和协作导航（CN），实现了三个场景的观测、动作和奖励规则。三个游戏共用一套接口，方便后续训练队友和比较算法。

[场景代码](fastap_scenarios/) · [场景配置](configs/scenarios/)

## 智能体与训练方法

**第一条基线是 vanilla QMIX。** 训练时，QMIX 使用全局状态将各角色的动作价值组合为团队价值；执行时，每个角色只根据自己的局部历史选择合法动作。LBF、PP 和 CN 的回放和评估记录原始团队奖励。生成冻结队友时，TD 更新也使用原始奖励；下述 LBF ego 实验按所用 EPyMARL 版本的默认设置，在计算 TD 目标前标准化奖励。[Fastap 论文](https://proceedings.mlr.press/v216/zhang23a.html) · [QMIX 原文](https://proceedings.mlr.press/v80/rashid18a.html)

| 模块 | 输入与职责 | 扩展入口 |
| --- | --- | --- |
| 智能体 | 本地观测、上一步动作、原始角色编号 → 各动作的 Q 值。默认共享一个 64 维 GRU，各角色单独保存记忆 | 新网络通过 `register_agent_backend` 注册训练控制器、权重说明和冻结策略加载器。[agent 模块](fastap_agents/) · [agent 配置](configs/agents/drqn.yaml) |
| 训练方法 | 从完整回合采样，利用全局状态、联合动作和团队奖励更新网络。默认使用单调混合网络、Double Q、经验回放和目标网络 | 新方法提供与 `train_qmix` 相同的调用入口，加入 `TRAINERS`，用方法 YAML 选择。[方法模块](fastap_methods/) · [QMIX 配置](configs/methods/qmix.yaml) |

默认折扣为 0.99，批量为 32 个回合，学习率为 0.0005，每 200 次优化更新同步目标网络。正式配置的 200 万个环境步对应原文基准学习曲线的横轴范围。原文没有单列 vanilla QMIX 的全部网络开关，本实现显式启用 GRU 和上一动作输入，并关闭 EPyMARL 的奖励标准化。队友生成的训练步数可用 `--steps` 单独指定。[论文补充材料](https://proceedings.mlr.press/v216/zhang23a/zhang23a-supp.pdf)

下面两条命令演示 LBF 的完整接口。第一条训练三人团队；第二条从其检查点加载第 3 个角色作为冻结队友，训练两个 ego。示例使用 125 步的小批量验证配置，每次运行使用新的输出目录。

```bash
source scripts/epymarl-env.sh
python3 -B scripts/run_baseline.py outputs/demo_team --scene lbf --control team --teammates 1 --smoke
python3 -B scripts/run_baseline.py outputs/demo_ego --scene lbf --control ego --teammate-checkpoint outputs/demo_team/checkpoints/125 --smoke
```

相同入口接受 `--scene pp` 或 `--scene cn`。用 `--agent-config` 和 `--method-config` 分别选择网络配置与训练配置。重复传入 `--teammate-checkpoint` 可组成队友池。训练时每局固定一组，ego 评估可加 `--switch-interval 5 8` 启用回合内重抽。同组重抽延续队友记忆，实际换组或新回合重新初始化。省略检查点时使用该游戏的规则队友。冻结队友需来自同一游戏的全队训练。

GPU 训练通过同一监控入口绑定物理 GPU 2。输出目录内，`config.json` 记录所选场景、agent、方法与队友来源，`metrics.jsonl` 记录训练和评估指标，`summary.json` 给出最终检查点路径。每个检查点包含局部策略、角色元数据、混合网络和优化器；执行冻结策略只需 `agent.th` 和 `agent.json`。

### 第一名队友及其 best response

在 LBF 中先用 QMIX 训练三人团队，从验证回报最高的检查点取出第 3 个角色作为队友 T1。固定 T1 的权重后，从头训练两个 ego，使它们与 T1 配合时的团队回报尽可能高。这两个 ego 的联合策略是本实验对 best response 的近似。

```bash
source scripts/epymarl-env.sh
envs/epymarl/bin/python scripts/train_teammate_response.py outputs/lbf_teammate_A
```

两阶段默认各 20 万步，训练种子分别为 101 和 202。这是第一组实验的预算。每阶段在相同的 64 张验证地图上选择检查点，最后另用 128 张留出地图比较「T1 + 原搭档」「T1 + 新训练的 ego」和「随机队友 + 原搭档」。`status.json` 标明当前阶段和已选权重，`heldout.json` 保存最终回报与成功采食率。复跑时使用新的输出目录。

### 基础验证：一个队友能否学会配合

在 LBF 中用 QMIX 训练一支三人团队。物理 GPU 2 批量更新棋盘、计算动作和奖励，并存储完整回放。每个角色只看到自己的局部观测。训练批量为 32 局，回放预热后每新增一局做一次梯度更新。启动时用原版 CPU 环境逐步核对 GPU 的状态、观测、动作掩码、奖励和终止标记。

```bash
source scripts/epymarl-env.sh
envs/epymarl/bin/python -B scripts/train_lbf_cooperation.py outputs/cooperation_run --num-envs 8
```

三个阶段依次是同一食物的邻格、还需走一至两步、标准随机出生。阶段共用一套模型和优化器，转段时清空回放。默认阶段上限分别为 5 万、5 万和 10 万个有效环境步，总预算 20 万步。可分别用 `--stage-steps` 和 `--total-steps` 指定。每阶段保留验证成绩最好的检查点；达到上限仍未通过时结束训练。定期验证与最终独立测试各使用 128 张地图，在同一地图上固定两个搭档，只将第三名替换成随机合法动作。

邻格阶段要求首步同步采集率至少 80%，并通过成对单侧检验（p < 0.05）。后两阶段要求至少采到一份的成功率显著优于随机队友，且平均回报更高。报告也给出三份全部采完的比例。运行进度见输出目录的 `status.json`，采样与梯度更新时间见 `metrics.jsonl`，最终选中模型在三个阶段的独立成绩见 `heldout.json`。在新的输出目录使用 `--benchmark-only` 可单独比较 1、8、32 个并行环境的吞吐。

### 生成和检验一名 LBF 队友

在标准随机地图上用 vanilla QMIX 从头联合训练三个角色，默认训练 200 万个有效环境步。每 5 万步在固定验证集上按三份采完率、平均回报的顺序选择一个检查点。运行命令中的输出目录需为空。

```bash
source scripts/epymarl-env.sh
envs/epymarl/bin/python -B scripts/train_lbf_cooperation.py outputs/single_T1 --single-teammate --num-envs 32 --eval-maps 256 --heldout-seed 720000
```

输出目录的 `T1/` 含可直接加载的权重，使用 `load_frozen_agent(path).policy_factory(2)` 取得第 3 个角色。选定后在原版 CPU 环境的 1,024 张独立地图上固定两名搭档，对照训练后的 T1、合法随机动作和初始 T1，并记录初始整队的成绩。`protocol.json` 记录事先设定的判据，`heldout.json` 记录逐图回报、至少采到一份和三份全采完的配对比较。

### 生成和检验一名 PP 队友

三个捕食者联合训练，固定逃逸策略控制一只猎物，场上有两个障碍物。每局 25 步，每发生一次捕食者与猎物接触，团队得 10 分，持续接触可重复计分。PP 沿用原始环境奖励。

```bash
source scripts/epymarl-env.sh
envs/epymarl/bin/python -B scripts/train_lbf_cooperation.py outputs/pp_single_T1 --scene pp --single-teammate --num-envs 32 --eval-maps 256 --heldout-seed 990000
```

训练预算为 200 万步。每 5 万步按验证集平均回报选检查点，分数相同时比较至少接触一次的回合比例。选定后从 `T1/` 的权重加载第 3 个捕食者，与固定的两个搭档在原版 CPU 环境上测试 1,024 个独立开局。报告列出随机 T1、初始 T1 和初始整队的同图成绩。训练后的 T1 相对两种替换队友的平均回报差，其配对 bootstrap 95% 区间下界都大于零时，实验判定该队友对团队有正贡献。

对比回合时限对 QMIX 的影响时，使用同一命令和种子，另选输出目录并加上 `--no-bootstrap-at-time-limit`。默认设置在第 25 步仍计入下一状态的估值，此选项让第 25 步的学习目标等于该步实际奖励。两组均按 25 步的实际回报评估。

两组完成后运行 `envs/epymarl/bin/python -B scripts/compare_pp_time_limit.py 原组目录 对照组目录`。脚本核对配置及初始权重，在 1,024 个新开局复测两组策略，并输出同图回报差的 95% 区间。

### 生成和检验一名 CN 队友

```bash
source scripts/epymarl-env.sh
envs/epymarl/bin/python -B scripts/train_lbf_cooperation.py outputs/cn_single_T1_horizon25_01 --scene cn --single-teammate --num-envs 32 --eval-maps 256 --heldout-seed 1810000 --no-bootstrap-at-time-limit
```

单队友阵容有两名可控智能体、一名队友和四个地标。QMIX 训练 200 万步，每局 25 步，按验证集平均回报选择模型。每步对四个地标分别计算最近智能体的距离，团队奖励为距离和的负值，每对碰撞的智能体再扣 1 分。

独立测试通过双智能体接口加载 `T1/`，在同一批 1,024 个开局上比较已训练、随机和未训练的队友。`heldout.json` 包含回报、地标距离和碰撞次数。`compatibility.json` 记录 GPU 与单局环境的逐步对照，以及导出的队友在两个控制接口中的动作、状态和回报一致性。

### 生成 LBF 队友集合

默认生成 8 个单队友策略。T1 复用已验证的权重，其余成员使用不同随机种子，各在物理 GPU 2 上训练 200 万步。每张棋盘上有两名搭档和一名队友。模型、环境、奖励和用于选检查点的验证集与 T1 相同。

```bash
source scripts/epymarl-env.sh
envs/epymarl/bin/python -B scripts/generate_lbf_pool.py outputs/lbf_teammate_pool_01 --size 8
```

每完成一名成员的 1,024 图独立测试，`pool.json` 就会增加一条记录，分别标明能否证实采到食物和采完全图的贡献。实验保留预定种子的全部结果。后续 ego 训练可用 `--teammate-pool outputs/lbf_teammate_pool_01/pool.json` 读取启动时已发布的成员，也可用 `--teammate-checkpoint` 指定个别成员。

全体完成后，`crossplay.json` 记录在另外 256 张相同地图上互换队友的成绩，每行固定一对原搭档，每列更换第 3 个角色。文件也记录了各个队友在 T1 原团队同一批观测及动作历史上选择不同动作的比例。

### 训练 LBF 的 QMIX ego

```bash
source scripts/epymarl-env.sh
envs/epymarl/bin/python -B scripts/train_lbf_pool_ego.py outputs/lbf_pool_ego_qmix_01
```

两个 ego 从零训练并共享 GRU 权重。棋盘为 6×6，有三份食物、3×3 局部视野和 25 步时限。每局抽取一名或两名冻结队友，局内固定。训练预算为 200 万个有效环境步，主要结果使用期末检查点的原始团队回报。[LBF 场景及 QMIX 网络](https://proceedings.mlr.press/v216/zhang23a/zhang23a-supp.pdf#page=3) · [原论文图 3](https://proceedings.mlr.press/v216/zhang23a/zhang23a.pdf#page=7)

作者未公布两种阵容的比例和队友权重。本实验每批 32 局两种人数各半，从 T1–T8 等概率抽取队友。双队友从池中不放回抽取，两个角色各用自己的观测和记忆。两种阵容分别报告回报，并以等权均值汇总。每约 5 万步验证一次，期末使用另外 1,024 个地图种子评估，同时保留初始权重及验证集表现最好的检查点供对照。

### 生成 PP 队友集合

```bash
source scripts/epymarl-env.sh
envs/epymarl/bin/python -B scripts/generate_teammate_pool.py outputs/pp_teammate_pool_01 --scene pp --size 8
```

T1 复用通过检验的 25 步终止模型。T2–T8 使用不同种子，各训练 200 万步，第 25 步的目标等于当步奖励。每名成员在相同的 1,024 个开局上与随机队友、未训练队友对照。`pool.json` 保留所有预定种子的权重路径、测试得分和检验结论，供 `--scene pp --teammate-pool` 加载。全部完成后，在另外 256 个开局上评估每对搭档与每名队友的组合，接触局数、平均团队回报和动作差异写入 `crossplay.json`。

### 生成 CN 队友集合

```bash
source scripts/epymarl-env.sh
envs/epymarl/bin/python -B scripts/generate_teammate_pool.py outputs/cn_teammate_pool_01 --scene cn --size 8
```

T1 直接复用单队友实验选定的权重。T2–T8 各用独立种子训练 200 万步，沿用相同的四地标奖励和 25 步时限。每名队友在相同的 1,024 个测试开局上与随机队友、未训练队友作对照。后续 ego 训练可用 `--scene cn --teammate-pool outputs/cn_teammate_pool_01/pool.json` 加载集合。

`crossplay.json` 在另外 256 个开局上评估 8×8 种组合。每行固定两名原搭档，每列换入一名队友。每格给出平均团队回报、每步四个地标的最近距离之和、每局碰撞对数的累计值。

## 项目存储

所有项目文件均存放在 `/mnt/ssd/users/linhui/projects/Fastap_realize/` 下，包括虚拟环境、依赖缓存、模型、数据、临时文件和训练输出。安装、下载和训练前必须显式检查相关写入路径，避免使用 `/home/linhui/` 或系统盘默认缓存目录。

Git 同步代码、配置与依赖清单。虚拟环境、缓存、模型、数据、训练输出和凭据不提交到仓库；默认忽略规则见 `.gitignore`。

## Fastap 的三个游戏场景

场景名分别为 `lbf`、`pp` 和 `cn`。每步由游戏统一计算一次团队奖励。队友组和训练算法可以单独替换。游戏规则参照[论文补充材料 B.2](https://proceedings.mlr.press/v216/zhang23a/zhang23a-supp.pdf#page=4)。

| 游戏 | 在场角色与目标 | 动作、奖励和终止 |
| --- | --- | --- |
| LBF 协作采食 | 6×6 棋盘，2 个 ego、1～2 名队友、3 份食物。角色只能看到周围 3×3 区域 | 四向移动、等待、采集。相邻且同时采集的角色等级之和达到门槛后，团队获得所采食物等级除以开局食物总等级的奖励。食物采完或达到 25 步时结束。 |
| PP 追捕 | 2 个 ego 捕食者、1 名队友捕食者、1 只更快的猎物、2 个固定障碍物。队友可换策略，捕食者人数固定 | 五个离散动作驱动连续空间中的运动。每步每对捕食者与猎物接触给团队 +10，给猎物 −10。两个捕食者同时接触则为 +20 和 −20。捕获后继续运行，直至时限。 |
| CN 协作导航 | 2 个 ego、1～2 名队友，始终有 4 个固定地标。每人看到自身速度、位置及所有其他在场角色和地标的相对位置 | 五个离散动作。每步对四个地标分别找最近的在场角色，取四个欧氏距离之和的负值，再扣除角色间碰撞罚分。地标被覆盖后保留，直至时限。 |

**复现取值。** LBF 的角色等级均匀取 1～3，食物等级取论文给出的下限。PP/CN 的连续运动参考 [MPE 粒子动力学](https://github.com/openai/multiagent-particle-envs/blob/master/multiagent/core.py)，支持停、左、右、下、上；初始位置的采样范围不构成硬墙。论文没有指定这两个场景的时限，默认均为 25 步，可通过 `world_options.episode_limit` 调整。PP 默认给猎物使用确定性的躲避策略，可以用 `prey_policy_factory` 注入带记忆的策略，每局独立初始化。CN 默认对每一对发生重叠的不同角色扣 1 分，每对只计一次，`world_options.collision_penalty` 可调。CN 增减一名队友时保留时钟、地标和留场角色，加入者的初始位置重新采样。

**场景、智能体和训练方法的接口**

1. **游戏层。** `make_world(key, seed=...)` 返回独立的游戏实例。通过 `reset(n_teammates=...)` 开局，用 `observations()`、`avail_actions()` 和 `state()` 读取所有协作角色的观测、动作掩码与全局状态。`step(joint_actions)` 接收按 ego、队友排列的动作，返回 `(reward, done, info)`。仅因时限结束时 `info["episode_limit"]` 为真。添加新游戏时实现 [GameWorld 协议](fastap_scenarios/game.py)，再加入[场景注册表](fastap_scenarios/registry.py)。
2. **智能体层。** `make_env(key, control="team", n_teammates=...)` 允许外部控制全部协作角色，供后续生成队友。`control="ego"` 则只接收两个动作，队友由注入的 [TeammateGroup](fastap_scenarios/policies.py) 提供，各自读取局部观测和合法动作。默认使用规则策略。评估时设置 `switch_interval=(5, 8)` 可开启局内重抽，默认的通用场景配置使用固定队友。`env.last_joint_actions` 保存刚结束一步内实际提交的全队动作，PP 的 `env.world.last_prey_action` 另记猎物动作。
3. **方法层。** [场景 YAML](configs/scenarios) 只选择游戏、控制方式和游戏参数。接入已有 EPyMARL 训练器时，在其 `src` 可导入的 Python 进程中调用 `fastap_scenarios.epymarl.register_epymarl()`，将选中的场景 YAML 叠加到算法配置上，使用已注册的 `fastap_episode` runner。网络、优化器和训练步数由所选方法管理。旧的 `fastap_lbf` 入口仍支持原 LBF 实验。

以下例子只推进游戏一步。将 `"cn"` 换成 `"lbf"` 或 `"pp"` 可以使用相同调用方式。

```python
from fastap_scenarios import make_env

env = make_env("cn", control="team", n_teammates=1, seed=7)
try:
    obs, state = env.reset()
    reward, done, info = env.step([0, 0, 0])
finally:
    env.close()
```

从项目根目录复现三张游戏开局图和场景测试。预览使用实际世界坐标，输出到 `outputs/scenarios/seed7.svg`。

```bash
source scripts/epymarl-env.sh
envs/epymarl/bin/python -m fastap_scenarios.preview --seed 7
CUDA_VISIBLE_DEVICES='' envs/epymarl/bin/python -m unittest discover -s tests -v
```

## QMIX + LBF 环境

使用 [EPyMARL 的 2023 年 2 月提交](https://github.com/uoe-agents/epymarl/tree/f355a55262ac9afecdb53368fec6337c549cc160)、Python 3.9.23、Gym 0.21 和 LBF 1.0.15。选择此提交是为了使用与 Fastap 预印本同时期的框架。PyTorch 使用支持本机 Blackwell GPU 的 Linux CUDA 12.8 构建。安装清单面向 QMIX + LBF，小实验无需安装星际争霸游戏。

在项目根目录用 Bash 准备环境。先用 `df -h .`、`pwd -P` 核对 SSD 空间和项目根目录。每个新终端先加载项目缓存与临时目录配置。

```bash
source scripts/epymarl-env.sh
mkdir -p third_party
git clone https://github.com/uoe-agents/epymarl.git third_party/epymarl
git -C third_party/epymarl checkout --detach f355a55262ac9afecdb53368fec6337c549cc160
python3 -m pip install --target "$XDG_CACHE_HOME/bootstrap" uv==0.8.22
uv_bin="$XDG_CACHE_HOME/bootstrap/bin/uv"
"$uv_bin" python install 3.9.23
"$uv_bin" venv --managed-python --python 3.9.23 envs/epymarl
"$uv_bin" pip install --python envs/epymarl/bin/python pip==23.3.2 wheel==0.38.4 setuptools==65.5.0
envs/epymarl/bin/python -m pip install --no-build-isolation -r requirements-epymarl-lbf.txt
```

在可以访问物理 GPU 2 的宿主机运行，输出目录每次使用新名字。监控入口在启动时识别 GPU 2，检查已有任务和显存预算，训练期间记录该进程的显存占用。

```bash
source scripts/epymarl-env.sh
python3 -B scripts/run_gpu2_smoke.py outputs/qmix_lbf_smoke_01
```

示例使用普通 QMIX 与 8×8、2 个智能体、3 个食物的 LBF 环境。每回合最多 25 步。训练脚本检查有限的 TD 损失、模型权重的实际变化和测试回报；各次结果保存在运行目录内的 `summary.json` 和 `monitor.json`。

## Fastap 的 6×6 LBF 场景

两个受控角色整局留在棋盘上，由 EPyMARL 为它们选择动作。另外一到两个队友各自根据自己的局部观测行动。棋盘有三份食物，视野半径为一格，每局最多 25 步。相邻角色同时采集且合计等级达到食物等级时，团队获得一次共享奖励，数值为该食物等级占初始食物总等级的比例。

论文附录 8.2 给出了食物等级下限，未给出完整的采样分布。本实现将角色等级均匀采样为 1～3，三份食物均取开局等级最低的三个角色的等级之和。队友人数改变时，留场角色和食物保留各自的位置，新加入的角色从空格中随机选择出生点。

本阶段用两组规则队友检查动作接口。一组只有一个队友，从合法动作中随机选择；另一组有两个队友，相邻有食物时优先采集，其余时间随机行动。后续可通过 `TeammateGroup` 传入训练好的策略，每个队友单独维护自己的策略状态。见 [队友接口](fastap_scenarios/policies.py)。

训练回合按论文算法 2 在开局抽取一组队友，整局固定。评估回合按算法 3 重新抽取队友组，默认每次等待 5～8 步，抽中当前组也计为一次抽取。固定队友评估可将 `switch_interval` 设为 `null`。论文表 1 另提到训练时使用 5～8 步切换，本阶段联调采用算法 2 的固定队友设置。见 [场景配置](configs/fastap_lbf.yaml)。

要在 CPU 上观察同一初始棋盘下的两种评估回合，可以运行下面的示例。两个受控角色在示例中原地待命，输出包括初始棋盘和每次抽取的步数、队友组名及人数。

```bash
source scripts/epymarl-env.sh
envs/epymarl/bin/python -m fastap_lbf --seed 7
envs/epymarl/bin/python -m unittest discover -s tests -v
```

要用 QMIX 验证受控策略的参数更新，运行下列命令，每次使用新的输出目录。共享显存预算和已有任务由同一个 GPU 2 监控脚本核对。

```bash
source scripts/epymarl-env.sh
python3 -B scripts/run_gpu2_smoke.py --scene fastap outputs/fastap_lbf_scene_smoke_02
```
