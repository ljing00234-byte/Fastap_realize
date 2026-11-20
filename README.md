# Fastap_realize

本仓库用于复现论文 *Fast Teammate Adaptation in the Presence of Sudden Policy Change*。

## 游戏场景搭建模块

本模块按论文搭建了协作采食（LBF）、追捕（PP）和协作导航（CN），实现了三个场景的观测、动作和奖励规则。三个游戏共用一套接口，方便后续训练队友和比较算法。

[场景代码](fastap_scenarios/) · [场景配置](configs/scenarios/)

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
