# Fastap_realize

项目复现代码仓库。

所有项目文件均存放在 `/mnt/ssd/users/linhui/projects/Fastap_realize/` 下，包括虚拟环境、依赖缓存、模型、数据、临时文件和训练输出。安装、下载和训练前必须显式检查相关写入路径，避免使用 `/home/linhui/` 或系统盘默认缓存目录。

Git 同步代码、配置与依赖清单。虚拟环境、缓存、模型、数据、训练输出和凭据不提交到仓库；默认忽略规则见 `.gitignore`。

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

本阶段用两组规则队友检查动作接口。一组只有一个队友，从合法动作中随机选择；另一组有两个队友，相邻有食物时优先采集，其余时间随机行动。后续可通过 `TeammateGroup` 传入训练好的策略，每个队友单独维护自己的策略状态。见 [队友接口](fastap_lbf/env.py)。

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
