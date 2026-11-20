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
