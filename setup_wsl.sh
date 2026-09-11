#!/usr/bin/env bash
# 在 WSL2 (Ubuntu) 中一键搭建环境:  bash setup_wsl.sh
set -e

# ── 前置条件（Windows 侧）─────────────────────────────
# 1. 安装 NVIDIA 驱动（桌面版即可）
# 2. WSL2 已启用；验证: 在 WSL 里 `nvidia-smi` 能显示显卡
# ────────────────────────────────────────────────────────

# 1. 安装 Python 3.12（torch 2.4.0 只支持到 3.12；系统默认若是 3.13/3.14 需单独装）
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.12 python3.12-venv

# 2. 用 3.12 建虚拟环境（Ubuntu 24.04+ 禁止系统级 pip 装包，必须用 venv）
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip

# 3. 安装 PyTorch（阿里云镜像用 -f；官方源用 --index-url https://download.pytorch.org/whl/cu121）
pip install torch==2.4.0+cu121 -i https://mirrors.aliyun.com/pypi/simple/ -f https://mirrors.aliyun.com/pytorch-wheels/cu121/

# 4. 其余依赖
pip install -r requirements.txt

echo ""
echo "完成。验证 GPU（应输出 2.4.0+cu121 True）:"
echo "  python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"
echo ""
echo "以后每次使用前先激活环境:"
echo "  cd ~/qwen-intent-finetune && source .venv/bin/activate"
echo "然后: python generate_data.py -> python train.py -> python inference.py"