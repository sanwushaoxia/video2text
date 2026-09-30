#!/usr/bin/env bash
# 创建独立 conda 环境并安装项目全量依赖 (requirements.txt)
# 用法: bash scripts/setup_bs_roformer_env.sh [环境名]  (默认 video2text-roformer)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${1:-video2text-roformer}"

if ! command -v conda >/dev/null 2>&1; then
  echo "需要 conda/miniconda。无 conda 时手动: python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt" >&2
  exit 1
fi

conda create -n "$ENV_NAME" python=3.12 -y
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

pip install -r "$ROOT/requirements.txt"

echo "环境 $ENV_NAME 就绪。验证:"
echo "  conda activate $ENV_NAME"
echo "  audio-separator --list_models --list_filter roformer | head"
