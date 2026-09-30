#!/usr/bin/env bash
# multi-stems: RoFormer 背景/人声 + pyannote N 说话人轨
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VIDEO="${1:-桔梗犬夜叉片段.mp4}"
OUT_DIR="${2:-./out_multi}"
MODE="${3:-diarize}"   # diarize | diarize_bss
MIN_SP="${MIN_SPEAKERS:-1}"
MAX_SP="${MAX_SPEAKERS:-6}"

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$ROOT/models/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$ROOT/models/huggingface/hub}"

PY="${PY:-python}"

echo "==> 部署检查"
"$PY" scripts/verify_diarize_setup.py || {
  echo "请先: Accept pyannote community-1 协议, 并设置 HF_TOKEN 或项目根 hf_token"
  exit 1
}

echo "==> 预下载 pyannote (若已缓存会很快)"
"$PY" scripts/prefetch_pyannote_models.py

echo "==> multi-stems 分离 mode=$MODE out=$OUT_DIR"
"$PY" src/cli/separate_audio.py "$VIDEO" \
  --out-dir "$OUT_DIR" \
  --multi-stems \
  --split-mode "$MODE" \
  --min-speakers "$MIN_SP" \
  --max-speakers "$MAX_SP" \
  --diarization-model pyannote/speaker-diarization-community-1

echo "==> 输出:"
ls -la "$OUT_DIR"/*_speaker_*.wav "$OUT_DIR"/*_split_report.json 2>/dev/null || true
