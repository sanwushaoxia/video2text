#!/usr/bin/env python3
"""预下载 pyannote 日记化 pipeline 到 HuggingFace 本地缓存。"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from video2text.diarization import (  # noqa: E402
    DEFAULT_DIARIZATION_MODEL,
    LEGACY_DIARIZATION_MODEL,
    _resolve_hf_token,
)
from video2text.paths import ensure_huggingface_env, huggingface_home  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="预下载 pyannote speaker-diarization 模型")
    p.add_argument(
        "--model",
        default=DEFAULT_DIARIZATION_MODEL,
        help=f"默认 {DEFAULT_DIARIZATION_MODEL}",
    )
    p.add_argument("--hf-token", default=None, help="HF Token (或 HF_TOKEN 环境变量 / hf_token 文件)")
    p.add_argument(
        "--legacy-fallback",
        action="store_true",
        help=f"主模型失败时尝试 {LEGACY_DIARIZATION_MODEL}",
    )
    args = p.parse_args(argv)

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    try:
        token = _resolve_hf_token(None)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    ensure_huggingface_env()
    print(f"HF 缓存: {huggingface_home()}")

    if os.environ.get("HF_ENDPOINT"):
        print(f"HF_ENDPOINT: {os.environ['HF_ENDPOINT']}")
    else:
        print("提示: 若 huggingface.co 不可达, 请先 export HF_ENDPOINT=https://hf-mirror.com")

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

    try:
        from pyannote.audio import Pipeline
    except ImportError:
        print("需要 pyannote.audio: pip install -r requirements.txt", file=sys.stderr)
        return 1

    model_id = args.model
    print(f"预下载: {model_id}")
    try:
        Pipeline.from_pretrained(model_id, token=token)
    except Exception as exc:
        if not args.legacy_fallback:
            print(f"失败: {exc}", file=sys.stderr)
            return 1
        print(f"主模型失败 ({exc}), 尝试 {LEGACY_DIARIZATION_MODEL}...")
        model_id = LEGACY_DIARIZATION_MODEL
        Pipeline.from_pretrained(model_id, token=token)

    print(f"pyannote 已缓存: {model_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
