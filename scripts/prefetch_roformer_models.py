#!/usr/bin/env python3
"""预下载 BS-RoFormer 模型到持久目录 (manifest + ckpt)。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from video2text.bs_roformer_split import (  # noqa: E402
    DEFAULT_ROFORMER_ENSEMBLE_PRESET,
    DEFAULT_ROFORMER_INST_MODEL,
    DEFAULT_ROFORMER_MODE,
    DEFAULT_ROFORMER_MODEL,
    DEFAULT_ROFORMER_VOCALS_MODEL,
    prefetch_roformer_models,
    resolve_roformer_model_dir,
)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="预下载 BS-RoFormer 模型 (download_checks.json + ckpt) 到持久目录"
    )
    p.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="目标目录 (默认项目 models/roformer; 可用 VIDEO2TEXT_ROFORMER_MODEL_DIR)",
    )
    p.add_argument(
        "--roformer-mode",
        choices=["single", "dual", "ensemble"],
        default=DEFAULT_ROFORMER_MODE,
    )
    p.add_argument(
        "--roformer-ensemble-preset",
        default=DEFAULT_ROFORMER_ENSEMBLE_PRESET,
    )
    p.add_argument("--roformer-model", default=DEFAULT_ROFORMER_MODEL)
    p.add_argument("--roformer-vocals-model", default=DEFAULT_ROFORMER_VOCALS_MODEL)
    p.add_argument("--roformer-inst-model", default=DEFAULT_ROFORMER_INST_MODEL)
    args = p.parse_args(argv)

    target = resolve_roformer_model_dir(args.model_dir)
    print(f"目标目录: {target}")
    print("提示: 若 GitHub 不可达, 请先设置 HTTPS_PROXY 再运行本脚本。")

    try:
        prefetch_roformer_models(
            model_dir=target,
            roformer_mode=args.roformer_mode,
            ensemble_preset=args.roformer_ensemble_preset,
            model_filename=args.roformer_model,
            vocals_model=args.roformer_vocals_model,
            inst_model=args.roformer_inst_model,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print("预下载完成 (manifest + ckpt 均已校验)。后续 separate_audio 将复用此目录。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
