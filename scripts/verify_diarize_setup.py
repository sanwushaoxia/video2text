#!/usr/bin/env python3
"""检查 multi-stems diarize 部署依赖与模型缓存。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from video2text.bs_roformer_split import (  # noqa: E402
    DEFAULT_ROFORMER_ENSEMBLE_PRESET,
    _ensemble_ckpt_filenames,
)
from video2text.paths import ensure_huggingface_env, huggingface_home, roformer_model_dir  # noqa: E402
from video2text.diarization import (  # noqa: E402
    DEFAULT_DIARIZATION_MODEL,
    _default_hf_token_file,
    _resolve_hf_token,
)


def _check(name: str, ok: bool, detail: str = "") -> bool:
    mark = "OK" if ok else "FAIL"
    line = f"  [{mark}] {name}"
    if detail:
        line += f" — {detail}"
    print(line)
    return ok


def main() -> int:
    print("diarize multi-stems 部署检查\n")
    all_ok = True

    try:
        import pyannote.audio  # noqa: F401

        all_ok &= _check("pyannote.audio", True)
    except ImportError:
        all_ok &= _check("pyannote.audio", False, "pip install -r requirements.txt")

    try:
        import speechbrain  # noqa: F401

        all_ok &= _check("speechbrain (diarize_bss)", True)
    except ImportError:
        all_ok &= _check("speechbrain (diarize_bss)", False, "pip install -r requirements.txt")

    ensure_huggingface_env()
    model_dir = roformer_model_dir()
    ckpts = _ensemble_ckpt_filenames(DEFAULT_ROFORMER_ENSEMBLE_PRESET)
    roformer_ok = all((model_dir / c).is_file() for c in ckpts)
    all_ok &= _check(
        "RoFormer 缓存",
        roformer_ok,
        str(model_dir) if roformer_ok else f"运行 prefetch_roformer_models.py 或 migrate_models_to_project.py → {model_dir}",
    )

    token_ok = False
    token_detail = ""
    try:
        _resolve_hf_token(None)
        token_ok = True
    except RuntimeError as exc:
        token_detail = str(exc).split("\n")[0]
    all_ok &= _check(
        "HF Token",
        token_ok,
        "已配置" if token_ok else f"设置 HF_TOKEN 或 {_default_hf_token_file()} — {token_detail}",
    )

    if token_ok:
        ensure_huggingface_env()
        print(f"  HF 缓存: {huggingface_home()}")
        try:
            from pyannote.audio import Pipeline

            token = _resolve_hf_token(None)
            Pipeline.from_pretrained(DEFAULT_DIARIZATION_MODEL, token=token)
            all_ok &= _check("pyannote pipeline 加载", True, DEFAULT_DIARIZATION_MODEL)
        except Exception as exc:
            all_ok &= _check("pyannote pipeline 加载", False, str(exc)[:120])

    print()
    if all_ok:
        print("全部通过。可运行: bash scripts/run_multi_stems_diarize.sh <video>")
        return 0
    print("存在未就绪项, 请按上方提示修复。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
