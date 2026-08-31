#!/usr/bin/env python3
"""检查 src/ 模块的 import 与模块级变量 (拆包后防回归).

用法:
  python tools/check_imports.py
  PYTHONPATH=src python tools/check_imports.py --smoke
"""
from __future__ import annotations

import argparse
import ast
import importlib
import pkgutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

# 模块级变量: 名称 -> 应定义于哪个文件
REQUIRED_MODULE_GLOBALS: dict[str, list[str]] = {
    "video2text.ffmpeg_util": [
        "_FFMPEG_BIN",
        "_FFPROBE_BIN",
        "_FFMPEG_SUBTITLES_FILTER",
        "_FFMPEG_FOR_SUBTITLES",
    ],
    "video2text.ocr": [
        "_CUDA_OCR_WORKING",
        "_CUDA_WARNED",
        "_NVIDIA_OCR_DLL_PREPARED",
        "_OCR_WORKER",
    ],
    "video2text.dub": [
        "_DUB_GENDER_DEFAULT_VOICES",
        "_SOVITS_API_VERSION",
    ],
}

STDLIB_USED_BY: dict[str, set[str]] = {}


def _collect_stdlib_usage(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
            for alias in node.names:
                imported.add(alias.name)
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            used.add(node.value.id)
    common = {
        "os",
        "sys",
        "re",
        "json",
        "shutil",
        "subprocess",
        "tempfile",
        "wave",
        "asyncio",
        "argparse",
        "platform",
    }
    return {m for m in used if m in common and m not in imported}


def check_module_globals() -> list[str]:
    errors: list[str] = []
    for mod_name, names in REQUIRED_MODULE_GLOBALS.items():
        mod = importlib.import_module(mod_name)
        for name in names:
            if not hasattr(mod, name):
                errors.append(f"{mod_name} 缺少模块变量 {name}")
    return errors


def check_stdlib_imports() -> list[str]:
    errors: list[str] = []
    for path in sorted((SRC / "video2text").glob("*.py")):
        missing = _collect_stdlib_usage(path)
        for m in sorted(missing):
            errors.append(f"{path.relative_to(ROOT)} 使用了 {m} 但未 import")
    return errors


def import_all_modules() -> list[str]:
    errors: list[str] = []
    import video2text

    names = [
        m.name
        for m in pkgutil.walk_packages(video2text.__path__, video2text.__name__ + ".")
    ]
    cli_names = [
        f"cli.{p.stem}"
        for p in (SRC / "cli").glob("*.py")
        if p.stem not in ("__init__", "_bootstrap")
    ]
    for name in sorted(set(names + cli_names)):
        try:
            importlib.import_module(name)
        except Exception as exc:
            errors.append(f"import {name} 失败: {exc}")
    return errors


def smoke_configure_ffmpeg() -> list[str]:
    errors: list[str] = []
    try:
        from video2text.ffmpeg_util import configure_ffmpeg, ffprobe_bin, _require_ffmpeg

        configure_ffmpeg()
        _require_ffmpeg()
        if ffprobe_bin() is None:
            errors.append("configure_ffmpeg 后 ffprobe_bin() 仍为 None (可接受但需 ffmpeg 回退)")
    except Exception as exc:
        errors.append(f"smoke configure_ffmpeg: {exc}")
    return errors


def smoke_media_duration() -> list[str]:
    errors: list[str] = []
    vocals = ROOT / "out" / "桔梗犬夜叉片段_vocals.wav"
    if not vocals.is_file():
        return errors
    try:
        from video2text.ffmpeg_util import configure_ffmpeg
        from video2text.media import get_media_duration

        configure_ffmpeg()
        dur = get_media_duration(vocals)
        if dur <= 0:
            errors.append(f"get_media_duration 返回无效时长: {dur}")
    except Exception as exc:
        errors.append(f"smoke get_media_duration: {exc}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="检查 video2text 模块 import 完整性")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="额外运行 configure_ffmpeg / get_media_duration 冒烟测试",
    )
    args = parser.parse_args()

    all_errors: list[str] = []
    all_errors.extend(check_module_globals())
    all_errors.extend(check_stdlib_imports())
    all_errors.extend(import_all_modules())
    if args.smoke:
        all_errors.extend(smoke_configure_ffmpeg())
        all_errors.extend(smoke_media_duration())

    if all_errors:
        print("检查失败:\n", file=sys.stderr)
        for err in all_errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print("check_imports: 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
