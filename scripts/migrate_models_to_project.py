#!/usr/bin/env python3
"""将 RoFormer / Whisper / HF / SpeechBrain 模型迁入项目 models/ 目录。"""
from __future__ import annotations

import argparse
import shutil
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
from video2text.paths import (  # noqa: E402
    ensure_models_subdirs,
    huggingface_home,
    legacy_huggingface_home,
    legacy_roformer_model_dir,
    legacy_whisper_model_dir,
    models_dir,
    repo_root,
    roformer_model_dir,
    whisper_model_dir,
)


def _dir_has_real_content(path: Path) -> bool:
    if not path.is_dir():
        return path.is_file()
    for item in path.rglob("*"):
        if item.is_file() and item.name != ".gitkeep":
            return True
    return False


def _transfer_dir_contents(
    src: Path, dst: Path, *, move: bool, dry_run: bool, label: str = ""
) -> None:
    if not src.is_dir():
        print(f"  跳过 (源不存在): {src}")
        return
    dst.mkdir(parents=True, exist_ok=True)
    items = [p for p in src.iterdir() if p.name != ".gitkeep"]
    if not items:
        print(f"  跳过 (源为空): {src}")
        return
    prefix = f"{label} " if label else ""
    for item in items:
        target = dst / item.name
        if target.exists():
            print(f"  {prefix}跳过 (已存在): {target.name}")
            continue
        action = "移动" if move else "复制"
        print(f"  {prefix}{action}: {item} -> {target}")
        if dry_run:
            continue
        if move:
            shutil.move(str(item), str(target))
        elif item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def _transfer(src: Path, dst: Path, *, move: bool, dry_run: bool) -> None:
    if not src.exists():
        print(f"  跳过 (源不存在): {src}")
        return
    if _dir_has_real_content(dst):
        print(f"  跳过 (目标已有内容): {dst}")
        return
    action = "移动" if move else "复制"
    print(f"  {action}: {src} -> {dst}")
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if move:
        if dst.exists():
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        shutil.move(str(src), str(dst))
    else:
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def _transfer_glob(src_dir: Path, dst_dir: Path, pattern: str, *, move: bool, dry_run: bool) -> None:
    if not src_dir.is_dir():
        print(f"  跳过 (源目录不存在): {src_dir}")
        return
    files = list(src_dir.glob(pattern))
    if not files:
        print(f"  跳过 (无匹配 {pattern}): {src_dir}")
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in files:
        dst = dst_dir / src.name
        if dst.is_file():
            print(f"  跳过 (已存在): {dst}")
            continue
        action = "移动" if move else "复制"
        print(f"  {action}: {src} -> {dst}")
        if dry_run:
            continue
        if move:
            shutil.move(str(src), str(dst))
        else:
            shutil.copy2(src, dst)


def _verify_roformer(model_dir: Path) -> list[str]:
    missing: list[str] = []
    ckpts = _ensemble_ckpt_filenames(DEFAULT_ROFORMER_ENSEMBLE_PRESET)
    for name in ckpts + ["download_checks.json"]:
        if not (model_dir / name).is_file():
            missing.append(name)
    for yaml in (
        "config_melbandroformer_inst.yaml",
        "config_mel_band_roformer_instrumental_becruily.yaml",
    ):
        if not (model_dir / yaml).is_file():
            missing.append(yaml)
    return missing


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="迁移外部模型缓存到项目 models/")
    p.add_argument("--dry-run", action="store_true", help="仅打印计划，不执行")
    p.add_argument(
        "--copy",
        action="store_true",
        help="复制而非移动 (默认移动)",
    )
    p.add_argument(
        "--skip-huggingface",
        action="store_true",
        help="跳过 HuggingFace 缓存迁移",
    )
    args = p.parse_args(argv)
    move = not args.copy
    dry = args.dry_run

    print(f"项目根: {repo_root()}")
    print(f"目标 models: {models_dir()}\n")

    if not dry:
        ensure_models_subdirs()

    print("[1/4] RoFormer")
    legacy_rf = legacy_roformer_model_dir()
    if legacy_rf:
        _transfer_dir_contents(
            legacy_rf, roformer_model_dir(), move=move, dry_run=dry, label="RoFormer"
        )
    else:
        print("  无旧 RoFormer 目录 (AppData 或 ~/.cache/video2text/roformer_models)")

    print("\n[2/4] Whisper")
    _transfer_glob(
        legacy_whisper_model_dir(),
        whisper_model_dir(),
        "*.pt",
        move=move,
        dry_run=dry,
    )

    print("\n[3/4] HuggingFace (pyannote 等)")
    if args.skip_huggingface:
        print("  已跳过 (--skip-huggingface)")
    else:
        hf_dst = huggingface_home()
        hf_src = legacy_huggingface_home()
        if _dir_has_real_content(hf_dst):
            print(f"  跳过 (目标已有内容): {hf_dst}")
        else:
            _transfer_dir_contents(hf_src, hf_dst, move=move, dry_run=dry, label="HF")
        hub_src = hf_src / "hub"
        hub_dst = hf_dst / "hub"
        if hub_src.is_dir():
            _transfer_dir_contents(hub_src, hub_dst, move=move, dry_run=dry, label="HF hub")

    print("\n[4/4] SpeechBrain (pretrained_models/)")
    legacy_sb = repo_root() / "pretrained_models"
    dst_sb = models_dir() / "speechbrain"
    if legacy_sb.is_dir():
        _transfer_dir_contents(legacy_sb, dst_sb, move=move, dry_run=dry, label="SB")
    else:
        print(f"  无 {legacy_sb}")

    print("\n校验 RoFormer (instrumental_full):")
    missing = _verify_roformer(roformer_model_dir())
    if missing:
        print(f"  缺失: {', '.join(missing)}")
        print("  请运行: python scripts/prefetch_roformer_models.py")
        return 1
    print(f"  OK: {roformer_model_dir()}")
    print("\n迁移完成。运行: python scripts/verify_diarize_setup.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
