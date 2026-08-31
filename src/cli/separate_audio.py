#!/usr/bin/env python3
"""Demucs 人声/伴奏分离；可选三轨 (背景/男声/女声)。"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from video2text.ffmpeg_util import configure_ffmpeg, extract_stereo_audio
from video2text.separate import _separate_vocal_stems, separate_three_stems


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="分离视频/音频中的人声与伴奏")
    p.add_argument("input", type=Path, help="输入视频或 wav")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="输出目录 (默认与输入同目录)",
    )
    p.add_argument("--stem", default=None, help="输出文件名前缀 (默认输入文件名)")
    p.add_argument(
        "--three-stems",
        action="store_true",
        help="输出三轨: 背景 + 男声 + 女声 (Demucs + 自动性别切分)",
    )
    p.add_argument(
        "--split-mode",
        choices=["whisper_f0", "srt_f0", "diarize"],
        default="srt_f0",
        help="三轨模式: srt_f0=SRT+F0+文本规则(推荐); whisper_f0; diarize",
    )
    p.add_argument(
        "--srt",
        type=Path,
        default=None,
        help="字幕 SRT (split-mode=srt_f0 必填; 默认同目录 <stem>.srt 或 *_whisper/*.srt)",
    )
    p.add_argument(
        "--gender-backend",
        choices=["slice", "sepformer"],
        default="slice",
        help="男女声组装: slice=按时间轴切 vocals; sepformer=SepFormer 双路(实验)",
    )
    p.add_argument(
        "--f0-threshold",
        type=float,
        default=200.0,
        help="基频阈值 Hz (>= 女声, < 男声; 默认 200, 可配合 --no-adaptive-f0)",
    )
    p.add_argument(
        "--no-adaptive-f0",
        action="store_true",
        help="禁用自适应 F0 阈值 (仅用 --f0-threshold)",
    )
    p.add_argument(
        "--min-voiced-ratio",
        type=float,
        default=0.25,
        help="F0 可信度下限 (默认 0.25)",
    )
    p.add_argument(
        "--demucs-model",
        default="htdemucs_ft",
        choices=["htdemucs", "htdemucs_ft", "mdx_extra_q"],
        help="Demucs 模型 (默认 htdemucs_ft, 人声/背景更干净)",
    )
    p.add_argument(
        "--demucs-shifts",
        type=int,
        default=1,
        help="Demucs shifts 次数 (>=2 更慢但更准, 默认 1)",
    )
    p.add_argument(
        "--whisper-model",
        default="base",
        help="split-mode=whisper_f0 时的 Whisper 模型 (默认 base)",
    )
    p.add_argument(
        "--language",
        default=None,
        help="Whisper 语言码, 如 ja/zh (whisper_f0 模式)",
    )
    p.add_argument(
        "--whisper-device",
        default=None,
        help="Whisper 设备 cpu/cuda",
    )
    p.add_argument(
        "--min-speakers",
        type=int,
        default=1,
        help="diarize 模式最少说话人数 (默认 1)",
    )
    p.add_argument(
        "--max-speakers",
        type=int,
        default=4,
        help="diarize 模式最多说话人数 (默认 4)",
    )
    p.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace Token (diarize 模式; 或设环境变量 HF_TOKEN)",
    )
    p.add_argument(
        "--slice-pad-ms",
        type=int,
        default=120,
        help="切片起点 padding 毫秒 (默认 120, 避免 onset 落在 SRT 边界外)",
    )
    p.add_argument(
        "--slice-pad-end-ms",
        type=int,
        default=None,
        help="切片终点 padding 毫秒 (默认同 --slice-pad-ms)",
    )
    p.add_argument(
        "--align-short-segments",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="短句 onset/VAD 边界收紧 (srt_f0 默认开启)",
    )
    p.add_argument(
        "--fill-gap-ms",
        type=int,
        default=400,
        help="段间空隙回填上限毫秒 (默认 400; 0 禁用)",
    )
    p.add_argument(
        "--shout-min-ms",
        type=int,
        default=600,
        help="喊名最短切片毫秒 (默认 600)",
    )
    p.add_argument(
        "--shout-tail-pad-ms",
        type=int,
        default=250,
        help="喊名尾音延长毫秒 (默认 250)",
    )
    p.add_argument(
        "--shout-all-islands",
        action="store_true",
        help="喊名 OCR 长窗内所有语声岛都归入该段 (默认仅长窗>5s 自动启用)",
    )
    p.add_argument(
        "--recover-window-vocals",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="SRT 窗内未分配人声回收 (srt_f0 默认开启)",
    )
    p.add_argument(
        "--speaker-map",
        type=Path,
        default=None,
        help="可选 speaker_map.json, 按 index 覆盖 gender 后重切",
    )
    p.add_argument(
        "--recover-vocal-bleed",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="从 no_vocals 回收 Demucs 泄漏语声并清理背景轨 (srt_f0 默认开启)",
    )
    p.add_argument("--ffmpeg", default=None)
    p.add_argument("--ffprobe", default=None)
    args = p.parse_args(argv)

    src = args.input.expanduser().resolve()
    if not src.is_file():
        print(f"找不到输入: {src}", file=sys.stderr)
        return 1

    try:
        configure_ffmpeg(args.ffmpeg, args.ffprobe)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    out_dir = args.out_dir.expanduser().resolve() if args.out_dir else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.stem or src.stem

    srt_path = None
    if args.srt:
        srt_path = args.srt.expanduser().resolve()
    elif args.three_stems and args.split_mode == "srt_f0":
        for candidate in (
            src.parent / f"{stem}_whisper" / f"{stem}.srt",
            src.parent / f"{stem}.srt",
            out_dir / f"{stem}.srt",
        ):
            if candidate.is_file():
                srt_path = candidate.resolve()
                break

    if args.three_stems and args.split_mode == "srt_f0":
        if srt_path is None or not srt_path.is_file():
            print(
                "split-mode=srt_f0 需要 --srt 或同目录下可找到的 .srt 字幕",
                file=sys.stderr,
            )
            return 1

    with tempfile.TemporaryDirectory(prefix="v2t_stems_") as tmp:
        work = Path(tmp)
        if src.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg"}:
            audio_wav = src
        else:
            audio_wav = work / "source.wav"
            print(f"提取音轨: {src.name}")
            extract_stereo_audio(src, audio_wav)

        if args.three_stems:
            try:
                separate_three_stems(
                    audio_wav,
                    out_dir,
                    stem,
                    work,
                    split_mode=args.split_mode,
                    gender_backend=args.gender_backend,
                    srt_path=srt_path,
                    f0_threshold=args.f0_threshold,
                    adaptive_threshold=not args.no_adaptive_f0,
                    min_voiced_ratio=args.min_voiced_ratio,
                    whisper_model=args.whisper_model,
                    language=args.language,
                    whisper_device=args.whisper_device,
                    min_speakers=args.min_speakers,
                    max_speakers=args.max_speakers,
                    hf_token=args.hf_token,
                    demucs_model=args.demucs_model,
                    demucs_shifts=max(1, args.demucs_shifts),
                    slice_pad_ms=max(0, args.slice_pad_ms),
                    slice_pad_end_ms=args.slice_pad_end_ms,
                    align_short_segments=args.align_short_segments,
                    fill_gap_ms=max(0, args.fill_gap_ms),
                    shout_min_ms=max(0, args.shout_min_ms),
                    shout_tail_pad_ms=max(0, args.shout_tail_pad_ms),
                    shout_all_islands=args.shout_all_islands,
                    recover_window_vocals=args.recover_window_vocals,
                    speaker_map_path=(
                        args.speaker_map.expanduser().resolve()
                        if args.speaker_map
                        else None
                    ),
                    recover_vocal_bleed=args.recover_vocal_bleed,
                )
            except (RuntimeError, ValueError) as exc:
                print(str(exc), file=sys.stderr)
                return 2
            return 0

        print("Demucs 人声分离...")
        instrumental, vocals = _separate_vocal_stems(
            audio_wav,
            work,
            demucs_model=args.demucs_model,
            demucs_shifts=max(1, args.demucs_shifts),
        )

        vocals_out = out_dir / f"{stem}_vocals.wav"
        bgm_out = out_dir / f"{stem}_no_vocals.wav"
        shutil.copy2(vocals, vocals_out)
        shutil.copy2(instrumental, bgm_out)
        print(f"已写入: {vocals_out}")
        print(f"已写入: {bgm_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
