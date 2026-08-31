#!/usr/bin/env python3
"""提取指定语言字幕（Whisper 音频转写 或 画面 OCR）。"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from video2text.ffmpeg_util import configure_ffmpeg, extract_audio
from video2text.ocr import extract_burned_in_subtitles
from video2text.srt import segments_to_srt
from video2text.translate import (
    _lang_short_code,
    translate_segments,
    write_translated_srt,
)
from video2text.whisper_transcribe import detect_and_transcribe, load_whisper_model


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="提取视频字幕 (Whisper / OCR)")
    p.add_argument("video", type=Path, help="输入视频")
    p.add_argument(
        "--source",
        choices=["whisper", "ocr"],
        default="whisper",
        help="字幕来源 (默认 whisper)",
    )
    p.add_argument("--model", default="base", help="Whisper 模型 (source=whisper)")
    p.add_argument("--language", default=None, help="Whisper 语言代码, 如 zh/ja")
    p.add_argument(
        "--task",
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Whisper 任务 (默认 transcribe)",
    )
    p.add_argument("--device", default=None, help="Whisper 设备 cpu/cuda")
    p.add_argument("--ocr-lang", default="zh", help="OCR 语言 (source=ocr)")
    p.add_argument("--ocr-fps", type=float, default=2.0)
    p.add_argument("--ocr-crop-ratio", type=float, default=0.28)
    p.add_argument("--ocr-use-gpu", action="store_true")
    p.add_argument("--ocr-no-det", action="store_true")
    p.add_argument("--ocr-workers", type=int, default=None)
    p.add_argument("--translate-to", default=None, metavar="LANG", help="翻译字幕目标语言")
    p.add_argument("--translate-from", default=None, metavar="LANG")
    p.add_argument(
        "--translate-engine",
        default="auto",
        choices=["auto", "bing", "alibaba", "google", "baidu"],
    )
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--ffmpeg", default=None)
    p.add_argument("--ffprobe", default=None)
    args = p.parse_args(argv)

    video_path = args.video.expanduser().resolve()
    if not video_path.is_file():
        print(f"找不到视频: {video_path}", file=sys.stderr)
        return 1

    try:
        configure_ffmpeg(args.ffmpeg, args.ffprobe)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    out_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir
        else video_path.parent / f"{video_path.stem}_whisper"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    srt_path = out_dir / f"{video_path.stem}.srt"
    txt_path = out_dir / f"{video_path.stem}.txt"
    json_path = out_dir / f"{video_path.stem}.json"

    if args.source == "ocr":
        print(f"OCR 提取字幕 (语言={args.ocr_lang})")
        with tempfile.TemporaryDirectory(prefix="v2t_ocr_") as tmp:
            segments = extract_burned_in_subtitles(
                video_path,
                Path(tmp),
                ocr_lang=args.ocr_lang,
                fps=args.ocr_fps,
                crop_ratio=args.ocr_crop_ratio,
                use_gpu=args.ocr_use_gpu,
                no_det=args.ocr_no_det,
                workers=args.ocr_workers,
            )
        full_text = " ".join((s.get("text") or "").strip() for s in segments).strip()
        detected, conf, used_lang, device = args.ocr_lang, 1.0, args.ocr_lang, "ocr"
    else:
        model, device = load_whisper_model(args.model, args.device)
        with tempfile.TemporaryDirectory(prefix="v2t_audio_") as tmp:
            wav_path = Path(tmp) / "audio.wav"
            extract_audio(video_path, wav_path)
            result, detected, conf = detect_and_transcribe(
                model, wav_path, language=args.language, task=args.task
            )
        segments = result.get("segments") or []
        full_text = (result.get("text") or "").strip()
        used_lang = result.get("language") or detected

    if args.translate_to:
        source_lang = args.translate_from or used_lang or detected
        segments = translate_segments(
            segments, source_lang, args.translate_to, args.translate_engine
        )
        srt_path = write_translated_srt(segments, srt_path, args.translate_to)
        full_text = " ".join((s.get("text") or "").strip() for s in segments).strip()
        txt_path = out_dir / f"{video_path.stem}_{_lang_short_code(args.translate_to)}.txt"

    srt_path.write_text(segments_to_srt(segments), encoding="utf-8")
    txt_path.write_text(full_text + ("\n" if full_text else ""), encoding="utf-8")
    meta = {
        "video": str(video_path),
        "subs_source": args.source,
        "detected_language": detected,
        "used_language": used_lang,
        "text": full_text,
        "segments": segments,
    }
    json_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入: {srt_path}")
    print(f"已写入: {txt_path}")
    print(f"已写入: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
