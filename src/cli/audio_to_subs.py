#!/usr/bin/env python3
"""Whisper 音频转字幕。"""
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
from video2text.srt import segments_to_srt
from video2text.whisper_transcribe import detect_and_transcribe, load_whisper_model


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Whisper 音频转字幕")
    p.add_argument("video", type=Path, help="输入视频或音频")
    p.add_argument("--model", default="base")
    p.add_argument("--language", default=None)
    p.add_argument(
        "--task",
        default="transcribe",
        choices=["transcribe", "translate"],
    )
    p.add_argument("--device", default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--ffmpeg", default=None)
    p.add_argument("--ffprobe", default=None)
    args = p.parse_args(argv)

    video_path = args.video.expanduser().resolve()
    if not video_path.is_file():
        print(f"找不到输入: {video_path}", file=sys.stderr)
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

    model, device = load_whisper_model(args.model, args.device)
    with tempfile.TemporaryDirectory(prefix="v2t_whisper_") as tmp:
        wav_path = Path(tmp) / "audio.wav"
        extract_audio(video_path, wav_path)
        result, detected, conf = detect_and_transcribe(
            model, wav_path, language=args.language, task=args.task
        )

    segments = result.get("segments") or []
    full_text = (result.get("text") or "").strip()
    srt_path = out_dir / f"{video_path.stem}.srt"
    txt_path = out_dir / f"{video_path.stem}.txt"
    json_path = out_dir / f"{video_path.stem}.json"

    srt_path.write_text(segments_to_srt(segments), encoding="utf-8")
    txt_path.write_text(full_text + ("\n" if full_text else ""), encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "input": str(video_path),
                "model": args.model,
                "device": device,
                "detected_language": detected,
                "confidence": conf,
                "text": full_text,
                "segments": segments,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"语言: {detected} ({conf:.1%})")
    print(f"已写入: {srt_path}")
    print(f"已写入: {txt_path}")
    print(f"已写入: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
