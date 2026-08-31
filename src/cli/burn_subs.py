#!/usr/bin/env python3
"""将 SRT 字幕烧录到视频画面下方。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from video2text.ffmpeg_util import configure_ffmpeg
from video2text.render import render_video


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="烧录字幕到视频")
    p.add_argument("video", type=Path, help="输入视频")
    p.add_argument("--srt", type=Path, required=True, help="字幕 SRT 文件")
    p.add_argument("-o", "--output", type=Path, default=None, help="输出视频路径")
    p.add_argument("--font-size", type=int, default=24)
    p.add_argument("--margin-v", type=int, default=36)
    p.add_argument("--box", action="store_true", help="使用不透明黑底字幕框")
    p.add_argument("--ffmpeg", default=None)
    p.add_argument("--ffprobe", default=None)
    args = p.parse_args(argv)

    video_path = args.video.expanduser().resolve()
    srt_path = args.srt.expanduser().resolve()
    if not video_path.is_file():
        print(f"找不到视频: {video_path}", file=sys.stderr)
        return 1
    if not srt_path.is_file():
        print(f"找不到 SRT: {srt_path}", file=sys.stderr)
        return 1

    try:
        configure_ffmpeg(args.ffmpeg, args.ffprobe)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    out_path = (
        args.output.expanduser().resolve()
        if args.output
        else video_path.parent / f"{video_path.stem}_subtitled{video_path.suffix}"
    )
    print(f"烧录 {srt_path.name} -> {out_path.name}")
    render_video(
        video_path,
        out_path,
        srt_path=srt_path,
        font_size=args.font_size,
        margin_v=args.margin_v,
        box=args.box,
    )
    print(f"完成: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
