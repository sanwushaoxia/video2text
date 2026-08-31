#!/usr/bin/env python3
"""字幕转 AI 配音（edge / RVC / GPT-SoVITS）。"""
from __future__ import annotations

import argparse
import sys

from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from video2text.pipeline import main as pipeline_main


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="从 SRT 生成 AI 配音 (等价于 transcribe_video.py --dub-only)",
        add_help=True,
    )
    p.add_argument("video", help="输入视频")
    p.add_argument("--srt", default=None, help="配音用 SRT (或 --dub-srt)")
    p.add_argument("--dub-srt", default=None, help="配音专用 SRT")
    p.add_argument("--dub-lang", required=True, help="配音语言, 如 zh/en/ja")
    p.add_argument("--out-dir", default=None)
    p.add_argument(
        "passthrough",
        nargs=argparse.REMAINDER,
        help="其余参数原样传给引擎 (如 --dub-engine edge --dub-speaker-map map.json)",
    )
    args, unknown = p.parse_known_args(argv)

    forward: list[str] = [args.video, "--dub-only", f"--dub-lang={args.dub_lang}"]
    if args.srt:
        forward.append(f"--srt={args.srt}")
    if args.dub_srt:
        forward.append(f"--dub-srt={args.dub_srt}")
    if args.out_dir:
        forward.append(f"--out-dir={args.out_dir}")
    forward.extend(args.passthrough or [])
    forward.extend(unknown)
    return pipeline_main(forward)


if __name__ == "__main__":
    raise SystemExit(main())
