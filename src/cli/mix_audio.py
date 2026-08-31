#!/usr/bin/env python3
"""混合 AI 配音与背景音乐。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from video2text.ffmpeg_util import configure_ffmpeg
from video2text.mix import _mix_bgm_and_dub


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="混合配音轨与伴奏轨")
    p.add_argument("--dub", type=Path, required=True, help="AI 配音 wav")
    p.add_argument("--bgm", type=Path, required=True, help="伴奏/无人声 wav")
    p.add_argument("-o", "--output", type=Path, required=True, help="输出混合 wav")
    p.add_argument("--bgm-volume", type=float, default=1.0)
    p.add_argument("--dub-volume", type=float, default=1.0)
    p.add_argument("--ffmpeg", default=None)
    p.add_argument("--ffprobe", default=None)
    args = p.parse_args(argv)

    dub_path = args.dub.expanduser().resolve()
    bgm_path = args.bgm.expanduser().resolve()
    out_path = args.output.expanduser().resolve()
    for label, path in (("配音", dub_path), ("伴奏", bgm_path)):
        if not path.is_file():
            print(f"找不到{label}: {path}", file=sys.stderr)
            return 1

    try:
        configure_ffmpeg(args.ffmpeg, args.ffprobe)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"混合 -> {out_path.name}")
    _mix_bgm_and_dub(
        bgm_path, dub_path, out_path, args.bgm_volume, args.dub_volume
    )
    print(f"完成: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
