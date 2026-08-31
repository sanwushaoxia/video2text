#!/usr/bin/env python3
"""完整视频处理流水线（转写 / OCR / 翻译 / 配音 / 烧录）。"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from video2text.pipeline import main


if __name__ == "__main__":
    raise SystemExit(main())
