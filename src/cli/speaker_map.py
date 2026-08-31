#!/usr/bin/env python3
"""自动生成 speaker_map.json（多角色配音）。"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from video2text.speaker_map import main


if __name__ == "__main__":
    raise SystemExit(main())
