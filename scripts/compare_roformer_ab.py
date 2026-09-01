#!/usr/bin/env python3
"""对比 no_vocals 在 #22 / #26 窗口的 RMS (客观 A/B)。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

WINDOWS = [
    ("#22 BGM 93-104s", 93.0, 104.0),
    ("#26 好温暖 129-142.5s", 129.0, 142.5),
    ("full", 0.0, 160.0),
]


def rms(path: Path, t0: float, t1: float) -> float:
    y, sr = sf.read(path, always_2d=True)
    g0, g1 = int(t0 * sr), int(t1 * sr)
    seg = y[g0:g1]
    if seg.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(seg**2)))


def nv_v_ratio(nv_path: Path, v_path: Path, t0: float, t1: float) -> float:
    y_nv, sr = sf.read(nv_path, always_2d=True)
    y_v, _ = sf.read(v_path, always_2d=True)
    g0, g1 = int(t0 * sr), int(t1 * sr)
    nv = np.sqrt(np.mean(y_nv[g0:g1] ** 2))
    v = np.sqrt(np.mean(y_v[g0:g1] ** 2))
    if v < 1e-7:
        return float("inf") if nv > 1e-4 else 0.0
    return float(nv / v)


def main() -> int:
    p = argparse.ArgumentParser(description="RoFormer A/B no_vocals RMS 对比")
    p.add_argument("dirs", nargs="+", type=Path, help="输出目录 (含 *_no_vocals.wav)")
    p.add_argument("--stem", default="桔梗犬夜叉片段")
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()

    rows: list[dict] = []
    print(f"{'dir':<24} {'window':<22} {'nv_rms':>10} {'nv/v':>8}")
    print("-" * 68)
    for d in args.dirs:
        d = d.expanduser().resolve()
        nv = d / f"{args.stem}_no_vocals.wav"
        voc = d / f"{args.stem}_vocals.wav"
        if not nv.is_file():
            print(f"跳过 {d}: 无 {nv.name}")
            continue
        for label, t0, t1 in WINDOWS:
            nv_r = rms(nv, t0, t1)
            ratio = nv_v_ratio(nv, voc, t0, t1) if voc.is_file() else None
            ratio_s = f"{ratio:.2f}" if ratio is not None and ratio != float("inf") else "inf"
            print(f"{d.name:<24} {label:<22} {nv_r:10.6f} {ratio_s:>8}")
            rows.append(
                {
                    "dir": str(d),
                    "window": label,
                    "t0": t0,
                    "t1": t1,
                    "nv_rms": round(nv_r, 6),
                    "nv_v_ratio": None if ratio == float("inf") else round(ratio, 4) if ratio else None,
                }
            )

    if args.json_out:
        args.json_out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写入 {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
