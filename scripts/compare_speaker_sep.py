#!/usr/bin/env python3
"""对比多角色分离输出: 各 speaker 轨能量、重叠段处理统计。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def _glob_speaker_wavs(out_dir: Path, stem: str) -> list[Path]:
    return sorted(out_dir.glob(f"{stem}_speaker_*.wav"))


def rms(path: Path, t0: float, t1: float) -> float:
    y, sr = sf.read(path, always_2d=True)
    g0, g1 = int(t0 * sr), int(t1 * sr)
    seg = y[g0:g1]
    if seg.size == 0:
        return 0.0
    mono = seg.mean(axis=1) if seg.ndim > 1 else seg
    return float(np.sqrt(np.mean(mono**2)))


def active_sec(path: Path, threshold_ratio: float = 0.04) -> float:
    y, sr = sf.read(path, always_2d=True)
    mono = y.mean(axis=1) if y.ndim > 1 else y
    peak = float(np.max(np.abs(mono)))
    if peak < 1e-8:
        return 0.0
    thr = peak * threshold_ratio
    return float(np.count_nonzero(np.abs(mono) >= thr)) / sr


def load_report(out_dir: Path, stem: str) -> dict | None:
    report = out_dir / f"{stem}_split_report.json"
    if not report.is_file():
        return None
    return json.loads(report.read_text(encoding="utf-8"))


def main() -> int:
    p = argparse.ArgumentParser(description="多角色分离 A/B 对比")
    p.add_argument("dirs", nargs="+", type=Path, help="输出目录 (含 *_speaker_*.wav)")
    p.add_argument("--stem", default="桔梗犬夜叉片段")
    p.add_argument(
        "--windows",
        nargs="*",
        default=["93:104", "129:142.5"],
        help="RMS 窗口 t0:t1 (秒), 默认两段重叠测试窗",
    )
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()

    windows: list[tuple[str, float, float]] = []
    for w in args.windows:
        if ":" not in w:
            continue
        a, b = w.split(":", 1)
        windows.append((w, float(a), float(b)))

    rows: list[dict] = []
    print(f"{'dir':<24} {'track':<28} {'active_s':>8} {'window':>12} {'rms':>10}")
    print("-" * 88)

    for d in args.dirs:
        d = d.expanduser().resolve()
        tracks = _glob_speaker_wavs(d, args.stem)
        if not tracks:
            print(f"跳过 {d}: 无 {args.stem}_speaker_*.wav")
            continue

        meta = load_report(d, args.stem)
        stats = (meta or {}).get("stats", {})
        overlap_bss = (meta or {}).get("overlap_bss")

        for track in tracks:
            act = active_sec(track)
            for label, t0, t1 in windows:
                r = rms(track, t0, t1)
                print(f"{d.name:<24} {track.name:<28} {act:8.2f} {label:>12} {r:10.6f}")
                rows.append(
                    {
                        "dir": str(d),
                        "track": track.name,
                        "active_sec": round(act, 2),
                        "window": label,
                        "t0": t0,
                        "t1": t1,
                        "rms": round(r, 6),
                    }
                )

        if stats:
            print(
                f"  report: {stats.get('speaker_count')} speakers, "
                f"coverage {stats.get('coverage_ratio', 0):.0%}, "
                f"unassigned {stats.get('unassigned_vocal_sec', 0):.1f}s"
            )
            rows.append({"dir": str(d), "stats": stats})
        if overlap_bss:
            print(
                f"  overlap_bss: {overlap_bss.get('overlap_count')} regions, "
                f"{overlap_bss.get('overlap_separated_sec', 0):.1f}s "
                f"({overlap_bss.get('backend_used', 'n/a')})"
            )
            rows.append({"dir": str(d), "overlap_bss": overlap_bss})

    if args.json_out:
        args.json_out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写入 {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
