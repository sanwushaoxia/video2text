"""从 Demucs no_vocals 泄漏轨回收语声, 并清理背景轨。"""
from __future__ import annotations

from pathlib import Path

from video2text.f0_analysis import load_vocals_mono
from video2text.segment_align import detect_speech_islands


def _segment_confidence(seg: dict) -> float:
    raw = seg.get("confidence")
    if raw is None:
        return 0.5
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.5


def _iter_segment_windows(
    seg: dict,
    *,
    pad_sec: float,
) -> tuple[float, float]:
    start = float(seg.get("srt_start", seg.get("start", 0.0))) - pad_sec
    end = float(seg.get("srt_end", seg.get("end", start))) + pad_sec
    return max(0.0, start), end


def _leak_islands_for_segment(
    no_vocals,
    sr: int,
    seg: dict,
    win_start: float,
    win_end: float,
    *,
    pad_sec: float,
    threshold_ratio: float,
) -> list[tuple[float, float]]:
    islands = detect_speech_islands(
        no_vocals,
        sr,
        win_start,
        win_end,
        pad_start_sec=pad_sec,
        pad_end_sec=pad_sec,
        threshold_ratio=threshold_ratio,
    )
    if islands:
        return islands

    stored = seg.get("speech_islands") or []
    parsed: list[tuple[float, float]] = []
    for item in stored:
        if isinstance(item, dict):
            parsed.append((float(item["start"]), float(item["end"])))
    return parsed


def _segment_needs_bleed_recovery(
    seg: dict,
    vocals,
    no_vocals,
    sr: int,
    *,
    pad_sec: float,
    min_nv_voc_ratio: float = 1.25,
) -> bool:
    import numpy as np

    cue = seg.get("align_cue_type")
    if cue in ("shout", "phrase_long_window"):
        return True

    win_start, win_end = _iter_segment_windows(seg, pad_sec=pad_sec)
    g0 = max(0, int(win_start * sr))
    g1 = min(len(vocals), int(win_end * sr))
    if g1 <= g0:
        return False

    v_rms = float(np.sqrt(np.mean(vocals[g0:g1] ** 2)))
    nv_rms = float(np.sqrt(np.mean(no_vocals[g0:g1] ** 2)))
    if nv_rms < 1e-7:
        return False
    if v_rms < 1e-7:
        return nv_rms > 1e-4
    return nv_rms / v_rms >= min_nv_voc_ratio


def apply_vocal_bleed_recovery(
    assignments: list[dict],
    vocals_path: Path,
    no_vocals_path: Path,
    male_path: Path,
    female_path: Path,
    *,
    sr: int = 44100,
    window_pad_sec: float = 0.12,
    leak_ratio: float = 0.55,
    transfer_gain: float = 0.92,
    bgm_attenuate: float = 0.95,
    island_threshold_ratio: float = 0.06,
) -> dict:
    """
    在标注语声窗内, 将 Demucs 漏进 no_vocals 的人声转移到男/女轨, 并衰减背景轨。
    """
    import numpy as np
    import soundfile as sf

    vocals, _ = load_vocals_mono(vocals_path, sr=sr)
    no_vocals, _ = load_vocals_mono(no_vocals_path, sr=sr)
    male, _ = load_vocals_mono(male_path, sr=sr)
    female, _ = load_vocals_mono(female_path, sr=sr)

    n = min(len(vocals), len(no_vocals), len(male), len(female))
    vocals = np.asarray(vocals[:n], dtype=np.float32)
    no_vocals = np.asarray(no_vocals[:n], dtype=np.float32)
    male = np.asarray(male[:n], dtype=np.float32)
    female = np.asarray(female[:n], dtype=np.float32)

    owner = np.full(n, -1, dtype=np.int32)
    owner_gender: list[str | None] = [None] * n

    segs = [
        s
        for s in assignments
        if not s.get("gap_fill") and s.get("index") is not None
    ]
    segs.sort(key=_segment_confidence, reverse=True)

    bleed_recovered_sec: dict[str, float] = {}
    total_transferred = 0

    for seg in segs:
        if not _segment_needs_bleed_recovery(
            seg, vocals, no_vocals, sr, pad_sec=window_pad_sec
        ):
            continue
        win_start, win_end = _iter_segment_windows(seg, pad_sec=window_pad_sec)
        gender = seg.get("gender", "female")
        islands = _leak_islands_for_segment(
            no_vocals,
            sr,
            seg,
            win_start,
            win_end,
            pad_sec=window_pad_sec,
            threshold_ratio=island_threshold_ratio,
        )
        if not islands:
            g0 = max(0, int(win_start * sr))
            g1 = min(n, int(win_end * sr))
            if g1 > g0:
                islands = [(win_start, win_end)]

        seg_transferred = 0
        for isl_start, isl_end in islands:
            g0 = max(0, int(isl_start * sr))
            g1 = min(n, int(isl_end * sr))
            for i in range(g0, g1):
                if owner[i] >= 0 and owner_gender[i] != gender:
                    continue

                nv = float(no_vocals[i])
                v = float(vocals[i])
                if abs(nv) < 1e-7:
                    continue

                excess = abs(nv) - abs(v) * leak_ratio
                if excess <= abs(nv) * 0.08:
                    continue

                transfer = (1.0 if nv >= 0 else -1.0) * excess * transfer_gain
                target = male if gender == "male" else female
                if abs(transfer) >= abs(float(target[i])):
                    target[i] = transfer
                else:
                    target[i] = float(target[i]) + transfer * 0.65

                vocals[i] = float(vocals[i]) + transfer * 0.55
                no_vocals[i] = float(no_vocals[i]) - transfer * bgm_attenuate
                owner[i] = int(seg.get("index") or 0)
                owner_gender[i] = gender
                seg_transferred += 1
                total_transferred += 1

        if seg_transferred:
            key = str(seg.get("index"))
            bleed_recovered_sec[key] = round(
                bleed_recovered_sec.get(key, 0.0) + seg_transferred / sr,
                3,
            )

    for track in (male, female, vocals, no_vocals):
        peak = float(np.max(np.abs(track)))
        if peak > 0.99:
            track /= peak

    sf.write(str(male_path), male, sr)
    sf.write(str(female_path), female, sr)
    sf.write(str(vocals_path), vocals, sr)
    sf.write(str(no_vocals_path), no_vocals, sr)

    return {
        "bleed_recovered_sec": round(total_transferred / sr, 2),
        "bleed_recovered_by_index": bleed_recovered_sec,
        "no_vocals_cleaned": True,
    }
