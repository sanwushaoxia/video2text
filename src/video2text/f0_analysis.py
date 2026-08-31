from __future__ import annotations


def analyze_segment_f0(
    audio,
    sr: int,
    start: float,
    end: float,
    *,
    fmin: float = 80.0,
    fmax: float = 400.0,
) -> tuple[float | None, float]:
    """返回 (median_f0_hz, voiced_ratio)。"""
    import numpy as np

    i0 = max(0, int(start * sr))
    i1 = min(len(audio), int(end * sr))
    clip = audio[i0:i1]
    if len(clip) < sr * 0.12:
        return None, 0.0

    import librosa

    f0, voiced_flag, _ = librosa.pyin(
        clip,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=2048,
        hop_length=256,
    )
    voiced = voiced_flag if voiced_flag is not None else np.zeros_like(f0, dtype=bool)
    valid = f0[~np.isnan(f0) & voiced]
    if valid.size == 0:
        return None, 0.0
    voiced_ratio = float(valid.size) / max(1, int(np.sum(~np.isnan(f0))))
    return float(np.median(valid)), voiced_ratio


def compute_adaptive_f0_threshold(
    f0_values: list[float],
    *,
    fallback: float = 200.0,
) -> float:
    """从有效 F0 样本估计男女分界 (动漫男声常高于 170Hz)。"""
    valid = sorted(f for f in f0_values if f is not None and f >= 80.0)
    if len(valid) < 6:
        return fallback

    best_threshold = fallback
    best_gap = 0.0
    for i in range(2, len(valid) - 2):
        left = valid[:i]
        right = valid[i:]
        gap = sum(right) / len(right) - sum(left) / len(left)
        if gap > best_gap:
            best_gap = gap
            best_threshold = (left[-1] + right[0]) / 2.0

    if best_gap < 25.0:
        return fallback
    return max(140.0, min(240.0, best_threshold))


def classify_segment_gender(
    f0: float | None,
    voiced_ratio: float,
    *,
    f0_threshold: float = 200.0,
    min_voiced_ratio: float = 0.25,
) -> tuple[str, str]:
    """纯 F0 判定性别, 返回 (male|female, reason)。"""
    if f0 is not None and voiced_ratio >= min_voiced_ratio:
        gender = "female" if f0 >= f0_threshold else "male"
        return gender, f"F0={f0:.0f}Hz"

    if f0 is not None:
        gender = "female" if f0 >= f0_threshold else "male"
        return gender, f"F0={f0:.0f}Hz (low_conf)"

    return "female", "default_female"


def load_vocals_mono(vocals_path, *, sr: int = 44100):
    import librosa

    return librosa.load(vocals_path, sr=sr, mono=True)
