"""SpeechBrain SepFormer 双说话人分离 (实验性, 重叠/串音场景)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _require_sepformer():
    try:
        from speechbrain.inference.separation import SepformerSeparation
    except ImportError as exc:
        raise RuntimeError(
            "SepFormer 需要 speechbrain, 请安装: pip install -r requirements-sepformer.txt"
        ) from exc
    return SepformerSeparation


def separate_vocals_sepformer(
    vocals_path: Path,
    work_dir: Path,
    *,
    chunk_sec: float = 15.0,
    overlap_sec: float = 1.0,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    将人声轨分离为两路 (8kHz mono)。
    返回 (source0, source1, sample_rate)。
    """
    import librosa
    import torch

    SepformerSeparation = _require_sepformer()
    work_dir.mkdir(parents=True, exist_ok=True)
    run_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"SepFormer 双路分离 (device={run_device}, chunk={chunk_sec}s)...")

    model = SepformerSeparation.from_hparams(
        source="speechbrain/sepformer-wsj02mix",
        savedir=str(work_dir / "sepformer_wsj02mix"),
        run_opts={"device": run_device},
    )

    target_sr = 8000
    audio, _ = librosa.load(str(vocals_path), sr=target_sr, mono=True)
    if audio.size == 0:
        raise RuntimeError("人声轨为空, 无法 SepFormer 分离")

    chunk_samples = int(chunk_sec * target_sr)
    overlap_samples = int(overlap_sec * target_sr)
    if chunk_samples <= overlap_samples * 2:
        chunk_samples = max(chunk_samples, overlap_samples * 2 + target_sr)

    out0 = np.zeros_like(audio)
    out1 = np.zeros_like(audio)
    weight = np.zeros_like(audio)

    step = chunk_samples - overlap_samples
    pos = 0
    while pos < len(audio):
        end = min(len(audio), pos + chunk_samples)
        chunk = audio[pos:end]
        pad = chunk_samples - len(chunk)
        if pad > 0:
            chunk = np.pad(chunk, (0, pad))

        tensor = torch.from_numpy(chunk).float().unsqueeze(0)
        with torch.no_grad():
            est = model.separate_batch(tensor)
        est = est.squeeze(0).cpu().numpy()
        s0 = est[0][: len(chunk) - (pad if pad > 0 else 0)]
        s1 = est[1][: len(chunk) - (pad if pad > 0 else 0)]

        sl = slice(pos, pos + len(s0))
        out0[sl] += s0
        out1[sl] += s1
        weight[sl] += 1.0

        if end >= len(audio):
            break
        pos += step

    weight = np.maximum(weight, 1.0)
    out0 /= weight
    out1 /= weight
    return out0, out1, target_sr


def _segment_rms(track: np.ndarray, sr: int, start: float, end: float) -> float:
    i0 = max(0, int(start * sr))
    i1 = min(len(track), int(end * sr))
    if i1 <= i0:
        return 0.0
    clip = track[i0:i1]
    return float(np.sqrt(np.mean(clip ** 2)))


def map_sepformer_tracks_to_gender(
    source0: np.ndarray,
    source1: np.ndarray,
    sr: int,
    assignments: list[dict],
    *,
    min_confidence: float = 0.55,
) -> dict[str, int]:
    """根据高置信 SRT 段, 判定哪一路 SepFormer 输出对应 male/female。"""
    male_score = {0: 0.0, 1: 0.0}
    female_score = {0: 0.0, 1: 0.0}

    for seg in assignments:
        conf = float(seg.get("confidence") or 0.0)
        if conf < min_confidence:
            continue
        start = float(seg["start"])
        end = float(seg["end"])
        gender = seg.get("gender")
        e0 = _segment_rms(source0, sr, start, end)
        e1 = _segment_rms(source1, sr, start, end)
        if gender == "male":
            male_score[0] += e0
            male_score[1] += e1
        elif gender == "female":
            female_score[0] += e0
            female_score[1] += e1

    track_for: dict[str, int] = {}
    if male_score[0] + male_score[1] > 0:
        track_for["male"] = 0 if male_score[0] >= male_score[1] else 1
    else:
        track_for["male"] = 0
    if female_score[0] + female_score[1] > 0:
        track_for["female"] = 0 if female_score[0] >= female_score[1] else 1
    else:
        track_for["female"] = 1 - track_for["male"]

    if track_for["male"] == track_for["female"]:
        track_for["female"] = 1 - track_for["male"]
    return track_for


def build_gender_stems_sepformer(
    source0: np.ndarray,
    source1: np.ndarray,
    sep_sr: int,
    track_for_gender: dict[str, int],
    assignments: list[dict],
    out_male: Path,
    out_female: Path,
    *,
    output_sr: int = 44100,
) -> None:
    """按 SRT 时间段从 SepFormer 两路输出组装男/女轨。"""
    import librosa
    import soundfile as sf

    sources = [source0, source1]
    male_track = sources[track_for_gender.get("male", 0)]
    female_track = sources[track_for_gender.get("female", 1)]

    duration = max(len(male_track), len(female_track)) / sep_sr
    male_out = np.zeros(int(duration * output_sr), dtype=np.float32)
    female_out = np.zeros(int(duration * output_sr), dtype=np.float32)

    for seg in assignments:
        start = float(seg["start"])
        end = float(seg["end"])
        i0 = max(0, int(start * sep_sr))
        i1 = min(len(male_track), int(end * sep_sr))
        if i1 <= i0:
            continue

        clip_m = male_track[i0:i1]
        clip_f = female_track[i0:i1]
        if clip_m.size == 0:
            continue

        clip_m = librosa.resample(clip_m, orig_sr=sep_sr, target_sr=output_sr)
        clip_f = librosa.resample(clip_f, orig_sr=sep_sr, target_sr=output_sr)

        o0 = int(start * output_sr)
        o1 = min(len(male_out), o0 + len(clip_m))
        if o1 <= o0:
            continue
        clip_len = o1 - o0
        if seg.get("gender") == "male":
            male_out[o0:o1] += clip_m[:clip_len]
        else:
            female_out[o0:o1] += clip_f[:clip_len]

    for track in (male_out, female_out):
        peak = float(np.max(np.abs(track)))
        if peak > 0.99:
            track /= peak

    out_male.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_male), male_out, output_sr)
    sf.write(str(out_female), female_out, output_sr)
