"""BS-RoFormer vocals 后 N 说话人轨切分 (日记化 + 可选重叠盲分离)。"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from video2text.diarization import (
    DEFAULT_DIARIZATION_MODEL,
    DiarizationResult,
    run_diarization,
    turns_to_assignments,
)
from video2text.f0_analysis import load_vocals_mono
from video2text.media import get_media_duration
from video2text.overlap_separation import apply_overlap_separation
from video2text.speaker_embedding import (
    DEFAULT_ANIME_EMBEDDING_MODEL,
    SpeakerEmbedder,
    build_speaker_profiles,
)


def _sanitize_speaker_id(speaker_id: str) -> str:
    """文件名安全: SPEAKER_00 -> speaker_00。"""
    s = speaker_id.strip().lower()
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def _segment_confidence(seg: dict) -> float:
    raw = seg.get("confidence")
    if raw is None:
        return 0.5
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.5


def _compute_stem_coverage_stats(
    audio: np.ndarray,
    sr: int,
    owner: np.ndarray,
    *,
    energy_ratio: float = 0.04,
) -> dict[str, float]:
    n = len(audio)
    global_peak = float(np.max(np.abs(audio))) if n else 0.0
    assigned = owner >= 0
    stem_assigned_sec = float(assigned.sum()) / sr

    if global_peak < 1e-6:
        return {
            "stem_assigned_sec": round(stem_assigned_sec, 2),
            "unassigned_vocal_sec": 0.0,
        }

    threshold = global_peak * energy_ratio
    vocal = np.abs(audio) >= threshold
    unassigned_vocal = (~assigned) & vocal
    return {
        "stem_assigned_sec": round(stem_assigned_sec, 2),
        "unassigned_vocal_sec": round(float(unassigned_vocal.sum()) / sr, 2),
    }


def build_speaker_stems(
    vocals_path: Path,
    assignments: list[dict],
    out_paths: dict[str, Path],
    *,
    sr: int = 44100,
    overlap_writes: dict[str, list[tuple[int, int, np.ndarray]]] | None = None,
    overlap_region_ranges: list[tuple[int, int]] | None = None,
) -> dict:
    """
    按 speaker_id 将 vocals 切分到 N 轨。
    overlap_writes: speaker_id -> [(g0, g1, samples)] 来自盲分离。
    overlap_region_ranges: 重叠区间样本索引, 单说话人切分跳过这些位置。
    """
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            "写入 wav 需要 soundfile: pip install soundfile"
        ) from exc

    audio, _ = load_vocals_mono(vocals_path, sr=sr)
    n = len(audio)
    speaker_ids = sorted(
        {str(a.get("speaker_id", "unknown")) for a in assignments} | set(out_paths.keys())
    )
    if not speaker_ids:
        raise RuntimeError("无有效 speaker_id, 无法输出多轨")

    owner = np.full(n, -1, dtype=np.int32)
    owner_conf = np.zeros(n, dtype=np.float32)

    overlap_mask = np.zeros(n, dtype=bool)
    if overlap_region_ranges:
        for g0, g1 in overlap_region_ranges:
            overlap_mask[g0:g1] = True

    indexed = list(enumerate(assignments))
    indexed.sort(key=lambda item: _segment_confidence(item[1]), reverse=True)

    for seg_idx, seg in indexed:
        speaker = str(seg.get("speaker_id", "unknown"))
        if speaker not in speaker_ids:
            continue
        start_i = max(0, int(float(seg["start"]) * sr))
        end_i = min(n, int(float(seg["end"]) * sr))
        if end_i <= start_i:
            continue
        conf = _segment_confidence(seg)
        center = (float(seg["start"]) + float(seg["end"])) / 2.0

        for i in range(start_i, end_i):
            if overlap_mask[i]:
                continue
            prev = owner[i]
            if prev < 0:
                owner[i] = seg_idx
                owner_conf[i] = conf
                continue

            prev_seg = assignments[prev]
            prev_speaker = str(prev_seg.get("speaker_id", "unknown"))
            if prev_speaker == speaker:
                if conf > owner_conf[i]:
                    owner[i] = seg_idx
                    owner_conf[i] = conf
                continue

            if conf > owner_conf[i]:
                owner[i] = seg_idx
                owner_conf[i] = conf
            elif conf == owner_conf[i]:
                prev_center = (
                    float(prev_seg["start"]) + float(prev_seg["end"])
                ) / 2.0
                t = i / sr
                if abs(t - center) < abs(t - prev_center):
                    owner[i] = seg_idx

    tracks: dict[str, np.ndarray] = {
        sp: np.zeros(n, dtype=np.float32) for sp in speaker_ids
    }
    for i in range(n):
        seg_idx = owner[i]
        if seg_idx < 0:
            continue
        sp = str(assignments[seg_idx].get("speaker_id", "unknown"))
        if sp in tracks:
            tracks[sp][i] = audio[i]

    if overlap_writes:
        for speaker, writes in overlap_writes.items():
            if speaker not in tracks:
                continue
            for g0, g1, samples in writes:
                g1 = min(g1, n)
                slen = min(len(samples), g1 - g0)
                if slen <= 0:
                    continue
                tracks[speaker][g0 : g0 + slen] = samples[:slen]

    for sp, track in tracks.items():
        peak = float(np.max(np.abs(track)))
        if peak > 0.99:
            track /= peak

    out_paths[speaker_ids[0]].parent.mkdir(parents=True, exist_ok=True)
    for sp in speaker_ids:
        out = out_paths.get(sp)
        if out is None:
            continue
        sf.write(str(out), tracks[sp], sr)

    coverage = _compute_stem_coverage_stats(audio, sr, owner)
    per_speaker_sec = {
        sp: round(float(np.count_nonzero(tracks[sp])) / sr, 2) for sp in speaker_ids
    }
    return {**coverage, "per_speaker_sec": per_speaker_sec}


def split_vocals_by_speaker(
    vocals_path: Path,
    out_dir: Path,
    stem: str,
    report_path: Path,
    *,
    split_mode: str = "diarize",
    min_speakers: int = 1,
    max_speakers: int = 4,
    hf_token: str | None = None,
    diarization_model: str = DEFAULT_DIARIZATION_MODEL,
    embedding_model: str = DEFAULT_ANIME_EMBEDDING_MODEL,
    use_embedding: bool = True,
    bss_backend: str = "auto",
    work_dir: Path | None = None,
    separator_meta: dict | None = None,
) -> tuple[list[dict], dict[str, Path]]:
    """
    日记化 vocals → N 条 speaker 轨 + JSON 报告。
    split_mode: diarize | diarize_bss
    """
    use_bss = split_mode == "diarize_bss"
    if split_mode not in ("diarize", "diarize_bss"):
        raise ValueError(f"speaker_stems 仅支持 diarize/diarize_bss, 收到: {split_mode}")

    diar: DiarizationResult = run_diarization(
        vocals_path,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        hf_token=hf_token,
        model_id=diarization_model,
    )
    assignments, report = turns_to_assignments(diar.turns)

    meta: dict = {
        "vocals": str(vocals_path),
        "split_mode": split_mode,
        "diarization_model": diarization_model,
        "speaker_ids": diar.speaker_ids,
        "turn_count": len(diar.turns),
        "overlap_region_count": len(diar.overlap_regions),
    }
    if separator_meta:
        meta["separator"] = separator_meta

    audio, sr = load_vocals_mono(vocals_path, sr=44100)
    overlap_stats: dict = {}
    overlap_writes: dict | None = None
    overlap_ranges: list[tuple[int, int]] | None = None

    if use_bss and diar.overlap_regions:
        embedder = SpeakerEmbedder(model_id=embedding_model) if use_embedding else None
        profiles = {}
        if embedder is not None:
            profiles = build_speaker_profiles(embedder, audio, sr, assignments)
            meta["embedding_model"] = embedder.model_id
        if profiles and work_dir is not None:
            overlap_stats = apply_overlap_separation(
                vocals_path,
                audio,
                sr,
                diar.overlap_regions,
                assignments,
                profiles,
                embedder,
                work_dir,
                bss_backend=bss_backend,
            )
            overlap_writes = overlap_stats.pop("overlap_writes", None)
            overlap_ranges = []
            for region in diar.overlap_regions:
                g0 = max(0, int(region.start * sr))
                g1 = min(len(audio), int(region.end * sr))
                overlap_ranges.append((g0, g1))
            for entry in report:
                entry["overlap_separated"] = any(
                    region.start <= float(entry["start"]) < region.end
                    or region.start < float(entry["end"]) <= region.end
                    for region in diar.overlap_regions
                )
            meta["overlap_bss"] = {
                k: v for k, v in overlap_stats.items() if k != "overlap_writes"
            }

    speaker_ids = diar.speaker_ids or sorted(
        {str(a["speaker_id"]) for a in assignments}
    )
    out_paths: dict[str, Path] = {}
    for sp in speaker_ids:
        safe = _sanitize_speaker_id(sp)
        out_paths[sp] = out_dir / f"{stem}_speaker_{safe}.wav"

    stem_stats = build_speaker_stems(
        vocals_path,
        assignments,
        out_paths,
        sr=sr,
        overlap_writes=overlap_writes,
        overlap_region_ranges=overlap_ranges,
    )

    vocals_duration = get_media_duration(vocals_path)
    coverage_sec = sum(max(0.0, float(r["end"]) - float(r["start"])) for r in report)
    meta["segments"] = report
    meta["stats"] = {
        "speaker_count": len(speaker_ids),
        "turn_count": len(report),
        "labeled_coverage_sec": round(coverage_sec, 2),
        "vocals_duration_sec": round(vocals_duration, 2),
        "coverage_ratio": round(coverage_sec / vocals_duration, 3)
        if vocals_duration > 0
        else 0.0,
        **{k: v for k, v in stem_stats.items() if k != "per_speaker_sec"},
        "per_speaker_sec": stem_stats.get("per_speaker_sec", {}),
    }

    report_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for sp, path in out_paths.items():
        print(f"已写入: {path}")
    print(f"已写入: {report_path}")
    stats = meta["stats"]
    print(
        f"  {stats['speaker_count']} 说话人, {stats['turn_count']} 段, "
        f"覆盖 {stats['labeled_coverage_sec']:.1f}s / {stats['vocals_duration_sec']:.1f}s "
        f"({stats['coverage_ratio']:.0%})"
    )
    if meta.get("overlap_bss"):
        ob = meta["overlap_bss"]
        print(
            f"  重叠盲分离: {ob.get('overlap_count', 0)} 段, "
            f"{ob.get('overlap_separated_sec', 0):.1f}s "
            f"(backend={ob.get('backend_used', 'n/a')})"
        )

    return report, out_paths
