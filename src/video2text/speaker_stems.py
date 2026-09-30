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
)
from video2text.diarization_refine import (
    RefineOptions,
    refine_diarization_turns,
    turns_to_assignments_with_confidence,
)
from video2text.f0_analysis import load_vocals_mono
from video2text.media import get_media_duration
from video2text.overlap_separation import apply_overlap_separation
from video2text.speaker_embedding import (
    DEFAULT_ANIME_EMBEDDING_MODEL,
    SpeakerEmbedder,
    assign_to_nearest_speaker,
    build_speaker_profiles,
    cosine_similarity,
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
    energy_ratio: float = 0.03,
) -> dict[str, float]:
    n = len(audio)
    global_peak = float(np.max(np.abs(audio))) if n else 0.0
    assigned = owner != -1
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


_FILL_OWNER = -2


def _resolve_stem_speaker(
    emb: np.ndarray,
    profiles: dict[str, np.ndarray],
    default_speaker: str,
    *,
    min_similarity: float,
    relabel_margin: float,
) -> str:
    if emb.size == 0 or float(np.linalg.norm(emb)) < 1e-6 or not profiles:
        return default_speaker
    scores = {
        sp: cosine_similarity(emb, profile) for sp, profile in profiles.items()
    }
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_sp, best_score = ranked[0]
    current_score = scores.get(default_speaker, 0.0)
    if (
        best_sp != default_speaker
        and best_score >= min_similarity
        and (best_score - current_score) >= relabel_margin
    ):
        return best_sp
    assigned = assign_to_nearest_speaker(
        emb, profiles, min_similarity=min_similarity
    )
    return assigned or default_speaker


def _fill_unassigned_vocal_runs(
    audio: np.ndarray,
    sr: int,
    owner: np.ndarray,
    tracks: dict[str, np.ndarray],
    speaker_ids: list[str],
    profiles: dict[str, np.ndarray],
    embedder: SpeakerEmbedder,
    *,
    energy_ratio: float = 0.03,
    min_run_sec: float = 0.2,
    min_similarity: float = 0.25,
) -> int:
    """对未分配但有人声能量的区间, 按 embedding 填充到最近 speaker。"""
    n = len(audio)
    global_peak = float(np.max(np.abs(audio))) if n else 0.0
    if global_peak < 1e-6 or not profiles:
        return 0

    threshold = global_peak * energy_ratio
    min_samples = max(1, int(min_run_sec * sr))
    filled_samples = 0
    i = 0
    while i < n:
        if owner[i] >= 0 or abs(audio[i]) < threshold:
            i += 1
            continue
        run_start = i
        while i < n and owner[i] < 0 and abs(audio[i]) >= threshold:
            i += 1
        run_end = i
        if run_end - run_start < min_samples:
            continue
        t0 = run_start / sr
        t1 = run_end / sr
        emb = embedder.embed_segment(audio, sr, t0, t1)
        speaker = assign_to_nearest_speaker(
            emb, profiles, min_similarity=min_similarity
        )
        if speaker is None or speaker not in tracks:
            continue
        tracks[speaker][run_start:run_end] = audio[run_start:run_end]
        owner[run_start:run_end] = _FILL_OWNER
        filled_samples += run_end - run_start
    return filled_samples


def build_speaker_stems(
    vocals_path: Path,
    assignments: list[dict],
    out_paths: dict[str, Path],
    *,
    sr: int = 44100,
    overlap_writes: dict[str, list[tuple[int, int, np.ndarray]]] | None = None,
    overlap_region_ranges: list[tuple[int, int]] | None = None,
    profiles: dict[str, np.ndarray] | None = None,
    embedder: SpeakerEmbedder | None = None,
    fill_unassigned: bool = True,
    stem_min_similarity: float = 0.30,
    stem_relabel_margin: float = 0.08,
    fill_min_similarity: float = 0.25,
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
    effective_speakers: dict[int, str] = {}

    for seg_idx, seg in indexed:
        speaker = str(seg.get("speaker_id", "unknown"))
        if speaker not in speaker_ids:
            continue
        start_i = max(0, int(float(seg["start"]) * sr))
        end_i = min(n, int(float(seg["end"]) * sr))
        if end_i <= start_i:
            continue
        conf = _segment_confidence(seg)
        if profiles and embedder is not None:
            emb = embedder.embed_segment(
                audio, sr, float(seg["start"]), float(seg["end"])
            )
            if emb.size > 0 and float(np.linalg.norm(emb)) > 1e-6:
                if seg.get("low_confidence"):
                    assigned = assign_to_nearest_speaker(
                        emb, profiles, min_similarity=fill_min_similarity
                    )
                    if assigned:
                        speaker = assigned
                else:
                    speaker = _resolve_stem_speaker(
                        emb,
                        profiles,
                        speaker,
                        min_similarity=stem_min_similarity,
                        relabel_margin=stem_relabel_margin,
                    )
                if speaker in profiles:
                    conf = max(conf, cosine_similarity(emb, profiles[speaker]))
        effective_speakers[seg_idx] = speaker
        center = (float(seg["start"]) + float(seg["end"])) / 2.0

        for i in range(start_i, end_i):
            if overlap_mask[i]:
                continue
            prev = owner[i]
            if prev < 0:
                owner[i] = seg_idx
                owner_conf[i] = conf
                continue

            prev_seg_idx = prev
            prev_speaker = effective_speakers.get(
                prev_seg_idx,
                str(assignments[prev_seg_idx].get("speaker_id", "unknown")),
            )
            if prev_speaker == speaker:
                if conf > owner_conf[i]:
                    owner[i] = seg_idx
                    owner_conf[i] = conf
                continue

            if conf > owner_conf[i]:
                owner[i] = seg_idx
                owner_conf[i] = conf
            elif conf == owner_conf[i]:
                prev_seg = assignments[prev_seg_idx]
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
        if seg_idx == _FILL_OWNER:
            continue
        sp = effective_speakers.get(
            seg_idx,
            str(assignments[seg_idx].get("speaker_id", "unknown")),
        )
        if sp in tracks:
            tracks[sp][i] = audio[i]

    filled_samples = 0
    if fill_unassigned and profiles and embedder is not None:
        filled_samples = _fill_unassigned_vocal_runs(
            audio,
            sr,
            owner,
            tracks,
            speaker_ids,
            profiles,
            embedder,
            min_similarity=fill_min_similarity,
        )

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
    return {
        **coverage,
        "per_speaker_sec": per_speaker_sec,
        "fill_unassigned_samples": filled_samples,
    }


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
    refine_embedding: bool = True,
    merge_turn_gap: float = 0.4,
    min_turn_sec: float = 0.5,
    min_similarity: float = 0.35,
    fill_unassigned: bool = True,
    diarize_preprocess: bool = True,
    overlap_min_sec: float = 0.15,
    max_overlap_sec: float = 8.0,
    drop_low_confidence: bool = False,
    stem_min_similarity: float = 0.30,
    stem_relabel_margin: float = 0.08,
    fill_min_similarity: float = 0.25,
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
        diarize_preprocess=diarize_preprocess,
        work_dir=work_dir,
        overlap_min_sec=overlap_min_sec,
    )

    audio, sr = load_vocals_mono(vocals_path, sr=44100)
    embedder: SpeakerEmbedder | None = None
    profiles: dict[str, np.ndarray] = {}
    refine_stats: dict = {}

    if use_embedding:
        embedder = SpeakerEmbedder(model_id=embedding_model)

    refined_turns = diar.turns
    confidences: list[float] | None = None
    if refine_embedding and embedder is not None:
        refined_turns, confidences, stats = refine_diarization_turns(
            diar.turns,
            audio,
            sr,
            embedder,
            options=RefineOptions(
                merge_turn_gap=merge_turn_gap,
                min_turn_sec=min_turn_sec,
                min_similarity=min_similarity,
                use_embedding=True,
                drop_low_confidence=drop_low_confidence,
            ),
        )
        refine_stats = {
            "merged_turn_count": stats.merged_turn_count,
            "relabeled_count": stats.relabeled_count,
            "filtered_count": stats.filtered_count,
            "avg_confidence": round(stats.avg_confidence, 3),
            "embedding_similarities": stats.embedding_similarities,
        }
        if embedder is not None:
            profiles = build_speaker_profiles(
                embedder, audio, sr, [
                    {"speaker_id": t.speaker_id, "start": t.start, "end": t.end}
                    for t in refined_turns
                ]
            )
    else:
        confidences = [float(t.confidence) for t in refined_turns]

    assignments, report = turns_to_assignments_with_confidence(
        refined_turns, confidences, min_similarity=min_similarity
    )

    if embedder is not None and not profiles:
        profiles = build_speaker_profiles(embedder, audio, sr, assignments)

    speaker_ids = sorted({t.speaker_id for t in refined_turns}) or diar.speaker_ids

    meta: dict = {
        "vocals": str(vocals_path),
        "split_mode": split_mode,
        "diarization_model": diarization_model,
        "speaker_ids": speaker_ids,
        "turn_count": len(refined_turns),
        "overlap_region_count": len(diar.overlap_regions),
        "overlap_detection_source": diar.overlap_detection_source,
        "refine_stats": refine_stats,
    }
    if separator_meta:
        meta["separator"] = separator_meta
    if embedder is not None:
        meta["embedding_model"] = embedder.model_id

    overlap_stats: dict = {}
    overlap_writes: dict | None = None
    overlap_ranges: list[tuple[int, int]] | None = None

    if use_bss and diar.overlap_regions:
        if not profiles and embedder is not None:
            profiles = build_speaker_profiles(embedder, audio, sr, assignments)
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
                min_overlap_sec=overlap_min_sec,
                max_overlap_sec=max_overlap_sec,
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

    if not speaker_ids:
        speaker_ids = sorted({str(a["speaker_id"]) for a in assignments})
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
        profiles=profiles or None,
        embedder=embedder,
        fill_unassigned=fill_unassigned,
        stem_min_similarity=stem_min_similarity,
        stem_relabel_margin=stem_relabel_margin,
        fill_min_similarity=fill_min_similarity,
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
