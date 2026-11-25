"""BS-RoFormer vocals 后 N 说话人轨切分 (日记化 + 可选重叠盲分离)。"""
from __future__ import annotations

from contextlib import ExitStack
import json
import re
from pathlib import Path

import numpy as np

from video2text.performance import stage, timed_stage
from video2text.diarization import (
    DEFAULT_DIARIZATION_MODEL,
    DiarizationResult,
    OverlapRegion,
    fuse_diarization,
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
from video2text.overlap_tracking import build_anchors
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


_FILL_OWNER = -2
_OVERLAP_OWNER = -3
_REVIEW_OWNER = -4


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


_BLOCK_SAMPLES = 1 << 18
_SEEKABLE_AUDIO_SECONDS = 600
# NumPy 1.x widens scalar float32 comparisons with a Python float, unlike 2.x.
_WEAK_SCALAR_PROMOTION = bool(np.float32(0.1) == 0.1)


class _SeekableMono:
    """Read exact sample ranges from an immutable, native-rate speech stem."""

    def __init__(self, path: Path, frames: int):
        self.path = path
        self.frames = frames

    def __len__(self):
        return self.frames

    def __getitem__(self, samples: slice) -> np.ndarray:
        import soundfile as sf

        start, stop, step = samples.indices(self.frames)
        if step != 1:
            raise ValueError("音频区间读取需要连续样本")
        audio, _ = sf.read(
            self.path, start=start, frames=max(0, stop - start), dtype="float32", always_2d=True,
        )
        return audio[:, 0] if audio.shape[1] == 1 else audio.T.mean(axis=0)


def _open_speaker_audio(path: Path, sr: int) -> np.ndarray | _SeekableMono:
    import soundfile as sf

    if path.suffix.lower() == ".wav":
        try:
            info = sf.info(path)
        except RuntimeError:
            pass
        else:
            if info.samplerate == sr and info.frames >= sr * _SEEKABLE_AUDIO_SECONDS:
                return _SeekableMono(path, info.frames)
    return load_vocals_mono(path, sr=sr)[0]


def _index_spans(spans, n):
    """Index spans per block while preserving the caller's precedence order."""
    blocks = [[] for _ in range((n + _BLOCK_SAMPLES - 1) // _BLOCK_SAMPLES)]
    for start, end, *values in spans:
        start, end, _ = slice(start, end).indices(n)
        if end <= start:
            continue
        for i in range(start // _BLOCK_SAMPLES, (end - 1) // _BLOCK_SAMPLES + 1):
            offset = i * _BLOCK_SAMPLES
            left, right = max(start, offset), min(end, offset + _BLOCK_SAMPLES)
            blocks[i].append((left - offset, right - offset, left - start, values))
    return blocks


def _block_owners(size, offset, sr, segments, overlap, review, centers, speaker_index):
    owner = np.full(size, -1, dtype=np.int32)
    confidence = np.zeros(size, dtype=np.float32)
    for start, end, _, _ in review:
        owner[start:end] = _REVIEW_OWNER
    for start, end, _, _ in overlap:
        owner[start:end] = _OVERLAP_OWNER
    for start, end, _, (index, score) in segments:
        current = owner[start:end]
        scores = confidence[start:end]
        comparison = scores if _WEAK_SCALAR_PROMOTION else scores.astype(np.float64)
        replace = (current == -1) | ((current >= 0) & (score > comparison))
        tied = np.flatnonzero((current >= 0) & (score == comparison) & ~replace)
        if tied.size:
            previous = current[tied]
            times = (tied + start + offset) / sr
            closer = np.abs(times - centers[index]) < np.abs(times - centers[previous])
            tied_changes = tied[closer & (speaker_index[previous] != speaker_index[index])]
        else:
            tied_changes = tied
        current[replace] = index
        scores[replace] = score
        current[tied_changes] = index
    return owner


def _unassigned_runs(blocks, global_peak, threshold_ratio=0.03):
    previous = False
    start = 0
    end = 0
    for _, offset, audio, owner in blocks():
        absolute = np.abs(audio)
        if not _WEAK_SCALAR_PROMOTION:
            absolute = absolute.astype(np.float64)
        mask = (owner == -1) & (absolute >= global_peak * threshold_ratio)
        for index in np.flatnonzero(np.diff(mask, prepend=previous)):
            if mask[index]:
                start = offset + int(index)
            else:
                yield start, offset + int(index)
        previous = bool(mask[-1])
        end = offset + len(mask)
    if previous:
        yield start, end


@timed_stage("write_speaker_tracks")
def build_speaker_stems(
    vocals_path: Path,
    assignments: list[dict],
    out_paths: dict[str, Path],
    *,
    sr: int = 44100,
    overlap_writes: dict[str, list[tuple[int, int, np.ndarray | Path]]] | None = None,
    overlap_region_ranges: list[tuple[int, int]] | None = None,
    review_ranges: list[tuple[int, int]] | None = None,
    review_path: Path | None = None,
    profiles: dict[str, np.ndarray] | None = None,
    embedder: SpeakerEmbedder | None = None,
    fill_unassigned: bool = True,
    stem_min_similarity: float = 0.30,
    stem_relabel_margin: float = 0.08,
    fill_min_similarity: float = 0.25,
    audio: np.ndarray | _SeekableMono | None = None,
) -> dict:
    """Render sample-aligned stems with bounded working buffers."""
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("写入 wav 需要 soundfile: pip install soundfile") from exc

    if audio is None:
        audio = _open_speaker_audio(vocals_path, sr)
    n = len(audio)
    speakers = sorted({str(a.get("speaker_id", "unknown")) for a in assignments} | set(out_paths))
    if not speakers and review_path is None:
        raise RuntimeError("无有效 speaker_id, 无法输出多轨")
    speaker_numbers = {speaker: i for i, speaker in enumerate(speakers)}
    centers = np.zeros(len(assignments))
    effective = np.zeros(len(assignments), dtype=np.int32)
    segments = []
    indexed = sorted(enumerate(assignments), key=lambda item: _segment_confidence(item[1]), reverse=True)
    for index, seg in indexed:
        speaker = str(seg.get("speaker_id", "unknown"))
        start, end = float(seg["start"]), float(seg["end"])
        left, right = max(0, int(start * sr)), min(n, int(end * sr))
        if right <= left:
            continue
        confidence = _segment_confidence(seg)
        if profiles and embedder is not None:
            emb = embedder.embed_segment(audio, sr, start, end)
            if emb.size and float(np.linalg.norm(emb)) > 1e-6:
                if seg.get("low_confidence"):
                    speaker = assign_to_nearest_speaker(
                        emb, profiles, min_similarity=fill_min_similarity,
                    ) or speaker
                else:
                    speaker = _resolve_stem_speaker(
                        emb, profiles, speaker, min_similarity=stem_min_similarity,
                        relabel_margin=stem_relabel_margin,
                    )
                if speaker in profiles:
                    confidence = max(confidence, cosine_similarity(emb, profiles[speaker]))
        # -1 represents previously relabeled speakers with no output track.
        effective[index] = speaker_numbers.get(speaker, -1)
        centers[index] = (start + end) / 2.0
        segments.append((left, right, index, confidence))

    by_block = _index_spans(segments, n)
    overlap = _index_spans(overlap_region_ranges or [], n)
    review = _index_spans(review_ranges if review_path and review_ranges else [], n)
    patches = []
    for speaker, writes in (overlap_writes or {}).items():
        if speaker_numbers.get(speaker) is None:
            continue
        for start, end, samples in writes:
            available = end - start if isinstance(samples, Path) else len(samples)
            length = min(available, end - start, n - start)
            if 0 <= start and length > 0:
                patches.append((start, start + length, speaker_numbers[speaker], samples))
    patches = _index_spans(patches, n)

    def blocks():
        for i, offset in enumerate(range(0, n, _BLOCK_SAMPLES)):
            clip = audio[offset:offset + _BLOCK_SAMPLES]
            owner = _block_owners(len(clip), offset, sr, by_block[i], overlap[i],
                                  review[i], centers, effective)
            yield i, offset, clip, owner

    peak = max((float(np.max(np.abs(audio[i:i + _BLOCK_SAMPLES])))
                for i in range(0, n, _BLOCK_SAMPLES)), default=0.0)
    filled = []
    if peak >= 1e-6 and fill_unassigned and profiles and embedder is not None:
        min_samples = max(1, int(0.2 * sr))
        for left, right in _unassigned_runs(blocks, peak):
            if right - left >= min_samples:
                emb = embedder.embed_segment(audio, sr, left / sr, right / sr)
                chosen = assign_to_nearest_speaker(emb, profiles, min_similarity=fill_min_similarity)
                if chosen in speaker_numbers:
                    filled.append((left, right, speaker_numbers[chosen]))
    fills = _index_spans(filled, n)

    def render():
        for i, _, clip, owner in blocks():
            track = np.zeros((len(speakers), len(clip)), dtype=np.float32)
            positions = np.flatnonzero(owner >= 0)
            slots = effective[owner[positions]]
            valid = slots >= 0
            track[slots[valid], positions[valid]] = clip[positions[valid]]
            for start, end, _, (speaker,) in fills[i]:
                track[speaker, start:end] = clip[start:end]
                owner[start:end] = _FILL_OWNER
            for start, end, offset, (speaker, samples) in patches[i]:
                if isinstance(samples, Path):
                    samples = np.load(samples, mmap_mode="r", allow_pickle=False)
                track[speaker, start:end] = samples[offset:offset + end - start]
            yield clip, owner, track

    scales = np.ones(len(speakers), dtype=np.float32)
    if review_path is None:
        # The legacy peak rule also increases peaks in (0.99, 1.0).
        near_peaks = np.zeros(len(speakers), dtype=np.float32)
        for _, _, track in render():
            near_peaks = np.maximum(near_peaks, np.max(np.abs(track), axis=1))
        scales = np.where(near_peaks.astype(np.float64) > 0.99, near_peaks, scales)

    assigned = review_count = unassigned_vocal = 0
    active = np.zeros(len(speakers), dtype=np.int64)
    with ExitStack() as stack:
        streams = {}
        for speaker, path in out_paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            streams[speaker_numbers[speaker]] = stack.enter_context(
                sf.SoundFile(path, "w", sr, 1, subtype="FLOAT" if review_path else "PCM_16")
            )
        review_stream = None
        if review_path is not None:
            review_path.parent.mkdir(parents=True, exist_ok=True)
            review_stream = stack.enter_context(sf.SoundFile(review_path, "w", sr, 1, subtype="FLOAT"))
        for clip, owner, track in render():
            track /= scales[:, None]
            active += np.count_nonzero(track, axis=1)
            for i, stream in streams.items():
                stream.write(track[i])
            needs_review = owner == _REVIEW_OWNER
            unassigned = (owner == -1) | needs_review
            if review_stream is not None:
                review_stream.write(np.where(unassigned, clip, 0))
            assigned += len(owner) - np.count_nonzero(unassigned)
            review_count += np.count_nonzero(needs_review)
            if peak >= 1e-6:
                unassigned_vocal += np.count_nonzero(unassigned & (np.abs(clip) >= peak * 0.03))

    stats = {
        "stem_assigned_sec": round(int(assigned) / sr, 2),
        "unassigned_vocal_sec": round(int(unassigned_vocal) / sr, 2),
        "per_speaker_sec": {speaker: round(int(count) / sr, 2) for speaker, count in zip(speakers, active)},
        "fill_unassigned_samples": sum(right - left for left, right, _ in filled),
    }
    if review_path is not None:
        stats["review_sec"] = round(int(review_count) / sr, 3)
    return stats


def _profile_segments(
    assignments: list[dict], overlaps: list[OverlapRegion], min_turn_sec: float,
) -> list[dict]:
    """为叠话归属提取足够长的单人片段。"""
    clean = []
    for segment in assignments:
        start, end = segment["start"], segment["end"]
        for region in overlaps:
            if region.end <= start:
                continue
            if region.start >= end:
                break
            if region.start - start >= min_turn_sec:
                clean.append(dict(segment, start=start, end=region.start))
            start = max(start, region.end)
        if end - start >= min_turn_sec:
            clean.append(dict(segment, start=start))
    return clean


def _labeled_coverage_sec(segments, duration: float) -> float:
    coverage = 0.0
    previous_end = 0.0
    for entry in sorted(segments, key=lambda segment: float(segment["start"])):
        start = max(previous_end, float(entry["start"]))
        end = min(duration, float(entry["end"]))
        if end > start:
            coverage += end - start
            previous_end = end
    return coverage


def split_vocals_by_speaker(
    vocals_path: Path,
    out_dir: Path,
    stem: str,
    report_path: Path,
    *,
    split_mode: str = "diarize",
    original_audio_path: Path | None = None,
    diarization_source: str = "vocals",
    min_speakers: int = 1,
    max_speakers: int = 4,
    hf_token: str | None = None,
    diarization_model: str = DEFAULT_DIARIZATION_MODEL,
    embedding_model: str = DEFAULT_ANIME_EMBEDDING_MODEL,
    use_embedding: bool = True,
    bss_backend: str = "auto",
    overlap_tracking: bool = False,
    save_overlap_review: bool = False,
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
    overlap_review_threshold: float | None = None,
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
    if (overlap_tracking or save_overlap_review) and not use_bss:
        raise ValueError("叠话联合归属和复核音频需要 diarize_bss")
    if diarization_source not in ("vocals", "mix", "joint"):
        raise ValueError(f"未知的日记化输入: {diarization_source}")
    if diarization_source == "joint" and not use_bss:
        raise ValueError("双路叠话检测需要 diarize_bss")
    if diarization_source in ("mix", "joint") and original_audio_path is None:
        raise ValueError("原混音日记化需要原始音频")
    if overlap_review_threshold is not None and (
        not use_bss or not 0 < overlap_review_threshold < 1
    ):
        raise ValueError("叠话复核阈值需要 diarize_bss，且必须介于 0 和 1 之间")

    diarization_args = dict(
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        hf_token=hf_token,
        model_id=diarization_model,
        diarize_preprocess=diarize_preprocess,
        work_dir=work_dir,
        overlap_min_sec=overlap_min_sec,
        overlap_review_threshold=overlap_review_threshold,
    )
    diarization_input = original_audio_path if diarization_source in ("mix", "joint") else vocals_path
    diar: DiarizationResult = run_diarization(diarization_input, **diarization_args)
    if diarization_source == "joint":
        diar = fuse_diarization(diar, run_diarization(vocals_path, **diarization_args))
    if (use_bss and not diar.turns and original_audio_path is not None
            and diarization_source == "vocals"):
        diarization_input = original_audio_path
        diar = run_diarization(original_audio_path, **diarization_args)

    with stage("decode_vocals"):
        sr = 44100
        audio = _open_speaker_audio(vocals_path, sr)
    embedder: SpeakerEmbedder | None = None
    profiles: dict[str, np.ndarray] = {}
    refine_stats: dict = {}

    if use_embedding and diar.turns:
        embedder = SpeakerEmbedder(model_id=embedding_model, hf_token=hf_token)
        embedder.cache_segments(audio)

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
    else:
        confidences = [float(t.confidence) for t in refined_turns]

    assignments, report = turns_to_assignments_with_confidence(
        refined_turns, confidences, min_similarity=min_similarity
    )

    confirmed_overlap = diar.overlap_detection_source != "proximity"
    tracking_anchors = {} if overlap_tracking else None
    if embedder is not None:
        profile_segments = assignments
        if use_bss and confirmed_overlap and diar.overlap_regions:
            exclude = sorted(diar.overlap_regions + diar.overlap_review_regions, key=lambda r: r.start)
            profile_segments = _profile_segments(assignments, exclude, min_turn_sec)
        profiles = build_speaker_profiles(embedder, audio, sr, profile_segments)
        if overlap_tracking and confirmed_overlap and diar.overlap_regions:
            tracking_anchors = build_anchors(embedder, audio, sr, profile_segments)

    speaker_ids = sorted({t.speaker_id for t in refined_turns}) or diar.speaker_ids

    meta: dict = {
        "vocals": str(vocals_path),
        "split_mode": split_mode,
        "diarization_model": diarization_model,
        "diarization_source": (
            "joint" if diarization_source == "joint"
            else "vocals" if diarization_input == vocals_path else "mix"
        ),
        "speaker_ids": speaker_ids,
        "turn_count": len(refined_turns),
        "overlap_region_count": len(diar.overlap_regions),
        "overlap_detection_source": diar.overlap_detection_source,
        "refine_stats": refine_stats,
    }
    if diarization_input != vocals_path and (diarization_source != "joint" or diar.overlap_detection_source == "joint"):
        meta["diarization_used_original_mix"] = True
    if separator_meta:
        meta["separator"] = separator_meta
    if embedder is not None:
        meta["embedding_model"] = embedder.model_id

    overlap_writes: dict | None = None
    overlap_ranges: list[tuple[int, int]] = []
    review_ranges: list[tuple[int, int]] = []
    review_path = out_dir / f"{stem}_dialogue_review.wav" if use_bss else None

    if use_bss:
        overlap_stats = apply_overlap_separation(
            audio, sr, diar.overlap_regions, assignments, profiles, embedder, work_dir,
            bss_backend=bss_backend, min_overlap_sec=overlap_min_sec,
            max_overlap_sec=max_overlap_sec, min_similarity=min_similarity,
            confirmed=confirmed_overlap,
            tracking_anchors=tracking_anchors,
            preview_dir=out_dir / f"{stem}_overlap_sources" if save_overlap_review else None,
        )
        overlap_writes = overlap_stats.pop("overlap_writes")
        for region in overlap_stats["regions"]:
            bounds = (region["start_sample"], region["end_sample"])
            if region["status"] == "separated":
                overlap_ranges.append(bounds)
            elif region["status"] == "review":
                review_ranges.append(bounds)
        for region in diar.overlap_review_regions:
            start = max(0, round(region.start * sr))
            end = min(len(audio), round(region.end * sr))
            if end <= start:
                continue
            review_ranges.append((start, end))
            overlap_stats["regions"].append({
                "start_sample": start, "end_sample": end, "speaker_ids": [],
                "status": "review", "reason": region.reason or "overlap_posterior",
            })
        for entry in report:
            entry["overlap_separated"] = any(
                start < float(entry["end"]) * sr and end > float(entry["start"]) * sr
                for start, end in overlap_ranges
            )
        meta["overlap_bss"] = overlap_stats
        if overlap_tracking:
            meta["overlap_assignment"] = "tracking"
        if overlap_review_threshold is not None:
            meta["overlap_review_threshold"] = overlap_review_threshold
        meta["dialogue_review"] = str(review_path.resolve())
        if not diar.turns:
            review_ranges.append((0, len(audio)))
            meta["dialogue_review_reason"] = "no_diarization_turns"

    if not speaker_ids:
        speaker_ids = sorted({str(a["speaker_id"]) for a in assignments})
    out_paths: dict[str, Path] = {}
    for sp in speaker_ids:
        safe = _sanitize_speaker_id(sp)
        out_paths[sp] = out_dir / f"{stem}_speaker_{safe}.wav"
    if save_overlap_review:
        meta["speaker_tracks"] = {speaker: str(path.resolve()) for speaker, path in out_paths.items()}

    stem_stats = build_speaker_stems(
        vocals_path,
        assignments,
        out_paths,
        sr=sr,
        overlap_writes=overlap_writes,
        overlap_region_ranges=overlap_ranges,
        review_ranges=review_ranges,
        review_path=review_path,
        profiles=profiles or None,
        embedder=embedder,
        fill_unassigned=fill_unassigned,
        stem_min_similarity=stem_min_similarity,
        stem_relabel_margin=stem_relabel_margin,
        fill_min_similarity=fill_min_similarity,
        audio=audio,
    )

    vocals_duration = get_media_duration(vocals_path)
    coverage_sec = _labeled_coverage_sec(report, vocals_duration)
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
    if review_path is not None:
        print(f"已写入: {review_path}")
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
