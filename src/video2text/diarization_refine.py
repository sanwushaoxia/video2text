"""日记化后处理: 合并碎段、embedding 改标、低置信过滤。"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from video2text.diarization import DiarizationTurn
from video2text.speaker_embedding import (
    SpeakerEmbedder,
    assign_to_nearest_speaker,
    cosine_similarity,
)


@dataclass
class RefineOptions:
    merge_turn_gap: float = 0.4
    min_turn_sec: float = 0.5
    min_similarity: float = 0.35
    relabel_margin: float = 0.08
    use_embedding: bool = True
    drop_low_confidence: bool = False


@dataclass
class RefineStats:
    merged_turn_count: int = 0
    relabeled_count: int = 0
    filtered_count: int = 0
    avg_confidence: float = 0.0
    embedding_similarities: list[dict] = field(default_factory=list)


def merge_adjacent_turns(
    turns: list[DiarizationTurn],
    *,
    max_gap: float = 0.4,
) -> list[DiarizationTurn]:
    """合并同 speaker 相邻碎段。"""
    if not turns:
        return []
    ordered = sorted(turns, key=lambda t: (t.start, t.speaker_id))
    merged: list[DiarizationTurn] = [ordered[0]]
    for turn in ordered[1:]:
        prev = merged[-1]
        gap = turn.start - prev.end
        if turn.speaker_id == prev.speaker_id and gap <= max_gap:
            merged[-1] = DiarizationTurn(
                start=prev.start,
                end=max(prev.end, turn.end),
                speaker_id=prev.speaker_id,
            )
        else:
            merged.append(turn)
    return merged


def _turn_duration(turn: DiarizationTurn) -> float:
    return max(0.0, float(turn.end) - float(turn.start))


def build_profiles_for_refine(
    embedder: SpeakerEmbedder,
    audio: np.ndarray,
    sr: int,
    turns: list[DiarizationTurn],
    *,
    min_turn_sec: float = 0.5,
) -> dict[str, np.ndarray]:
    """仅用足够长的 turn 构建 speaker profile。"""
    buckets: dict[str, list[np.ndarray]] = {}
    for turn in turns:
        if _turn_duration(turn) < min_turn_sec:
            continue
        emb = embedder.embed_segment(audio, sr, turn.start, turn.end)
        if emb.size == 0 or float(np.linalg.norm(emb)) < 1e-6:
            continue
        buckets.setdefault(turn.speaker_id, []).append(emb)

    profiles: dict[str, np.ndarray] = {}
    for speaker, embs in buckets.items():
        mean = np.mean(np.stack(embs, axis=0), axis=0).astype(np.float32)
        norm = float(np.linalg.norm(mean))
        if norm > 1e-8:
            mean /= norm
            profiles[speaker] = mean
    return profiles


def _score_turn_embedding(
    embedder: SpeakerEmbedder,
    audio: np.ndarray,
    sr: int,
    turn: DiarizationTurn,
    profiles: dict[str, np.ndarray],
) -> tuple[str | None, float, dict[str, float]]:
    emb = embedder.embed_segment(audio, sr, turn.start, turn.end)
    if emb.size == 0 or float(np.linalg.norm(emb)) < 1e-6:
        return None, 0.0, {}

    scores = {
        sp: cosine_similarity(emb, profile) for sp, profile in profiles.items()
    }
    if not scores:
        return None, 0.0, {}

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_sp, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = best_score - second_score
    assigned = assign_to_nearest_speaker(emb, profiles, min_similarity=0.0)
    return assigned or best_sp, margin, scores


def refine_turn_labels(
    turns: list[DiarizationTurn],
    profiles: dict[str, np.ndarray],
    embedder: SpeakerEmbedder,
    audio: np.ndarray,
    sr: int,
    *,
    relabel_margin: float = 0.08,
    min_similarity: float = 0.35,
) -> tuple[list[DiarizationTurn], RefineStats]:
    """embedding 二次校验: 改标、更新置信度。"""
    stats = RefineStats()
    refined: list[DiarizationTurn] = []
    confidences: list[float] = []

    for turn in turns:
        best_sp, margin, scores = _score_turn_embedding(
            embedder, audio, sr, turn, profiles
        )
        speaker = turn.speaker_id
        confidence = 0.75

        if best_sp and scores:
            current_score = scores.get(turn.speaker_id, 0.0)
            best_score = scores.get(best_sp, 0.0)
            confidence = max(current_score, best_score)
            stats.embedding_similarities.append(
                {
                    "start": round(turn.start, 3),
                    "end": round(turn.end, 3),
                    "speaker_id": turn.speaker_id,
                    "scores": {k: round(v, 3) for k, v in scores.items()},
                    "margin": round(margin, 3),
                }
            )
            if (
                best_sp != turn.speaker_id
                and best_score >= min_similarity
                and (best_score - current_score) >= relabel_margin
            ):
                speaker = best_sp
                stats.relabeled_count += 1
                confidence = best_score
            else:
                confidence = current_score if current_score >= min_similarity else best_score

        refined.append(
            DiarizationTurn(
                start=turn.start,
                end=turn.end,
                speaker_id=speaker,
                confidence=confidence,
            )
        )
        confidences.append(confidence)

    if confidences:
        stats.avg_confidence = float(np.mean(confidences))
    return refined, stats


def filter_low_confidence_turns(
    turns: list[DiarizationTurn],
    confidences: list[float],
    *,
    min_similarity: float = 0.35,
) -> tuple[list[DiarizationTurn], int]:
    """丢弃 embedding 置信度过低的 turn。"""
    kept: list[DiarizationTurn] = []
    dropped = 0
    for turn, conf in zip(turns, confidences):
        if conf < min_similarity:
            dropped += 1
            continue
        kept.append(turn)
    return kept, dropped


def turns_to_assignments_with_confidence(
    turns: list[DiarizationTurn],
    confidences: list[float] | None = None,
    *,
    min_similarity: float | None = None,
) -> tuple[list[dict], list[dict]]:
    assignments: list[dict] = []
    report: list[dict] = []
    for idx, turn in enumerate(turns, start=1):
        conf = (
            confidences[idx - 1]
            if confidences and idx - 1 < len(confidences)
            else getattr(turn, "confidence", 0.75)
        )
        entry = {
            "start": turn.start,
            "end": turn.end,
            "speaker_id": turn.speaker_id,
            "index": idx,
            "text": "",
            "confidence": round(float(conf), 3),
            "reason": f"diarization+refine:{turn.speaker_id}",
        }
        if min_similarity is not None and float(conf) < min_similarity:
            entry["low_confidence"] = True
        assignments.append(dict(entry))
        report.append(dict(entry))
    return assignments, report


def refine_diarization_turns(
    turns: list[DiarizationTurn],
    audio: np.ndarray,
    sr: int,
    embedder: SpeakerEmbedder | None,
    *,
    options: RefineOptions | None = None,
) -> tuple[list[DiarizationTurn], list[float], RefineStats]:
    """完整 refine 流水线。"""
    opts = options or RefineOptions()
    stats = RefineStats()
    original_count = len(turns)

    merged = merge_adjacent_turns(turns, max_gap=opts.merge_turn_gap)
    stats.merged_turn_count = original_count - len(merged)

    if not opts.use_embedding or embedder is None:
        confidences = [0.75] * len(merged)
        stats.avg_confidence = 0.75
        return merged, confidences, stats

    profiles = build_profiles_for_refine(
        embedder, audio, sr, merged, min_turn_sec=opts.min_turn_sec
    )
    if len(profiles) < 2:
        confidences = [0.75] * len(merged)
        stats.avg_confidence = 0.75
        return merged, confidences, stats

    refined, refine_stats = refine_turn_labels(
        merged,
        profiles,
        embedder,
        audio,
        sr,
        relabel_margin=opts.relabel_margin,
        min_similarity=opts.min_similarity,
    )
    stats.relabeled_count = refine_stats.relabeled_count
    stats.embedding_similarities = refine_stats.embedding_similarities[:20]
    stats.avg_confidence = refine_stats.avg_confidence

    confidences = [float(t.confidence) for t in refined]
    _, dropped = filter_low_confidence_turns(
        refined, confidences, min_similarity=opts.min_similarity
    )
    stats.filtered_count = dropped
    if opts.drop_low_confidence:
        filtered, _ = filter_low_confidence_turns(
            refined, confidences, min_similarity=opts.min_similarity
        )
        confidences = [float(t.confidence) for t in filtered]
        return filtered, confidences, stats
    return refined, confidences, stats
