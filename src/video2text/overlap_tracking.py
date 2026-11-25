"""整片独白声纹与连续分窗的联合角色归属。"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import numpy as np

from video2text.performance import timed_stage
from video2text.speaker_embedding import cosine_similarity


@dataclass
class Window:
    start: int
    end: int
    read_start: int
    sources: tuple[np.ndarray, ...]
    report: dict

    @property
    def core(self) -> slice:
        return slice(self.start - self.read_start, self.end - self.read_start)


@timed_stage("tracking_anchors")
def build_anchors(embedder, audio, sr, segments):
    """从时间上分散的干净独白建立每个角色的参考声纹。"""
    anchors = {}
    for speaker in {segment["speaker_id"] for segment in segments}:
        turns = sorted(
            (s for s in segments if s["speaker_id"] == speaker and not s.get("low_confidence")),
            key=lambda s: s["start"],
        )
        kept = []
        for index in np.linspace(0, len(turns) - 1, min(len(turns), 10), dtype=int):
            turn = turns[index]
            start, end = float(turn["start"]), float(turn["end"])
            center = (start + end) / 2
            embedding = embedder.embed_segment(audio, sr, max(start, center - 2), min(end, center + 2))
            if np.isfinite(embedding).all() and np.linalg.norm(embedding) > 1e-6:
                kept.append((center, embedding))
        if kept:
            anchors[speaker] = kept
    return anchors


def _score_window(window, sr, active, profiles, anchors, embedder, minimum):
    rows = []
    center = (window.start + window.end) / (2 * sr)
    for signal in window.sources:
        emb = embedder.embed_waveform(signal[window.core], sr)
        scores = []
        for speaker in active:
            base = cosine_similarity(emb, profiles[speaker])
            clips = anchors.get(speaker, [])
            if clips:
                nearest = min(clips, key=lambda clip: abs(clip[0] - center))
                best = max(cosine_similarity(emb, anchor) for _, anchor in clips)
                base = 0.55 * base + 0.25 * best + 0.20 * cosine_similarity(emb, nearest[1])
            scores.append(base)
        rows.append(scores)
    values = np.asarray(rows)
    orders = []
    emissions = []
    for order in permutations(range(len(active))):
        selected = values[np.arange(len(active)), order]
        if np.isfinite(selected).all() and np.min(selected) >= minimum:
            orders.append(order)
            emissions.append(float(np.mean(selected)))
    return orders, np.asarray(emissions)


def _link(left, right, left_orders, right_orders, sr):
    """对相邻预测在共同时间轴上的波形给连续性加分。"""
    start = max(left.read_start, right.read_start)
    end = min(left.read_start + len(left.sources[0]), right.read_start + len(right.sources[0]))
    scores = np.zeros((len(left_orders), len(right_orders)))
    if end - start < round(0.04 * sr):
        return scores
    left_span = slice(start - left.read_start, end - left.read_start)
    right_span = slice(start - right.read_start, end - right.read_start)
    similarities = np.array([
        [abs(cosine_similarity(a[left_span], b[right_span])) for b in right.sources]
        for a in left.sources
    ])
    for i, a in enumerate(left_orders):
        for j, b in enumerate(right_orders):
            # If the common context is silent, cosine similarity is zero.
            scores[i, j] = 0.15 * np.mean(similarities[np.argsort(a), np.argsort(b)])
    return scores


def _decode_chain(windows, candidates, sr, active):
    orders, emissions = zip(*candidates)
    links = [_link(a, b, oa, ob, sr)
             for a, b, oa, ob in zip(windows, windows[1:], orders, orders[1:])]
    forward = [emissions[0]]
    for emission, link in zip(emissions[1:], links):
        forward.append(emission + np.max(forward[-1][:, None] + link, axis=0))
    backward = np.zeros_like(emissions[-1])
    for i in range(len(windows) - 1, -1, -1):
        total = forward[i] + backward
        ranked = np.argsort(total)
        accepted = len(ranked) == 1 or total[ranked[-1]] - total[ranked[-2]] >= 0.03
        if accepted:
            windows[i].report["source_speakers"] = [active[j] for j in orders[i][ranked[-1]]]
        if i:
            backward = np.max(links[i - 1] + (emissions[i] + backward)[None, :], axis=1)


def track_windows(windows, sr, active, profiles, anchors, embedder, minimum):
    chain, candidates = [], []
    for window in windows:
        try:
            orders, emissions = _score_window(window, sr, active, profiles, anchors, embedder, minimum)
        except Exception as exc:
            window.report.update(reason="inference_failed", error=str(exc))
            orders = []
        if chain and (chain[-1].end != window.start or not orders):
            _decode_chain(chain, candidates, sr, active)
            chain, candidates = [], []
        if orders:
            chain.append(window)
            candidates.append((orders, emissions))
    if chain:
        _decode_chain(chain, candidates, sr, active)
