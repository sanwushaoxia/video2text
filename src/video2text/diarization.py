"""pyannote 说话人日记化 (Community-1 / 3.1) 与重叠区间检测。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
LEGACY_DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"


@dataclass
class DiarizationTurn:
    start: float
    end: float
    speaker_id: str


@dataclass
class OverlapRegion:
    start: float
    end: float
    speaker_ids: list[str] = field(default_factory=list)


@dataclass
class DiarizationResult:
    turns: list[DiarizationTurn]
    overlap_regions: list[OverlapRegion]
    model: str
    speaker_ids: list[str] = field(default_factory=list)


def _resolve_hf_token(hf_token: str | None) -> str:
    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError(
            "pyannote 需要 HuggingFace Token: 在 "
            "https://huggingface.co/pyannote/speaker-diarization-community-1 "
            "接受协议后设置环境变量 HF_TOKEN"
        )
    return token


def _load_diarization_pipeline(model_id: str, token: str):
    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "日记化需要 pyannote.audio, 请安装: pip install -r requirements-diarize.txt"
        ) from exc

    import torch

    kwargs: dict = {"token": token}
    try:
        pipeline = Pipeline.from_pretrained(model_id, **kwargs)
    except TypeError:
        pipeline = Pipeline.from_pretrained(model_id, use_auth_token=token)

    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    return pipeline


def _iter_diarization_output(output) -> list[tuple[float, float, str]]:
    """兼容 community-1 新 API 与 3.1 Annotation。"""
    if hasattr(output, "speaker_diarization"):
        rows: list[tuple[float, float, str]] = []
        for turn, speaker in output.speaker_diarization:
            rows.append((float(turn.start), float(turn.end), str(speaker)))
        if rows:
            return rows
        if hasattr(output, "exclusive_speaker_diarization"):
            for turn, speaker in output.exclusive_speaker_diarization:
                rows.append((float(turn.start), float(turn.end), str(speaker)))
        return rows

    rows = []
    for turn, _, speaker in output.itertracks(yield_label=True):
        rows.append((float(turn.start), float(turn.end), str(speaker)))
    return rows


def run_diarization(
    vocals_path: Path,
    *,
    min_speakers: int = 1,
    max_speakers: int = 4,
    hf_token: str | None = None,
    model_id: str = DEFAULT_DIARIZATION_MODEL,
) -> DiarizationResult:
    """运行 pyannote 日记化, 返回说话片段与重叠区间。"""
    token = _resolve_hf_token(hf_token)
    print(
        f"说话人日记化 ({model_id}, speakers={min_speakers}..{max_speakers})..."
    )

    pipeline = _load_diarization_pipeline(model_id, token)
    try:
        output = pipeline(
            str(vocals_path),
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
    except TypeError:
        output = pipeline(str(vocals_path))

    turns: list[DiarizationTurn] = []
    for start, end, speaker in _iter_diarization_output(output):
        if end <= start:
            continue
        turns.append(DiarizationTurn(start=start, end=end, speaker_id=speaker))
    turns.sort(key=lambda t: (t.start, t.speaker_id))

    speaker_ids = sorted({t.speaker_id for t in turns})
    overlap_regions = detect_overlap_regions(turns)
    return DiarizationResult(
        turns=turns,
        overlap_regions=overlap_regions,
        model=model_id,
        speaker_ids=speaker_ids,
    )


def detect_overlap_regions(
    turns: list[DiarizationTurn],
    *,
    min_overlap_sec: float = 0.15,
) -> list[OverlapRegion]:
    """从日记化片段中检测多人同时说话的区间。"""
    if len(turns) < 2:
        return []

    events: list[tuple[float, int, str]] = []
    for turn in turns:
        events.append((turn.start, 1, turn.speaker_id))
        events.append((turn.end, -1, turn.speaker_id))
    events.sort(key=lambda item: (item[0], -item[1]))

    active: dict[str, int] = {}
    overlap_regions: list[OverlapRegion] = []
    prev_t: float | None = None

    def active_speakers() -> list[str]:
        return sorted(sp for sp, count in active.items() if count > 0)

    for t, delta, speaker in events:
        if prev_t is not None and t > prev_t:
            speakers = active_speakers()
            if len(speakers) >= 2 and (t - prev_t) >= min_overlap_sec:
                if overlap_regions and overlap_regions[-1].end >= prev_t - 1e-6:
                    region = overlap_regions[-1]
                    merged = sorted(set(region.speaker_ids) | set(speakers))
                    overlap_regions[-1] = OverlapRegion(
                        start=region.start,
                        end=t,
                        speaker_ids=merged,
                    )
                else:
                    overlap_regions.append(
                        OverlapRegion(start=prev_t, end=t, speaker_ids=speakers)
                    )
        active[speaker] = active.get(speaker, 0) + delta
        if active[speaker] <= 0:
            active.pop(speaker, None)
        prev_t = t

    return overlap_regions


def turns_to_assignments(turns: list[DiarizationTurn]) -> tuple[list[dict], list[dict]]:
    """日记化片段 → assignments / report 列表。"""
    assignments: list[dict] = []
    report: list[dict] = []
    for idx, turn in enumerate(turns, start=1):
        entry = {
            "start": turn.start,
            "end": turn.end,
            "speaker_id": turn.speaker_id,
            "index": idx,
            "text": "",
            "confidence": 0.75,
            "reason": f"diarization:{turn.speaker_id}",
        }
        assignments.append(dict(entry))
        report.append(dict(entry))
    return assignments, report
