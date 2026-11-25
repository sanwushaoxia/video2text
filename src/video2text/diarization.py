"""pyannote 说话人日记化 (Community-1 / 3.1) 与重叠区间检测。"""
from __future__ import annotations

import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from functools import lru_cache
import json
from pathlib import Path
from threading import RLock

import numpy as np

from video2text.performance import timed_stage
from video2text.paths import ensure_huggingface_env, huggingface_hub_cache, repo_root
from video2text.stage_cache import get_cache, inference_environment, model_artifacts, versions

DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
LEGACY_DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
HF_TOKEN_FILE_ENV = "VIDEO2TEXT_HF_TOKEN_FILE"
_pipeline_lock = RLock()


@dataclass
class DiarizationTurn:
    start: float
    end: float
    speaker_id: str
    confidence: float = 0.75


@dataclass
class OverlapRegion:
    start: float
    end: float
    speaker_ids: list[str] = field(default_factory=list)
    reason: str | None = None


@dataclass
class DiarizationResult:
    turns: list[DiarizationTurn]
    overlap_regions: list[OverlapRegion]
    model: str
    speaker_ids: list[str] = field(default_factory=list)
    overlap_detection_source: str = "timeline"
    overlap_review_regions: list[OverlapRegion] = field(default_factory=list)


def _hf_token_search_paths() -> list[Path]:
    paths: list[Path] = []
    path_env = os.environ.get(HF_TOKEN_FILE_ENV, "").strip()
    if path_env:
        paths.append(Path(path_env).expanduser().resolve())
    repo_root_path = repo_root()
    paths.append((repo_root_path / "hf_token").resolve())
    local_app = os.environ.get("LOCALAPPDATA", "").strip()
    if local_app:
        paths.append((Path(local_app) / "video2text" / "hf_token").resolve())
    paths.append((Path.home() / ".cache" / "video2text" / "hf_token").resolve())
    return paths


def _default_hf_token_file() -> Path:
    for path in _hf_token_search_paths():
        if path.is_file():
            return path
    return _hf_token_search_paths()[0]


def _read_hf_token_file() -> str | None:
    for path in _hf_token_search_paths():
        if not path.is_file():
            continue
        token = path.read_text(encoding="utf-8").strip()
        if token:
            return token
    return None


def _resolve_hf_token(hf_token: str | None) -> str:
    token = (
        hf_token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or _read_hf_token_file()
    )
    if not token:
        repo_token = repo_root() / "hf_token"
        raise RuntimeError(
            "pyannote 需要 HuggingFace Token。\n"
            "  1) Accept: https://huggingface.co/pyannote/speaker-diarization-community-1\n"
            "  2) 任选其一:\n"
            "     export HF_TOKEN=hf_xxxx\n"
            "     python ... --hf-token hf_xxxx\n"
            f"     写入 {repo_token} 或 {_default_hf_token_file()} (单行, 勿提交 git)\n"
            "  国内镜像: export HF_ENDPOINT=https://hf-mirror.com"
        )
    return token


def _ensure_hf_hub_env() -> None:
    ensure_huggingface_env()


@lru_cache(maxsize=1)
@timed_stage("load_diarization_model")
def _load_diarization_pipeline(model_id: str, token: str):
    _ensure_hf_hub_env()
    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "日记化需要 pyannote.audio, 请安装: pip install -r requirements.txt"
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


def prepare_diarization_audio(
    vocals_path: Path,
    work_dir: Path,
    *,
    sr: int = 16000,
    highpass_hz: float = 80.0,
) -> Path:
    """峰值归一化 + 高通, 仅用于 pyannote 输入。"""
    import librosa
    import soundfile as sf

    work_dir.mkdir(parents=True, exist_ok=True)
    audio, _ = librosa.load(vocals_path, sr=sr, mono=True)
    audio = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(audio)))
    if peak > 1e-8:
        audio = audio / peak * 0.95
    try:
        from scipy.signal import butter, sosfilt

        sos = butter(2, highpass_hz, btype="highpass", fs=sr, output="sos")
        audio = sosfilt(sos, audio).astype(np.float32)
    except ImportError:
        pass
    out = work_dir / "diarize_input.wav"
    sf.write(str(out), audio, sr)
    return out


def _extract_native_overlap(output) -> list[OverlapRegion]:
    """尝试从 pyannote community-1 原生 overlap / multilabel 输出提取重叠。"""
    regions: list[OverlapRegion] = []
    candidates = [
        getattr(output, name, None)
        for name in (
            "speaker_overlap",
            "overlap",
            "overlapped_speech",
            "multilabel",
        )
    ]
    for ann in candidates:
        if ann is None:
            continue
        try:
            if hasattr(ann, "itertracks"):
                for turn, _, label in ann.itertracks(yield_label=True):
                    label_str = str(label).lower()
                    if "overlap" in label_str or "+" in label_str:
                        regions.append(
                            OverlapRegion(
                                start=float(turn.start),
                                end=float(turn.end),
                                speaker_ids=[],
                            )
                        )
            elif hasattr(ann, "__iter__"):
                for item in ann:
                    if hasattr(item, "start") and hasattr(item, "end"):
                        regions.append(
                            OverlapRegion(
                                start=float(item.start),
                                end=float(item.end),
                                speaker_ids=[],
                            )
                        )
        except Exception:
            continue
    return regions


def _merge_overlap_regions(regions: list[OverlapRegion]) -> list[OverlapRegion]:
    if not regions:
        return []
    ordered = sorted(regions, key=lambda r: r.start)
    merged: list[OverlapRegion] = [ordered[0]]
    for region in ordered[1:]:
        prev = merged[-1]
        if region.start <= prev.end + 1e-3:
            merged[-1] = OverlapRegion(
                start=prev.start,
                end=max(prev.end, region.end),
                speaker_ids=sorted(set(prev.speaker_ids) | set(region.speaker_ids)),
            )
        else:
            merged.append(region)
    return merged


def detect_proximity_overlap_candidates(
    turns: list[DiarizationTurn],
    *,
    window_sec: float = 0.5,
    min_overlap_sec: float = 0.15,
    max_region_sec: float = 3.0,
) -> list[OverlapRegion]:
    """相邻双 speaker turn 近邻启发式 overlap 候选。"""
    if len(turns) < 2:
        return []
    ordered = sorted(turns, key=lambda t: t.start)
    regions: list[OverlapRegion] = []
    for i, t1 in enumerate(ordered):
        for t2 in ordered[i + 1 :]:
            if t2.start - t1.end > window_sec:
                break
            if t1.speaker_id == t2.speaker_id:
                continue
            overlap_start = max(t1.start, t2.start)
            overlap_end = min(t1.end, t2.end)
            if overlap_end - overlap_start >= min_overlap_sec:
                regions.append(
                    OverlapRegion(
                        start=overlap_start,
                        end=overlap_end,
                        speaker_ids=[t1.speaker_id, t2.speaker_id],
                    )
                )
                continue
            gap = t2.start - t1.end
            if 0 <= gap <= window_sec:
                start = t1.end
                end = t2.start
                if end - start >= min_overlap_sec:
                    regions.append(
                        OverlapRegion(
                            start=start,
                            end=end,
                            speaker_ids=[t1.speaker_id, t2.speaker_id],
                        )
                    )
                else:
                    start = min(t1.start, t2.start)
                    end = max(t1.end, t2.end)
                    if (
                        min_overlap_sec <= (end - start) <= max_region_sec
                        and t2.start <= t1.end + window_sec
                    ):
                        regions.append(
                            OverlapRegion(
                                start=start,
                                end=end,
                                speaker_ids=[t1.speaker_id, t2.speaker_id],
                            )
                        )
    return _merge_overlap_regions(regions)


def _posterior_review_regions(pipeline, audio: Path, threshold: float, min_sec: float):
    """从 powerset 模型的多人类别后验补充待复核区间。"""
    segmentation = getattr(pipeline, "_segmentation", None)
    conversion = getattr(segmentation, "conversion", None)
    if not hasattr(conversion, "mapping"):
        return []

    from pyannote.audio import Inference

    overlap_classes = conversion.mapping.sum(dim=-1).cpu().numpy() >= 2

    def overlap_probability(log_probs):
        return np.exp(log_probs[..., overlap_classes]).sum(axis=-1, keepdims=True)

    inference = Inference(
        segmentation.model, duration=segmentation.duration, step=segmentation.step,
        device=segmentation.device, batch_size=segmentation.batch_size,
        skip_conversion=True, pre_aggregation_hook=overlap_probability,
    )
    posterior = inference(str(audio))
    window = posterior.sliding_window
    start = window.start + (window.duration - window.step) / 2
    return _threshold_overlap_posterior(posterior.data[:, 0], start, window.step, threshold, min_sec)


def _threshold_overlap_posterior(scores, start, step, threshold, min_sec):
    boundaries = np.flatnonzero(np.diff(np.r_[False, scores >= threshold, False]))
    regions = []
    for left, right in boundaries.reshape(-1, 2):
        begin = max(0.0, start + left * step)
        end = start + right * step
        if end - begin >= min_sec:
            regions.append(OverlapRegion(begin, end))
    return regions


def _activity_sweep(left, right):
    events = defaultdict(list)
    for side, spans in enumerate((left, right)):
        for start, end, labels in spans:
            for label in labels:
                events[start].append((side, label, 1))
                events[end].append((side, label, -1))
    active = [Counter(), Counter()]
    times = sorted(events)
    for start, end in zip(times, times[1:]):
        for side, label, count in events[start]:
            active[side][label] += count
            if not active[side][label]:
                del active[side][label]
        yield start, end, set(active[0]), set(active[1])


def fuse_diarization(mix: DiarizationResult, vocals: DiarizationResult) -> DiarizationResult:
    """保留原混音的角色时间轴，用两路检测的一致性决定是否自动分离。"""
    from scipy.optimize import linear_sum_assignment

    if not mix.turns:
        return vocals
    mix_regions = mix.overlap_regions if mix.overlap_detection_source != "proximity" else []
    vocal_regions = vocals.overlap_regions if vocals.overlap_detection_source != "proximity" else []
    names = [sorted({t.speaker_id for t in result.turns}) for result in (mix, vocals)]
    affinity = np.zeros(tuple(map(len, names)))
    mix_index, vocal_index = ({name: i for i, name in enumerate(group)} for group in names)
    # Only unambiguous activity contributes to the cross-view speaker mapping.
    solo_spans = ([(t.start, t.end, [t.speaker_id]) for t in result.turns]
                  for result in (mix, vocals))
    for left, right, a, b in _activity_sweep(*solo_spans):
        if len(a) == len(b) == 1:
            affinity[mix_index[a.pop()], vocal_index[b.pop()]] += right - left
    rows, columns = linear_sum_assignment(affinity, maximize=True)
    mapping = {names[1][j]: names[0][i] for i, j in zip(rows, columns) if affinity[i, j] >= 0.25}

    confirmed, review = [], []
    overlap_spans = ([(r.start, r.end, r.speaker_ids or [None]) for r in regions]
                     for regions in (mix_regions, vocal_regions))
    for left, right, a, b in _activity_sweep(*overlap_spans):
        if not (a or b):
            continue
        if None not in a and len(a) in (2, 3) and a == {mapping.get(s) for s in b}:
            destination, speakers, reason = confirmed, sorted(a), None
        else:
            destination, speakers, reason = review, sorted(a - {None}), "joint_disagreement"
        if (destination and destination[-1].end == left
                and destination[-1].speaker_ids == speakers and destination[-1].reason == reason):
            destination[-1].end = right
        else:
            destination.append(OverlapRegion(left, right, speakers, reason))
    return DiarizationResult(
        mix.turns, confirmed, mix.model, mix.speaker_ids, "joint",
        review + mix.overlap_review_regions + vocals.overlap_review_regions,
    )


@timed_stage("diarization")
def run_diarization(
    vocals_path: Path,
    *,
    min_speakers: int = 1,
    max_speakers: int = 4,
    hf_token: str | None = None,
    model_id: str = DEFAULT_DIARIZATION_MODEL,
    diarize_preprocess: bool = True,
    work_dir: Path | None = None,
    overlap_min_sec: float = 0.15,
    overlap_review_threshold: float | None = None,
) -> DiarizationResult:
    options = dict(
        min_speakers=min_speakers, max_speakers=max_speakers, model_id=model_id,
        diarize_preprocess=diarize_preprocess, overlap_min_sec=overlap_min_sec,
        overlap_review_threshold=overlap_review_threshold,
    )
    cache = get_cache()
    if cache is None:
        return _infer_diarization(vocals_path, hf_token=hf_token, work_dir=work_dir, **options)
    local = Path(model_id).expanduser()
    model_root = local if local.exists() else huggingface_hub_cache() / ("models--" + model_id.replace("/", "--"))
    signature = {
        "input": cache.fingerprint(vocals_path), "options": options,
        "preprocess_applied": diarize_preprocess and work_dir is not None,
        "weights": model_artifacts(model_root),
        "code": cache.fingerprint(Path(__file__)),
        "packages": versions("pyannote.audio", "torch", "librosa", "soundfile"),
        "environment": inference_environment(),
    }

    def build(directory):
        result = _infer_diarization(vocals_path, hf_token=hf_token, work_dir=work_dir, **options)
        (directory / "timeline.json").write_text(json.dumps(asdict(result)), encoding="utf-8")

    def read(directory):
        return json.loads((directory / "timeline.json").read_text(encoding="utf-8"))

    (artifact,) = cache.files(
        "diarization", signature, ("timeline.json",), build,
        validate=lambda directory: "turns" in read(directory),
    )
    payload = read(artifact.parent)
    payload["turns"] = [DiarizationTurn(**turn) for turn in payload["turns"]]
    for field in ("overlap_regions", "overlap_review_regions"):
        payload[field] = [OverlapRegion(**row) for row in payload[field]]
    return DiarizationResult(**payload)


def _infer_diarization(
    vocals_path: Path, *, hf_token, work_dir, min_speakers, max_speakers,
    model_id, diarize_preprocess, overlap_min_sec, overlap_review_threshold,
) -> DiarizationResult:
    """运行 pyannote 日记化, 返回说话片段与重叠区间。"""
    token = _resolve_hf_token(hf_token)
    print(
        f"说话人日记化 ({model_id}, speakers={min_speakers}..{max_speakers})..."
    )

    diarize_input = vocals_path
    if diarize_preprocess and work_dir is not None:
        diarize_input = prepare_diarization_audio(vocals_path, work_dir)
        print(f"  日记化预处理: {diarize_input.name}")

    with _pipeline_lock:
        pipeline = _load_diarization_pipeline(model_id, token)
        try:
            output = pipeline(
                str(diarize_input),
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
        except TypeError:
            output = pipeline(str(diarize_input))
        posterior_regions = (
            _posterior_review_regions(pipeline, diarize_input, overlap_review_threshold, overlap_min_sec)
            if overlap_review_threshold is not None else []
        )

    turns: list[DiarizationTurn] = []
    for start, end, speaker in _iter_diarization_output(output):
        if end <= start:
            continue
        turns.append(DiarizationTurn(start=start, end=end, speaker_id=speaker))
    turns.sort(key=lambda t: (t.start, t.speaker_id))

    speaker_ids = sorted({t.speaker_id for t in turns})
    overlap_source = "timeline"
    overlap_regions = detect_overlap_regions(turns, min_overlap_sec=0.0)
    if not overlap_regions:
        native = _extract_native_overlap(output)
        if native:
            overlap_regions = _merge_overlap_regions(native)
            overlap_source = "native"
        else:
            overlap_regions = detect_proximity_overlap_candidates(
                turns, min_overlap_sec=overlap_min_sec
            )
            if overlap_regions:
                overlap_source = "proximity"
    return DiarizationResult(
        turns=turns,
        overlap_regions=overlap_regions,
        model=model_id,
        speaker_ids=speaker_ids,
        overlap_detection_source=overlap_source,
        overlap_review_regions=posterior_regions,
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
            if len(speakers) >= 2:
                if (overlap_regions and overlap_regions[-1].end == prev_t
                        and overlap_regions[-1].speaker_ids == speakers):
                    overlap_regions[-1].end = t
                else:
                    overlap_regions.append(
                        OverlapRegion(start=prev_t, end=t, speaker_ids=speakers)
                    )
        active[speaker] = active.get(speaker, 0) + delta
        if active[speaker] <= 0:
            active.pop(speaker, None)
        prev_t = t

    return [r for r in overlap_regions if r.end - r.start >= min_overlap_sec]


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
            "confidence": round(float(turn.confidence), 3),
            "reason": f"diarization:{turn.speaker_id}",
        }
        assignments.append(dict(entry))
        report.append(dict(entry))
    return assignments, report


def estimate_speaker_count(
    vocals_path: Path,
    *,
    max_speakers: int = 8,
    hf_token: str | None = None,
    model_id: str = DEFAULT_DIARIZATION_MODEL,
    diarize_preprocess: bool = True,
    work_dir: Path | None = None,
) -> dict:
    """仅运行日记化, 返回说话人数量与每人时长分布, 不做切分。

    用于分离前先确认音频里一共有多少个人发声。
    """
    result = run_diarization(
        vocals_path,
        min_speakers=1,
        max_speakers=max_speakers,
        hf_token=hf_token,
        model_id=model_id,
        diarize_preprocess=diarize_preprocess,
        work_dir=work_dir,
    )

    per_speaker_sec: dict[str, float] = {}
    per_speaker_turns: dict[str, int] = {}
    for t in result.turns:
        dur = max(0.0, t.end - t.start)
        per_speaker_sec[t.speaker_id] = per_speaker_sec.get(t.speaker_id, 0.0) + dur
        per_speaker_turns[t.speaker_id] = per_speaker_turns.get(t.speaker_id, 0) + 1

    # 按时长降序排列, 最能代表"主要说话人"
    sorted_speakers = sorted(
        per_speaker_sec.items(), key=lambda item: item[1], reverse=True
    )

    total_speech_sec = sum(per_speaker_sec.values())

    return {
        "speaker_count": len(result.speaker_ids),
        "speaker_ids": result.speaker_ids,
        "per_speaker_sec": {
            sp: round(sec, 2) for sp, sec in sorted_speakers
        },
        "per_speaker_turns": {
            sp: per_speaker_turns[sp] for sp, _ in sorted_speakers
        },
        "total_speech_sec": round(total_speech_sec, 2),
        "overlap_region_count": len(result.overlap_regions),
        "overlap_detection_source": result.overlap_detection_source,
        "turn_count": len(result.turns),
        "model": result.model,
    }
