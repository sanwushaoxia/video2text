from __future__ import annotations

import json
from pathlib import Path

from video2text.f0_analysis import (
    analyze_segment_f0,
    classify_segment_gender,
    load_vocals_mono,
)
from video2text.media import get_media_duration
from video2text.speaker_map import (
    apply_speaker_map_override,
    assign_gender_to_segments,
)
from video2text.srt import parse_srt
from video2text.whisper_transcribe import detect_and_transcribe, load_whisper_model


def segments_from_whisper(
    vocals_path: Path,
    *,
    model_name: str = "base",
    language: str | None = None,
    device: str | None = None,
    task: str = "transcribe",
) -> tuple[list[dict], str, float]:
    """Whisper 转写 vocals, 返回 (segments, detected_lang, confidence)。"""
    model, _device = load_whisper_model(model_name, device)
    result, detected, conf = detect_and_transcribe(
        model, vocals_path, language=language, task=task
    )
    segments = result.get("segments") or []
    return segments, detected, conf


def segments_from_srt(srt_path: Path) -> list[dict]:
    """从 SRT 读取完整时间轴。"""
    body = srt_path.read_text(encoding="utf-8")
    segments = parse_srt(body)
    if not segments:
        raise RuntimeError(f"SRT 无有效字幕段: {srt_path}")
    return segments


def segments_from_diarization(
    vocals_path: Path,
    *,
    min_speakers: int = 1,
    max_speakers: int = 4,
    hf_token: str | None = None,
) -> list[dict]:
    """pyannote 说话人日记化, 返回 {start, end, speaker_id} 列表。"""
    from video2text.diarization import run_diarization, turns_to_assignments

    diar = run_diarization(
        vocals_path,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        hf_token=hf_token,
    )
    assignments, _ = turns_to_assignments(diar.turns)
    return [
        {
            "start": float(a["start"]),
            "end": float(a["end"]),
            "speaker_id": str(a["speaker_id"]),
        }
        for a in assignments
    ]


def label_speakers_by_f0(
    vocals_path: Path,
    turns: list[dict],
    *,
    f0_threshold: float = 200.0,
    min_voiced_ratio: float = 0.25,
    sr: int = 44100,
) -> dict[str, str]:
    """对每个 speaker_id 聚合 F0, 返回 speaker_id -> male|female。"""
    audio, _ = load_vocals_mono(vocals_path, sr=sr)
    speaker_f0s: dict[str, list[float]] = {}

    for turn in turns:
        speaker = str(turn.get("speaker_id", "unknown"))
        f0, voiced = analyze_segment_f0(
            audio, sr, float(turn["start"]), float(turn["end"])
        )
        if f0 is not None and voiced >= min_voiced_ratio * 0.5:
            speaker_f0s.setdefault(speaker, []).append(f0)

    labels: dict[str, str] = {}
    for speaker, f0_list in speaker_f0s.items():
        if not f0_list:
            labels[speaker] = "female"
            continue
        median_f0 = sorted(f0_list)[len(f0_list) // 2]
        gender, _ = classify_segment_gender(
            median_f0,
            1.0,
            f0_threshold=f0_threshold,
            min_voiced_ratio=min_voiced_ratio,
        )
        labels[speaker] = gender

    for turn in turns:
        speaker = str(turn.get("speaker_id", "unknown"))
        labels.setdefault(speaker, "female")

    return labels


def assign_gender_diarize_f0(
    vocals_path: Path,
    turns: list[dict],
    *,
    f0_threshold: float = 200.0,
    min_voiced_ratio: float = 0.25,
    sr: int = 44100,
) -> tuple[list[dict], list[dict]]:
    """日记化片段 + speaker F0 聚类, 返回 (assignments, report)。"""
    speaker_labels = label_speakers_by_f0(
        vocals_path,
        turns,
        f0_threshold=f0_threshold,
        min_voiced_ratio=min_voiced_ratio,
        sr=sr,
    )
    audio, _ = load_vocals_mono(vocals_path, sr=sr)
    assignments: list[dict] = []
    report: list[dict] = []

    for idx, turn in enumerate(turns, start=1):
        start = float(turn["start"])
        end = float(turn["end"])
        speaker = str(turn.get("speaker_id", "unknown"))
        gender = speaker_labels.get(speaker, "female")
        f0, voiced_ratio = analyze_segment_f0(audio, sr, start, end)
        entry = {
            "start": start,
            "end": end,
            "gender": gender,
            "index": idx,
            "speaker_id": speaker,
            "text": "",
            "f0_hz": round(f0, 1) if f0 is not None else None,
            "voiced_ratio": round(voiced_ratio, 2),
            "reason": f"speaker_{speaker}->{gender}",
            "confidence": None,
        }
        assignments.append(entry)
        report.append(dict(entry))

    return assignments, report


def _segment_confidence(seg: dict) -> float:
    raw = seg.get("confidence")
    if raw is None:
        return 0.5
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.5


def build_gender_stems(
    vocals_path: Path,
    assignments: list[dict],
    out_male: Path,
    out_female: Path,
    *,
    sr: int = 44100,
) -> dict:
    """按 SRT/日记化时间轴将 vocals 切分到男/女两轨。"""
    import numpy as np

    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            "写入 wav 需要 soundfile (通常随 librosa 安装): pip install soundfile"
        ) from exc

    audio, _ = load_vocals_mono(vocals_path, sr=sr)
    n = len(audio)
    owner = np.full(n, -1, dtype=np.int32)
    owner_conf = np.zeros(n, dtype=np.float32)

    indexed = list(enumerate(assignments))
    indexed.sort(key=lambda item: _segment_confidence(item[1]), reverse=True)

    for seg_idx, seg in indexed:
        start_i = max(0, int(float(seg["start"]) * sr))
        end_i = min(n, int(float(seg["end"]) * sr))
        if end_i <= start_i:
            continue
        conf = _segment_confidence(seg)
        gender = seg.get("gender", "female")
        center = (float(seg["start"]) + float(seg["end"])) / 2.0

        for i in range(start_i, end_i):
            prev = owner[i]
            if prev < 0:
                owner[i] = seg_idx
                owner_conf[i] = conf
                continue

            prev_seg = assignments[prev]
            prev_gender = prev_seg.get("gender", "female")
            if prev_gender == gender:
                if conf > owner_conf[i]:
                    owner[i] = seg_idx
                    owner_conf[i] = conf
                continue

            if conf > owner_conf[i]:
                owner[i] = seg_idx
                owner_conf[i] = conf
            elif conf == owner_conf[i]:
                prev_center = (float(prev_seg["start"]) + float(prev_seg["end"])) / 2.0
                t = i / sr
                if abs(t - center) < abs(t - prev_center):
                    owner[i] = seg_idx

    coverage_stats = _compute_stem_coverage_stats(audio, sr, owner)

    male = np.zeros(n, dtype=np.float32)
    female = np.zeros(n, dtype=np.float32)
    for i in range(n):
        seg_idx = owner[i]
        if seg_idx < 0:
            continue
        gender = assignments[seg_idx].get("gender", "female")
        sample = audio[i]
        if gender == "male":
            male[i] = sample
        else:
            female[i] = sample

    for track in (male, female):
        peak = float(np.max(np.abs(track)))
        if peak > 0.99:
            track /= peak

    out_male.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_male), male, sr)
    sf.write(str(out_female), female, sr)
    return coverage_stats


def _compute_stem_coverage_stats(
    audio,
    sr: int,
    owner,
    *,
    energy_ratio: float = 0.04,
) -> dict[str, float]:
    import numpy as np

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


def _build_split_stats(
    report: list[dict],
    vocals_duration: float,
    coverage_sec: float | None = None,
    stem_stats: dict | None = None,
) -> dict:
    if coverage_sec is None:
        coverage_sec = sum(
            max(0.0, float(r["end"]) - float(r["start"])) for r in report
        )
    male_sec = sum(
        max(0.0, float(r["end"]) - float(r["start"]))
        for r in report
        if r.get("gender") == "male"
    )
    female_sec = sum(
        max(0.0, float(r["end"]) - float(r["start"]))
        for r in report
        if r.get("gender") == "female"
    )
    low_conf = sum(
        1 for r in report if float(r.get("confidence") or 0) < 0.5
    )
    stats = {
        "male_segments": sum(1 for r in report if r.get("gender") == "male"),
        "female_segments": sum(1 for r in report if r.get("gender") == "female"),
        "male_audio_sec": round(male_sec, 2),
        "female_audio_sec": round(female_sec, 2),
        "labeled_coverage_sec": round(coverage_sec, 2),
        "vocals_duration_sec": round(vocals_duration, 2),
        "coverage_ratio": round(
            coverage_sec / vocals_duration, 3
        )
        if vocals_duration > 0
        else 0.0,
        "low_confidence_count": low_conf,
    }
    if stem_stats:
        stats["stem_assigned_sec"] = stem_stats.get("stem_assigned_sec", 0.0)
        stats["unassigned_vocal_sec"] = stem_stats.get("unassigned_vocal_sec", 0.0)
    return stats


def split_vocals_by_gender(
    vocals_path: Path,
    out_male: Path,
    out_female: Path,
    report_path: Path,
    *,
    split_mode: str = "whisper_f0",
    srt_path: Path | None = None,
    f0_threshold: float = 200.0,
    adaptive_threshold: bool = True,
    min_voiced_ratio: float = 0.25,
    whisper_model: str = "base",
    language: str | None = None,
    whisper_device: str | None = None,
    min_speakers: int = 1,
    max_speakers: int = 4,
    hf_token: str | None = None,
    speaker_map_path: Path | None = None,
    separator_meta: dict | None = None,
) -> list[dict]:
    """自动标注性别并输出男/女两轨 + JSON 报告。"""
    meta: dict = {
        "vocals": str(vocals_path),
        "split_mode": split_mode,
        "f0_threshold": f0_threshold,
        "adaptive_threshold": adaptive_threshold,
    }
    if separator_meta:
        meta["separator"] = separator_meta
    vocals_duration = get_media_duration(vocals_path)
    gender_meta: dict = {}

    if split_mode == "srt_f0":
        if srt_path is None or not srt_path.is_file():
            raise ValueError("split_mode=srt_f0 需要 --srt 指向有效字幕文件")
        segments = segments_from_srt(srt_path)
        meta["srt"] = str(srt_path.resolve())
        meta["srt_segment_count"] = len(segments)
        assignments, report, gender_meta = assign_gender_to_segments(
            segments,
            vocals_path,
            f0_threshold=f0_threshold,
            adaptive_threshold=adaptive_threshold,
            min_voiced_ratio=min_voiced_ratio,
        )
    elif split_mode == "whisper_f0":
        segments, detected, conf = segments_from_whisper(
            vocals_path,
            model_name=whisper_model,
            language=language,
            device=whisper_device,
        )
        if not segments:
            raise RuntimeError("Whisper 未返回有效字幕段, 无法切分男女声")
        meta["whisper"] = {
            "model": whisper_model,
            "detected_language": detected,
            "confidence": conf,
            "segment_count": len(segments),
        }
        assignments, report, gender_meta = assign_gender_to_segments(
            segments,
            vocals_path,
            f0_threshold=f0_threshold,
            adaptive_threshold=adaptive_threshold,
            min_voiced_ratio=min_voiced_ratio,
        )
    elif split_mode == "diarize":
        turns = segments_from_diarization(
            vocals_path,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            hf_token=hf_token,
        )
        if not turns:
            raise RuntimeError("日记化未检测到说话片段")
        speaker_labels = label_speakers_by_f0(
            vocals_path,
            turns,
            f0_threshold=f0_threshold,
            min_voiced_ratio=min_voiced_ratio,
        )
        meta["diarization"] = {
            "speaker_count": len(speaker_labels),
            "speaker_labels": speaker_labels,
            "turn_count": len(turns),
        }
        assignments, report = assign_gender_diarize_f0(
            vocals_path,
            turns,
            f0_threshold=f0_threshold,
            min_voiced_ratio=min_voiced_ratio,
        )
        gender_meta = {"f0_threshold_used": f0_threshold}
    else:
        raise ValueError(f"未知 split_mode: {split_mode}")

    if speaker_map_path is not None and speaker_map_path.is_file():
        mapping = apply_speaker_map_override(assignments, report, speaker_map_path)
        meta["speaker_map"] = str(speaker_map_path.resolve())
        meta["speaker_map_override_count"] = len(mapping)

    stem_stats = build_gender_stems(
        vocals_path,
        assignments,
        out_male,
        out_female,
    )

    meta.update(gender_meta)
    meta["segments"] = report
    coverage = gender_meta.get("coverage_sec")
    meta["stats"] = _build_split_stats(
        report, vocals_duration, coverage, stem_stats
    )
    report_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"已写入: {out_male}")
    print(f"已写入: {out_female}")
    print(f"已写入: {report_path}")
    stats = meta["stats"]
    extra = ""
    if "unassigned_vocal_sec" in stats:
        extra = (
            f", stem {stats['stem_assigned_sec']:.1f}s, "
            f"未分配人声 {stats['unassigned_vocal_sec']:.1f}s"
        )
    print(
        f"  覆盖 {stats['labeled_coverage_sec']:.1f}s / "
        f"{stats['vocals_duration_sec']:.1f}s "
        f"({stats['coverage_ratio']:.0%}), "
        f"男 {stats['male_segments']} 段 / 女 {stats['female_segments']} 段"
        f"{extra}"
    )
    return report
