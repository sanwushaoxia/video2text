from __future__ import annotations

import json
import os
from pathlib import Path

from video2text.f0_analysis import (
    analyze_segment_f0,
    classify_segment_gender,
    load_vocals_mono,
)
from video2text.media import get_media_duration
from video2text.sepformer_split import (
    build_gender_stems_sepformer,
    map_sepformer_tracks_to_gender,
    separate_vocals_sepformer,
)
from video2text.segment_align import align_segment_bounds
from video2text.speaker_map import (
    apply_speaker_map_override,
    assign_gender_to_segments,
)
from video2text.vocal_bleed import apply_vocal_bleed_recovery
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
    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "diarize 模式需要 pyannote.audio, 请安装: pip install -r requirements-diarize.txt"
        ) from exc

    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError(
            "pyannote 需要 HuggingFace Token: 在 https://huggingface.co/pyannote/speaker-diarization-3.1 "
            "接受协议后设置环境变量 HF_TOKEN"
        )

    print(
        f"说话人日记化 (pyannote, speakers={min_speakers}..{max_speakers})..."
    )
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=token,
    )
    diarization = pipeline(
        str(vocals_path),
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    segments: list[dict] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append(
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker_id": str(speaker),
            }
        )
    segments.sort(key=lambda s: (s["start"], s.get("speaker_id", "")))
    return segments


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


def _fill_inter_segment_gaps(
    audio,
    sr: int,
    assignments: list[dict],
    *,
    fill_gap_ms: int,
    energy_ratio: float = 0.06,
) -> list[dict]:
    """在相邻标注段之间的短空隙中, 按邻近段 gender 回填虚拟段。"""
    import numpy as np

    if fill_gap_ms <= 0 or len(assignments) < 2:
        return []

    max_gap = fill_gap_ms / 1000.0
    sorted_segs = sorted(assignments, key=lambda s: float(s["start"]))
    gap_fills: list[dict] = []

    for prev, nxt in zip(sorted_segs, sorted_segs[1:]):
        gap_start = float(prev["end"])
        gap_end = float(nxt["start"])
        gap_dur = gap_end - gap_start
        if gap_dur <= 0 or gap_dur > max_gap:
            continue

        g0 = max(0, int(gap_start * sr))
        g1 = min(len(audio), int(gap_end * sr))
        if g1 <= g0:
            continue

        region = audio[g0:g1]
        peak = float(np.max(np.abs(region))) if region.size else 0.0
        if peak < 1e-6:
            continue
        rms = float(np.sqrt(np.mean(region ** 2)))
        if rms < peak * energy_ratio:
            continue

        mid = (gap_start + gap_end) / 2.0
        prev_mid = (float(prev["start"]) + float(prev["end"])) / 2.0
        nxt_mid = (float(nxt["start"]) + float(nxt["end"])) / 2.0
        gender = prev["gender"] if abs(mid - prev_mid) <= abs(mid - nxt_mid) else nxt["gender"]

        gap_fills.append(
            {
                "start": gap_start,
                "end": gap_end,
                "gender": gender,
                "index": None,
                "text": "",
                "reason": "gap_fill",
                "confidence": 0.35,
                "aligned_start": round(gap_start, 3),
                "aligned_end": round(gap_end, 3),
                "gap_fill": True,
            }
        )

    return gap_fills


def _recover_unassigned_in_windows(
    audio,
    sr: int,
    owner,
    owner_conf,
    all_segments: list[dict],
    *,
    window_pad_sec: float = 0.12,
    energy_ratio: float = 0.04,
) -> dict[str, float]:
    """在 SRT 窗内回收 owner=-1 且有人声能量的采样。"""
    import numpy as np

    n = len(audio)
    global_peak = float(np.max(np.abs(audio))) if n else 0.0
    if global_peak < 1e-6:
        return {}

    threshold = global_peak * energy_ratio
    recovered_by_index: dict[str, float] = {}

    indexed = [
        (idx, seg)
        for idx, seg in enumerate(all_segments)
        if not seg.get("gap_fill") and seg.get("index") is not None
    ]
    indexed.sort(key=lambda item: _segment_confidence(item[1]), reverse=True)

    for seg_idx, seg in indexed:
        win_start = float(seg.get("srt_start", seg["start"])) - window_pad_sec
        win_end = float(seg.get("srt_end", seg["end"])) + window_pad_sec
        g0 = max(0, int(win_start * sr))
        g1 = min(n, int(win_end * sr))
        if g1 <= g0:
            continue

        conf = _segment_confidence(seg) * 0.92
        recovered = 0
        for i in range(g0, g1):
            if owner[i] >= 0:
                continue
            if abs(float(audio[i])) < threshold:
                continue
            owner[i] = seg_idx
            owner_conf[i] = conf
            recovered += 1

        if recovered:
            key = str(seg.get("index"))
            recovered_by_index[key] = round(
                recovered_by_index.get(key, 0.0) + recovered / sr,
                3,
            )

    return recovered_by_index


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


def build_gender_stems(
    vocals_path: Path,
    assignments: list[dict],
    out_male: Path,
    out_female: Path,
    *,
    sr: int = 44100,
    slice_pad_ms: int = 120,
    slice_pad_end_ms: int | None = None,
    fill_gap_ms: int = 400,
    recover_window_vocals: bool = True,
    recover_window_pad_ms: int = 120,
) -> dict:
    """按时间段将 vocals 切分到男/女两轨 (含 padding / overlap 解析 / 空隙回填)。"""
    import numpy as np

    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            "写入 wav 需要 soundfile (通常随 librosa 安装): pip install soundfile"
        ) from exc

    audio, _ = load_vocals_mono(vocals_path, sr=sr)
    n = len(audio)
    pad_start = slice_pad_ms / 1000.0
    pad_end = (slice_pad_end_ms if slice_pad_end_ms is not None else slice_pad_ms) / 1000.0

    gap_fills = _fill_inter_segment_gaps(
        audio, sr, assignments, fill_gap_ms=fill_gap_ms
    )
    all_segments = list(assignments) + gap_fills

    owner = np.full(n, -1, dtype=np.int32)
    owner_conf = np.zeros(n, dtype=np.float32)

    indexed = list(enumerate(all_segments))
    indexed.sort(key=lambda item: _segment_confidence(item[1]), reverse=True)

    for seg_idx, seg in indexed:
        start_i = max(0, int((float(seg["start"]) - pad_start) * sr))
        end_i = min(n, int((float(seg["end"]) + pad_end) * sr))
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

            prev_seg = all_segments[prev]
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

    recover_pad = recover_window_pad_ms / 1000.0
    recovered_by_index = {}
    if recover_window_vocals:
        recovered_by_index = _recover_unassigned_in_windows(
            audio,
            sr,
            owner,
            owner_conf,
            all_segments,
            window_pad_sec=recover_pad,
        )

    coverage_stats = _compute_stem_coverage_stats(audio, sr, owner)

    male = np.zeros(n, dtype=np.float32)
    female = np.zeros(n, dtype=np.float32)
    for i in range(n):
        seg_idx = owner[i]
        if seg_idx < 0:
            continue
        gender = all_segments[seg_idx].get("gender", "female")
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
    return {
        **coverage_stats,
        "recovered_by_index": recovered_by_index,
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
        if stem_stats.get("recovered_by_index"):
            stats["recovered_total_sec"] = round(
                sum(stem_stats["recovered_by_index"].values()), 2
            )
    return stats


def split_vocals_by_gender(
    vocals_path: Path,
    out_male: Path,
    out_female: Path,
    report_path: Path,
    *,
    split_mode: str = "whisper_f0",
    gender_backend: str = "slice",
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
    work_dir: Path | None = None,
    slice_pad_ms: int = 120,
    slice_pad_end_ms: int | None = None,
    align_short_segments: bool | None = None,
    fill_gap_ms: int = 400,
    shout_min_ms: int = 600,
    shout_tail_pad_ms: int = 250,
    shout_all_islands: bool = False,
    recover_window_vocals: bool | None = None,
    speaker_map_path: Path | None = None,
    no_vocals_path: Path | None = None,
    recover_vocal_bleed: bool | None = None,
    bleed_leak_ratio: float = 0.70,
    bleed_island_threshold: float = 0.12,
    bleed_min_nv_voc_ratio: float = 1.5,
    bleed_min_excess_ratio: float = 0.15,
    bleed_bgm_attenuate: float = 0.85,
    bleed_fade_ms: float = 8.0,
    separator_meta: dict | None = None,
) -> list[dict]:
    """自动标注性别并输出男/女两轨 + JSON 报告。"""
    meta: dict = {
        "vocals": str(vocals_path),
        "split_mode": split_mode,
        "gender_backend": gender_backend,
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

    if align_short_segments is None:
        align_short_segments = split_mode == "srt_f0"
    if recover_window_vocals is None:
        recover_window_vocals = split_mode == "srt_f0"
    if recover_vocal_bleed is None:
        recover_vocal_bleed = split_mode == "srt_f0"

    audio_mono, sr = load_vocals_mono(vocals_path, sr=44100)
    if align_short_segments and gender_backend == "slice":
        assignments = align_segment_bounds(
            assignments,
            audio_mono,
            sr,
            enabled=True,
            pad_start_sec=slice_pad_ms / 1000.0,
            pad_end_sec=(
                (slice_pad_end_ms if slice_pad_end_ms is not None else slice_pad_ms)
                / 1000.0
            ),
            shout_min_ms=shout_min_ms,
            shout_tail_pad_ms=shout_tail_pad_ms,
            shout_all_islands=shout_all_islands,
        )
        for entry in report:
            idx = entry.get("index")
            for a in assignments:
                if a.get("index") == idx:
                    entry["aligned_start"] = a.get("aligned_start")
                    entry["aligned_end"] = a.get("aligned_end")
                    if "srt_start" in a:
                        entry["srt_start"] = a["srt_start"]
                        entry["srt_end"] = a["srt_end"]
                    if a.get("align_cue_type"):
                        entry["align_cue_type"] = a["align_cue_type"]
                    if a.get("speech_islands"):
                        entry["speech_islands"] = a["speech_islands"]
                    break
    else:
        for entry in report:
            entry["aligned_start"] = round(float(entry["start"]), 3)
            entry["aligned_end"] = round(float(entry["end"]), 3)

    gap_fills = _fill_inter_segment_gaps(
        audio_mono, sr, assignments, fill_gap_ms=fill_gap_ms
    )
    if gap_fills:
        for gf in gap_fills:
            report.append(
                {
                    **gf,
                    "index": len(report) + 1,
                    "f0_hz": None,
                    "voiced_ratio": None,
                }
            )

    meta["slice_pad_ms"] = slice_pad_ms
    meta["slice_pad_end_ms"] = (
        slice_pad_end_ms if slice_pad_end_ms is not None else slice_pad_ms
    )
    meta["align_short_segments"] = align_short_segments
    meta["fill_gap_ms"] = fill_gap_ms
    meta["shout_min_ms"] = shout_min_ms
    meta["shout_tail_pad_ms"] = shout_tail_pad_ms
    meta["shout_all_islands"] = shout_all_islands
    meta["recover_window_vocals"] = recover_window_vocals
    meta["recover_vocal_bleed"] = recover_vocal_bleed
    if gap_fills:
        meta["gap_fill_count"] = len(gap_fills)

    for a in assignments:
        if "srt_start" not in a:
            a["srt_start"] = round(float(a["start"]), 3)
            a["srt_end"] = round(float(a["end"]), 3)

    stem_stats: dict = {}
    if gender_backend == "sepformer":
        if work_dir is None:
            raise ValueError("gender_backend=sepformer 需要 work_dir")
        src0, src1, sep_sr = separate_vocals_sepformer(
            vocals_path, work_dir / "sepformer"
        )
        track_map = map_sepformer_tracks_to_gender(
            src0, src1, sep_sr, assignments
        )
        build_gender_stems_sepformer(
            src0,
            src1,
            sep_sr,
            track_map,
            assignments,
            out_male,
            out_female,
        )
        meta["sepformer"] = {
            "model": "speechbrain/sepformer-wsj02mix",
            "track_map": track_map,
        }
    else:
        stem_stats = build_gender_stems(
            vocals_path,
            assignments,
            out_male,
            out_female,
            sr=sr,
            slice_pad_ms=slice_pad_ms,
            slice_pad_end_ms=slice_pad_end_ms,
            fill_gap_ms=fill_gap_ms,
            recover_window_vocals=bool(recover_window_vocals),
            recover_window_pad_ms=slice_pad_ms,
        )

    recovered = stem_stats.get("recovered_by_index") or {}
    for entry in report:
        key = str(entry.get("index", ""))
        if key in recovered:
            entry["recovered_sec"] = recovered[key]

    bleed_stats: dict = {}
    if (
        recover_vocal_bleed
        and gender_backend == "slice"
        and no_vocals_path is not None
        and no_vocals_path.is_file()
    ):
        bleed_stats = apply_vocal_bleed_recovery(
            assignments,
            vocals_path,
            no_vocals_path,
            out_male,
            out_female,
            sr=sr,
            window_pad_sec=slice_pad_ms / 1000.0,
            leak_ratio=bleed_leak_ratio,
            island_threshold_ratio=bleed_island_threshold,
            min_nv_voc_ratio=bleed_min_nv_voc_ratio,
            min_excess_ratio=bleed_min_excess_ratio,
            bgm_attenuate=bleed_bgm_attenuate,
            fade_ms=bleed_fade_ms,
        )
        meta["vocal_bleed"] = {
            **bleed_stats,
            "leak_ratio": bleed_leak_ratio,
            "island_threshold_ratio": bleed_island_threshold,
            "min_nv_voc_ratio": bleed_min_nv_voc_ratio,
            "min_excess_ratio": bleed_min_excess_ratio,
            "bgm_attenuate": bleed_bgm_attenuate,
            "fade_ms": bleed_fade_ms,
        }
        meta["no_vocals"] = str(no_vocals_path.resolve())
        for entry in report:
            key = str(entry.get("index", ""))
            by_idx = bleed_stats.get("bleed_recovered_by_index") or {}
            if key in by_idx:
                entry["bleed_recovered_sec"] = by_idx[key]
        print(
            f"  背景泄漏回收 {bleed_stats.get('bleed_recovered_sec', 0):.2f}s "
            f"(已清理 no_vocals)"
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
