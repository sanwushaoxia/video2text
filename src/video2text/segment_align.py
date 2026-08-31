"""SRT 段边界精修: 分类对齐 (shout / phrase / short) + 多岛 VAD."""
from __future__ import annotations

from video2text.speaker_map import classify_cue_align_type


def detect_speech_islands(
    audio,
    sr: int,
    start_sec: float,
    end_sec: float,
    *,
    pad_start_sec: float = 0.12,
    pad_end_sec: float = 0.12,
    frame_ms: float = 30.0,
    threshold_ratio: float = 0.08,
    hangover_ms: float = 200.0,
    merge_gap_ms: float = 300.0,
) -> list[tuple[float, float]]:
    """在时间窗内检测所有语声岛 (start_sec, end_sec)。"""
    import numpy as np

    n = len(audio)
    region_start_i = max(0, int((start_sec - pad_start_sec) * sr))
    region_end_i = min(n, int((end_sec + pad_end_sec) * sr))
    if region_end_i <= region_start_i:
        return []

    frame = max(1, int(sr * frame_ms / 1000.0))
    hangover_frames = max(1, int(hangover_ms / frame_ms))
    merge_gap_frames = max(1, int(merge_gap_ms / frame_ms))

    region = audio[region_start_i:region_end_i]
    rms: list[float] = []
    for i in range(0, len(region), frame):
        chunk = region[i : i + frame]
        if chunk.size:
            rms.append(float(np.sqrt(np.mean(chunk ** 2))))

    if not rms:
        return []

    peak = max(rms)
    if peak < 1e-6:
        return []

    threshold = peak * threshold_ratio
    voiced_flags: list[bool] = []
    silent_run = 0
    for val in rms:
        if val >= threshold:
            voiced_flags.append(True)
            silent_run = 0
        else:
            silent_run += 1
            voiced_flags.append(silent_run < hangover_frames)

    raw_islands: list[tuple[int, int]] = []
    in_island = False
    island_start = 0
    for idx, voiced in enumerate(voiced_flags):
        if voiced and not in_island:
            in_island = True
            island_start = idx
        elif not voiced and in_island:
            in_island = False
            raw_islands.append((island_start, idx))
    if in_island:
        raw_islands.append((island_start, len(voiced_flags)))

    if not raw_islands:
        return []

    merged: list[tuple[int, int]] = [raw_islands[0]]
    for start_f, end_f in raw_islands[1:]:
        prev_start, prev_end = merged[-1]
        if start_f - prev_end <= merge_gap_frames:
            merged[-1] = (prev_start, end_f)
        else:
            merged.append((start_f, end_f))

    islands: list[tuple[float, float]] = []
    for start_f, end_f in merged:
        s_i = region_start_i + start_f * frame
        e_i = min(region_end_i, region_start_i + end_f * frame)
        if e_i > s_i:
            islands.append((s_i / sr, e_i / sr))
    return islands


def _detect_speech_end(
    audio,
    sr: int,
    start_i: int,
    search_end_i: int,
    *,
    frame_ms: float = 30.0,
    hangover_ms: float = 200.0,
    threshold_ratio: float = 0.08,
) -> int:
    """从 start_i 起向后找语音结束点 (能量低于阈值持续 hangover)。"""
    import numpy as np

    if search_end_i <= start_i:
        return start_i

    frame = max(1, int(sr * frame_ms / 1000.0))
    hangover_frames = max(1, int(hangover_ms / frame_ms))
    region = audio[start_i:search_end_i]
    if region.size == 0:
        return start_i

    peaks: list[float] = []
    for i in range(0, len(region), frame):
        chunk = region[i : i + frame]
        if chunk.size:
            peaks.append(float(np.sqrt(np.mean(chunk ** 2))))

    if not peaks:
        return search_end_i

    peak = max(peaks)
    if peak < 1e-6:
        return search_end_i

    threshold = peak * threshold_ratio
    silent_run = 0
    last_voiced = 0
    for idx, rms in enumerate(peaks):
        if rms >= threshold:
            silent_run = 0
            last_voiced = idx
        else:
            silent_run += 1
            if silent_run >= hangover_frames:
                break

    end_frame = last_voiced + 1
    end_i = start_i + end_frame * frame
    return min(search_end_i, max(start_i + frame, end_i))


def _detect_onset(
    audio,
    sr: int,
    start_i: int,
    end_i: int,
    *,
    frame_ms: float = 10.0,
    threshold_ratio: float = 0.12,
) -> int:
    """在区间内找第一个显著能量上升点 (辅音/音节 onset)。"""
    import numpy as np

    if end_i <= start_i:
        return start_i

    frame = max(1, int(sr * frame_ms / 1000.0))
    region = audio[start_i:end_i]
    if region.size < frame * 2:
        return start_i

    rms = []
    for i in range(0, len(region) - frame, frame):
        chunk = region[i : i + frame]
        rms.append(float(np.sqrt(np.mean(chunk ** 2))))

    if not rms:
        return start_i

    peak = max(rms)
    if peak < 1e-6:
        return start_i

    threshold = peak * threshold_ratio
    for idx, val in enumerate(rms):
        if val >= threshold:
            return start_i + idx * frame
    return start_i


def _apply_min_duration(
    start_sec: float,
    end_sec: float,
    *,
    min_sec: float,
    max_end_sec: float,
) -> tuple[float, float]:
    dur = end_sec - start_sec
    if dur >= min_sec:
        return start_sec, min(end_sec, max_end_sec)
    extra = (min_sec - dur) / 2.0
    new_start = max(0.0, start_sec - extra)
    new_end = min(max_end_sec, end_sec + extra)
    if new_end - new_start < min_sec:
        new_end = min(max_end_sec, new_start + min_sec)
    return new_start, new_end


def _format_islands(islands: list[tuple[float, float]]) -> list[dict]:
    return [
        {"start": round(s, 3), "end": round(e, 3)}
        for s, e in islands
    ]


def align_segment_bounds(
    assignments: list[dict],
    audio,
    sr: int,
    *,
    enabled: bool = True,
    pad_start_sec: float = 0.12,
    pad_end_sec: float = 0.12,
    onset_lead_sec: float = 0.03,
    long_window_sec: float = 5.0,
    shout_min_ms: int = 600,
    shout_tail_pad_ms: int = 250,
    short_min_ms: int = 300,
    shout_all_islands: bool = False,
) -> list[dict]:
    """
    分类对齐 [start,end]:
    - shout: 首岛/全岛 + min 时长 + 尾音 pad
    - phrase_long_window: 全窗 phrase VAD (onset → 末岛/窗末)
    - short_dialogue: onset/VAD + min 300ms
    - none: 保留 SRT 窗
    """
    if not enabled or not assignments:
        return assignments

    aligned: list[dict] = []
    n = len(audio)
    shout_min_sec = shout_min_ms / 1000.0
    shout_tail_sec = shout_tail_pad_ms / 1000.0
    short_min_sec = short_min_ms / 1000.0

    for seg in assignments:
        entry = dict(seg)
        srt_start = float(seg.get("start", 0.0))
        srt_end = float(seg.get("end", srt_start))
        cue_type = classify_cue_align_type(seg, long_window_sec=long_window_sec)
        entry["align_cue_type"] = cue_type

        if cue_type == "none":
            entry["aligned_start"] = round(srt_start, 3)
            entry["aligned_end"] = round(srt_end, 3)
            aligned.append(entry)
            continue

        entry["srt_start"] = round(srt_start, 3)
        entry["srt_end"] = round(srt_end, 3)
        search_start_i = max(0, int((srt_start - pad_start_sec) * sr))
        search_end_i = min(n, int((srt_end + pad_end_sec) * sr))
        max_end_sec = srt_end + pad_end_sec

        islands = detect_speech_islands(
            audio,
            sr,
            srt_start,
            srt_end,
            pad_start_sec=pad_start_sec,
            pad_end_sec=pad_end_sec,
        )
        entry["speech_islands"] = _format_islands(islands)

        if cue_type == "shout":
            if not islands:
                onset_i = _detect_onset(
                    audio, sr, search_start_i, min(search_end_i, search_start_i + sr)
                )
                onset_i = max(0, onset_i - int(onset_lead_sec * sr))
                speech_end_i = _detect_speech_end(audio, sr, onset_i, search_end_i)
                aligned_start = onset_i / sr
                aligned_end = speech_end_i / sr
            elif shout_all_islands or (srt_end - srt_start) > long_window_sec:
                aligned_start = islands[0][0]
                aligned_end = islands[-1][1]
            else:
                aligned_start = islands[0][0]
                aligned_end = islands[0][1]

            aligned_end = min(max_end_sec, aligned_end + shout_tail_sec)
            aligned_start, aligned_end = _apply_min_duration(
                aligned_start,
                aligned_end,
                min_sec=shout_min_sec,
                max_end_sec=max_end_sec,
            )
        elif cue_type == "phrase_long_window":
            srt_dur = max(0.001, srt_end - srt_start)
            if islands:
                aligned_start = islands[0][0]
                aligned_end = islands[-1][1]
            else:
                onset_i = _detect_onset(audio, sr, search_start_i, search_end_i)
                onset_i = max(0, onset_i - int(onset_lead_sec * sr))
                speech_end_i = _detect_speech_end(
                    audio, sr, onset_i, search_end_i, hangover_ms=350.0
                )
                aligned_start = onset_i / sr
                aligned_end = speech_end_i / sr
            island_dur = aligned_end - aligned_start
            if island_dur < srt_dur * 0.4 or island_dur < 2.0:
                aligned_start = srt_start
                aligned_end = srt_end
            else:
                aligned_start = max(0.0, aligned_start - onset_lead_sec)
                aligned_end = min(max_end_sec, aligned_end)
        else:
            core_start_i = max(0, int(srt_start * sr))
            core_end_i = min(n, int(srt_end * sr))
            onset_i = _detect_onset(
                audio,
                sr,
                search_start_i,
                min(search_end_i, core_end_i + sr // 2),
            )
            onset_i = max(0, onset_i - int(onset_lead_sec * sr))
            speech_end_i = _detect_speech_end(
                audio,
                sr,
                max(onset_i, core_start_i),
                search_end_i,
            )
            if speech_end_i <= onset_i:
                speech_end_i = min(search_end_i, onset_i + int(0.8 * sr))
            aligned_start = onset_i / sr
            aligned_end = speech_end_i / sr
            aligned_start, aligned_end = _apply_min_duration(
                aligned_start,
                aligned_end,
                min_sec=short_min_sec,
                max_end_sec=max_end_sec,
            )

        entry["aligned_start"] = round(aligned_start, 3)
        entry["aligned_end"] = round(aligned_end, 3)
        entry["start"] = aligned_start
        entry["end"] = aligned_end
        aligned.append(entry)

    return aligned
