from __future__ import annotations

import re
from pathlib import Path


def sec_to_srt_time(t: float) -> str:
    """秒 -> SRT 时间码 00:00:00,000"""
    if t < 0:
        t = 0.0
    ms = int(round(t * 1000.0))
    h, rem = divmod(ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, milli = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{milli:03d}"

_SRT_TIME_ARROW_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})"
)

def _srt_time_parts_to_sec(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

def parse_srt(body: str) -> list[dict]:
    """解析 SRT 为与 Whisper segments 兼容的分段列表。"""
    segments: list[dict] = []
    for block in re.split(r"\n\s*\n", body.strip()):
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        time_line_idx = 1 if lines[0].isdigit() and len(lines) > 1 else 0
        cue_index: int | None = int(lines[0]) if lines[0].isdigit() else None
        match = _SRT_TIME_ARROW_RE.match(lines[time_line_idx])
        if not match:
            continue
        h1, m1, s1, ms1, h2, m2, s2, ms2 = match.groups()
        text = " ".join(lines[time_line_idx + 1 :]).strip()
        if not text:
            continue
        seg: dict = {
            "start": _srt_time_parts_to_sec(h1, m1, s1, ms1),
            "end": _srt_time_parts_to_sec(h2, m2, s2, ms2),
            "text": text,
        }
        if cue_index is not None:
            seg["index"] = cue_index
        segments.append(seg)
    return segments

def segments_to_srt(segments) -> str:
    lines = []
    for i, seg in enumerate(segments, start=1):
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        lines.append(str(i))
        lines.append(f"{sec_to_srt_time(start)} --> {sec_to_srt_time(end)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)

def resolve_dub_srt_path(
    dub_srt: str | None,
    srt: str | None,
    out_dir: Path | None,
    video_stem: str,
    dub_lang: str | None,
) -> Path | None:
    """解析用于 AI 配音的字幕文件路径 (--dub-srt 优先)。"""
    if dub_srt:
        return Path(dub_srt).expanduser().resolve()
    if srt:
        return Path(srt).expanduser().resolve()
    if dub_lang and out_dir is not None:
        candidate = out_dir / f"{video_stem}_{_lang_short_code(dub_lang)}.srt"
        if candidate.is_file():
            return candidate.resolve()
    return None

def load_dub_segments(
    dub_srt_path: Path | None,
    fallback_segments: list[dict],
) -> tuple[list[dict], Path | None]:
    """
    加载配音分段: 若指定了配音字幕文件则直接解析使用, 否则用 fallback (如 Whisper 结果)。
    """
    if dub_srt_path is None:
        return fallback_segments, None
    if not dub_srt_path.is_file():
        raise FileNotFoundError(f"找不到配音字幕: {dub_srt_path}")
    segments = parse_srt(dub_srt_path.read_text(encoding="utf-8"))
    if not segments:
        raise RuntimeError(f"配音字幕无有效分段: {dub_srt_path}")
    print(f"配音使用字幕: {dub_srt_path.name} (直接朗读, 不经过转写/翻译)")
    return segments, dub_srt_path

def _is_meaningful_subtitle(text: str, min_cjk: int = 2) -> bool:
    """过滤 OCR 噪声 (单字、纯符号等)。"""
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    if cjk >= min_cjk:
        return True
    core = re.sub(r"[\s\d\W_]+", "", text)
    return len(core) >= min_cjk
