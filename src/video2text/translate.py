from __future__ import annotations

import sys
from pathlib import Path

from video2text.srt import segments_to_srt


_TRANSLATOR_LANG_MAP = {
    "zh": "zh",
    "zh-cn": "zh",
    "zh-tw": "zh-TW",
    "en": "en",
    "ja": "ja",
    "ko": "ko",
    "fr": "fr",
    "de": "de",
    "es": "es",
    "ru": "ru",
    "pt": "pt",
    "it": "it",
    "ar": "ar",
    "hi": "hi",
    "th": "th",
    "vi": "vi",
    "auto": "auto",
}

_TRANSLATOR_ENGINES = ("bing", "alibaba", "google", "baidu")

def _to_translator_lang(code: str | None) -> str:
    if not code:
        return "auto"
    normalized = code.strip().lower().replace("_", "-")
    return _TRANSLATOR_LANG_MAP.get(normalized, normalized)

def _lang_short_code(code: str) -> str:
    lang = _to_translator_lang(code)
    if lang.lower() == "zh-tw":
        return "zh-TW"
    return lang.split("-")[0].lower()

def _translate_text(
    text: str,
    source_lang: str,
    target_lang: str,
    engine: str,
) -> str:
    import translators as ts

    if engine == "auto":
        engines = _TRANSLATOR_ENGINES
    else:
        engines = (engine,)

    last_exc: Exception | None = None
    for name in engines:
        try:
            return ts.translate_text(
                text,
                translator=name,
                from_language=source_lang,
                to_language=target_lang,
            )
        except Exception as exc:
            last_exc = exc
            print(f"  {name} 翻译失败: {exc}", file=sys.stderr)
    raise RuntimeError(f"所有翻译引擎均失败: {last_exc}") from last_exc

def translated_srt_path(base_srt: Path, target_lang: str) -> Path:
    short = _lang_short_code(target_lang)
    return base_srt.parent / f"{base_srt.stem}_{short}.srt"

def translate_segments(
    segments: list[dict],
    source_lang: str | None,
    target_lang: str,
    engine: str = "auto",
) -> list[dict]:
    """将分段字幕翻译为目标语言, 保留时间轴。"""
    if not segments:
        return segments

    src = _to_translator_lang(source_lang)
    tgt = _to_translator_lang(target_lang)
    if src != "auto" and _lang_short_code(src) == _lang_short_code(tgt):
        print(f"源语言与目标语言均为 {tgt}, 跳过翻译")
        return segments

    out = [dict(seg) for seg in segments]
    total = sum(1 for seg in out if (seg.get("text") or "").strip())
    if total == 0:
        return out

    engine_label = engine if engine != "auto" else "bing/alibaba/google (自动)"
    print(f"翻译字幕: {src} -> {tgt} ({total} 段, 引擎={engine_label}, 需联网)...")

    done = 0
    for seg in out:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        done += 1
        print(f"  翻译 [{done}/{total}]: {text[:40]}{'...' if len(text) > 40 else ''}")
        seg["text"] = _translate_text(text, src, tgt, engine).strip()
    return out

def write_translated_srt(
    segments: list[dict],
    source_srt: Path,
    target_lang: str,
) -> Path:
    out_path = translated_srt_path(source_srt, target_lang)
    out_path.write_text(segments_to_srt(segments), encoding="utf-8")
    print(f"已写入译文字幕: {out_path}")
    return out_path
