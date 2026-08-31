#!/usr/bin/env python3
"""One-time splitter: transcribe_video.py -> src/video2text/*.py"""
from __future__ import annotations

import ast
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_FILE = ROOT / "transcribe_video.py"
OUT_PKG = ROOT / "src" / "video2text"

# function/class/name -> module (without package prefix)
ASSIGNMENTS: dict[str, str] = {}

MODULE_NAMES = [
    "ffmpeg_util",
    "srt",
    "translate",
    "ocr",
    "media",
    "whisper_transcribe",
    "separate",
    "dub",
    "mix",
    "render",
    "pipeline",
]

FUNCTION_MODULES: dict[str, str] = {
    "_run_subprocess_text": "ffmpeg_util",
    "_ffmpeg_install_hint": "ffmpeg_util",
    "_add_executable": "ffmpeg_util",
    "_find_ffmpeg_executable": "ffmpeg_util",
    "_find_ffprobe_executable": "ffmpeg_util",
    "configure_ffmpeg": "ffmpeg_util",
    "_require_ffmpeg": "ffmpeg_util",
    "_imageio_ffmpeg_executable": "ffmpeg_util",
    "_ffmpeg_has_subtitles_filter": "ffmpeg_util",
    "_require_ffmpeg_for_subtitles": "ffmpeg_util",
    "_require_ffprobe": "ffmpeg_util",
    "_load_whisper_audio": "ffmpeg_util",
    "_duration_via_ffmpeg": "ffmpeg_util",
    "extract_audio": "ffmpeg_util",
    "extract_stereo_audio": "ffmpeg_util",
    "sec_to_srt_time": "srt",
    "_srt_time_parts_to_sec": "srt",
    "parse_srt": "srt",
    "segments_to_srt": "srt",
    "resolve_dub_srt_path": "srt",
    "load_dub_segments": "srt",
    "_to_translator_lang": "translate",
    "_lang_short_code": "translate",
    "_translate_text": "translate",
    "translated_srt_path": "translate",
    "translate_segments": "translate",
    "write_translated_srt": "translate",
    "_normalize_ocr_text": "ocr",
    "_is_meaningful_subtitle": "srt",
    "_ocr_lang_list": "ocr",
    "_nvidia_pip_dll_bin_dirs": "ocr",
    "_prepare_nvidia_ocr_dll_paths": "ocr",
    "_ocr_engine_using_gpu": "ocr",
    "_cuda_ocr_working": "ocr",
    "_resolve_ocr_use_gpu": "ocr",
    "_build_ocr_engine": "ocr",
    "_load_frame_gray_small": "ocr",
    "_frame_changed": "ocr",
    "_parse_ocr_result": "ocr",
    "_ocr_worker_init": "ocr",
    "_ocr_single_frame": "ocr",
    "_resolve_ocr_workers": "ocr",
    "_select_ocr_frames": "ocr",
    "_run_ocr_on_frames": "ocr",
    "_merge_ocr_jitter_segments": "ocr",
    "extract_burned_in_subtitles": "ocr",
    "_duration_via_wave": "media",
    "get_media_duration": "media",
    "_normalize_audio_to_wav": "media",
    "_create_silence_wav": "media",
    "_trim_audio_silence": "media",
    "_atempo_filter_chain": "media",
    "_normalize_audio_peak": "media",
    "_fit_audio_to_max_duration": "media",
    "_trim_audio_to_duration": "media",
    "_concat_wav_files": "media",
    "_extract_audio_clip": "media",
    "_whisper_model_cache_path": "whisper_transcribe",
    "load_whisper_model": "whisper_transcribe",
    "detect_and_transcribe": "whisper_transcribe",
    "_separate_vocal_stems": "separate",
    "_find_reference_window": "separate",
    "extract_voice_reference": "separate",
    "_load_ref_text_segments": "separate",
    "prepare_vocal_assets": "separate",
    "VocalAssets": "separate",
    "_resolve_dub_voice": "dub",
    "load_dub_speaker_map": "dub",
    "_looks_like_edge_voice": "dub",
    "_resolve_segment_edge_voice": "dub",
    "_tts_to_file": "dub",
    "DubEngineConfig": "dub",
    "_sovits_lang": "dub",
    "_detect_gpt_sovits_api_version": "dub",
    "_prepare_sovits_ref_audio": "dub",
    "_rvc_convert_http": "dub",
    "_rvc_convert_inprocess": "dub",
    "_rvc_convert": "dub",
    "_http_download": "dub",
    "_gpt_sovits_save_response": "dub",
    "_gpt_sovits_tts": "dub",
    "synthesize_segment": "dub",
    "build_dub_engine_config": "dub",
    "resolve_dub_language": "dub",
    "generate_dub_audio": "dub",
    "finalize_dub_audio": "dub",
    "_mix_bgm_and_dub": "mix",
    "render_video": "render",
    "build_argparser": "pipeline",
    "_validate_dub_engine_args": "pipeline",
    "run_dub_workflow": "pipeline",
    "main": "pipeline",
}

CONST_MODULES: dict[str, str] = {
    "_FFMPEG_BIN": "ffmpeg_util",
    "_FFPROBE_BIN": "ffmpeg_util",
    "_FFMPEG_SUBTITLES_FILTER": "ffmpeg_util",
    "_FFMPEG_FOR_SUBTITLES": "ffmpeg_util",
    "_SRT_TIME_ARROW_RE": "srt",
    "_TRANSLATOR_LANG_MAP": "translate",
    "_TRANSLATOR_ENGINES": "translate",
    "_OCR_LANG_MAP": "ocr",
    "_CUDA_OCR_WORKING": "ocr",
    "_CUDA_WARNED": "ocr",
    "_NVIDIA_OCR_DLL_PREPARED": "ocr",
    "_OCR_WORKER": "ocr",
    "_DUB_LOCALE_MAP": "dub",
    "_DUB_DEFAULT_VOICES": "dub",
    "_DUB_GENDER_DEFAULT_VOICES": "dub",
    "_SPEAKER_GENDER_ALIASES": "dub",
    "_SOVITS_API_VERSION": "dub",
    "_SOVITS_LANG_MAP": "dub",
    "_DUB_MAX_SPEEDUP": "media",
}

MODULE_IMPORTS: dict[str, str] = {
    "ffmpeg_util": textwrap.dedent(
        """
        from __future__ import annotations

        import platform
        import re
        import shutil
        import subprocess
        import sys
        from pathlib import Path
        """
    ).strip(),
    "srt": textwrap.dedent(
        """
        from __future__ import annotations

        import re
        from pathlib import Path
        """
    ).strip(),
    "translate": textwrap.dedent(
        """
        from __future__ import annotations

        import sys
        from pathlib import Path

        from video2text.srt import segments_to_srt
        """
    ).strip(),
    "ocr": textwrap.dedent(
        """
        from __future__ import annotations

        import os
        import sys
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from pathlib import Path

        from video2text.ffmpeg_util import _require_ffmpeg
        from video2text.srt import _is_meaningful_subtitle
        """
    ).strip(),
    "media": textwrap.dedent(
        """
        from __future__ import annotations

        import re
        import shutil
        import subprocess
        import wave
        from pathlib import Path

        from video2text.ffmpeg_util import _require_ffmpeg, _run_subprocess_text
        """
    ).strip(),
    "whisper_transcribe": textwrap.dedent(
        """
        from __future__ import annotations

        import json
        import shutil
        from pathlib import Path

        from video2text.ffmpeg_util import _load_whisper_audio
        """
    ).strip(),
    "separate": textwrap.dedent(
        """
        from __future__ import annotations

        import shutil
        import subprocess
        import sys
        from dataclasses import dataclass
        from pathlib import Path

        from video2text.ffmpeg_util import extract_stereo_audio
        from video2text.media import _extract_audio_clip, get_media_duration
        from video2text.srt import _is_meaningful_subtitle, parse_srt
        """
    ).strip(),
    "dub": textwrap.dedent(
        """
        from __future__ import annotations

        import asyncio
        import json
        import re
        import shutil
        import subprocess
        import sys
        import urllib.error
        import urllib.parse
        import urllib.request
        from dataclasses import dataclass
        from pathlib import Path

        from video2text.ffmpeg_util import _run_subprocess_text
        from video2text.media import (
            _concat_wav_files,
            _create_silence_wav,
            _extract_audio_clip,
            _fit_audio_to_max_duration,
            _normalize_audio_peak,
            _normalize_audio_to_wav,
            _trim_audio_silence,
            get_media_duration,
        )
        from video2text.mix import _mix_bgm_and_dub
        from video2text.separate import _separate_vocal_stems
        from video2text.srt import _is_meaningful_subtitle
        """
    ).strip(),
    "mix": textwrap.dedent(
        """
        from __future__ import annotations

        import subprocess
        from pathlib import Path

        from video2text.ffmpeg_util import _require_ffmpeg, extract_stereo_audio
        from video2text.separate import _separate_vocal_stems
        """
    ).strip(),
    "render": textwrap.dedent(
        """
        from __future__ import annotations

        import shutil
        import subprocess
        from pathlib import Path

        from video2text.ffmpeg_util import _require_ffmpeg, _require_ffmpeg_for_subtitles
        from video2text.srt import sec_to_srt_time
        """
    ).strip(),
    "pipeline": textwrap.dedent(
        """
        from __future__ import annotations

        import argparse
        import json
        import sys
        import tempfile
        from pathlib import Path

        from video2text.dub import (
            build_dub_engine_config,
            finalize_dub_audio,
            generate_dub_audio,
            resolve_dub_language,
        )
        from video2text.ffmpeg_util import configure_ffmpeg, extract_audio
        from video2text.media import get_media_duration
        from video2text.ocr import extract_burned_in_subtitles
        from video2text.render import render_video
        from video2text.separate import VocalAssets, prepare_vocal_assets
        from video2text.srt import (
            load_dub_segments,
            parse_srt,
            resolve_dub_srt_path,
            segments_to_srt,
        )
        from video2text.translate import (
            _lang_short_code,
            _to_translator_lang,
            translate_segments,
            write_translated_srt,
        )
        from video2text.whisper_transcribe import detect_and_transcribe, load_whisper_model
        """
    ).strip(),
}


def main() -> None:
    source = SRC_FILE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)

    modules: dict[str, list[str]] = {m: [] for m in MODULE_NAMES}
    module_set = set(MODULE_NAMES)

    # module doc/header from original (without path bootstrap)
    header_end = source.find("_FFMPEG_BIN")
    if header_end == -1:
        raise SystemExit("header not found")

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    mod = CONST_MODULES.get(target.id)
                    if mod:
                        modules[mod].append("".join(lines[node.lineno - 1 : node.end_lineno]))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            mod = CONST_MODULES.get(node.target.id)
            if mod:
                modules[mod].append("".join(lines[node.lineno - 1 : node.end_lineno]))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            mod = FUNCTION_MODULES.get(node.name)
            if mod:
                modules[mod].append("".join(lines[node.lineno - 1 : node.end_lineno]))
        elif isinstance(node, ast.ClassDef):
            mod = FUNCTION_MODULES.get(node.name)
            if mod:
                modules[mod].append("".join(lines[node.lineno - 1 : node.end_lineno]))

    OUT_PKG.mkdir(parents=True, exist_ok=True)
    (OUT_PKG / "__init__.py").write_text(
        '"""video2text: video subtitle extraction, burn-in, dubbing, and audio tools."""\n',
        encoding="utf-8",
    )

    for mod_name, chunks in modules.items():
        if not chunks:
            continue
        header = MODULE_IMPORTS.get(mod_name, "from __future__ import annotations\n")
        body = "\n\n".join(c.rstrip() for c in chunks)
        path = OUT_PKG / f"{mod_name}.py"
        path.write_text(header + "\n\n\n" + body + "\n", encoding="utf-8")
        print(f"wrote {path} ({len(chunks)} items)")

    print("done - manual fixes required for pipeline imports and ocr _is_meaningful_subtitle export")


if __name__ == "__main__":
    main()
