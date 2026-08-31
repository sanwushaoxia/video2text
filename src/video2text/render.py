from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from video2text.ffmpeg_util import (
    _require_ffmpeg,
    _require_ffmpeg_for_subtitles,
    _run_subprocess_text,
)


def render_video(
    video_path: Path,
    output_path: Path,
    audio_path: Path | None = None,
    srt_path: Path | None = None,
    font_size: int = 22,
    margin_v: int = 40,
    box: bool = False,
) -> None:
    """
    输出最终视频: 可选烧录字幕 + 可选替换音轨 (消除原配音)。
    audio_path 不为 None 时用 AI 配音替换原音轨。
    """
    ffmpeg = (
        _require_ffmpeg_for_subtitles()
        if srt_path is not None
        else _require_ffmpeg()
    )
    cmd = [ffmpeg, "-y", "-i", str(video_path.resolve())]

    if audio_path is not None:
        cmd.extend(["-i", str(audio_path.resolve())])

    burn_srt: Path | None = None
    ffmpeg_cwd: str | None = None
    if srt_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Windows: 避免 C: 盘符与中文路径导致 subtitles 滤镜解析失败
        burn_srt = output_path.parent / "__v2t_burn.srt"
        shutil.copy2(srt_path, burn_srt)
        ffmpeg_cwd = str(output_path.parent.resolve())
        if box:
            border = "BorderStyle=3,Outline=2,Shadow=0"
        else:
            border = "BorderStyle=1,Outline=2,Shadow=1"
        force_style = (
            f"Alignment=2,FontSize={font_size},PrimaryColour=&H00FFFFFF,"
            f"OutlineColour=&H00000000,BackColour=&H00000000,{border},"
            f"MarginV={margin_v}"
        )
        vf = f"subtitles={burn_srt.name}:force_style='{force_style}'"
        cmd.extend(["-vf", vf])

    cmd.extend(["-map", "0:v:0"])
    if audio_path is not None:
        cmd.extend(["-map", "1:a:0", "-c:a", "aac", "-b:a", "192k"])
    else:
        cmd.extend(["-c:a", "copy"])

    cmd.append(str(output_path.resolve()))
    try:
        if ffmpeg_cwd:
            proc = _run_subprocess_text(cmd, cwd=ffmpeg_cwd)
        else:
            proc = _run_subprocess_text(cmd)
    finally:
        if burn_srt is not None:
            burn_srt.unlink(missing_ok=True)

    if proc.returncode != 0:
        raise RuntimeError(
            "生成视频失败 (ffmpeg):\n{}".format((proc.stderr or "")[-2000:])
        )
