from __future__ import annotations

import subprocess
from pathlib import Path

from video2text.ffmpeg_util import _require_ffmpeg, extract_stereo_audio
from video2text.separate import _separate_vocal_stems


def _mix_bgm_and_dub(
    bgm_path: Path,
    dub_path: Path,
    out_path: Path,
    bgm_volume: float = 1.0,
    dub_volume: float = 1.0,
) -> None:
    """将伴奏与 AI 配音混合为最终音轨。"""
    ffmpeg = _require_ffmpeg()
    # dub 需 asplit 两路: sidechain 侧链 + amix 直混; 同一 pad 不可复用 (conda ffmpeg 6.x)。
    # acompressor 不宜放在 filter_complex 链中, 改用 amix + sidechaincompress + alimiter。
    filter_graph = (
        f"[1:a]volume={dub_volume},asplit=2[dsc][dm];"
        f"[0:a]volume={bgm_volume}[bgm];"
        "[bgm][dsc]sidechaincompress=threshold=0.02:ratio=7:attack=25:release=450"
        ":level_sc=1[bgmduck];"
        "[bgmduck][dm]amix=inputs=2:duration=first:dropout_transition=0,"
        "alimiter=limit=0.98"
    )
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(bgm_path),
        "-i",
        str(dub_path),
        "-filter_complex",
        filter_graph,
        "-ac",
        "2",
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        stderr = (proc.stderr or b"").decode("utf-8", errors="replace")
        raise RuntimeError(f"混合 BGM 与配音失败 (ffmpeg):\n{stderr[-2000:]}")
