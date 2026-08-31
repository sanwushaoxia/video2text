from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

from video2text.ffmpeg_util import (
    _duration_via_ffmpeg,
    _require_ffmpeg,
    _run_subprocess_text,
    ffprobe_bin,
)


def _duration_via_wave(path: Path) -> float | None:
    if path.suffix.lower() != ".wav":
        return None
    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            if rate <= 0:
                return None
            return handle.getnframes() / float(rate)
    except wave.Error:
        return None

def get_media_duration(path: Path) -> float:
    """获取音视频时长 (秒)。"""
    probe = ffprobe_bin()
    if probe:
        cmd = [
            probe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
        proc = _run_subprocess_text(cmd, check=False)
        raw = (proc.stdout or "").strip()
        if proc.returncode == 0 and raw and raw.upper() != "N/A":
            try:
                duration = float(raw)
                if duration > 0:
                    return duration
            except ValueError:
                pass
    try:
        duration = _duration_via_ffmpeg(path)
        if duration > 0:
            return duration
    except RuntimeError:
        pass
    duration = _duration_via_wave(path)
    if duration is not None and duration > 0:
        return duration
    raise RuntimeError(f"无法获取媒体时长: {path}")

def _normalize_audio_to_wav(src: Path, dst: Path, sample_rate: int = 44100) -> None:
    ffmpeg = _require_ffmpeg()
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(src),
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        str(dst),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

def _create_silence_wav(duration: float, dst: Path, sample_rate: int = 44100) -> None:
    if duration <= 0.01:
        return
    ffmpeg = _require_ffmpeg()
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"anullsrc=r={sample_rate}:cl=mono",
        "-t",
        f"{duration:.3f}",
        "-acodec",
        "pcm_s16le",
        str(dst),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

def _trim_audio_silence(
    src: Path,
    dst: Path,
    *,
    sample_rate: int = 44100,
    threshold: str = "-50dB",
    min_silence: float = 0.06,
) -> None:
    """裁掉首尾静音 (GPT-SoVITS 等引擎常在句尾留长段空白)。"""
    ffmpeg = _require_ffmpeg()
    af = (
        f"silenceremove=start_periods=1:start_duration={min_silence}:"
        f"start_threshold={threshold}:detection=peak,"
        "areverse,"
        f"silenceremove=start_periods=1:start_duration={min_silence}:"
        f"start_threshold={threshold}:detection=peak,"
        "areverse"
    )
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(src),
        "-af",
        af,
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        str(dst),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        if get_media_duration(dst) <= 0.01:
            if src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
    except RuntimeError:
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)

def _atempo_filter_chain(speed_ratio: float) -> str:
    """生成 atempo 滤镜链; speed_ratio>1 加速缩短时长。"""
    filters: list[str] = []
    ratio = speed_ratio
    while ratio > 2.0:
        filters.append("atempo=2.0")
        ratio /= 2.0
    while ratio < 0.5:
        filters.append("atempo=0.5")
        ratio /= 0.5
    filters.append(f"atempo={ratio:.6f}")
    return ",".join(filters)

_DUB_MAX_SPEEDUP = 1.12

def _normalize_audio_peak(
    src: Path,
    dst: Path,
    *,
    target_peak_db: float = -3.0,
    sample_rate: int = 44100,
) -> None:
    """把单段配音峰值归一化 (GPT-SoVITS 原始输出往往极轻)。"""
    ffmpeg = _require_ffmpeg()
    probe = subprocess.run(
        [ffmpeg, "-i", str(src), "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    match = re.search(r"max_volume: ([-\d.]+) dB", probe.stderr or "")
    if not match:
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        return
    peak_db = float(match.group(1))
    if peak_db <= -55.0:
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        return
    gain_db = target_peak_db - peak_db
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(src),
        "-af",
        f"volume={gain_db:.2f}dB",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        str(dst),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

def _fit_audio_to_max_duration(
    src: Path,
    dst: Path,
    max_duration: float,
    *,
    sample_rate: int = 44100,
    max_speedup: float = _DUB_MAX_SPEEDUP,
) -> None:
    """对齐字幕时长: 轻微加速或截断尾部, 避免过快导致听不清。"""
    dur = get_media_duration(src)
    if dur <= max_duration + 0.03:
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        return
    ffmpeg = _require_ffmpeg()
    ratio = dur / max_duration
    if ratio <= max_speedup:
        af = _atempo_filter_chain(ratio)
    else:
        fade_start = max(0.0, max_duration - 0.06)
        af = f"afade=t=out:st={fade_start:.3f}:d=0.06"
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(src),
        "-t",
        f"{max_duration:.3f}",
        "-af",
        af,
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        str(dst),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

def _trim_audio_to_duration(
    src: Path,
    dst: Path,
    duration: float,
    *,
    sample_rate: int = 44100,
) -> None:
    """截断音轨到指定秒数 (对齐视频时长)。"""
    ffmpeg = _require_ffmpeg()
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(src),
        "-t",
        f"{duration:.3f}",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        str(dst),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

def _concat_wav_files(parts: list[Path], dst: Path) -> None:
    ffmpeg = _require_ffmpeg()
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        list_path = Path(f.name)
        for p in parts:
            escaped = str(p.resolve()).replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")
    try:
        cmd = [
            ffmpeg,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-acodec",
            "pcm_s16le",
            str(dst),
        ]
        proc = _run_subprocess_text(cmd)
        if proc.returncode != 0:
            raise RuntimeError(
                "拼接配音音轨失败 (ffmpeg):\n{}".format((proc.stderr or "")[-2000:])
            )
    finally:
        list_path.unlink(missing_ok=True)

def _extract_audio_clip(
    src: Path,
    dst: Path,
    start: float,
    end: float,
    sample_rate: int = 44100,
) -> None:
    ffmpeg = _require_ffmpeg()
    duration = max(0.1, end - start)
    cmd = [
        ffmpeg,
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(src),
        "-t",
        f"{duration:.3f}",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        str(dst),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
