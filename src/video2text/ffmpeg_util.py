from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path


# configure_ffmpeg() 写入; 勿 from ffmpeg_util import _FFPROBE_BIN (会得到 stale None)
_FFMPEG_BIN: str | None = None
_FFPROBE_BIN: str | None = None
_FFMPEG_SUBTITLES_FILTER: dict[str, bool] = {}
_FFMPEG_FOR_SUBTITLES: str | None = None


def ffprobe_bin() -> str | None:
    """当前已配置的 ffprobe 路径 (configure_ffmpeg 之后有效)。"""
    return _FFPROBE_BIN


def _run_subprocess_text(
    cmd: list[str],
    *,
    check: bool = False,
    cwd: str | None = None,
) -> subprocess.CompletedProcess:
    """运行子进程并以 UTF-8 解码输出 (避免 Windows 默认 GBK 解码 ffmpeg 失败)。"""
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=check,
        cwd=cwd,
    )

def _ffmpeg_install_hint() -> str:
    system = platform.system()
    if system == "Windows":
        return (
            "未找到 ffmpeg。Windows 安装方式 (任选其一):\n"
            "  1. pip install imageio-ffmpeg  (推荐, 脚本会自动使用内置 ffmpeg)\n"
            "  2. 从 https://www.gyan.dev/ffmpeg/builds/ 下载, 解压后将 bin 加入系统 PATH\n"
            "  3. conda install -c conda-forge ffmpeg\n"
            "  4. winget install Gyan.FFmpeg\n"
            "  也可用 --ffmpeg 指定 ffmpeg.exe 的完整路径"
        )
    if system == "Darwin":
        return (
            "未找到 ffmpeg。macOS 安装方式:\n"
            "  brew install ffmpeg\n"
            "  或 pip install imageio-ffmpeg"
        )
    return (
        "未找到 ffmpeg。Linux 安装方式:\n"
        "  sudo apt install ffmpeg\n"
        "  或 conda install -c conda-forge ffmpeg\n"
        "  或 pip install imageio-ffmpeg"
    )

def _add_executable(candidates: list[Path], path: str | Path | None) -> None:
    if not path:
        return
    resolved = Path(path).expanduser()
    if resolved.is_file() and resolved not in candidates:
        candidates.append(resolved)

def _find_ffmpeg_executable(ffmpeg_override: str | None = None) -> Path | None:
    candidates: list[Path] = []
    _add_executable(candidates, ffmpeg_override)
    _add_executable(candidates, os.environ.get("FFMPEG_PATH"))
    _add_executable(candidates, os.environ.get("VIDEO2TEXT_FFMPEG"))
    _add_executable(candidates, shutil.which("ffmpeg"))

    if platform.system() == "Windows":
        for rel in (
            r"C:\ffmpeg\bin\ffmpeg.exe",
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WinGet", "Links", "ffmpeg.exe"),
            os.path.join(os.environ.get("ProgramFiles", ""), "ffmpeg", "bin", "ffmpeg.exe"),
            os.path.join(os.environ.get("ProgramFiles(x86)", ""), "ffmpeg", "bin", "ffmpeg.exe"),
            os.path.join(os.environ.get("USERPROFILE", ""), "scoop", "shims", "ffmpeg.exe"),
        ):
            _add_executable(candidates, rel)

    try:
        import imageio_ffmpeg

        _add_executable(candidates, imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        pass

    return candidates[0] if candidates else None

def _find_ffprobe_executable(
    ffmpeg_path: Path,
    ffprobe_override: str | None = None,
) -> Path | None:
    candidates: list[Path] = []
    _add_executable(candidates, ffprobe_override)
    _add_executable(candidates, os.environ.get("FFPROBE_PATH"))
    _add_executable(candidates, os.environ.get("VIDEO2TEXT_FFPROBE"))
    _add_executable(candidates, shutil.which("ffprobe"))

    sibling_name = "ffprobe.exe" if platform.system() == "Windows" else "ffprobe"
    _add_executable(candidates, ffmpeg_path.parent / sibling_name)
    return candidates[0] if candidates else None

def configure_ffmpeg(ffmpeg: str | None = None, ffprobe: str | None = None) -> None:
    """解析 ffmpeg / ffprobe 路径, 兼容 Windows / Linux / macOS。"""
    global _FFMPEG_BIN, _FFPROBE_BIN, _FFMPEG_FOR_SUBTITLES

    found = _find_ffmpeg_executable(ffmpeg)
    if found is None:
        raise RuntimeError(_ffmpeg_install_hint())

    _FFMPEG_BIN = str(found.resolve())
    _FFMPEG_FOR_SUBTITLES = None
    ffprobe_path = _find_ffprobe_executable(found, ffprobe)
    _FFPROBE_BIN = str(ffprobe_path.resolve()) if ffprobe_path else None

    print(f"使用 ffmpeg: {_FFMPEG_BIN}")
    if _FFPROBE_BIN:
        print(f"使用 ffprobe: {_FFPROBE_BIN}")
    else:
        print("未找到 ffprobe, 将使用 ffmpeg 探测媒体时长")

def _require_ffmpeg() -> str:
    if _FFMPEG_BIN is None:
        configure_ffmpeg()
    return _FFMPEG_BIN

def _imageio_ffmpeg_executable() -> Path | None:
    try:
        import imageio_ffmpeg

        path = Path(imageio_ffmpeg.get_ffmpeg_exe())
        if path.is_file():
            return path
    except Exception:
        pass
    return None

def _ffmpeg_has_subtitles_filter(ffmpeg_path: str) -> bool:
    cached = _FFMPEG_SUBTITLES_FILTER.get(ffmpeg_path)
    if cached is not None:
        return cached
    proc = _run_subprocess_text([ffmpeg_path, "-filters"])
    has_filter = proc.returncode == 0 and " subtitles " in proc.stdout
    _FFMPEG_SUBTITLES_FILTER[ffmpeg_path] = has_filter
    return has_filter

def _require_ffmpeg_for_subtitles() -> str:
    """烧录字幕需要 libass 的 subtitles 滤镜; conda 默认 ffmpeg 常不带此滤镜。"""
    global _FFMPEG_FOR_SUBTITLES
    if _FFMPEG_FOR_SUBTITLES is not None:
        return _FFMPEG_FOR_SUBTITLES

    primary = _require_ffmpeg()
    if _ffmpeg_has_subtitles_filter(primary):
        _FFMPEG_FOR_SUBTITLES = primary
        return primary

    fallback = _imageio_ffmpeg_executable()
    if fallback is not None:
        fallback_str = str(fallback.resolve())
        if _ffmpeg_has_subtitles_filter(fallback_str):
            print(
                "当前 ffmpeg 不支持字幕烧录 (无 subtitles/libass 滤镜), "
                f"改用: {fallback_str}"
            )
            _FFMPEG_FOR_SUBTITLES = fallback_str
            return fallback_str

    raise RuntimeError(
        "当前 ffmpeg 不支持 subtitles 滤镜 (需 libass)。\n"
        "请 pip install imageio-ffmpeg, 或使用带 libass 的完整版 ffmpeg "
        "(如 conda-forge: conda install -c conda-forge ffmpeg, 或 gyan.dev 构建)。"
    )

def _require_ffprobe() -> str:
    if _FFMPEG_BIN is None:
        configure_ffmpeg()
    if _FFPROBE_BIN:
        return _FFPROBE_BIN
    raise RuntimeError(
        "未找到 ffprobe, 请安装完整版 ffmpeg, 或使用 --ffprobe 指定路径"
    )

def _load_whisper_audio(path: Path | str, sr: int = 16000):
    """与 whisper.load_audio 相同, 但使用本脚本已配置的 ffmpeg (无需 PATH 中有 ffmpeg)。"""
    import numpy as np

    ffmpeg = _require_ffmpeg()
    cmd = [
        ffmpeg,
        "-nostdin",
        "-threads",
        "0",
        "-i",
        str(path),
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sr),
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            "Failed to load audio:\n{}".format(
                proc.stderr.decode(errors="replace")[-2000:]
            )
        )
    return np.frombuffer(proc.stdout, np.int16).flatten().astype(np.float32) / 32768.0

def _duration_via_ffmpeg(path: Path) -> float:
    ffmpeg = _require_ffmpeg()
    cmd = [ffmpeg, "-hide_banner", "-i", str(path)]
    proc = _run_subprocess_text(cmd)
    stderr = proc.stderr or ""
    match = re.search(
        r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)",
        stderr,
    )
    if not match:
        raise RuntimeError(f"无法获取媒体时长: {path}")
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)

def extract_audio(video_path: Path, wav_path: Path, sample_rate: int = 16000) -> None:
    """从视频抽出单声道 16kHz wav, Whisper 推荐输入。"""
    ffmpeg = _require_ffmpeg()
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        str(wav_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

def extract_stereo_audio(
    video_path: Path,
    wav_path: Path,
    sample_rate: int = 44100,
) -> None:
    """从视频抽出立体声 wav, 用于人声/伴奏分离。"""
    ffmpeg = _require_ffmpeg()
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "2",
        str(wav_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
