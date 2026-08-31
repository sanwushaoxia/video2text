from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from video2text.ffmpeg_util import _require_ffmpeg
from video2text.media import get_media_duration
from video2text.srt import _is_meaningful_subtitle


_OCR_LANG_MAP = {
    "zh": ["ch"],
    "ja": ["japan"],
    "en": ["en"],
    "ko": ["korean"],
}

def _normalize_ocr_text(text: str) -> str:
    text = re.sub(r"\s+", "", text)
    text = text.replace(" ", "")
    return text.strip()

def _ocr_lang_list(lang: str) -> list[str]:
    short = lang.split("-")[0].lower()
    return _OCR_LANG_MAP.get(short, [short])

_CUDA_OCR_WORKING: bool | None = None
_CUDA_WARNED = False

_NVIDIA_OCR_DLL_PREPARED = False

_OCR_WORKER: dict = {}

def _nvidia_pip_dll_bin_dirs() -> list[Path]:
    """查找 pip 安装的 nvidia-cublas / nvidia-cudnn 的 bin 目录。"""
    dirs: list[Path] = []
    try:
        import site

        search_roots = {Path(p) for p in site.getsitepackages()}
        try:
            search_roots.add(Path(site.getusersitepackages()))
        except Exception:
            pass
        for root in search_roots:
            nvidia_root = root / "nvidia"
            for sub in ("cublas", "cudnn"):
                bin_dir = nvidia_root / sub / "bin"
                if bin_dir.is_dir():
                    dirs.append(bin_dir.resolve())
    except Exception:
        pass
    seen: set[str] = set()
    unique: list[Path] = []
    for d in dirs:
        key = str(d).lower()
        if key not in seen:
            seen.add(key)
            unique.append(d)
    return unique

def _prepare_nvidia_ocr_dll_paths() -> list[Path]:
    """将 pip 版 NVIDIA CUDA/cuDNN DLL 目录加入当前进程的加载路径 (Windows 免手动 export PATH)。"""
    global _NVIDIA_OCR_DLL_PREPARED
    bin_dirs = _nvidia_pip_dll_bin_dirs()
    if _NVIDIA_OCR_DLL_PREPARED:
        return bin_dirs
    _NVIDIA_OCR_DLL_PREPARED = True
    if not bin_dirs:
        return bin_dirs

    if platform.system() == "Windows" and hasattr(os, "add_dll_directory"):
        for bin_dir in bin_dirs:
            try:
                os.add_dll_directory(str(bin_dir))
            except OSError:
                pass

    prepend = os.pathsep.join(str(d) for d in bin_dirs)
    path_val = os.environ.get("PATH", "")
    if prepend and prepend not in path_val:
        os.environ["PATH"] = f"{prepend}{os.pathsep}{path_val}" if path_val else prepend

    try:
        import onnxruntime as ort

        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls(cuda=True, cudnn=True)
    except Exception:
        pass

    return bin_dirs

def _ocr_engine_using_gpu(engine) -> bool:
    try:
        providers = engine.text_det.infer.session.get_providers()
        return bool(providers) and providers[0] == "CUDAExecutionProvider"
    except Exception:
        return False

def _cuda_ocr_working() -> bool:
    global _CUDA_OCR_WORKING
    if _CUDA_OCR_WORKING is not None:
        return _CUDA_OCR_WORKING
    _prepare_nvidia_ocr_dll_paths()
    prev_log_level = os.environ.get("ORT_LOG_SEVERITY_LEVEL")
    os.environ["ORT_LOG_SEVERITY_LEVEL"] = "4"
    try:
        import onnxruntime as ort

        if "CUDAExecutionProvider" not in ort.get_available_providers():
            _CUDA_OCR_WORKING = False
            return False
        from rapidocr_onnxruntime import RapidOCR

        probe = RapidOCR(det_use_cuda=True, rec_use_cuda=True, no_cls=True, no_det=True)
        _CUDA_OCR_WORKING = _ocr_engine_using_gpu(probe)
    except Exception:
        _CUDA_OCR_WORKING = False
    finally:
        if prev_log_level is None:
            os.environ.pop("ORT_LOG_SEVERITY_LEVEL", None)
        else:
            os.environ["ORT_LOG_SEVERITY_LEVEL"] = prev_log_level
    return _CUDA_OCR_WORKING

def _resolve_ocr_use_gpu(use_gpu: bool) -> bool:
    """只探测一次 CUDA OCR; 不可用则静默回退 CPU, 避免多进程重复报错。"""
    global _CUDA_WARNED
    if not use_gpu:
        return False
    if _cuda_ocr_working():
        return True
    if not _CUDA_WARNED:
        _CUDA_WARNED = True
        print(
            "CUDA OCR 不可用 (onnxruntime-gpu 与当前 CUDA/cuDNN 版本不匹配), 已回退 CPU。\n"
            "  快速解决: 命令加 --no-ocr-use-gpu\n"
            "  或安装: pip install onnxruntime-gpu==1.20.2 nvidia-cublas-cu12 nvidia-cudnn-cu12",
            file=sys.stderr,
        )
    return False

def _build_ocr_engine(use_gpu: bool, no_det: bool):
    from rapidocr_onnxruntime import RapidOCR

    kwargs: dict = {"no_cls": True}
    if no_det:
        kwargs["no_det"] = True
    if use_gpu:
        kwargs["det_use_cuda"] = True
        kwargs["rec_use_cuda"] = True
    return RapidOCR(**kwargs)

def _load_frame_gray_small(path: Path, width: int = 128, height: int = 32):
    import cv2
    import numpy as np

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros((height, width), dtype=np.uint8)
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

def _frame_changed(prev, curr, threshold: float) -> bool:
    import numpy as np

    if prev is None:
        return True
    diff = np.abs(prev.astype(np.float32) - curr.astype(np.float32)).mean()
    return float(diff) >= threshold

def _parse_ocr_result(result, min_confidence: float) -> str | None:
    if not result:
        return None
    parts: list[str] = []
    for item in result:
        if len(item) < 3:
            continue
        conf = float(item[2])
        if conf < min_confidence:
            continue
        text = _normalize_ocr_text(str(item[1]))
        if text:
            parts.append(text)
    if not parts:
        return None
    merged = "".join(parts)
    if not _is_meaningful_subtitle(merged):
        return None
    return merged

def _ocr_worker_init(use_gpu: bool, no_det: bool, min_confidence: float) -> None:
    global _OCR_WORKER
    if use_gpu:
        _prepare_nvidia_ocr_dll_paths()
    _OCR_WORKER = {
        "engine": _build_ocr_engine(use_gpu, no_det),
        "no_det": no_det,
        "min_confidence": min_confidence,
    }

def _ocr_single_frame(task: tuple[int, float, str]) -> tuple[int, float, str | None]:
    i, t, frame_path = task
    engine = _OCR_WORKER["engine"]
    use_det = not _OCR_WORKER["no_det"]
    result, _ = engine(frame_path, use_det=use_det, use_cls=False, use_rec=True)
    text = _parse_ocr_result(result, _OCR_WORKER["min_confidence"])
    return (i, t, text)

def _resolve_ocr_workers(use_gpu: bool, workers: int | None) -> int:
    if use_gpu:
        return 1
    if workers is not None:
        return max(1, workers)
    return min(os.cpu_count() or 1, 4)

def _select_ocr_frames(
    frame_files: list[Path],
    fps: float,
    skip_unchanged: bool,
    change_threshold: float,
    min_sample_interval: float = 1.0,
) -> list[tuple[int, float, Path]]:
    if not skip_unchanged:
        return [(i, i / fps, p) for i, p in enumerate(frame_files)]

    tasks: list[tuple[int, float, Path]] = []
    prev_gray = None
    last_sample_t = -min_sample_interval
    for i, frame_path in enumerate(frame_files):
        t = i / fps
        gray = _load_frame_gray_small(frame_path)
        changed = _frame_changed(prev_gray, gray, change_threshold)
        periodic = (t - last_sample_t) >= min_sample_interval - 1e-6
        if changed or periodic:
            tasks.append((i, t, frame_path))
            last_sample_t = t
            if changed:
                prev_gray = gray
    return tasks

def _run_ocr_on_frames(
    ocr_tasks: list[tuple[int, float, Path]],
    use_gpu: bool,
    no_det: bool,
    min_confidence: float,
    workers: int,
) -> list[tuple[int, float, str]]:
    if not ocr_tasks:
        return []

    serializable = [(i, t, str(p)) for i, t, p in ocr_tasks]
    results: list[tuple[int, float, str]] = []

    if workers <= 1:
        engine = _build_ocr_engine(use_gpu, no_det)
        use_det = not no_det
        for idx, (i, t, frame_path) in enumerate(serializable):
            result, _ = engine(frame_path, use_det=use_det, use_cls=False, use_rec=True)
            text = _parse_ocr_result(result, min_confidence)
            if text:
                results.append((i, t, text))
            if (idx + 1) % max(1, len(serializable) // 10) == 0:
                print(f"  OCR 进度: {idx + 1}/{len(serializable)}")
        return results

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_ocr_worker_init,
        initargs=(use_gpu, no_det, min_confidence),
    ) as pool:
        futures = [pool.submit(_ocr_single_frame, task) for task in serializable]
        done = 0
        for fut in as_completed(futures):
            done += 1
            i, t, text = fut.result()
            if text:
                results.append((i, t, text))
            if done % max(1, len(serializable) // 10) == 0:
                print(f"  OCR 进度: {done}/{len(serializable)}")

    results.sort(key=lambda x: x[0])
    return results

def _merge_ocr_jitter_segments(segments: list[dict], max_gap: float = 1.0) -> list[dict]:
    """合并相邻 OCR 抖动 (互为子串且间隔很短), 避免误吞正常字幕段。"""
    merged: list[dict] = []
    for seg in segments:
        if not merged:
            merged.append(dict(seg))
            continue
        prev = merged[-1]
        gap = float(seg["start"]) - float(prev["end"])
        if gap <= max_gap:
            if seg["text"] in prev["text"]:
                prev["end"] = seg["end"]
                continue
            if prev["text"] in seg["text"]:
                merged[-1] = dict(seg)
                continue
        merged.append(dict(seg))
    return merged

def extract_burned_in_subtitles(
    video_path: Path,
    work_dir: Path,
    ocr_lang: str = "zh",
    fps: float = 2.0,
    crop_ratio: float = 0.28,
    min_confidence: float = 0.5,
    skip_unchanged: bool = True,
    change_threshold: float = 2.0,
    use_gpu: bool = True,
    no_det: bool = False,
    workers: int | None = None,
) -> list[dict]:
    """
    OCR 识别视频画面底部烧录字幕, 返回 Whisper 兼容的分段列表。
    """
    ffmpeg = _require_ffmpeg()
    frames_dir = work_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    crop_y = 1.0 - crop_ratio
    vf = f"fps={fps},crop=in_w:in_h*{crop_ratio:.4f}:0:in_h*{crop_y:.4f}"
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-vf",
        vf,
        "-q:v",
        "2",
        str(frames_dir / "frame_%06d.jpg"),
    ]
    print(f"OCR 抽帧: {video_path.name} (fps={fps}, 底部 {crop_ratio:.0%} 区域)...")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    if not frame_files:
        raise RuntimeError("OCR 抽帧失败, 未生成任何画面")

    lang_list = _ocr_lang_list(ocr_lang)
    ocr_tasks = _select_ocr_frames(
        frame_files, fps, skip_unchanged, change_threshold
    )
    use_gpu = _resolve_ocr_use_gpu(use_gpu)
    num_workers = _resolve_ocr_workers(use_gpu, workers)
    gpu_label = "GPU" if use_gpu else "CPU"
    det_label = "无检测" if no_det else "含检测"
    skip_label = f"跳变帧 {len(ocr_tasks)}/{len(frame_files)}" if skip_unchanged else "全帧"
    print(
        f"OCR 识别语言: {ocr_lang} ({', '.join(lang_list)}), "
        f"抽帧 {len(frame_files)} → {skip_label}, "
        f"{gpu_label}/{det_label}, workers={num_workers}"
    )

    ocr_results = _run_ocr_on_frames(
        ocr_tasks, use_gpu, no_det, min_confidence, num_workers
    )
    timeline = [(t, text) for _, t, text in ocr_results]

    if not timeline:
        raise RuntimeError(
            "OCR 未识别到字幕文字; 可尝试调高 --ocr-fps 或调整 --ocr-crop-ratio"
        )

    duration = get_media_duration(video_path)
    segments: list[dict] = []
    cur_text = timeline[0][1]
    cur_start = timeline[0][0]
    for t, text in timeline[1:]:
        if text == cur_text:
            continue
        segments.append({"start": cur_start, "end": t, "text": cur_text})
        cur_text = text
        cur_start = t
    segments.append({"start": cur_start, "end": duration, "text": cur_text})

    merged = _merge_ocr_jitter_segments(segments)

    print(f"OCR 得到 {len(merged)} 条字幕段")
    filtered = [s for s in merged if _is_meaningful_subtitle(s["text"])]
    if len(filtered) < len(merged):
        print(f"过滤 OCR 噪声后保留 {len(filtered)} 条")
    return filtered
