"""Opt-in stage timings and process high-water marks for separation."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter, process_time


_trace = ContextVar("video2text_performance", default=None)


def _gpu():
    torch = sys.modules.get("torch")
    return torch.cuda if torch is not None and torch.cuda.is_initialized() else None


def _rss_bytes():
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak * (1 if sys.platform == "darwin" else 1024))


def _io_counts():
    try:
        with open("/proc/self/io", encoding="ascii") as file:
            counters = dict(line.split(":", 1) for line in file)
        return {name: int(counters[name]) for name in ("read_bytes", "write_bytes")}
    except (OSError, KeyError, ValueError):
        return None


@contextmanager
def stage(name):
    trace = _trace.get()
    if trace is None:
        yield
        return
    cuda = _gpu()
    if cuda is not None:
        cuda.synchronize()
    started = perf_counter()
    event = {"stage": name, "status": "ok"}
    trace["events"].append(event)
    try:
        yield
    except BaseException:
        event["status"] = "failed"
        raise
    finally:
        cuda = _gpu()
        if cuda is not None and event["status"] == "ok":
            cuda.synchronize()
        event["wall_sec"] = round(perf_counter() - started, 6)


def timed_stage(name):
    def decorator(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            with stage(name):
                return function(*args, **kwargs)

        return wrapped

    return decorator


def cache_event(name, hit):
    if (trace := _trace.get()) is not None:
        trace["cache"].append({"stage": name, "hit": hit})


def record_audio_duration(path):
    if _trace.get() is not None:
        from video2text.media import get_media_duration

        record_audio_seconds(get_media_duration(path))


def record_audio_seconds(seconds):
    if (trace := _trace.get()) is not None:
        trace["audio_sec"] = seconds


def mark_failed():
    if (trace := _trace.get()) is not None:
        trace["status"] = "failed"


@contextmanager
def collect_performance(path, source, out_dir, stem):
    if path is None:
        yield
        return
    path = Path(path).expanduser().resolve()
    trace = {"input": str(source), "events": [], "cache": [], "python": platform.python_version(),
             "platform": platform.platform(), "status": "ok", "cpu_count": os.cpu_count(),
             "threads": {name: os.environ.get(name) for name in
                         ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}
    token = _trace.set(trace)
    started = perf_counter()
    cpu_started = process_time()
    io_before = _io_counts()
    try:
        yield
    except BaseException:
        trace["status"] = "failed"
        raise
    finally:
        _trace.reset(token)
        trace["wall_sec"] = round(perf_counter() - started, 6)
        trace["process_cpu_sec"] = round(process_time() - cpu_started, 6)
        io_after = _io_counts()
        if io_before is not None and io_after is not None:
            trace["process_disk_io"] = {name: io_after[name] - io_before[name] for name in io_after}
        if trace.get("audio_sec"):
            trace["real_time_factor"] = round(trace["wall_sec"] / trace["audio_sec"], 6)
        trace["peak_rss_bytes"] = _rss_bytes()
        cuda = _gpu()
        if cuda is not None:
            trace["gpu_peak_allocated_bytes"] = cuda.max_memory_allocated()
            trace["gpu_peak_reserved_bytes"] = cuda.max_memory_reserved()
        files = list(out_dir.glob(f"{stem}_*"))
        files.extend((out_dir / f"{stem}_overlap_sources").glob("*.wav"))
        trace["output_bytes"] = sum(item.stat().st_size for item in files
                                    if item.is_file() and item != path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(trace, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
