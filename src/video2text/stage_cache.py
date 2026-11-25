"""Persistent completed-stage artifacts with cross-process publication."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

from video2text.performance import cache_event


_active = ContextVar("video2text_stage_cache", default=None)
_SCHEMA = 1


@contextmanager
def use_cache(directory):
    token = _active.set(StageCache(Path(directory).expanduser().resolve()) if directory else None)
    try:
        yield
    finally:
        _active.reset(token)


def get_cache():
    return _active.get()


def versions(*packages):
    result = {}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def inference_environment(*, fast=False):
    settings = {name: os.environ.get(name) for name in ("MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    if fast:
        loaded = sys.modules.get("torch")
        requested = settings["MKL_NUM_THREADS"] or settings["OMP_NUM_THREADS"]
        threads = loaded.get_num_threads() if loaded is not None else (
            int(requested) if requested and requested.isdecimal() else None
        )
        if os.environ.get("CUDA_VISIBLE_DEVICES") in ("", "-1"):
            return {"device": "cpu", "cuda": None, "threads": settings, "torch_threads": threads}
        if loaded is None and threads:
            from video2text.cuda_probe import wheel_cuda_identity

            identity = wheel_cuda_identity()
            if identity is not None:
                device, cuda = identity
                return {"device": device, "cuda": cuda, "threads": settings, "torch_threads": threads}
    import torch

    cuda = torch.cuda.is_available()
    return {"device": torch.cuda.get_device_name() if cuda else "cpu",
            "cuda": torch.version.cuda if cuda else None, "threads": settings,
            "torch_threads": torch.get_num_threads()}


def model_artifacts(root: Path):
    """Identify installed checkpoint/config revisions without reading model tensors."""
    if root.is_file():
        paths = [root]
        base = root.parent
    else:
        paths = sorted(p for p in root.rglob("*") if p.is_file()
                       and len(p.parts) - len(root.parts) <= 5
                       and not p.name.endswith(".lock"))
        base = root
    return {
        "path": str(root.resolve()),
        "files": [(str(path.relative_to(base)), path.stat().st_size, path.stat().st_mtime_ns)
                  for path in paths],
    }


def copy_bundle(source: Path, target: Path, origin: Path):
    """Copy immutable cache assets and attach report paths to this publication."""
    for path in source.rglob("*"):
        if not path.is_file() or path.name in ("complete.json", ".bundle-origin"):
            continue
        output = target / path.relative_to(source)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, output)
    _relocate_reports(target, origin, target)


def _relocate_reports(directory: Path, origin: Path, target: Path):
    def relocate(value):
        if isinstance(value, dict):
            return {key: relocate(item) for key, item in value.items()}
        if isinstance(value, list):
            return [relocate(item) for item in value]
        if isinstance(value, str) and value.startswith(str(origin)):
            path = Path(value)
            if path.is_relative_to(origin):
                return str(target / path.relative_to(origin))
        return value

    for report in directory.glob("*_split_report.json"):
        data = json.loads(report.read_text(encoding="utf-8"))
        report.write_text(json.dumps(relocate(data), ensure_ascii=False, indent=2), encoding="utf-8")


class StageCache:
    def __init__(self, directory: Path):
        self.directory = directory
        self._digests = {}

    def fingerprint(self, path: Path):
        path = path.resolve()
        status = path.stat()
        key = (str(path), status.st_size, status.st_mtime_ns, status.st_ctime_ns)
        if key not in self._digests:
            digest = sha256()
            with path.open("rb") as file:
                while chunk := file.read(4 * 1024 * 1024):
                    digest.update(chunk)
            self._digests[key] = digest.hexdigest()
        return self._digests[key]

    def files(self, stage, parameters, names, build, *, validate=None):
        """Call build with a private stage directory and publish a complete file set."""
        from filelock import FileLock

        encoded = json.dumps([_SCHEMA, stage, parameters], sort_keys=True, separators=(",", ":"))
        key = sha256(encoded.encode()).hexdigest()
        parent = self.directory / stage
        parent.mkdir(parents=True, exist_ok=True)
        target = parent / key

        def result():
            return target if names is None else tuple(target / name for name in names)

        with FileLock(str(parent / f"{key}.lock")):
            manifest = target / "complete.json"
            if manifest.is_file():
                try:
                    sizes = json.loads(manifest.read_text(encoding="utf-8"))
                    expected = sizes if names is None else names
                    if (expected and all((target / name).is_file() and (target / name).stat().st_size == sizes[name]
                            for name in expected) and (validate is None or validate(target))):
                        cache_event(stage, True)
                        return result()
                except (KeyError, OSError, TypeError, ValueError):
                    pass
            cache_event(stage, False)
            prefix = f".build-{key}-"
            for abandoned in parent.glob(f"{prefix}*"):
                if abandoned.is_dir():
                    shutil.rmtree(abandoned)
            with TemporaryDirectory(prefix=prefix, dir=parent) as temp:
                directory = Path(temp) / "payload"
                directory.mkdir()
                build(directory)
                artifacts = names if names is not None else (
                    path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()
                )
                sizes = {name: (directory / name).stat().st_size for name in artifacts}
                if not sizes or any(size == 0 for size in sizes.values()):
                    raise RuntimeError(f"缓存阶段 {stage} 产生了空产物")
                (directory / "complete.json").write_text(json.dumps(sizes), encoding="utf-8")
                if target.exists():
                    # The stage lock keeps readers from observing replacement in progress.
                    shutil.rmtree(target)
                directory.rename(target)
        return result()


@contextmanager
def publish_directory(target: Path, enabled: bool):
    if not enabled:
        yield target
        return
    from filelock import FileLock

    target.parent.mkdir(parents=True, exist_ok=True)
    lock = target.parent / f".{target.name}.publish.lock"
    with FileLock(str(lock)):
        if target.exists():
            raise FileExistsError(f"输出目录已存在: {target}")
        with TemporaryDirectory(prefix=".video2text-", dir=target.parent) as temp:
            staged = Path(temp) / "bundle"
            staged.mkdir()
            yield staged
            _relocate_reports(staged, staged, target)
            if target.exists():
                raise FileExistsError(f"输出目录已存在: {target}")
            staged.rename(target)
