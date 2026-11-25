"""Lightweight CUDA identity for result-cache lookups with standard Linux PyTorch wheels."""
from __future__ import annotations

from ctypes import CDLL, byref, c_int, create_string_buffer
from importlib.util import find_spec
import os
from pathlib import Path
import re
import subprocess
import sys


def _device_name(runtime_name, bundled_runtime):
    try:
        runtime = CDLL(runtime_name)
    except OSError:
        runtime = CDLL(bundled_runtime)

    count = c_int()
    if runtime.cudaGetDeviceCount(byref(count)) or count.value < 1:
        return
    driver = CDLL("libcuda.so.1")
    device = c_int()
    name = create_string_buffer(256)
    if not driver.cuDeviceGet(byref(device), 0) and not driver.cuDeviceGetName(name, len(name), device):
        print(name.value.decode())


def wheel_cuda_identity() -> tuple[str, str] | None:
    if sys.platform != "linux" or os.environ.get("PYTORCH_NVML_BASED_CUDA_CHECK") == "1":
        return None
    spec = find_spec("torch")
    if spec is None or not spec.origin:
        return None
    package = Path(spec.origin).parent
    version_file = package / "version.py"
    try:
        match = re.search(r"""(?m)^cuda(?:\s*:[^=\n]+)?\s*=\s*['"](\d+\.\d+)['"]""",
                          version_file.read_text(encoding="utf-8"))
        if not match:
            return None
        version = match[1]
        major = version.split(".")[0]
        library = f"libcudart.so.{major}"
        nvidia = package.parent / "nvidia"
        bundled = next((path for path in (
            nvidia / f"cu{major}" / "lib" / library,
            nvidia / "cuda_runtime" / "lib" / library,
        ) if path.is_file()), None)
        if bundled is None:
            return None
        # Keep CUDA runtime/driver initialization out of the process loading PyTorch on a miss.
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), library, str(bundled)],
            capture_output=True, text=True, timeout=2, check=False,
        )
        name = result.stdout.strip()
        return (name, version) if result.returncode == 0 and name else None
    except (OSError, subprocess.SubprocessError):
        return None


if __name__ == "__main__":
    _device_name(*sys.argv[1:])
