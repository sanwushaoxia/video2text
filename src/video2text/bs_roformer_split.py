"""BS-RoFormer / Mel-Band RoFormer vocal separation via audio-separator."""
from __future__ import annotations

from contextvars import ContextVar
import json
import os
import re
import shutil
from importlib import resources
from pathlib import Path
from zipfile import is_zipfile

from video2text.paths import roformer_model_dir as _default_roformer_dir
from video2text.performance import cache_event, stage
from video2text.stage_cache import model_artifacts

DEFAULT_ROFORMER_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
DEFAULT_ROFORMER_VOCALS_MODEL = "bs_roformer_vocals_revive_v3e_unwa.ckpt"
DEFAULT_ROFORMER_INST_MODEL = "mel_band_roformer_instrumental_fv7z_gabox.ckpt"
DEFAULT_ROFORMER_MODE = "ensemble"
DEFAULT_ROFORMER_ENSEMBLE_PRESET = "instrumental_full"
ROFORMER_MODEL_DIR_ENV = "VIDEO2TEXT_ROFORMER_MODEL_DIR"

_STEM_LABEL_RE = re.compile(r"_\(([^)]+)\)_")
_BUNDLED_ROFORMER_DIR = Path(__file__).resolve().parents[2] / "assets" / "roformer"
_BUNDLED_DOWNLOAD_CHECKS = _BUNDLED_ROFORMER_DIR / "download_checks.json"
_worker_models = ContextVar("video2text_roformer_worker", default=None)


def enable_cpu_model_reuse() -> None:
    """Retain one single-checkpoint separator per serial CPU worker."""
    _worker_models.set({})


def default_roformer_model_dir() -> Path:
    """默认持久化模型目录: 项目 models/roformer。"""
    return _default_roformer_dir()


def requested_roformer_precision() -> str:
    return os.environ.get("VIDEO2TEXT_ROFORMER_PRECISION", "fp32").strip().lower()


def resolve_roformer_model_dir(model_dir: Path | str | None = None) -> Path:
    if model_dir is not None and str(model_dir).strip():
        path = Path(model_dir).expanduser().resolve()
    else:
        path = default_roformer_model_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_download_checks_manifest(model_dir: Path) -> Path:
    """确保 model_dir 内有 download_checks.json (联网或 bundled 兜底)。"""
    target = model_dir / "download_checks.json"
    if target.is_file() and target.stat().st_size > 100:
        return target

    if _BUNDLED_DOWNLOAD_CHECKS.is_file():
        shutil.copy2(_BUNDLED_DOWNLOAD_CHECKS, target)
        print(f"  使用内置 manifest: {target}")
        return target

    try:
        import requests

        url = (
            "https://raw.githubusercontent.com/TRvlvr/application_data/"
            "main/filelists/download_checks.json"
        )
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        target.write_bytes(resp.content)
        print(f"  已下载 manifest: {target}")
        return target
    except Exception as exc:
        raise RuntimeError(
            "无法获取 RoFormer 模型清单 download_checks.json。\n"
            f"  模型目录: {model_dir}\n"
            "  请检查网络/代理 (HTTPS_PROXY), 或运行:\n"
            "    python scripts/prefetch_roformer_models.py\n"
            "  也可手动将 download_checks.json 放入上述目录。"
        ) from exc


def roformer_network_error_message(model_dir: Path) -> str:
    return (
        "BS-RoFormer 模型下载失败 (无法访问 GitHub)。\n"
        f"  模型目录: {model_dir}\n"
        "  每个 MDXC 模型需 ckpt + yaml 配置; 缺 yaml 也会触发联网下载。\n"
        "  1) 配置代理后重试, 例如:\n"
        "     export HTTPS_PROXY=http://127.0.0.1:7890\n"
        "  2) 预下载模型到持久目录:\n"
        "     python scripts/prefetch_roformer_models.py\n"
        "  3) 指定目录: --roformer-model-dir 或环境变量 "
        f"{ROFORMER_MODEL_DIR_ENV}\n"
        "  4) 减小下载量: --roformer-mode single"
    )


def _require_audio_separator():
    try:
        from audio_separator.separator import Separator
    except ImportError as exc:
        raise RuntimeError(
            "BS-RoFormer 需要 audio-separator: "
            "pip install -r requirements.txt "
            '或 pip install "audio-separator[gpu]"'
        ) from exc
    return Separator


def _stem_label(filename: str) -> str:
    """从 audio-separator 输出名提取 stem 标签, 如 source_(Vocals)_model... -> vocals。"""
    match = _STEM_LABEL_RE.search(Path(filename).name)
    if match:
        return match.group(1).lower().strip()
    return Path(filename).stem.lower()


def _pick_stem(output_files: list[str], keywords: tuple[str, ...]) -> Path | None:
    """按 stem 标签匹配, 避免模型名含 instrumental/vocals 时误匹配。"""
    lowered = tuple(k.lower() for k in keywords)
    for item in output_files:
        label = _stem_label(item)
        if any(key == label or label.replace(" ", "_") == key.replace(" ", "_") for key in lowered):
            path = Path(item)
            if path.is_file():
                return path.resolve()
    for item in output_files:
        stem = Path(item).stem.lower()
        if any(stem == key or stem.endswith(f"_{key}") for key in lowered):
            path = Path(item)
            if path.is_file():
                return path.resolve()
    return None


def _resolve_output_paths(outputs: list[str], out_dir: Path) -> list[str]:
    resolved: list[str] = []
    for p in outputs:
        path = Path(p)
        if not path.is_file():
            candidate = out_dir / path.name
            if candidate.is_file():
                path = candidate
        resolved.append(str(path.resolve()))
    return resolved


def _stems_from_outputs(output_paths: list[str]) -> tuple[Path, Path]:
    vocals = _pick_stem(output_paths, ("vocals",))
    instrumental = _pick_stem(
        output_paths,
        ("instrumental", "no_vocals", "no vocals", "other"),
    )
    if vocals is None or instrumental is None:
        raise RuntimeError(f"BS-RoFormer 输出不完整: {output_paths}")
    return instrumental, vocals


def _make_separator(
    out_dir: Path,
    model_dir: Path,
    *,
    ensemble_preset: str | None = None,
    overlap: int | None = None,
    output_format: str = "wav",
    native_fp16: bool = False,
):
    ensure_download_checks_manifest(model_dir)
    Separator = _require_audio_separator()
    mdxc_params: dict = {
        "segment_size": 256,
        "override_model_segment_size": False,
        "batch_size": None,
        "overlap": overlap,
        "pitch_shift": 0,
    }
    kwargs: dict = {
        "output_dir": str(out_dir),
        "output_format": output_format,
        "model_file_dir": str(model_dir),
        "mdxc_params": mdxc_params,
    }
    if ensemble_preset:
        kwargs["ensemble_preset"] = ensemble_preset
    if native_fp16:
        kwargs["use_native_fp16"] = True
    return Separator(**kwargs)


def _raise_if_network_error(exc: BaseException, *, model_dir: Path) -> None:
    if isinstance(exc, (ConnectionError, TimeoutError)):
        raise RuntimeError(roformer_network_error_message(model_dir)) from exc
    if isinstance(exc, OSError) and (
        "Network" in str(exc) or "connection" in str(exc).lower()
    ):
        raise RuntimeError(roformer_network_error_message(model_dir)) from exc
    try:
        import requests

        if isinstance(
            exc,
            (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ConnectTimeout,
                requests.exceptions.ReadTimeout,
            ),
        ):
            raise RuntimeError(roformer_network_error_message(model_dir)) from exc
    except ImportError:
        pass
    if type(exc).__name__ in {"ConnectionError", "ConnectTimeout", "ReadTimeout"}:
        raise RuntimeError(roformer_network_error_message(model_dir)) from exc


def _load_model_safe(separator, model_filename: str, *, model_dir: Path) -> None:
    try:
        separator.load_model(model_filename=model_filename)
    except Exception as exc:
        _raise_if_network_error(exc, model_dir=model_dir)
        raise


def _download_model_files_safe(separator, model_filename: str, *, model_dir: Path) -> None:
    try:
        separator.download_model_files(model_filename)
    except Exception as exc:
        _raise_if_network_error(exc, model_dir=model_dir)
        raise


def _ensemble_ckpt_filenames(preset: str) -> list[str]:
    with resources.open_text("audio_separator", "ensemble_presets.json") as f:
        data = json.load(f)
    presets = data.get("presets", {})
    if preset not in presets:
        raise ValueError(f"未知 ensemble preset: {preset}")
    return list(presets[preset]["models"])


def _artifact_local_name(file_ref: str) -> str:
    if file_ref.startswith("http"):
        return file_ref.split("/")[-1]
    return file_ref


def _artifact_download_urls(model_type: str, file_ref: str, *, is_vip: bool) -> list[str]:
    if file_ref.startswith("http"):
        return [file_ref]
    public_prefix = (
        "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models"
    )
    vip_prefix = "https://github.com/Anjok0109/ai_magic/releases/download/v5"
    audio_sep_prefix = (
        "https://github.com/nomadkaraoke/python-audio-separator/releases/download/model-configs"
    )
    prefix = vip_prefix if is_vip else public_prefix
    urls = [f"{prefix}/{file_ref}"]
    if model_type == "MDXC" and file_ref.endswith(".yaml"):
        urls.insert(0, f"{prefix}/mdx_model_data/mdx_c_configs/{file_ref}")
    else:
        urls.append(f"{audio_sep_prefix}/{file_ref}")
    return urls


def _expected_remote_size(urls: list[str]) -> int | None:
    try:
        import requests
    except ImportError:
        return None
    for url in urls:
        try:
            resp = requests.head(url, timeout=30, allow_redirects=True)
            if resp.status_code == 200:
                content_length = resp.headers.get("content-length")
                if content_length:
                    return int(content_length)
        except Exception:
            continue
    return None


_MIN_ARTIFACT_BYTES = {
    ".ckpt": 50 * 1024 * 1024,
    ".pth": 10 * 1024 * 1024,
    ".onnx": 5 * 1024 * 1024,
    ".th": 10 * 1024 * 1024,
    ".yaml": 100,
}


def _min_size_for(name: str) -> int:
    return _MIN_ARTIFACT_BYTES.get(Path(name).suffix.lower(), 1024)


def _is_artifact_complete(path: Path, expected_size: int | None) -> bool:
    if not path.is_file():
        return False
    size = path.stat().st_size
    if expected_size is None and size < _min_size_for(path.name):
        return False
    if expected_size is not None and size != expected_size:
        return False
    if path.suffix.lower() in {".ckpt", ".pth", ".th"}:
        with path.open("rb") as file:
            # PyTorch also accepts legacy non-ZIP checkpoints. For ZIP checkpoints,
            # a missing central directory identifies an interrupted download locally.
            if file.read(4) == b"PK\x03\x04":
                return is_zipfile(file)
    return True


def ensure_bundled_model_yaml(model_dir: Path, yaml_name: str) -> Path | None:
    """本地缺失时从 assets/roformer 复制 yaml 配置。"""
    target = model_dir / yaml_name
    if _is_artifact_complete(target, None):
        return target
    bundled = _BUNDLED_ROFORMER_DIR / yaml_name
    if not bundled.is_file():
        return None
    shutil.copy2(bundled, target)
    print(f"  使用内置 yaml: {yaml_name}")
    return target


def _collect_model_artifacts(separator, model_filename: str) -> list[tuple[str, list[str]]]:
    grouped = separator.list_supported_model_files()
    for model_type, models in grouped.items():
        for model_friendly_name, model_info in models.items():
            filename = model_info["filename"]
            download_files = model_info["download_files"]
            if filename != model_filename and model_filename not in download_files:
                continue
            is_vip = "VIP" in model_friendly_name
            return [
                (
                    _artifact_local_name(file_ref),
                    _artifact_download_urls(model_type, file_ref, is_vip=is_vip),
                )
                for file_ref in download_files
            ]
    raise ValueError(f"模型未在 manifest 中找到: {model_filename}")


def _prune_incomplete_artifacts(
    model_dir: Path, artifacts: list[tuple[str, list[str]]]
) -> None:
    for local_name, urls in artifacts:
        path = model_dir / local_name
        if not path.is_file():
            continue
        expected = _expected_remote_size(urls)
        if _is_artifact_complete(path, expected):
            continue
        size = path.stat().st_size
        print(f"  移除不完整文件: {path.name} ({size // (1024 * 1024)} MiB)")
        path.unlink()


def _verify_model_artifacts(
    model_dir: Path, artifacts: list[tuple[str, list[str]]], *, model_filename: str
) -> None:
    problems: list[str] = []
    for local_name, urls in artifacts:
        path = model_dir / local_name
        expected = _expected_remote_size(urls)
        if _is_artifact_complete(path, expected):
            continue
        if path.is_file():
            problems.append(f"{local_name} ({path.stat().st_size} bytes)")
        else:
            problems.append(f"{local_name} (missing)")
    if problems:
        raise RuntimeError(
            f"模型文件校验失败 {model_filename}: {', '.join(problems)}\n"
            f"  模型目录: {model_dir}\n"
            "  请运行: python scripts/prefetch_roformer_models.py"
        )


def _download_url_to_file(url: str, target: Path) -> bool:
    try:
        import requests
    except ImportError:
        return False
    try:
        resp = requests.get(url, stream=True, timeout=120)
        if resp.status_code != 200:
            return False
        with target.open("wb") as handle:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    handle.write(chunk)
        return _is_artifact_complete(target, None)
    except Exception:
        if target.is_file():
            target.unlink(missing_ok=True)
        return False


def _ensure_yaml_artifact(
    model_dir: Path, local_name: str, urls: list[str]
) -> None:
    target = model_dir / local_name
    expected = _expected_remote_size(urls)
    if _is_artifact_complete(target, expected):
        return
    if target.is_file() and not _is_artifact_complete(target, expected):
        target.unlink()
    if ensure_bundled_model_yaml(model_dir, local_name):
        return
    for url in urls:
        if _download_url_to_file(url, target):
            return
        if target.is_file():
            target.unlink(missing_ok=True)
    if ensure_bundled_model_yaml(model_dir, local_name):
        return
    raise RuntimeError(
        f"无法获取模型配置 {local_name}。\n"
        f"  模型目录: {model_dir}\n"
        "  请检查网络/代理, 或运行:\n"
        "    python scripts/prefetch_roformer_models.py"
    )


def _ensure_model_artifacts_cached(
    separator, model_dir: Path, model_filename: str
) -> None:
    artifacts = _collect_model_artifacts(separator, model_filename)
    if all(_is_artifact_complete(model_dir / name, None) for name, _ in artifacts):
        return
    yaml_artifacts = [(n, u) for n, u in artifacts if n.endswith(".yaml")]
    ckpt_artifacts = [(n, u) for n, u in artifacts if not n.endswith(".yaml")]

    for local_name, urls in yaml_artifacts:
        _ensure_yaml_artifact(model_dir, local_name, urls)

    _prune_incomplete_artifacts(model_dir, ckpt_artifacts)
    missing_ckpts = [
        name
        for name, urls in ckpt_artifacts
        if not _is_artifact_complete(model_dir / name, _expected_remote_size(urls))
    ]
    if missing_ckpts:
        _download_model_files_safe(separator, model_filename, model_dir=model_dir)
    _verify_model_artifacts(model_dir, artifacts, model_filename=model_filename)


def _prefetch_single_ckpt(separator, model_filename: str, *, model_dir: Path) -> None:
    print(f"  下载: {model_filename}")
    artifacts = _collect_model_artifacts(separator, model_filename)
    _prune_incomplete_artifacts(model_dir, artifacts)
    for local_name, urls in artifacts:
        if local_name.endswith(".yaml"):
            _ensure_yaml_artifact(model_dir, local_name, urls)
    _download_model_files_safe(separator, model_filename, model_dir=model_dir)
    _verify_model_artifacts(model_dir, artifacts, model_filename=model_filename)
    for local_name, _ in artifacts:
        path = model_dir / local_name
        size_mib = path.stat().st_size // (1024 * 1024)
        if size_mib == 0 and path.stat().st_size > 0:
            print(f"    OK {local_name} ({path.stat().st_size} B)")
        else:
            print(f"    OK {local_name} ({size_mib} MiB)")


def prefetch_roformer_models(
    *,
    model_dir: Path | None = None,
    roformer_mode: str = DEFAULT_ROFORMER_MODE,
    ensemble_preset: str | None = DEFAULT_ROFORMER_ENSEMBLE_PRESET,
    model_filename: str = DEFAULT_ROFORMER_MODEL,
    vocals_model: str | None = None,
    inst_model: str | None = None,
) -> Path:
    """预下载 manifest + 当前配置所需 ckpt 到持久目录。"""
    resolved = resolve_roformer_model_dir(model_dir)
    ensure_download_checks_manifest(resolved)

    work = resolved.parent / ".prefetch_work"
    work.mkdir(parents=True, exist_ok=True)
    out_dir = work / "bs_roformer_prefetch"
    out_dir.mkdir(parents=True, exist_ok=True)

    mode = roformer_mode.lower().strip()
    ensure_download_checks_manifest(resolved)

    separator = _make_separator(out_dir, resolved, ensemble_preset=None)

    if mode == "ensemble":
        preset = ensemble_preset or DEFAULT_ROFORMER_ENSEMBLE_PRESET
        ckpts = _ensemble_ckpt_filenames(preset)
        print(f"预下载 ensemble preset: {preset} ({len(ckpts)} 个 ckpt)")
        for ckpt in ckpts:
            _prefetch_single_ckpt(separator, ckpt, model_dir=resolved)
    elif mode == "dual":
        voc = vocals_model or DEFAULT_ROFORMER_VOCALS_MODEL
        ins = inst_model or DEFAULT_ROFORMER_INST_MODEL
        print(f"预下载 dual: vocals={voc}, inst={ins}")
        _prefetch_single_ckpt(separator, voc, model_dir=resolved)
        _prefetch_single_ckpt(separator, ins, model_dir=resolved)
    else:
        print(f"预下载 single: {model_filename}")
        _prefetch_single_ckpt(separator, model_filename, model_dir=resolved)

    print(f"模型已缓存并校验通过: {resolved}")
    return resolved


def _run_single_separation(
    audio_wav: Path,
    work_dir: Path,
    model_dir: Path,
    model_filename: str,
    *,
    ensemble_preset: str | None = None,
    overlap: int | None = None,
    output_format: str = "wav",
    run_label: str = "",
    native_fp16: bool = False,
) -> tuple[Path, Path]:
    out_dir = work_dir / "bs_roformer"
    if run_label:
        out_dir = work_dir / f"bs_roformer_{run_label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    session = _worker_models.get()
    if session is not None:
        import torch

        if ensemble_preset or run_label or torch.cuda.is_available():
            session = None
    key = None
    if session is not None:
        key = (model_artifacts(model_dir), model_filename, overlap, output_format)
    reused = session is not None and session.get("key") == key and "separator" in session
    if session is not None:
        cache_event("roformer_model", reused)
    if reused:
        separator = session["separator"]
        separator.output_dir = str(out_dir)
        separator.model_instance.output_dir = str(out_dir)
    else:
        if session is not None:
            session.clear()
        separator = _make_separator(
            out_dir, model_dir, ensemble_preset=ensemble_preset,
            overlap=overlap, output_format=output_format, native_fp16=native_fp16,
        )
    load_name = model_filename
    if ensemble_preset:
        print(f"  ensemble preset: {ensemble_preset}")
    else:
        print(f"  model: {model_filename}")
    print(f"  模型目录: {model_dir}")
    if not reused:
        with stage("load_roformer_model"):
            with stage("verify_roformer_artifacts"):
                checkpoints = _ensemble_ckpt_filenames(ensemble_preset) if ensemble_preset else (model_filename,)
                for ckpt in checkpoints:
                    _ensure_model_artifacts_cached(separator, model_dir, ckpt)
            _load_model_safe(separator, load_name, model_dir=model_dir)
            if native_fp16 and separator.effective_precision != "native_fp16":
                raise RuntimeError(f"模型 {model_filename} 不支持在当前设备上使用原生 FP16")
    try:
        outputs = separator.separate(str(audio_wav))
    except Exception as exc:
        if session is not None:
            session.clear()
        _raise_if_network_error(exc, model_dir=model_dir)
        raise
    if not outputs:
        raise RuntimeError("BS-RoFormer 未返回任何输出文件")
    if session is not None and not reused:
        session.update(key=(model_artifacts(model_dir), model_filename, overlap, output_format),
                       separator=separator)
    output_paths = _resolve_output_paths(outputs, out_dir)
    return _stems_from_outputs(output_paths)


def separate_vocals_bs_roformer(
    audio_wav: Path,
    work_dir: Path,
    *,
    model_filename: str = DEFAULT_ROFORMER_MODEL,
    roformer_mode: str = DEFAULT_ROFORMER_MODE,
    ensemble_preset: str | None = DEFAULT_ROFORMER_ENSEMBLE_PRESET,
    vocals_model: str | None = None,
    inst_model: str | None = None,
    overlap: int | None = None,
    output_format: str = "wav",
    model_dir: Path | str | None = None,
    precision: str = "fp32",
) -> tuple[Path, Path]:
    """
    RoFormer 双轨分离: 返回 (instrumental/no_vocals, vocals)。

    roformer_mode:
      - single: 单模型 (model_filename)
      - dual: vocals 与 instrumental 各用专精模型
      - ensemble: audio-separator ensemble preset (instrumental_clean 等)
    """
    import torch

    work_dir.mkdir(parents=True, exist_ok=True)
    resolved_model_dir = resolve_roformer_model_dir(model_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mode = roformer_mode.lower().strip()
    if precision not in ("fp32", "fp16"):
        raise ValueError("VIDEO2TEXT_ROFORMER_PRECISION 需要是 fp32 或 fp16")
    if precision == "fp16" and (mode != "single" or device != "cuda"):
        raise ValueError("RoFormer FP16 需要 CUDA 和 --roformer-mode single")

    if mode == "ensemble":
        preset = ensemble_preset or DEFAULT_ROFORMER_ENSEMBLE_PRESET
        if not preset:
            raise ValueError("roformer_mode=ensemble 需要指定 ensemble_preset")
        print(
            f"分离人声与背景音乐 (RoFormer ensemble={preset}, "
            f"overlap={overlap or 'default'}, device={device})..."
        )
        return _run_single_separation(
            audio_wav,
            work_dir,
            resolved_model_dir,
            model_filename,
            ensemble_preset=preset,
            overlap=overlap,
            output_format=output_format,
        )

    if mode == "dual":
        voc_model = vocals_model or DEFAULT_ROFORMER_VOCALS_MODEL
        ins_model = inst_model or DEFAULT_ROFORMER_INST_MODEL
        print(
            f"分离人声与背景音乐 (RoFormer dual, device={device}, "
            f"vocals={voc_model}, inst={ins_model})..."
        )
        _, vocals = _run_single_separation(
            audio_wav,
            work_dir,
            resolved_model_dir,
            voc_model,
            overlap=overlap,
            output_format=output_format,
            run_label="vocals",
        )
        import gc

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        instrumental, _ = _run_single_separation(
            audio_wav,
            work_dir,
            resolved_model_dir,
            ins_model,
            overlap=overlap,
            output_format=output_format,
            run_label="inst",
        )
        return instrumental, vocals

    if mode != "single":
        raise ValueError(f"未知 roformer_mode: {roformer_mode}")

    print(
        f"分离人声与背景音乐 (RoFormer {model_filename}, "
        f"overlap={overlap or 'default'}, device={device}, precision={precision})..."
    )
    return _run_single_separation(
        audio_wav,
        work_dir,
        resolved_model_dir,
        model_filename,
        overlap=overlap,
        output_format=output_format,
        native_fp16=precision == "fp16",
    )
