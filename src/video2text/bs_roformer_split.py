"""BS-RoFormer / Mel-Band RoFormer vocal separation via audio-separator."""
from __future__ import annotations

import re
from pathlib import Path

DEFAULT_ROFORMER_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
DEFAULT_ROFORMER_VOCALS_MODEL = "bs_roformer_vocals_revive_v3e_unwa.ckpt"
DEFAULT_ROFORMER_INST_MODEL = "mel_band_roformer_instrumental_fv7z_gabox.ckpt"
DEFAULT_ROFORMER_MODE = "ensemble"
DEFAULT_ROFORMER_ENSEMBLE_PRESET = "instrumental_full"

_STEM_LABEL_RE = re.compile(r"_\(([^)]+)\)_")


def _require_audio_separator():
    try:
        from audio_separator.separator import Separator
    except ImportError as exc:
        raise RuntimeError(
            "BS-RoFormer 需要 audio-separator: "
            "pip install -r requirements-bs-roformer.txt "
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
    # 兼容无括号标签的旧输出: 仅匹配文件名 stem 部分 (不含路径)
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
):
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
    return Separator(**kwargs)


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
) -> tuple[Path, Path]:
    out_dir = work_dir / "bs_roformer"
    if run_label:
        out_dir = work_dir / f"bs_roformer_{run_label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    separator = _make_separator(
        out_dir,
        model_dir,
        ensemble_preset=ensemble_preset,
        overlap=overlap,
        output_format=output_format,
    )
    load_name = model_filename
    if ensemble_preset:
        print(f"  ensemble preset: {ensemble_preset}")
    else:
        print(f"  model: {model_filename}")
    separator.load_model(model_filename=load_name)
    outputs = separator.separate(str(audio_wav))
    if not outputs:
        raise RuntimeError("BS-RoFormer 未返回任何输出文件")
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
    model_dir = work_dir / "roformer_models"
    model_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mode = roformer_mode.lower().strip()

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
            model_dir,
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
            model_dir,
            voc_model,
            overlap=overlap,
            output_format=output_format,
            run_label="vocals",
        )
        import gc

        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        instrumental, _ = _run_single_separation(
            audio_wav,
            work_dir,
            model_dir,
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
        f"overlap={overlap or 'default'}, device={device})..."
    )
    return _run_single_separation(
        audio_wav,
        work_dir,
        model_dir,
        model_filename,
        overlap=overlap,
        output_format=output_format,
    )
