"""项目内 models/ 目录路径与 HuggingFace 环境。"""
from __future__ import annotations

import os
from pathlib import Path

MODELS_DIR_ENV = "VIDEO2TEXT_MODELS_DIR"
ROFORMER_MODEL_DIR_ENV = "VIDEO2TEXT_ROFORMER_MODEL_DIR"
WHISPER_MODEL_DIR_ENV = "VIDEO2TEXT_WHISPER_MODEL_DIR"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def models_dir() -> Path:
    env = os.environ.get(MODELS_DIR_ENV, "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (repo_root() / "models").resolve()


def roformer_model_dir() -> Path:
    env = os.environ.get(ROFORMER_MODEL_DIR_ENV, "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (models_dir() / "roformer").resolve()


def whisper_model_dir() -> Path:
    env = os.environ.get(WHISPER_MODEL_DIR_ENV, "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (models_dir() / "whisper").resolve()


def huggingface_home() -> Path:
    env = os.environ.get("HF_HOME", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (models_dir() / "huggingface").resolve()


def huggingface_hub_cache() -> Path:
    env = os.environ.get("HF_HUB_CACHE", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (huggingface_home() / "hub").resolve()


def speechbrain_model_dir(name: str) -> Path:
    safe = name.replace("/", "_").strip()
    return (models_dir() / "speechbrain" / safe).resolve()


def ensure_models_subdirs() -> None:
    for path in (
        roformer_model_dir(),
        whisper_model_dir(),
        huggingface_home(),
        huggingface_hub_cache(),
        models_dir() / "speechbrain",
    ):
        path.mkdir(parents=True, exist_ok=True)


def ensure_huggingface_env() -> None:
    """在 import huggingface_hub / pyannote 前调用。"""
    if not os.environ.get("HF_HUB_ENABLE_HF_TRANSFER"):
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    if not os.environ.get("HF_HOME"):
        os.environ["HF_HOME"] = str(huggingface_home())
    if not os.environ.get("HF_HUB_CACHE"):
        os.environ["HF_HUB_CACHE"] = str(huggingface_hub_cache())
    huggingface_home().mkdir(parents=True, exist_ok=True)
    huggingface_hub_cache().mkdir(parents=True, exist_ok=True)


def legacy_roformer_model_dir() -> Path | None:
    """迁移脚本用：旧 AppData / ~/.cache 路径。"""
    local_app = os.environ.get("LOCALAPPDATA", "").strip()
    if local_app:
        path = Path(local_app) / "video2text" / "roformer_models"
        if path.is_dir():
            return path.resolve()
    path = Path.home() / ".cache" / "video2text" / "roformer_models"
    if path.is_dir():
        return path.resolve()
    return None


def legacy_whisper_model_dir() -> Path:
    return (Path.home() / ".cache" / "whisper").resolve()


def legacy_huggingface_home() -> Path:
    return (Path.home() / ".cache" / "huggingface").resolve()
