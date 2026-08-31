from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from video2text.ffmpeg_util import _load_whisper_audio


def _whisper_model_cache_path(model_name: str) -> Path:
    return Path.home() / ".cache" / "whisper" / f"{model_name}.pt"

def load_whisper_model(model_name: str, device: str | None):
    import whisper
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"加载 Whisper 模型: {model_name} (device={device})")

    try:
        return whisper.load_model(model_name, device=device), device
    except RuntimeError as exc:
        message = str(exc)
        if "SHA256 checksum" not in message:
            raise

        cache_path = _whisper_model_cache_path(model_name)
        if cache_path.is_file():
            size_mb = cache_path.stat().st_size / (1024 * 1024)
            print(
                f"检测到损坏的模型缓存 {cache_path} ({size_mb:.1f} MB), 正在删除并重新下载...",
                file=sys.stderr,
            )
            cache_path.unlink(missing_ok=True)

        print("正在重新下载 Whisper 模型 (请保持网络畅通)...", file=sys.stderr)
        return whisper.load_model(model_name, device=device), device

def detect_and_transcribe(
    model,
    audio_path: Path,
    language: str | None,
    task: str = "transcribe",
):
    """
    识别语言并转写.
    language=None 时由 Whisper 自动检测.
    """
    import whisper

    audio = _load_whisper_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

    # 语言检测 (即使用户指定了 language, 也打印一次检测结果供参考)
    _, probs = model.detect_language(mel)
    detected = max(probs, key=probs.get)
    print(
        "语言检测: {} (置信度 {:.2%})".format(detected, probs[detected])
    )
    top5 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
    print(
        "Top-5: "
        + ", ".join("{}={:.1%}".format(lang, p) for lang, p in top5)
    )

    use_lang = language or detected
    print(f"转写语言: {use_lang}, task={task}")

    result = model.transcribe(
        audio,
        language=use_lang,
        task=task,
        verbose=False,
    )
    # result["language"] 为 Whisper 最终使用的语言码
    return result, detected, float(probs[detected])
