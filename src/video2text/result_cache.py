"""Opt-in reuse of a fully rendered multi-stem result."""
from __future__ import annotations

import json
from pathlib import Path

from video2text.ffmpeg_util import ffprobe_bin
from video2text.paths import huggingface_hub_cache, models_dir
from video2text.performance import record_audio_seconds, stage
from video2text.stage_cache import copy_bundle, inference_environment, model_artifacts, versions


def render_cached_result(cache, source, destination, settings, ffmpeg, build):
    # Keep output location and credentials independent of the inference result.
    from video2text.bs_roformer_split import requested_roformer_precision, resolve_roformer_model_dir

    root = resolve_roformer_model_dir(settings.get("roformer_model_dir"))
    package = Path(__file__).resolve().parent
    probe = ffprobe_bin()
    signature = {
        "source": cache.fingerprint(source),
        "input_format": source.suffix.lower(),
        "settings": settings,
        "roformer_precision": requested_roformer_precision(),
        "ffmpeg": cache.fingerprint(Path(ffmpeg)),
        "ffprobe": cache.fingerprint(Path(probe)) if probe else None,
        "models": [model_artifacts(path) for path in (root, models_dir() / "speechbrain",
                    huggingface_hub_cache(), Path.cwd() / "checkpoints")],
        "code": {path.name: cache.fingerprint(path) for path in sorted(package.glob("*.py"))},
        "versions": versions("audio-separator", "torch", "torchaudio", "onnxruntime", "pyannote.audio",
                             "librosa", "soundfile", "numpy", "scipy", "anime-speaker-embedding",
                             "clearvoice", "speechbrain"),
        "environment": inference_environment(fast=True),
    }

    def produce(directory):
        build(directory)
        report, = directory.glob("*_split_report.json")
        with report.open(encoding="utf-8") as file:
            payload = json.load(file)
        metadata = {"root": str(directory), "audio_sec": payload["stats"]["vocals_duration_sec"]}
        (directory / ".bundle-origin").write_text(json.dumps(metadata), encoding="utf-8")

    artifact = cache.files(
        "multi_stem_result", signature, None, produce,
        validate=lambda path: (path / ".bundle-origin").is_file() and any(path.glob("*_split_report.json")),
    )
    metadata = json.loads((artifact / ".bundle-origin").read_text(encoding="utf-8"))
    record_audio_seconds(metadata["audio_sec"])
    with stage("publish_cached_result"):
        copy_bundle(artifact, destination, Path(metadata["root"]))
