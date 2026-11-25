#!/usr/bin/env python3
"""Compare pre-change and current stem renderers in isolated, alternating processes."""
from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from video2text.performance import _rss_bytes

SR = 44100


def prepare(source, path, frames):
    clip, sr = sf.read(source, dtype="float32", always_2d=True)
    if sr != SR:
        raise ValueError("基准输入需要 44.1 kHz WAV")
    mono = clip.mean(axis=1)
    with sf.SoundFile(path, "w", SR, 1, subtype="FLOAT") as out:
        for _ in range(frames // len(mono)):
            out.write(mono)
        out.write(mono[:frames % len(mono)])


def digest_wav(path):
    digest = sha256()
    with sf.SoundFile(path) as source:
        for block in source.blocks(blocksize=1 << 18, dtype="float32"):
            digest.update(block.tobytes())
    return digest.hexdigest()


def worker(input_path, baseline):
    if baseline is None:
        from video2text.speaker_stems import build_speaker_stems
    else:
        spec = importlib.util.spec_from_file_location("stems_before", baseline)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        build_speaker_stems = module.build_speaker_stems
    import librosa

    # Load numerical libraries before starting the timed section.
    librosa.load
    frames = sf.info(input_path).frames
    speakers = ["a", "b", "c", "d"]
    assignments = [
        {"start": i / SR, "end": min(i + 2 * SR, frames) / SR,
         "speaker_id": speakers[(i // (2 * SR)) % 4], "confidence": .75}
        for i in range(0, frames, 2 * SR)
    ]
    with TemporaryDirectory(prefix="video2text-bench-output-") as temp:
        directory = Path(temp)
        outputs = {s: directory / f"{s}.wav" for s in speakers}
        review = directory / "review.wav"
        started = perf_counter()
        stats = build_speaker_stems(input_path, assignments, outputs, review_path=review,
                                   fill_unassigned=False)
        elapsed = perf_counter() - started
        return {
            "wall_sec": round(elapsed, 6), "peak_rss_bytes": _rss_bytes(),
            "pcm_sha256": [digest_wav(p) for p in (*outputs.values(), review)], "stats": stats,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--baseline", type=Path, help="优化前的 speaker_stems.py")
    parser.add_argument("--seconds", type=int, default=22)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.input, args.baseline)))
        return
    if args.baseline is None or args.pairs < 1 or args.seconds < 1:
        parser.error("需要 --baseline、正的 --pairs 和 --seconds")
    observations = []
    with TemporaryDirectory(prefix="video2text-bench-input-") as temp:
        audio = Path(temp) / "fixture.wav"
        prepare(args.input, audio, args.seconds * SR)
        for pair in range(args.pairs):
            ordered = ("before", "after") if pair % 2 == 0 else ("after", "before")
            result = {}
            for variant in ordered:
                command = [sys.executable, str(Path(__file__).resolve()), str(audio), "--worker"]
                if variant == "before":
                    command.extend(("--baseline", str(args.baseline)))
                data = json.loads(subprocess.check_output(command, text=True))
                print(pair + 1, variant, data["wall_sec"], data["peak_rss_bytes"], flush=True)
                result[variant] = data
            for key in ("pcm_sha256", "stats"):
                if result["before"][key] != result["after"][key]:
                    raise SystemExit(f"输出差异: {key}, 配对 {pair + 1}")
            observations.append(result)
    data = {"input": str(args.input), "seconds": args.seconds, "paired_runs": observations,
            "same_pcm_and_stats": True}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
