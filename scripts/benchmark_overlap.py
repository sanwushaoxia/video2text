#!/usr/bin/env python3
"""生成固定的双人语音混音，测量叠话分离和可选的背景分轨误差。"""
from __future__ import annotations

import argparse
from itertools import permutations
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from video2text.overlap_separation import separate_overlap_2spk

SAMPLE_RATE = 44100


def synthesize(out_dir: Path, voice: str, text: str) -> np.ndarray:
    path = out_dir / f"{voice}.wav"
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
        f"flite=text={text}:voice={voice}", "-ar", str(SAMPLE_RATE), str(path),
    ], check=True)
    audio, _ = sf.read(path, dtype="float32")
    return audio * (0.04 / np.sqrt(np.mean(audio**2)))


def make_fixture(out_dir: Path):
    a = synthesize(out_dir, "slt", "We need to get the train before the end of the day. The blue tickets are on the table.")
    b = synthesize(out_dir, "awb", "Yesterday I went into the forest and found a small house next to the river.")
    start = SAMPLE_RATE // 2
    end = min(len(a), start + len(b))
    length = max(len(a), start + len(b))
    a = np.pad(a, (0, length - len(a)))
    b = np.pad(b, (start, length - start - len(b)))
    time = np.arange(length) / SAMPLE_RATE
    music = sum(0.01 * np.sin(2 * np.pi * hz * time) for hz in (110, 139, 165))
    background = np.column_stack((music, np.roll(music, 35)))
    mixture = (a + b)[:, None] + background
    for name, signal in (("ref_a", a), ("ref_b", b), ("ref_bg", background), ("mix_bg", mixture)):
        sf.write(out_dir / f"{name}.wav", signal, SAMPLE_RATE)
    return (a[start:end], b[start:end]), mixture, background


def si_sdr(estimate: np.ndarray, reference: np.ndarray) -> float:
    estimate = estimate - estimate.mean()
    reference = reference - reference.mean()
    target = reference * (np.dot(estimate, reference) / np.dot(reference, reference))
    noise = estimate - target
    return float(10 * np.log10((np.dot(target, target) + 1e-15) / (np.dot(noise, noise) + 1e-15)))


def evaluate_background(directory: Path, mixture: np.ndarray, reference: np.ndarray) -> dict:
    def load(stem):
        audio, sr = sf.read(directory / f"mix_bg_{stem}.wav", always_2d=True)
        if sr != SAMPLE_RATE or audio.shape != mixture.shape:
            raise ValueError(f"{directory.name}/{stem} 的声道、采样率或样本数不匹配")
        return audio

    def measure(estimate):
        speech = mixture - reference
        residual = estimate - reference
        projection = np.sum(residual * speech) / np.sum(speech**2)
        return {
            "background_snr_db": round(float(10 * np.log10(
                np.sum(reference**2) / max(np.sum(residual**2), 1e-15)
            )), 2),
            "speech_leakage_db": round(float(20 * np.log10(max(abs(projection), 1e-8))), 2),
        }

    return {
        "direct": measure(load("no_vocals")),
        "mixture_minus_vocals": measure(mixture - load("vocals")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--backend", choices=("auto", "mossformer2", "sepformer"), default="sepformer")
    parser.add_argument("--background-dir", type=Path, action="append", default=[])
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (a, b), mixture, background = make_fixture(args.out_dir)

    rows = []
    gains = []
    with TemporaryDirectory() as work:
        for factor in (0.5, 1.0, 2.0):
            refs = (a, b * factor)
            clip = refs[0] + refs[1]
            *sources, backend = separate_overlap_2spk(
                clip, SAMPLE_RATE, Path(work) / str(factor), backend=args.backend,
            )
            baseline = [si_sdr(clip, ref) for ref in refs]
            # 这里用参考音轨确定排列，单独测量分离质量。
            separated = max(
                ([si_sdr(x, ref) for x, ref in zip(order, refs)] for order in permutations(sources)),
                key=sum,
            )
            delta = [after - before for before, after in zip(baseline, separated)]
            gains.extend(delta)
            rows.append({
                "speaker_b_gain": factor, "backend": backend,
                "mixture_si_sdr_db": [round(x, 2) for x in baseline],
                "separated_si_sdr_db": [round(x, 2) for x in separated],
                "improvement_db": [round(x, 2) for x in delta],
            })
    result = {
        "median_overlap_improvement_db": round(float(np.median(gains)), 2),
        "overlap": rows,
        "background": {str(d): evaluate_background(d, mixture, background) for d in args.background_dir},
    }
    output = json.dumps(result, indent=2, ensure_ascii=False)
    (args.out_dir / "benchmark.json").write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
