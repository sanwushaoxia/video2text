#!/usr/bin/env python3
"""比较背景轨保真度并生成响度匹配的盲听背景及替换配音样片。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import re
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf

from evaluate_overlap_generalization import read_aligned


def integrated_loudness(waveform, sr):
    with TemporaryDirectory() as temp:
        path = Path(temp) / "measure.wav"
        sf.write(path, waveform.T, sr, subtype="FLOAT")
        proc = subprocess.run([
            "ffmpeg", "-hide_banner", "-i", str(path), "-af", "loudnorm=print_format=json",
            "-f", "null", "-",
        ], capture_output=True, check=True, text=True)
    value = float(json.loads(re.search(r'\{[^{}]*"input_i"[^{}]*\}', proc.stderr, re.S).group())["input_i"])
    return value if np.isfinite(value) else None


def equalize(audios, sr, target=-24.0):
    loudness = [integrated_loudness(audio, sr) for audio in audios]
    gains = [10 ** ((target - value) / 20) if value is not None else 1.0 for value in loudness]
    levels = [audio * gain for audio, gain in zip(audios, gains)]
    # One gain for both alternatives preserves their listening level match after mixing.
    peak = max(float(np.max(np.abs(level))) for level in levels)
    attenuation = min(1.0, 0.95 / max(peak, 1e-8))
    return [level * attenuation for level in levels], loudness


def snr(reference, error):
    return float(10 * np.log10(max(np.sum(reference**2), 1e-12) / max(np.sum(error**2), 1e-12)))


def measure_background(estimate, reference, voices, effects, transitions, sr):
    difference = estimate - reference
    matrix = np.column_stack([voice.ravel() for voice in voices])
    coefficients, *_ = np.linalg.lstsq(matrix, difference.ravel(), rcond=1e-4)
    leakage = matrix @ coefficients
    leakage_db = -snr(matrix.sum(axis=1), leakage)
    effect_scores = []
    for left, right in effects:
        part = slice(round(left * sr), round(right * sr))
        goal = np.diff(reference[:, part], axis=-1)
        error = np.diff(difference[:, part], axis=-1)
        effect_scores.append({
            "start": left, "end": right, "transient_snr_db": round(snr(goal, error), 2),
        })
    # Difference from the reference sample step at annotated dialogue edit points.
    steps = []
    for t in sorted(set(transitions)):
        i = round(t * sr)
        if 0 < i < reference.shape[1]:
            steps.append(float(np.max(np.abs(difference[:, i] - difference[:, i - 1]))))
    return {
        "background_snr_db": round(snr(reference, difference), 2),
        "projected_dialogue_leakage_db": round(leakage_db, 2),
        "effect_windows": effect_scores,
        "max_boundary_step_error": round(max(steps, default=0), 6),
    }


def replacement_track(path, case, sr, frames, channels):
    if path is None:
        return None
    import librosa

    source, _ = librosa.load(path, sr=sr, mono=True)
    track = np.zeros(frames, dtype=np.float32)
    turns = sorted(case["turns"], key=lambda turn: turn["start"])
    for turn in turns[:2]:
        left, right = round(turn["start"] * sr), min(frames, round(turn["end"] * sr))
        count = min(len(source), right - left)
        if count <= 0:
            continue
        clip = source[:count].copy()
        fade = min(round(0.02 * sr), count // 2)
        if fade:
            ramp = np.linspace(0, 1, fade)
            clip[:fade] *= ramp
            clip[-fade:] *= ramp[::-1]
        track[left:left + count] += clip
    goal = np.sqrt(np.mean(track**2))
    track *= 0.04 / max(goal, 1e-5)
    return np.repeat(track[None, :], channels, axis=0)


def render_video(video, audio, start, output):
    duration = sf.info(audio).duration
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-ss", str(start), "-i", str(video), "-i", str(audio),
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", "veryfast",
        "-c:a", "aac", "-t", str(duration), "-shortest", str(output),
    ], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--replacement", type=Path, help="用于试听的独立替换配音 WAV")
    parser.add_argument("--video", type=Path, help="单片段清单的原视频，同时生成 A/B 配音视频")
    parser.add_argument("--seed", type=int, default=581)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    root = args.manifest.resolve().parent
    cases = json.loads(args.manifest.read_text(encoding="utf-8"))["cases"]
    if args.video is not None and (len(cases) != 1 or args.replacement is None):
        parser.error("--video 需要一条片段记录及 --replacement")
    rng = random.Random(args.seed)
    key, measurements = {}, {}
    with (args.out_dir / "listen.csv").open("w", newline="", encoding="utf-8") as csv_file:
        rows = csv.writer(csv_file)
        rows.writerow(["case", "blind_id", "背景对白残留_0无3明显", "音效缺失_0无3明显", "配音切点_0自然3突兀", "备注"])
        for case in cases:
            mix_path = root / case["audio"]
            info = sf.info(mix_path)
            sr, frames = info.samplerate, info.frames

            def read(path):
                return read_aligned(path, sr, frames, mono=False)

            mix, vocals = read(mix_path), read(mix_path.with_name("vocals.wav"))
            variants = [
                ("direct", read(mix_path.with_name("estimated_background.wav"))),
                ("mixture_minus_vocals", mix - vocals),
            ]
            if "background" in case:
                reference = read(root / case["background"])
                refs = [np.repeat(read_aligned(root / path, sr, frames, mono=True)[None, :],
                                  reference.shape[0], axis=0)
                        for path in case["references"].values()]
                transitions = [t[k] for t in case["turns"] for k in ("start", "end")]
                measurements[case["id"]] = {
                    name: measure_background(
                        signal, reference, refs, case.get("effect_windows", []), transitions, sr,
                    )
                    for name, signal in variants
                }
            rng.shuffle(variants)
            signals = [signal for _, signal in variants]
            backgrounds, _ = equalize(signals, sr)
            replacement = replacement_track(args.replacement, case, sr, mix.shape[1], mix.shape[0])
            dubbed = None
            if replacement is not None:
                peak = max(float(np.max(np.abs(level + replacement))) for level in backgrounds)
                attenuation = min(1.0, 0.95 / max(peak, 1e-8))
                dubbed = [(level + replacement) * attenuation for level in backgrounds]
            directory = args.out_dir / case["id"]
            directory.mkdir(exist_ok=True)
            key[case["id"]] = {}
            for index, letter in enumerate(("A", "B")):
                sf.write(directory / f"{letter}_background.wav", backgrounds[index].T, sr, subtype="FLOAT")
                if dubbed is not None:
                    dub_path = directory / f"{letter}_replacement.wav"
                    sf.write(dub_path, dubbed[index].T, sr, subtype="FLOAT")
                    if args.video is not None:
                        render_video(args.video, dub_path, case.get("source_start_sec", 0),
                                     directory / f"{letter}_replacement.mp4")
                key[case["id"]][letter] = variants[index][0]
                rows.writerow([case["id"], letter, "", "", "", ""])
    for name, data in (("answer_key.json", key), ("objective_metrics.json", measurements)):
        (args.out_dir / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.out_dir / "listen.csv")


if __name__ == "__main__":
    main()
