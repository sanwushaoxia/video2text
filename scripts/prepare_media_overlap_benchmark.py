#!/usr/bin/env python3
"""建立多语言参考混音，并导入具有人工 RTTM 的真实影视片段。"""
from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd

from prepare_generalization_benchmark import DiarizationTurn, trim_turns, write_fixture

SR = 16000


def normalized_mono(source):
    audio, sr = sf.read(source, dtype="float32", always_2d=True)
    factor = gcd(sr, SR)
    audio = resample_poly(audio.mean(axis=1), SR // factor, sr // factor)
    audio = audio[:4 * SR]
    return audio * (0.045 / max(float(np.sqrt(np.mean(audio**2))), 1e-6))


def make_case(root, name, language, partition, voices, seed, provenance):
    sr = SR
    tail = sr
    step = sr // 5
    start = sum(len(anchor) + step for _, anchor, _ in voices) + step
    n = start + max(i * step + len(utterance) for i, (_, _, utterance) in enumerate(voices)) + 2 * tail
    references, turns = {}, []
    cursor = 0
    for i, (speaker, anchor, speech) in enumerate(voices):
        offset = start + i * step
        waveform = np.zeros(n, dtype=np.float32)
        waveform[cursor:cursor + len(anchor)] = anchor
        waveform[offset:offset + len(speech)] = speech * (1, 0.58, 0.75)[i]
        if seed % 2:
            delay = round(0.043 * sr)
            waveform[delay:] += 0.20 * waveform[:-delay]
        turns.extend([
            DiarizationTurn(cursor / sr, (cursor + len(anchor)) / sr, speaker),
            DiarizationTurn(offset / sr, (offset + len(speech)) / sr, speaker),
        ])
        references[speaker] = waveform
        cursor += len(anchor) + step

    rng = np.random.default_rng(seed)
    t = np.arange(n) / sr
    music = sum(np.sin(2 * np.pi * f * t) for f in (131, 196, 262))
    bg = 0.009 * music + rng.normal(0, 0.002, n)
    event_positions = (start + step, n - tail)
    for at in event_positions:
        burst = 0.07 * np.exp(-np.arange(tail) / (0.07 * sr)) * rng.normal(size=tail)
        bg[at:at + tail] += burst
    background = np.column_stack((bg, np.roll(bg, 11))).astype(np.float32)
    mixture = sum(references.values())[:, None] + background
    metadata = {
        "group": name, "partition": partition, "kind": "controlled", "language": language,
        "provenance": provenance, "speakers": len(voices),
        "effect_windows": [[at / sr, at / sr + 0.25] for at in event_positions],
    }
    return write_fixture(root, metadata, name, mixture, sr, turns, refs=references, background=background)


def import_scene(root, item, base):
    start, end = float(item["start"]), float(item["end"])
    if (not np.isfinite(start + end) or start < 0 or end <= start
            or item["partition"] not in ("dev", "holdout")):
        raise ValueError(f"无效影视区间: {item['id']}")
    if not item["id"].replace("_", "").replace("-", "").isalnum():
        raise ValueError("片段 ID 只允许字母、数字、下划线和连字符")
    turns = []
    for line in (base / item["rttm"]).read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields and fields[0] == "SPEAKER" and fields[1] == item["rttm_recording"]:
            left, length = map(float, fields[3:5])
            if not np.isfinite(left + length) or left < 0 or length <= 0:
                raise ValueError(f"非法 RTTM 时间: {line}")
            if fields[7] not in item["speakers"]:
                raise ValueError(f"缺少跨作品可核对的演员 ID: {fields[7]}")
            turns.append(DiarizationTurn(left, left + length, item["speakers"][fields[7]]))
    clipped = trim_turns(turns, start, end)
    if not clipped:
        raise ValueError(f"{item['id']} 无有效人工标注")
    with TemporaryDirectory() as temp:
        wav = Path(temp) / "scene.wav"
        subprocess.run([
            "ffmpeg", "-v", "error", "-ss", str(start), "-i", str(base / item["media"]),
            "-t", str(end - start), "-vn", "-ac", "2", "-ar", str(SR), str(wav),
        ], check=True)
        audio, _ = sf.read(wav, dtype="float32")
    metadata = {k: item[k] for k in ("group", "partition", "language", "provenance")}
    return write_fixture(root, dict(metadata, kind="real", source_start_sec=start),
                         item["id"], audio, SR, clipped)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--jvs-demo", type=Path)
    parser.add_argument("--mandarin", type=Path, nargs=3,
                        help="AISHELL-3 官方补充页的 raw1、raw2、raw3")
    parser.add_argument("--scenes", type=Path, help="有独立人工 RTTM 的影视片段 JSON")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    if args.jvs_demo:
        with ZipFile(args.jvs_demo) as archive:
            for partition, speakers, seed in (("dev", (1, 2), 30), ("holdout", (3, 4), 31)):
                voices = []
                for index in speakers:
                    speaker = f"jvs{index:03}"
                    names = sorted(name for name in archive.namelist()
                                   if f"/{speaker}/" in name and name.endswith(".wav")
                                   and "whisper" not in name and "falsetto" not in name)
                    voices.append((speaker, *(normalized_mono(BytesIO(archive.read(name)))
                                              for name in (names[0], names[-1]))))
                cases.append(make_case(args.out_dir, f"jvs_{partition}", "ja", partition, voices, seed,
                                       "https://github.com/MontrealCorpusTools/japanese-jvs-demo"))
    if args.mandarin:
        voices = []
        for index, path in enumerate(args.mandarin, 1):
            signal = normalized_mono(path)
            middle = signal.size // 2
            voices.append((f"aishell3_sample_{index}", signal[:middle], signal[middle:]))
        cases.append(make_case(args.out_dir, "aishell3_holdout", "zh", "holdout", voices, 51,
                               "https://sos1sos2sixteen.github.io/aishell3v2/"))
    if args.scenes:
        data = json.loads(args.scenes.read_text(encoding="utf-8"))
        cases.extend(import_scene(args.out_dir, item, args.scenes.parent) for item in data["scenes"])
    if not cases:
        parser.error("至少提供一种输入")
    from evaluate_overlap_generalization import validate_cases
    validate_cases(cases)
    manifest = args.out_dir / "manifest.json"
    manifest.write_text(json.dumps({"cases": cases}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{manifest} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
