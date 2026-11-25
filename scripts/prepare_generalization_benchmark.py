#!/usr/bin/env python3
"""从 AMI 官方人工标注生成分组留出的短录音和有参考分轨的混音。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from video2text.diarization import DiarizationTurn, detect_overlap_regions


def read_turns(archive: ZipFile, meeting: str) -> list[DiarizationTurn]:
    member = next((name for name in archive.namelist()
                   if "/only_words/rttms/" in name and name.endswith(f"/{meeting}.rttm")), None)
    if member is None:
        raise ValueError(f"AMI 标注包缺少 {meeting} 的分轨时间标注")
    turns = []
    for line in archive.read(member).decode().splitlines():
        fields = line.split()
        start, duration = map(float, fields[3:5])
        turns.append(DiarizationTurn(start, start + duration, f"{meeting}:{fields[7]}"))
    return sorted(turns, key=lambda turn: turn.start)


def trim_turns(turns, start, end):
    return [
        DiarizationTurn(max(t.start, start) - start, min(t.end, end) - start, t.speaker_id)
        for t in turns if t.start < end and t.end > start
    ]


def real_windows(turns, length):
    # 固定采样网格。各取一个含短叠话和频繁叠话的窗口。
    duration = 16
    candidates = []
    for start in range(16, int(length) - duration, duration):
        clipped = trim_turns(turns, start, start + duration)
        regions = detect_overlap_regions(clipped, min_overlap_sec=0.15)
        seconds = sum(region.end - region.start for region in regions)
        if seconds >= 0.8 and len({t.speaker_id for t in clipped}) >= 2:
            candidates.append((seconds, start, clipped))
    if len(candidates) < 2:
        raise ValueError("无法提取两个独立的真实叠话窗口")
    candidates.sort(key=lambda item: item[:2])
    for index in (len(candidates) // 4, 3 * len(candidates) // 4):
        _, start, clipped = candidates[index]
        yield start, duration, clipped


def solo_examples(turns):
    overlaps = detect_overlap_regions(turns, min_overlap_sec=0)
    per_speaker = {}
    for turn in turns:
        if turn.end - turn.start < 2.0:
            continue
        end = min(turn.end, turn.start + 4.0)
        if any(r.start < end and r.end > turn.start for r in overlaps):
            continue
        per_speaker.setdefault(turn.speaker_id, []).append((turn.start, end))
    return {speaker: spans for speaker, spans in per_speaker.items() if len(spans) >= 2}


def write_fixture(root, metadata, name, audio, sr, turns, *, refs=None, background=None):
    directory = root / name
    directory.mkdir(exist_ok=True)
    sf.write(directory / "mix.wav", audio, sr, subtype="FLOAT")
    entry = {
        **metadata, "id": name, "audio": f"{name}/mix.wav",
        "turns": [{"start": t.start, "end": t.end, "speaker_id": t.speaker_id} for t in turns],
    }
    if refs:
        entry["references"] = {}
        for index, (speaker, ref) in enumerate(refs.items()):
            filename = f"speaker_{index}.wav"
            sf.write(directory / filename, ref, sr, subtype="FLOAT")
            entry["references"][speaker] = f"{name}/{filename}"
        sf.write(directory / "background.wav", background, sr, subtype="FLOAT")
        entry["background"] = f"{name}/background.wav"
    return entry


def controlled_mix(audio, sr, examples, speakers, seed):
    rng = np.random.default_rng(seed)
    clips = []
    for speaker, (anchor, speech) in examples[:speakers]:
        segments = [audio[round(a * sr):round(b * sr)].copy() for a, b in (anchor, speech)]
        clips.append((speaker, *(x * (0.05 / max(np.sqrt(np.mean(x**2)), 1e-7)) for x in segments)))
    overlap_start = sum(len(anchor) + sr // 5 for _, anchor, _ in clips) + sr // 3
    length = overlap_start + max(len(speech) + i * (sr // 7) for i, (_, _, speech) in enumerate(clips))
    turns, refs = [], {}
    offset = 0
    for index, (speaker, anchor, speech) in enumerate(clips):
        start = overlap_start + index * (sr // 7)
        ref = np.zeros(length, dtype=np.float32)
        ref[offset:offset + len(anchor)] = anchor
        amplitude = (1.0, 0.65, 0.8)[index]
        ref[start:start + len(speech)] = speech * amplitude
        if speakers == 3:
            delay = round(0.037 * sr)
            ref[delay:] += 0.18 * ref[:-delay]
        turns.extend([
            DiarizationTurn(offset / sr, (offset + len(anchor)) / sr, speaker),
            DiarizationTurn(start / sr, (start + len(speech)) / sr, speaker),
        ])
        offset += len(anchor) + sr // 5
        refs[speaker] = ref
    time = np.arange(length) / sr
    tones = sum(np.sin(2 * np.pi * hz * time) for hz in (110, 165, 220))
    background = np.column_stack((tones, np.roll(tones, 10))).astype(np.float32) * 0.014
    background += rng.normal(0, 0.005, background.shape).astype(np.float32)
    return sum(refs.values())[:, None] + background, turns, refs, background


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--rttm-zip", required=True, type=Path)
    parser.add_argument("--dev", required=True, type=Path, help="ES2004a Mix-Headset WAV")
    parser.add_argument("--holdout", required=True, type=Path, help="EN2002a Mix-Headset WAV")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    seen_speakers = set()
    with ZipFile(args.rttm_zip) as archive:
        for partition, meeting, path in (
            ("dev", "ES2004a", args.dev), ("holdout", "EN2002a", args.holdout),
        ):
            audio, sr = sf.read(path, dtype="float32", always_2d=True)
            audio = audio.mean(axis=1)
            turns = read_turns(archive, meeting)
            speakers = {turn.speaker_id.split(":", 1)[1] for turn in turns}
            if seen_speakers & speakers:
                raise ValueError("开发会议和留出会议共享说话人")
            seen_speakers.update(speakers)
            metadata = {"group": meeting, "partition": partition, "language": "en"}
            for i, (start, duration, clipped) in enumerate(real_windows(turns, len(audio) / sr)):
                signal = audio[start * sr:(start + duration) * sr]
                entries.append(write_fixture(
                    args.out_dir, dict(metadata, kind="real", source_start_sec=start),
                    f"{meeting}_real_{i}", signal, sr, clipped,
                ))
            examples = [(speaker, spans[:2]) for speaker, spans in sorted(solo_examples(turns).items())]
            if len(examples) < 3:
                raise ValueError(f"{meeting} 无法从人工标注提取三位说话人的纯独白")
            for n in (2, 3):
                mix, labeled, refs, bg = controlled_mix(audio, sr, examples, n, seed=n)
                entries.append(write_fixture(
                    args.out_dir, dict(metadata, kind="controlled", speakers=n),
                    f"{meeting}_mix_{n}", mix, sr, labeled, refs=refs, background=bg,
                ))
    manifest = args.out_dir / "manifest.json"
    manifest.write_text(json.dumps({"cases": entries}, indent=2) + "\n", encoding="utf-8")
    print(f"{manifest} ({len(entries)} cases)")


if __name__ == "__main__":
    main()
