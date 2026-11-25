#!/usr/bin/env python3
"""把人工指定的重叠声源归属写入新副本，仅改动选中的样本区间。"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf


def apply_corrections(report_path: Path, corrections: dict, output: Path) -> Path:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    paths = report["speaker_tracks"]
    review_path = Path(report["dialogue_review"])
    bss = report["overlap_bss"]
    sr = bss["sample_rate"]
    by_id = {row["id"]: row for row in bss["regions"] if "preview" in row}
    if not isinstance(corrections, dict) or not corrections:
        raise ValueError("请至少指定一个窗口的声源归属")
    if output.exists():
        raise FileExistsError(f"输出目录已存在: {output}")
    info = sf.info(review_path)
    edits = []
    for key, mapping in corrections.items():
        if key not in by_id:
            raise ValueError(f"窗口缺少可用声源缓存: {key}")
        row = by_id[key]
        if (not isinstance(mapping, list) or not all(isinstance(s, str) for s in mapping)
                or len(mapping) != len(set(mapping)) or not set(mapping).issubset(paths)):
            raise ValueError(f"{key} 的声源归属必须是一组不重复的现有角色 ID")
        left, right = row["start_sample"], row["end_sample"]
        preview = sf.info(row["preview"])
        if (not 0 <= left < right <= info.frames or preview.samplerate != sr
                or preview.frames != right - left or preview.channels != len(mapping)):
            raise ValueError(f"{key} 的缓存长度、采样率或声源数量与归属不符")
        edits.append((left, right, mapping, row))
    edits.sort(key=lambda item: item[0])
    if any(a[1] > b[0] for a, b in zip(edits, edits[1:])):
        raise ValueError("所选修正区间彼此重叠")

    originals = {speaker: Path(file) for speaker, file in paths.items()}
    originals[None] = review_path
    names = [path.name for path in originals.values()]
    if len(set(names)) != len(names):
        raise ValueError("输出轨文件名冲突")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".overlap-correction-", dir=output.parent) as temp:
        staged = Path(temp) / "bundle"
        staged.mkdir()
        with ExitStack() as stack:
            files = {}
            for speaker, original in originals.items():
                copy = staged / original.name
                copyfile(original, copy)
                file = stack.enter_context(sf.SoundFile(copy, mode="r+"))
                if (file.frames, file.samplerate, file.channels) != (info.frames, sr, 1):
                    raise ValueError(f"输出源音轨时间轴不一致: {original}")
                files[speaker] = file
            for left, right, mapping, row in edits:
                sources, _ = sf.read(row["preview"], dtype="float32", always_2d=True)
                zero = np.zeros(right - left, dtype=np.float32)
                assigned = dict(zip(mapping, sources.T))
                for speaker, file in files.items():
                    file.seek(left)
                    file.write(assigned.get(speaker, zero))
                row.update(status="separated", source_speakers=mapping, assignment="manual")
                row.pop("reason", None)

        report["speaker_tracks"] = {speaker: str(output.resolve() / file.name)
                                    for speaker, file in originals.items() if speaker is not None}
        report["dialogue_review"] = str(output.resolve() / review_path.name)
        accepted = [row for row in bss["regions"] if row["status"] == "separated"]
        bss["overlap_count"] = len(accepted)
        bss["overlap_separated_sec"] = round(sum(r["end_sample"] - r["start_sample"] for r in accepted) / sr, 2)
        bss["backend_used"] = ", ".join(sorted({r["backend"] for r in accepted if r.get("backend")})) or None
        for entry in report.get("segments", []):
            entry["overlap_separated"] = any(
                r["start_sample"] < float(entry["end"]) * sr and r["end_sample"] > float(entry["start"]) * sr
                for r in accepted
            )
        if "stats" in report:
            report["stats_before_manual_correction"] = report.pop("stats")
        report["manual_corrections"] = {**report.get("manual_corrections", {}), **corrections}
        staged_report = staged / report_path.name
        staged_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        staged.rename(output)
    return output / report_path.name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("corrections", type=Path, help='JSON 例如 {"0:0": ["SPEAKER_01", "SPEAKER_00"]}')
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(apply_corrections(args.report, json.loads(args.corrections.read_text()), args.out_dir))
    except (KeyError, OSError, ValueError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
