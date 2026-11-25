#!/usr/bin/env python3
"""用固定清单比较人工边界、提取人声与原混音的叠话分离。"""
from __future__ import annotations

import argparse
from itertools import permutations
import json
from pathlib import Path
from shutil import copyfile
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import soundfile as sf
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from video2text.diarization import (
    DiarizationResult,
    DiarizationTurn,
    OverlapRegion,
    detect_overlap_regions,
)
from video2text.overlap_separation import separate_overlap_2spk, separate_overlap_3spk
from video2text.speaker_stems import split_vocals_by_speaker
from video2text.speaker_embedding import DEFAULT_ANIME_EMBEDDING_MODEL
from benchmark_overlap import si_sdr


FRAMES_PER_SECOND = 100
OUTPUT_SAMPLE_RATE = 44100
SILENCE_SCORE_DB = -40.0


def validate_cases(cases):
    partitions = {}
    ids = set()
    for case in cases:
        if case["id"] in ids:
            raise ValueError(f"重复评测 ID: {case['id']}")
        ids.add(case["id"])
        if case["partition"] not in ("dev", "holdout"):
            raise ValueError(f"未知集合: {case['partition']}")
        tags = {("group", case["group"])}
        tags.update(("speaker", t["speaker_id"]) for t in case["turns"])
        for tag in tags:
            previous = partitions.setdefault(tag, case["partition"])
            if previous != case["partition"]:
                raise ValueError(f"开发集和留出集共享录音或说话人: {tag[1]}")


def activity_matrix(turns, frames):
    names = sorted({t.speaker_id for t in turns})
    matrix = np.zeros((frames, len(names)), dtype=bool)
    speaker_index = {speaker: i for i, speaker in enumerate(names)}
    for t in turns:
        start = max(0, round(t.start * FRAMES_PER_SECOND))
        end = min(frames, round(t.end * FRAMES_PER_SECOND))
        matrix[start:end, speaker_index[t.speaker_id]] = True
    return names, matrix


def region_mask(regions, frames):
    mask = np.zeros(frames, dtype=bool)
    for start, end in regions:
        left = max(0, round(start * FRAMES_PER_SECOND))
        right = min(frames, round(end * FRAMES_PER_SECOND))
        mask[left:right] = True
    return mask


def compare_timeline(reference, estimate, frames):
    names, ref = activity_matrix(reference, frames)
    predicted, hyp = activity_matrix(estimate, frames)
    ref_count = ref.sum(axis=1)
    hyp_count = hyp.sum(axis=1)
    shared = ref.astype(np.int32).T @ hyp.astype(np.int32)
    rows, cols = linear_sum_assignment(shared, maximize=True)
    mapping = {names[r]: predicted[c] for r, c in zip(rows, cols)}
    correct = (ref[:, rows] & hyp[:, cols]).sum(axis=1)
    wrong = np.minimum(ref_count, hyp_count) - correct
    counts = {
        "speaker_reference_frames": int(ref_count.sum()),
        "speaker_confusion_frames": int(wrong.sum()),
        "speaker_missed_frames": int(np.maximum(ref_count - hyp_count, 0).sum()),
        "speaker_false_alarm_frames": int(np.maximum(hyp_count - ref_count, 0).sum()),
    }
    return mapping, counts


def tally_overlap(reference, prediction, metadata, frames):
    mask = region_mask([(r.start, r.end) for r in reference], frames)
    hyp_mask = region_mask([(r.start, r.end) for r in prediction], frames)
    regions = metadata["overlap_bss"]["regions"]
    sr = metadata["overlap_bss"]["sample_rate"]

    def status_mask(status):
        return region_mask(
            [(item["start_sample"] / sr, item["end_sample"] / sr)
             for item in regions if item["status"] == status],
            frames,
        )

    separated = status_mask("separated")
    review = status_mask("review") & ~separated
    return {
        "overlap_reference_frames": int(mask.sum()),
        "overlap_detected_frames": int(hyp_mask.sum()),
        "overlap_detected_true_frames": int((mask & hyp_mask).sum()),
        "overlap_separated_true_frames": int((mask & separated).sum()),
        "overlap_separated_false_frames": int((~mask & separated).sum()),
        "overlap_review_true_frames": int((mask & review).sum()),
    }


def read_aligned(path, rate, frames, *, mono):
    import librosa

    audio, _ = librosa.load(path, sr=rate, mono=mono)
    if abs(audio.shape[-1] - frames) > round(0.05 * rate):
        raise ValueError(f"参考和输出长度不匹配: {path}")
    audio = librosa.util.fix_length(audio, size=frames)
    return audio if mono else np.atleast_2d(audio)


def overlap_sdr(case, directory, outputs, mapping, waveform_size, regions):
    mask = np.zeros(waveform_size, dtype=bool)
    for region in regions:
        mask[round(region.start * OUTPUT_SAMPLE_RATE):round(region.end * OUTPUT_SAMPLE_RATE)] = True
    mixture = read_aligned(
        directory / case["audio"], OUTPUT_SAMPLE_RATE, waveform_size, mono=True,
    )[mask]
    rows = []
    for speaker, path in case["references"].items():
        ref = read_aligned(directory / path, OUTPUT_SAMPLE_RATE, waveform_size, mono=True)[mask]
        if not np.any(ref):
            continue
        track = outputs.get(mapping.get(speaker))
        estimate = (
            read_aligned(track, OUTPUT_SAMPLE_RATE, waveform_size, mono=True)[mask]
            if track is not None else np.zeros_like(ref)
        )
        score = si_sdr(estimate, ref) if np.linalg.norm(estimate) > 1e-7 else SILENCE_SCORE_DB
        rows.append({
            "speaker": speaker,
            "mixture_si_sdr_db": round(si_sdr(mixture, ref), 3),
            "track_si_sdr_db": round(max(SILENCE_SCORE_DB, score), 3),
        })
    return rows


def oracle_source_quality(case, directory, vocals, regions, backend, work):
    """用人工边界和最佳排列单独测量波形分离，排除检测与角色归属的影响。"""
    import librosa

    sr = OUTPUT_SAMPLE_RATE
    audio, _ = librosa.load(vocals, sr=sr, mono=True)
    refs = {speaker: read_aligned(directory / file, sr, audio.size, mono=True)
            for speaker, file in case["references"].items()}
    rows = []
    for index, region in enumerate(regions):
        if len(region.speaker_ids) not in (2, 3) or region.end - region.start < 0.3:
            continue
        bounds = slice(round(region.start * sr), round(region.end * sr))
        sources = [refs[speaker][bounds] for speaker in region.speaker_ids]
        if any(np.linalg.norm(ref) < 1e-6 for ref in sources):
            continue
        separate = separate_overlap_2spk if len(sources) == 2 else separate_overlap_3spk
        row = {"start": region.start, "end": region.end, "speakers": len(sources)}
        try:
            *estimates, used = separate(audio[bounds], sr, work / f"source_{index}", backend=backend)
        except RuntimeError as exc:
            rows.append(dict(row, error=str(exc)))
            continue
        scores = max(
            ([si_sdr(est, ref) for est, ref in zip(order, sources)] for order in permutations(estimates)),
            key=sum,
        )
        delta = [score - si_sdr(audio[bounds], ref) for score, ref in zip(scores, sources)]
        rows.append(dict(row, backend=used, si_sdri_db=[round(score, 3) for score in delta]))
    return rows


def correct_role_frames(case, directory, outputs, mapping, report, regions, samples):
    """用有参考轨的窗口判定角色是否匹配且各路均比原混音更干净。"""
    sr = OUTPUT_SAMPLE_RATE
    mix = read_aligned(directory / case["audio"], sr, samples, mono=True)
    refs = {speaker: read_aligned(directory / file, sr, samples, mono=True)
            for speaker, file in case["references"].items()}
    tracks = {speaker: read_aligned(file, sr, samples, mono=True) for speaker, file in outputs.items()}
    correct = 0
    compared = 0
    for window in report["overlap_bss"]["regions"]:
        actual = window.get("source_speakers")
        if window["status"] != "separated" or not actual:
            continue
        for region in regions:
            left = max(window["start_sample"], round(region.start * sr))
            right = min(window["end_sample"], round(region.end * sr), samples)
            span_frames = round(right / sr * FRAMES_PER_SECOND) - round(left / sr * FRAMES_PER_SECOND)
            if span_frames <= 0:
                continue
            compared += span_frames
            speakers = region.speaker_ids
            expected = [mapping.get(speaker) for speaker in speakers]
            if set(actual) != set(expected) or not set(speakers).issubset(refs):
                continue
            ref = [refs[speaker][left:right] for speaker in speakers]
            separated = [tracks[speaker][left:right] for speaker in expected]
            if any(np.std(wave) < 1e-7 for wave in ref + separated):
                continue
            scores = np.array([[si_sdr(estimate, goal) for goal in ref] for estimate in separated])
            rows, cols = linear_sum_assignment(scores, maximize=True)
            baseline = [si_sdr(mix[left:right], goal) for goal in ref]
            if np.array_equal(rows, cols) and np.all(np.diag(scores) > baseline):
                correct += span_frames
    return {"reference_checked_role_frames": compared, "reference_correct_role_frames": correct}


def background_scores(case, directory, estimated):
    if "background" not in case or not estimated.is_file():
        return None
    mix_file = directory / case["audio"]
    info = sf.info(mix_file)
    args = (info.samplerate, info.frames)
    mix = read_aligned(mix_file, *args, mono=False)
    reference = read_aligned(directory / case["background"], *args, mono=False)
    background = read_aligned(estimated, *args, mono=False)
    error = background - reference
    speech = mix - reference
    power = np.sum(reference**2)
    projection = np.sum(error * speech) / np.sum(speech**2)
    return {
        "snr_db": round(float(10 * np.log10(power / max(np.sum(error**2), 1e-15))), 3),
        "speech_leakage_db": round(float(20 * np.log10(max(abs(projection), 1e-8))), 3),
    }


def ratios(counts):
    def ratio(numerator, denominator):
        return round(numerator / denominator, 4) if denominator else None

    overlap = counts["overlap_reference_frames"]
    speakers = counts["speaker_reference_frames"]
    errors = sum(counts[f"speaker_{k}_frames"] for k in ("confusion", "missed", "false_alarm"))
    return {
        "overlap_precision": ratio(counts["overlap_detected_true_frames"], counts["overlap_detected_frames"]),
        "overlap_recall": ratio(counts["overlap_detected_true_frames"], overlap),
        "overlap_separated_fraction": ratio(counts["overlap_separated_true_frames"], overlap),
        "overlap_review_fraction": ratio(counts["overlap_review_true_frames"], overlap),
        "overlap_missed_fraction": ratio(overlap - counts["overlap_detected_true_frames"], overlap),
        "overlap_not_separated_fraction": ratio(overlap - counts["overlap_separated_true_frames"], overlap),
        "overlap_output_precision": ratio(counts["overlap_separated_true_frames"],
                                         counts["overlap_separated_true_frames"]
                                         + counts["overlap_separated_false_frames"]),
        "diarization_error_rate": ratio(errors, speakers),
        "speaker_confusion_rate": ratio(counts["speaker_confusion_frames"], speakers),
    }


def score_variant(case, directory, output, variant, backend, embedding_model, review_threshold=0.4):
    mix = directory / case["audio"]
    vocals = mix.with_name("vocals.wav")
    reference = [DiarizationTurn(**turn) for turn in case["turns"]]
    reference_regions = detect_overlap_regions(reference, min_overlap_sec=0.0)
    duration = sf.info(mix).duration
    frames = round(duration * FRAMES_PER_SECOND)
    out_dir = output / case["id"] / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    if not vocals.is_file():
        raise FileNotFoundError(vocals)

    with TemporaryDirectory() as temp:
        work = Path(temp)
        options = dict(
            split_mode="diarize_bss", work_dir=work, bss_backend=backend,
            embedding_model=embedding_model,
            overlap_tracking=variant in ("tracking", "oracle_tracking", "joint_tracking"),
            save_overlap_review=variant in ("tracking", "oracle_tracking", "joint_tracking"),
        )
        acoustic_rows = None
        if variant in ("oracle", "oracle_tracking"):
            speakers = sorted({turn.speaker_id for turn in reference})
            diar = DiarizationResult(reference, reference_regions, "human", speakers)
            with patch("video2text.speaker_stems.run_diarization", return_value=diar):
                _, outputs = split_vocals_by_speaker(
                    vocals, out_dir, "result", out_dir / "split_report.json",
                    refine_embedding=False, **options,
                )
            if "references" in case and variant == "oracle":
                acoustic_rows = oracle_source_quality(
                    case, directory, vocals, reference_regions, backend, work,
                )
        else:
            source = ("joint" if variant in ("joint", "joint_tracking")
                      else "mix" if variant in ("mix", "tracking") else "vocals")
            _, outputs = split_vocals_by_speaker(
                vocals, out_dir, "result", out_dir / "split_report.json",
                diarization_source=source,
                original_audio_path=mix if source != "vocals" else None,
                overlap_review_threshold=review_threshold if variant == "review" else None,
                **options,
            )
    report = json.loads((out_dir / "split_report.json").read_text())
    segments = [
        DiarizationTurn(float(row["start"]), float(row["end"]), row["speaker_id"])
        for row in report["segments"]
    ]
    mapping, counts = compare_timeline(reference, segments, frames)
    meta = report["overlap_bss"]
    candidate_regions = [
        (r["start_sample"] / meta["sample_rate"], r["end_sample"] / meta["sample_rate"])
        for r in meta["regions"] if r["status"] != "candidate"
    ]
    counts.update(tally_overlap(reference_regions, [OverlapRegion(*r) for r in candidate_regions], report, frames))
    row = {
        "case": case["id"], "group": case["group"], "kind": case["kind"], "variant": variant,
        "counts": counts, "rates": ratios(counts),
    }
    if "references" in case:
        samples = sf.info(out_dir / "result_dialogue_review.wav").frames
        row["si_sdr"] = overlap_sdr(case, directory, outputs, mapping, samples, reference_regions)
        row["role_verification"] = correct_role_frames(
            case, directory, outputs, mapping, report, reference_regions, samples,
        )
    if acoustic_rows is not None:
        row["oracle_waveform_si_sdri"] = acoustic_rows
    return row


def summarize(rows):
    result = {}
    for variant in sorted({row["variant"] for row in rows}):
        groups = [("all", [r for r in rows if r["variant"] == variant])]
        groups.extend((kind, [r for r in groups[0][1] if r["kind"] == kind])
                      for kind in ("real", "controlled"))
        result[variant] = {}
        for kind, group in groups:
            if not group:
                continue
            counts = {key: sum(r["counts"][key] for r in group) for key in group[0]["counts"]}
            rate = ratios(counts)
            scores = [s["track_si_sdr_db"] - s["mixture_si_sdr_db"]
                      for r in group for s in r.get("si_sdr", [])]
            if scores:
                rate["median_end_to_end_si_sdri_db"] = round(float(np.median(scores)), 3)
            verification = [r for r in group if "role_verification" in r]
            if verification:
                correct = sum(r["role_verification"]["reference_correct_role_frames"] for r in verification)
                total = sum(r["counts"]["overlap_reference_frames"] for r in verification)
                rate["reference_correct_role_fraction"] = round(correct / total, 4) if total else None
            result[variant][kind] = {**counts, **rate}
    return result


def prepare_vocals(mix: Path, model: str) -> None:
    from video2text.separate import _separate_vocal_stems

    with TemporaryDirectory() as temp:
        background, vocals = _separate_vocal_stems(
            mix, Path(temp), roformer_mode="single", roformer_model=model,
        )
        copyfile(background, mix.with_name("estimated_background.wav"))
        copyfile(vocals, mix.with_name("vocals.wav"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--partition", choices=("dev", "holdout"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", choices=("oracle", "vocals", "mix", "review", "tracking",
                                              "oracle_tracking", "joint", "joint_tracking"), nargs="+",
                        default=("oracle", "vocals", "mix", "review"))
    parser.add_argument("--backend", choices=("auto", "mossformer2", "sepformer", "sepformer-whamr"),
                        default="sepformer")
    parser.add_argument("--embedding-model", default=DEFAULT_ANIME_EMBEDDING_MODEL)
    parser.add_argument("--prepare-vocals", action="store_true", help="生成尚未缓存的背景和人声轨")
    parser.add_argument("--roformer-model", default="melband_roformer_inst_v1e_plus.ckpt")
    parser.add_argument("--review-threshold", type=float, default=0.4)
    args = parser.parse_args()
    cases = json.loads(args.manifest.read_text())["cases"]
    validate_cases(cases)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    directory = args.manifest.resolve().parent
    rows, backgrounds = [], {}
    for case in cases:
        if case["partition"] != args.partition:
            continue
        mix = directory / case["audio"]
        if args.prepare_vocals and not all(
            mix.with_name(name).is_file() for name in ("estimated_background.wav", "vocals.wav")
        ):
            prepare_vocals(mix, args.roformer_model)
        bg = background_scores(case, directory, mix.with_name("estimated_background.wav"))
        if bg is not None:
            backgrounds[case["id"]] = bg
        for variant in dict.fromkeys(args.variants):
            row = score_variant(
                case, directory, output, variant, args.backend, args.embedding_model, args.review_threshold,
            )
            rows.append(row)
            print(case["id"], variant, row["rates"], flush=True)
    report = {
        "partition": args.partition, "embedding_model": args.embedding_model,
        "backend": args.backend, "review_threshold": args.review_threshold,
        "summary": summarize(rows), "cases": rows, "background": backgrounds,
    }
    path = output / f"{args.partition}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
