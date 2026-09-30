#!/usr/bin/env python3
"""BS-RoFormer 人声/伴奏分离；可选三轨 (男/女) 或多轨 (N speaker)。"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from video2text.ffmpeg_util import configure_ffmpeg, extract_stereo_audio
from video2text.separate import _separate_vocal_stems, separate_multi_stems, separate_three_stems
from video2text.diarization import estimate_speaker_count


def _roformer_backend_label(args) -> str:
    if args.roformer_mode == "ensemble":
        return f"BS-RoFormer ensemble ({args.roformer_ensemble_preset})"
    if args.roformer_mode == "dual":
        return (
            f"BS-RoFormer dual (voc={args.roformer_vocals_model}, "
            f"inst={args.roformer_inst_model})"
        )
    return f"BS-RoFormer ({args.roformer_model})"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="分离视频/音频中的人声与伴奏")
    p.add_argument("input", type=Path, help="输入视频或 wav")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="输出目录 (默认与输入同目录)",
    )
    p.add_argument("--stem", default=None, help="输出文件名前缀 (默认输入文件名)")
    p.add_argument(
        "--three-stems",
        action="store_true",
        help="输出三轨: 背景 + 男声 + 女声 (BS-RoFormer + 自动性别切分)",
    )
    p.add_argument(
        "--multi-stems",
        action="store_true",
        help="输出 N 条 speaker 轨 (BS-RoFormer + pyannote 日记化)",
    )
    p.add_argument(
        "--count-speakers",
        action="store_true",
        help="仅运行日记化预检, 报告说话人数量与每人时长后退出 (不做切分)",
    )
    p.add_argument(
        "--split-mode",
        choices=["whisper_f0", "srt_f0", "diarize", "diarize_bss"],
        default="srt_f0",
        help="三轨: srt_f0/whisper_f0/diarize; 多轨: diarize/diarize_bss",
    )
    p.add_argument(
        "--srt",
        type=Path,
        default=None,
        help="字幕 SRT (split-mode=srt_f0 必填; 默认同目录 <stem>.srt 或 *_whisper/*.srt)",
    )
    p.add_argument(
        "--f0-threshold",
        type=float,
        default=200.0,
        help="基频阈值 Hz (>= 女声, < 男声; 默认 200, 可配合 --no-adaptive-f0)",
    )
    p.add_argument(
        "--no-adaptive-f0",
        action="store_true",
        help="禁用自适应 F0 阈值 (仅用 --f0-threshold)",
    )
    p.add_argument(
        "--min-voiced-ratio",
        type=float,
        default=0.25,
        help="F0 可信度下限 (默认 0.25)",
    )
    p.add_argument(
        "--roformer-model",
        default="model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        help="BS-RoFormer 模型 ckpt (roformer-mode=single)",
    )
    p.add_argument(
        "--roformer-mode",
        choices=["single", "dual", "ensemble"],
        default="ensemble",
        help="RoFormer 分离模式 (默认 ensemble 专保伴奏)",
    )
    p.add_argument(
        "--roformer-ensemble-preset",
        default="instrumental_full",
        help="ensemble preset (默认 instrumental_full 最大保伴奏)",
    )
    p.add_argument(
        "--roformer-vocals-model",
        default="bs_roformer_vocals_revive_v3e_unwa.ckpt",
        help="dual 模式 vocals 模型 (默认 revive v3e)",
    )
    p.add_argument(
        "--roformer-inst-model",
        default="mel_band_roformer_instrumental_fv7z_gabox.ckpt",
        help="dual 模式 instrumental 模型 (默认 fv7z bleedless)",
    )
    p.add_argument(
        "--roformer-overlap",
        type=int,
        default=None,
        help="MDXC overlap 窗口数 (默认模型 yaml 值, 提高如 16 可增质量)",
    )
    p.add_argument(
        "--roformer-model-dir",
        type=Path,
        default=None,
        help="RoFormer 模型目录 (默认项目 models/roformer; 可用 VIDEO2TEXT_ROFORMER_MODEL_DIR)",
    )
    p.add_argument(
        "--whisper-model",
        default="base",
        help="split-mode=whisper_f0 时的 Whisper 模型 (默认 base)",
    )
    p.add_argument(
        "--language",
        default=None,
        help="Whisper 语言码, 如 ja/zh (whisper_f0 模式)",
    )
    p.add_argument(
        "--whisper-device",
        default=None,
        help="Whisper 设备 cpu/cuda",
    )
    p.add_argument(
        "--min-speakers",
        type=int,
        default=1,
        help="diarize 模式最少说话人数 (默认 1)",
    )
    p.add_argument(
        "--max-speakers",
        type=int,
        default=4,
        help="diarize 模式最多说话人数 (默认 4)",
    )
    p.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace Token (diarize 模式; 或设环境变量 HF_TOKEN)",
    )
    p.add_argument(
        "--speaker-map",
        type=Path,
        default=None,
        help="可选 speaker_map.json, 按 index 覆盖 gender 后重切 (仅 --three-stems)",
    )
    p.add_argument(
        "--diarization-model",
        default="pyannote/speaker-diarization-community-1",
        help="pyannote 日记化模型 (multi-stems / diarize 模式)",
    )
    p.add_argument(
        "--embedding-model",
        default="litagin/anime_speaker_embedding_by_va_ecapa_tdnn_groupnorm",
        help="说话人 embedding 模型 (diarize_bss 重叠归属)",
    )
    p.add_argument(
        "--bss-backend",
        choices=["auto", "mossformer2", "sepformer"],
        default="auto",
        help="重叠段盲分离 backend (diarize_bss)",
    )
    p.add_argument(
        "--no-embedding",
        action="store_true",
        help="diarize_bss 时禁用 anime embedding (仅用日记化 speaker_id)",
    )
    p.add_argument(
        "--refine-embedding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="日记化后用 embedding 合并碎段/改标/过滤 (默认开启)",
    )
    p.add_argument(
        "--merge-turn-gap",
        type=float,
        default=0.4,
        help="合并同 speaker 相邻碎段的最大间隔秒 (默认 0.4)",
    )
    p.add_argument(
        "--min-turn-sec",
        type=float,
        default=0.5,
        help="参与 speaker profile 的最短 turn 秒 (默认 0.5)",
    )
    p.add_argument(
        "--min-similarity",
        type=float,
        default=0.35,
        help="embedding 改标/过滤最低相似度 (默认 0.35)",
    )
    p.add_argument(
        "--drop-low-confidence",
        action="store_true",
        help="丢弃 refine 后置信度低于 --min-similarity 的 turn (默认保留)",
    )
    p.add_argument(
        "--stem-min-similarity",
        type=float,
        default=0.30,
        help="切分/填充时 embedding 最低相似度 (默认 0.30)",
    )
    p.add_argument(
        "--stem-relabel-margin",
        type=float,
        default=0.08,
        help="切分覆写 speaker 所需 embedding margin (默认 0.08)",
    )
    p.add_argument(
        "--fill-min-similarity",
        type=float,
        default=0.25,
        help="未分配区间填充时的 embedding 最低相似度 (默认 0.25)",
    )
    p.add_argument(
        "--fill-unassigned",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="对未分配人声区间按 embedding 填充 (默认开启)",
    )
    p.add_argument(
        "--diarize-preprocess",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="日记化前对 vocals 峰值归一化+高通 (默认开启)",
    )
    p.add_argument(
        "--overlap-min-sec",
        type=float,
        default=0.15,
        help="重叠检测/BSS 最小时长秒 (默认 0.15)",
    )
    p.add_argument(
        "--max-overlap-sec",
        type=float,
        default=8.0,
        help="重叠段 BSS 盲分离最大时长秒 (默认 8.0; 旧值 3.0 会跳过长重叠)",
    )
    p.add_argument("--ffmpeg", default=None)
    p.add_argument("--ffprobe", default=None)
    args = p.parse_args(argv)

    src = args.input.expanduser().resolve()
    if not src.is_file():
        print(f"找不到输入: {src}", file=sys.stderr)
        return 1

    try:
        configure_ffmpeg(args.ffmpeg, args.ffprobe)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    out_dir = args.out_dir.expanduser().resolve() if args.out_dir else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.stem or src.stem

    if args.three_stems and args.multi_stems:
        print("--three-stems 与 --multi-stems 不能同时使用", file=sys.stderr)
        return 1

    if args.multi_stems and args.split_mode not in ("diarize", "diarize_bss"):
        print(
            "--multi-stems 需要 --split-mode diarize 或 diarize_bss",
            file=sys.stderr,
        )
        return 1

    if args.three_stems and args.split_mode == "diarize_bss":
        print(
            "--three-stems 不支持 diarize_bss, 请使用 --multi-stems",
            file=sys.stderr,
        )
        return 1

    srt_path = None
    if args.srt:
        srt_path = args.srt.expanduser().resolve()
    elif args.three_stems and args.split_mode == "srt_f0":
        for candidate in (
            src.parent / f"{stem}_whisper" / f"{stem}.srt",
            src.parent / f"{stem}.srt",
            out_dir / f"{stem}.srt",
        ):
            if candidate.is_file():
                srt_path = candidate.resolve()
                break

    if args.three_stems and args.split_mode == "srt_f0":
        if srt_path is None or not srt_path.is_file():
            print(
                "split-mode=srt_f0 需要 --srt 或同目录下可找到的 .srt 字幕",
                file=sys.stderr,
            )
            return 1

    roformer_kwargs = {
        "roformer_model": args.roformer_model,
        "roformer_mode": args.roformer_mode,
        "roformer_ensemble_preset": args.roformer_ensemble_preset,
        "roformer_vocals_model": args.roformer_vocals_model,
        "roformer_inst_model": args.roformer_inst_model,
        "roformer_overlap": args.roformer_overlap,
        "roformer_model_dir": (
            args.roformer_model_dir.expanduser().resolve()
            if args.roformer_model_dir
            else None
        ),
    }

    with tempfile.TemporaryDirectory(prefix="v2t_stems_") as tmp:
        work = Path(tmp)
        if src.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg"}:
            audio_wav = src
        else:
            audio_wav = work / "source.wav"
            print(f"提取音轨: {src.name}")
            extract_stereo_audio(src, audio_wav)

        if args.count_speakers:
            import json

            try:
                instrumental, vocals = _separate_vocal_stems(
                    audio_wav, work, **roformer_kwargs
                )
            except RuntimeError as exc:
                print(str(exc), file=sys.stderr)
                return 2

            report = estimate_speaker_count(
                vocals,
                max_speakers=args.max_speakers,
                hf_token=args.hf_token,
                model_id=args.diarization_model,
                diarize_preprocess=args.diarize_preprocess,
                work_dir=work,
            )
            report_path = out_dir / f"{stem}_speaker_count.json"
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"已写入: {report_path}")
            print(
                f"  识别到 {report['speaker_count']} 位说话人, "
                f"{report['turn_count']} 段, "
                f"{report['overlap_region_count']} 个重叠区间"
            )
            for sp, sec in report["per_speaker_sec"].items():
                turns = report["per_speaker_turns"].get(sp, 0)
                print(f"    {sp}: {sec:.1f}s ({turns} 段)")
            return 0

        if args.multi_stems:
            try:
                separate_multi_stems(
                    audio_wav,
                    out_dir,
                    stem,
                    work,
                    split_mode=args.split_mode,
                    min_speakers=args.min_speakers,
                    max_speakers=args.max_speakers,
                    hf_token=args.hf_token,
                    diarization_model=args.diarization_model,
                    embedding_model=args.embedding_model,
                    use_embedding=not args.no_embedding,
                    bss_backend=args.bss_backend,
                    refine_embedding=args.refine_embedding,
                    merge_turn_gap=args.merge_turn_gap,
                    min_turn_sec=args.min_turn_sec,
                    min_similarity=args.min_similarity,
                    fill_unassigned=args.fill_unassigned,
                    diarize_preprocess=args.diarize_preprocess,
                    overlap_min_sec=args.overlap_min_sec,
                    max_overlap_sec=args.max_overlap_sec,
                    drop_low_confidence=args.drop_low_confidence,
                    stem_min_similarity=args.stem_min_similarity,
                    stem_relabel_margin=args.stem_relabel_margin,
                    fill_min_similarity=args.fill_min_similarity,
                    roformer_model=args.roformer_model,
                    roformer_mode=args.roformer_mode,
                    roformer_ensemble_preset=args.roformer_ensemble_preset,
                    roformer_vocals_model=args.roformer_vocals_model,
                    roformer_inst_model=args.roformer_inst_model,
                    roformer_overlap=args.roformer_overlap,
                    roformer_model_dir=roformer_kwargs["roformer_model_dir"],
                )
            except (RuntimeError, ValueError) as exc:
                print(str(exc), file=sys.stderr)
                return 2
            return 0

        if args.three_stems:
            try:
                separate_three_stems(
                    audio_wav,
                    out_dir,
                    stem,
                    work,
                    split_mode=args.split_mode,
                    srt_path=srt_path,
                    f0_threshold=args.f0_threshold,
                    adaptive_threshold=not args.no_adaptive_f0,
                    min_voiced_ratio=args.min_voiced_ratio,
                    whisper_model=args.whisper_model,
                    language=args.language,
                    whisper_device=args.whisper_device,
                    min_speakers=args.min_speakers,
                    max_speakers=args.max_speakers,
                    hf_token=args.hf_token,
                    roformer_model=args.roformer_model,
                    roformer_mode=args.roformer_mode,
                    roformer_ensemble_preset=args.roformer_ensemble_preset,
                    roformer_vocals_model=args.roformer_vocals_model,
                    roformer_inst_model=args.roformer_inst_model,
                    roformer_overlap=args.roformer_overlap,
                    roformer_model_dir=roformer_kwargs["roformer_model_dir"],
                    speaker_map_path=(
                        args.speaker_map.expanduser().resolve()
                        if args.speaker_map
                        else None
                    ),
                )
            except (RuntimeError, ValueError) as exc:
                print(str(exc), file=sys.stderr)
                return 2
            return 0

        print(f"{_roformer_backend_label(args)} 人声分离...")
        instrumental, vocals = _separate_vocal_stems(audio_wav, work, **roformer_kwargs)

        vocals_out = out_dir / f"{stem}_vocals.wav"
        bgm_out = out_dir / f"{stem}_no_vocals.wav"
        shutil.copy2(vocals, vocals_out)
        shutil.copy2(instrumental, bgm_out)
        print(f"已写入: {vocals_out}")
        print(f"已写入: {bgm_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
