from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from video2text.dub import (
    DubEngineConfig,
    build_dub_engine_config,
    finalize_dub_audio,
    generate_dub_audio,
    load_dub_speaker_map,
    resolve_dub_language,
)
from video2text.ffmpeg_util import configure_ffmpeg, extract_audio
from video2text.media import get_media_duration
from video2text.ocr import extract_burned_in_subtitles
from video2text.render import render_video
from video2text.separate import VocalAssets, _load_ref_text_segments, prepare_vocal_assets
from video2text.srt import (
    load_dub_segments,
    parse_srt,
    resolve_dub_srt_path,
    segments_to_srt,
)
from video2text.translate import (
    _lang_short_code,
    _to_translator_lang,
    translate_segments,
    write_translated_srt,
)
from video2text.whisper_transcribe import detect_and_transcribe, load_whisper_model


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Whisper 视频语言识别 + 字幕烧录到画面下方"
    )
    p.add_argument("video", type=str, help="输入视频路径")
    p.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help="Whisper 模型大小 (默认 base; subs-source=whisper 时有效)",
    )
    p.add_argument(
        "--subs-source",
        default="whisper",
        choices=["whisper", "ocr"],
        help="字幕来源: whisper=音频转写; ocr=识别画面烧录字幕 (如中文字幕+原音视频)",
    )
    p.add_argument(
        "--ocr-lang",
        default="zh",
        help="OCR 识别语言 (subs-source=ocr 时), 如 zh/ja/en",
    )
    p.add_argument(
        "--ocr-fps",
        type=float,
        default=2.0,
        help="OCR 抽帧频率 (帧/秒, 默认 2)",
    )
    p.add_argument(
        "--ocr-crop-ratio",
        type=float,
        default=0.28,
        help="OCR 裁剪画面底部比例 (默认 0.28)",
    )
    p.add_argument(
        "--ocr-skip-unchanged",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="跳过画面未变化的帧 (默认开启, 烧录字幕加速)",
    )
    p.add_argument(
        "--ocr-change-threshold",
        type=float,
        default=2.0,
        help="跳变帧检测阈值, 越大越保守 (默认 2.0)",
    )
    p.add_argument(
        "--ocr-use-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="OCR 使用 CUDA GPU 加速 (需 onnxruntime-gpu, 默认开启)",
    )
    p.add_argument(
        "--ocr-no-det",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="跳过文字检测, 直接识别裁剪区域 (单行硬字幕更快; 多行字幕请保持默认关闭)",
    )
    p.add_argument(
        "--ocr-workers",
        type=int,
        default=None,
        help="OCR 并行进程数 (默认 CPU 时 min(cores,4); GPU 时为 1)",
    )
    p.add_argument(
        "--language",
        default=None,
        help="强制语言码, 如 zh/en/ja; 默认自动检测",
    )
    p.add_argument(
        "--task",
        default="transcribe",
        choices=["transcribe", "translate"],
        help="transcribe=原语言字幕; translate=译成英文",
    )
    p.add_argument(
        "--device",
        default=None,
        help="cuda / cpu; 默认自动",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="输出目录; 默认与视频同目录下的 <视频名>_whisper",
    )
    p.add_argument(
        "--font-size",
        type=int,
        default=22,
        help="烧录字幕字号",
    )
    p.add_argument(
        "--margin-v",
        type=int,
        default=40,
        help="字幕距画面底部的像素边距",
    )
    p.add_argument(
        "--no-burn",
        action="store_true",
        help="只导出字幕/文本, 不生成带字幕视频",
    )
    p.add_argument(
        "--burn-only",
        action="store_true",
        help="跳过转写, 用已有 SRT 重新烧录 (默认找输出目录下同名 .srt)",
    )
    p.add_argument(
        "--dub-only",
        action="store_true",
        help="跳过转写, 用已有 SRT 生成 AI 配音 (配合 --dub-srt 或 --srt 指定目标语言字幕)",
    )
    p.add_argument(
        "--srt",
        default=None,
        help="指定 SRT 路径; 与 --burn-only / --dub-only 一起用",
    )
    p.add_argument(
        "--dub-srt",
        default=None,
        help="AI 配音专用字幕 (如中文字幕); 直接按此文件朗读, 不经过转写/自动翻译",
    )
    p.add_argument(
        "--box",
        action="store_true",
        help="字幕用黑色不透明底框; 默认透明底 + 描边",
    )
    p.add_argument(
        "--dub",
        action="store_true",
        help="启用 AI 配音; 默认保留背景音乐, 仅替换人声",
    )
    p.add_argument(
        "--no-keep-bgm",
        action="store_true",
        help="AI 配音时不保留原视频背景音乐 (仅输出 TTS 人声)",
    )
    p.add_argument(
        "--bgm-volume",
        type=float,
        default=1.0,
        help="混合时背景音乐音量 (默认 1.0)",
    )
    p.add_argument(
        "--dub-volume",
        type=float,
        default=1.0,
        help="混合时 AI 配音音量 (默认 1.0)",
    )
    p.add_argument(
        "--dub-lang",
        default=None,
        help="AI 配音语言码, 如 zh/en/ja; 默认 translate 任务为 en, 否则与转写语言一致",
    )
    p.add_argument(
        "--dub-voice",
        default=None,
        help="edge-tts 音色名 (dub-engine=edge/rvc 时); 默认按语言自动选择; 未映射段也用此音色",
    )
    p.add_argument(
        "--dub-voice-female",
        default=None,
        help="女声 edge-tts 音色 (--dub-speaker-map 中 female/女 时使用)",
    )
    p.add_argument(
        "--dub-voice-male",
        default=None,
        help="男声 edge-tts 音色 (--dub-speaker-map 中 male/男 时使用)",
    )
    p.add_argument(
        "--dub-speaker-map",
        default=None,
        metavar="JSON",
        help="说话人映射 JSON: SRT 序号 -> male/female/女/男 或 edge-tts 音色名",
    )
    p.add_argument(
        "--dub-engine",
        default="edge",
        choices=["edge", "rvc", "gpt-sovits"],
        help="配音引擎: edge=预设音色; rvc=edge-tts+RVC 音色转换; gpt-sovits=本地 API 克隆",
    )
    p.add_argument(
        "--voice-ref",
        default=None,
        help="音色参考音频 (wav/mp3); 未指定时从分离人声轨自动截取",
    )
    p.add_argument(
        "--voice-ref-text",
        default=None,
        help="参考音频对应文本 (GPT-SoVITS 用; 默认自动匹配原语言字幕)",
    )
    p.add_argument(
        "--voice-ref-lang",
        default="ja",
        help="参考音频语言码 (GPT-SoVITS prompt_language, 默认 ja)",
    )
    p.add_argument(
        "--rvc-model",
        default=None,
        help="RVC 模型 .pth 路径 (dub-engine=rvc)",
    )
    p.add_argument(
        "--rvc-index",
        default=None,
        help="RVC index 文件路径 (可选, 提升相似度)",
    )
    p.add_argument(
        "--rvc-api",
        default=None,
        help="RVC HTTP 服务地址 (如 http://127.0.0.1:7860; 提供 /convert 接口)",
    )
    p.add_argument(
        "--gpt-sovits-api",
        default="http://127.0.0.1:9880",
        help="GPT-SoVITS 本地 API 地址 (dub-engine=gpt-sovits)",
    )
    p.add_argument(
        "--sovits-speed",
        type=float,
        default=0.95,
        help="GPT-SoVITS 语速 (略慢更清晰, 默认 0.95)",
    )
    p.add_argument(
        "--sovits-text-split",
        default="cut0",
        help="GPT-SoVITS 分句方式 (短字幕推荐 cut0, 默认 cut0)",
    )
    p.add_argument(
        "--translate-engine",
        default="auto",
        choices=["auto", "bing", "alibaba", "google", "baidu"],
        help="翻译引擎 (默认 auto: 依次尝试 bing/alibaba/google)",
    )
    p.add_argument(
        "--translate-to",
        default=None,
        metavar="LANG",
        help="将字幕翻译为目标语言, 如 zh/en/ja (需联网); 可与 --dub 联用",
    )
    p.add_argument(
        "--translate-from",
        default=None,
        metavar="LANG",
        help="翻译源语言; 默认自动 (转写时用检测到的语言)",
    )
    p.add_argument(
        "--translate-srt-only",
        action="store_true",
        help="仅翻译已有 SRT (需 --translate-to; 默认读输出目录下同名 .srt 或 --srt)",
    )
    p.add_argument(
        "--ffmpeg",
        default=None,
        help="ffmpeg 可执行文件路径 (Windows 未加入 PATH 时可手动指定)",
    )
    p.add_argument(
        "--ffprobe",
        default=None,
        help="ffprobe 可执行文件路径 (可选; 未指定时尝试与 ffmpeg 同目录)",
    )
    return p

def _validate_dub_engine_args(args) -> None:
    if args.dub_engine == "rvc" and not args.rvc_model and not args.rvc_api:
        raise SystemExit(
            "--dub-engine rvc 需要 --rvc-model (本地推理) 或 --rvc-api (HTTP 服务)"
        )
    if args.dub_speaker_map:
        map_path = Path(args.dub_speaker_map).expanduser().resolve()
        if not map_path.is_file():
            raise SystemExit(f"找不到 --dub-speaker-map: {map_path}")
        try:
            load_dub_speaker_map(map_path)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

def run_dub_workflow(
    video_path: Path,
    out_dir: Path,
    video_stem: str,
    segments: list[dict],
    args,
    dub_lang: str,
    dub_wav_path: Path,
) -> tuple[str, Path, DubEngineConfig]:
    """生成配音 WAV 并可选与 BGM 混音, 返回 (voice_label, final_audio, config)。"""
    _validate_dub_engine_args(args)
    keep_bgm = not args.no_keep_bgm
    need_voice_ref = args.dub_engine == "gpt-sovits" or bool(args.voice_ref)
    need_stems = keep_bgm or need_voice_ref
    ref_text_segments = _load_ref_text_segments(out_dir, video_stem, segments)
    video_duration = get_media_duration(video_path)

    with tempfile.TemporaryDirectory(prefix="whisper_stems_") as stem_tmp:
        stem_dir = Path(stem_tmp)
        vocal_assets: VocalAssets | None = None
        if need_stems:
            vocal_assets = prepare_vocal_assets(
                video_path,
                out_dir,
                video_stem,
                segments,
                voice_ref_override=args.voice_ref,
                voice_ref_text_override=args.voice_ref_text,
                ref_text_segments=ref_text_segments,
                work_dir=stem_dir,
                need_stems=keep_bgm,
                need_voice_ref=need_voice_ref,
            )

        engine_config = build_dub_engine_config(args, dub_lang, vocal_assets)

        with tempfile.TemporaryDirectory(prefix="whisper_dub_") as dub_tmp:
            resolved_voice = generate_dub_audio(
                segments,
                dub_wav_path,
                dub_lang,
                video_duration,
                Path(dub_tmp),
                engine_config,
            )

        print(f"已写入: {dub_wav_path}")
        final_audio = finalize_dub_audio(
            dub_wav_path,
            out_dir,
            video_stem,
            keep_bgm=keep_bgm,
            bgm_volume=args.bgm_volume,
            dub_volume=args.dub_volume,
            vocal_assets=vocal_assets,
            video_path=video_path,
            work_dir=stem_dir,
        )

    return resolved_voice, final_audio, engine_config

def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)
    video_path = Path(args.video).expanduser().resolve()
    if not video_path.is_file():
        print(f"找不到视频: {video_path}", file=sys.stderr)
        return 1

    try:
        configure_ffmpeg(args.ffmpeg, args.ffprobe)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else video_path.parent / f"{video_path.stem}_whisper"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    srt_path = (
        Path(args.srt).expanduser().resolve()
        if args.srt
        else out_dir / f"{video_path.stem}.srt"
    )
    txt_path = out_dir / f"{video_path.stem}.txt"
    json_path = out_dir / f"{video_path.stem}.json"
    subtitled_path = out_dir / f"{video_path.stem}_subtitled{video_path.suffix}"
    dubbed_path = out_dir / f"{video_path.stem}_dubbed{video_path.suffix}"
    subtitled_dubbed_path = (
        out_dir / f"{video_path.stem}_subtitled_dubbed{video_path.suffix}"
    )

    if args.burn_only and args.dub_only:
        print("--burn-only 与 --dub-only 不能同时使用", file=sys.stderr)
        return 1

    if args.translate_srt_only:
        if not args.translate_to:
            print("--translate-srt-only 需要 --translate-to, 如 --translate-to zh", file=sys.stderr)
            return 1
        if not srt_path.is_file():
            print(f"找不到 SRT: {srt_path}", file=sys.stderr)
            return 1
        segments = parse_srt(srt_path.read_text(encoding="utf-8"))
        if not segments:
            print("SRT 中无有效字幕段", file=sys.stderr)
            return 2
        source_lang = args.translate_from or "auto"
        translated = translate_segments(
            segments, source_lang, args.translate_to, args.translate_engine
        )
        write_translated_srt(translated, srt_path, args.translate_to)
        return 0

    if args.dub_only:
        dub_lang_arg = args.dub_lang or args.translate_to
        if not dub_lang_arg:
            print("--dub-only 需要 --dub-lang (如 zh), 且应与配音字幕语言一致", file=sys.stderr)
            return 1
        dub_lang = resolve_dub_language("transcribe", "", dub_lang_arg)
        try:
            dub_srt_resolved = resolve_dub_srt_path(
                args.dub_srt, args.srt, out_dir, video_path.stem, dub_lang_arg
            )
            if dub_srt_resolved is None:
                print(
                    "请用 --dub-srt 或 --srt 指定目标语言字幕文件, "
                    f"或在输出目录提供 {video_path.stem}_{_lang_short_code(dub_lang_arg)}.srt",
                    file=sys.stderr,
                )
                return 1
            segments, srt_path = load_dub_segments(dub_srt_resolved, [])
            srt_body = segments_to_srt(segments)
        except (FileNotFoundError, RuntimeError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        if args.translate_to:
            print("提示: 已指定 --dub-srt/--srt, 跳过 --translate-to 自动翻译", file=sys.stderr)
        dub_wav_path = out_dir / f"{video_path.stem}_dub.wav"
        print(
            f"从 {srt_path.name} 生成 AI 配音 (语言={dub_lang}, 引擎={args.dub_engine}) -> {dub_wav_path.name}"
        )
        try:
            resolved_voice, final_audio, engine_config = run_dub_workflow(
                video_path,
                out_dir,
                video_path.stem,
                segments,
                args,
                dub_lang,
                dub_wav_path,
            )
        except (RuntimeError, SystemExit) as exc:
            if isinstance(exc, SystemExit):
                raise
            print(str(exc), file=sys.stderr)
            return 2
        audio_label = "保留背景音乐" if not args.no_keep_bgm else "仅 AI 配音"
        if args.no_burn:
            print(f"替换人声配音 ({audio_label}) -> {dubbed_path.name}")
            render_video(video_path, dubbed_path, audio_path=final_audio)
            print(f"完成: {dubbed_path}")
            return 0
        out_path = subtitled_dubbed_path
        print(f"烧录字幕 + AI 配音 ({audio_label}) -> {out_path.name}")
        render_video(
            video_path,
            out_path,
            audio_path=final_audio,
            srt_path=srt_path,
            font_size=args.font_size,
            margin_v=args.margin_v,
            box=args.box,
        )
        print(f"完成: {out_path}")
        return 0

    if args.burn_only:
        if not srt_path.is_file():
            print(f"找不到 SRT: {srt_path}", file=sys.stderr)
            return 1
        srt_body = srt_path.read_text(encoding="utf-8")
        if not srt_body.strip():
            print("SRT 为空, 跳过烧录", file=sys.stderr)
            return 2
        out_path = subtitled_path
        print(f"仅烧录 (复用 {srt_path.name}) -> {out_path.name}")
        render_video(
            video_path,
            out_path,
            srt_path=srt_path,
            font_size=args.font_size,
            margin_v=args.margin_v,
            box=args.box,
        )
        print(f"完成: {out_path}")
        return 0

    if args.subs_source == "ocr":
        print("=" * 60)
        print(f"字幕来源: 画面 OCR (语言={args.ocr_lang})")
        print("=" * 60)
        try:
            with tempfile.TemporaryDirectory(prefix="whisper_ocr_") as ocr_tmp:
                segments = extract_burned_in_subtitles(
                    video_path,
                    Path(ocr_tmp),
                    ocr_lang=args.ocr_lang,
                    fps=args.ocr_fps,
                    crop_ratio=args.ocr_crop_ratio,
                    skip_unchanged=args.ocr_skip_unchanged,
                    change_threshold=args.ocr_change_threshold,
                    use_gpu=args.ocr_use_gpu,
                    no_det=args.ocr_no_det,
                    workers=args.ocr_workers,
                )
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        full_text = " ".join((s.get("text") or "").strip() for s in segments).strip()
        detected = args.ocr_lang
        conf = 1.0
        used_lang = args.ocr_lang
        device = args.device or "ocr"
        print("-" * 60)
        print("OCR 识别文本:")
        print(full_text if full_text else "(空)")
        print("=" * 60)
    else:
        model, device = load_whisper_model(args.model, args.device)

        with tempfile.TemporaryDirectory(prefix="whisper_audio_") as tmp:
            wav_path = Path(tmp) / "audio.wav"
            print(f"抽取音频: {video_path.name} -> wav")
            extract_audio(video_path, wav_path)
            result, detected, conf = detect_and_transcribe(
                model,
                wav_path,
                language=args.language,
                task=args.task,
            )

        full_text = (result.get("text") or "").strip()
        segments = result.get("segments") or []
        used_lang = result.get("language") or detected

        print("=" * 60)
        print(f"检测到的语言: {detected} (置信度 {conf:.2%})")
        print(f"转写使用语言: {used_lang}")
        print("-" * 60)
        print("完整转写文本:")
        print(full_text if full_text else "(空)")
        print("=" * 60)

    srt_body = segments_to_srt(segments)
    srt_path.write_text(srt_body, encoding="utf-8")
    txt_path.write_text(full_text + ("\n" if full_text else ""), encoding="utf-8")
    meta = {
        "video": str(video_path),
        "subs_source": args.subs_source,
        "model": args.model if args.subs_source == "whisper" else None,
        "device": device,
        "detected_language": detected,
        "detected_confidence": conf,
        "used_language": used_lang,
        "task": args.task if args.subs_source == "whisper" else "ocr",
        "text": full_text,
        "segments": [
            {
                "id": s.get("id"),
                "start": s.get("start"),
                "end": s.get("end"),
                "text": (s.get("text") or "").strip(),
            }
            for s in segments
        ],
    }
    json_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"已写入: {txt_path}")
    print(f"已写入: {srt_path}")
    print(f"已写入: {json_path}")

    dub_lang = resolve_dub_language(
        args.task,
        used_lang,
        args.dub_lang or args.translate_to or (args.ocr_lang if args.subs_source == "ocr" else None),
    )
    dub_srt_resolved = resolve_dub_srt_path(
        args.dub_srt,
        args.srt,
        out_dir if args.subs_source != "ocr" else None,
        video_path.stem,
        args.dub_lang or args.translate_to,
    )
    use_external_dub_srt = dub_srt_resolved is not None

    if args.translate_to and not use_external_dub_srt:
        source_lang = args.translate_from or used_lang or detected
        segments = translate_segments(
            segments, source_lang, args.translate_to, args.translate_engine
        )
        srt_path = write_translated_srt(segments, srt_path, args.translate_to)
        srt_body = segments_to_srt(segments)
        full_text = " ".join((seg.get("text") or "").strip() for seg in segments).strip()
        txt_path = out_dir / f"{video_path.stem}_{_lang_short_code(args.translate_to)}.txt"
        txt_path.write_text(full_text + ("\n" if full_text else ""), encoding="utf-8")
        meta["translated_language"] = _to_translator_lang(args.translate_to)
        meta["translated_text"] = full_text
        meta["segments"] = [
            {
                "id": s.get("id"),
                "start": s.get("start"),
                "end": s.get("end"),
                "text": (s.get("text") or "").strip(),
            }
            for s in segments
        ]
        json_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"已写入: {txt_path}")
    elif args.translate_to and use_external_dub_srt:
        print(
            "提示: 已指定 --dub-srt, 跳过 --translate-to 自动翻译; "
            "配音与烧录将使用配音字幕文件",
            file=sys.stderr,
        )

    dub_wav_path = out_dir / f"{video_path.stem}_dub.wav"
    final_audio_path = dub_wav_path
    resolved_voice: str | None = None
    burn_srt_path = srt_path

    if args.dub:
        try:
            dub_segments, dub_srt_file = load_dub_segments(
                dub_srt_resolved,
                segments,
            )
        except (FileNotFoundError, RuntimeError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        if dub_srt_file is not None:
            burn_srt_path = dub_srt_file
            srt_body = segments_to_srt(dub_segments)
        else:
            dub_segments = segments
        if not dub_segments:
            print("无有效字幕段, 无法生成 AI 配音", file=sys.stderr)
            return 2
        keep_bgm = not args.no_keep_bgm
        bgm_note = ", 保留背景音乐" if keep_bgm else ""
        print(
            f"生成 AI 配音 (语言={dub_lang}, 引擎={args.dub_engine}{bgm_note}) -> {dub_wav_path.name}"
        )
        try:
            resolved_voice, final_audio_path, engine_config = run_dub_workflow(
                video_path,
                out_dir,
                video_path.stem,
                dub_segments,
                args,
                dub_lang,
                dub_wav_path,
            )
        except (RuntimeError, SystemExit) as exc:
            if isinstance(exc, SystemExit):
                raise
            print(str(exc), file=sys.stderr)
            return 2
        meta["dub"] = {
            "enabled": True,
            "engine": engine_config.engine,
            "language": dub_lang,
            "voice": resolved_voice,
            "audio": str(dub_wav_path),
            "final_audio": str(final_audio_path),
            "keep_bgm": keep_bgm,
            "voice_ref": str(engine_config.voice_ref) if engine_config.voice_ref else None,
            "srt": str(burn_srt_path) if dub_srt_file else None,
        }
        json_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    if args.no_burn and not args.dub:
        print("已跳过烧录 (--no-burn)")
        return 0

    if args.no_burn and args.dub:
        audio_label = "保留背景音乐" if not args.no_keep_bgm else "仅 AI 配音"
        print(f"替换人声配音 ({audio_label}) -> {dubbed_path.name}")
        render_video(video_path, dubbed_path, audio_path=final_audio_path)
        print(f"完成: {dubbed_path}")
        return 0

    if not srt_body.strip() and not args.dub:
        print("无有效字幕段, 跳过烧录")
        return 2

    if args.dub:
        out_path = subtitled_dubbed_path
        audio_label = "保留背景音乐" if not args.no_keep_bgm else "仅 AI 配音"
        print(f"烧录字幕 + AI 配音 ({audio_label}) -> {out_path.name}")
        render_video(
            video_path,
            out_path,
            audio_path=final_audio_path,
            srt_path=burn_srt_path if srt_body.strip() else None,
            font_size=args.font_size,
            margin_v=args.margin_v,
            box=args.box,
        )
    else:
        out_path = subtitled_path
        print(f"烧录字幕到视频下方 -> {out_path.name}")
        render_video(
            video_path,
            out_path,
            srt_path=srt_path,
            font_size=args.font_size,
            margin_v=args.margin_v,
            box=args.box,
        )
    print(f"完成: {out_path}")
    return 0
