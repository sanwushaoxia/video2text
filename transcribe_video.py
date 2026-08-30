#!/usr/bin/env python3
"""
基于 OpenAI Whisper: 识别视频语言、转写音频为文字, 并将字幕烧录到视频下方。

依赖:
  - ffmpeg (系统已安装)
  - openai-whisper (pip install -r requirements.txt)

示例:
  python transcribe_video.py input.mp4
  python transcribe_video.py input.mp4 --model medium --language zh
  python transcribe_video.py input.mp4 --out-dir ./output --no-burn
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# 保证可直接运行本文件
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _require_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError("未找到 ffmpeg, 请先安装: sudo apt install ffmpeg")
    return path


def extract_audio(video_path: Path, wav_path: Path, sample_rate: int = 16000) -> None:
    """从视频抽出单声道 16kHz wav, Whisper 推荐输入。"""
    ffmpeg = _require_ffmpeg()
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        str(wav_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def sec_to_srt_time(t: float) -> str:
    """秒 -> SRT 时间码 00:00:00,000"""
    if t < 0:
        t = 0.0
    ms = int(round(t * 1000.0))
    h, rem = divmod(ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, milli = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{milli:03d}"


def segments_to_srt(segments) -> str:
    lines = []
    for i, seg in enumerate(segments, start=1):
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        lines.append(str(i))
        lines.append(f"{sec_to_srt_time(start)} --> {sec_to_srt_time(end)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def burn_subtitles(
    video_path: Path,
    srt_path: Path,
    output_path: Path,
    font_size: int = 22,
    margin_v: int = 40,
    box: bool = False,
) -> None:
    """
    用 ffmpeg 将 SRT 烧录到画面下方。
    默认 BorderStyle=1: 白字黑描边、透明底, 不挡画面。
    box=True 时 BorderStyle=3: 不透明黑底框 (旧样式)。
    """
    ffmpeg = _require_ffmpeg()
    # ffmpeg subtitles filter 路径中的特殊字符需转义
    srt_escaped = (
        str(srt_path.resolve())
        .replace("\\", "/")
        .replace(":", "\\:")
        .replace("'", "\\'")
    )
    if box:
        # 不透明底框
        border = "BorderStyle=3,Outline=2,Shadow=0"
    else:
        # 透明底 + 描边, 浅色/深色画面都可读
        border = "BorderStyle=1,Outline=2,Shadow=1"
    force_style = (
        f"Alignment=2,FontSize={font_size},PrimaryColour=&H00FFFFFF,"
        f"OutlineColour=&H00000000,BackColour=&H00000000,{border},"
        f"MarginV={margin_v}"
    )
    vf = f"subtitles='{srt_escaped}':force_style='{force_style}'"
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-vf",
        vf,
        "-c:a",
        "copy",
        str(output_path),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "烧录字幕失败 (ffmpeg):\n{}".format(proc.stderr[-2000:])
        )


def load_whisper_model(model_name: str, device: str | None):
    import whisper
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"加载 Whisper 模型: {model_name} (device={device})")
    return whisper.load_model(model_name, device=device), device


def detect_and_transcribe(
    model,
    audio_path: Path,
    language: str | None,
    task: str = "transcribe",
):
    """
    识别语言并转写.
    language=None 时由 Whisper 自动检测.
    """
    import whisper

    audio = whisper.load_audio(str(audio_path))
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

    # 语言检测 (即使用户指定了 language, 也打印一次检测结果供参考)
    _, probs = model.detect_language(mel)
    detected = max(probs, key=probs.get)
    print(
        "语言检测: {} (置信度 {:.2%})".format(detected, probs[detected])
    )
    top5 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
    print(
        "Top-5: "
        + ", ".join("{}={:.1%}".format(lang, p) for lang, p in top5)
    )

    use_lang = language or detected
    print(f"转写语言: {use_lang}, task={task}")

    result = model.transcribe(
        str(audio_path),
        language=use_lang,
        task=task,
        verbose=False,
    )
    # result["language"] 为 Whisper 最终使用的语言码
    return result, detected, float(probs[detected])


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Whisper 视频语言识别 + 字幕烧录到画面下方"
    )
    p.add_argument("video", type=str, help="输入视频路径")
    p.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help="Whisper 模型大小 (默认 base; 中文建议 small/medium)",
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
        "--srt",
        default=None,
        help="指定 SRT 路径; 与 --burn-only 一起用, 或覆盖默认字幕文件",
    )
    p.add_argument(
        "--box",
        action="store_true",
        help="字幕用黑色不透明底框; 默认透明底 + 描边",
    )
    return p


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)
    video_path = Path(args.video).expanduser().resolve()
    if not video_path.is_file():
        print(f"找不到视频: {video_path}", file=sys.stderr)
        return 1

    _require_ffmpeg()

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

    if args.burn_only:
        if not srt_path.is_file():
            print(f"找不到 SRT: {srt_path}", file=sys.stderr)
            return 1
        srt_body = srt_path.read_text(encoding="utf-8")
        if not srt_body.strip():
            print("SRT 为空, 跳过烧录", file=sys.stderr)
            return 2
        print(f"仅烧录 (复用 {srt_path.name}) -> {subtitled_path.name}")
        burn_subtitles(
            video_path,
            srt_path,
            subtitled_path,
            font_size=args.font_size,
            margin_v=args.margin_v,
            box=args.box,
        )
        print(f"完成: {subtitled_path}")
        return 0

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
        "model": args.model,
        "device": device,
        "detected_language": detected,
        "detected_confidence": conf,
        "used_language": used_lang,
        "task": args.task,
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

    if args.no_burn:
        print("已跳过烧录 (--no-burn)")
        return 0

    if not srt_body.strip():
        print("无有效字幕段, 跳过烧录")
        return 2

    print(f"烧录字幕到视频下方 -> {subtitled_path.name}")
    burn_subtitles(
        video_path,
        srt_path,
        subtitled_path,
        font_size=args.font_size,
        margin_v=args.margin_v,
        box=args.box,
    )
    print(f"完成: {subtitled_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
