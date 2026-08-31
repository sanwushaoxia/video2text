"""
根据字幕时间轴 + 人声轨基频 (F0), 自动推断每句为 male/female, 生成 speaker_map.json。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from video2text.srt import parse_srt
from video2text.f0_analysis import (
    analyze_segment_f0,
    classify_segment_gender,
    compute_adaptive_f0_threshold,
)

def _find_whisper_assets(whisper_dir: Path) -> tuple[Path, Path]:
    """在 *_whisper 目录中查找主 SRT 与人声 wav。"""
    if not whisper_dir.is_dir():
        raise FileNotFoundError(f"目录不存在: {whisper_dir}")

    vocals_candidates = sorted(whisper_dir.glob("*_vocals.wav"))
    if not vocals_candidates:
        raise FileNotFoundError(f"未找到 *_vocals.wav: {whisper_dir}")
    vocals = vocals_candidates[0]

    srt_candidates = [
        p
        for p in sorted(whisper_dir.glob("*.srt"))
        if not p.name.endswith("_zh.srt") and not p.name.endswith("_ja.srt")
    ]
    if not srt_candidates:
        srt_candidates = sorted(whisper_dir.glob("*.srt"))
    if not srt_candidates:
        raise FileNotFoundError(f"未找到 .srt: {whisper_dir}")
    srt = srt_candidates[0]
    return srt, vocals


def _text_gender_hint(text: str) -> str | None:
    """基于台词内容的轻量规则 (辅助 F0, 尤其短句/喊名)。"""
    t = text.strip()
    if not t:
        return None

    if t in ("犬夜叉", "犬夜叉。", "犬夜叉！", "犬夜叉..."):
        return "female"
    if t in ("桔梗", "桔梗。", "桔梗！", "桔梗..."):
        return "male"

    # --- 男声优先 (避免被下方女声泛规则覆盖) ---
    if t == "可是":
        return "male"
    if "我变成凡人" in t and ("你会" in t or "怎么样" in t):
        return "male"
    if "我怎么可能忘记" in t:
        return "male"
    if any(kw in t for kw in ("却没能", "没能为你", "没能救", "我没能")):
        return "male"
    if "桔梗" in t and "你是我" in t:
        return "male"
    if "一桔梗" in t or t.startswith("桔梗我"):
        return "male"
    if "要是那时候我变成凡人" in t:
        return "male"
    if "桔梗让我们" in t and "不要伤心" in t:
        return "male"
    if "她会一直保佑" in t or ("保佑我们" in t and "她" in t):
        return "male"

    # --- 女声 ---
    if "但是你来了" in t or (t.startswith("但是") and "你来了" in t):
        return "female"
    if "这样就够了" in t:
        return "female"
    if "犬夜叉" in t and "你哭" in t:
        return "female"
    if "犬夜叉" in t and "变成凡人" in t:
        return "female"
    if "我终于成为平凡的女人" in t:
        return "female"
    if "犬夜叉" in t and any(kw in t for kw in ("吗", "？", "?", "记得", "之前")):
        return "female"
    if "犬夜叉" in t and "要是" in t and "变成凡人" not in t:
        return "female"
    if any(kw in t for kw in ("守玉之人", "平凡的女人", "若是玉")):
        return "female"
    if any(kw in t for kw in ("桔梗的灵魂", "第一次看到")):
        return "female"
    if "告别" in t and "桔梗让" not in t:
        return "female"
    if t == "好温暖" or t.startswith("好温暖"):
        return "female"

    return None


def _is_strong_text_hint(text: str, hint: str) -> bool:
    """高置信台词规则: 可覆盖明显的 F0 误判。"""
    t = text.strip()
    if hint == "male":
        return (
            t in ("桔梗", "桔梗。", "桔梗！")
            or t == "可是"
            or ("我变成凡人" in t and ("你会" in t or "怎么样" in t))
            or "我怎么可能忘记" in t
            or any(
                kw in t
                for kw in (
                    "却没能",
                    "没能为你",
                    "没能救",
                    "我没能",
                    "你是我第一个",
                    "一桔梗",
                    "要是那时候我变成凡人",
                    "桔梗让我们",
                    "她会一直保佑",
                )
            )
        )
    if hint == "female":
        return (
            t in ("犬夜叉", "犬夜叉。", "犬夜叉！")
            or "但是你来了" in t
            or "这样就够了" in t
            or any(
                kw in t
                for kw in (
                    "守玉",
                    "若是玉",
                    "平凡的女人",
                    "你哭起来",
                    "还记得吗",
                    "很久之前",
                    "第一次看到",
                    "我终于成为平凡的女人",
                    "桔梗的灵魂",
                )
            )
            or ("告别" in t and "桔梗让" not in t)
            or t == "好温暖"
            or t.startswith("好温暖")
        )
    return False


_SHOUT_NAME_VARIANTS = frozenset(
    {
        "桔梗",
        "桔梗。",
        "桔梗！",
        "桔梗...",
        "犬夜叉",
        "犬夜叉。",
        "犬夜叉！",
        "犬夜叉...",
    }
)


def is_shout_name(text: str) -> bool:
    """纯喊名 (桔梗 / 犬夜叉 及标点变体)。"""
    t = text.strip()
    if t in _SHOUT_NAME_VARIANTS:
        return True
    core = t.rstrip("。！？!?….")
    return core in ("桔梗", "犬夜叉")


def is_phrase_long_window_cue(seg: dict, *, long_window_sec: float = 5.0) -> bool:
    """情感/长句: 不应走 shout 单 burst 收紧。"""
    text = (seg.get("text") or "").strip()
    if not text or is_shout_name(text):
        return False
    start = float(seg.get("start", 0.0))
    end = float(seg.get("end", start))
    duration = max(0.0, end - start)
    hint = _text_gender_hint(text)
    if hint and _is_strong_text_hint(text, hint):
        return True
    return len(text) > 4 and duration > long_window_sec


def classify_cue_align_type(seg: dict, *, long_window_sec: float = 5.0) -> str:
    """
    对齐策略: shout | phrase_long_window | short_dialogue | none
    """
    text = (seg.get("text") or "").strip()
    start = float(seg.get("start", 0.0))
    end = float(seg.get("end", start))
    duration = max(0.0, end - start)

    if is_phrase_long_window_cue(seg, long_window_sec=long_window_sec):
        return "phrase_long_window"
    if is_shout_name(text) or (duration > long_window_sec and len(text) <= 4):
        return "shout"
    if duration <= 3.0:
        return "short_dialogue"
    return "none"


def apply_speaker_map_override(
    assignments: list[dict],
    report: list[dict],
    speaker_map_path: Path,
) -> dict[str, str]:
    """按 speaker_map.json 覆盖 gender 标签 (不改时间轴)。"""
    raw = json.loads(speaker_map_path.read_text(encoding="utf-8"))
    mapping = {str(k): str(v) for k, v in raw.items()}
    for row in assignments:
        key = str(row.get("index", ""))
        if key in mapping:
            row["gender"] = mapping[key]
    for row in report:
        key = str(row.get("index", ""))
        if key in mapping:
            row["gender"] = mapping[key]
            prev = row.get("reason") or ""
            if "speaker_map" not in prev:
                row["reason"] = f"{prev} + speaker_map".strip(" +")
    return mapping


def _refine_mapping_with_hints(
    mapping: dict[str, str],
    report: list[dict],
    *,
    f0_threshold: float,
) -> None:
    """用台词规则修正 F0 不确定或与内容明显冲突的条目。"""
    for row in report:
        key = row["index"]
        hint = _text_gender_hint(row.get("text") or "")
        if not hint:
            continue
        f0 = row.get("f0_hz")
        voiced = float(row.get("voiced_ratio") or 0.0)
        strong = _is_strong_text_hint(row.get("text") or "", hint)
        if strong:
            mapping[key] = hint
            row["gender"] = hint
            row["reason"] = "text_hint (strong)"
            continue
        if f0 is None or voiced < 0.35:
            mapping[key] = hint
            row["gender"] = hint
            row["reason"] = "text_hint (override)"
            continue
        audio_gender = "female" if f0 >= f0_threshold else "male"
        if hint != audio_gender and abs(f0 - f0_threshold) < 45:
            mapping[key] = hint
            row["gender"] = hint
            row["reason"] = f"text_hint (F0={f0:.0f}Hz borderline)"


def _estimate_confidence(row: dict, *, f0_threshold: float) -> float:
    """估计单条判定置信度 0~1。"""
    reason = row.get("reason") or ""
    if reason == "text_hint (strong)":
        return 0.95
    if reason.startswith("text_hint"):
        return 0.82
    if reason == "dialogue_alternate":
        return 0.68

    f0 = row.get("f0_hz")
    voiced = float(row.get("voiced_ratio") or 0.0)
    if f0 is None or voiced < 0.2:
        return 0.2
    if voiced < 0.32:
        return 0.35

    margin = abs(float(f0) - f0_threshold)
    if margin >= 70:
        return 0.88
    if margin >= 45:
        return 0.65
    return 0.42


def _smooth_dialogue_turns(
    mapping: dict[str, str],
    report: list[dict],
    *,
    f0_threshold: float,
    max_confidence: float = 0.52,
) -> None:
    """
    双人对话场景: 对 F0 低置信条目, 参考相邻高置信句的交替说话规律修正。
    仅在前/后邻句之一置信度足够高且当前条无 strong 文本规则时生效。
    """
    confidences = [_estimate_confidence(row, f0_threshold=f0_threshold) for row in report]
    n = len(report)

    for i in range(n):
        if confidences[i] > max_confidence:
            continue
        if (report[i].get("reason") or "").startswith("text_hint (strong)"):
            continue

        text_hint = _text_gender_hint(report[i].get("text") or "")
        if text_hint and _is_strong_text_hint(report[i].get("text") or "", text_hint):
            continue

        f0 = report[i].get("f0_hz")
        voiced = float(report[i].get("voiced_ratio") or 0.0)
        current_g = report[i]["gender"]
        if f0 is not None and voiced >= 0.28:
            audio_g = "female" if float(f0) >= f0_threshold else "male"
            margin = abs(float(f0) - f0_threshold)
            if margin >= 38 and audio_g == current_g:
                continue
        # 极低音高在动漫 BGM 下常为误判, 允许交替修正
        suspicious_low = f0 is not None and float(f0) < 115 and voiced >= 0.2
        borderline = f0 is not None and abs(float(f0) - f0_threshold) < 42
        if not suspicious_low and not borderline and confidences[i] > 0.38:
            continue

        for neighbor, nc in (
            (i - 1, confidences[i - 1] if i > 0 else 0.0),
            (i + 1, confidences[i + 1] if i < n - 1 else 0.0),
        ):
            if neighbor < 0 or neighbor >= n or nc < 0.72:
                continue
            if (report[neighbor].get("reason") or "").startswith("text_hint (strong)"):
                nc = max(nc, 0.9)
            if nc < 0.72:
                continue
            prev_g = report[neighbor]["gender"]
            alt = "female" if prev_g == "male" else "male"
            if text_hint and text_hint != alt and not suspicious_low:
                continue
            key = report[i]["index"]
            mapping[key] = alt
            report[i]["gender"] = alt
            report[i]["reason"] = f"dialogue_alternate (ref #{report[neighbor]['index']})"
            confidences[i] = 0.68
            break


def _classify_gender(
    f0: float | None,
    voiced_ratio: float,
    text: str,
    *,
    f0_threshold: float,
    min_voiced_ratio: float,
    use_text_hints: bool,
) -> tuple[str, str]:
    """返回 (gender, reason)。"""
    if not use_text_hints:
        return classify_segment_gender(
            f0,
            voiced_ratio,
            f0_threshold=f0_threshold,
            min_voiced_ratio=min_voiced_ratio,
        )

    hint = _text_gender_hint(text)

    if f0 is not None and voiced_ratio >= min_voiced_ratio:
        audio_gender = "female" if f0 >= f0_threshold else "male"
        if hint and hint != audio_gender and voiced_ratio < 0.45:
            return hint, f"text_hint (F0={f0:.0f}Hz 置信度低)"
        return audio_gender, f"F0={f0:.0f}Hz"

    if hint:
        return hint, "text_hint"

    if f0 is not None:
        audio_gender = "female" if f0 >= f0_threshold else "male"
        return audio_gender, f"F0={f0:.0f}Hz (low_conf)"

    return "female", "default_female"


def build_speaker_map(
    segments: list[dict],
    vocals_path: Path,
    *,
    f0_threshold: float = 200.0,
    min_voiced_ratio: float = 0.25,
    use_text_hints: bool = True,
    use_dialogue_smooth: bool = True,
    sr: int = 16000,
) -> tuple[dict[str, str], list[dict]]:
    import librosa

    audio, _ = librosa.load(vocals_path, sr=sr, mono=True)
    mapping: dict[str, str] = {}
    report: list[dict] = []
    cue_counter = 0

    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        cue_counter += 1
        cue_key = str(seg.get("index", cue_counter))
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        duration = max(0.0, end - start)

        f0, voiced_ratio = analyze_segment_f0(audio, sr, start, end)
        gender, reason = _classify_gender(
            f0,
            voiced_ratio,
            text,
            f0_threshold=f0_threshold,
            min_voiced_ratio=min_voiced_ratio,
            use_text_hints=use_text_hints,
        )
        mapping[cue_key] = gender
        report.append(
            {
                "index": cue_key,
                "start": start,
                "end": end,
                "duration": round(duration, 2),
                "gender": gender,
                "f0_hz": round(f0, 1) if f0 is not None else None,
                "voiced_ratio": round(voiced_ratio, 2),
                "reason": reason,
                "text": text,
            }
        )

    if use_text_hints:
        _refine_mapping_with_hints(
            mapping, report, f0_threshold=f0_threshold
        )
    if use_dialogue_smooth and use_text_hints:
        _smooth_dialogue_turns(
            mapping, report, f0_threshold=f0_threshold
        )

    return mapping, report


def assign_gender_to_segments(
    segments: list[dict],
    vocals_path: Path,
    *,
    f0_threshold: float = 200.0,
    adaptive_threshold: bool = True,
    min_voiced_ratio: float = 0.25,
    use_text_hints: bool = True,
    use_dialogue_smooth: bool = True,
    sr: int = 44100,
) -> tuple[list[dict], list[dict], dict]:
    """
    为分段标注 gender, 供三轨切分使用。
    返回 (assignments, report, meta)；assignments 可直接传入 build_gender_stems。
    """
    import librosa

    audio, _ = librosa.load(vocals_path, sr=sr, mono=True)
    threshold = f0_threshold
    if adaptive_threshold:
        f0_samples: list[float] = []
        for seg in segments:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start))
            if end <= start:
                continue
            f0, voiced = analyze_segment_f0(audio, sr, start, end)
            if f0 is not None and voiced >= min_voiced_ratio * 0.5:
                f0_samples.append(f0)
        threshold = compute_adaptive_f0_threshold(
            f0_samples, fallback=f0_threshold
        )

    _, report = build_speaker_map(
        segments,
        vocals_path,
        f0_threshold=threshold,
        min_voiced_ratio=min_voiced_ratio,
        use_text_hints=use_text_hints,
        use_dialogue_smooth=use_dialogue_smooth,
        sr=sr,
    )

    for row in report:
        row["confidence"] = round(
            _estimate_confidence(row, f0_threshold=threshold), 3
        )

    assignments: list[dict] = []
    coverage_sec = 0.0
    for row in report:
        start = float(row["start"])
        end = float(row["end"])
        coverage_sec += max(0.0, end - start)
        assignments.append(
            {
                "start": start,
                "end": end,
                "gender": row["gender"],
                "index": row.get("index"),
                "speaker_id": row.get("speaker_id"),
                "text": row.get("text", ""),
                "f0_hz": row.get("f0_hz"),
                "voiced_ratio": row.get("voiced_ratio"),
                "reason": row.get("reason"),
                "confidence": row.get("confidence"),
            }
        )

    low_conf = sum(
        1 for row in report if float(row.get("confidence") or 0) < 0.5
    )
    meta = {
        "f0_threshold_requested": f0_threshold,
        "f0_threshold_used": round(threshold, 1),
        "adaptive_threshold": adaptive_threshold,
        "coverage_sec": round(coverage_sec, 2),
        "segment_count": len(report),
        "low_confidence_count": low_conf,
    }
    return assignments, report, meta


def main() -> int:
    parser = argparse.ArgumentParser(
        description="从人声轨自动推断 SRT 各句男/女声, 生成 speaker_map.json",
    )
    parser.add_argument(
        "--srt",
        type=Path,
        default=None,
        help="字幕 SRT 路径",
    )
    parser.add_argument(
        "--vocals",
        type=Path,
        default=None,
        help="Demucs 人声 wav (如 *_vocals.wav)",
    )
    parser.add_argument(
        "--whisper-dir",
        type=Path,
        default=None,
        help="自动查找目录内的 .srt 与 *_vocals.wav",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="输出 JSON 路径 (默认: whisper-dir/speaker_map.json 或与 srt 同目录)",
    )
    parser.add_argument(
        "--f0-threshold",
        type=float,
        default=200.0,
        help="基频阈值 Hz: >= 阈值判女声, 以下判男声 (默认 170)",
    )
    parser.add_argument(
        "--min-voiced-ratio",
        type=float,
        default=0.25,
        help="F0 可信度: 有效浊音帧占比下限 (默认 0.25)",
    )
    parser.add_argument(
        "--no-text-hints",
        action="store_true",
        help="禁用台词规则辅助 (纯 F0 判定)",
    )
    parser.add_argument(
        "--no-dialogue-smooth",
        action="store_true",
        help="禁用双人对话交替平滑 (仅 F0 + 台词规则)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="可选: 写入详细分析报告 JSON",
    )
    args = parser.parse_args()

    srt_path = args.srt
    vocals_path = args.vocals

    if args.whisper_dir:
        auto_srt, auto_vocals = _find_whisper_assets(args.whisper_dir.expanduser().resolve())
        srt_path = srt_path or auto_srt
        vocals_path = vocals_path or auto_vocals

    if not srt_path or not vocals_path:
        parser.error("需要 --srt 与 --vocals, 或指定 --whisper-dir")

    srt_path = srt_path.expanduser().resolve()
    vocals_path = vocals_path.expanduser().resolve()
    if not srt_path.is_file():
        print(f"找不到 SRT: {srt_path}", file=sys.stderr)
        return 1
    if not vocals_path.is_file():
        print(f"找不到人声: {vocals_path}", file=sys.stderr)
        return 1

    out_path = args.output
    if out_path is None:
        if args.whisper_dir:
            out_path = args.whisper_dir.expanduser().resolve() / "speaker_map.json"
        else:
            out_path = srt_path.parent / "speaker_map.json"
    out_path = out_path.expanduser().resolve()

    try:
        import librosa  # noqa: F401
    except ImportError:
        print(
            "缺少 librosa, 请安装: pip install librosa",
            file=sys.stderr,
        )
        return 1

    segments = parse_srt(srt_path.read_text(encoding="utf-8"))
    if not segments:
        print(f"SRT 无有效字幕段: {srt_path}", file=sys.stderr)
        return 1

    print(f"字幕: {srt_path.name} ({len(segments)} 段)")
    print(f"人声: {vocals_path.name}")
    print(f"F0 阈值: {args.f0_threshold} Hz")

    mapping, report = build_speaker_map(
        segments,
        vocals_path,
        f0_threshold=args.f0_threshold,
        min_voiced_ratio=args.min_voiced_ratio,
        use_text_hints=not args.no_text_hints,
        use_dialogue_smooth=not args.no_dialogue_smooth,
    )

    out_path.write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\n已写入: {out_path} ({len(mapping)} 条)\n")
    print(f"{'#':>3}  {'性别':^6}  {'F0':>6}  {'依据':<22}  文本")
    print("-" * 72)
    for row in report:
        f0s = f"{row['f0_hz']:.0f}" if row["f0_hz"] is not None else "  -"
        gzh = "女" if row["gender"] == "female" else "男"
        print(
            f"{row['index']:>3}  {gzh:^6}  {f0s:>6}  {row['reason']:<22}  "
            f"{row['text'][:40]}{'...' if len(row['text']) > 40 else ''}"
        )

    if args.report:
        report_path = args.report.expanduser().resolve()
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\n详细报告: {report_path}")

    male_n = sum(1 for v in mapping.values() if v == "male")
    female_n = len(mapping) - male_n
    print(f"\n统计: 男 {male_n} / 女 {female_n} / 共 {len(mapping)}")
    print("请试听核对; 不准的条目可直接编辑 speaker_map.json 后重新配音。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
