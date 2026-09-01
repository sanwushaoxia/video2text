from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from video2text.ffmpeg_util import extract_stereo_audio
from video2text.gender_split import split_vocals_by_gender
from video2text.media import _extract_audio_clip, get_media_duration
from video2text.speaker_stems import split_vocals_by_speaker
from video2text.srt import _is_meaningful_subtitle, parse_srt
from video2text.diarization import DEFAULT_DIARIZATION_MODEL
from video2text.speaker_embedding import DEFAULT_ANIME_EMBEDDING_MODEL
from video2text.bs_roformer_split import (
    DEFAULT_ROFORMER_ENSEMBLE_PRESET,
    DEFAULT_ROFORMER_INST_MODEL,
    DEFAULT_ROFORMER_MODE,
    DEFAULT_ROFORMER_MODEL,
    DEFAULT_ROFORMER_VOCALS_MODEL,
    separate_vocals_bs_roformer,
)


class VocalAssets:
    instrumental: Path | None = None
    vocals: Path | None = None
    voice_ref: Path | None = None
    voice_ref_text: str | None = None


@dataclass
class MultiStemResult:
    instrumental: Path
    vocals: Path
    speaker_paths: dict[str, Path]
    report: Path


@dataclass
class ThreeStemResult:
    instrumental: Path
    vocals: Path
    male_vocals: Path
    female_vocals: Path
    report: Path

def _find_reference_window(
    segments: list[dict],
    min_sec: float = 5.0,
    max_sec: float = 15.0,
) -> tuple[float, float, list[dict]]:
    """从字幕分段中选取参考窗口 (优先较长对白段)。"""
    usable = [
        seg
        for seg in segments
        if _is_meaningful_subtitle((seg.get("text") or "").strip())
        and float(seg.get("end", 0)) - float(seg.get("start", 0)) >= 0.4
    ]
    if not usable:
        usable = [seg for seg in segments if float(seg.get("end", 0)) > float(seg.get("start", 0))]
    if not usable:
        raise RuntimeError("无法从字幕分段中选取参考音频窗口")

    usable.sort(
        key=lambda s: float(s.get("end", 0)) - float(s.get("start", 0)),
        reverse=True,
    )
    picked: list[dict] = []
    total = 0.0
    for seg in usable:
        dur = float(seg.get("end", 0)) - float(seg.get("start", 0))
        if picked and total + dur > max_sec:
            break
        picked.append(seg)
        total += dur
        if total >= min_sec:
            break

    if not picked:
        picked = [usable[0]]
    picked.sort(key=lambda s: float(s.get("start", 0)))
    start = float(picked[0].get("start", 0))
    end = float(picked[-1].get("end", start))
    if end - start < 1.0:
        end = start + min(max_sec, max(min_sec, 5.0))
    if end - start > max_sec:
        end = start + max_sec
    return start, end, picked

def extract_voice_reference(
    vocals_wav: Path,
    segments: list[dict],
    out_path: Path,
    ref_text_segments: list[dict] | None = None,
    min_sec: float = 5.0,
    max_sec: float = 15.0,
) -> tuple[Path, str]:
    """从人声轨截取参考片段, 返回 (路径, 参考文本)。"""
    start, end, _ = _find_reference_window(segments, min_sec, max_sec)
    _extract_audio_clip(vocals_wav, out_path, start, end)

    text_source = ref_text_segments or segments
    ref_parts: list[str] = []
    for seg in text_source:
        seg_start = float(seg.get("start", 0))
        seg_end = float(seg.get("end", seg_start))
        if seg_end <= start or seg_start >= end:
            continue
        text = (seg.get("text") or "").strip()
        if text:
            ref_parts.append(text)
    ref_text = " ".join(ref_parts).strip()
    print(
        f"参考音频: {out_path.name} ({end - start:.1f}s, "
        f"{'有' if ref_text else '无'}参考文本)"
    )
    return out_path, ref_text

def _build_roformer_meta(
    *,
    roformer_model: str,
    roformer_mode: str,
    ensemble_preset: str | None,
    vocals_model: str | None,
    inst_model: str | None,
    overlap: int | None,
) -> dict:
    meta: dict = {
        "backend": "bs_roformer",
        "mode": roformer_mode,
    }
    if roformer_mode == "ensemble" and ensemble_preset:
        meta["ensemble_preset"] = ensemble_preset
    elif roformer_mode == "dual":
        meta["vocals_model"] = vocals_model or DEFAULT_ROFORMER_VOCALS_MODEL
        meta["inst_model"] = inst_model or DEFAULT_ROFORMER_INST_MODEL
    else:
        meta["model"] = roformer_model
    if overlap is not None:
        meta["overlap"] = overlap
    return meta


def _separate_vocal_stems(
    audio_wav: Path,
    work_dir: Path,
    *,
    roformer_model: str = DEFAULT_ROFORMER_MODEL,
    roformer_mode: str = DEFAULT_ROFORMER_MODE,
    roformer_ensemble_preset: str | None = DEFAULT_ROFORMER_ENSEMBLE_PRESET,
    roformer_vocals_model: str | None = None,
    roformer_inst_model: str | None = None,
    roformer_overlap: int | None = None,
) -> tuple[Path, Path]:
    """BS-RoFormer 双轨分离: 返回 (no_vocals/instrumental, vocals)。"""
    return separate_vocals_bs_roformer(
        audio_wav,
        work_dir,
        model_filename=roformer_model,
        roformer_mode=roformer_mode,
        ensemble_preset=roformer_ensemble_preset,
        vocals_model=roformer_vocals_model,
        inst_model=roformer_inst_model,
        overlap=roformer_overlap,
    )


def separate_three_stems(
    audio_wav: Path,
    out_dir: Path,
    stem: str,
    work_dir: Path,
    *,
    split_mode: str = "whisper_f0",
    srt_path: Path | None = None,
    f0_threshold: float = 200.0,
    adaptive_threshold: bool = True,
    min_voiced_ratio: float = 0.25,
    whisper_model: str = "base",
    language: str | None = None,
    whisper_device: str | None = None,
    min_speakers: int = 1,
    max_speakers: int = 4,
    hf_token: str | None = None,
    roformer_model: str = DEFAULT_ROFORMER_MODEL,
    roformer_mode: str = DEFAULT_ROFORMER_MODE,
    roformer_ensemble_preset: str | None = DEFAULT_ROFORMER_ENSEMBLE_PRESET,
    roformer_vocals_model: str | None = None,
    roformer_inst_model: str | None = None,
    roformer_overlap: int | None = None,
    speaker_map_path: Path | None = None,
) -> ThreeStemResult:
    """分离背景/人声, 再自动切分男/女两轨。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    instrumental, vocals = _separate_vocal_stems(
        audio_wav,
        work_dir,
        roformer_model=roformer_model,
        roformer_mode=roformer_mode,
        roformer_ensemble_preset=roformer_ensemble_preset,
        roformer_vocals_model=roformer_vocals_model,
        roformer_inst_model=roformer_inst_model,
        roformer_overlap=roformer_overlap,
    )

    separator_meta = _build_roformer_meta(
        roformer_model=roformer_model,
        roformer_mode=roformer_mode,
        ensemble_preset=roformer_ensemble_preset,
        vocals_model=roformer_vocals_model,
        inst_model=roformer_inst_model,
        overlap=roformer_overlap,
    )

    bgm_out = out_dir / f"{stem}_no_vocals.wav"
    vocals_out = out_dir / f"{stem}_vocals.wav"
    male_out = out_dir / f"{stem}_male_vocals.wav"
    female_out = out_dir / f"{stem}_female_vocals.wav"
    report_out = out_dir / f"{stem}_split_report.json"

    shutil.copy2(instrumental, bgm_out)
    shutil.copy2(vocals, vocals_out)
    print(f"已写入: {bgm_out}")
    print(f"已写入: {vocals_out}")

    split_vocals_by_gender(
        vocals_out,
        male_out,
        female_out,
        report_out,
        split_mode=split_mode,
        srt_path=srt_path,
        f0_threshold=f0_threshold,
        adaptive_threshold=adaptive_threshold,
        min_voiced_ratio=min_voiced_ratio,
        whisper_model=whisper_model,
        language=language,
        whisper_device=whisper_device,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        hf_token=hf_token,
        speaker_map_path=speaker_map_path,
        separator_meta=separator_meta,
    )

    return ThreeStemResult(
        instrumental=bgm_out,
        vocals=vocals_out,
        male_vocals=male_out,
        female_vocals=female_out,
        report=report_out,
    )


def separate_multi_stems(
    audio_wav: Path,
    out_dir: Path,
    stem: str,
    work_dir: Path,
    *,
    split_mode: str = "diarize",
    min_speakers: int = 1,
    max_speakers: int = 4,
    hf_token: str | None = None,
    diarization_model: str = DEFAULT_DIARIZATION_MODEL,
    embedding_model: str = DEFAULT_ANIME_EMBEDDING_MODEL,
    use_embedding: bool = True,
    bss_backend: str = "auto",
    roformer_model: str = DEFAULT_ROFORMER_MODEL,
    roformer_mode: str = DEFAULT_ROFORMER_MODE,
    roformer_ensemble_preset: str | None = DEFAULT_ROFORMER_ENSEMBLE_PRESET,
    roformer_vocals_model: str | None = None,
    roformer_inst_model: str | None = None,
    roformer_overlap: int | None = None,
) -> MultiStemResult:
    """分离背景/人声, 再按日记化输出 N 条 speaker 轨。"""
    if split_mode not in ("diarize", "diarize_bss"):
        raise ValueError(
            f"multi-stems 仅支持 diarize/diarize_bss, 收到: {split_mode}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    instrumental, vocals = _separate_vocal_stems(
        audio_wav,
        work_dir,
        roformer_model=roformer_model,
        roformer_mode=roformer_mode,
        roformer_ensemble_preset=roformer_ensemble_preset,
        roformer_vocals_model=roformer_vocals_model,
        roformer_inst_model=roformer_inst_model,
        roformer_overlap=roformer_overlap,
    )

    separator_meta = _build_roformer_meta(
        roformer_model=roformer_model,
        roformer_mode=roformer_mode,
        ensemble_preset=roformer_ensemble_preset,
        vocals_model=roformer_vocals_model,
        inst_model=roformer_inst_model,
        overlap=roformer_overlap,
    )

    bgm_out = out_dir / f"{stem}_no_vocals.wav"
    vocals_out = out_dir / f"{stem}_vocals.wav"
    report_out = out_dir / f"{stem}_split_report.json"

    shutil.copy2(instrumental, bgm_out)
    shutil.copy2(vocals, vocals_out)
    print(f"已写入: {bgm_out}")
    print(f"已写入: {vocals_out}")

    _, speaker_paths = split_vocals_by_speaker(
        vocals_out,
        out_dir,
        stem,
        report_out,
        split_mode=split_mode,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        hf_token=hf_token,
        diarization_model=diarization_model,
        embedding_model=embedding_model,
        use_embedding=use_embedding,
        bss_backend=bss_backend,
        work_dir=work_dir,
        separator_meta=separator_meta,
    )

    return MultiStemResult(
        instrumental=bgm_out,
        vocals=vocals_out,
        speaker_paths=speaker_paths,
        report=report_out,
    )


def _load_ref_text_segments(out_dir: Path, video_stem: str, fallback: list[dict]) -> list[dict]:
    """优先使用原语言转写 SRT 作为 GPT-SoVITS 参考文本来源。"""
    for name in (f"{video_stem}.srt", f"{video_stem}_ja.srt"):
        candidate = out_dir / name
        if candidate.is_file():
            segs = parse_srt(candidate.read_text(encoding="utf-8"))
            if segs:
                return segs
    return fallback

def prepare_vocal_assets(
    video_path: Path,
    out_dir: Path,
    video_stem: str,
    timing_segments: list[dict],
    *,
    voice_ref_override: str | None,
    voice_ref_text_override: str | None,
    ref_text_segments: list[dict] | None,
    work_dir: Path,
    need_stems: bool,
    need_voice_ref: bool,
    roformer_model: str = DEFAULT_ROFORMER_MODEL,
    roformer_mode: str = DEFAULT_ROFORMER_MODE,
    roformer_ensemble_preset: str | None = DEFAULT_ROFORMER_ENSEMBLE_PRESET,
    roformer_vocals_model: str | None = None,
    roformer_inst_model: str | None = None,
    roformer_overlap: int | None = None,
) -> VocalAssets:
    """分离人声/伴奏并可选提取音色参考片段。"""
    assets = VocalAssets()
    if not need_stems and not need_voice_ref:
        return assets

    source_audio = work_dir / "source_audio.wav"
    print("提取原视频音轨...")
    extract_stereo_audio(video_path, source_audio)

    if need_stems:
        instrumental, vocals = _separate_vocal_stems(
            source_audio,
            work_dir,
            roformer_model=roformer_model,
            roformer_mode=roformer_mode,
            roformer_ensemble_preset=roformer_ensemble_preset,
            roformer_vocals_model=roformer_vocals_model,
            roformer_inst_model=roformer_inst_model,
            roformer_overlap=roformer_overlap,
        )
        assets.instrumental = instrumental
        assets.vocals = vocals
        saved_vocals = out_dir / f"{video_stem}_vocals.wav"
        shutil.copy2(vocals, saved_vocals)
        assets.vocals = saved_vocals
        print(f"已保存人声轨: {saved_vocals}")

    if need_voice_ref:
        if voice_ref_override:
            ref_path = Path(voice_ref_override).expanduser().resolve()
            if not ref_path.is_file():
                raise FileNotFoundError(f"找不到参考音频: {ref_path}")
            assets.voice_ref = ref_path
            assets.voice_ref_text = voice_ref_text_override
            print(f"使用指定参考音频: {ref_path.name}")
        else:
            vocals_path = assets.vocals
            if vocals_path is None:
                _, vocals_path = _separate_vocal_stems(
                    source_audio,
                    work_dir,
                    roformer_model=roformer_model,
                    roformer_mode=roformer_mode,
                    roformer_ensemble_preset=roformer_ensemble_preset,
                    roformer_vocals_model=roformer_vocals_model,
                    roformer_inst_model=roformer_inst_model,
                    roformer_overlap=roformer_overlap,
                )
                assets.vocals = vocals_path
            ref_out = out_dir / f"{video_stem}_voice_ref.wav"
            ref_segments = ref_text_segments or timing_segments
            _, ref_text = extract_voice_reference(
                vocals_path,
                timing_segments,
                ref_out,
                ref_text_segments=ref_segments,
                min_sec=3.0,
                max_sec=10.0,
            )
            assets.voice_ref = ref_out
            assets.voice_ref_text = voice_ref_text_override or ref_text

    return assets
