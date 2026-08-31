from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from video2text.ffmpeg_util import extract_stereo_audio
from video2text.gender_split import split_vocals_by_gender
from video2text.media import _extract_audio_clip, get_media_duration
from video2text.srt import _is_meaningful_subtitle, parse_srt
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

def _separate_vocal_stems_demucs(
    audio_wav: Path,
    work_dir: Path,
    *,
    demucs_model: str = "htdemucs",
    demucs_shifts: int = 1,
) -> tuple[Path, Path]:
    """Demucs 双轨分离: 返回 (no_vocals, vocals)。"""
    import torch

    demucs_out = work_dir / "demucs"
    demucs_out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"分离人声与背景音乐 (Demucs {demucs_model}, shifts={demucs_shifts}, "
        f"device={device}, 首次运行会下载模型)..."
    )
    cmd = [
        sys.executable,
        "-m",
        "demucs.separate",
        "--two-stems",
        "vocals",
        "-n",
        demucs_model,
        "-d",
        device,
        "-o",
        str(demucs_out),
    ]
    if demucs_shifts > 1:
        cmd.extend(["--shifts", str(demucs_shifts)])
    cmd.append(str(audio_wav))
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        stderr = (proc.stderr or b"").decode("utf-8", errors="replace")
        raise RuntimeError(f"人声分离失败 (demucs):\n{stderr[-2000:]}")

    stem_dir = demucs_out / demucs_model / audio_wav.stem
    instrumental = stem_dir / "no_vocals.wav"
    vocals = stem_dir / "vocals.wav"
    if not instrumental.is_file() or not vocals.is_file():
        inst_found = list((demucs_out / demucs_model).rglob("no_vocals.wav"))
        voc_found = list((demucs_out / demucs_model).rglob("vocals.wav"))
        if not inst_found or not voc_found:
            raise RuntimeError("未找到 demucs 输出的人声/伴奏轨")
        instrumental = inst_found[0]
        vocals = voc_found[0]
    return instrumental, vocals


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
    separator_backend: str = "bs_roformer",
    demucs_model: str = "htdemucs",
    demucs_shifts: int = 1,
    roformer_model: str = DEFAULT_ROFORMER_MODEL,
    roformer_mode: str = DEFAULT_ROFORMER_MODE,
    roformer_ensemble_preset: str | None = DEFAULT_ROFORMER_ENSEMBLE_PRESET,
    roformer_vocals_model: str | None = None,
    roformer_inst_model: str | None = None,
    roformer_overlap: int | None = None,
) -> tuple[Path, Path]:
    """双轨分离 dispatcher: 返回 (no_vocals/instrumental, vocals)。"""
    if separator_backend == "bs_roformer":
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
    if separator_backend != "demucs":
        raise ValueError(f"未知 separator_backend: {separator_backend}")
    return _separate_vocal_stems_demucs(
        audio_wav,
        work_dir,
        demucs_model=demucs_model,
        demucs_shifts=demucs_shifts,
    )


def separate_three_stems(
    audio_wav: Path,
    out_dir: Path,
    stem: str,
    work_dir: Path,
    *,
    split_mode: str = "whisper_f0",
    gender_backend: str = "slice",
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
    separator_backend: str = "bs_roformer",
    demucs_model: str = "htdemucs",
    demucs_shifts: int = 1,
    roformer_model: str = DEFAULT_ROFORMER_MODEL,
    roformer_mode: str = DEFAULT_ROFORMER_MODE,
    roformer_ensemble_preset: str | None = DEFAULT_ROFORMER_ENSEMBLE_PRESET,
    roformer_vocals_model: str | None = None,
    roformer_inst_model: str | None = None,
    roformer_overlap: int | None = None,
    slice_pad_ms: int = 120,
    slice_pad_end_ms: int | None = None,
    align_short_segments: bool | None = None,
    fill_gap_ms: int = 400,
    shout_min_ms: int = 600,
    shout_tail_pad_ms: int = 250,
    shout_all_islands: bool = False,
    recover_window_vocals: bool | None = None,
    speaker_map_path: Path | None = None,
    recover_vocal_bleed: bool | None = None,
    bleed_leak_ratio: float = 0.70,
    bleed_island_threshold: float = 0.12,
    bleed_min_nv_voc_ratio: float = 1.5,
    bleed_min_excess_ratio: float = 0.15,
    bleed_bgm_attenuate: float = 0.85,
    bleed_fade_ms: float = 8.0,
) -> ThreeStemResult:
    """分离背景/人声, 再自动切分男/女两轨。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    instrumental, vocals = _separate_vocal_stems(
        audio_wav,
        work_dir,
        separator_backend=separator_backend,
        demucs_model=demucs_model,
        demucs_shifts=demucs_shifts,
        roformer_model=roformer_model,
        roformer_mode=roformer_mode,
        roformer_ensemble_preset=roformer_ensemble_preset,
        roformer_vocals_model=roformer_vocals_model,
        roformer_inst_model=roformer_inst_model,
        roformer_overlap=roformer_overlap,
    )

    separator_meta = {"backend": separator_backend}
    if separator_backend == "bs_roformer":
        separator_meta = _build_roformer_meta(
            roformer_model=roformer_model,
            roformer_mode=roformer_mode,
            ensemble_preset=roformer_ensemble_preset,
            vocals_model=roformer_vocals_model,
            inst_model=roformer_inst_model,
            overlap=roformer_overlap,
        )
    else:
        separator_meta["model"] = demucs_model
        separator_meta["demucs_shifts"] = demucs_shifts

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
        gender_backend=gender_backend,
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
        work_dir=work_dir,
        slice_pad_ms=slice_pad_ms,
        slice_pad_end_ms=slice_pad_end_ms,
        align_short_segments=align_short_segments,
        fill_gap_ms=fill_gap_ms,
        shout_min_ms=shout_min_ms,
        shout_tail_pad_ms=shout_tail_pad_ms,
        shout_all_islands=shout_all_islands,
        recover_window_vocals=recover_window_vocals,
        speaker_map_path=speaker_map_path,
        no_vocals_path=bgm_out,
        recover_vocal_bleed=recover_vocal_bleed,
        bleed_leak_ratio=bleed_leak_ratio,
        bleed_island_threshold=bleed_island_threshold,
        bleed_min_nv_voc_ratio=bleed_min_nv_voc_ratio,
        bleed_min_excess_ratio=bleed_min_excess_ratio,
        bleed_bgm_attenuate=bleed_bgm_attenuate,
        bleed_fade_ms=bleed_fade_ms,
        separator_meta=separator_meta,
    )

    return ThreeStemResult(
        instrumental=bgm_out,
        vocals=vocals_out,
        male_vocals=male_out,
        female_vocals=female_out,
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
    separator_backend: str = "bs_roformer",
    demucs_model: str = "htdemucs",
    demucs_shifts: int = 1,
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
            separator_backend=separator_backend,
            demucs_model=demucs_model,
            demucs_shifts=demucs_shifts,
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
                    separator_backend=separator_backend,
                    demucs_model=demucs_model,
                    demucs_shifts=demucs_shifts,
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
