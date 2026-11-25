"""重叠语声段盲源分离 (2/3 说话人, MossFormer2 / SepFormer)。"""
from __future__ import annotations

from functools import lru_cache
from itertools import islice, permutations
from math import ceil
from pathlib import Path

import numpy as np

from video2text.performance import timed_stage
from video2text.diarization import OverlapRegion
from video2text.overlap_tracking import Window, track_windows
from video2text.paths import speechbrain_model_dir
from video2text.speaker_embedding import cosine_similarity

MIN_OVERLAP_SEC = 0.2
MAX_OVERLAP_SEC = 8.0
_SPOOL_AUDIO_SECONDS = 600


@lru_cache(maxsize=1)
def _load_mossformer2():
    from clearvoice import ClearVoice

    return ClearVoice(task="speech_separation", model_names=["MossFormer2_SS_16K"])


def _separate_mossformer2(audio: np.ndarray, sr: int, work_dir: Path):
    import soundfile as sf

    clip_path = work_dir / "overlap_clip.wav"
    out_dir = work_dir / "mossformer2_out"
    out_dir.mkdir()
    sf.write(str(clip_path), audio, sr)

    _load_mossformer2()(input_path=str(clip_path), online_write=True, output_path=str(out_dir))

    outputs = sorted(out_dir.rglob("*.wav"))
    if len(outputs) != 2:
        raise RuntimeError(f"MossFormer2 输出轨数异常: {len(outputs)}")
    sources = [sf.read(str(path), dtype="float32") for path in outputs]
    output_sr = sources[0][1]
    if any(samples.ndim != 1 or rate != output_sr for samples, rate in sources):
        raise RuntimeError("MossFormer2 输出声道或采样率不一致")
    return tuple(samples for samples, _ in sources), output_sr


@lru_cache(maxsize=3)
def _load_sepformer(name: str, device: str):
    from speechbrain.inference.separation import SepformerSeparation

    savedir = speechbrain_model_dir(name)
    required = ("hyperparams.yaml", "encoder.ckpt", "decoder.ckpt", "masknet.ckpt")
    source = str(savedir) if all((savedir / f).is_file() for f in required) else f"speechbrain/{name}"
    return SepformerSeparation.from_hparams(
        source=source, savedir=str(savedir), run_opts={"device": device}
    )


def _resample(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return audio
    import librosa

    return librosa.resample(audio, orig_sr=source_sr, target_sr=target_sr)


def _separate_sepformer(audio: np.ndarray, sr: int, speakers: int, model_name: str):
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_sepformer(model_name, device)
    model_sr = int(model.hparams.sample_rate)
    waveform = _resample(audio, sr, model_sr)
    # separate_file 会独立归一化每一路。保留 separate_batch 的相对增益。
    with torch.inference_mode():
        est = model.separate_batch(torch.from_numpy(waveform).unsqueeze(0))
    est = est.detach().cpu().numpy()
    if est.shape != (1, waveform.size, speakers):
        raise RuntimeError(f"SepFormer 输出形状不符合预期: {est.shape}")
    return tuple(est[0].T), model_sr


def _align_sources(sources, model_sr: int, mix: np.ndarray, sr: int) -> tuple[np.ndarray, ...]:
    expected = round(mix.size * model_sr / sr)
    aligned = []
    for source in sources:
        source = np.asarray(source, dtype=np.float32)
        if source.ndim != 1 or not np.isfinite(source).all() or abs(source.size - expected) > 1:
            raise RuntimeError("分离结果时长或波形异常")
        source = _resample(source, model_sr, sr)[:mix.size]
        aligned.append(np.pad(source, (0, mix.size - source.size)))

    # 输出的极性和增益可能不同。以原混音拟合各路，再统一限制峰值。
    matrix = np.column_stack(aligned)
    gains, _, rank, _ = np.linalg.lstsq(matrix, mix, rcond=1e-4)
    if rank != len(aligned) or not np.isfinite(gains).all():
        raise RuntimeError("分离结果缺少独立信号")
    calibrated = matrix * gains
    peak = float(np.max(np.abs(calibrated)))
    if peak < 1e-8:
        raise RuntimeError("分离结果无有效信号")
    calibrated *= min(1.0, 0.99 / peak)
    return tuple(calibrated.T)


def _separate_overlap(
    audio: np.ndarray, sr: int, work_dir: Path, speakers: int, backend: str,
) -> tuple[tuple[np.ndarray, ...], str]:
    backends = ("mossformer2", "sepformer") if backend == "auto" else (backend,)
    if speakers == 3 and backend in ("auto", "sepformer-whamr"):
        backends = ("sepformer",)
    last_error = None
    for name in backends:
        try:
            if name == "mossformer2" and speakers == 2:
                sources, model_sr = _separate_mossformer2(audio, sr, work_dir)
            elif name in ("sepformer", "sepformer-whamr"):
                model_name = f"sepformer-wsj0{speakers}mix" if name == "sepformer" else name
                sources, model_sr = _separate_sepformer(audio, sr, speakers, model_name)
            else:
                raise ValueError(f"{name} 不支持 {speakers} 路分离")
            return _align_sources(sources, model_sr, audio, sr), name
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"{speakers} 路重叠分离失败: {last_error}") from last_error


def separate_overlap_2spk(
    audio: np.ndarray, sr: int, work_dir: Path, *, backend: str = "auto",
) -> tuple[np.ndarray, np.ndarray, str]:
    sources, used = _separate_overlap(audio, sr, work_dir, 2, backend)
    return (*sources, used)


def separate_overlap_3spk(
    audio: np.ndarray, sr: int, work_dir: Path, *, backend: str = "auto",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    sources, used = _separate_overlap(audio, sr, work_dir, 3, backend)
    return (*sources, used)


def _match_sources(sources, sr, active, profiles, embedder, min_similarity):
    scores = []
    for samples in sources:
        emb = embedder.embed_waveform(samples, sr)
        scores.append([cosine_similarity(emb, profiles[speaker]) for speaker in active])

    ranked = []
    for order in permutations(range(len(active))):
        similarities = [row[col] for row, col in zip(scores, order)]
        if all(np.isfinite(score) and score >= min_similarity for score in similarities):
            ranked.append((sum(similarities) / len(similarities), order))
    ranked.sort(reverse=True)
    if not ranked or (len(ranked) > 1 and ranked[0][0] - ranked[1][0] < 0.03):
        return None
    return [active[col] for col in ranked[0][1]]


def _inference_windows(start: int, end: int, length: int, sr: int, cap: int):
    """等分长区间，并在推理时保留少量边界上下文。"""
    context = min(round(0.12 * sr), cap // 8)
    count = ceil((end - start) / (cap - 2 * context))
    if end - start <= cap:
        count = 1
    for i in range(count):
        left = start + (end - start) * i // count
        right = start + (end - start) * (i + 1) // count
        padding = min(context, (cap - (right - left)) // 2)
        yield left, right, max(0, left - padding), min(length, right + padding)


@timed_stage("overlap_separation")
def apply_overlap_separation(
    audio,
    sr: int,
    overlap_regions: list[OverlapRegion],
    assignments: list[dict],
    speaker_profiles: dict[str, np.ndarray],
    embedder,
    work_dir: Path | None,
    *,
    bss_backend: str = "auto",
    min_overlap_sec: float = MIN_OVERLAP_SEC,
    max_overlap_sec: float = MAX_OVERLAP_SEC,
    min_similarity: float = 0.35,
    confirmed: bool = True,
    tracking_anchors: dict | None = None,
    preview_dir: Path | None = None,
) -> dict:
    """仅在每一路均成功归属后提交结果，逐段记录回退原因。"""
    overlap_writes: dict[str, list[tuple[int, int, np.ndarray | Path]]] = {}
    regions = []
    speaker_ids = {turn["speaker_id"] for turn in assignments}
    spool = None
    if work_dir is not None and len(audio) >= sr * _SPOOL_AUDIO_SECONDS:
        spool = work_dir / "overlap_render"

    for index, region in enumerate(overlap_regions):
        start = max(0, round(region.start * sr))
        end = min(len(audio), round(region.end * sr))
        if end <= start:
            continue
        active = sorted(set(region.speaker_ids))
        if not active:
            active = sorted({
                turn["speaker_id"] for turn in assignments
                if turn["start"] < region.end and turn["end"] > region.start
            })
        info = {
            "start_sample": start, "end_sample": end, "speaker_ids": active,
            "status": "review",
        }
        if not confirmed:
            regions.append(dict(info, status="candidate", reason="proximity_only"))
            continue
        if end - start < round(min_overlap_sec * sr) or max_overlap_sec < min_overlap_sec:
            regions.append(info)
            info["reason"] = "duration_out_of_range"
            continue
        if len(active) not in (2, 3):
            regions.append(info)
            info["reason"] = "unsupported_speaker_count"
            continue
        profiles_ready = embedder is not None and set(active).issubset(speaker_ids & speaker_profiles.keys())
        if not profiles_ready and preview_dir is None:
            regions.append(info)
            info["reason"] = "missing_speaker_profiles"
            continue
        if work_dir is None:
            regions.append(info)
            info["reason"] = "missing_work_dir"
            continue

        separate = separate_overlap_2spk if len(active) == 2 else separate_overlap_3spk
        cap = max(1, round(max_overlap_sec * sr))
        windows = enumerate(_inference_windows(start, end, len(audio), sr, cap))
        previous_end = None
        # Keep the lookahead bounded even when a whole recording is an overlap.
        batch_limit = 16 if tracking_anchors is not None else 1
        while (batch := list(islice(windows, batch_limit))):
            pending = []
            for window, (left, right, read_start, read_end) in batch:
                window_info = dict(info, id=f"{index}:{window}", start_sample=left, end_sample=right)
                regions.append(window_info)
                try:
                    *sources, backend = separate(
                        audio[read_start:read_end], sr, work_dir / f"overlap_{index}_{window}",
                        backend=bss_backend,
                    )
                    window_info["backend"] = backend
                    pending.append(Window(left, right, read_start, tuple(sources), window_info))
                except Exception as exc:
                    window_info.update(reason="inference_failed", error=str(exc))

            if profiles_ready and tracking_anchors is not None:
                track_windows(pending, sr, active, speaker_profiles, tracking_anchors, embedder, min_similarity)

            for window in pending:
                left, right, read_start = window.start, window.end, window.read_start
                core, sources, window_info = window.core, window.sources, window.report
                if preview_dir is not None:
                    import soundfile as sf

                    preview_dir.mkdir(parents=True, exist_ok=True)
                    preview = preview_dir / (window_info["id"].replace(":", "_") + ".wav")
                    sf.write(preview, np.column_stack([samples[core] for samples in sources]), sr, subtype="FLOAT")
                    window_info["preview"] = str(preview.resolve())
                if not profiles_ready:
                    window_info["reason"] = "missing_speaker_profiles"
                    continue
                if window_info.get("reason") == "inference_failed":
                    continue
                matched = window_info.get("source_speakers")
                if tracking_anchors is None:
                    try:
                        matched = _match_sources(
                            [samples[core] for samples in sources], sr, active,
                            speaker_profiles, embedder, min_similarity,
                        )
                    except Exception as exc:
                        window_info.update(reason="inference_failed", error=str(exc))
                        continue
                if matched is None:
                    window_info["reason"] = "uncertain_speaker_assignment"
                    continue

                for source, (speaker, samples) in enumerate(zip(matched, sources)):
                    writes = overlap_writes.setdefault(speaker, [])
                    if previous_end == left:
                        previous = writes[-1][2]
                        fade = min(round(0.02 * sr), left - read_start, previous.size)
                        if fade:
                            weight = np.linspace(0, 1, fade, dtype=np.float32)
                            previous[-fade:] = (
                                previous[-fade:] * (1 - weight)
                                + samples[core.start - fade:core.start] * weight
                            )
                    if spool is not None and writes:
                        # Keep the newest patch per speaker writable for the next crossfade.
                        patch_start, patch_end, patch = writes[-1]
                        spool.mkdir(parents=True, exist_ok=True)
                        window_name = window_info["id"].replace(":", "_")
                        path = spool / f"{window_name}_{source}.npy"
                        np.save(path, patch, allow_pickle=False)
                        writes[-1] = (patch_start, min(patch_end, patch_start + len(patch)), path)
                    writes.append((left, right, samples[core].copy()))
                window_info["status"] = "separated"
                window_info["source_speakers"] = matched
                previous_end = right

    accepted = [region for region in regions if region["status"] == "separated"]
    backends = sorted({region["backend"] for region in accepted})
    separated_samples = sum(r["end_sample"] - r["start_sample"] for r in accepted)
    return {
        "overlap_count": len(accepted),
        "overlap_separated_sec": round(separated_samples / sr, 2),
        "backend_used": ", ".join(backends) or None,
        "sample_rate": sr,
        "regions": regions,
        "overlap_writes": overlap_writes,
    }
