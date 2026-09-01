"""重叠语声段 2 说话人盲源分离 (MossFormer2 / SepFormer fallback)。"""
from __future__ import annotations

from pathlib import Path

import numpy as np

MIN_OVERLAP_SEC = 0.2
MAX_OVERLAP_SEC = 3.0


def _require_soundfile():
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("overlap 分离需要 soundfile") from exc
    return sf


def _load_mono_segment(vocals_path: Path, start: float, end: float, target_sr: int = 16000):
    import librosa

    audio, sr = librosa.load(
        vocals_path, sr=target_sr, mono=True, offset=start, duration=max(0.0, end - start)
    )
    return np.asarray(audio, dtype=np.float32), sr


def _separate_mossformer2(audio: np.ndarray, sr: int, work_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """MossFormer2 via ClearVoice (optional dependency)."""
    import librosa

    try:
        from clearvoice import ClearVoice
    except ImportError as exc:
        raise RuntimeError("MossFormer2 需要 clearvoice 包") from exc

    sf = _require_soundfile()
    clip_path = work_dir / "overlap_clip.wav"
    out_dir = work_dir / "mossformer2_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    sf.write(str(clip_path), audio, sr)

    cv = ClearVoice(task="speech_separation", model_names=["MossFormer2_SS_16K"])
    cv(str(clip_path), str(out_dir))

    outputs = sorted(out_dir.glob("*.wav"))
    if len(outputs) < 2:
        raise RuntimeError(f"MossFormer2 输出不完整: {outputs}")
    spk0, _ = librosa.load(outputs[0], sr=sr, mono=True)
    spk1, _ = librosa.load(outputs[1], sr=sr, mono=True)
    return np.asarray(spk0, dtype=np.float32), np.asarray(spk1, dtype=np.float32)


def _separate_sepformer(audio: np.ndarray, sr: int, work_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """SpeechBrain SepFormer WSJ0-2mix fallback。"""
    import torch

    try:
        from speechbrain.inference.separation import SepformerSeparation
    except ImportError as exc:
        raise RuntimeError(
            "overlap 分离需要 speechbrain, pip install -r requirements-separation.txt"
        ) from exc

    sf = _require_soundfile()
    clip_path = work_dir / "overlap_clip.wav"
    sf.write(str(clip_path), audio, sr)

    separator = SepformerSeparation.from_hparams(
        source="speechbrain/sepformer-wsj02mix",
        savedir=str(work_dir / "sepformer_wsj02mix"),
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    )
    est = separator.separate_file(path=str(clip_path))
    if hasattr(est, "detach"):
        est = est.detach().cpu().numpy()
    else:
        est = np.asarray(est)
    if est.ndim == 1:
        raise RuntimeError("SepFormer 输出格式异常")
    if est.shape[0] == 2:
        return est[0].astype(np.float32), est[1].astype(np.float32)
    return est[:, 0].astype(np.float32), est[:, 1].astype(np.float32)


def separate_overlap_2spk(
    audio: np.ndarray,
    sr: int,
    work_dir: Path,
    *,
    backend: str = "auto",
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    分离 2 路重叠语声。
    backend: auto | mossformer2 | sepformer
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    backends = []
    if backend == "auto":
        backends = ["mossformer2", "sepformer"]
    else:
        backends = [backend]

    last_exc: Exception | None = None
    for name in backends:
        try:
            if name == "mossformer2":
                spk0, spk1 = _separate_mossformer2(audio, sr, work_dir)
            elif name == "sepformer":
                spk0, spk1 = _separate_sepformer(audio, sr, work_dir)
            else:
                raise ValueError(f"未知 overlap backend: {name}")
            return spk0, spk1, name
        except Exception as exc:
            last_exc = exc
            continue
    raise RuntimeError(f"overlap 2spk 分离失败: {last_exc}") from last_exc


def apply_overlap_separation(
    vocals_path: Path,
    audio: np.ndarray,
    sr: int,
    overlap_regions: list,
    assignments: list[dict],
    speaker_profiles: dict[str, np.ndarray],
    embedder,
    work_dir: Path,
    *,
    bss_backend: str = "auto",
) -> dict:
    """
    对重叠区间做 2spk 分离, 将样本写入 assignments 的 speaker 轨缓冲。
    返回 stats: overlap_count, overlap_separated_sec, backend_used。
    """
    from video2text.speaker_embedding import assign_to_nearest_speaker

    if not overlap_regions or not speaker_profiles:
        return {"overlap_count": 0, "overlap_separated_sec": 0.0}

    overlap_writes: dict[str, list[tuple[int, int, np.ndarray]]] = {}
    total_sec = 0.0
    backend_used: str | None = None
    count = 0

    for region in overlap_regions:
        dur = region.end - region.start
        if dur < MIN_OVERLAP_SEC or dur > MAX_OVERLAP_SEC:
            continue
        g0 = max(0, int(region.start * sr))
        g1 = min(len(audio), int(region.end * sr))
        if g1 - g0 < int(MIN_OVERLAP_SEC * sr):
            continue

        clip = audio[g0:g1]
        spk0, spk1, used = separate_overlap_2spk(
            clip, sr, work_dir / f"overlap_{count}", backend=bss_backend
        )
        backend_used = used
        count += 1
        total_sec += dur

        min_len = min(len(spk0), len(spk1), g1 - g0)
        emb0 = embedder.embed_waveform(spk0[:min_len], sr)
        emb1 = embedder.embed_waveform(spk1[:min_len], sr)
        id0 = assign_to_nearest_speaker(emb0, speaker_profiles)
        id1 = assign_to_nearest_speaker(emb1, speaker_profiles)

        active = list(getattr(region, "speaker_ids", []) or [])
        if id0 is None and active:
            id0 = active[0]
        if id1 is None and len(active) > 1:
            id1 = active[1]
        elif id1 is None and id0 is not None:
            for sp in active:
                if sp != id0:
                    id1 = sp
                    break

        if id0:
            overlap_writes.setdefault(id0, []).append((g0, g0 + min_len, spk0[:min_len]))
        if id1 and id1 != id0:
            overlap_writes.setdefault(id1, []).append((g0, g0 + min_len, spk1[:min_len]))

    return {
        "overlap_count": count,
        "overlap_separated_sec": round(total_sec, 2),
        "backend_used": backend_used,
        "overlap_writes": overlap_writes,
    }
