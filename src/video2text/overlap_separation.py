"""重叠语声段盲源分离 (2/3 说话人, MossFormer2 / SepFormer)。"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from video2text.paths import speechbrain_model_dir

MIN_OVERLAP_SEC = 0.2
MAX_OVERLAP_SEC = 8.0


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
            "overlap 分离需要 speechbrain, pip install -r requirements.txt"
        ) from exc

    sf = _require_soundfile()
    clip_path = work_dir / "overlap_clip.wav"
    sf.write(str(clip_path), audio, sr)

    separator = SepformerSeparation.from_hparams(
        source="speechbrain/sepformer-wsj02mix",
        savedir=str(speechbrain_model_dir("sepformer-wsj02mix")),
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    )
    est = separator.separate_file(path=str(clip_path))
    if hasattr(est, "detach"):
        est = est.detach().cpu().numpy()
    else:
        est = np.asarray(est)
    # speechbrain separate_file 返回 (batch=1, T, spks); 兼容旧 (spks, T) / (T, spks)
    if est.ndim == 3 and est.shape[0] == 1:
        est = est[0]
    if est.ndim == 1:
        raise RuntimeError("SepFormer 输出格式异常")
    if est.shape[0] == 2:
        return est[0].astype(np.float32), est[1].astype(np.float32)
    if est.shape[-1] == 2:
        return est[:, 0].astype(np.float32), est[:, 1].astype(np.float32)
    raise RuntimeError(f"SepFormer 输出形状不符合预期: {est.shape}")


def _separate_sepformer_3mix(
    audio: np.ndarray, sr: int, work_dir: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SpeechBrain SepFormer WSJ0-3mix, 输出 3 路。"""
    import torch

    try:
        from speechbrain.inference.separation import SepformerSeparation
    except ImportError as exc:
        raise RuntimeError(
            "3 路 overlap 分离需要 speechbrain, pip install -r requirements.txt"
        ) from exc

    sf = _require_soundfile()
    clip_path = work_dir / "overlap_clip.wav"
    sf.write(str(clip_path), audio, sr)

    separator = SepformerSeparation.from_hparams(
        source="speechbrain/sepformer-wsj03mix",
        savedir=str(speechbrain_model_dir("sepformer-wsj03mix")),
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    )
    est = separator.separate_file(path=str(clip_path))
    if hasattr(est, "detach"):
        est = est.detach().cpu().numpy()
    else:
        est = np.asarray(est)
    # speechbrain separate_file 返回 (batch=1, T, spks); 兼容旧 (spks, T) / (T, spks)
    if est.ndim == 3 and est.shape[0] == 1:
        est = est[0]
    if est.ndim == 1:
        raise RuntimeError("SepFormer 3mix 输出格式异常")
    # 输出形如 (3, T) 或 (T, 3)
    if est.shape[0] == 3:
        return est[0].astype(np.float32), est[1].astype(np.float32), est[2].astype(np.float32)
    if est.shape[-1] == 3:
        return (
            est[:, 0].astype(np.float32),
            est[:, 1].astype(np.float32),
            est[:, 2].astype(np.float32),
        )
    raise RuntimeError(f"SepFormer 3mix 输出形状不符合预期: {est.shape}")


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


def separate_overlap_3spk(
    audio: np.ndarray,
    sr: int,
    work_dir: Path,
    *,
    backend: str = "auto",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """
    分离 3 路重叠语声。
    3 路 backend: sepformer (wsj03mix); mossformer2 不支持 3 路。
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    backends: list[str] = []
    if backend == "auto":
        backends = ["sepformer"]
    elif backend == "sepformer":
        backends = ["sepformer"]
    elif backend == "mossformer2":
        raise RuntimeError("MossFormer2 不支持 3 路分离, 请用 sepformer")
    else:
        backends = [backend]

    last_exc: Exception | None = None
    for name in backends:
        try:
            if name == "sepformer":
                spk0, spk1, spk2 = _separate_sepformer_3mix(audio, sr, work_dir)
            else:
                raise ValueError(f"未知 overlap 3spk backend: {name}")
            return spk0, spk1, spk2, name
        except Exception as exc:
            last_exc = exc
            continue
    raise RuntimeError(f"overlap 3spk 分离失败: {last_exc}") from last_exc


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
    min_overlap_sec: float = MIN_OVERLAP_SEC,
    max_overlap_sec: float = MAX_OVERLAP_SEC,
) -> dict:
    """
    对重叠区间做盲分离, 将样本写入 assignments 的 speaker 轨缓冲。
    根据重叠区间涉及的说话人数量自动选 2 路或 3 路分离。
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
        if dur < min_overlap_sec or dur > max_overlap_sec:
            continue
        g0 = max(0, int(region.start * sr))
        g1 = min(len(audio), int(region.end * sr))
        if g1 - g0 < int(min_overlap_sec * sr):
            continue

        clip = audio[g0:g1]
        active = list(getattr(region, "speaker_ids", []) or [])

        # 3+ 说话人重叠: 用 SepFormer 3mix; 否则 2 路
        use_3spk = len(active) >= 3

        if use_3spk:
            try:
                spk0, spk1, spk2, used = separate_overlap_3spk(
                    clip, sr, work_dir / f"overlap_{count}", backend=bss_backend
                )
                backend_used = used
                count += 1
                total_sec += dur
                min_len = min(len(spk0), len(spk1), len(spk2), g1 - g0)
                sources = [
                    (spk0[:min_len], 0),
                    (spk1[:min_len], 1),
                    (spk2[:min_len], 2),
                ]
            except RuntimeError:
                # 3 路失败则回退 2 路
                use_3spk = False

        if not use_3spk:
            spk0, spk1, used = separate_overlap_2spk(
                clip, sr, work_dir / f"overlap_{count}", backend=bss_backend
            )
            backend_used = used
            count += 1
            total_sec += dur
            min_len = min(len(spk0), len(spk1), g1 - g0)
            sources = [
                (spk0[:min_len], 0),
                (spk1[:min_len], 1),
            ]

        # 用 embedding 把每路归属到已知 speaker
        assigned_ids: list[str | None] = []
        for src_samples, _ in sources:
            if src_samples.size < int(sr * 0.06):
                # <60ms 的源无法可靠 embed, 不送模型 (ECAPA conv 对超短输入会崩)
                assigned_ids.append(None)
                continue
            emb = embedder.embed_waveform(src_samples, sr)
            sid = assign_to_nearest_speaker(emb, speaker_profiles)
            assigned_ids.append(sid)

        # embedding 匹配不上时, 用 region 的 speaker_ids 兜底
        for idx, sid in enumerate(assigned_ids):
            if sid is None and idx < len(active):
                assigned_ids[idx] = active[idx]

        # 确保不重复写入同一 speaker
        seen: set[str] = set()
        for idx, (src_samples, _) in enumerate(sources):
            sid = assigned_ids[idx]
            if sid is None or sid in seen:
                continue
            seen.add(sid)
            overlap_writes.setdefault(sid, []).append(
                (g0, g0 + min_len, src_samples)
            )

    return {
        "overlap_count": count,
        "overlap_separated_sec": round(total_sec, 2),
        "backend_used": backend_used,
        "overlap_writes": overlap_writes,
    }
