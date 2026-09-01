"""说话人 embedding: 动漫 ECAPA (litagin) + 通用 fallback。"""
from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT_ANIME_EMBEDDING_MODEL = (
    "litagin/anime_speaker_embedding_by_va_ecapa_tdnn_groupnorm"
)
FALLBACK_EMBEDDING_MODEL = "speechbrain/spkrec-ecapa-voxceleb"


class SpeakerEmbedder:
    """从音频片段提取说话人 embedding。"""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_ANIME_EMBEDDING_MODEL,
        device: str | None = None,
    ):
        self.model_id = model_id
        self._classifier = None
        self._device = device

    def _ensure_loaded(self):
        if self._classifier is not None:
            return
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError as exc:
            raise RuntimeError(
                "speaker embedding 需要 speechbrain, "
                "请安装: pip install -r requirements-separation.txt"
            ) from exc

        import torch

        run_opts = {"device": self._device or ("cuda" if torch.cuda.is_available() else "cpu")}
        try:
            self._classifier = EncoderClassifier.from_hparams(
                source=self.model_id,
                savedir=f"pretrained_models/{self.model_id.replace('/', '_')}",
                run_opts=run_opts,
            )
        except Exception:
            if self.model_id == FALLBACK_EMBEDDING_MODEL:
                raise
            print(f"  动漫 embedding 加载失败, 回退 {FALLBACK_EMBEDDING_MODEL}")
            self.model_id = FALLBACK_EMBEDDING_MODEL
            self._classifier = EncoderClassifier.from_hparams(
                source=FALLBACK_EMBEDDING_MODEL,
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
                run_opts=run_opts,
            )

    def embed_waveform(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """mono float32 audio → L2-normalized embedding。"""
        import torch

        self._ensure_loaded()
        if audio.size == 0:
            return np.zeros(192, dtype=np.float32)
        wav = torch.from_numpy(np.asarray(audio, dtype=np.float32))
        if wav.dim() > 1:
            wav = wav.mean(dim=-1)
        emb = self._classifier.encode_batch(wav.unsqueeze(0))
        vec = emb.squeeze().detach().cpu().numpy().astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 1e-8:
            vec /= norm
        return vec

    def embed_segment(
        self,
        audio: np.ndarray,
        sr: int,
        start: float,
        end: float,
        *,
        min_samples: int = 1600,
    ) -> np.ndarray:
        g0 = max(0, int(start * sr))
        g1 = min(len(audio), int(end * sr))
        seg = audio[g0:g1]
        if seg.size < min_samples:
            return np.zeros(192, dtype=np.float32)
        return self.embed_waveform(seg, sr)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def build_speaker_profiles(
    embedder: SpeakerEmbedder,
    audio: np.ndarray,
    sr: int,
    turns: list[dict],
) -> dict[str, np.ndarray]:
    """按 speaker_id 聚合 embedding 均值。"""
    buckets: dict[str, list[np.ndarray]] = {}
    for turn in turns:
        speaker = str(turn.get("speaker_id", "unknown"))
        emb = embedder.embed_segment(
            audio, sr, float(turn["start"]), float(turn["end"])
        )
        if emb.size == 0 or float(np.linalg.norm(emb)) < 1e-6:
            continue
        buckets.setdefault(speaker, []).append(emb)

    profiles: dict[str, np.ndarray] = {}
    for speaker, embs in buckets.items():
        mean = np.mean(np.stack(embs, axis=0), axis=0).astype(np.float32)
        norm = float(np.linalg.norm(mean))
        if norm > 1e-8:
            mean /= norm
        profiles[speaker] = mean
    return profiles


def assign_to_nearest_speaker(
    embedding: np.ndarray,
    profiles: dict[str, np.ndarray],
    *,
    min_similarity: float = 0.35,
) -> str | None:
    if not profiles or embedding.size == 0:
        return None
    best_sp: str | None = None
    best_score = -1.0
    for speaker, profile in profiles.items():
        score = cosine_similarity(embedding, profile)
        if score > best_score:
            best_score = score
            best_sp = speaker
    if best_score < min_similarity:
        return None
    return best_sp
