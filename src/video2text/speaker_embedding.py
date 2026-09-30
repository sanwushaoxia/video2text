"""说话人 embedding: 动漫 ECAPA (litagin) + 通用 fallback。"""
from __future__ import annotations

import numpy as np

from video2text.paths import ensure_huggingface_env, speechbrain_model_dir

DEFAULT_ANIME_EMBEDDING_MODEL = (
    "litagin/anime_speaker_embedding_by_va_ecapa_tdnn_groupnorm"
)
FALLBACK_EMBEDDING_MODEL = "speechbrain/spkrec-ecapa-voxceleb"

ANIME_VARIANTS = {
    "va": "litagin/anime_speaker_embedding_by_va_ecapa_tdnn_groupnorm",
    "char": "litagin/anime_speaker_embedding_ecapa_tdnn_groupnorm",
}


def _resolve_backend(model_id: str) -> tuple[str, str | None]:
    """返回 ('anime', variant) 或 ('speechbrain', None)。"""
    mid = model_id.strip().lower()
    if mid in ("va", "char"):
        return "anime", mid
    if "by_va" in mid or mid == ANIME_VARIANTS["va"].lower():
        return "anime", "va"
    if "anime_speaker_embedding" in mid and "by_va" not in mid:
        return "anime", "char"
    return "speechbrain", None


class SpeakerEmbedder:
    """从音频片段提取说话人 embedding。"""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_ANIME_EMBEDDING_MODEL,
        device: str | None = None,
    ):
        self.model_id = model_id
        self._device = device
        self._backend: str | None = None
        self._anime = None
        self._classifier = None

    def _is_loaded(self) -> bool:
        return self._backend is not None

    def _load_anime(self, variant: str) -> None:
        try:
            from anime_speaker_embedding import AnimeSpeakerEmbedding
        except ImportError as exc:
            raise RuntimeError(
                "动漫 embedding 需要 anime_speaker_embedding, "
                "请安装: pip install -r requirements.txt"
            ) from exc

        import torch

        ensure_huggingface_env()
        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._anime = AnimeSpeakerEmbedding(variant=variant, device=device)
        self._backend = "anime"
        self.model_id = ANIME_VARIANTS[variant]

    def _load_speechbrain(self, model_id: str) -> None:
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError as exc:
            raise RuntimeError(
                "speaker embedding 需要 speechbrain, "
                "请安装: pip install -r requirements.txt"
            ) from exc

        import torch

        run_opts = {
            "device": self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        }
        savedir_name = (
            "spkrec-ecapa-voxceleb"
            if model_id == FALLBACK_EMBEDDING_MODEL
            else model_id.replace("/", "_")
        )
        self._classifier = EncoderClassifier.from_hparams(
            source=model_id,
            savedir=str(speechbrain_model_dir(savedir_name)),
            run_opts=run_opts,
        )
        self._backend = "speechbrain"
        self.model_id = model_id

    def _ensure_loaded(self):
        if self._is_loaded():
            return

        backend, variant = _resolve_backend(self.model_id)
        if backend == "anime":
            try:
                self._load_anime(variant or "va")
            except Exception:
                if self.model_id == FALLBACK_EMBEDDING_MODEL:
                    raise
                print(f"  动漫 embedding 加载失败, 回退 {FALLBACK_EMBEDDING_MODEL}")
                self._load_speechbrain(FALLBACK_EMBEDDING_MODEL)
            return

        try:
            self._load_speechbrain(self.model_id)
        except Exception:
            if self.model_id == FALLBACK_EMBEDDING_MODEL:
                raise
            print(f"  embedding 加载失败, 回退 {FALLBACK_EMBEDDING_MODEL}")
            self._load_speechbrain(FALLBACK_EMBEDDING_MODEL)

    def _embed_anime(self, audio: np.ndarray, sr: int) -> np.ndarray:
        import librosa
        import torch
        import torch.nn.functional as F

        wav = np.asarray(audio, dtype=np.float32).ravel()
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)

        with torch.inference_mode():
            x = torch.from_numpy(wav).float().unsqueeze(0).to(self._anime.device)
            emb = self._anime(x).squeeze(0).squeeze(0)
            emb = F.normalize(emb, dim=0)
            return emb.cpu().numpy().astype(np.float32)

    def _embed_speechbrain(self, audio: np.ndarray) -> np.ndarray:
        import torch

        wav = torch.from_numpy(np.asarray(audio, dtype=np.float32))
        if wav.dim() > 1:
            wav = wav.mean(dim=-1)
        emb = self._classifier.encode_batch(wav.unsqueeze(0))
        vec = emb.squeeze().detach().cpu().numpy().astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 1e-8:
            vec /= norm
        return vec

    def embed_waveform(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """mono float32 audio → L2-normalized embedding。"""
        self._ensure_loaded()
        wav = np.asarray(audio)
        if wav.size == 0:
            return np.zeros(192, dtype=np.float32)
        if wav.ndim > 1:
            if wav.shape[0] == 1:
                # (1, T) 为上游形状 bug 的退化产物, 展平而非沿时间维平均
                wav = wav[0]
            elif wav.shape[-1] == 1:
                wav = wav[:, 0]
            else:
                wav = wav.mean(axis=-1)
        # 短于 ~60ms 的片段提不出有效 fbank 帧, 直接返回零向量,
        # 避免 ECAPA 首层 conv (反射填充) 对超短输入崩溃
        if wav.size < int(sr * 0.06):
            return np.zeros(192, dtype=np.float32)
        if self._backend == "anime":
            return self._embed_anime(wav, sr)
        return self._embed_speechbrain(wav)

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
