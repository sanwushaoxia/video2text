from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from video2text.ffmpeg_util import _run_subprocess_text, extract_stereo_audio
from video2text.media import (
    _concat_wav_files,
    _create_silence_wav,
    _extract_audio_clip,
    _fit_audio_to_max_duration,
    _normalize_audio_peak,
    _normalize_audio_to_wav,
    _trim_audio_silence,
    _trim_audio_to_duration,
    get_media_duration,
)
from video2text.mix import _mix_bgm_and_dub
from video2text.separate import VocalAssets, _separate_vocal_stems
from video2text.srt import _is_meaningful_subtitle


_DUB_LOCALE_MAP = {
    "zh": "zh-CN",
    "en": "en-US",
    "ja": "ja-JP",
    "ko": "ko-KR",
    "fr": "fr-FR",
    "de": "de-DE",
    "es": "es-ES",
    "ru": "ru-RU",
    "pt": "pt-BR",
    "it": "it-IT",
    "ar": "ar-SA",
    "hi": "hi-IN",
    "th": "th-TH",
    "vi": "vi-VN",
}

_DUB_DEFAULT_VOICES = {
    "zh": "zh-CN-XiaoxiaoNeural",
    "en": "en-US-JennyNeural",
    "ja": "ja-JP-NanamiNeural",
    "ko": "ko-KR-SunHiNeural",
    "fr": "fr-FR-DeniseNeural",
    "de": "de-DE-KatjaNeural",
    "es": "es-ES-ElviraNeural",
    "ru": "ru-RU-SvetlanaNeural",
    "pt": "pt-BR-FranciscaNeural",
    "it": "it-IT-ElsaNeural",
}

_DUB_GENDER_DEFAULT_VOICES: dict[str, dict[str, str]] = {
    "zh": {
        "female": "zh-CN-XiaoxiaoNeural",
        "male": "zh-CN-YunxiNeural",
    },
    "en": {
        "female": "en-US-JennyNeural",
        "male": "en-US-GuyNeural",
    },
    "ja": {
        "female": "ja-JP-NanamiNeural",
        "male": "ja-JP-KeitaNeural",
    },
    "ko": {
        "female": "ko-KR-SunHiNeural",
        "male": "ko-KR-InJoonNeural",
    },
    "fr": {
        "female": "fr-FR-DeniseNeural",
        "male": "fr-FR-HenriNeural",
    },
    "de": {
        "female": "de-DE-KatjaNeural",
        "male": "de-DE-ConradNeural",
    },
    "es": {
        "female": "es-ES-ElviraNeural",
        "male": "es-ES-AlvaroNeural",
    },
    "ru": {
        "female": "ru-RU-SvetlanaNeural",
        "male": "ru-RU-DmitryNeural",
    },
    "pt": {
        "female": "pt-BR-FranciscaNeural",
        "male": "pt-BR-AntonioNeural",
    },
    "it": {
        "female": "it-IT-ElsaNeural",
        "male": "it-IT-DiegoNeural",
    },
}

_SPEAKER_GENDER_ALIASES = {
    "male": "male",
    "m": "male",
    "man": "male",
    "男": "male",
    "male_voice": "male",
    "female": "female",
    "f": "female",
    "woman": "female",
    "女": "female",
    "female_voice": "female",
}

async def _resolve_dub_voice(dub_lang: str, voice_override: str | None) -> str:
    if voice_override:
        return voice_override
    short = dub_lang.split("-")[0].lower()
    if short in _DUB_DEFAULT_VOICES:
        return _DUB_DEFAULT_VOICES[short]
    import edge_tts

    locale_prefix = _DUB_LOCALE_MAP.get(short, dub_lang)
    voices = await edge_tts.list_voices()
    for v in voices:
        name = v.get("ShortName") or ""
        locale = v.get("Locale") or ""
        if locale.startswith(locale_prefix) and "Neural" in name:
            return name
    raise RuntimeError(
        f"未找到语言 {dub_lang!r} 的 edge-tts 音色, 请用 --dub-voice 手动指定"
    )

def load_dub_speaker_map(path: Path) -> dict[str, str]:
    """加载说话人映射 JSON: SRT 序号 -> male/female 或 edge-tts 音色名。"""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"说话人映射 JSON 解析失败 ({path}): {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"说话人映射必须是 JSON 对象: {path}")
    mapping: dict[str, str] = {}
    for key, value in raw.items():
        if value is None:
            continue
        label = str(value).strip()
        if label:
            mapping[str(key).strip()] = label
    if not mapping:
        raise ValueError(f"说话人映射为空: {path}")
    return mapping

def _looks_like_edge_voice(name: str) -> bool:
    return "Neural" in name or re.match(r"^[a-z]{2}-[A-Z]{2}-", name) is not None

def _resolve_segment_edge_voice(
    cue_key: str,
    *,
    speaker_map: dict[str, str] | None,
    voice_pool: dict[str, str],
) -> tuple[str, str | None]:
    """根据说话人映射返回 (edge_voice, 性别标签用于日志)。"""
    if not speaker_map:
        return voice_pool["default"], None
    entry = speaker_map.get(cue_key, "").strip()
    if not entry:
        return voice_pool["default"], None
    gender_key = _SPEAKER_GENDER_ALIASES.get(entry.lower())
    if gender_key:
        return voice_pool[gender_key], gender_key
    if _looks_like_edge_voice(entry):
        return entry, None
    return voice_pool["default"], None

async def _tts_to_file(text: str, voice: str, out_path: Path) -> None:
    import edge_tts

    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(out_path))

class DubEngineConfig:
    engine: str = "edge"
    voice: str | None = None
    voice_ref: Path | None = None
    voice_ref_text: str | None = None
    voice_ref_lang: str = "ja"
    dub_lang: str = "zh"
    rvc_model: Path | None = None
    rvc_index: Path | None = None
    rvc_api: str | None = None
    gpt_sovits_api: str = "http://127.0.0.1:9880"
    sovits_ref_clip: Path | None = None
    sovits_speed: float = 0.95
    sovits_text_split: str = "cut0"
    speaker_map: dict[str, str] | None = None
    voice_female: str | None = None
    voice_male: str | None = None

_SOVITS_API_VERSION: dict[str, str] = {}

_SOVITS_LANG_MAP = {
    "zh": "zh",
    "en": "en",
    "ja": "ja",
    "ko": "ko",
    "yue": "yue",
    "auto": "auto",
}

def _sovits_lang(code: str) -> str:
    short = (code or "auto").split("-")[0].lower()
    return _SOVITS_LANG_MAP.get(short, short)

def _detect_gpt_sovits_api_version(api_base: str) -> str:
    base = api_base.rstrip("/")
    cached = _SOVITS_API_VERSION.get(base)
    if cached:
        return cached
    version = "v1"
    try:
        with urllib.request.urlopen(f"{base}/openapi.json", timeout=5) as resp:
            spec = json.loads(resp.read().decode("utf-8"))
        if "/tts" in spec.get("paths", {}):
            version = "v2"
    except Exception:
        pass
    _SOVITS_API_VERSION[base] = version
    return version

def _prepare_sovits_ref_audio(ref_path: Path, work_dir: Path) -> tuple[Path, float]:
    """GPT-SoVITS 要求参考音频 3~10 秒, 过长则自动裁剪。"""
    duration = get_media_duration(ref_path)
    if 3.0 <= duration <= 10.0:
        return ref_path, duration

    clip_len = min(8.0, duration)
    if duration > 10.0:
        start = max(0.0, (duration - clip_len) / 2.0)
    else:
        start = 0.0
        clip_len = duration

    clipped = work_dir / f"{ref_path.stem}_sovits_clip.wav"
    _extract_audio_clip(ref_path, clipped, start, start + clip_len)
    clip_dur = get_media_duration(clipped)
    if clip_dur < 3.0:
        raise RuntimeError(
            f"GPT-SoVITS 参考音频过短 ({clip_dur:.1f}s), 需要 3~10 秒; "
            f"请用 --voice-ref 指定更长的参考片段"
        )
    return clipped, clip_dur

def _rvc_convert_http(
    api_base: str,
    src_wav: Path,
    out_wav: Path,
    model_path: Path,
    index_path: Path | None,
) -> None:
    """调用 RVC WebUI / 兼容服务的 /convert 接口。"""
    import json as json_mod

    boundary = "----video2text-rvc"
    body = bytearray()
    for name, value, filename, content_type in (
        ("model_path", str(model_path.resolve()), None, None),
        ("f0_up_key", "0", None, None),
        ("index_rate", "0.75", None, None),
        ("filter_radius", "3", None, None),
        ("rms_mix_rate", "0.25", None, None),
        ("protect", "0.33", None, None),
    ):
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        body.extend(f"{value}\r\n".encode())

    if index_path and index_path.is_file():
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(
            f'Content-Disposition: form-data; name="index_path"\r\n\r\n'.encode()
        )
        body.extend(f"{str(index_path.resolve())}\r\n".encode())

    audio_bytes = src_wav.read_bytes()
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(
        f'Content-Disposition: form-data; name="audio"; filename="{src_wav.name}"\r\n'.encode()
    )
    body.extend(b"Content-Type: audio/wav\r\n\r\n")
    body.extend(audio_bytes)
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())

    url = api_base.rstrip("/") + "/convert"
    req = urllib.request.Request(
        url,
        data=bytes(body),
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            ctype = resp.headers.get("Content-Type", "")
            payload = resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"RVC API HTTP {exc.code}: {detail[-500:]}") from exc

    if "application/json" in ctype:
        info = json_mod.loads(payload.decode("utf-8", errors="replace"))
        raise RuntimeError(f"RVC API 返回 JSON 而非音频: {info}")
    out_wav.write_bytes(payload)

def _rvc_convert_inprocess(
    src_wav: Path,
    out_wav: Path,
    model_path: Path,
    index_path: Path | None,
) -> None:
    from rvc_python.inference import RVCInference

    import torch

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    rvc = RVCInference(device=device)
    rvc.load_model(str(model_path), index_path=str(index_path) if index_path else "")
    rvc.infer_file(str(src_wav), str(out_wav))

def _rvc_convert(
    src_wav: Path,
    out_wav: Path,
    config: DubEngineConfig,
) -> None:
    if config.rvc_api:
        if not config.rvc_model:
            raise RuntimeError("RVC API 模式需要 --rvc-model")
        _rvc_convert_http(
            config.rvc_api, src_wav, out_wav, config.rvc_model, config.rvc_index
        )
        return
    if not config.rvc_model:
        raise RuntimeError("RVC 模式需要 --rvc-model 或 --rvc-api")
    try:
        _rvc_convert_inprocess(src_wav, out_wav, config.rvc_model, config.rvc_index)
    except ImportError as exc:
        raise RuntimeError(
            "未安装 rvc-python, 请 pip install -r requirements-rvc.txt "
            "或配置 --rvc-api 指向本地 RVC 服务"
        ) from exc

def _http_download(url: str, out_path: Path, *, timeout: float = 180.0) -> None:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
        content_type = (resp.headers.get("Content-Type") or "").lower()
    if len(data) < 256 and "json" in content_type:
        try:
            payload = json.loads(data.decode("utf-8"))
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and payload.get("message"):
            raise RuntimeError(
                f"API 错误: {payload.get('message')}: {payload.get('Exception', '')}"
            )
    if len(data) < 256:
        raise RuntimeError(f"API 返回过短: {data[:200]!r}")
    out_path.write_bytes(data)

def _gpt_sovits_save_response(data: bytes, content_type: str, out_path: Path) -> None:
    content_type = (content_type or "").lower()
    if "json" in content_type:
        payload = json.loads(data.decode("utf-8"))
        if isinstance(payload, dict):
            if payload.get("message") and payload.get("message") != "success":
                raise RuntimeError(
                    f"GPT-SoVITS 错误: {payload.get('message')}: {payload.get('Exception', '')}"
                )
            for key in ("audio", "wav", "data", "file"):
                candidate = payload.get(key)
                if isinstance(candidate, str) and Path(candidate).is_file():
                    shutil.copy2(candidate, out_path)
                    return
        raise RuntimeError(f"GPT-SoVITS 返回未知 JSON: {data[:200]!r}")

    raw_path = out_path.with_suffix(".raw.bin")
    raw_path.write_bytes(data)
    _normalize_audio_to_wav(raw_path, out_path, 44100)
    raw_path.unlink(missing_ok=True)

def _gpt_sovits_tts(
    text: str,
    out_path: Path,
    config: DubEngineConfig,
) -> None:
    if not config.voice_ref or not config.voice_ref.is_file():
        raise RuntimeError("GPT-SoVITS 需要参考音频 (--voice-ref 或自动提取)")
    ref_audio = config.sovits_ref_clip or config.voice_ref
    prompt_text = config.voice_ref_text or text[:80]
    api_base = config.gpt_sovits_api.rstrip("/")
    api_version = _detect_gpt_sovits_api_version(api_base)

    if api_version == "v2":
        payload = {
            "text": text,
            "text_lang": _sovits_lang(config.dub_lang),
            "ref_audio_path": str(ref_audio.resolve()),
            "prompt_text": prompt_text,
            "prompt_lang": _sovits_lang(config.voice_ref_lang),
            "speed_factor": config.sovits_speed,
            "text_split_method": config.sovits_text_split,
            "media_type": "wav",
            "streaming_mode": False,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            f"{api_base}/tts",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
    else:
        params = urllib.parse.urlencode(
            {
                "text": text,
                "text_language": config.dub_lang,
                "refer_wav_path": str(ref_audio.resolve()),
                "prompt_text": prompt_text,
                "prompt_language": config.voice_ref_lang,
            }
        )
        req = urllib.request.Request(f"{api_base}/?{params}", method="GET")

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = resp.read()
            content_type = resp.headers.get("Content-Type") or ""
        _gpt_sovits_save_response(data, content_type, out_path)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GPT-SoVITS API HTTP {exc.code}: {detail[-500:]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"无法连接 GPT-SoVITS API ({config.gpt_sovits_api}): {exc}"
        ) from exc

async def synthesize_segment(
    text: str,
    out_path: Path,
    *,
    config: DubEngineConfig,
    edge_voice: str,
    work_dir: Path,
    segment_index: int,
) -> None:
    """按引擎合成单段配音。"""
    if config.engine == "gpt-sovits":
        _gpt_sovits_tts(text, out_path, config)
        return

    if config.engine == "rvc":
        edge_mp3 = work_dir / f"edge_{segment_index:04d}.mp3"
        await _tts_to_file(text, edge_voice, edge_mp3)
        edge_wav = work_dir / f"edge_{segment_index:04d}.wav"
        _normalize_audio_to_wav(edge_mp3, edge_wav, 44100)
        try:
            _rvc_convert(edge_wav, out_path, config)
        except Exception as exc:
            print(f"  RVC 转换失败, 回退 edge-tts: {exc}", file=sys.stderr)
            shutil.copy2(edge_wav, out_path)
        return

    if out_path.suffix.lower() == ".wav":
        tmp_mp3 = out_path.with_suffix(".mp3")
        await _tts_to_file(text, edge_voice, tmp_mp3)
        _normalize_audio_to_wav(tmp_mp3, out_path, 44100)
    else:
        await _tts_to_file(text, edge_voice, out_path)

def build_dub_engine_config(args, dub_lang: str, vocal_assets: VocalAssets | None) -> DubEngineConfig:
    voice_ref = None
    voice_ref_text = args.voice_ref_text
    if args.voice_ref:
        voice_ref = Path(args.voice_ref).expanduser().resolve()
    elif vocal_assets and vocal_assets.voice_ref:
        voice_ref = vocal_assets.voice_ref
        if not voice_ref_text:
            voice_ref_text = vocal_assets.voice_ref_text

    ref_lang = args.voice_ref_lang or "ja"
    speaker_map = None
    if getattr(args, "dub_speaker_map", None):
        speaker_map = load_dub_speaker_map(
            Path(args.dub_speaker_map).expanduser().resolve()
        )
    return DubEngineConfig(
        engine=args.dub_engine,
        voice=args.dub_voice,
        voice_ref=voice_ref,
        voice_ref_text=voice_ref_text,
        voice_ref_lang=ref_lang,
        dub_lang=dub_lang,
        rvc_model=Path(args.rvc_model).expanduser().resolve() if args.rvc_model else None,
        rvc_index=Path(args.rvc_index).expanduser().resolve() if args.rvc_index else None,
        rvc_api=args.rvc_api,
        gpt_sovits_api=args.gpt_sovits_api,
        sovits_speed=args.sovits_speed,
        sovits_text_split=args.sovits_text_split,
        speaker_map=speaker_map,
        voice_female=args.dub_voice_female,
        voice_male=args.dub_voice_male,
    )

def resolve_dub_language(task: str, used_lang: str, dub_lang: str | None) -> str:
    """确定 AI 配音使用的语言码。"""
    if dub_lang:
        return dub_lang.split("-")[0].lower()
    if task == "translate":
        return "en"
    return (used_lang or "en").split("-")[0].lower()

def generate_dub_audio(
    segments,
    out_wav: Path,
    dub_lang: str,
    total_duration: float,
    work_dir: Path,
    engine_config: DubEngineConfig,
) -> str:
    """按字幕时间轴生成 AI 配音 WAV, 返回实际使用的 edge-tts 音色名 (或引擎描述)。"""
    async def _run() -> str:
        voice_pool: dict[str, str] = {}
        if engine_config.engine in ("edge", "rvc"):
            default_voice = await _resolve_dub_voice(dub_lang, engine_config.voice)
            voice_pool["default"] = default_voice
            lang_short = dub_lang.split("-")[0].lower()
            gender_defaults = _DUB_GENDER_DEFAULT_VOICES.get(lang_short, {})
            voice_pool["female"] = (
                engine_config.voice_female
                or gender_defaults.get("female")
                or default_voice
            )
            voice_pool["male"] = (
                engine_config.voice_male
                or gender_defaults.get("male")
                or default_voice
            )
            if engine_config.engine == "rvc":
                print(
                    f"AI 配音: edge-tts + RVC 音色转换 (语言={dub_lang}, 基础音色={default_voice})"
                )
            elif engine_config.speaker_map:
                print(
                    f"AI 配音语言: {dub_lang}, 多说话人 edge-tts "
                    f"(女={voice_pool['female']}, 男={voice_pool['male']}, "
                    f"映射 {len(engine_config.speaker_map)} 条)"
                )
            else:
                print(f"AI 配音语言: {dub_lang}, 音色: {default_voice}")
            resolved_voice = (
                f"multi(f={voice_pool['female']}, m={voice_pool['male']})"
                if engine_config.speaker_map
                else default_voice
            )
        else:
            resolved_voice = engine_config.engine
            api_ver = _detect_gpt_sovits_api_version(engine_config.gpt_sovits_api)
            print(
                f"AI 配音: GPT-SoVITS v{api_ver[-1]} (语言={dub_lang}, "
                f"API={engine_config.gpt_sovits_api})"
            )
            if engine_config.speaker_map:
                print(
                    "  提示: --dub-speaker-map 仅对 edge/rvc 引擎生效, 已忽略",
                    file=sys.stderr,
                )
            if engine_config.voice_ref:
                clip, clip_dur = _prepare_sovits_ref_audio(
                    engine_config.voice_ref, work_dir
                )
                engine_config.sovits_ref_clip = clip
                print(
                    f"  参考音频: {engine_config.voice_ref.name} "
                    f"-> {clip.name} ({clip_dur:.1f}s, GPT-SoVITS 要求 3~10s)"
                )
                if engine_config.voice_ref_text:
                    print(f"  参考文本: {engine_config.voice_ref_text[:60]}...")
                    print(
                        "  提示: 参考文本必须与参考音频内容一致 (错字会导致发音含糊); "
                        "中文合成需 GPT-SoVITS 安装 G2PWModel"
                    )

        parts: list[Path] = []
        cursor = 0.0
        sample_rate = 44100
        cue_counter = 0

        for i, seg in enumerate(segments):
            text = (seg.get("text") or "").strip()
            if not text or not _is_meaningful_subtitle(text):
                continue
            cue_counter += 1
            cue_key = str(seg.get("index", cue_counter))
            start = float(seg.get("start", 0.0))
            if start > cursor + 0.05:
                silence = work_dir / f"silence_{i:04d}.wav"
                _create_silence_wav(start - cursor, silence, sample_rate)
                parts.append(silence)
                cursor = start

            segment_voice = ""
            gender_tag: str | None = None
            if engine_config.engine in ("edge", "rvc"):
                segment_voice, gender_tag = _resolve_segment_edge_voice(
                    cue_key,
                    speaker_map=engine_config.speaker_map,
                    voice_pool=voice_pool,
                )

            wav_tts = work_dir / f"tts_{i:04d}.wav"
            tag = {"male": "男", "female": "女"}.get(gender_tag or "", "")
            tag_note = f" ({tag})" if tag else ""
            print(
                f"  合成配音 [{cue_key}]{tag_note}: "
                f"{text[:40]}{'...' if len(text) > 40 else ''}"
            )
            try:
                await synthesize_segment(
                    text,
                    wav_tts,
                    config=engine_config,
                    edge_voice=segment_voice if engine_config.engine != "gpt-sovits" else "",
                    work_dir=work_dir,
                    segment_index=i,
                )
            except Exception as exc:
                print(f"  跳过无效配音段 [{i + 1}]: {exc}", file=sys.stderr)
                continue

            try:
                trimmed = work_dir / f"tts_{i:04d}_trim.wav"
                boosted = work_dir / f"tts_{i:04d}_boost.wav"
                _normalize_audio_peak(wav_tts, boosted, sample_rate=sample_rate)
                try:
                    get_media_duration(boosted)
                except RuntimeError as exc:
                    raise RuntimeError(f"GPT-SoVITS 返回无效音频: {exc}") from exc
                _trim_audio_silence(boosted, trimmed, sample_rate=sample_rate)
                end = float(seg.get("end", start))
                slot = max(end - start, 0.25)
                fitted = work_dir / f"tts_{i:04d}_fit.wav"
                _fit_audio_to_max_duration(
                    trimmed, fitted, slot, sample_rate=sample_rate
                )
                seg_dur = get_media_duration(fitted)
                if seg_dur <= 0.01:
                    raise RuntimeError("配音段时长为 0")
                parts.append(fitted)
                cursor += seg_dur
            except Exception as exc:
                print(f"  跳过无效配音段 [{i + 1}]: {exc}", file=sys.stderr)
                continue

        if cursor < total_duration - 0.05:
            tail = work_dir / "silence_tail.wav"
            _create_silence_wav(total_duration - cursor, tail, sample_rate)
            parts.append(tail)
            cursor = total_duration

        if not parts:
            _create_silence_wav(total_duration, out_wav, sample_rate)
        else:
            _concat_wav_files(parts, out_wav)
            out_dur = get_media_duration(out_wav)
            if out_dur > total_duration + 0.05:
                clipped = work_dir / "dub_clipped.wav"
                _trim_audio_to_duration(
                    out_wav, clipped, total_duration, sample_rate=sample_rate
                )
                shutil.move(clipped, out_wav)

        return resolved_voice if engine_config.engine != "gpt-sovits" else "gpt-sovits"

    return asyncio.run(_run())

def finalize_dub_audio(
    dub_wav_path: Path,
    out_dir: Path,
    video_stem: str,
    *,
    keep_bgm: bool,
    bgm_volume: float,
    dub_volume: float,
    vocal_assets: VocalAssets | None,
    video_path: Path | None = None,
    work_dir: Path | None = None,
) -> Path:
    """生成最终配音音轨; keep_bgm=True 时保留原视频背景音乐。"""
    if not keep_bgm:
        return dub_wav_path

    mixed_path = out_dir / f"{video_stem}_mixed.wav"
    instrumental = vocal_assets.instrumental if vocal_assets else None

    if instrumental is None:
        if video_path is None or work_dir is None:
            raise RuntimeError("保留 BGM 需要 video_path 或预分离的 vocal_assets")
        source_audio = work_dir / "source_audio.wav"
        print("提取原视频音轨...")
        extract_stereo_audio(video_path, source_audio)
        instrumental, _ = _separate_vocal_stems(source_audio, work_dir)

    print(f"混合 AI 配音与背景音乐 -> {mixed_path.name}")
    _mix_bgm_and_dub(instrumental, dub_wav_path, mixed_path, bgm_volume, dub_volume)
    print(f"已写入: {mixed_path}")
    return mixed_path
