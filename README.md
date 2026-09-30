# video2text（Whisper 视频语音识别与字幕）

基于 [OpenAI Whisper](https://github.com/openai/whisper) 的视频转文字工具：自动识别视频语言、将音频转成文字，并把字幕烧录到画面下方。可选 **AI 配音**（edge-tts / RVC / GPT-SoVITS），默认 **保留背景音乐**（BS-RoFormer ensemble 人声分离）。

**同时支持 Linux 与 Windows**（Python + Whisper + edge-tts + ffmpeg）。

---

## 项目结构

源码已拆分为单一职责模块，位于 `src/`：

```
src/
  video2text/          # 核心库
    ffmpeg_util.py     # ffmpeg / ffprobe 封装
    srt.py             # SRT 解析与生成
    translate.py       # 字幕翻译
    ocr.py             # 画面烧录字幕 OCR
    whisper_transcribe.py  # Whisper 转写
    separate.py        # 人声/背景分离 (BS-RoFormer)
    bs_roformer_split.py  # BS-RoFormer / Mel-Band (audio-separator)
    f0_analysis.py     # 基频 (F0) 分析
    gender_split.py    # 自动男/女声切分
    diarization.py     # pyannote Community-1 日记化
    speaker_embedding.py  # 动漫 ECAPA embedding
    speaker_stems.py   # N 说话人轨切分
    overlap_separation.py  # 重叠段 MossFormer2/SepFormer 盲分离
    dub.py             # AI 配音 (edge / RVC / GPT-SoVITS)
    mix.py             # 配音与 BGM 混音
    render.py          # 字幕烧录 / 音轨替换
    speaker_map.py     # 多角色说话人推断
    pipeline.py        # 完整流水线 CLI
  cli/                 # 单一功能命令行入口
    extract_subs.py    # 提取字幕 (Whisper / OCR)
    burn_subs.py       # 烧录字幕到视频
    separate_audio.py  # 人声/伴奏分离 (默认 BS-RoFormer ensemble)
    audio_to_subs.py   # 音频 → 字幕 (Whisper)
    subs_to_audio.py   # 字幕 → AI 配音
    mix_audio.py       # 混合配音与伴奏
    speaker_map.py     # 生成 speaker_map.json
    transcribe.py      # 完整流水线 (同 transcribe_video.py)

requirements.txt              # 统一依赖 (PyPI PyTorch / CUDA 13 + Whisper + 分离 + 日记化 + OCR 等)
scripts/
  setup_bs_roformer_env.sh    # 创建独立 conda 环境并安装全量依赖 (可选)
  compare_roformer_ab.py      # no_vocals 客观 A/B (RMS / nv/v)
  compare_speaker_sep.py      # 多角色轨能量 / 重叠分离 A/B
```

根目录保留兼容入口：`transcribe_video.py`、`generate_speaker_map.py`（薄包装，转发到 `src/video2text`）。

---

## 命令速查（每个特性的完整可执行命令）

以下命令均可直接复制执行（`video.mp4` 换成你的视频）。前置：`pip install -r requirements.txt`；Linux 需系统 `ffmpeg`（Windows 用 imageio-ffmpeg 自带）。

### 0. 完整流水线（转写 → 字幕 → 配音 → 烧录，一键）

```bash
# 最简: 自动检测语言 + 转写 + 烧录字幕
python transcribe_video.py video.mp4

# 指定语言/模型/输出目录与字幕样式
python transcribe_video.py video.mp4 --model small --language ja --out-dir ./out --font-size 28 --margin-v 48

# 只出 SRT/文本, 不烧录
python transcribe_video.py video.mp4 --no-burn

# 已有 SRT 只重烧录 / 恢复不透明底框 / 强制 CPU
python transcribe_video.py video.mp4 --burn-only --srt ./out/video.srt
python transcribe_video.py video.mp4 --burn-only --box
python transcribe_video.py video.mp4 --device cpu

# 画面 OCR 硬字幕 + 中文 AI 配音 + 保留 BGM
python transcribe_video.py video.mp4 --subs-source ocr --ocr-lang zh --dub --dub-lang zh

# 完整: 翻译 + 配音 (GPT-SoVITS 需本地 API http://127.0.0.1:9880)
python transcribe_video.py video.mp4 --subs-source ocr --ocr-lang zh --dub --dub-lang zh \
  --dub-engine gpt-sovits --voice-ref-lang ja

# 等价入口: python src/cli/transcribe.py (参数完全相同)
```

### 1. 音频/视频 → 字幕（Whisper 或 OCR）

```bash
# Whisper 转写 → <stem>.srt/.txt/.json (输出到 ./video_whisper/)
python src/cli/audio_to_subs.py video.mp4
python src/cli/audio_to_subs.py video.mp4 --model small --language zh --out-dir ./out

# OCR 画面硬字幕
python src/cli/extract_subs.py video.mp4 --source ocr --ocr-lang zh --ocr-fps 2
# OCR + GPU 加速 (extract_subs 默认 GPU 关; 流水线里默认开)
python src/cli/extract_subs.py video.mp4 --source ocr --ocr-lang zh --ocr-use-gpu
# Whisper 模式 + 翻译为中文
python src/cli/extract_subs.py video.mp4 --source whisper --translate-to zh
```

### 2. 人声/背景分离（BS-RoFormer，二轨）

```bash
# 默认: ensemble 专保伴奏 → video_vocals.wav + video_no_vocals.wav
python src/cli/separate_audio.py video.mp4 --out-dir ./out

# 换 ensemble preset / 单模型 / dual 双模型
python src/cli/separate_audio.py video.mp4 --roformer-ensemble-preset instrumental_clean
python src/cli/separate_audio.py video.mp4 --roformer-mode single \
  --roformer-model model_bs_roformer_ep_317_sdr_12.9755.ckpt
python src/cli/separate_audio.py video.mp4 --roformer-mode dual \
  --roformer-vocals-model bs_roformer_vocals_revive_v3e_unwa.ckpt \
  --roformer-inst-model mel_band_roformer_instrumental_fv7z_gabox.ckpt

# 指定模型目录与 overlap
python src/cli/separate_audio.py video.mp4 --roformer-model-dir ./models/roformer --roformer-overlap 16
```

### 3. 三轨分离（背景 / 男声 / 女声）

```bash
# 有 SRT (推荐, 精度最高): --srt 必给 (或自动查找 <stem>_whisper/<stem>.srt)
python src/cli/separate_audio.py video.mp4 --out-dir ./out_3stem \
  --three-stems --split-mode srt_f0 --srt ./out/video.srt

# 无 SRT: Whisper 自动时间轴
python src/cli/separate_audio.py video.mp4 --out-dir ./out_3stem \
  --three-stems --split-mode whisper_f0 --language ja

# pyannote 日记化时间轴 (需 HF_TOKEN)
python src/cli/separate_audio.py video.mp4 --out-dir ./out_3stem \
  --three-stems --split-mode diarize --max-speakers 2

# 调参: F0 阈值 / 置信度
python src/cli/separate_audio.py video.mp4 --three-stems --split-mode srt_f0 \
  --srt ./out/video.srt --f0-threshold 200 --min-voiced-ratio 0.25 --no-adaptive-f0
```

### 4. 多轨分离（背景 / N 说话人，pyannote 日记化）

```bash
# 前置: HuggingFace 接受 pyannote 协议 + HF_TOKEN (或 --hf-token / 项目根 hf_token 文件)
export HF_TOKEN=hf_xxxx

# 先预检说话人数量 → *_speaker_count.json
python src/cli/separate_audio.py video.mp4 --out-dir ./out_count \
  --count-speakers --max-speakers 6

# 按预检数量精确分离 (diarize=纯切分)
python src/cli/separate_audio.py video.mp4 --out-dir ./out_multi \
  --multi-stems --split-mode diarize --min-speakers 2 --max-speakers 2

# 含重叠对白: 日记化 + 局部盲分离 (MossFormer2→SepFormer 自动回退)
python src/cli/separate_audio.py video.mp4 --out-dir ./out_bss \
  --multi-stems --split-mode diarize_bss --max-speakers 6 --bss-backend auto

# 常用调参: 重叠窗口 / embedding 改标
python src/cli/separate_audio.py video.mp4 --multi-stems --split-mode diarize_bss \
  --overlap-min-sec 0.15 --max-overlap-sec 8 \
  --merge-turn-gap 0.4 --min-similarity 0.35 --no-refine-embedding
```

### 5. 说话人映射（speaker_map.json）

```bash
# 前置: 目录内需同时有 *.srt 与 *_vocals.wav (先跑 1 和 2)
python src/cli/speaker_map.py --whisper-dir ./out/video_whisper

# 或显式指定
python src/cli/speaker_map.py --srt ./out/video.srt --vocals ./out/video_vocals.wav -o ./out/speaker_map.json

# 等价入口: python generate_speaker_map.py --whisper-dir ./out/video_whisper
```

### 6. 字幕 → AI 配音（edge / RVC / GPT-SoVITS）

```bash
# edge-tts (默认引擎, 需联网)
# 注意: subs_to_audio 的选项必须写在视频路径之前 (argparse REMAINDER 限制)
python src/cli/subs_to_audio.py --dub-lang zh --srt ./out/video.srt video.mp4 --dub-engine edge

# 多角色男女声
python src/cli/subs_to_audio.py --dub-lang zh --srt ./out/video.srt video.mp4 \
  --dub-engine edge --dub-speaker-map ./out/speaker_map.json \
  --dub-voice-female zh-CN-XiaoxiaoNeural --dub-voice-male zh-CN-YunxiNeural

# GPT-SoVITS 音色克隆 (需本地启动 API, 默认 http://127.0.0.1:9880)
python src/cli/subs_to_audio.py --dub-lang zh --srt ./out/video.srt video.mp4 \
  --dub-engine gpt-sovits --voice-ref-lang ja

# RVC (需角色 .pth: --rvc-model 必填; rvc-python 需单独安装或用 --rvc-api 服务)
python src/cli/subs_to_audio.py --dub-lang zh --srt ./out/video.srt video.mp4 \
  --dub-engine rvc --rvc-model /path/to/character.pth
```

### 7. 混音 / 烧录（单独执行）

```bash
# 配音 + BGM 混音 (三个参数均必填)
python src/cli/mix_audio.py --dub ./out/video_dub.wav --bgm ./out/video_no_vocals.wav -o ./out/video_mixed.wav
python src/cli/mix_audio.py --dub ./out/video_dub.wav --bgm ./out/video_no_vocals.wav \
  -o ./out/video_mixed.wav --bgm-volume 0.8 --dub-volume 1.2

# 烧录字幕 (--srt 必填)
python src/cli/burn_subs.py video.mp4 --srt ./out/video.srt -o ./out/video_subtitled.mp4
python src/cli/burn_subs.py video.mp4 --srt ./out/video.srt --font-size 28 --margin-v 48 --box
```

### 8. 辅助脚本

```bash
python scripts/prefetch_roformer_models.py      # 预下载 BS-RoFormer ckpt
python scripts/prefetch_pyannote_models.py      # 预下载 pyannote 模型 (需 HF_TOKEN)
python scripts/verify_diarize_setup.py          # 日记化部署自检
python scripts/migrate_models_to_project.py --dry-run   # 迁移旧缓存到 models/
bash scripts/run_multi_stems_diarize.sh video.mp4 ./out_multi diarize   # 一键多轨 (PY=python3 可换解释器)
python scripts/compare_roformer_ab.py ./out_a ./out_b --json-out ab_compare.json  # 分离 A/B
python scripts/compare_speaker_sep.py ./out_multi ./out_bss --stem 桔梗犬夜叉片段
python tools/check_imports.py --smoke           # 开发自检 import 完整性
```

---

**开发：** 修改 `src/` 后运行 `python tools/check_imports.py --smoke` 检查 import 与模块变量是否完整。

**分步示例：**

```bash
# 1. Whisper 提取字幕
python src/cli/audio_to_subs.py video.mp4 --language ja --out-dir ./out

# 2. 人声分离 (二轨, 默认 ensemble 保伴奏)
python src/cli/separate_audio.py video.mp4 --out-dir ./out
# 输出: video_no_vocals.wav (背景) + video_vocals.wav (人声)

# 2b. 三轨 (背景 + 男声 + 女声)
python src/cli/separate_audio.py video.mp4 --out-dir ./out --three-stems --split-mode whisper_f0 --language ja
# 纯音频日记化 (需 HF_TOKEN; 依赖已含于 requirements.txt)
python src/cli/separate_audio.py video.mp4 --out-dir ./out --three-stems --split-mode diarize --max-speakers 4

# 3. 生成多角色映射
python src/cli/speaker_map.py --whisper-dir ./out

# 4. 字幕转配音
# 注意: subs_to_audio 的选项必须写在位置参数 (视频路径) 之前 (argparse REMAINDER 限制)
python src/cli/subs_to_audio.py --dub-lang zh --srt ./out/video.srt \
  video.mp4 --dub-engine edge --dub-speaker-map ./out/speaker_map.json

# 5. 混合 BGM（若上一步未自动混音）
python src/cli/mix_audio.py --dub ./out/video_dub.wav --bgm ./out/video_no_vocals.wav -o ./out/video_mixed.wav

# 6. 烧录字幕
python src/cli/burn_subs.py video.mp4 --srt ./out/video.srt -o ./out/video_subtitled.mp4

# 一键完整流程（与原先相同）
python transcribe_video.py video.mp4 --dub --dub-lang zh
```

---

## 人声/背景分离 (`separate_audio.py`)

### 默认行为（无需额外参数）

```bash
conda activate py312   # 或任何已安装 requirements.txt 的环境
python src/cli/separate_audio.py 桔梗犬夜叉片段.mp4 --out-dir ./out
```

| 默认项 | 值 | 说明 |
|--------|-----|------|
| `--roformer-mode` | `ensemble` | 多模型融合推理 |
| `--roformer-ensemble-preset` | `instrumental_full` | 专保伴奏 (v1e+ + becruily inst) |

输出：

| 文件 | 说明 |
|------|------|
| `*_no_vocals.wav` | 背景/伴奏 (立体声) |
| `*_vocals.wav` | 人声 |

### RoFormer 分离模式 (`--roformer-mode`)

| 模式 | 说明 | 典型场景 |
|------|------|---------|
| `ensemble` (**默认**) | audio-separator 内置 preset，多 ckpt 融合 | **保伴奏 + 去人声**，实践效果最佳 |
| `single` | 单个 ckpt | A/B 对比、指定模型 |
| `dual` | vocals 与 instrumental 各用专精 ckpt 分别推理 | 精细调参；注意 inst 模型名勿误匹配 stem |

**常用 ensemble preset：**

| preset | 说明 |
|--------|------|
| `instrumental_full` (**默认**) | 最大保伴奏 (v1e+ + becruily) |
| `instrumental_clean` | 最少人声泄漏 (fv7z bleedless + resurrection) |
| `instrumental_balanced` | 噪声与饱满度折中 |
| `vocal_full` | 气声/情感句 capture 更强 |

```bash
# 换 preset
python src/cli/separate_audio.py video.mp4 --out-dir ./out \
  --roformer-ensemble-preset instrumental_clean

# single 单模型
python src/cli/separate_audio.py video.mp4 --out-dir ./out \
  --roformer-mode single --roformer-model model_bs_roformer_ep_317_sdr_12.9755.ckpt

# dual 双模型
python src/cli/separate_audio.py video.mp4 --out-dir ./out \
  --roformer-mode dual \
  --roformer-vocals-model bs_roformer_vocals_revive_v3e_unwa.ckpt \
  --roformer-inst-model bs_roformer_instrumental_resurrection_unwa.ckpt

# 提高推理 overlap (更慢, 略增质量)
python src/cli/separate_audio.py video.mp4 --out-dir ./out --roformer-overlap 16
```

**客观 A/B 对比**（#22 纯 BGM / #26「好温暖」窗口 RMS）：

```bash
python scripts/compare_roformer_ab.py ./out_a ./out_b --json-out ab_compare.json
```

### BS-RoFormer 环境

分离依赖已并入主 `requirements.txt`，直接安装即可：

```bash
pip install -r requirements.txt
```

若需独立环境（避免与其他项目冲突），可用脚本一键创建并安装全量依赖：

```bash
bash scripts/setup_bs_roformer_env.sh video2text-roformer
conda activate video2text-roformer
```

### 首次运行 / 网络与离线

首次分离需从 GitHub 下载 `download_checks.json`、ckpt 权重与 **yaml 模型配置**。若出现 `ConnectionError` / `ConnectTimeout`（无法访问 GitHub）：

**1. 配置代理后重试**

```bash
export HTTPS_PROXY=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
python src/cli/separate_audio.py 桔梗犬夜叉片段.mp4 --out-dir ./out
```

**2. 预下载到项目 models/ 目录（推荐）**

模型默认位于项目 [`models/`](models/) 子目录（Git LFS 跟踪大文件），下载一次即可复用：

| 子目录 | 内容 |
|--------|------|
| `models/roformer/` | BS-RoFormer ckpt + yaml + manifest |
| `models/whisper/` | Whisper `.pt` 权重 |
| `models/huggingface/` | pyannote 等 HF 模型 |
| `models/speechbrain/` | ECAPA embedding、SepFormer |

从旧 AppData / 用户缓存 **一次性迁移**：

```bash
python scripts/migrate_models_to_project.py --dry-run   # 预览
python scripts/migrate_models_to_project.py             # 移动至 models/
```

预下载（新环境无缓存时）：

```bash
python scripts/prefetch_roformer_models.py
python scripts/prefetch_pyannote_models.py   # 需 hf_token
```

环境变量覆盖（可选）：

```bash
export VIDEO2TEXT_MODELS_DIR=/path/to/models
export VIDEO2TEXT_ROFORMER_MODEL_DIR=/path/to/roformer
export HF_HOME=/path/to/huggingface
```

**Git LFS**（首次 clone 后）：

```bash
git lfs install
git lfs pull
```

**3. CLI 指定模型目录**

```bash
python src/cli/separate_audio.py video.mp4 --out-dir ./out \
  --roformer-model-dir ./models/roformer
```

**4. 减小首次下载量**（manifest 仍需要；ckpt 仅 1 个）

```bash
python src/cli/separate_audio.py video.mp4 --out-dir ./out \
  --roformer-mode single --roformer-model model_bs_roformer_ep_317_sdr_12.9755.ckpt
```

离线 manifest 兜底：仓库 [`assets/roformer/download_checks.json`](assets/roformer/download_checks.json) 会在联网失败时自动复制到模型目录；**ckpt 仍需 prefetch 或手动放入模型目录**。

默认 `instrumental_full` ensemble 在模型目录中需 **5 个文件**（prefetch 会校验）：

| 文件 | 说明 |
|------|------|
| `download_checks.json` | 模型清单（可 bundled） |
| `melband_roformer_inst_v1e_plus.ckpt` | ensemble 模型 1 (~871 MiB) |
| `config_melbandroformer_inst.yaml` | 模型 1 配置（可 bundled） |
| `mel_band_roformer_instrumental_becruily.ckpt` | ensemble 模型 2 (~871 MiB) |
| `config_mel_band_roformer_instrumental_becruily.yaml` | 模型 2 配置（可 bundled） |

yaml 配置离线兜底：[`assets/roformer/*.yaml`](assets/roformer/) 在 GitHub 不可达时自动复制到模型目录；**ckpt 无法 bundled，仍需 prefetch 或代理下载**。

若仅缺 yaml（ckpt 已下完），可手动放入模型目录后重跑 `separate_audio`，无需重下 ckpt。

---

## 三轨自动分离 (背景 / 男声 / 女声)

加 `--three-stems` 在二轨基础上自动切分男/女声，**无需人工编辑 speaker_map**：

| 输出文件 | 说明 |
|---------|------|
| `*_no_vocals.wav` | 背景音 (默认 BS-RoFormer ensemble) |
| `*_vocals.wav` | 混合人声 |
| `*_male_vocals.wav` | 男声片段叠加 |
| `*_female_vocals.wav` | 女声片段叠加 |
| `*_split_report.json` | 每段 F0、性别、置信度、对齐信息 |

**原理：** 上游 BS-RoFormer 分离背景与人声；男女声按**时间段切分**到不同轨道 (不是 FFT 滤波, 也不能在两人**同时重叠**说话时物理分离)。

**模式选型：**

| `--split-mode` | 时间轴来源 | 适用场景 |
|----------------|-----------|---------|
| `srt_f0` (**推荐**) | 已有 SRT/OCR 字幕 + F0 + 文本规则 | 硬字幕/完整 SRT，覆盖全片对白 |
| `whisper_f0` | Whisper 自动转写句级时间戳 + F0 | 无 SRT、需 ASR 生成时间轴 |
| `diarize` | pyannote 说话人日记化 + F0 | 不想跑 ASR、说话人数未知 |

**三轨推荐命令：**

```bash
# 无 SRT：Whisper 自动时间轴 + F0
python src/cli/separate_audio.py 桔梗犬夜叉片段.mp4 --out-dir ./out_3stem \
  --three-stems --split-mode whisper_f0 --language ja

# 有 SRT：精度更高
python src/cli/separate_audio.py 桔梗犬夜叉片段.mp4 --out-dir ./out_3stem \
  --three-stems --split-mode srt_f0 \
  --srt 桔梗犬夜叉片段_whisper/桔梗犬夜叉片段.srt

# pyannote 日记化 (需 HF_TOKEN)
export HF_TOKEN=hf_xxxx
python src/cli/separate_audio.py video.mp4 --out-dir ./out \
  --three-stems --split-mode diarize --max-speakers 2

# 用已有 speaker_map 覆盖性别后重切
python src/cli/separate_audio.py 桔梗犬夜叉片段.mp4 --out-dir ./out_3stem \
  --three-stems --split-mode srt_f0 \
  --srt 桔梗犬夜叉片段_whisper/桔梗犬夜叉片段.srt \
  --speaker-map 桔梗犬夜叉片段_whisper/speaker_map.json
```

**准确率边界：**

- 童声、旁白、BGM 残留可能导致 F0 误判；查看 `*_split_report.json` 中 `confidence`、`coverage_ratio` 核对。
- 多人同性别会合并到同一轨；pyannote 可能 over/under-segment。
- 重叠语音仍会串音；当前按 SRT 时间轴切分，无法物理分离同时说话。

**调参：** `--f0-threshold 200`、`--min-voiced-ratio 0.25`。

**RoFormer 参数：**

| 参数 | 默认 | 说明 |
|------|------|------|
| `--roformer-mode` | ensemble | `single` / `dual` / `ensemble` |
| `--roformer-ensemble-preset` | instrumental_full | ensemble 专保伴奏 |
| `--roformer-model` | ep_317 ckpt | single 模式 |
| `--roformer-vocals-model` | revive v3e | dual 模式人声 ckpt |
| `--roformer-inst-model` | fv7z bleedless | dual 模式伴奏 ckpt |
| `--roformer-overlap` | 模型 yaml | 提高如 `16` 增质量 |
| `--roformer-model-dir` | 见上文 | 持久模型缓存目录 |
| `VIDEO2TEXT_ROFORMER_MODEL_DIR` | 同上 | 环境变量等价于 `--roformer-model-dir` |
| `--speaker-map` | 无 | 用已有 map 覆盖 gender 后重切 |

**Report 字段** (`*_split_report.json`)：

- `separator`: 上游 mode、preset/模型名
- 每段: `start/end`, `gender`, `f0_hz`, `confidence`
- 汇总: `stats.stem_assigned_sec`, `stats.unassigned_vocal_sec`

---

## 多轨分离 (背景 / N 说话人)

加 `--multi-stems` 在二轨基础上按 **pyannote 日记化** 输出 N 条 speaker 轨（不再局限男/女二元）：

| 输出文件 | 说明 |
|---------|------|
| `*_no_vocals.wav` | 背景音 |
| `*_vocals.wav` | 混合人声 |
| `*_speaker_speaker_00.wav` 等 | 各说话人片段叠加 |
| `*_split_report.json` | 每段 speaker_id、重叠分离标记、覆盖率 |

**原理：** 日记化标出「谁何时说」→ 非重叠段从 vocals **时间切分**；`diarize_bss` 模式下对重叠段 (0.15–8s, `--overlap-min-sec`/`--max-overlap-sec` 可调) 调用 MossFormer2/SepFormer **局部盲分离**，再用 anime ECAPA embedding 归属到已有 speaker。

**依赖：**

```bash
pip install -r requirements.txt   # 已含分离/日记化/盲分离全部依赖
python scripts/prefetch_roformer_models.py

# HuggingFace: Accept community-1 协议后任选其一
export HF_TOKEN=hf_xxxx
# 或写入项目根目录 hf_token (单行, 已 gitignore)
# 国内: export HF_ENDPOINT=https://hf-mirror.com HF_HUB_ENABLE_HF_TRANSFER=0

python scripts/verify_diarize_setup.py      # 检查依赖 + 预拉 pyannote
python scripts/prefetch_pyannote_models.py  # 仅预下载日记化模型
bash scripts/run_multi_stems_diarize.sh 桔梗犬夜叉片段.mp4 ./out_multi diarize
```

**双人对话（犬夜叉+桔梗）推荐：**

```bash
export HF_TOKEN=hf_xxxx
python src/cli/separate_audio.py 桔梗犬夜叉片段.mp4 --out-dir ./out_multi \
  --multi-stems --split-mode diarize --min-speakers 2 --max-speakers 2
```

**推荐命令：**

```bash
# 交替对白为主: 仅日记化 + 切分
python src/cli/separate_audio.py video.mp4 --out-dir ./out_multi \
  --multi-stems --split-mode diarize --max-speakers 4

# 含重叠对白: 日记化 + 局部盲分离
python src/cli/separate_audio.py video.mp4 --out-dir ./out_bss \
  --multi-stems --split-mode diarize_bss --max-speakers 4 \
  --bss-backend auto

# A/B 对比两个输出目录
python scripts/compare_speaker_sep.py ./out_multi ./out_bss --stem 桔梗犬夜叉片段
```

| `--split-mode` | 说明 |
|----------------|------|
| `diarize` | pyannote Community-1 + 时间切分 |
| `diarize_bss` | 上述 + 重叠段 MossFormer2/SepFormer + embedding 归属 |

| 参数 | 默认 | 说明 |
|------|------|------|
| `--count-speakers` | 关 | 仅预检说话人数量, 输出 `*_speaker_count.json` 后退出 |
| `--diarization-model` | community-1 | pyannote pipeline |
| `--embedding-model` | litagin anime ECAPA | 重叠段 speaker 匹配 |
| `--bss-backend` | auto | mossformer2 → sepformer 回退 (2 路); 3 路自动用 sepformer 3mix |
| `--no-embedding` | 关 | 禁用 embedding (diarize_bss / refine) |
| `--refine-embedding` / `--no-refine-embedding` | 开 | 日记化后 embedding 合并碎段、改标、低置信过滤 |
| `--merge-turn-gap` | 0.4 | 合并同 speaker 相邻碎段最大间隔 (秒) |
| `--min-turn-sec` | 0.5 | 参与 speaker profile 的最短 turn (秒) |
| `--min-similarity` | 0.35 | embedding 改标/过滤最低相似度 |
| `--fill-unassigned` / `--no-fill-unassigned` | 开 | 未分配人声区间按 embedding 填充 |
| `--diarize-preprocess` / `--no-diarize-preprocess` | 开 | 日记化前 vocals 峰值归一化 + 80Hz 高通 |
| `--overlap-min-sec` | 0.15 | 重叠检测与 BSS 最小时长 (秒) |
| `--max-overlap-sec` | 8.0 | 重叠段 BSS 盲分离最大时长 (秒) |

**纯音频调优流程（不用 SRT）：**

1. 先预检说话人数量：`--count-speakers --max-speakers 6`，输出 `*_speaker_count.json` 列出每位说话人时长
2. 已知人数后精确分离：`--min-speakers N --max-speakers N`，可选 `--roformer-mode dual`
3. 默认已开启 embedding refine 与未分配填充；若改标过激可 `--no-refine-embedding` 或提高 `--min-similarity`
4. 含快速交替/叠话：试 `--split-mode diarize_bss --overlap-min-sec 0.15 --max-overlap-sec 8`
5. A/B 对比与验收：

```bash
# 第一步: 说话人计数预检
python src/cli/separate_audio.py 桔梗犬夜叉片段.mp4 --out-dir ./out_count \
  --count-speakers --max-speakers 6

# 第二步: 精确分离 (把 N 换成预检结果的 speaker_count)
python src/cli/separate_audio.py 桔梗犬夜叉片段.mp4 --out-dir ./out_multi_v2 \
  --multi-stems --split-mode diarize --min-speakers N --max-speakers N

python scripts/compare_speaker_sep.py ./out_multi ./out_multi_v2 --stem 桔梗犬夜叉片段
```

验收参考（`桔梗犬夜叉片段`）：`split_report.json` 中 `coverage_ratio` ≥ 0.55；对比脚本默认窗 `93:104`、`129:142.5` 上正确角色轨 RMS 应高于错误轨；人工试听交替对白 ≥80% 在正确 `speaker_*` 轨。

**准确率边界：**

- 交替对白、N≤6：通常可用；短喊名/气声边界可能偏移。
- 2 人重叠：`diarize_bss` 优于纯切分，但 WSJ0 训练模型对日语动漫需 A/B。
- 3 人重叠：`diarize_bss` 下自动调用 SepFormer 3mix 做 3 路盲分离，再用 embedding 归属；WSJ0-3mix 训练模型对日语动漫需 A/B。
- 4+ 人同时说话：仍可能串音，建议先 `--count-speakers` 确认人数后分段处理。

---

1. 识别视频音轨语言（打印语种与置信度 Top-5）  
2. 将音频转写为文字（含时间轴），或 **OCR 识别画面烧录字幕**（`--subs-source ocr`）  
3. 可选 **自动翻译字幕**（`--translate-to zh`）  
4. 把字幕烧录到画面下方（也可只导出文本/SRT）  
5. **（可选）** AI 配音：替换人声，**默认保留 BGM**；支持 edge-tts / RVC / GPT-SoVITS 音色克隆

---

## Linux 部署与使用

### 1. 依赖

- Python 3.10+（推荐 conda，如环境名 `py312`）
- 系统 `ffmpeg`（含 `ffprobe`）
- GPU 可选（有 CUDA 时自动用 GPU）
- 启用 `--dub` 时需联网（edge-tts 调用 Microsoft TTS 服务）

### 2. 安装

```bash
# ffmpeg
sudo apt update
sudo apt install -y ffmpeg
# 验证
ffmpeg -version

# 进入项目并安装 Python 依赖
cd /path/to/ai_model/model/video2text
conda activate py312          # 若使用 conda
pip install -r requirements.txt   # 已含 PyTorch (PyPI, CUDA 13) + BS-RoFormer 等全部依赖

# 无 NVIDIA GPU 时改用 CPU 版 PyTorch:
#   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
#   pip install -r requirements.txt  # 已装的 CPU 版会被识别, 不重复下载
```

### 3. 使用

```bash
cd /path/to/ai_model/model/video2text
conda activate py312

# 自动检测语言 + 转写 + 烧录字幕到视频下方
python transcribe_video.py /path/to/video.mp4

# 指定中文、更大模型（中文建议 small / medium）
python transcribe_video.py /path/to/video.mp4 --model small --language zh

# 只导出字幕/文本，不生成带字幕视频
python transcribe_video.py /path/to/video.mp4 --no-burn

# 指定输出目录与字幕样式 (默认透明底 + 白字黑描边)
python transcribe_video.py /path/to/video.mp4 \
  --out-dir ./output \
  --font-size 28 \
  --margin-v 48

# 已有 SRT 时只重新烧录, 不跑 Whisper
python transcribe_video.py /path/to/video.mp4 --burn-only

# 恢复不透明黑底框
python transcribe_video.py /path/to/video.mp4 --burn-only --box

# 强制 CPU
python transcribe_video.py /path/to/video.mp4 --device cpu

# 翻译为英文字幕 + 英文 AI 配音 + 烧录字幕（消除原配音）
python transcribe_video.py /path/to/video.mp4 --task translate --dub

# 原语言字幕 + 中文 AI 配音
python transcribe_video.py /path/to/video.mp4 --language zh --dub

# 指定配音语言与音色
python transcribe_video.py /path/to/video.mp4 --task translate --dub \
  --dub-lang en --dub-voice en-US-GuyNeural

# 只生成 AI 配音视频，不烧录字幕
python transcribe_video.py /path/to/video.mp4 --task translate --dub --no-burn

# 画面 OCR 中文字幕 + 中文配音 + 保留 BGM（适合硬字幕视频）
python transcribe_video.py /path/to/video.mp4 --subs-source ocr --ocr-lang zh --dub --dub-lang zh

# GPT-SoVITS 音色克隆（需本地 API: http://127.0.0.1:9880）
python transcribe_video.py /path/to/video.mp4 --subs-source ocr --ocr-lang zh --dub --dub-lang zh \
  --dub-engine gpt-sovits --voice-ref-lang ja

# RVC 音色转换（需角色 .pth 模型; rvc-python 不含于主依赖, 需单独安装或用 --rvc-api）
python transcribe_video.py /path/to/video.mp4 --dub-only --dub-srt ./out/demo.srt --dub-lang zh \
  --dub-engine rvc --rvc-model /path/to/character.pth

# 多说话人：按 speaker_map.json 分男女 edge 音色
# 前置: --whisper-dir 指向的目录需同时含 *.srt 与 *_vocals.wav (先跑 separate_audio.py)
python generate_speaker_map.py --whisper-dir ./out/video_whisper
python transcribe_video.py /path/to/video.mp4 --dub-only --dub-srt ./out/demo.srt --dub --dub-lang zh \
  --dub-engine edge --dub-speaker-map ./out/video_whisper/speaker_map.json \
  --dub-voice-female zh-CN-XiaoxiaoNeural --dub-voice-male zh-CN-YunxiNeural
```

### 4. Linux 常见问题

| 现象 | 处理 |
|------|------|
| `未找到 ffmpeg` | `sudo apt install ffmpeg`，确认 `which ffmpeg` 有路径 |
| CUDA 报错 | 改用 `--device cpu`，或重装匹配驱动的 PyTorch |
| 首次很慢 | 正在下载 Whisper 权重，之后会走本地缓存 |
| AI 配音失败 / 超时 | 确认网络可访问 edge-tts 服务；可用 `--dub-voice` 手动指定音色 |

---

## Windows 部署与使用

### 1. 依赖

- Python 3.10+（[python.org](https://www.python.org/downloads/) 或 Miniconda/Anaconda）
- `ffmpeg`：**推荐** `pip install -r requirements.txt`（含 `imageio-ffmpeg`，脚本会自动使用内置 ffmpeg，无需手动配置 PATH）；也可自行安装系统 ffmpeg
- GPU 可选（需安装 CUDA 版 PyTorch）
- 启用 `--dub` 时需联网（edge-tts 调用 Microsoft TTS 服务）

### 2. 安装 ffmpeg（可选）

执行 `pip install -r requirements.txt` 后，脚本会自动使用 `imageio-ffmpeg` 自带的 ffmpeg，**一般无需额外安装**。

若需使用系统版 ffmpeg（含 ffprobe，部分场景更完整），可任选其一：

1. `winget install Gyan.FFmpeg`
2. 从 [gyan.dev builds](https://www.gyan.dev/ffmpeg/builds/) 下载，解压后将 `bin` 加入系统 **Path**
3. `conda install -c conda-forge ffmpeg`

也可不修改 PATH，运行时手动指定路径：

```powershell
python transcribe_video.py D:\videos\demo.mp4 --ffmpeg C:\ffmpeg\bin\ffmpeg.exe
```

验证系统 ffmpeg（若已安装）：

```powershell
ffmpeg -version
```

### 3. 安装 Python 依赖

在 **PowerShell** 或 **cmd** 中：

```powershell
cd C:\path\to\ai_model\model\video2text

# 若使用 conda
conda activate py312

# 建议先建虚拟环境（可选）
# python -m venv .venv
# .\.venv\Scripts\Activate.ps1

pip install -r requirements.txt   # 已含 PyTorch (PyPI, CUDA 13) + 全部依赖

# 无 NVIDIA GPU 时改用 CPU 版 PyTorch:
#   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
#   pip install -r requirements.txt  # 已装的 CPU 版会被识别, 不重复下载
```

### 4. 使用

路径可用正斜杠 `/` 或反斜杠 `\`；含空格时请加引号。

```powershell
cd C:\path\to\ai_model\model\video2text
conda activate py312

# 自动检测语言 + 转写 + 烧录字幕到视频下方
python transcribe_video.py D:\videos\demo.mp4

# 指定中文、更大模型
python transcribe_video.py D:\videos\demo.mp4 --model small --language zh

# 只导出字幕/文本
python transcribe_video.py D:\videos\demo.mp4 --no-burn

# 指定输出目录与字幕样式（PowerShell 下一行写完即可；或用 ` 续行）
python transcribe_video.py D:\videos\demo.mp4 --out-dir D:\videos\output --font-size 28 --margin-v 48

# 强制 CPU
python transcribe_video.py D:\videos\demo.mp4 --device cpu

# 翻译为英文字幕 + 英文 AI 配音 + 烧录字幕（消除原配音）
python transcribe_video.py D:\videos\demo.mp4 --task translate --dub

# 原语言字幕 + 中文 AI 配音
python transcribe_video.py D:\videos\demo.mp4 --language zh --dub

# 指定配音语言与音色
python transcribe_video.py D:\videos\demo.mp4 --task translate --dub --dub-lang en --dub-voice en-US-GuyNeural

# 只生成 AI 配音视频，不烧录字幕
python transcribe_video.py D:\videos\demo.mp4 --task translate --dub --no-burn

# 画面 OCR 中文字幕 + 中文配音 + 保留 BGM
python transcribe_video.py D:\videos\demo.mp4 --subs-source ocr --ocr-lang zh --dub --dub-lang zh

# GPT-SoVITS 克隆原角色声线 (PowerShell 下一行写完, 勿用 bash 续行符)
python transcribe_video.py D:\videos\demo.mp4 --subs-source ocr --ocr-lang zh --dub --dub-lang zh --dub-engine gpt-sovits
```

### 5. Windows 常见问题

| 现象 | 处理 |
|------|------|
| `未找到 ffmpeg` | 先执行 `pip install -r requirements.txt`（含 imageio-ffmpeg）；或安装系统 ffmpeg / 使用 `--ffmpeg` 指定路径 |
| 烧录字幕失败 | 换带 libass 的 ffmpeg 完整/essentials 包；或先 `--no-burn` 只出 SRT |
| 执行策略禁止激活 venv | PowerShell 可临时：`Set-ExecutionPolicy -Scope Process RemoteSigned` |
| 路径含中文/空格 | 给视频路径加引号，例如 `"D:\我的视频\demo.mp4"` |
| CUDA / 驱动问题 | 使用 `--device cpu` |
| AI 配音失败 / 超时 | 确认网络可访问 edge-tts 服务；可用 `--dub-voice` 手动指定音色 |

---

## 输出说明（两端相同）

默认写到与视频同级的 `<视频名>_whisper/`：

| 文件 | 说明 |
|------|------|
| `*.txt` | 完整转写文本 |
| `*.srt` | 带时间轴的字幕 |
| `*.json` | 语言检测结果 + 分段时间戳 |
| `*_subtitled.mp4`（或原后缀） | 底部烧录字幕后的视频 |
| `*_dub.wav` | AI 配音音轨（启用 `--dub` 时） |
| `*_mixed.wav` | 配音 + 背景音乐混合轨（默认保留 BGM 时） |
| `*_vocals.wav` | 分离的人声轨 (默认 BS-RoFormer) |
| `*_no_vocals.wav` | 分离的背景/伴奏轨 |
| `*_voice_ref.wav` | 自动截取的音色参考片段 |
| `*_subtitled_dubbed.mp4` | 烧录字幕 + AI 配音 |
| `*_dubbed.mp4` | 仅替换配音（配合 `--no-burn --dub` 时） |

控制台会打印：检测到的语言、置信度 Top-5、完整转写文本；启用 `--dub` 时还会打印配音语言、音色及合成进度。

---

## 常用参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--model` | `base` | `tiny` / `base` / `small` / `medium` / `large` / `large-v2` / `large-v3`；越大越准越慢 |
| `--language` | 自动检测 | 如 `zh` `en` `ja` |
| `--task` | `transcribe` | `translate` 则译成英文字幕 |
| `--device` | 自动 | `cuda` / `cpu` |
| `--out-dir` | `<视频名>_whisper` | 输出目录 |
| `--no-burn` | 关 | 不烧录，只出文本/SRT |
| `--burn-only` | 关 | 跳过转写，用已有 SRT 重烧 |
| `--box` | 关 | 黑色不透明底框；默认透明底 + 描边 |
| `--font-size` / `--margin-v` | 22 / 40 | 字幕字号与距底边距 |
| `--subs-source` | `whisper` | `whisper`=音频转写; `ocr`=识别画面烧录字幕 |
| `--ocr-lang` / `--ocr-fps` | `zh` / `2` | OCR 语言与抽帧频率 |
| `--translate-to` | - | 自动翻译字幕（如 `zh`） |
| `--dub-srt` | - | 配音专用字幕（跳过转写/翻译） |
| `--dub-only` | 关 | 跳过转写，仅从 SRT 生成配音 |
| `--dub` | 关 | 启用 AI 配音（默认保留 BGM） |
| `--no-keep-bgm` | 关 | 不保留背景音乐 |
| `--bgm-volume` / `--dub-volume` | `1.0` / `1.0` | 混音音量 |
| `--dub-lang` | 自动 | 配音语言，如 `zh` |
| `--dub-engine` | `edge` | `edge` / `rvc` / `gpt-sovits` |
| `--dub-voice` | 自动 | edge-tts 默认音色；未在映射中标注的段也用此音色 |
| `--dub-voice-female` | 按语言 | 女声（映射值为 `female`/`女` 时） |
| `--dub-voice-male` | 按语言 | 男声（映射值为 `male`/`男` 时） |
| `--dub-speaker-map` | - | 说话人 JSON：SRT 序号 → `male`/`female` 或 edge 音色名 |
| `--voice-ref` | 自动 | 音色参考音频；未指定时从人声自动截取 |
| `--voice-ref-text` | 自动 | GPT-SoVITS 参考文本 |
| `--voice-ref-lang` | `ja` | 参考音频语言（SoVITS prompt_language） |
| `--rvc-model` / `--rvc-index` | - | RVC 模型与 index |
| `--rvc-api` | - | RVC HTTP 服务（如 `http://127.0.0.1:7860`） |
| `--gpt-sovits-api` | `http://127.0.0.1:9880` | GPT-SoVITS API 地址 |
| `--ffmpeg` | 自动 | ffmpeg 路径 |
| `--ffprobe` | 自动 | ffprobe 路径（可选） |

中文视频建议：`--model small` 或 `medium`，并可加 `--language zh`。

**AI 配音与音色克隆：**

| 引擎 | 相似度 | 前置条件 |
|------|--------|----------|
| `edge`（默认） | 低 | 联网；`--dub-voice` 或 `--dub-speaker-map` 分男女音色 |
| `rvc` | 中高 | `--rvc-model`（必填，API 模式同样需要）+ 可选 `--rvc-index`/`--rvc-api`；rvc-python 需单独安装（Python 3.10/3.11 + pip<24.1，其 faiss-cpu/omegaconf 硬 pin 无法并入主 requirements），不便降级 pip 时用 `--rvc-api`+`--rvc-model` 指向本地 RVC 服务 |
| `gpt-sovits` | 高 | 本地启动 GPT-SoVITS API；自动截取 `--voice-ref` |

- 默认 **保留背景音乐**（BS-RoFormer ensemble 分离人声）；用 `--no-keep-bgm` 关闭。
- 硬字幕视频（无独立字幕轨）请用 `--subs-source ocr --ocr-lang zh`。
- 已有中文字幕时用 `--dub-only --dub-srt xxx_zh.srt --dub-lang zh`，勿走音频转写+翻译。
- **多角色男女声**：先 `python generate_speaker_map.py --whisper-dir <输出目录>` 自动生成 `speaker_map.json`，再 `--dub-speaker-map`、`--dub-voice-female`、`--dub-voice-male`（仅 `edge`/`rvc`）。
- `--task translate` 仅能将 Whisper 字幕译为**英文**；中文字幕请用 `--translate-to zh` 或 OCR。

---

## 流程说明

1. 字幕来源：Whisper 音频转写 **或** OCR 画面字幕 **或** 已有 SRT  
2. 可选：自动翻译字幕（`--translate-to`）  
3. **（可选 `--dub`）** 按引擎合成配音：`edge-tts` / `RVC` / `GPT-SoVITS`  
4. BS-RoFormer ensemble 分离人声与 BGM；默认将 AI 配音与 BGM 混合  
5. 用 ffmpeg 输出最终视频：可选烧录字幕 + 替换音轨  

首次运行会下载 Whisper / RoFormer 权重到本机缓存。
