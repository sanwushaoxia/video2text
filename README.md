# video2text（Whisper 视频语音识别与字幕）

从视频或音频生成带时间轴的字幕，支持 Whisper 转写、画面硬字幕 OCR 和已有 SRT。可选翻译、烧录字幕和 AI 配音（edge-tts / RVC / GPT-SoVITS）。配音默认通过 RoFormer 分离保留背景音。[OpenAI Whisper](https://github.com/openai/whisper)

独立分轨命令支持人声/背景二轨、按性别三轨、按说话人多轨，以及叠话分离、试听和局部修正。批处理支持阶段缓存、完整结果缓存和性能报告。

**同时支持 Linux 与 Windows**（Python + Whisper + edge-tts + ffmpeg）。

[命令速查](#命令速查) · [模型文件与 Git LFS](#模型文件与-git-lfs) · [Linux 安装](#linux-部署与使用) · [Windows 安装](#windows-部署与使用)

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
    diarization_refine.py # 声纹改标与碎段合并
    speaker_embedding.py  # 动漫 ECAPA embedding
    speaker_stems.py   # N 说话人轨切分
    overlap_separation.py  # 重叠段 MossFormer2/SepFormer 盲分离
    overlap_tracking.py    # 结合邻近独白和相邻窗口做叠话归属
    paths.py           # 模型目录与 Hugging Face 缓存路径
    stage_cache.py     # 阶段缓存与原子发布
    result_cache.py    # 完整分轨结果缓存
    performance.py     # 阶段计时与资源统计
    dub.py             # AI 配音 (edge / RVC / GPT-SoVITS)
    mix.py             # 配音与 BGM 混音
    render.py          # 字幕烧录 / 音轨替换
    speaker_map.py     # 多角色说话人推断
    pipeline.py        # 完整流水线 CLI
  cli/                 # 单一功能命令行入口
    extract_subs.py    # 提取字幕 (Whisper / OCR)
    burn_subs.py       # 烧录字幕到视频
    separate_audio.py  # 人声/伴奏分离 (默认 BS-RoFormer ensemble)
    correct_overlap.py # 从试听缓存局部修正叠话归属
    audio_to_subs.py   # 音频 → 字幕 (Whisper)
    subs_to_audio.py   # 字幕 → AI 配音
    mix_audio.py       # 混合配音与伴奏
    speaker_map.py     # 生成 speaker_map.json
    transcribe.py      # 完整流水线 (同 transcribe_video.py)

requirements.txt              # Python 依赖
assets/roformer/              # 内置 RoFormer 模型清单和 YAML
models/                       # 模型权重与缓存，部分由 Git LFS 提供
scripts/
  setup_bs_roformer_env.sh    # 创建独立 conda 环境并安装全量依赖 (可选)
  prefetch_roformer_models.py # 预下载 RoFormer 权重与配置
  prefetch_pyannote_models.py # 预下载 pyannote 模型
  batch_separate_audio.py     # 批量分轨与逐项报告
  compare_roformer_ab.py      # no_vocals 客观 A/B (RMS / nv/v)
  compare_speaker_sep.py      # 多角色轨能量 / 重叠分离 A/B
tests/                        # 分轨、缓存及输出流程回归测试
```

根目录保留兼容入口：`transcribe_video.py`、`generate_speaker_map.py`（薄包装，转发到 `src/video2text`）。

---

## 命令速查

在仓库根目录运行示例，将 `video.mp4` 和输入路径换成你的文件。先安装 `requirements.txt` 中的 Python 依赖。Linux 建议安装系统 FFmpeg，Windows 可用 `imageio-ffmpeg` 提供的 FFmpeg。使用默认 RoFormer 分离前，可按[模型文件与 Git LFS](#模型文件与-git-lfs)拉取权重。

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

# OCR 字幕 + GPT-SoVITS 配音 (需本地 API http://127.0.0.1:9880)
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
# --srt / --dub-lang / --out-dir 放在视频路径之前，引擎参数放在路径之后
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
# --srt / --dub-lang / --out-dir 放在视频路径之前，引擎参数放在路径之后
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

### RoFormer 权重与离线运行

默认 `instrumental_full` 使用下表中的 5 个文件。仓库随 Git 提供清单和 YAML，两个 ckpt 通过 Git LFS 下载。权重和配置齐全时，推理直接从磁盘加载。

| 文件 | 用途 |
|------|------|
| `download_checks.json` | 模型清单 |
| `melband_roformer_inst_v1e_plus.ckpt` | ensemble 权重 1，约 871 MiB |
| `config_melbandroformer_inst.yaml` | 模型 1 配置 |
| `mel_band_roformer_instrumental_becruily.ckpt` | ensemble 权重 2，约 871 MiB |
| `config_mel_band_roformer_instrumental_becruily.yaml` | 模型 2 配置 |

使用其他 preset 或模型时，先运行预下载命令，并传入和分轨时相同的模型参数。自定义目录用 `--model-dir` 预下载，分轨时用 `--roformer-model-dir` 指向同一目录。缺少清单或默认配置时，程序可从仓库内的文件补齐。离线机器可从已下载的机器复制整套文件。若访问 GitHub 超时，可先将 `HTTPS_PROXY` 设为实际可用的代理地址再重试。[内置模型清单与配置](assets/roformer/)

```bash
# 下载并校验默认 ensemble
python scripts/prefetch_roformer_models.py

# 只需一个权重的模式示例
python scripts/prefetch_roformer_models.py --roformer-mode single \
  --roformer-model model_bs_roformer_ep_317_sdr_12.9755.ckpt
python src/cli/separate_audio.py video.mp4 --out-dir ./out \
  --roformer-mode single --roformer-model model_bs_roformer_ep_317_sdr_12.9755.ckpt
```

---

## 模型文件与 Git LFS

| 默认目录 | 内容与获取方式 |
|----------|----------------|
| `models/roformer/` | 默认 ensemble 的两个权重走 LFS，清单和 YAML 随 Git。其他权重可用 RoFormer 预下载脚本获取 |
| `models/whisper/` | Whisper 权重由转写命令首次加载时下载，本地目录受 `.gitignore` 管理 |
| `models/huggingface/` | Hugging Face 缓存。已提交的普通文件按 LFS 规则处理，缺少的模型从 Hub 获取 |
| `models/speechbrain/` | SpeechBrain 本地模型目录。SepFormer 等模型首次用到时可从 Hub 获取 |

默认以仓库的 `models/` 为根目录，可用 `VIDEO2TEXT_MODELS_DIR` 覆盖。RoFormer 和 Whisper 的独立覆盖变量分别是 `VIDEO2TEXT_ROFORMER_MODEL_DIR`、`VIDEO2TEXT_WHISPER_MODEL_DIR`。Hugging Face 用 `HF_HOME` 覆盖缓存根目录，用 `HF_HUB_CACHE` 单独覆盖 Hub 缓存。Git LFS 始终检出到仓库内的跟踪路径，程序按运行时设置的目录加载权重。

### 下载仓库里的大文件

Ubuntu/Debian 可执行 `sudo apt install git-lfs`。Windows 可安装带有 Git LFS 的 Git for Windows，或单独安装 Git LFS。先验证客户端并初始化过滤器，再克隆仓库。默认克隆会自动下载当前分支的 LFS 文件。[Git LFS 安装文档](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage)

```bash
git lfs version
git lfs install
git clone https://github.com/sanwushaoxia/video2text.git
cd video2text
```

已有工作副本时，先切换到 `main`，再执行以下命令。如果模型文件只有以 `version https://git-lfs.github.com/spec/v1` 开头的几行文本，`git lfs pull origin` 会下载并检出实际内容。完整下载后可用 `git lfs fsck` 检查当前提交的本地 LFS 对象。

```bash
git lfs install --local
git pull --ff-only origin main
git lfs pull origin
git lfs fsck
```

只要源码或部分模型时，可在克隆时跳过权重。下面只下载 RoFormer 目录中的 LFS 文件，两个默认权重合计约 1.70 GiB。若需要仓库里的全部权重，再运行 `git lfs pull origin`。在 PowerShell 中用 `$env:GIT_LFS_SKIP_SMUDGE = "1"` 设置变量，完成 `git clone` 后用 `Remove-Item Env:GIT_LFS_SKIP_SMUDGE` 清除，再执行拉取命令。[Git LFS pull 手册](https://github.com/git-lfs/git-lfs/blob/main/docs/man/git-lfs-pull.adoc)

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/sanwushaoxia/video2text.git
cd video2text
git lfs pull --include="models/roformer/**" origin
git lfs ls-files --include="models/roformer/**" --size
```

### 按需下载与复用缓存

首次使用说话人日记化前，在 Hugging Face 接受 Community-1 的访问条款，并设定 `HF_TOKEN` 或在仓库根目录的 `hf_token` 文件里写入一行 token。即使已有模型缓存，当前日记化入口仍会读取 token。[Community-1 模型页](https://huggingface.co/pyannote/speaker-diarization-community-1)

```bash
python scripts/prefetch_pyannote_models.py
python scripts/migrate_models_to_project.py --dry-run
```

后一个命令预览旧 AppData 或用户缓存的迁移，确认后去掉 `--dry-run` 执行。当前仓库的 `models/speechbrain/sepformer-wsj02mix/` 内有指向原机器路径的符号链接。换机器后如果 SepFormer 加载失败，可用 `VIDEO2TEXT_MODELS_DIR` 指向新的本地模型目录，将已有的 RoFormer 权重和配置复制过去，再让 SpeechBrain 在新目录中按需建立缓存。Windows 上如果 Hugging Face 缓存里的符号链接被检出为文本，也可用新的 `HF_HOME` 重建缓存。

### 提交与上传大文件

仓库的 `.gitattributes` 已为 `models/` 下的 `*.ckpt`、`*.pt` 普通文件，以及 `models/huggingface/`、`models/speechbrain/` 下的普通文件配置 LFS。先在仓库中启用 LFS，再暂存实际要发布的权重。以下以 `new_model.ckpt` 为例，请换成实际文件名。[仓库 LFS 规则](.gitattributes)

```bash
git lfs version
git lfs install --local
git check-attr filter -- models/roformer/new_model.ckpt
git add -- models/roformer/new_model.ckpt
git lfs status
git show :models/roformer/new_model.ckpt | git lfs pointer --check --stdin
git commit -m "Add RoFormer model weights"
git push origin main
```

`git check-attr` 应显示 `filter: lfs`，暂存区的指针检查退出码应为 0。已暂存但尚未提交的普通 blob 可在 LFS 启用后用 `git add --renormalize -- models/roformer/new_model.ckpt` 重新经过过滤，再检查一次指针。规则未覆盖的新路径可先用 `git lfs track "路径或通配规则"` 配置，并将 `.gitattributes` 与权重一同提交。逐个暂存权重和必要配置，避免把整个缓存、锁文件或凭据一起提交。

普通 `git push` 会先通过 LFS 钩子上传对象，再更新 Git 分支。上传中断可重试同一命令，服务端已有的对象会被复用。GitHub LFS 用量计入仓库所有者的存储和下载额度。[GitHub LFS 计费](https://docs.github.com/en/billing/concepts/product-billing/git-lfs)

### 修复未推送提交里的大文件

如果推送因 `GH001` 被拒，先确认超限文件所在的提交。GitHub 对普通 Git 文件的上限是 100 MiB。涉及历史提交时，后续再提交一次删除或启用 LFS，旧 blob 仍会包含在推送里。[GitHub 大文件限制](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)

下面适用于当前在 `main`、工作区干净、问题提交都尚未发布，且这些提交中的 `.gitattributes` 已覆盖相关文件的情况。先更新远端跟踪分支并检查 `git status --short --branch` 与 `git log origin/main..main --oneline`，确认只需处理本地提交。

```bash
git fetch origin
git status --short --branch
git log origin/main..main --oneline
git branch backup/main-before-lfs main
git lfs migrate import --fixup --include-ref=refs/heads/main --exclude-ref=refs/remotes/origin/main
git lfs install --local
git lfs checkout
git lfs fsck origin/main..main
git status --short --branch
git push origin main
```

若刚启用 LFS 时，迁移误将原有普通 blob 显示为工作区修改，先用禁用过滤器的 `git -c filter.lfs.process= -c filter.lfs.clean= -c filter.lfs.smudge= -c filter.lfs.required=false status --short` 确认工作区确实干净，再把相同的 `-c` 参数放在迁移命令的 `git` 与 `lfs` 之间重试。迁移只重写本地待发布提交，随后用普通推送。[Git LFS migrate 手册](https://github.com/git-lfs/git-lfs/blob/main/docs/man/git-lfs-migrate.adoc)

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
| `*_dialogue_review.wav` | `diarize_bss` 中未能可靠分离的叠话，以及未分配到角色轨的人声底轨片段，可能含背景残留 |
| `*_split_report.json` | 每段 speaker_id、重叠分离标记、覆盖率 |

日记化标出每位角色的发声时间。非重叠段直接切分，`diarize_bss` 对同时发声的区间调用 MossFormer2/SepFormer，再用单人片段提取的声纹做一对一归属。长重叠分窗处理，成功归属的相邻窗做短交叉淡化。相邻轮次推测出的重叠只记入报告。报告的 `overlap_bss.regions` 记录每段的样本边界、处理状态和原因。说话人轨与待复核轨以相同长度的 32 位浮点 WAV 输出。

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
| `--diarization-source` | vocals | 多轨日记化输入，`vocals` 用提取人声，`mix` 用原混音，`joint` 对齐双路角色后将有分歧的叠话送去复核 |
| `--overlap-tracking` | 未启用 | 叠话归属结合整片独白声纹、邻近独白与连续分窗 |
| `--save-overlap-review` | 未启用 | 保存每个已尝试分离的窗口，按 WAV 声道顺序试听并人工归属 |
| `--background-output` | direct | `residual` 从对齐后的原混音减去提取对白，可与直接估计背景做 A/B |
| `--overlap-review-threshold` | 未启用 | `diarize_bss` 中用 Community-1 的多人发声概率补充待复核区间。可从 `0.4` 开始验证，会增加一次分割模型推理 |
| `--embedding-model` | litagin anime ECAPA | 重叠段 speaker 匹配 |
| `--bss-backend` | auto | 双人支持 MossFormer2、SepFormer 和 `sepformer-whamr`。`auto` 双人先尝试 MossFormer2，三人使用 SepFormer 3mix |
| `--no-embedding` | 关 | 禁用 embedding (diarize_bss / refine) |
| `--refine-embedding` / `--no-refine-embedding` | 开 | 日记化后 embedding 合并碎段、改标、低置信过滤 |
| `--merge-turn-gap` | 0.4 | 合并同 speaker 相邻碎段最大间隔 (秒) |
| `--min-turn-sec` | 0.5 | 参与 speaker profile 的最短 turn (秒) |
| `--min-similarity` | 0.35 | embedding 改标/过滤最低相似度 |
| `--fill-unassigned` / `--no-fill-unassigned` | 开 | 未分配人声区间按 embedding 填充 |
| `--diarize-preprocess` / `--no-diarize-preprocess` | 开 | 日记化前 vocals 峰值归一化 + 80Hz 高通 |
| `--overlap-min-sec` | 0.15 | BSS 最短时长 (秒)，更短的重叠保留到待复核轨 |
| `--max-overlap-sec` | 8.0 | BSS 单窗最长时长 (秒) |

对有叠话的片段，可同时比较原混音与分离后人声的日记化结果，结合邻近独白和相邻窗口判断声源归属，并保存试听所需的波形。

```bash
python src/cli/separate_audio.py video.mp4 --out-dir ./out_review \
  --multi-stems --split-mode diarize_bss --diarization-source joint \
  --overlap-tracking --save-overlap-review
```

修正时使用[影视评测与局部修正](#影视评测与局部修正)中的命令。

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

可用包含 `flite` 滤镜的 FFmpeg 生成固定双人混音，并测量已知叠话边界下的 SI-SDR。传入用 `mix_bg.wav` 生成分轨的目录后，还会测量背景轨误差和对白残留。

```bash
python scripts/benchmark_overlap.py ./out_overlap_benchmark
python src/cli/separate_audio.py ./out_overlap_benchmark/mix_bg.wav --out-dir ./out_overlap_background
python scripts/benchmark_overlap.py ./out_overlap_benchmark --background-dir ./out_overlap_background
PYTHONPATH=src python -m unittest discover -s tests
```

### 跨录音留出评测

以 AMI 的 ES2004a 和 EN2002a 分别作为开发集和留出集。每组包含两段真实会议片段，以及由该场会议独白制作的双人、三人混音。真实片段使用 AMI 人工说话人标注，受控混音另有逐角色和背景参考 WAV。音频来自 AMI Meeting Corpus（CC BY 4.0，AMI Consortium），时间标注来自 pyannote 的 `only_words` RTTM。[AMI 语料](https://groups.inf.ed.ac.uk/ami/corpus/) · [音频镜像](https://huggingface.co/datasets/FluidInference/ami-corpus-mirror/tree/722d8891643e1e4dc62cfd0d198fa05a1646c3cc) · [固定 RTTM 版本](https://github.com/pyannote/AMI-diarization-setup/tree/67c2d539286e89f68952d5dcf83912bd9f01dfae)

```bash
python scripts/prepare_generalization_benchmark.py out_generalization \
  --rttm-zip ami-rttm.zip --dev ES2004a.Mix-Headset.wav --holdout EN2002a.Mix-Headset.wav
python scripts/evaluate_overlap_generalization.py out_generalization/manifest.json \
  --partition dev --output out_generalization/dev --prepare-vocals
python scripts/evaluate_overlap_generalization.py out_generalization/manifest.json \
  --partition holdout --output out_generalization/holdout --prepare-vocals
```

评测同时记录叠话精确率、召回率、实际分离比例、待复核比例、角色标签错误和背景对白泄漏。已知边界和最佳源排列的波形 SI-SDR 单独报告，便于定位分离模型本身的误差。使用 `--embedding-model wespeaker` 可比较 Community-1 自带的通用声纹。使用 `--backend sepformer-whamr` 可比较带混响训练的双人模型，三人仍使用 3mix。

### 影视评测与局部修正

`scripts/prepare_media_overlap_benchmark.py` 可生成日语 JVS 与普通话 AISHELL-3 受控混音，包含原对白、独立背景和音效参考。用 `--scenes` 可导入自有影视片段及独立人工 RTTM。片段清单中的 `group` 是作品 ID，`speakers` 将 RTTM 标签映射到跨作品一致的演员 ID。评测拒绝在开发集与留出集之间共用作品或演员。[JVS 演示录音](https://github.com/MontrealCorpusTools/japanese-jvs-demo) · [AISHELL-3 官方补充音频](https://sos1sos2sixteen.github.io/aishell3v2/)

每个 `scenes` 条目提供 `id`、`group`、`partition`（`dev` 或 `holdout`）、`language`、`provenance`、`media`、`rttm`、`rttm_recording`、`start`、`end` 和 `speakers`。媒体和 RTTM 路径相对于清单。评测可通过 `--variants oracle oracle_tracking vocals mix tracking joint_tracking` 同时核对人工边界、单路和双路配置。有参考波形时，`reference_correct_role_fraction` 统计所有输出归属正确且各路 SI-SDR 都优于原混音的叠话时长比例。

开启 `--save-overlap-review` 后，报告中带 `preview` 的区间可按声道试听。修正 JSON 以窗口 ID 为键，按声道顺序列出角色 ID，例如 `{"0:0": ["SPEAKER_01", "SPEAKER_00"]}`。以下命令从缓存声源写入新目录，原分轨及修正区间之外的样本保持原样。

```bash
python src/cli/correct_overlap.py out/video_split_report.json corrections.json --out-dir out_corrected
```

背景验收脚本对比两种背景输出，并生成配平响度的匿名 A/B 音频、替换配音片段和评分表。单片段清单可用 `--video` 提供原视频，同时导出 A/B 配音成片。独立背景参考存在时，同时报告对白泄漏、已标注音效窗的瞬态误差和切点误差。评分前可暂不查看 `answer_key.json`。

```bash
python scripts/evaluate_background_quality.py out_media_validation/manifest.json \
  --out-dir out_background_review --replacement independent_dialogue.wav
```

### 性能基准、阶段缓存和批处理

多轨混轨使用定长工作缓冲。与处理采样率一致且至少 10 分钟的 WAV 人声轨按需读取样本区间，其余输入共享一次人声解码。至少 10 分钟的输入会分批把已确认的叠话波形写入工作目录，混轨时按需读回。处理长时间连续叠话时，临时磁盘用量随实际分离的叠话时长和轨数增长，CLI 完成后清理工作目录。

Linux 性能报告记录阶段墙钟时间、实时系数、进程 CPU 时间、磁盘计数、峰值 RSS、产物大小和缓存命中。CUDA 已初始化时也记录进程显存峰值，计时边界同步。阶段可嵌套，进程峰值在长驻批处理 worker 中为累计值。

为缓存和性能报告选择输出目录之外的位置。`--cache-dir` 根据源文件内容、参数、已安装模型和运行环境复用解码、RoFormer 与日记化产物。失败后可用同样的命令重试，已完成的阶段可复用。为同一输入和配置重复导出时，加上 `--cache-results` 可跳过解码、模型推理和混轨，直接发布完整音轨与报告。该选项需要 `--multi-stems`，缓存目录会额外保存一套分轨。使用 `--atomic-output` 时，指定一个新输出目录，等全部分轨和报告成功后再发布。

```bash
python src/cli/separate_audio.py video.mp4 --multi-stems --split-mode diarize_bss \
  --out-dir out_result --atomic-output --cache-dir out_stage_cache --cache-results \
  --performance-report out_result_performance.json

python scripts/batch_separate_audio.py a.mp4 b.mp4 --out-dir out_batch \
  --cache-dir out_stage_cache --device cpu --workers 2 --threads-per-worker 2
```

批处理的每项任务都有独立目录、日志和性能 JSON。自动选择设备时用单个 worker。CPU 单权重 RoFormer 可通过 `--reuse-cpu-roformer` 在各 worker 内复用一份已加载模型。额外分离参数放在 `--` 之后。批量缓存完整多轨结果时，先在 `--` 之前指定 `--cache-dir`，再传入 `-- --multi-stems --split-mode diarize_bss --cache-results`。批次重试可把 `batch_report.json` 中失败的输入重新传入。

在 i5-13490F 上，单任务、单 RoFormer 权重测试中，CPU 线程从 2 增至 4 后耗时下降 34%–40%，增至 8 的差异约 1%。单任务可将 `OMP_NUM_THREADS`、`MKL_NUM_THREADS` 和 `OPENBLAS_NUM_THREADS` 同时设为 4。批处理用 `--threads-per-worker` 控制每个 worker 的线程数。

本地 RoFormer 权重及配置完整时，推理直接从磁盘加载。需要下载或恢复文件时才查询远端大小。CUDA 单模型可用 `VIDEO2TEXT_ROFORMER_PRECISION=fp16` 选择原生 FP16，两种精度各自使用缓存。默认使用 FP32。RTX 3060 上已用 `melband_roformer_inst_v1e_plus.ckpt` 对比日语和普通话参考片段，其他模型可通过同样的分离评测核验。使用标准 Linux PyTorch CUDA wheel 并显式设置 `OMP_NUM_THREADS` 或 `MKL_NUM_THREADS` 时，完整缓存命中可通过独立 CUDA 运行库检查设备，避免在主进程加载 PyTorch。

```bash
VIDEO2TEXT_ROFORMER_PRECISION=fp16 python src/cli/separate_audio.py video.mp4 \
  --multi-stems --split-mode diarize_bss --roformer-mode single \
  --roformer-model melband_roformer_inst_v1e_plus.ckpt --out-dir out_fast
```

混轨基准使用独立进程交错运行优化前后的实现，对所有解码 WAV 的 PCM 哈希和覆盖统计做一致性检查。长片测试可将短参考片段重复到指定时长。

```bash
python scripts/benchmark_speaker_tracks.py path/to/vocals.wav \
  --baseline path/to/before/speaker_stems.py --seconds 3600 --pairs 1 \
  --out out_bench/renderer.json
```

验收参考（`桔梗犬夜叉片段`）：在已标注的对白窗里检查角色归属和串轨。报告中的 `coverage_ratio` 以整段音频时长为分母，包含无对白的时段。

**准确率边界：**

- 交替对白、N≤6：通常可用；短喊名/气声边界可能偏移。
- 2 人重叠：`diarize_bss` 优于纯切分，但 WSJ0 训练模型对日语动漫需 A/B。
- 3 人重叠：`diarize_bss` 下自动调用 SepFormer 3mix 做 3 路盲分离，再用 embedding 归属；WSJ0-3mix 训练模型对日语动漫需 A/B。
- 4+ 人同时说话：原片段保留在待复核轨。

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
- Git LFS（用于克隆或更新仓库里的模型权重）
- GPU 可选（有 CUDA 时自动用 GPU）
- 启用 `--dub` 时需联网（edge-tts 调用 Microsoft TTS 服务）

### 2. 安装

```bash
sudo apt update
sudo apt install -y ffmpeg git-lfs
ffmpeg -version
git lfs version

# 进入项目并安装 Python 依赖
cd /path/to/video2text
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
- Git LFS（获取模型前运行 `git lfs version` 验证安装）
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
cd C:\path\to\video2text

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

Whisper 首次加载时会下载所选权重。默认 RoFormer ensemble 可先通过 Git LFS 拉取，其他模型按需下载。具体路径见[模型文件与 Git LFS](#模型文件与-git-lfs)。
