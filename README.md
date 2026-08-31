# video2text（Whisper 视频语音识别与字幕）

基于 [OpenAI Whisper](https://github.com/openai/whisper) 的视频转文字工具：自动识别视频语言、将音频转成文字，并把字幕烧录到画面下方。可选 **AI 配音**（edge-tts / RVC / GPT-SoVITS），默认 **保留背景音乐**（Demucs 人声分离）。

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
    separate.py        # Demucs 人声分离
    f0_analysis.py     # 基频 (F0) 分析
    gender_split.py    # 自动男/女声切分
    dub.py             # AI 配音 (edge / RVC / GPT-SoVITS)
    mix.py             # 配音与 BGM 混音
    render.py          # 字幕烧录 / 音轨替换
    speaker_map.py     # 多角色说话人推断
    pipeline.py        # 完整流水线 CLI
  cli/                 # 单一功能命令行入口
    extract_subs.py    # 提取字幕 (Whisper / OCR)
    burn_subs.py       # 烧录字幕到视频
    separate_audio.py  # 人声/伴奏分离
    audio_to_subs.py   # 音频 → 字幕 (Whisper)
    subs_to_audio.py   # 字幕 → AI 配音
    mix_audio.py       # 混合配音与伴奏
    speaker_map.py     # 生成 speaker_map.json
    transcribe.py      # 完整流水线 (同 transcribe_video.py)
```

根目录保留兼容入口：`transcribe_video.py`、`generate_speaker_map.py`（薄包装，转发到 `src/video2text`）。

**开发：** 修改 `src/` 后运行 `python tools/check_imports.py --smoke` 检查 import 与模块变量是否完整。

**分步示例：**

```bash
# 1. Whisper 提取字幕
python src/cli/audio_to_subs.py video.mp4 --language ja --out-dir ./out

# 2. 人声分离 (二轨)
python src/cli/separate_audio.py video.mp4 --out-dir ./out

# 2b. 三轨自动分离 (背景 + 男声 + 女声, 无需人工 speaker_map)
python src/cli/separate_audio.py video.mp4 --out-dir ./out --three-stems --split-mode whisper_f0 --language ja
# 纯音频日记化 (需 pip install -r requirements-diarize.txt 与 HF_TOKEN)
python src/cli/separate_audio.py video.mp4 --out-dir ./out --three-stems --split-mode diarize --max-speakers 4

# 3. 生成多角色映射
python src/cli/speaker_map.py --whisper-dir ./out

# 4. 字幕转配音
python src/cli/subs_to_audio.py video.mp4 --dub-lang zh --srt ./out/video.srt \
  --dub-engine edge --dub-speaker-map ./out/speaker_map.json

# 5. 混合 BGM（若上一步未自动混音）
python src/cli/mix_audio.py --dub ./out/video_dub.wav --bgm ./out/video_no_vocals.wav -o ./out/video_mixed.wav

# 6. 烧录字幕
python src/cli/burn_subs.py video.mp4 --srt ./out/video.srt -o ./out/video_subtitled.mp4

# 一键完整流程（与原先相同）
python transcribe_video.py video.mp4 --dub --dub-lang zh
```

---

## 三轨自动分离 (背景 / 男声 / 女声)

默认 `separate_audio.py` 仅做 Demucs **二轨**分离 (`*_no_vocals.wav` + `*_vocals.wav`)。加 `--three-stems` 可自动输出三轨，**无需人工编辑 speaker_map**：

| 输出文件 | 说明 |
|---------|------|
| `*_no_vocals.wav` | 背景音 (Demucs) |
| `*_male_vocals.wav` | 男声片段叠加 |
| `*_female_vocals.wav` | 女声片段叠加 |
| `*_split_report.json` | 每段 F0、性别、置信度 (自动质检, 非手工 map) |

**原理：** Demucs 深度学习分离背景与人声；男女声按**时间段切分**到不同轨道 (不是 FFT 滤波, 也不能在两人**同时重叠**说话时物理分离)。

**模式选型：**

| `--split-mode` | 时间轴来源 | 适用场景 |
|----------------|-----------|---------|
| `srt_f0` (**推荐**) | 已有 SRT/OCR 字幕 + F0 + 文本规则 | 硬字幕/完整 SRT，覆盖全片对白 |
| `whisper_f0` | Whisper 自动转写句级时间戳 + F0 | 无 SRT、需 ASR 生成时间轴 |
| `diarize` | pyannote 说话人日记化 + F0 | 不想跑 ASR、说话人数未知 |

| `--gender-backend` | 说明 |
|--------------------|------|
| `slice` (默认) | 按 SRT 时间段从 Demucs 人声轨切到男/女轨 |
| `sepformer` (实验) | SpeechBrain SepFormer 双路分离后再组装；重叠串音场景可 A/B |

```bash
# 推荐: OCR/SRT 全片时间轴 + 自适应 F0 + 更强 Demucs
python src/cli/separate_audio.py 桔梗犬夜叉片段.mp4 --out-dir ./out_v2 \
  --three-stems --split-mode srt_f0 \
  --srt 桔梗犬夜叉片段_whisper/桔梗犬夜叉片段.srt \
  --demucs-model htdemucs_ft --demucs-shifts 2

# 短喊名/短句 (如「桔梗」「犬夜叉」): 边界 padding + onset 对齐 + 段间空隙回填
python src/cli/separate_audio.py 桔梗犬夜叉片段.mp4 --out-dir ./out_v3 \
  --three-stems --split-mode srt_f0 \
  --srt 桔梗犬夜叉片段_whisper/桔梗犬夜叉片段.srt \
  --demucs-model htdemucs_ft --demucs-shifts 2 \
  --slice-pad-ms 120 --align-short-segments --fill-gap-ms 400

# out_v4: 分类对齐 (shout/phrase) + 长窗多岛 + 窗内人声回收
python src/cli/separate_audio.py 桔梗犬夜叉片段.mp4 --out-dir ./out_v4 \
  --three-stems --split-mode srt_f0 \
  --srt 桔梗犬夜叉片段_whisper/桔梗犬夜叉片段.srt \
  --demucs-model htdemucs_ft --demucs-shifts 2 \
  --slice-pad-ms 120 --align-short-segments --fill-gap-ms 400 \
  --shout-min-ms 600 --shout-tail-pad-ms 250 \
  --recover-window-vocals \
  --recover-vocal-bleed \
  --speaker-map 桔梗犬夜叉片段_whisper/speaker_map.json

# 轻量: Whisper + F0 (无 SRT 时)
python src/cli/separate_audio.py input.mp4 --out-dir ./out \
  --three-stems --split-mode whisper_f0 --language ja --whisper-model base

# SepFormer 实验 (需 pip install -r requirements-sepformer.txt)
python src/cli/separate_audio.py input.mp4 --out-dir ./out_sepformer \
  --three-stems --split-mode srt_f0 --srt ./subs.srt --gender-backend sepformer

# 纯音频: pyannote 日记化 (需额外依赖与 HF Token)
pip install -r requirements-diarize.txt
# 在 HuggingFace 接受 pyannote/speaker-diarization-3.1 协议后:
export HF_TOKEN=hf_xxxx   # Windows: set HF_TOKEN=hf_xxxx
python src/cli/separate_audio.py input.mp4 --out-dir ./out \
  --three-stems --split-mode diarize --min-speakers 1 --max-speakers 4
```

**准确率边界：**

- 童声、旁白、BGM 残留可能导致 F0 误判；查看 `*_split_report.json` 中 `confidence`、`coverage_ratio` 核对。
- 多人同性别会合并到同一轨；pyannote 可能 over/under-segment。
- 重叠语音仍会串音；可试 `--gender-backend sepformer` 做 A/B（WSJ0 英文模型，日语动漫效果需实测）。

**调参：** `--f0-threshold 200` (Hz, 默认 200 + 自适应)、`--min-voiced-ratio 0.25`、`--demucs-model htdemucs_ft`、`--demucs-shifts 2`。

**短音 / 喊名切分 (「ki」落在边界外、段间两轨皆空):**

| 参数 | 默认 | 说明 |
|------|------|------|
| `--slice-pad-ms` | 120 | 每段切片前后扩展毫秒, 包住略早于 SRT 起点的 onset |
| `--slice-pad-end-ms` | 同 start | 仅终点 padding (可选) |
| `--align-short-segments` / `--no-align-short-segments` | srt_f0 时开启 | 分类对齐: shout / phrase / short |
| `--fill-gap-ms` | 400 | 相邻 SRT 段间短空隙回填; `0` 禁用 |
| `--shout-min-ms` | 600 | 喊名最短切片 (含 ki+kyo 尾音) |
| `--shout-tail-pad-ms` | 250 | 喊名 VAD 结束后额外延长 |
| `--shout-all-islands` | 长窗>5s 自动 | 强制 OCR 长窗内所有语声岛归入该段 |
| `--recover-window-vocals` / `--no-recover-window-vocals` | srt_f0 时开启 | SRT 窗内未分配人声回收 |
| `--recover-vocal-bleed` / `--no-recover-vocal-bleed` | srt_f0 时开启 | 从 `no_vocals` 回收 Demucs 泄漏 (如 ki/好温暖), 并清理背景轨 |
| `--speaker-map` | 无 | 用已有 map 覆盖 gender 后重切 |

**Report 新字段** (`*_split_report.json`):

- 每段: `aligned_start/end`, `srt_start/end`, `align_cue_type`, `speech_islands`, `recovered_sec`, `bleed_recovered_sec`
- 汇总: `stats.stem_assigned_sec`, `stats.unassigned_vocal_sec` (目标较 out_v3 明显下降)

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
pip install -r requirements.txt

# 有 NVIDIA GPU 时, 建议安装对应 CUDA 的 PyTorch
# 参见 https://pytorch.org/get-started/locally/
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

# RVC 音色转换（需角色 .pth 模型）
pip install -r requirements-rvc.txt
python transcribe_video.py /path/to/video.mp4 --dub-only --dub-srt ./out/demo.srt --dub-lang zh \
  --dub-engine rvc --rvc-model /path/to/character.pth

# 多说话人：按 speaker_map.json 分男女 edge 音色
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

pip install -r requirements.txt

# 有 NVIDIA GPU 时, 按官网命令安装 CUDA 版 PyTorch
# https://pytorch.org/get-started/locally/
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

# GPT-SoVITS 克隆原角色声线
python transcribe_video.py D:\videos\demo.mp4 --subs-source ocr --ocr-lang zh --dub --dub-lang zh \
  --dub-engine gpt-sovits
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
| `*_vocals.wav` | Demucs 分离的人声轨 |
| `*_voice_ref.wav` | 自动截取的音色参考片段 |
| `*_subtitled_dubbed.mp4` | 烧录字幕 + AI 配音 |
| `*_dubbed.mp4` | 仅替换配音（配合 `--no-burn --dub` 时） |

控制台会打印：检测到的语言、置信度 Top-5、完整转写文本；启用 `--dub` 时还会打印配音语言、音色及合成进度。

---

## 常用参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--model` | `base` | `tiny` / `base` / `small` / `medium` / `large-v3`；越大越准越慢 |
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
| `rvc` | 中高 | `--rvc-model` 或 `--rvc-api`；可选 `pip install -r requirements-rvc.txt` |
| `gpt-sovits` | 高 | 本地启动 GPT-SoVITS API；自动截取 `--voice-ref` |

- 默认 **保留背景音乐**（Demucs 分离人声）；用 `--no-keep-bgm` 关闭。
- 硬字幕视频（无独立字幕轨）请用 `--subs-source ocr --ocr-lang zh`。
- 已有中文字幕时用 `--dub-only --dub-srt xxx_zh.srt --dub-lang zh`，勿走音频转写+翻译。
- **多角色男女声**：先 `python generate_speaker_map.py --whisper-dir <输出目录>` 自动生成 `speaker_map.json`，再 `--dub-speaker-map`、`--dub-voice-female`、`--dub-voice-male`（仅 `edge`/`rvc`）。
- `--task translate` 仅能将 Whisper 字幕译为**英文**；中文字幕请用 `--translate-to zh` 或 OCR。

---

## 流程说明

1. 字幕来源：Whisper 音频转写 **或** OCR 画面字幕 **或** 已有 SRT  
2. 可选：自动翻译字幕（`--translate-to`）  
3. **（可选 `--dub`）** 按引擎合成配音：`edge-tts` / `RVC` / `GPT-SoVITS`  
4. Demucs 分离人声与 BGM；默认将 AI 配音与 BGM 混合  
5. 用 ffmpeg 输出最终视频：可选烧录字幕 + 替换音轨  

首次运行会下载 Whisper / Demucs 权重到本机缓存。
