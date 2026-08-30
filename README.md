# video2text（Whisper 视频语音识别与字幕）

基于 [OpenAI Whisper](https://github.com/openai/whisper) 的视频转文字工具：自动识别视频语言、将音频转成文字，并把字幕烧录到画面下方。

**同时支持 Linux 与 Windows**（Python + Whisper + 系统 `ffmpeg`）。

---

## 功能概览

1. 识别视频音轨语言（打印语种与置信度 Top-5）  
2. 将音频转写为文字（含时间轴）  
3. 把字幕烧录到画面下方（也可只导出文本/SRT）

---

## Linux 部署与使用

### 1. 依赖

- Python 3.10+（推荐 conda，如环境名 `py312`）
- 系统 `ffmpeg`
- GPU 可选（有 CUDA 时自动用 GPU）

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
```

### 4. Linux 常见问题

| 现象 | 处理 |
|------|------|
| `未找到 ffmpeg` | `sudo apt install ffmpeg`，确认 `which ffmpeg` 有路径 |
| CUDA 报错 | 改用 `--device cpu`，或重装匹配驱动的 PyTorch |
| 首次很慢 | 正在下载 Whisper 权重，之后会走本地缓存 |

---

## Windows 部署与使用

### 1. 依赖

- Python 3.10+（[python.org](https://www.python.org/downloads/) 或 Miniconda/Anaconda）
- `ffmpeg`，且加入系统 **PATH**
- GPU 可选（需安装 CUDA 版 PyTorch）

### 2. 安装 ffmpeg

1. 从 [ffmpeg 官网](https://ffmpeg.org/download.html) 或 [gyan.dev builds](https://www.gyan.dev/ffmpeg/builds/) 下载 Windows 构建（如 `ffmpeg-release-essentials.zip`）。  
2. 解压到例如 `C:\ffmpeg\`。  
3. 将 `C:\ffmpeg\bin` 加入系统环境变量 **Path**。  
4. **重新打开** PowerShell /「命令提示符」，验证：

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
```

### 5. Windows 常见问题

| 现象 | 处理 |
|------|------|
| `未找到 ffmpeg` | 确认已加入 PATH，并**新开**终端；在终端执行 `ffmpeg -version` |
| 烧录字幕失败 | 换带 libass 的 ffmpeg 完整/essentials 包；或先 `--no-burn` 只出 SRT |
| 执行策略禁止激活 venv | PowerShell 可临时：`Set-ExecutionPolicy -Scope Process RemoteSigned` |
| 路径含中文/空格 | 给视频路径加引号，例如 `"D:\我的视频\demo.mp4"` |
| CUDA / 驱动问题 | 使用 `--device cpu` |

---

## 输出说明（两端相同）

默认写到与视频同级的 `<视频名>_whisper/`：

| 文件 | 说明 |
|------|------|
| `*.txt` | 完整转写文本 |
| `*.srt` | 带时间轴的字幕 |
| `*.json` | 语言检测结果 + 分段时间戳 |
| `*_subtitled.mp4`（或原后缀） | 底部烧录字幕后的视频 |

控制台会打印：检测到的语言、置信度 Top-5、完整转写文本。

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

中文视频建议：`--model small` 或 `medium`，并可加 `--language zh`。

---

## 流程说明

1. `ffmpeg` 抽取 16kHz 单声道音频  
2. Whisper `detect_language` 识别语种  
3. Whisper `transcribe` 得到分段文本与时间戳  
4. 生成 SRT，再用 `ffmpeg` `subtitles` 滤镜烧录到画面底部居中  

首次运行会下载对应 Whisper 权重到本机缓存目录（Linux / Windows 均会缓存，之后无需重复下载）。
