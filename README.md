# STORYVID 🎬

Turn one prompt into a complete, narrated video made of consistent AI-generated visuals, synchronized voice narration, dynamic motion, and timed captions.

```
prompt ──▶ story ──▶ scene split ──▶ images ──▶ local voice ──▶ sync & motion ──▶ final.mp4
          (LLM)         (LLM)       (GenAI)       (TTS)           (FFmpeg)        (FFmpeg)
```

| Stage | Where it runs | Supported Engines & Providers |
|---|---|---|
| **Story & Scene Split** | Cloud or Local | NVIDIA NIM, OpenAI, Groq, OpenRouter, Ollama/Local, Custom AI endpoint |
| **Image Generation** | Cloud or Local | Google Gemini, Pollinations AI (100% Free), OpenAI (DALL-E), Custom AI endpoint |
| **Voice (TTS)** | **Local** | Kokoro (default), Chatterbox (voice clone), Piper, espeak-ng |
| **Sync, Motion, Captions** | Local | FFmpeg (Ken Burns zoom/pan, audio-driven duration, SRT/VTT) |

There are two ways to use StoryVid: an interactive **Web UI** (`storyvid_web.py`) with live SSE progress, in-page scene editor, and video player, or the **CLI** (`storyvid.py`). Both share identical project storage and caching.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Prerequisites & Requirements](#prerequisites--requirements)
3. [Configuring AI Providers & Keys](#configuring-ai-providers--keys)
   - [Option A: Web UI Settings (Recommended)](#option-a-web-ui-settings-recommended)
   - [Option B: .env File](#option-b-env-file)
   - [Option C: Environment Variables](#option-c-environment-variables)
   - [Free / No-Key Mode](#free--no-key-mode)
4. [Supported Story LLM Providers](#supported-story-llm-providers)
5. [Supported Image Generation Providers](#supported-image-generation-providers)
6. [Voice Engines (Local TTS)](#voice-engines-local-tts)
7. [Web UI Guide](#web-ui-guide)
8. [CLI Reference](#cli-reference)
9. [Configuration Reference](#configuration-reference)
10. [Output Structure & Caching](#output-structure--caching)
11. [Troubleshooting](#troubleshooting)

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Sami001-OG/storyvid.git
cd storyvid
```

### 2. Set up Python environment

> ⚠️ **Important:** StoryVid requires **Python 3.10 to 3.12** (Python 3.12 recommended). `kokoro-onnx` depends on `onnxruntime`, which does not yet provide pre-built wheels for Python 3.13+ or pre-release builds like Python 3.15.

**Windows (PowerShell / Command Prompt):**
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Make sure **FFmpeg** is installed and on your system `PATH`:
- **Windows:** `winget install Gyan.FFmpeg` or `choco install ffmpeg`
- **macOS:** `brew install ffmpeg`
- **Ubuntu/Debian:** `sudo apt update && sudo apt install ffmpeg`

### 4. Start the Web UI

```bash
python storyvid_web.py
```

Open your browser to:
👉 **[http://127.0.0.1:8765](http://127.0.0.1:8765)**

*(Or run via CLI: `python storyvid.py "a cyber detective in neo-tokyo" -n 8`)*

---

## Prerequisites & Requirements

- **Python 3.10 – 3.12**: Verified with Python 3.12 on Windows, macOS, and Linux.
- **FFmpeg**: Required for audio normalization, duration measuring, Ken Burns motion video rendering, subtitle burning, and stream concatenation.
- **Python Packages**: `openai`, `google-genai`, `kokoro-onnx`, `numpy`, `rich`, `pydantic>=2.0`.
- **Hardware**: Runs comfortably on CPU using Kokoro TTS. A dedicated GPU (NVIDIA CUDA or Apple Silicon MPS) is optional and enables Chatterbox voice cloning.

---

## Configuring AI Providers & Keys

StoryVid provides flexible AI integration for both Story generation and Image generation.

### Option A: Web UI Settings (Recommended)

1. Start the server: `python storyvid_web.py`
2. Open **[http://127.0.0.1:8765](http://127.0.0.1:8765)**
3. Under **NEW JOB**, expand **⚙ AI INTEGRATION & PRESETS**
4. Select your preferred **Story LLM** and **Image Generation** providers
5. Enter your API key(s) or custom endpoint URLs
6. Click **💾 Save as defaults** — your configuration is saved automatically to `.env` and loaded on future launches!

### Option B: .env File

Create a `.env` file in the project root:

```ini
# --- Story LLM ---
SV_LLM_URL=https://integrate.api.nvidia.com/v1
SV_LLM_MODEL=meta/llama-3.2-11b-vision-instruct
NVIDIA_API_KEY=nvapi-...

# --- Image Generation ---
SV_IMAGE_PROVIDER=gemini
SV_IMAGE_MODEL=gemini-2.5-flash-image
GEMINI_API_KEY=AIzaSy...

# Or use Pollinations AI (Free, no key required):
# SV_IMAGE_PROVIDER=pollinations
```

*(Note: `.env` is git-ignored for your security).*

### Option C: Environment Variables

**Windows (PowerShell):**
```powershell
$env:NVIDIA_API_KEY="nvapi-..."
$env:GEMINI_API_KEY="AIzaSy..."
python storyvid_web.py
```

**macOS / Linux:**
```bash
export NVIDIA_API_KEY="nvapi-..."
export GEMINI_API_KEY="AIzaSy..."
python storyvid_web.py
```

### Free / No-Key Mode

You can run StoryVid without paid image generation keys:
1. Set **Image Provider** to `pollinations` (Pollinations AI generates images for free via Flux without requiring an API key or account).
2. For Story generation, you can use:
   - **Local LLMs via Ollama**: `http://localhost:11434/v1` with model `llama3.2`
   - **Groq Free Tier**: Fast inference with free developer credits
   - **NVIDIA NIM Free Trial**: 1,000 free inference credits on [build.nvidia.com](https://build.nvidia.com)

---

## Supported Story LLM Providers

The Story LLM invents the visual style, characters, and breaks the narrative down into an exact number of scenes. StoryVid works with any OpenAI-compatible API:

| Provider | Preset | Default Model | Base URL |
|---|---|---|---|
| **NVIDIA NIM** (Default) | `nvidia` | `meta/llama-3.2-11b-vision-instruct` | `https://integrate.api.nvidia.com/v1` |
| **OpenAI** | `openai` | `gpt-4o-mini` | `https://api.openai.com/v1` |
| **Groq** | `groq` | `llama-3.3-70b-versatile` | `https://api.groq.com/openai/v1` |
| **OpenRouter** | `openrouter` | `meta-llama/llama-3.3-70b-instruct` | `https://openrouter.ai/api/v1` |
| **Local (Ollama / vLLM)** | `local` | `llama3.2` | `http://localhost:11434/v1` |
| **Custom AI Endpoint** | `custom` | *Your choice* | *Any custom OpenAI-compatible URL* |

---

## Supported Image Generation Providers

| Provider | Key Required? | Description & Models |
|---|---|---|
| **Google Gemini** | Yes (`GEMINI_API_KEY`) | High-fidelity image generation. Default model: `gemini-2.5-flash-image` (or `imagen-3.0-generate-002`). |
| **Pollinations AI** | **No (100% Free)** | Fast, free image generation using Flux. No account or key required. Ideal for quick generation and testing. |
| **OpenAI DALL-E** | Yes (`OPENAI_API_KEY`) | Official OpenAI DALL-E image generation (`dall-e-3` or `dall-e-2`). |
| **Custom AI Endpoint** | Optional | Any OpenAI-compatible `/v1/images/generations` server (e.g. Local Stable Diffusion, Fooocus, vLLM, SwarmUI) returning Base64 (`b64_json`) or image URLs. |

---

## Voice Engines (Local TTS)

Local speech synthesis ensures all audio is generated directly on your machine with zero cloud TTS fees.

| Engine | Quality | Setup | Features |
|---|---|---|---|
| **Kokoro** (Default) | Very high, natural rhythm | Included in `requirements.txt` | Fast on CPU (~300MB weights auto-download to cache on first run). 19+ voices: `af_heart`, `am_adam`, `bf_emma`, `bm_george`, etc. |
| **Chatterbox** | Expressive, dramatic | `pip install chatterbox-tts` (GPU recommended) | Expressiveness slider (`--exaggeration` 0.25–1.5). Voice cloning from a 5–15s WAV sample (`--ref-audio`). |
| **Piper** | Fast, lightweight | `piper-tts` + `.onnx` voice file | Set path with `--voice` or `SV_PIPER_MODEL`. |
| **espeak-ng** | Robotic / testing | `espeak-ng` system package | Instant zero-setup fallback. |

Audition any voice directly in the Web UI using the **"Test voice"** button before generating a video.

---

## Web UI Guide

The Web UI (`storyvid_web.py` + `ui.html`) is a responsive, single-page terminal-styled dashboard:

```bash
python storyvid_web.py --port 8765
```

### Key Features

1. **Job Creation & Parameter Tuning:**
   - Prompt, Scene count (1–24), Aspect ratio (`16:9` widescreen or `9:16` vertical reels/shorts).
   - Narration style (`viral` with punchy hooks or `calm` cinematic).
   - Voice engine selection, voice picker, speech speed multiplier (0.6x–1.6x).
   - Video settings: 1080p HD, Ken Burns motion zoom/pan, Hardcoded subtitle burning.

2. **Interactive Scene Editor:**
   - View every generated scene card with its image and audio as soon as they finish rendering.
   - **Edit narration text**: re-generates only that scene's audio (`↻ voice`).
   - **Edit image prompt**: re-generates only that scene's image (`↻ image`).
   - Re-roll individual assets without regenerating the rest of the project.

3. **Re-rendering & Dirty State Detection:**
   - Whenever an asset is modified, the **RE-RENDER** button activates.
   - Re-rendering takes only seconds because clips and concat demuxers reuse existing media with stream copying (`-c copy`).

4. **In-Browser Video Player:**
   - Native HTML5 video player with HTTP Range requests (smooth seeking).
   - Interactive playback: clicking `▶ play` on any scene card jumps the video directly to that scene's timestamp.
   - Active scene card highlights in sync with the video.
   - Subtitle toggle powered by `subs.vtt`.

5. **Project Management:**
   - Resume interrupted jobs with one click.
   - Browse and load past projects.
   - Delete unwanted projects.

---

## CLI Reference

```bash
python storyvid.py "<prompt>" [options]
```

### Basic Options

| Option | Default | Description |
|---|---|---|
| `-n`, `--scenes` | `8` | Number of scenes to generate (1–24, exact count guaranteed) |
| `--aspect` | `16:9` | Aspect ratio: `16:9` (widescreen) or `9:16` (shorts/TikTok) |
| `--hd` | `off` | 1080p resolution instead of 720p |
| `--lang` | `English` | Language for story generation and narration |
| `--narration` | `viral` | Narration tone: `viral` (hook-heavy) or `calm` (flowing) |
| `--tts` | `auto` | TTS backend: `auto`, `kokoro`, `chatterbox`, `piper`, `espeak` |
| `--voice` | default | Voice identifier (e.g. `af_heart` for Kokoro) |
| `--speed` | `1.0` | Speech speed multiplier (0.6 to 1.6) |
| `--no-kenburns` | `off` | Disable subtle dynamic zoom and pan motion |
| `--burn-subs` | `off` | Burn timed subtitles directly into the video frames |
| `--workers` | `3` | Parallel workers for image generation |
| `--workdir` | auto | Custom project directory path |
| `--force` | `off` | Clear cached images/audio/clips and regenerate |

### AI Provider CLI Flags

| Flag | Default / Env | Description |
|---|---|---|
| `--llm-url` | `SV_LLM_URL` | Base URL for OpenAI-compatible LLM endpoint |
| `--llm-model` | `SV_LLM_MODEL` | Model name for story writing |
| `--llm-key` | `NVIDIA_API_KEY` / `SV_LLM_KEY` | API key for Story LLM |
| `--image-provider` | `gemini` | `gemini`, `pollinations`, `openai`, or `custom` |
| `--image-model` | `SV_IMAGE_MODEL` | Image model name |
| `--image-key` | `GEMINI_API_KEY` / `SV_IMAGE_KEY` | Image provider API key |
| `--image-url` | `SV_IMAGE_URL` | Custom OpenAI-compatible image endpoint URL |

### CLI Examples

**Generate with default NIM + Gemini:**
```bash
python storyvid.py "The secret library beneath the ocean" -n 6
```

**Generate with Pollinations AI (Free, no image key needed):**
```bash
python storyvid.py "A cyberpunk samurai in Neo-Tokyo" --image-provider pollinations -n 8
```

**Generate vertical 9:16 video for YouTube Shorts / TikTok with burned-in subtitles:**
```bash
python storyvid.py "Three bizarre historical facts that sound fake" --aspect 9:16 --burn-subs -n 6
```

**Use a local Ollama model:**
```bash
python storyvid.py "A time traveler stuck in ancient Rome" --llm-url "http://localhost:11434/v1" --llm-model "llama3.2" --image-provider pollinations
```

---

## Configuration Reference

Variables can be placed in `.env` or set in your environment:

| Variable | Default | Description |
|---|---|---|
| `SV_LLM_URL` | `https://integrate.api.nvidia.com/v1` | Story LLM endpoint URL |
| `SV_LLM_MODEL` | `meta/llama-3.2-11b-vision-instruct` | Story LLM model name |
| `NVIDIA_API_KEY` / `SV_LLM_KEY` | *empty* | API key for Story LLM |
| `SV_IMAGE_PROVIDER` | `gemini` | Image provider: `gemini`, `pollinations`, `openai`, `custom` |
| `SV_IMAGE_MODEL` | `gemini-2.5-flash-image` | Image generation model |
| `GEMINI_API_KEY` / `SV_IMAGE_KEY` | *empty* | API key for image generation |
| `SV_IMAGE_URL` | `https://api.openai.com/v1` | Endpoint URL for `openai` or `custom` image provider |
| `SV_OUT` | `out` | Directory for output project folders |
| `SV_CACHE` | `~/.cache/storyvid` | Directory for downloaded TTS model weights |
| `SV_PIPER_MODEL` | *empty* | Path to default Piper `.onnx` voice model |

---

## Output Structure & Caching

Every generated video is stored in its own project directory inside `out/`:

```
out/my-story-ab12cd/
├── project.json      # Complete configuration options used
├── story.json        # Generated story, style, characters, and scenes (editable)
├── images/           # scene_00.png, scene_01.png, ...
├── audio/            # scene_00.wav, scene_01.wav, ... (normalized PCM16)
├── clips/            # scene_00.mp4, ... (per-scene video clips)
├── subs.srt          # Timed subtitles (SubRip)
├── subs.vtt          # Timed subtitles for Web HTML5 player
└── final.mp4         # Final rendered video
```

- **Smart Caching:** Existing `story.json`, images, and audio files are never re-downloaded or re-generated unless explicitly requested.
- **Exact Scene Count Guarantee:** If an LLM returns fewer or more scenes than requested, StoryVid's boundary-aware alignment algorithm splits or merges scenes seamlessly to guarantee the exact frame count you asked for.

---

## Troubleshooting

| Problem | Cause | Solution |
|---|---|---|
| `ResolutionImpossible` / `onnxruntime` has no matching distribution | Using Python 3.13, 3.14, or 3.15 | Use **Python 3.12** or 3.11 (`py -3.12 -m venv .venv`). `onnxruntime` does not yet publish binary wheels for Python 3.13+. |
| `ffmpeg: command not found` | FFmpeg is not installed or not in PATH | Install FFmpeg (`winget install Gyan.FFmpeg` on Windows, `brew install ffmpeg` on macOS, `apt install ffmpeg` on Linux) and restart your terminal. |
| `429 RESOURCE_EXHAUSTED` from Gemini | Free Gemini tier image quotas exhausted | Switch image provider to **Pollinations AI** (free, no quota) in Web UI or `.env`, or enable Pay-As-You-Go on Google Cloud. |
| Model 410 Gone (`meta/llama-3.3-70b-instruct`) | Old NVIDIA model retired | Update to `meta/llama-3.2-11b-vision-instruct` (set automatically in current version). |
| Black placeholder frame generated | Safety filter or temporary provider timeout | Click `↻ image` on that scene card in the Web UI to re-roll with a slightly modified prompt. |
| Captions not burned into video (`--burn-subs`) | FFmpeg build lacks `libass` | Use a full FFmpeg build (e.g. Gyan.FFmpeg full on Windows). The `.srt` and `.vtt` caption files are still produced. |

---

## License

MIT License. Feel free to use, modify, and build upon StoryVid!
