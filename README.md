# STORYVID

Turn one prompt into a finished narrated video made of still images.

```
prompt ─▶ story ─▶ scenes ─▶ images ─▶ local voice ─▶ sync ─▶ render ─▶ final.mp4
         (NIM)     (NIM)    (Gemini)    (on-device)   (FFmpeg)  (FFmpeg)
```

| Stage | Where it runs | Engine |
|---|---|---|
| Story + scene split | cloud | NVIDIA NIM (any OpenAI-compatible model) |
| Images (one per scene) | cloud | Gemini image model, the **only** Gemini usage |
| Voice (TTS) | **local** | Chatterbox, Kokoro, Piper or espeak-ng |
| Sync, motion, captions, render | local | FFmpeg |

There are two ways to run it: a **CLI** (`storyvid.py`) and an **interactive web UI** (`storyvid_web.py` + `ui.html`). Both drive the same pipeline code and share the same project folders.

---

## Contents

1. [Quick start](#quick-start)
2. [Requirements](#requirements)
3. [How the pipeline works](#how-the-pipeline-works)
4. [Voice engines](#voice-engines)
5. [Output layout and caching](#output-layout-and-caching)
6. [CLI reference](#cli-reference)
7. [Web UI](#web-ui)
8. [Configuration (environment variables)](#configuration-environment-variables)
9. [Error handling](#error-handling)
10. [Security notes](#security-notes)
11. [Code map](#code-map)
12. [Known limits](#known-limits)
13. [Troubleshooting](#troubleshooting)

---

## Quick start

```bash
git clone https://github.com/Sami001-OG/storyvid.git
cd storyvid
# Use Python 3.10–3.12 (Python 3.12 or 3.11 recommended)
python -m venv .venv
source .venv/bin/activate              # Linux/macOS
# or: .venv\Scripts\activate           # Windows

pip install -r requirements.txt        # and install ffmpeg (see below)

export NVIDIA_API_KEY=nvapi-...        # story + scene split
export GEMINI_API_KEY=...              # images only

# web UI
python storyvid_web.py                 # open http://127.0.0.1:8765

# or CLI
python storyvid.py "a lighthouse keeper who befriends a storm" -n 8
```

Result: `out/<slug>-<hash>/final.mp4`.

## Requirements

- **Python 3.10–3.12** (`kokoro-onnx` depends on `onnxruntime`, which requires Python <= 3.12; pre-release versions such as Python 3.15 are not supported)
- **FFmpeg** on `PATH` (`apt install ffmpeg`, `brew install ffmpeg`, `pkg install ffmpeg`). For `--burn-subs` it must be built with libass (most builds are).
- **Python packages** (`requirements.txt`): `openai` (talks to NIM), `google-genai` (images), `rich` (terminal UI), `numpy`, `kokoro-onnx` (default local TTS).
- **Optional:** `chatterbox-tts` for the most expressive voice (Python 3.11, GPU strongly recommended), `piper-tts`, or `espeak-ng`.
- **API keys:** an NVIDIA NIM key from build.nvidia.com and a Gemini API key.

The web server uses only the Python standard library, so it adds no extra dependencies.

---

## How the pipeline works

Everything lives in `storyvid.py`. `run_pipeline()` runs five steps in order, and every step is cached on disk so a re-run only does the missing work.

### 1. Story and scene split (NVIDIA NIM)

Two separate model calls, because one call that must invent a story *and* cut it into scenes produces worse results than two focused ones.

**Stage 1, the "bible".** The model returns JSON with `title`, `visual_style`, `characters`, `story`.
- `story` is a complete tale of roughly `scenes × 30` words in the chosen language.
- `visual_style` is one line defining a single art style used for every frame.
- `characters` is a visual description of each recurring character (age, clothes, colours, features).

**Stage 2, the breakdown.** The model receives the bible and splits the story into exactly N consecutive scenes. Each scene has:
- `narration`: 15 to 45 words, in order, no gaps or repeats, polished for reading aloud.
- `image_prompt`: one self-contained still frame in English (subject, action, setting, camera angle, lighting, mood), re-describing every visible character.

**Narration styles** (`--narration`):
- `viral` (default): hook in the first sentence (shocking claim, question or mystery), short punchy sentences, rising tension, a twist, a satisfying last line, natural spoken rhythm.
- `calm`: warm, flowing, cinematic.

Both styles append **TTS rules** to the prompt: spell out numbers and symbols, no abbreviations, no parentheses, no em dashes, no emojis, no markdown. Text-to-speech engines stumble on those, so the story model is told to avoid them.

**Robust parsing.** Models often wrap JSON in markdown fences or add chatter. `extract_json()` strips fences and takes everything from the first `{` to the last `}`, then validates the result with Pydantic. Invalid output triggers a retry (up to 4 attempts).

The result is saved as `story.json`. **If `story.json` already exists it is reused**, so you can hand-edit it and re-run.

### 2. Images (Gemini)

One image per scene. The prompt sent to Gemini is assembled by `frame_prompt()`:

```
<visual_style>
Characters (identical appearance in every frame): <characters>
Scene: <image_prompt>
Aspect ratio <16:9|9:16>. No text, captions, letters, logos or watermarks.
```

Repeating the shared style and character block in every prompt is what keeps frames visually consistent. The aspect ratio is passed through Gemini's image config where the SDK supports it, and FFmpeg crops to exact size later regardless.

- Scenes are generated **in parallel** (`--workers`, default 3).
- Already-generated scenes are skipped (cache).
- On a transient error the call is retried with backoff. If a scene still fails (for example a safety block), a **black placeholder** is written so the video can still render, and the failure is logged. Auth errors abort immediately.
- The Gemini client is only created if at least one image is actually missing, so a fully cached project needs no Gemini key.

### 3. Voice (local TTS)

One WAV per scene, generated on your machine. See [Voice engines](#voice-engines).

- Voice generation runs **sequentially**, because local engines are not safely re-entrant and use the GPU/CPU heavily. A lock inside each engine also protects against concurrent web jobs.
- Each engine's raw output is normalised by FFmpeg to **mono 16-bit PCM WAV**. That makes duration maths and muxing uniform across engines.
- Chatterbox has no native speed control, so the speed setting is applied during normalisation with FFmpeg's `atempo` filter.

### 4. Sync

The key idea: **audio duration drives scene duration.**

For every scene, the real length of its WAV is measured (`wav_dur`). That scene's image is shown for `audio length + 0.5 s` (`PAD`). The half second of silence after each line gives the video breathing room between scenes. There is no guessing and no drift, because each clip is built from exactly the audio it contains.

### 5. Render (FFmpeg)

**Per-scene clip** (`build_clip`):
- The image is scaled and centre-cropped to fill the target size (1280×720 by default, 1920×1080 with `--hd`, swapped for 9:16).
- **Ken Burns motion** (default): the image is rendered at 2× size and `zoompan` slowly zooms 20%. Even-numbered scenes zoom in and odd-numbered scenes zoom out, so consecutive shots feel different. Rendering at 2× avoids the jitter `zoompan` produces on small sources. `--no-kenburns` uses a plain static frame.
- Audio is padded with silence (`apad`), resampled to 48 kHz and made stereo, then encoded as AAC. Video is H.264 (`libx264`, CRF 20, 30 fps, `yuv420p`, `faststart`).
- Clips are built in parallel across CPU cores.

**Join:** all clips share identical codec parameters, so FFmpeg's concat demuxer joins them with **stream copy** (`-c copy`), with no re-encode and no quality loss.

**Captions:** `subs.srt` and `subs.vtt` are always written. Narration is split into sentences, then into chunks of at most 8 words (balanced within a sentence). Each chunk's time slice is proportional to its character count within the scene's audio duration. These are **estimated timings, not forced alignment**, which is accurate enough for short scenes. With `--burn-subs` the SRT is hard-coded into the picture through FFmpeg's `subtitles` filter (needs libass; if missing, the burn step is skipped with a warning and the sidecar files are still written).

The final file is `final.mp4`.

---

## Voice engines

Select with `--tts` (CLI) or the **Voice engine** dropdown (web). `auto` picks the best available:

1. **Chatterbox** if installed **and** a GPU (CUDA or Apple MPS) is present.
2. **Kokoro** if installed.
3. **Piper** if installed and `SV_PIPER_MODEL` is set.
4. **espeak-ng** as a last resort.

| Engine | Quality | Needs | Notes |
|---|---|---|---|
| `chatterbox` | Most expressive, attention-grabbing | `pip install chatterbox-tts`, GPU | **Expressiveness** (`--exaggeration`, 0.25–1.5) controls dramatic delivery. `--ref-audio clip.wav` clones a voice from a 5–15 s sample. At high expressiveness the CFG weight is lowered automatically to keep pacing natural. |
| `kokoro` | Very good, fast, runs on CPU | `pip install kokoro-onnx` | Voices such as `af_heart`, `am_adam`, `bf_emma`, `bm_george`. The ~300 MB model files download automatically on first use to `~/.cache/storyvid`. `--tts-lang` sets the phonemiser language (`en-us`, `en-gb`, ...). |
| `piper` | Good, lightweight | `piper-tts` plus a voice `.onnx` | Pass the model path as `--voice` or set `SV_PIPER_MODEL`. |
| `espeak` | Robotic, zero setup | `espeak-ng` | Fallback and quick testing. |

Engines are loaded once per process and cached, so the web UI does not reload multi-gigabyte weights for every scene.

---

## Output layout and caching

Each project is one folder (default `out/<prompt-slug>-<hash>/`; the web UI adds a timestamp):

```
out/my-story-ab12cd/
├── project.json      all options used (shared by CLI and web UI)
├── story.json        title, style, characters, scenes (editable)
├── images/           scene_00.png ...   (Gemini output)
├── audio/            scene_00.wav ...   (local TTS, mono PCM16)
├── clips/            scene_00.mp4 ...   (per-scene video, rebuilt every render)
├── subs.srt          captions
├── subs.vtt          captions for the browser player
└── final.mp4         the result
```

**Resume and cache rules**
- CLI: the folder name is a hash of prompt + scene count + aspect + language, so re-running the **same command** resumes automatically. Change any of those and you get a new project.
- Existing `story.json`, images and WAVs are never regenerated.
- To redo something, delete it (for example `images/scene_03.png`) or pass `--force` (clears images, audio and clips).
- `--story path.json` supplies a story file instead of generating one.
- Clips are always rebuilt on render (cheap), so edits are never stale.

---

## CLI reference

```
python storyvid.py "<prompt>" [options]
```

| Option | Default | Meaning |
|---|---|---|
| `-n`, `--scenes` | 8 | Number of scenes (1–24) |
| `--aspect` | 16:9 | `16:9` or `9:16` (shorts/reels) |
| `--hd` | off | 1080p instead of 720p |
| `--lang` | English | Narration language (for the story model) |
| `--narration` | viral | `viral` or `calm` |
| `--tts` | auto | `auto`, `chatterbox`, `kokoro`, `piper`, `espeak` |
| `--voice` | engine default | Kokoro voice, Piper `.onnx` path, or espeak voice |
| `--tts-lang` | en-us | Phonemiser language (Kokoro/espeak) |
| `--speed` | 1.0 | Speech speed multiplier (0.6–1.6) |
| `--exaggeration` | 0.6 | Chatterbox expressiveness |
| `--ref-audio` | none | Chatterbox voice-clone reference clip |
| `--workers` | 3 | Parallel Gemini image calls (lower if rate-limited) |
| `--no-kenburns` | off | Disable zoom/pan |
| `--burn-subs` | off | Burn captions into the video |
| `--workdir` | auto | Output folder |
| `--story` | none | Use an existing `story.json` |
| `--force` | off | Discard cached images/audio/clips |

Terminal progress is drawn by `rich`: one bar per stage (STORY, IMAGES, VOICE, CLIPS, RENDER).

---

## Web UI

```bash
python storyvid_web.py [--host 127.0.0.1] [--port 8765]
```

A single-page, terminal-styled interface (`ui.html`) served by a stdlib HTTP server (`storyvid_web.py`). It works on phones.

### What you can do

- **Create a job:** prompt, scenes, aspect, narration style, language, voice engine, voice, speed, expressiveness, reference voice, 1080p, Ken Burns, burn subs.
- **Test voice:** synthesises a sample line with the current voice settings so you can audition before committing.
- **Watch it run:** live progress bars and a log stream. Scene images and audio appear as soon as each is ready.
- **Edit any scene afterwards.** Change the narration (only that voice is regenerated) or the image prompt (only that image is regenerated). `↻ image` and `↻ voice` re-roll a scene without changing its text.
- **Re-render:** edited projects highlight the RE-RENDER button. Rendering rebuilds the clips from the current assets.
- **Player:** an in-page video player with captions from `subs.vtt`. `▶ play` on a scene card jumps the video to that scene, and the card for the currently playing scene lights up.
- **Manage projects:** reopen past projects (shared with the CLI), resume a job that stopped part-way, delete a project.

### How it works

- The pipeline functions take a `Reporter` object. The CLI passes a `rich`-based reporter; the web server passes `WebReporter`, which turns progress into JSON events.
- Each project keeps an in-memory event list. The browser connects with **Server-Sent Events** and receives them live; reconnecting resumes from the last event ID. Event types: `busy`, `stage`, `tick`, `finish`, `log`, `story`, `asset`, `video`, `done`, `error`.
- Only **one job runs per project at a time**; a second action returns HTTP 409.
- On `asset`/`story`/`video`/`done` events the page re-fetches project state and updates cards in place. Text boxes you are editing are not overwritten.
- A project is "dirty" (needs re-render) when any asset or `story.json` is newer than `final.mp4`.
- Videos are served with HTTP **Range** support so browsers can seek.

### HTTP API

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | the UI |
| GET | `/api/status` | key presence, available TTS engines, models |
| GET | `/api/projects` | list projects |
| POST | `/api/projects` | create and start a job (JSON options) |
| GET | `/api/projects/<id>` | full state: story, scenes, asset URLs, video, dirty flag |
| GET | `/api/projects/<id>/events` | SSE stream |
| POST | `/api/projects/<id>/resume` | continue an incomplete project |
| POST | `/api/projects/<id>/render` | re-render (`kenburns`, `burn_subs` optional) |
| POST | `/api/projects/<id>/scenes/<i>` | edit a scene (`narration`, `image_prompt`, `regen: ["image","voice"]`) |
| DELETE | `/api/projects/<id>` | delete a project |
| POST | `/api/preview` | synthesise a voice sample (returns WAV) |
| GET | `/files/<id>/<path>` | project files (Range supported) |

---

## Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `NVIDIA_API_KEY` (or `SV_LLM_KEY`) | none | Key for the story model |
| `SV_LLM_URL` | `https://integrate.api.nvidia.com/v1` | Any OpenAI-compatible endpoint |
| `SV_LLM_MODEL` | `meta/llama-3.3-70b-instruct` | Story/scene model |
| `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | none | Key for image generation |
| `SV_IMAGE_MODEL` | `gemini-2.5-flash-image` | Gemini image model |
| `SV_OUT` | `out` | Projects folder |
| `SV_CACHE` | `~/.cache/storyvid` | Downloaded model files (Kokoro) |
| `SV_PIPER_MODEL` | none | Default Piper voice `.onnx` |

Keys are read only when needed: a fully cached project (story, images) needs neither key.

---

## Error handling

- **Retries with backoff** on every network call. Delays double up to 60 s, and if the provider sends a retry-after hint (rate limits) that value is used. Authentication and permission errors are **not** retried.
- **Failed image** after retries becomes a black placeholder (logged) instead of killing the run.
- **Malformed story JSON** is retried, then reported.
- **Everything is resumable.** After any failure, re-run the same command or click **resume**; completed work is not repeated.
- User-facing problems (missing key, missing FFmpeg, bad option, missing reference audio) raise a clear `PipelineError` message in the CLI and appear in the UI log.

---

## Security notes

- The web server binds to **localhost** by default and checks the `Host` header, which blocks DNS-rebinding attacks.
- Every state-changing request must carry a custom `X-SV: 1` header, which a malicious web page cannot send cross-origin without a CORS preflight (the server never allows one). This stops other sites from triggering paid API calls through your local server.
- File serving resolves the real path and refuses anything outside the project folder.
- UI text from the model is inserted with `textContent`, never as HTML.
- `--host 0.0.0.0` exposes the server with **no authentication**. Use it only on a network you trust.
- API keys are read from environment variables; the UI never accepts or stores keys.
- Never commit keys. `.env` and `out/` are in `.gitignore`.

---

## Code map

| File | Role |
|---|---|
| `storyvid.py` | The whole pipeline plus the CLI: options (`Opts`), story (`make_story`, `step_story`), images (`make_image`, `step_images`), voice (`TTS`, `step_voice`), subtitles (`build_cues`, `write_subs`), render (`build_clip`, `concat`, `step_render`), `run_pipeline`, reporters |
| `storyvid_web.py` | Stdlib HTTP server: project registry, SSE events, job threads, scene editing, Range file serving, preview |
| `ui.html` | The single-file web interface (HTML, CSS, vanilla JS) |
| `requirements.txt` | Python dependencies |

---

## Known limits

- Caption timing is estimated from character counts, not aligned to the audio word by word.
- Frames are independent generations, so character consistency relies on the repeated style/character prompt and is good but not perfect.
- Chatterbox on CPU is slow, so `auto` avoids it without a GPU.
- Model names (`gemini-2.5-flash-image`, `meta/llama-3.3-70b-instruct`) change over time; override them with the environment variables above if a provider renames or retires one.
- Testing status: the full web flow, sync, render, captions, Range requests and scene editing were verified end to end with the story and image calls mocked and real local TTS (Kokoro and espeak) and real FFmpeg. The live NIM and Gemini calls and the Chatterbox integration were written against their documented APIs but not exercised live in the test environment.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ResolutionImpossible` / `onnxruntime` has no matching distribution | You are using an unsupported Python version (e.g. Python 3.15 alpha); use Python 3.11 or 3.12 |
| `ffmpeg not found` | Install FFmpeg and make sure it is on `PATH` |
| `NVIDIA_API_KEY is not set` / `GEMINI_API_KEY is not set` | Export the key in the shell that launches the app |
| 429 / rate limit from Gemini | Lower `--workers` to 1 or 2; retries already honor the provider's delay |
| A frame came out black | Gemini refused or failed that prompt; edit the scene's image prompt and click `↻ image` |
| Kokoro download fails | Download `kokoro-v1.0.onnx` and `voices-v1.0.bin` manually into `~/.cache/storyvid` |
| No audio engine found | `pip install kokoro-onnx` or install `espeak-ng` |
| Captions not burned in | Your FFmpeg lacks libass; use the `.srt`/`.vtt` files or install a full FFmpeg build |
| Video does not seek in the browser | Make sure you are using `storyvid_web.py`, which supports Range requests |
