#!/usr/bin/env python3
"""
storyvid — prompt -> story -> scenes -> images -> local TTS -> synced video.

  story + scene split : NVIDIA NIM (OpenAI-compatible endpoint, any model)
  images              : Gemini (the ONLY Gemini usage)
  voice               : local (chatterbox | kokoro | piper | espeak-ng)
  sync + render       : FFmpeg, audio length drives scene length

CLI:
    export NVIDIA_API_KEY=nvapi-...  GEMINI_API_KEY=...
    python storyvid.py "a lighthouse keeper who befriends a storm" -n 8

Web UI:  python storyvid_web.py   (same pipeline, interactive)

Re-running the same command resumes from cache (story, images, audio).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from pydantic import BaseModel, field_validator
from rich.console import Console
from rich.progress import (BarColumn, MofNCompleteColumn, Progress,
                           SpinnerColumn, TextColumn, TimeElapsedColumn)

LLM_URL = os.getenv("SV_LLM_URL", "https://integrate.api.nvidia.com/v1")
LLM_MODEL = os.getenv("SV_LLM_MODEL", "meta/llama-3.2-11b-vision-instruct")
IMAGE_PROVIDER = os.getenv("SV_IMAGE_PROVIDER", "gemini")
IMAGE_MODEL = os.getenv("SV_IMAGE_MODEL", "gemini-2.5-flash-image")
IMAGE_URL = os.getenv("SV_IMAGE_URL", "https://api.openai.com/v1")
OUT = Path(os.getenv("SV_OUT", "out"))
CACHE = Path(os.getenv("SV_CACHE", Path.home() / ".cache" / "storyvid"))

def load_env():
    global LLM_MODEL, LLM_URL, IMAGE_PROVIDER, IMAGE_MODEL, IMAGE_URL, OUT, CACHE
    env_file = Path(__file__).resolve().parent / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip("'\"")
            if k and v:
                os.environ[k] = v
    LLM_URL = os.getenv("SV_LLM_URL", "https://integrate.api.nvidia.com/v1")
    LLM_MODEL = os.getenv("SV_LLM_MODEL", "meta/llama-3.2-11b-vision-instruct")
    IMAGE_PROVIDER = os.getenv("SV_IMAGE_PROVIDER", "gemini")
    IMAGE_MODEL = os.getenv("SV_IMAGE_MODEL", "gemini-2.5-flash-image")
    IMAGE_URL = os.getenv("SV_IMAGE_URL", "https://api.openai.com/v1")
    OUT = Path(os.getenv("SV_OUT", "out"))
    CACHE = Path(os.getenv("SV_CACHE", Path.home() / ".cache" / "storyvid"))

load_env()

def update_env(updates: dict[str, str]) -> None:
    """Update or append variables in .env file and reload runtime environment."""
    env_file = Path(__file__).resolve().parent / ".env"
    lines = []
    existing = {}
    if env_file.is_file():
        lines = env_file.read_text(encoding="utf-8").splitlines()
        for idx, line in enumerate(lines):
            line_str = line.strip()
            if line_str and not line_str.startswith("#") and "=" in line_str:
                k, _ = line_str.split("=", 1)
                existing[k.strip()] = idx

    for k, v in updates.items():
        if not k:
            continue
        v_str = str(v or "").strip()
        os.environ[k] = v_str
        new_line = f"{k}={v_str}"
        if k in existing:
            lines[existing[k]] = new_line
        else:
            lines.append(new_line)
            existing[k] = len(lines) - 1

    env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    load_env()

FPS = 30
PAD = 0.5  # seconds of silence after each scene's narration
KOKORO_BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
KOKORO_VOICES = ["af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky", "af_nova", "af_jessica",
                 "am_adam", "am_michael", "am_eric", "am_onyx", "am_puck", "am_fenrir",
                 "bf_emma", "bf_isabella", "bf_lily", "bm_george", "bm_lewis", "bm_fable"]
FATAL_STR = ("API_KEY_INVALID", "API key not valid", "PERMISSION_DENIED", "UNAUTHENTICATED")
FATAL_TYPE = ("AuthenticationError", "PermissionDeniedError")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

con = Console(legacy_windows=False)


class PipelineError(Exception):
    """User-facing failure (bad key, missing tool, bad options...)."""


# ───────────────────────── reporting ─────────────────────────
class Reporter:
    """No-op base. CLI and web UI subclass it."""

    def stage(self, name: str, total: int = 0) -> None: ...
    def tick(self, name: str) -> None: ...
    def finish(self, name: str) -> None: ...
    def log(self, msg: str, level: str = "info") -> None: ...
    def event(self, etype: str, /, **data) -> None: ...
    def close(self) -> None: ...


class CLIReporter(Reporter):
    def __init__(self) -> None:
        self.pg = Progress(SpinnerColumn(style="green"), TextColumn("[green]{task.description}"),
                           BarColumn(complete_style="green", finished_style="bright_green"),
                           MofNCompleteColumn(), TimeElapsedColumn(), console=con)
        self.tasks: dict[str, tuple[int, int]] = {}
        self.pg.start()

    def stage(self, name, total=0):
        self.tasks[name] = (self.pg.add_task(name.ljust(8), total=total or None), total or 1)

    def tick(self, name):
        self.pg.advance(self.tasks[name][0])

    def finish(self, name):
        tid, tot = self.tasks[name]
        self.pg.update(tid, total=tot, completed=tot)

    def log(self, msg, level="info"):
        color = {"warn": "yellow", "err": "red", "ok": "green"}.get(level, "dim")
        self.pg.console.print(f"[{color}]{msg}[/]")

    def close(self):
        self.pg.stop()


# ───────────────────────── options / models ─────────────────────────
@dataclass
class Opts:
    prompt: str = ""
    scenes: int = 8
    aspect: str = "16:9"
    hd: bool = False
    lang: str = "English"
    narration: str = "viral"        # viral | calm
    tts: str = "auto"               # auto | chatterbox | kokoro | piper | espeak
    voice: str | None = None
    tts_lang: str = "en-us"
    speed: float = 1.0
    exaggeration: float = 0.6       # chatterbox expressiveness
    ref_audio: str | None = None    # chatterbox voice-clone reference
    workers: int = 3
    kenburns: bool = True
    burn_subs: bool = False
    # LLM (Story) integration
    llm_url: str | None = None
    llm_model: str | None = None
    llm_key: str | None = None
    # Image integration
    image_provider: str = "gemini"  # gemini | pollinations | openai
    image_model: str | None = None
    image_key: str | None = None
    image_url: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "Opts":
        names = {f.name for f in fields(cls)}
        o = cls(**{k: v for k, v in d.items() if k in names})
        o.validate()
        return o

    def validate(self) -> None:
        def num(v, lo, hi, cast):
            try:
                return cast(min(max(cast(v), lo), hi))
            except (TypeError, ValueError):
                raise PipelineError(f"bad numeric option: {v!r}")
        self.prompt = (self.prompt or "").strip()
        if not self.prompt:
            raise PipelineError("empty prompt")
        if len(self.prompt) > 4000:
            raise PipelineError("prompt too long (max 4000 chars)")
        self.scenes = num(self.scenes, 1, 24, int)
        self.speed = num(self.speed, 0.6, 1.6, float)
        self.exaggeration = num(self.exaggeration, 0.0, 2.0, float)
        self.workers = num(self.workers, 1, 8, int)
        if self.aspect not in ("16:9", "9:16"):
            raise PipelineError("aspect must be 16:9 or 9:16")
        if self.narration not in NARRATION:
            raise PipelineError("narration must be viral or calm")
        if self.tts not in ("auto", "chatterbox", "kokoro", "piper", "espeak"):
            raise PipelineError(f"unknown tts backend: {self.tts}")
        self.lang = (self.lang or "English").strip()[:40]
        self.tts_lang = (self.tts_lang or "en-us").strip()[:16]
        self.voice = (self.voice or "").strip() or None
        self.ref_audio = (self.ref_audio or "").strip() or None
        self.hd, self.kenburns, self.burn_subs = bool(self.hd), bool(self.kenburns), bool(self.burn_subs)
        self.image_provider = (self.image_provider or IMAGE_PROVIDER).lower().strip()
        if self.image_provider not in ("gemini", "pollinations", "openai", "custom"):
            self.image_provider = "gemini"
        self.llm_url = (self.llm_url or "").strip() or None
        self.llm_model = (self.llm_model or "").strip() or None
        self.llm_key = (self.llm_key or "").strip() or None
        self.image_model = (self.image_model or "").strip() or None
        self.image_key = (self.image_key or "").strip() or None
        self.image_url = (self.image_url or "").strip() or None

    @property
    def size(self) -> tuple[int, int]:
        long_side, short_side = (1920, 1080) if self.hd else (1280, 720)
        return (long_side, short_side) if self.aspect == "16:9" else (short_side, long_side)


class Scene(BaseModel):
    narration: str
    image_prompt: str


class Story(BaseModel):
    title: str
    visual_style: str
    characters: str
    scenes: list[Scene]

    @field_validator("characters", mode="before")
    @classmethod
    def normalize_characters(cls, v):
        if isinstance(v, dict):
            return "; ".join(f"{k}: {val}" for k, val in v.items())
        if isinstance(v, list):
            return "; ".join(str(x) for x in v)
        return str(v)


class Bible(BaseModel):
    title: str
    visual_style: str
    characters: str
    story: str

    @field_validator("characters", mode="before")
    @classmethod
    def normalize_characters(cls, v):
        if isinstance(v, dict):
            return "; ".join(f"{k}: {val}" for k, val in v.items())
        if isinstance(v, list):
            return "; ".join(str(x) for x in v)
        return str(v)


class Breakdown(BaseModel):
    scenes: list[Scene]


def save_cfg(o: Opts, wd: Path) -> None:
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "project.json").write_text(json.dumps(asdict(o), indent=2), encoding="utf-8")


def load_cfg(wd: Path) -> Opts:
    p = wd / "project.json"
    if not p.exists():
        raise PipelineError("project.json missing")
    return Opts.from_dict(json.loads(p.read_text(encoding="utf-8")))


# ───────────────────────── utils ─────────────────────────
def retry(fn, label: str, rep: Reporter, tries: int = 6):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            fatal = type(e).__name__ in FATAL_TYPE or any(s in msg for s in FATAL_STR)
            if i == tries - 1 or fatal:
                raise
            m = re.search(r"(?:retry in |retryDelay['\"]?:\s*['\"]?)([\d.]+)s", msg, re.I)
            wait = min(90.0, float(m.group(1)) + 1) if m else min(60.0, 2.0 ** (i + 1))
            rep.log(f"{label}: {type(e).__name__}, retry {i + 1}/{tries - 1} in {wait:.0f}s", "warn")
            time.sleep(wait)


def ffmpeg(args: list[str], cwd: Path | None = None) -> None:
    p = subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args],
                       cwd=cwd, capture_output=True, text=True)
    if p.returncode:
        raise PipelineError(f"ffmpeg failed:\n{p.stderr.strip()}")


def run_pool(label: str, items: list, fn, workers: int, rep: Reporter) -> dict:
    out = {}
    rep.stage(label, len(items))
    with ThreadPoolExecutor(max(1, workers)) as ex:
        futs = {ex.submit(fn, i): i for i in items}
        for f in as_completed(futs):
            out[futs[f]] = f.result()
            rep.tick(label)
    return out


def wav_dur(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def find_asset(d: Path, stem: str) -> Path | None:
    return next(iter(sorted(d.glob(f"{stem}.*"))), None)


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


# ───────────────────────── 1. story (NVIDIA NIM) ─────────────────────────
NARRATION = {
    "viral": ("Open with a scroll-stopping hook in the very first sentence: a shocking claim, a "
              "question, or a mystery. Use short punchy sentences, vivid sensory detail, rising "
              "tension, a twist, and a satisfying last line. Natural spoken rhythm with contractions."),
    "calm": "Warm, flowing, cinematic storytelling with a gentle emotional arc.",
}
TTS_RULES = ("Write for text-to-speech: spell out numbers and symbols, no abbreviations, no "
             "parentheses, no em dashes, no emojis, no markdown.")


def extract_json(text: str) -> dict:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I).strip()
    a, b = clean.find("{"), clean.rfind("}")
    if a >= 0 and b > a:
        try:
            return json.loads(clean[a:b + 1])
        except Exception:
            pass
    for kw in ('"scenes"', '"title"'):
        idx = text.find(kw)
        if idx >= 0:
            brace = text.rfind("{", 0, idx)
            b = text.rfind("}")
            if brace >= 0 and b > brace:
                try:
                    return json.loads(text[brace:b + 1])
                except Exception:
                    pass
    raise ValueError("no JSON object in model output")


def llm_json(client, system: str, user: str, cls, label: str, rep: Reporter, model: str | None = None, temperature: float = 0.4):
    m = model or LLM_MODEL
    def call():
        r = client.chat.completions.create(
            model=m, temperature=temperature, max_tokens=4096,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        msg = r.choices[0].message
        raw = msg.content or getattr(msg, "reasoning_content", "") or ""
        return cls.model_validate(extract_json(raw))
    return retry(call, label, rep, tries=4)


def adjust_scenes(scenes: list[Scene], n: int, style: str, chars: str) -> list[Scene]:
    """Ensure the scene list has exactly n scenes, expanding or trimming as needed."""
    if not scenes:
        raise ValueError("model returned zero scenes")
    if len(scenes) == n:
        return scenes

    if len(scenes) > n:
        trimmed = list(scenes[:n - 1])
        tail = scenes[n - 1:]
        merged_narration = " ".join(s.narration.strip() for s in tail if s.narration.strip())
        final_prompt = tail[-1].image_prompt
        trimmed.append(Scene(narration=merged_narration, image_prompt=final_prompt))
        return trimmed

    current = list(scenes)
    while len(current) < n:
        best_idx = max(range(len(current)), key=lambda i: len(current[i].narration.split()))
        target = current[best_idx]
        words = target.narration.split()
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", target.narration) if s.strip()]
        if len(sentences) >= 2:
            mid = len(sentences) // 2
            part1 = " ".join(sentences[:mid])
            part2 = " ".join(sentences[mid:])
        elif len(words) >= 4:
            mid = len(words) // 2
            part1 = " ".join(words[:mid])
            part2 = " ".join(words[mid:])
        else:
            part1 = target.narration
            part2 = target.narration

        p1 = target.image_prompt
        p2 = f"{style}. Characters: {chars}. Scene continues: {target.image_prompt}. Alternative dynamic angle."
        current[best_idx] = Scene(narration=part1, image_prompt=p1)
        current.insert(best_idx + 1, Scene(narration=part2, image_prompt=p2))
    return current


def make_story(client, o: Opts, rep: Reporter, model: str | None = None) -> Story:
    n, lang, style = o.scenes, o.lang, NARRATION[o.narration]
    m = model or o.llm_model or LLM_MODEL
    bible = llm_json(
        client,
        "You are a master storyteller for short narrated videos. Respond with ONLY a JSON object, "
        'no markdown, with keys: "title", "visual_style", "characters", "story". '
        f'"story": a complete tale with beginning, middle and end, about {n * 30} words, written in '
        f"{lang}, meant to be read aloud, no headings. {style} {TTS_RULES} "
        '"visual_style": ONE line defining a single consistent art style for every frame. '
        '"characters": concise visual descriptions of each recurring character '
        "(age, clothing, colors, distinctive features).",
        o.prompt, Bible, "story", rep, model=m)

    bd = llm_json(
        client,
        f"You split stories into EXACTLY {n} consecutive scenes for a video of still images. "
        f'CRITICAL: The output JSON "scenes" array MUST contain EXACTLY {n} items, no more and no less. '
        'Respond with ONLY a JSON object: {"scenes":[{"narration":"...","image_prompt":"..."}]}. '
        f"narration: that scene's portion of the story in {lang}, 15-45 words, in order, no gaps or "
        f"repeats across scenes, polished for reading aloud. {style} {TTS_RULES} "
        "image_prompt: ONE self-contained still frame in English: subject, action, setting, camera "
        "angle, lighting, mood. Re-describe every visible character's full appearance in each "
        "image_prompt. Never ask for text, captions or logos in the image.",
        f"STYLE: {bible.visual_style}\nCHARACTERS: {bible.characters}\n\nSTORY:\n{bible.story}",
        Breakdown, "scenes", rep, model=m, temperature=0.4)
    if not bd.scenes:
        raise ValueError("model returned zero scenes")
    scenes = bd.scenes
    if len(scenes) != n:
        rep.log(f"model returned {len(scenes)} scenes (asked {n}); adjusting to exactly {n}", "info")
        scenes = adjust_scenes(scenes, n, bible.visual_style, bible.characters)
    return Story(title=bible.title, visual_style=bible.visual_style,
                 characters=bible.characters, scenes=scenes)


def step_story(o: Opts, wd: Path, rep: Reporter, story_file: str | None = None) -> Story:
    sp = Path(story_file) if story_file else wd / "story.json"
    if sp.exists():
        story = Story.model_validate_json(sp.read_text(encoding="utf-8"))
        rep.log(f"story loaded from {sp}")
    else:
        url = o.llm_url or LLM_URL
        model = o.llm_model or LLM_MODEL
        key = o.llm_key or os.getenv("NVIDIA_API_KEY") or os.getenv("SV_LLM_KEY") or os.getenv("OPENAI_API_KEY")
        if not key:
            raise PipelineError("Story LLM API key is not set (configure in Web UI or set NVIDIA_API_KEY)")
        try:
            from openai import OpenAI
        except ImportError:
            raise PipelineError("pip install openai")
        client = OpenAI(base_url=url, api_key=key, timeout=120)
        rep.stage("STORY", 1)
        rep.log(f"writing story with {model}")
        story = make_story(client, o, rep, model=model)
        rep.finish("STORY")
    (wd / "story.json").write_text(story.model_dump_json(indent=2), encoding="utf-8")
    rep.event("story")
    return story


# ───────────────────────── 2. images (Gemini / Pollinations / OpenAI) ─────────────────────────
def frame_prompt(story: Story, scene: Scene, aspect: str) -> str:
    return (f"{story.visual_style}\n"
            f"Characters (identical appearance in every frame): {story.characters}\n"
            f"Scene: {scene.image_prompt}\n"
            f"Aspect ratio {aspect}. No text, captions, letters, logos or watermarks.")


def make_image_gemini(client, prompt: str, aspect: str, wd: Path, idx: int, size: tuple[int, int],
                      rep: Reporter, o: Opts) -> Path:
    from google.genai import types
    stem = f"scene_{idx:02d}"
    cached = find_asset(wd / "images", stem)
    if cached:
        return cached
    model = o.image_model or IMAGE_MODEL
    kw: dict = {"response_modalities": ["TEXT", "IMAGE"]}
    if hasattr(types, "ImageConfig"):
        kw["image_config"] = types.ImageConfig(aspect_ratio=aspect)
    cfg = types.GenerateContentConfig(**kw)

    def call():
        r = client.models.generate_content(model=model, contents=prompt, config=cfg)
        for c in r.candidates or []:
            for p in (c.content.parts if c.content and c.content.parts else []):
                if p.inline_data and p.inline_data.data:
                    return p.inline_data.data, p.inline_data.mime_type or "image/png"
        raise RuntimeError("no image returned (likely safety block)")

    try:
        data, mime = retry(call, f"image {idx + 1}", rep, tries=4)
        ext = {"image/jpeg": ".jpg", "image/webp": ".webp"}.get(mime, ".png")
        path = wd / "images" / f"{stem}{ext}"
        path.write_bytes(data)
        return path
    except Exception as e:  # noqa: BLE001
        if type(e).__name__ in FATAL_TYPE or any(s in str(e) for s in FATAL_STR):
            raise
        rep.log(f"image {idx + 1} gemini failed ({e}); using black placeholder", "err")
        path = wd / "images" / f"{stem}.png"
        ffmpeg(["-f", "lavfi", "-i", f"color=c=black:s={size[0]}x{size[1]}", "-frames:v", "1", str(path)])
        return path


def make_image_pollinations(prompt: str, aspect: str, wd: Path, idx: int, size: tuple[int, int],
                            rep: Reporter, o: Opts) -> Path:
    stem = f"scene_{idx:02d}"
    cached = find_asset(wd / "images", stem)
    if cached:
        return cached
    clean_p = prompt[:600].replace("\n", " ").strip()
    encoded = urllib.parse.quote(clean_p)
    model = o.image_model or "flux"
    url = f"https://image.pollinations.ai/prompt/{encoded}?nologo=true&seed={idx + 42}&model={model}"

    def call():
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 storyvid"})
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = resp.read()
            if not data or len(data) < 500:
                raise RuntimeError("invalid image from pollinations")
            ctype = resp.headers.get_content_type()
            ext = ".jpg" if "jpeg" in ctype else ".png"
            return data, ext

    try:
        data, ext = retry(call, f"image {idx + 1}", rep, tries=3)
        path = wd / "images" / f"{stem}{ext}"
        path.write_bytes(data)
        return path
    except Exception as e:  # noqa: BLE001
        rep.log(f"image {idx + 1} pollinations failed ({e}); using black placeholder", "err")
        path = wd / "images" / f"{stem}.png"
        ffmpeg(["-f", "lavfi", "-i", f"color=c=black:s={size[0]}x{size[1]}", "-frames:v", "1", str(path)])
        return path


def make_image_openai(client, prompt: str, aspect: str, wd: Path, idx: int, size: tuple[int, int],
                      rep: Reporter, o: Opts) -> Path:
    stem = f"scene_{idx:02d}"
    cached = find_asset(wd / "images", stem)
    if cached:
        return cached
    model = o.image_model or "dall-e-3"
    sz = "1792x1024" if aspect == "16:9" else "1024x1792"

    def call():
        import base64
        r = None
        try:
            r = client.images.generate(model=model, prompt=prompt[:1000], size=sz, response_format="b64_json")
        except Exception:
            try:
                r = client.images.generate(model=model, prompt=prompt[:1000], response_format="b64_json")
            except Exception:
                r = client.images.generate(model=model, prompt=prompt[:1000])

        if r and getattr(r, "data", None) and len(r.data) > 0:
            item = r.data[0]
            if getattr(item, "b64_json", None):
                return base64.b64decode(item.b64_json), ".png"
            if getattr(item, "url", None):
                req = urllib.request.Request(item.url, headers={"User-Agent": "Mozilla/5.0 storyvid"})
                with urllib.request.urlopen(req, timeout=45) as resp:
                    data = resp.read()
                    ctype = resp.headers.get_content_type()
                    ext = ".jpg" if "jpeg" in ctype else ".png"
                    return data, ext
        raise RuntimeError("no image data in API response")

    try:
        data, ext = retry(call, f"image {idx + 1}", rep, tries=3)
        path = wd / "images" / f"{stem}{ext}"
        path.write_bytes(data)
        return path
    except Exception as e:  # noqa: BLE001
        rep.log(f"image {idx + 1} ({o.image_provider}) failed ({e}); using black placeholder", "err")
        path = wd / "images" / f"{stem}.png"
        ffmpeg(["-f", "lavfi", "-i", f"color=c=black:s={size[0]}x{size[1]}", "-frames:v", "1", str(path)])
        return path


def make_image(client, prompt: str, aspect: str, wd: Path, idx: int, size: tuple[int, int],
               rep: Reporter, o: Opts) -> Path:
    provider = (o.image_provider or IMAGE_PROVIDER).lower()
    if provider == "pollinations":
        return make_image_pollinations(prompt, aspect, wd, idx, size, rep, o)
    if provider in ("openai", "custom"):
        return make_image_openai(client, prompt, aspect, wd, idx, size, rep, o)
    return make_image_gemini(client, prompt, aspect, wd, idx, size, rep, o)


def step_images(story: Story, o: Opts, wd: Path, rep: Reporter, only: list[int] | None = None) -> dict:
    idxs = only if only is not None else list(range(len(story.scenes)))
    (wd / "images").mkdir(parents=True, exist_ok=True)
    provider = (o.image_provider or IMAGE_PROVIDER).lower()
    client = None
    if any(find_asset(wd / "images", f"scene_{i:02d}") is None for i in idxs):
        if provider == "gemini":
            key = o.image_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not key:
                raise PipelineError("GEMINI_API_KEY is not set (configure in Web UI or .env)")
            try:
                from google import genai
            except ImportError:
                raise PipelineError("pip install google-genai")
            client = genai.Client(api_key=key)
        elif provider in ("openai", "custom"):
            key = o.image_key or os.getenv("SV_IMAGE_KEY") or os.getenv("OPENAI_API_KEY") or "custom-key"
            base_url = o.image_url or os.getenv("SV_IMAGE_URL") or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            try:
                from openai import OpenAI
            except ImportError:
                raise PipelineError("pip install openai")
            client = OpenAI(api_key=key, base_url=base_url)
        elif provider == "pollinations":
            client = None
        else:
            raise PipelineError(f"unknown image provider: {provider}")

    def one(i: int) -> Path:
        p = make_image(client, frame_prompt(story, story.scenes[i], o.aspect), o.aspect, wd, i, o.size, rep, o)
        rep.event("asset", scene=i, kind="image")
        return p

    return run_pool("IMAGES", idxs, one, o.workers, rep)


# ───────────────────────── 3. local TTS ─────────────────────────
def _accel() -> str | None:
    try:
        import torch
    except ImportError:
        return None
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    return "mps" if mps and mps.is_available() else None


def resolve_backend(name: str) -> str:
    if name != "auto":
        return name
    if _has_module("chatterbox") and _accel():
        return "chatterbox"
    if _has_module("kokoro_onnx"):
        return "kokoro"
    if (shutil.which("piper") or _has_module("piper")) and os.getenv("SV_PIPER_MODEL"):
        return "piper"
    if shutil.which("espeak-ng") or shutil.which("espeak"):
        return "espeak"
    raise PipelineError("no local TTS found: pip install kokoro-onnx (or chatterbox-tts / piper-tts / espeak-ng)")


def tts_backends() -> list[dict]:
    return [
        {"name": "auto", "ok": True},
        {"name": "chatterbox", "ok": _has_module("chatterbox")},
        {"name": "kokoro", "ok": _has_module("kokoro_onnx")},
        {"name": "piper", "ok": bool(shutil.which("piper") or _has_module("piper"))},
        {"name": "espeak", "ok": bool(shutil.which("espeak-ng") or shutil.which("espeak"))},
    ]


def download(url: str, dest: Path, rep: Reporter) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    rep.log(f"downloading {dest.name} (first run only)...")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)


def write_pcm(samples, sr: int, out: Path) -> None:
    import numpy as np
    pcm = (np.clip(np.asarray(samples, dtype="float32"), -1.0, 1.0) * 32767).astype("<i2").tobytes()
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)


class TTS:
    """One loaded engine per backend. Per-call settings come from Opts."""

    def __init__(self, backend: str, rep: Reporter):
        self.backend = backend
        self.lock = threading.Lock()
        if backend == "kokoro":
            try:
                from kokoro_onnx import Kokoro
            except ImportError:
                raise PipelineError("pip install kokoro-onnx")
            model, voices = CACHE / "kokoro-v1.0.onnx", CACHE / "voices-v1.0.bin"
            for f in (model, voices):
                if not f.exists():
                    download(KOKORO_BASE + f.name, f, rep)
            self.k = Kokoro(str(model), str(voices))
        elif backend == "chatterbox":
            try:
                from chatterbox.tts import ChatterboxTTS
            except ImportError:
                raise PipelineError("pip install chatterbox-tts (python 3.11, GPU strongly recommended)")
            dev = _accel() or "cpu"
            rep.log(f"loading chatterbox on {dev} (first run downloads weights)...")
            self.model = ChatterboxTTS.from_pretrained(device=dev)
        elif backend == "piper":
            exe = shutil.which("piper")
            self.cmd = [exe] if exe else [sys.executable, "-m", "piper"]
        elif backend == "espeak":
            self.exe = shutil.which("espeak-ng") or shutil.which("espeak")
            if not self.exe:
                raise PipelineError("espeak-ng not found (apt install espeak-ng / pkg install espeak)")
        else:
            raise PipelineError(f"unknown tts backend: {backend}")

    def synth(self, text: str, out: Path, o: Opts) -> None:
        text = re.sub(r"[*_#`]", "", text).strip()
        if not text:
            raise PipelineError("empty narration")
        with self.lock:
            if self.backend == "kokoro":
                samples, sr = self.k.create(text, voice=o.voice or "af_heart", speed=o.speed, lang=o.tts_lang)
                write_pcm(samples, sr, out)
            elif self.backend == "chatterbox":
                ref = o.ref_audio
                if ref and not Path(ref).is_file():
                    raise PipelineError(f"reference audio not found: {ref}")
                wav = self.model.generate(text, audio_prompt_path=ref, exaggeration=o.exaggeration,
                                          cfg_weight=0.3 if o.exaggeration >= 0.7 else 0.5)
                write_pcm(wav.squeeze().detach().cpu().numpy(), self.model.sr, out)
            elif self.backend == "piper":
                model = o.voice or os.getenv("SV_PIPER_MODEL")
                if not model:
                    raise PipelineError("piper needs a voice model: set Voice to a .onnx path (or SV_PIPER_MODEL)")
                p = subprocess.run([*self.cmd, "-m", model, "-f", str(out),
                                    "--length-scale", f"{1 / o.speed:.3f}"],
                                   input=text, capture_output=True, text=True)
                if p.returncode:
                    raise PipelineError(f"piper failed: {p.stderr.strip()}")
            else:
                p = subprocess.run([self.exe, "-v", o.voice or o.tts_lang, "-s", str(int(165 * o.speed)),
                                    "-w", str(out), "--stdin"],
                                   input=text, capture_output=True, text=True)
                if p.returncode:
                    raise PipelineError(f"espeak failed: {p.stderr.strip()}")


_ENGINES: dict[str, TTS] = {}
_ENGINES_LOCK = threading.Lock()


def get_tts(name: str, rep: Reporter) -> TTS:
    backend = resolve_backend(name)
    with _ENGINES_LOCK:
        if backend not in _ENGINES:
            _ENGINES[backend] = TTS(backend, rep)
        return _ENGINES[backend]


def synth_wav(tts: TTS, o: Opts, text: str, path: Path) -> None:
    """Synthesize and normalize to mono PCM16 so duration math + muxing are uniform."""
    raw = path.with_suffix(".raw.wav")
    try:
        tts.synth(text, raw, o)
        af = ["-af", f"atempo={o.speed:.3f}"] if tts.backend == "chatterbox" and abs(o.speed - 1) > 0.01 else []
        ffmpeg(["-i", str(raw), *af, "-ac", "1", "-c:a", "pcm_s16le", str(path)])
    finally:
        raw.unlink(missing_ok=True)


def step_voice(story: Story, o: Opts, wd: Path, rep: Reporter, only: list[int] | None = None) -> dict:
    idxs = only if only is not None else list(range(len(story.scenes)))
    (wd / "audio").mkdir(parents=True, exist_ok=True)
    todo = [i for i in idxs if not (wd / "audio" / f"scene_{i:02d}.wav").exists()]
    tts = get_tts(o.tts, rep) if todo else None
    rep.log(f"voice: local · {tts.backend if tts else 'cached'}")

    def one(i: int) -> Path:
        path = wd / "audio" / f"scene_{i:02d}.wav"
        if not path.exists():
            synth_wav(tts, o, story.scenes[i].narration, path)
        rep.event("asset", scene=i, kind="audio")
        return path

    return run_pool("VOICE", idxs, one, 1, rep)


# ───────────────────────── 4. subtitles ─────────────────────────
def srt_ts(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def caption_chunks(text: str, max_words: int = 8) -> list[str]:
    out: list[str] = []
    for sent in re.split(r"(?<=[.!?])\s+", text.strip()):
        words = sent.split()
        if not words:
            continue
        k = math.ceil(len(words) / max_words)
        size = math.ceil(len(words) / k)
        out += [" ".join(words[i:i + size]) for i in range(0, len(words), size)]
    return out


def build_cues(scenes: list[Scene], durs: list[float]) -> list[tuple[float, float, str]]:
    cues, t0 = [], 0.0
    for sc, d in zip(scenes, durs):
        chunks = caption_chunks(sc.narration)
        total = sum(len(c) for c in chunks) or 1
        t = t0
        for c in chunks:
            span = d * len(c) / total
            cues.append((t, t + span, c))
            t += span
        t0 += d + PAD
    return cues


def write_subs(cues, srt: Path, vtt: Path) -> None:
    srt.write_text("\n".join(f"{n}\n{srt_ts(a)} --> {srt_ts(b)}\n{c}\n"
                             for n, (a, b, c) in enumerate(cues, 1)), encoding="utf-8")
    vtt.write_text("WEBVTT\n\n" + "\n".join(
        f"{srt_ts(a).replace(',', '.')} --> {srt_ts(b).replace(',', '.')}\n{c}\n" for a, b, c in cues),
        encoding="utf-8")


# ───────────────────────── 5. sync + render ─────────────────────────
def build_clip(img: Path, wav: Path, adur: float, out: Path, size: tuple[int, int], idx: int, kenburns: bool) -> None:
    W, H = size
    dur = adur + PAD
    frames = max(1, round(dur * FPS))
    if kenburns:
        z = f"1+0.2*on/{frames}" if idx % 2 == 0 else f"1.2-0.2*on/{frames}"
        vf = (f"scale={W * 2}:{H * 2}:force_original_aspect_ratio=increase,crop={W * 2}:{H * 2},"
              f"zoompan=z='{z}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s={W}x{H}:fps={FPS},"
              f"format=yuv420p")
        pre: list[str] = []
    else:
        vf = (f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
              f"fps={FPS},format=yuv420p")
        pre = ["-loop", "1"]
    ffmpeg([*pre, "-i", str(img), "-i", str(wav),
            "-vf", vf, "-af", "apad,aresample=48000,aformat=channel_layouts=stereo",
            "-t", f"{dur:.3f}", "-r", str(FPS),
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out)])


def concat(clips: list[Path], out: Path, wd: Path) -> None:
    lst = wd / "clips.txt"
    lst.write_text("".join(f"file '{c.resolve().as_posix()}'\n" for c in clips), encoding="utf-8")
    ffmpeg(["-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", "-movflags", "+faststart", str(out)])


def has_subtitles_filter() -> bool:
    p = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True)
    return bool(re.search(r"\bsubtitles\b", p.stdout))


def burn_subs(src: Path, srt: Path, out: Path, portrait: bool) -> None:
    fs = 14 if portrait else 18
    style = f"FontName=DejaVu Sans,FontSize={fs},Bold=1,Outline=2,Shadow=0,Alignment=2,MarginV=28"
    ffmpeg(["-i", src.name, "-vf", f"subtitles={srt.name}:force_style='{style}'",
            "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-c:a", "copy",
            "-movflags", "+faststart", out.name], cwd=src.parent)


def step_render(story: Story, o: Opts, wd: Path, rep: Reporter) -> Path:
    n = len(story.scenes)
    imgs, auds, durs = {}, {}, []
    for i in range(n):
        im = find_asset(wd / "images", f"scene_{i:02d}")
        au = wd / "audio" / f"scene_{i:02d}.wav"
        if not im or not au.exists():
            raise PipelineError(f"scene {i + 1} is missing its image or audio")
        imgs[i], auds[i] = im, au
        durs.append(wav_dur(au))
    (wd / "clips").mkdir(parents=True, exist_ok=True)

    def clip(i: int) -> Path:
        out = wd / "clips" / f"scene_{i:02d}.mp4"
        build_clip(imgs[i], auds[i], durs[i], out, o.size, i, o.kenburns)
        return out

    clips = run_pool("CLIPS", list(range(n)), clip, os.cpu_count() or 2, rep)
    srt, vtt = wd / "subs.srt", wd / "subs.vtt"
    write_subs(build_cues(story.scenes, durs), srt, vtt)

    rep.stage("RENDER", 1)
    raw, final = wd / "joined.mp4", wd / "final.mp4"
    concat([clips[i] for i in range(n)], raw, wd)
    if o.burn_subs and has_subtitles_filter():
        burn_subs(raw, srt, final, o.aspect == "9:16")
        raw.unlink()
    else:
        if o.burn_subs:
            rep.log("ffmpeg lacks libass; skipping burn-in (subs.srt / subs.vtt written)", "warn")
        raw.replace(final)
    rep.finish("RENDER")
    rep.event("video")
    rep.log(f"rendered {final.name} · {sum(durs) + PAD * n:.1f}s", "ok")
    return final


def run_pipeline(o: Opts, wd: Path, rep: Reporter, story_file: str | None = None) -> Path:
    load_env()
    if not shutil.which("ffmpeg"):
        raise PipelineError("ffmpeg not found in PATH")
    for d in ("images", "audio", "clips"):
        (wd / d).mkdir(parents=True, exist_ok=True)
    save_cfg(o, wd)
    story = step_story(o, wd, rep, story_file)
    rep.log(f"{story.title} · {len(story.scenes)} scenes · {o.aspect} · {o.size[0]}x{o.size[1]}")
    step_images(story, o, wd, rep)
    step_voice(story, o, wd, rep)
    return step_render(story, o, wd, rep)


# ───────────────────────── CLI ─────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description="prompt -> story -> images -> local TTS -> video")
    ap.add_argument("prompt", help="story idea")
    ap.add_argument("-n", "--scenes", type=int, default=8, help="number of scenes (default 8)")
    ap.add_argument("--aspect", choices=["16:9", "9:16"], default="16:9")
    ap.add_argument("--hd", action="store_true", help="1080p instead of 720p")
    ap.add_argument("--lang", default="English", help="narration language (for the story model)")
    ap.add_argument("--narration", choices=list(NARRATION), default="viral",
                    help="viral = hook-first punchy delivery (default), calm = cinematic")
    ap.add_argument("--tts", choices=["auto", "chatterbox", "kokoro", "piper", "espeak"], default="auto")
    ap.add_argument("--voice", help="kokoro voice (af_heart, am_adam...) | piper .onnx path | espeak voice")
    ap.add_argument("--tts-lang", default="en-us", help="phonemizer language (kokoro/espeak), e.g. en-us, en-gb")
    ap.add_argument("--speed", type=float, default=1.0, help="speech speed multiplier")
    ap.add_argument("--exaggeration", type=float, default=0.6,
                    help="chatterbox expressiveness 0.25-1.5 (higher = more dramatic)")
    ap.add_argument("--ref-audio", help="chatterbox: 5-15s reference clip to clone the voice from")
    ap.add_argument("--workers", type=int, default=3, help="parallel Gemini image calls")
    ap.add_argument("--no-kenburns", action="store_true", help="disable zoom/pan motion")
    ap.add_argument("--burn-subs", action="store_true", help="burn captions into the video")
    ap.add_argument("--llm-url", help="OpenAI-compatible LLM endpoint URL")
    ap.add_argument("--llm-model", help="story model name")
    ap.add_argument("--llm-key", help="story LLM API key")
    ap.add_argument("--image-provider", choices=["gemini", "pollinations", "openai"], default="gemini")
    ap.add_argument("--image-model", help="image model name (e.g. gemini-2.5-flash-image, flux, dall-e-3)")
    ap.add_argument("--image-key", help="image generation API key")
    ap.add_argument("--image-url", help="custom image generation endpoint URL")
    ap.add_argument("--workdir", help="output dir (default: out/<slug>-<hash>)")
    ap.add_argument("--story", help="use an existing story.json instead of generating")
    ap.add_argument("--force", action="store_true", help="discard cached images/audio")
    a = ap.parse_args()

    con.print("[bold green]▌STORYVID[/] [dim]// nim story → gemini frames → local voice → video[/]\n")
    o = Opts.from_dict({
        "prompt": a.prompt, "scenes": a.scenes, "aspect": a.aspect, "hd": a.hd, "lang": a.lang,
        "narration": a.narration, "tts": a.tts, "voice": a.voice, "tts_lang": a.tts_lang,
        "speed": a.speed, "exaggeration": a.exaggeration, "ref_audio": a.ref_audio,
        "workers": a.workers, "kenburns": not a.no_kenburns, "burn_subs": a.burn_subs,
        "llm_url": a.llm_url, "llm_model": a.llm_model, "llm_key": a.llm_key,
        "image_provider": a.image_provider, "image_model": a.image_model,
        "image_key": a.image_key, "image_url": a.image_url})

    key = hashlib.sha1(f"{o.prompt}|{o.scenes}|{o.aspect}|{o.lang}".encode()).hexdigest()[:6]
    slug = re.sub(r"[^a-z0-9]+", "-", o.prompt.lower())[:30].strip("-") or "story"
    wd = Path(a.workdir) if a.workdir else OUT / f"{slug}-{key}"
    if a.force:
        for d in ("images", "audio", "clips"):
            shutil.rmtree(wd / d, ignore_errors=True)

    rep = CLIReporter()
    try:
        final = run_pipeline(o, wd, rep, a.story)
    finally:
        rep.close()
    con.print(f"\n[bold green]DONE[/]  {final}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        con.print("\n[yellow]aborted; re-run to resume from cache[/]")
        sys.exit(130)
    except Exception as e:  # noqa: BLE001
        con.print(f"\n[red]error:[/] {e}\n[dim]re-run to resume from cache[/]")
        sys.exit(1)
