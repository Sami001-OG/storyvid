#!/usr/bin/env python3
"""
storyvid web UI — interactive front-end for the storyvid pipeline. Stdlib only.

    export NVIDIA_API_KEY=...  GEMINI_API_KEY=...
    python storyvid_web.py [--host 127.0.0.1] [--port 8765]

Open http://127.0.0.1:8765 . Projects live in ./out (SV_OUT) and are shared with the CLI.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import shutil
import sys
import threading
import time
import traceback
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import storyvid as sv

HERE = Path(__file__).resolve().parent
OUT = sv.OUT.resolve()
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,120}$")
mimetypes.add_type("text/vtt", ".vtt")
mimetypes.add_type("video/mp4", ".mp4")
mimetypes.add_type("audio/wav", ".wav")

TTS_DEFAULT: str | None = None


# ───────────────────────── projects / events ─────────────────────────
class Proj:
    def __init__(self, pid: str):
        self.id = pid
        self.wd = OUT / pid
        self.events: list[dict] = []
        self.cond = threading.Condition()
        self.busy = False

    def emit(self, type_: str, **d) -> None:
        with self.cond:
            self.events.append({"seq": len(self.events), "t": time.time(), "type": type_, **d})
            self.cond.notify_all()

    def claim(self) -> bool:
        with self.cond:
            if self.busy:
                return False
            self.busy = True
            return True


class WebReporter(sv.Reporter):
    def __init__(self, p: Proj):
        self.p = p

    def stage(self, name, total=0):
        self.p.emit("stage", name=name, total=total or 1)

    def tick(self, name):
        self.p.emit("tick", name=name)

    def finish(self, name):
        self.p.emit("finish", name=name)

    def log(self, msg, level="info"):
        self.p.emit("log", msg=msg, level=level)

    def event(self, etype, /, **data):
        self.p.emit(etype, **data)


PROJECTS: dict[str, Proj] = {}
PLOCK = threading.Lock()


def get_proj(pid: str) -> Proj | None:
    if not ID_RE.match(pid) or not (OUT / pid).is_dir():
        return None
    with PLOCK:
        if pid not in PROJECTS:
            PROJECTS[pid] = Proj(pid)
        return PROJECTS[pid]


def launch(p: Proj, label: str, fn) -> None:
    """Run fn(reporter) in a thread. Caller must have claimed p."""
    p.emit("busy", label=label)

    def work():
        err = None
        try:
            fn(WebReporter(p))
        except sv.PipelineError as e:
            err = str(e)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            err = f"{type(e).__name__}: {e}"
        finally:
            with p.cond:
                p.busy = False
        p.emit("error", message=err) if err else p.emit("done")

    threading.Thread(target=work, daemon=True).start()


def read_story(wd: Path) -> sv.Story | None:
    f = wd / "story.json"
    if not f.exists():
        return None
    return sv.Story.model_validate_json(f.read_text(encoding="utf-8"))


def furl(pid: str, path: Path) -> str:
    return f"/files/{pid}/{path.relative_to(OUT / pid).as_posix()}?v={int(path.stat().st_mtime * 1000)}"


def project_state(p: Proj) -> dict:
    wd = p.wd
    cfg = {}
    if (wd / "project.json").exists():
        cfg = json.loads((wd / "project.json").read_text(encoding="utf-8"))
    story = read_story(wd)
    scenes, newest = [], 0.0
    if story:
        newest = (wd / "story.json").stat().st_mtime
        for i, s in enumerate(story.scenes):
            img = sv.find_asset(wd / "images", f"scene_{i:02d}")
            aud = wd / "audio" / f"scene_{i:02d}.wav"
            for f in (img, aud):
                if f and f.exists():
                    newest = max(newest, f.stat().st_mtime)
            scenes.append({
                "narration": s.narration, "image_prompt": s.image_prompt,
                "image": furl(p.id, img) if img else None,
                "audio": furl(p.id, aud) if aud.exists() else None,
                "duration": sv.wav_dur(aud) if aud.exists() else None,
            })
    final = wd / "final.mp4"
    return {
        "id": p.id, "opts": cfg, "busy": p.busy, "pad": sv.PAD,
        "story": {"title": story.title, "visual_style": story.visual_style,
                  "characters": story.characters} if story else None,
        "scenes": scenes,
        "video": furl(p.id, final) if final.exists() else None,
        "vtt": furl(p.id, wd / "subs.vtt") if (wd / "subs.vtt").exists() else None,
        "srt": furl(p.id, wd / "subs.srt") if (wd / "subs.srt").exists() else None,
        "dirty": bool(final.exists() and newest > final.stat().st_mtime),
    }


def list_projects() -> list[dict]:
    rows = []
    if OUT.is_dir():
        for d in OUT.iterdir():
            if not d.is_dir() or not ID_RE.match(d.name):
                continue
            if not ((d / "project.json").exists() or (d / "story.json").exists()):
                continue
            title, prompt = d.name, ""
            try:
                if (d / "story.json").exists():
                    title = json.loads((d / "story.json").read_text(encoding="utf-8")).get("title", d.name)
                if (d / "project.json").exists():
                    prompt = json.loads((d / "project.json").read_text(encoding="utf-8")).get("prompt", "")
            except (OSError, ValueError):
                pass
            rows.append({"id": d.name, "title": title, "prompt": prompt,
                         "mtime": d.stat().st_mtime, "video": (d / "final.mp4").exists()})
    return sorted(rows, key=lambda r: -r["mtime"])


# ───────────────────────── http ─────────────────────────
class Err(Exception):
    def __init__(self, code: int, msg: str):
        super().__init__(msg)
        self.code, self.msg = code, msg


R_PROJ = re.compile(r"^/api/projects/([^/]+)$")
R_EVENTS = re.compile(r"^/api/projects/([^/]+)/events$")
R_SCENE = re.compile(r"^/api/projects/([^/]+)/scenes/(\d+)$")
R_ACT = re.compile(r"^/api/projects/([^/]+)/(resume|render)$")
R_FILE = re.compile(r"^/files/([^/]+)/(.+)$")


class Handler(BaseHTTPRequestHandler):
    server_version = "storyvid"

    def log_message(self, fmt, *args):  # quiet
        pass

    # ── helpers
    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n > 1_000_000:
            raise Err(413, "body too large")
        if not n:
            return {}
        try:
            d = json.loads(self.rfile.read(n))
        except ValueError:
            raise Err(400, "invalid JSON")
        if not isinstance(d, dict):
            raise Err(400, "JSON object expected")
        return d

    def _guard(self, mutating: bool) -> None:
        allowed = self.server.allowed_hosts  # type: ignore[attr-defined]
        if allowed and (self.headers.get("Host") or "").lower() not in allowed:
            raise Err(403, "bad host")
        if mutating and self.headers.get("X-SV") != "1":
            raise Err(403, "missing X-SV header")

    def _dispatch(self, fn, mutating: bool) -> None:
        try:
            self._guard(mutating)
            fn()
        except Err as e:
            self._json({"error": e.msg}, e.code)
        except sv.PipelineError as e:
            self._json({"error": str(e)}, 400)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            try:
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)
            except OSError:
                pass

    def _proj(self, pid: str) -> Proj:
        p = get_proj(unquote(pid))
        if not p:
            raise Err(404, "no such project")
        return p

    # ── verbs
    def do_GET(self):
        self._dispatch(self._get, False)

    def do_POST(self):
        self._dispatch(self._post, True)

    def do_DELETE(self):
        self._dispatch(self._delete, True)

    def _get(self):
        u = urlparse(self.path)
        path = u.path
        if path in ("/", "/index.html"):
            html = (HERE / "ui.html").read_bytes()
            return self._send(200, html, "text/html; charset=utf-8")
        if path in ("/api/status", "/api/settings"):
            sv.load_env()
            return self._json({
                "keys": {
                    "nvidia": bool(os.getenv("NVIDIA_API_KEY") or os.getenv("SV_LLM_KEY")),
                    "gemini": bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")),
                    "openai": bool(os.getenv("OPENAI_API_KEY")),
                },
                "presets": {
                    "nvidia_key": os.getenv("NVIDIA_API_KEY", "") or os.getenv("SV_LLM_KEY", ""),
                    "gemini_key": os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", ""),
                    "openai_key": os.getenv("OPENAI_API_KEY", ""),
                    "image_key": os.getenv("SV_IMAGE_KEY", "") or os.getenv("OPENAI_API_KEY", "") or os.getenv("GEMINI_API_KEY", ""),
                    "llm_url": sv.LLM_URL,
                    "llm_model": sv.LLM_MODEL,
                    "image_provider": sv.IMAGE_PROVIDER,
                    "image_model": sv.IMAGE_MODEL,
                    "image_url": sv.IMAGE_URL,
                },
                "ffmpeg": bool(shutil.which("ffmpeg")),
                "tts": {"list": sv.tts_backends(), "default": TTS_DEFAULT},
                "llm_model": sv.LLM_MODEL,
                "llm_url": sv.LLM_URL,
                "image_provider": sv.IMAGE_PROVIDER,
                "image_model": sv.IMAGE_MODEL,
                "image_url": sv.IMAGE_URL,
                "kokoro_voices": sv.KOKORO_VOICES,
            })
        if path == "/api/projects":
            return self._json(list_projects())
        m = R_EVENTS.match(path)
        if m:
            return self._sse(self._proj(m.group(1)), u.query)
        m = R_PROJ.match(path)
        if m:
            return self._json(project_state(self._proj(m.group(1))))
        m = R_FILE.match(path)
        if m:
            p = self._proj(m.group(1))
            target = (p.wd / unquote(m.group(2))).resolve()
            if not target.is_relative_to(p.wd.resolve()) or not target.is_file():
                raise Err(404, "not found")
            return self._file(target, "v" in parse_qs(u.query))
        raise Err(404, "not found")

    def _post(self):
        path = urlparse(self.path).path
        if path == "/api/settings":
            body = self._body()
            updates = {}
            if "nvidia_key" in body:
                updates["NVIDIA_API_KEY"] = body["nvidia_key"]
            if "gemini_key" in body:
                updates["GEMINI_API_KEY"] = body["gemini_key"]
            if "openai_key" in body:
                updates["OPENAI_API_KEY"] = body["openai_key"]
            if "llm_url" in body and body["llm_url"]:
                updates["SV_LLM_URL"] = body["llm_url"]
            if "llm_model" in body and body["llm_model"]:
                updates["SV_LLM_MODEL"] = body["llm_model"]
            if "image_provider" in body and body["image_provider"]:
                updates["SV_IMAGE_PROVIDER"] = body["image_provider"]
            if "image_model" in body and body["image_model"]:
                updates["SV_IMAGE_MODEL"] = body["image_model"]
            if "image_url" in body and body["image_url"]:
                updates["SV_IMAGE_URL"] = body["image_url"]
            if "image_key" in body:
                updates["SV_IMAGE_KEY"] = body["image_key"]
            sv.update_env(updates)
            return self._json({"ok": True, "settings": {
                "llm_url": sv.LLM_URL,
                "llm_model": sv.LLM_MODEL,
                "image_provider": sv.IMAGE_PROVIDER,
                "image_model": sv.IMAGE_MODEL,
                "image_url": sv.IMAGE_URL,
                "keys": {
                    "nvidia": bool(os.getenv("NVIDIA_API_KEY") or os.getenv("SV_LLM_KEY")),
                    "gemini": bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")),
                    "openai": bool(os.getenv("OPENAI_API_KEY")),
                }
            }})
        if path == "/api/projects":
            return self._create(self._body())
        if path == "/api/preview":
            return self._preview(self._body())
        m = R_ACT.match(path)
        if m:
            p = self._proj(m.group(1))
            return self._resume(p) if m.group(2) == "resume" else self._render(p, self._body())
        m = R_SCENE.match(path)
        if m:
            return self._scene(self._proj(m.group(1)), int(m.group(2)), self._body())
        raise Err(404, "not found")

    def _delete(self):
        m = R_PROJ.match(urlparse(self.path).path)
        if not m:
            raise Err(404, "not found")
        p = self._proj(m.group(1))
        if not p.claim():
            raise Err(409, "project is busy")
        try:
            shutil.rmtree(p.wd)
            with PLOCK:
                PROJECTS.pop(p.id, None)
        finally:
            p.busy = False
        self._json({"ok": True})

    # ── actions
    def _create(self, body: dict):
        o = sv.Opts.from_dict(body)
        slug = re.sub(r"[^a-z0-9]+", "-", o.prompt.lower())[:30].strip("-") or "story"
        pid = f"{slug}-{time.strftime('%y%m%d-%H%M%S')}-{os.urandom(2).hex()}"
        (OUT / pid).mkdir(parents=True, exist_ok=True)
        p = get_proj(pid)
        p.claim()
        sv.save_cfg(o, p.wd)
        launch(p, "pipeline", lambda rep: sv.run_pipeline(o, p.wd, rep))
        self._json({"id": pid}, 201)

    def _resume(self, p: Proj):
        o = sv.load_cfg(p.wd)
        if not p.claim():
            raise Err(409, "project is busy")
        launch(p, "resume", lambda rep: sv.run_pipeline(o, p.wd, rep))
        self._json({"started": True})

    def _render(self, p: Proj, body: dict):
        o = sv.load_cfg(p.wd)
        story = read_story(p.wd)
        if not story:
            raise Err(400, "no story yet")
        for k in ("burn_subs", "kenburns"):
            if k in body:
                setattr(o, k, bool(body[k]))
        if not p.claim():
            raise Err(409, "project is busy")
        sv.save_cfg(o, p.wd)
        launch(p, "render", lambda rep: sv.step_render(story, o, p.wd, rep))
        self._json({"started": True})

    def _scene(self, p: Proj, i: int, body: dict):
        o = sv.load_cfg(p.wd)
        story = read_story(p.wd)
        if not story or i >= len(story.scenes):
            raise Err(404, "no such scene")
        if not p.claim():
            raise Err(409, "project is busy")
        try:
            sc = story.scenes[i]
            todo = {k for k in (body.get("regen") or []) if k in ("image", "voice")}
            nar, prm = body.get("narration"), body.get("image_prompt")
            if isinstance(nar, str) and nar.strip() and nar.strip() != sc.narration:
                sc.narration = nar.strip()
                todo.add("voice")
            if isinstance(prm, str) and prm.strip() and prm.strip() != sc.image_prompt:
                sc.image_prompt = prm.strip()
                todo.add("image")
            if not todo:
                p.busy = False
                return self._json({"started": False})
            (p.wd / "story.json").write_text(story.model_dump_json(indent=2), encoding="utf-8")
            if "image" in todo:
                for f in (p.wd / "images").glob(f"scene_{i:02d}.*"):
                    f.unlink()
            if "voice" in todo:
                (p.wd / "audio" / f"scene_{i:02d}.wav").unlink(missing_ok=True)
        except BaseException:
            p.busy = False
            raise

        def work(rep):
            if "image" in todo:
                sv.step_images(story, o, p.wd, rep, only=[i])
            if "voice" in todo:
                sv.step_voice(story, o, p.wd, rep, only=[i])

        launch(p, f"scene {i + 1}", work)
        self._json({"started": True})

    def _preview(self, body: dict):
        text = str(body.get("text") or "Nobody believed him, until the storm answered back.")[:300]
        o = sv.Opts.from_dict({**body, "prompt": "preview"})
        tts = sv.get_tts(o.tts, sv.Reporter())
        d = OUT / ".preview"
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{os.urandom(4).hex()}.wav"
        try:
            sv.synth_wav(tts, o, text, path)
            data = path.read_bytes()
        finally:
            path.unlink(missing_ok=True)
        self._send(200, data, "audio/wav")

    # ── streaming
    def _sse(self, p: Proj, query: str):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        self.close_connection = True
        lei = self.headers.get("Last-Event-ID", "")
        since = parse_qs(query).get("since", ["0"])[0]
        i = int(lei) + 1 if lei.isdigit() else (int(since) if since.isdigit() else 0)
        try:
            while True:
                with p.cond:
                    while i >= len(p.events):
                        if not p.cond.wait(timeout=15):
                            break
                    batch = p.events[i:]
                if batch:
                    for ev in batch:
                        self.wfile.write(f"id: {ev['seq']}\ndata: {json.dumps(ev)}\n\n".encode())
                    i += len(batch)
                else:
                    self.wfile.write(b": ka\n\n")
                self.wfile.flush()
        except OSError:
            pass

    def _file(self, path: Path, cacheable: bool):
        size = path.stat().st_size
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        start, end, code = 0, size - 1, 200
        rng = self.headers.get("Range")
        if rng:
            m = re.match(r"^bytes=(\d*)-(\d*)$", rng.strip())
            if m and (m.group(1) or m.group(2)):
                a, b = m.groups()
                if a == "":
                    start = max(0, size - int(b))
                else:
                    start = int(a)
                    end = min(int(b), size - 1) if b else size - 1
                if start > end or start >= size:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                code = 206
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "private, max-age=3600" if cacheable else "no-cache")
        if code == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        try:
            with open(path, "rb") as f:
                f.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = f.read(min(65536, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def make_server(host: str, port: int) -> Server:
    srv = Server((host, port), Handler)
    port = srv.server_address[1]
    loopback = host in ("127.0.0.1", "localhost", "::1")
    srv.allowed_hosts = ({f"localhost:{port}", f"127.0.0.1:{port}", f"[::1]:{port}"}  # type: ignore[attr-defined]
                         if loopback else None)
    return srv


def main() -> int:
    global TTS_DEFAULT
    ap = argparse.ArgumentParser(description="storyvid web UI")
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to expose on your network (no auth!)")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        TTS_DEFAULT = sv.resolve_backend("auto")
    except sv.PipelineError:
        TTS_DEFAULT = None
    srv = make_server(a.host, a.port)
    sv.con.print(f"[bold green]▌STORYVID WEB[/]  http://{a.host}:{srv.server_address[1]}   "
                 f"[dim]out={OUT}  tts={TTS_DEFAULT}[/]")
    if a.host not in ("127.0.0.1", "localhost", "::1"):
        sv.con.print("[yellow]warning: exposed beyond localhost with no authentication[/]")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
