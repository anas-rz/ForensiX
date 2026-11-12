"""Check that the browser studio and forensix.identity.anomaly compute the same embeddings.

Opens the studio in headless Chrome (DevTools protocol), embeds the given files with the studio's
engine (onnxruntime-web on CPU), embeds them again in Python, and prints the cosine between the two
for every frame or speech segment. Needs Google Chrome and a running server:

    python -m forensix.server --port 8000 &
    python scripts/check_studio.py --url http://127.0.0.1:8000/ui/studio/ face.jpg clip.mp4 speech.wav

Videos must be in a codec browsers decode (H.264, VP9, AV1). Exit code 1 if any cosine is below
--min-cosine (default 0.95).
"""

import argparse
import asyncio
import functools
import http.server
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

JS = """
const E = await import("./engine.js"), out = {};
const enc = { face: await E.FaceEncoder.create("wasm"), voice: null };
for (const [name, kind, type] of FILES) {
  if (kind === "voice" && !enc.voice) enc.voice = await E.VoiceEncoder.create("wasm");
  const f = new File([await (await fetch(MEDIA + encodeURIComponent(name))).blob()], name, { type });
  const r = await enc[kind].embedFile(f);
  out[name] = { times: r.times, emb: r.embeddings.map((e) => Array.from(e)) };
}
return out;
"""


def serve(folder: Path, port: int):
    class H(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*"); super().end_headers()

        def log_message(self, *a):
            pass
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), functools.partial(H, directory=str(folder)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


async def run_in_chrome(url: str, js: str, timeout: float, profile: str, port: int = 9335):
    import websockets

    chrome = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chrome")
    if not chrome:
        raise SystemExit("Google Chrome or Chromium is needed")
    p = subprocess.Popen([chrome, "--headless=new", "--no-sandbox", f"--remote-debugging-port={port}",
                          f"--user-data-dir={profile}", "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json")); break
            except OSError:
                time.sleep(0.2)
        ws_url = [t for t in tabs if t["type"] == "page"][0]["webSocketDebuggerUrl"]
        async with websockets.connect(ws_url, max_size=None) as ws:
            n = 0

            async def call(method, **params):
                nonlocal n
                n += 1; i = n
                await ws.send(json.dumps({"id": i, "method": method, "params": params}))
                while True:
                    m = json.loads(await ws.recv())
                    if m.get("id") == i:
                        return m
            await call("Page.navigate", url=url); await asyncio.sleep(3)
            r = await asyncio.wait_for(call("Runtime.evaluate", expression=f"(async () => {{ {js} }})()",
                                            awaitPromise=True, returnByValue=True), timeout)
            res = r["result"]
            if "exceptionDetails" in res:
                raise SystemExit(f"Studio error: {res['exceptionDetails'].get('exception', {}).get('description')}")
            return res["result"]["value"]
    finally:
        p.kill()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--url", default="http://127.0.0.1:8000/ui/studio/")
    ap.add_argument("--min-cosine", type=float, default=0.95)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--profile", help="Chrome profile folder (keeps the model cache); default: a temporary one")
    args = ap.parse_args()

    from forensix.identity.anomaly import FaceEncoder, VoiceEncoder, kind_of

    media = tempfile.mkdtemp()
    files = []
    for f in args.files:
        shutil.copy(f, Path(media) / f.name)
        kind = kind_of(str(f))
        mime = {"voice": "audio/" + f.suffix.lstrip(".").lower(), "face": "video/mp4" if f.suffix.lower() in (".mp4", ".m4v") else
                ("video/webm" if f.suffix.lower() == ".webm" else "image/" + f.suffix.lstrip(".").lower().replace("jpg", "jpeg"))}[kind]
        files.append((f.name, kind, mime))
    srv = serve(Path(media), 8799)
    profile = args.profile or tempfile.mkdtemp()
    try:
        js = f"const FILES = {json.dumps(files)}; const MEDIA = 'http://127.0.0.1:8799/';" + JS
        browser = asyncio.run(run_in_chrome(args.url, js, args.timeout, profile))
    finally:
        srv.shutdown()
        if not args.profile:
            shutil.rmtree(profile, ignore_errors=True)
    enc = {}
    worst = 1.0
    for (name, kind, _), f in zip(files, args.files):
        enc.setdefault(kind, FaceEncoder() if kind == "face" else VoiceEncoder())
        E, times, _ = enc[kind].embed_media(str(f))
        J = np.array(browser[name]["emb"])
        if len(E) != len(J):
            print(f"{name}: python found {len(E)} {'frames with a face' if kind == 'face' else 'segments'}, the browser {len(J)}")
        k = min(len(E), len(J))
        cos = [float(E[i] @ J[i]) for i in range(k)]
        if cos:
            worst = min(worst, min(cos))
            print(f"{name}: {k} {'frames' if kind == 'face' else 'segments'}, cosine min {min(cos):.4f} mean {np.mean(cos):.4f}")
    print("OK" if worst >= args.min_cosine else f"FAIL: cosine {worst:.4f} < {args.min_cosine}")
    raise SystemExit(0 if worst >= args.min_cosine else 1)


if __name__ == "__main__":
    os.environ.setdefault("TMPDIR", tempfile.gettempdir())
    main()
