"""HTTP API for ForensiX detectors (FastAPI).

Models stay loaded between requests, so after the first load (or a preload at
startup) a request pays only for inference. See docs/server.md.
"""

import asyncio
import logging
import math
import secrets
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

import forensix
from forensix.server.config import ServerConfig
from forensix.server.manager import NEEDS_WEIGHTS, ModelManager
from forensix.server.media import MediaInfo, bound_video, probe, sha256_file

log = logging.getLogger("forensix.server")

AUDIO_PHYSICS = {"phys-voice", "phys-room"}
VIDEO_PHYSICS = {"phys-bloodflow", "phys-motion", "phys-lighting"}


def applicable(name: str, family: str, info: MediaInfo) -> str | None:
    """None if ``name`` can score this media, else why not."""
    kind, sound = info.kind, info.kind == "audio" or info.has_audio
    if family == "image":
        ok = kind == "image"
    elif family == "video" or name in VIDEO_PHYSICS:
        ok = kind == "video"
    elif family == "audio" or name in AUDIO_PHYSICS:
        ok = sound
    elif family == "audiovisual":
        ok = kind == "video" and info.has_audio
    elif name == "physdf":
        ok = kind != "image"
    elif name == "id-motion":
        ok = kind == "video"
    elif family == "identity":
        ok = kind in {"image", "video"}
    else:
        ok = True
    return None if ok else f"{name} ({family}) doesn't score {info.modality} media"


def to_jsonable(x):
    """Numpy values, tuples and non-string keys made JSON-safe; NaN and inf become null."""
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return to_jsonable(x.tolist())
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


def create_app(config: ServerConfig | None = None) -> FastAPI:
    config = config or ServerConfig.load()
    manager = ModelManager(config)
    state = {"ready": False, "started": time.time()}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        for name in config.preload:
            log.info("preloading %s", name)
            await run_in_threadpool(manager.get, name)
        state["ready"] = True
        yield

    app = FastAPI(title="ForensiX", version=_version(), lifespan=lifespan,
                  description="Deepfake and manipulated-media detection with ForensiX models.")
    app.state.manager = manager
    app.state.config = config

    def check_key(request: Request, admin: bool = False) -> None:
        expected = (config.admin_key or config.api_key) if admin else config.api_key
        if not expected:
            return
        given = request.headers.get("x-api-key") or ""
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            given = auth[7:]
        if not secrets.compare_digest(given.encode(), expected.encode()):
            raise HTTPException(401, "Missing or wrong API key")

    def user(request: Request) -> None:
        check_key(request)

    def admin(request: Request) -> None:
        check_key(request, admin=True)

    def model_or_404(name: str):
        try:
            return forensix.get_spec(name)
        except KeyError:
            raise HTTPException(404, f"Unknown model {name!r}. See GET /v1/models") from None

    # -- service -----------------------------------------------------------------

    @app.get("/health", tags=["service"])
    async def health():
        """Liveness: the process is up."""
        return {"status": "ok"}

    @app.get("/ready", tags=["service"])
    async def ready():
        """Readiness: preloaded models are in memory."""
        if not state["ready"]:
            raise HTTPException(503, "Loading models")
        return {"status": "ready", "loaded": manager.loaded()}

    @app.get("/v1/stats", tags=["service"], dependencies=[Depends(user)])
    async def stats():
        return {"uptime_s": round(time.time() - state["started"], 1), "device": config.device,
                "max_loaded": config.max_loaded, "models": manager.stats()}

    # -- models ------------------------------------------------------------------

    def describe(name: str) -> dict:
        spec = forensix.get_spec(name)
        configured = name in config.weights
        return {"name": name, "family": spec.family, "license": spec.license, "paper": spec.paper,
                "description": spec.description, "loaded": name in manager.loaded(),
                "available": name not in NEEDS_WEIGHTS or configured,
                "threshold_override": config.thresholds.get(name),
                "default_for": [m for m, names in config.default_models.items() if name in names]}

    @app.get("/v1/models", tags=["models"], dependencies=[Depends(user)])
    async def list_models(family: str | None = None):
        return {"models": [describe(n) for n in forensix.list_models(family)]}

    @app.get("/v1/models/{name}", tags=["models"], dependencies=[Depends(user)])
    async def get_model(name: str):
        model_or_404(name)
        return describe(name)

    @app.post("/v1/models/{name}/load", tags=["models"], dependencies=[Depends(admin)])
    async def load_model(name: str):
        """Load (and warm up) a model now, so later requests don't wait for it."""
        model_or_404(name)
        try:
            entry = await run_in_threadpool(manager.get, name)
        except LookupError as e:
            raise HTTPException(409, str(e)) from None
        return {"name": name, "loaded": True, "load_seconds": entry.load_seconds}

    @app.delete("/v1/models/{name}", tags=["models"], dependencies=[Depends(admin)])
    async def unload_model(name: str):
        model_or_404(name)
        return {"name": name, "unloaded": await run_in_threadpool(manager.unload, name)}

    # -- detection ---------------------------------------------------------------

    async def save_upload(upload: UploadFile, directory: Path) -> Path:
        limit = int(config.max_upload_mb * 1024 * 1024)
        path, size = directory / "upload", 0
        with open(path, "wb") as f:
            while chunk := await upload.read(1 << 20):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, f"File larger than {config.max_upload_mb:g} MB")
                f.write(chunk)
        if size == 0:
            raise HTTPException(400, "Empty file")
        return path

    async def run_model(name: str, path: Path, sha: str, info: MediaInfo, identity: str | None,
                        details: bool, segments: bool) -> dict:
        spec = forensix.get_spec(name)
        if reason := applicable(name, spec.family, info):
            return {"status": "not_applicable", "reason": reason}
        if spec.family == "identity" and not identity:
            return {"status": "not_applicable", "reason": f"{name} needs the identity form field"}
        start = time.perf_counter()
        try:
            pred, cached = await run_in_threadpool(manager.predict, name, path, sha, identity)
        except LookupError as e:  # needs trained weights
            return {"status": "unavailable", "reason": str(e)}
        except Exception as e:  # undecodable media, no face track, ...
            log.warning("%s failed: %r", name, e)
            return {"status": "error", "error": str(e)[:500], "seconds": round(time.perf_counter() - start, 3)}
        threshold = config.thresholds.get(name, pred.threshold)
        result = {
            "status": "ok", "family": spec.family, "score": to_jsonable(pred.fake_prob),
            "score_type": pred.score_type, "threshold": threshold,
            "label": None if threshold is None else ("fake" if pred.fake_prob >= threshold else "real"),
            "n_segments": len(pred.segment_fake_probs), "cached": cached,
            "seconds": round(time.perf_counter() - start, 3),
        }
        if spec.family == "video" and not pred.segment_fake_probs:
            result.update(status="no_face", label=None)  # these models return 0.5 without a face
        if segments:
            result["segments"] = to_jsonable(pred.segment_fake_probs)
        if details:
            result["details"] = to_jsonable(pred.details)
        return result

    async def detect_one(upload: UploadFile, models: list[str] | None, identity: str | None,
                         details: bool, segments: bool) -> dict:
        start = time.perf_counter()
        with tempfile.TemporaryDirectory(dir=config.tmp_dir) as tmp:
            raw = await save_upload(upload, Path(tmp))
            sha = await run_in_threadpool(sha256_file, raw)
            try:
                info = await run_in_threadpool(probe, raw)
            except ValueError as e:
                raise HTTPException(415, str(e)) from None
            path = raw.rename(raw.with_suffix(info.suffix(Path(upload.filename or "").suffix)))
            bounded = None
            if info.kind == "video":
                path, bounded = await run_in_threadpool(bound_video, path, info, config.max_video_seconds,
                                                        config.max_video_height, Path(tmp))
            names = models or config.default_models.get(info.modality, [])
            results = await asyncio.gather(*(run_model(n, path, sha, info, identity, details, segments)
                                             for n in names))
        return {
            "id": uuid.uuid4().hex, "filename": upload.filename, "sha256": sha,
            "media": {"kind": info.kind, "has_audio": info.has_audio, "duration_s": info.duration_s,
                      "bounded": bounded},
            "results": dict(zip(names, results)),
            "seconds": round(time.perf_counter() - start, 3),
        }

    def parse_models(models: str | None) -> list[str] | None:
        if not models:
            return None
        names = list(dict.fromkeys(m.strip() for m in models.split(",") if m.strip()))
        for n in names:
            model_or_404(n)
        return names

    @app.post("/v1/detect", tags=["detect"], dependencies=[Depends(user)])
    async def detect(
        file: UploadFile = File(..., description="Image, video or audio"),
        models: str | None = Form(None, description="Comma-separated model names; default: by media type"),
        identity: str | None = Form(None, description="Enrolled identity, for identity models"),
        details: bool = Form(False, description="Include each model's evidence"),
        segments: bool = Form(False, description="Include per-segment scores"),
    ):
        """Score one file with the requested models (or the defaults for its media type)."""
        return await detect_one(file, parse_models(models), identity, details, segments)

    @app.post("/v1/detect/batch", tags=["detect"], dependencies=[Depends(user)])
    async def detect_batch(
        files: list[UploadFile] = File(...),
        models: str | None = Form(None),
        identity: str | None = Form(None),
        details: bool = Form(False),
        segments: bool = Form(False),
    ):
        """Score several files. They run concurrently, so image and audio models batch them."""
        names = parse_models(models)

        async def one(f: UploadFile) -> dict:
            try:
                return await detect_one(f, names, identity, details, segments)
            except HTTPException as e:
                return {"filename": f.filename, "error": e.detail, "status_code": e.status_code}

        return {"results": await asyncio.gather(*(one(f) for f in files))}

    # -- identities --------------------------------------------------------------

    def identity_model(name: str) -> str:
        if model_or_404(name).family != "identity":
            raise HTTPException(400, f"{name} is not an identity model")
        return name

    @app.get("/v1/identities", tags=["identities"], dependencies=[Depends(user)])
    async def list_identities(model: str = Query("id-face")):
        try:
            names = await run_in_threadpool(manager.identities, identity_model(model))
        except LookupError as e:
            raise HTTPException(409, str(e)) from None
        return {"model": model, "identities": names}

    @app.post("/v1/identities/{identity}", tags=["identities"], dependencies=[Depends(admin)])
    async def enroll(identity: str, files: list[UploadFile] = File(...), model: str = Query("id-face")):
        """Enroll (or add references to) a person from photos and/or videos."""
        model = identity_model(model)
        with tempfile.TemporaryDirectory(dir=config.tmp_dir) as tmp:
            paths = []
            for i, f in enumerate(files):
                sub = Path(tmp) / str(i)
                sub.mkdir()
                raw = await save_upload(f, sub)
                try:
                    info = await run_in_threadpool(probe, raw)
                except ValueError as e:
                    raise HTTPException(415, f"{f.filename}: {e}") from None
                paths.append(raw.rename(raw.with_suffix(info.suffix(Path(f.filename or "").suffix))))
            try:
                result = await run_in_threadpool(manager.enroll, model, identity, paths)
            except LookupError as e:
                raise HTTPException(409, str(e)) from None
            except ValueError as e:
                raise HTTPException(422, str(e)) from None
        return {"model": model, **to_jsonable(result)}

    @app.delete("/v1/identities/{identity}", tags=["identities"], dependencies=[Depends(admin)])
    async def remove_identity(identity: str, model: str = Query("id-face")):
        removed = await run_in_threadpool(manager.remove_identity, identity_model(model), identity)
        if not removed:
            raise HTTPException(404, f"{identity!r} is not enrolled in {model}")
        return {"model": model, "identity": identity, "removed": True}

    return app


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("forensix")
    except Exception:
        return "unknown"

