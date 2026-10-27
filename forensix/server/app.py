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
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

import forensix
from forensix.server.calibration import Jobs
from forensix.server.config import ServerConfig
from forensix.server.manager import NEEDS_WEIGHTS, ModelManager
from forensix.server.media import MediaInfo, applicable, bound_video, probe, sha256_file

log = logging.getLogger("forensix.server")

VERDICTS = {("fake",): "fake", ("real",): "real", ("real", "fake"): "uncertain", (): "out_of_distribution"}


def confidence(pred, calibration: str) -> dict:
    """The calibrated reading of a prediction (see docs/server.md#confidence)."""
    d = pred.details
    return {"fake_probability": to_jsonable(pred.fake_prob), "verdict": VERDICTS[tuple(d["prediction_set"])],
            "prediction_set": d["prediction_set"], "p_real": d["p_real"], "p_fake": d["p_fake"],
            "alpha": d["alpha"], "calibration": calibration}


class CalibrateRequest(BaseModel):
    dataset: str = Field(description="Dataset id from POST /v1/datasets")
    models: list[str] = Field(min_length=1)
    alpha: float = Field(0.05, gt=0, lt=1, description="Target false-alarm rate on real media")
    folds: int = Field(5, ge=2, le=20)
    seed: int = 0
    activate: bool = Field(True, description="Use the new calibration for predictions right away")
    score_types: dict[str, str] = Field(default_factory=dict,
                                        description='"raw" for models whose score is not a probability (auto-detected)')


class CheckRequest(BaseModel):
    dataset: str
    models: list[str] | None = Field(None, description="Default: every model with an active calibration")
    level: float = Field(0.01, gt=0, lt=1, description="Significance level of the drift tests")


class ActiveRequest(BaseModel):
    version: int | None = Field(description="Version to use for predictions; null for uncalibrated scores")


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
    jobs = Jobs(manager.store, manager, config)
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

    def model_or_404(name: str) -> str:
        """The model's family, or 404."""
        try:
            return manager.family(name)
        except KeyError:
            raise HTTPException(404, f"Unknown model {name!r}. See GET /v1/models") from None

    fusion_members: dict[str, list[tuple[str, str]]] = {}

    def members(name: str) -> list[tuple[str, str]]:
        """(model, family) of each member of a configured fusion table."""
        if name not in fusion_members:
            from forensix.fusion import Fusion

            fusion = Fusion.load(config.fusions[name])
            fusion_members[name] = [(m.model, forensix.get_spec(m.model).family) for m in fusion.members]
        return fusion_members[name]

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
        if name in config.fusions:
            return {"name": name, "family": "fusion", "members": [m for m, _ in members(name)],
                    "table": config.fusions[name], "loaded": name in manager.loaded(), "available": True,
                    "threshold_override": config.thresholds.get(name),
                    "default_for": [m for m, names in config.default_models.items() if name in names]}
        spec = forensix.get_spec(name)
        configured = name in config.weights
        return {"name": name, "family": spec.family, "license": spec.license, "paper": spec.paper,
                "description": spec.description, "loaded": name in manager.loaded(),
                "available": name not in NEEDS_WEIGHTS or configured,
                "threshold_override": config.thresholds.get(name),
                "default_for": [m for m, names in config.default_models.items() if name in names]}

    @app.get("/v1/models", tags=["models"], dependencies=[Depends(user)])
    async def list_models(family: str | None = None):
        names = forensix.list_models(family) if family != "fusion" else []
        if family in (None, "fusion"):
            names += sorted(config.fusions)
        return {"models": [describe(n) for n in names]}

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

    async def save_upload(upload: UploadFile, directory: Path, limit_mb: float | None = None) -> Path:
        limit_mb = config.max_upload_mb if limit_mb is None else limit_mb
        limit = int(limit_mb * 1024 * 1024)
        path, size = directory / "upload", 0
        with open(path, "wb") as f:
            while chunk := await upload.read(1 << 20):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, f"File larger than {limit_mb:g} MB")
                f.write(chunk)
        if size == 0:
            raise HTTPException(400, "Empty file")
        return path

    async def run_model(name: str, path: Path, sha: str, info: MediaInfo, identity: str | None,
                        details: bool, segments: bool) -> dict:
        family = manager.family(name)
        checks = members(name) if family == "fusion" else [(name, family)]
        for member, member_family in checks:
            if reason := applicable(member, member_family, info):
                return {"status": "not_applicable", "reason": reason}
        if family == "identity" and not identity:
            return {"status": "not_applicable", "reason": f"{name} needs the identity form field"}
        start = time.perf_counter()
        try:
            pred, cached, calibration = await run_in_threadpool(manager.predict, name, path, sha, identity)
        except LookupError as e:  # needs trained weights
            return {"status": "unavailable", "reason": str(e)}
        except Exception as e:  # undecodable media, no face track, ...
            log.warning("%s failed: %r", name, e)
            return {"status": "error", "error": str(e)[:500], "seconds": round(time.perf_counter() - start, 3)}
        # a calibration sets its own threshold; config thresholds apply to uncalibrated scores
        threshold = pred.threshold if calibration else config.thresholds.get(name, pred.threshold)
        result = {
            "status": "ok", "family": family, "score": to_jsonable(pred.fake_prob),
            "score_type": pred.score_type, "threshold": to_jsonable(threshold),
            "label": None if threshold is None else ("fake" if pred.fake_prob >= threshold else "real"),
            "n_segments": len(pred.segment_fake_probs), "cached": cached, "calibration": calibration,
            "seconds": round(time.perf_counter() - start, 3),
        }
        if calibration:
            result["confidence"] = confidence(pred, calibration)
        if family == "video" and not pred.segment_fake_probs:
            result.update(status="no_face", label=None)  # these models return 0.5 without a face
        if segments:
            result["segments"] = to_jsonable(pred.segment_fake_probs)
        if details:
            result["details"] = to_jsonable(pred.details)
        return result

    async def prepare(upload: UploadFile, tmp: Path) -> tuple[Path, str, MediaInfo, dict | None]:
        """Save, hash, probe and bound an upload: ``(path, sha256, info, bounded)``."""
        raw = await save_upload(upload, tmp)
        sha = await run_in_threadpool(sha256_file, raw)
        try:
            info = await run_in_threadpool(probe, raw)
        except ValueError as e:
            raise HTTPException(415, str(e)) from None
        path = raw.rename(raw.with_suffix(info.suffix(Path(upload.filename or "").suffix)))
        bounded = None
        if info.kind == "video":
            path, bounded = await run_in_threadpool(bound_video, path, info, config.max_video_seconds,
                                                    config.max_video_height, tmp)
        return path, sha, info, bounded

    async def detect_one(upload: UploadFile, models: list[str] | None, identity: str | None,
                         details: bool, segments: bool) -> dict:
        start = time.perf_counter()
        with tempfile.TemporaryDirectory(dir=config.tmp_dir) as tmp:
            path, sha, info, bounded = await prepare(upload, Path(tmp))
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
        if model_or_404(name) != "identity":
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

    # -- calibration -------------------------------------------------------------

    @app.post("/v1/datasets", tags=["calibration"], status_code=201, dependencies=[Depends(admin)])
    async def upload_dataset(
        file: UploadFile = File(..., description="ZIP in the ForensiX dataset format (manifest.csv, or real/ and fake/)"),
        name: str | None = Form(None),
    ):
        """Upload a labelled dataset. It's validated and stored; calibrate models on it with POST /v1/calibrations."""
        with tempfile.TemporaryDirectory(dir=config.tmp_dir) as tmp:
            archive = await save_upload(file, Path(tmp), config.max_dataset_mb)
            try:
                return await run_in_threadpool(manager.store.add_dataset, archive, name or file.filename or "dataset",
                                               int(config.max_dataset_mb * 1024 * 1024))
            except (ValueError, zipfile.BadZipFile) as e:
                raise HTTPException(422, f"Invalid dataset: {e}") from None

    @app.get("/v1/datasets", tags=["calibration"], dependencies=[Depends(user)])
    async def list_datasets():
        return {"datasets": manager.store.datasets()}

    @app.get("/v1/datasets/{dataset_id}", tags=["calibration"], dependencies=[Depends(user)])
    async def get_dataset(dataset_id: str):
        if (info := manager.store.dataset_info(dataset_id)) is None:
            raise HTTPException(404, f"Unknown dataset {dataset_id!r}")
        return info

    @app.delete("/v1/datasets/{dataset_id}", tags=["calibration"], dependencies=[Depends(admin)])
    async def delete_dataset(dataset_id: str):
        if not manager.store.delete_dataset(dataset_id):
            raise HTTPException(404, f"Unknown dataset {dataset_id!r}")
        return {"id": dataset_id, "deleted": True}

    def check_job_inputs(dataset: str, models: list[str]) -> None:
        if manager.store.dataset_info(dataset) is None:
            raise HTTPException(404, f"Unknown dataset {dataset!r}")
        for m in models:
            model_or_404(m)

    @app.post("/v1/calibrations", tags=["calibration"], status_code=202, dependencies=[Depends(admin)])
    async def start_calibration(req: CalibrateRequest):
        """Calibrate models on a dataset, in the background. Poll GET /v1/jobs/{id}."""
        check_job_inputs(req.dataset, req.models)
        return jobs.submit("calibrate", req.model_dump())

    @app.post("/v1/calibrations/check", tags=["calibration"], status_code=202, dependencies=[Depends(admin)])
    async def start_check(req: CheckRequest):
        """Check active calibrations against new labelled data (drift), in the background."""
        models = req.models or [m for m in manager.store.models() if manager.store.active(m)]
        if not models:
            raise HTTPException(409, "No model has an active calibration")
        check_job_inputs(req.dataset, models)
        return jobs.submit("check", {**req.model_dump(), "models": models})

    @app.get("/v1/jobs", tags=["calibration"], dependencies=[Depends(user)])
    async def list_jobs():
        return {"jobs": [{k: j[k] for k in ("id", "kind", "status", "created")} for j in jobs.all()]}

    @app.get("/v1/jobs/{job_id}", tags=["calibration"], dependencies=[Depends(user)])
    async def get_job(job_id: str):
        if (job := jobs.get(job_id)) is None:
            raise HTTPException(404, f"Unknown job {job_id!r}")
        return to_jsonable(job)

    def calibration_status(model: str) -> dict:
        store, active = manager.store, manager.store.active(model)
        versions = []
        for n in store.versions(model):
            cal, report = store.version(model, n)
            versions.append({"version": n, "created": report.get("created", cal.created),
                             "dataset": report.get("dataset_name"), "n_real": len(cal.real_scores),
                             "n_fake": len(cal.fake_scores), "auc": report.get("auc"), "alpha": cal.alpha,
                             "score_type": cal.score_type, "warnings": report.get("warnings", [])})
        in_use = manager.calibration(model)
        return {"model": model, "active_version": active[0] if active else None,
                "in_use": in_use[0] if in_use else None, "versions": versions}

    @app.get("/v1/calibrations", tags=["calibration"], dependencies=[Depends(user)])
    async def list_calibrations():
        models = sorted(set(manager.store.models()) | set(config.calibrations))
        return {"calibrations": to_jsonable([calibration_status(m) for m in models])}

    @app.get("/v1/calibrations/{model}", tags=["calibration"], dependencies=[Depends(user)])
    async def get_calibration(model: str):
        model_or_404(model)
        return to_jsonable(calibration_status(model))

    @app.get("/v1/calibrations/{model}/versions/{version}", tags=["calibration"], dependencies=[Depends(user)])
    async def get_calibration_version(model: str, version: int):
        """The fitted table (usable with forensix.load(..., calibration=...)) and its report."""
        if (found := manager.store.version(model, version)) is None:
            raise HTTPException(404, f"{model} has no calibration version {version}")
        return to_jsonable({"table": found[0].to_dict(), "report": found[1]})

    @app.put("/v1/calibrations/{model}/active", tags=["calibration"], dependencies=[Depends(admin)])
    async def set_active_calibration(model: str, req: ActiveRequest):
        """Switch the version used for predictions (roll back, or null for raw scores)."""
        model_or_404(model)
        try:
            manager.store.set_active(model, req.version)
        except KeyError as e:
            raise HTTPException(404, str(e)) from None
        return calibration_status(model)

    return app


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("forensix")
    except Exception:
        return "unknown"

