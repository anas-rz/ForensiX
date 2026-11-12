"""HTTP API for ForensiX detectors (FastAPI).

Models stay loaded between requests, so after the first load (or a preload at
startup) a request pays only for inference. See docs/server.md.
"""

import asyncio
import json
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
from fastapi.responses import FileResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from typing import Literal

from pydantic import BaseModel, Field

import forensix
from forensix.server.calibration import Jobs
from forensix.server.config import ServerConfig
from forensix.server.manager import NEEDS_WEIGHTS, ModelManager
from forensix.server.media import MediaInfo, applicable, bound_video, probe, sha256_file
from forensix.server.telemetry import Telemetry

log = logging.getLogger("forensix.server")

VERDICTS = {("fake",): "fake", ("real",): "real", ("real", "fake"): "uncertain", (): "out_of_distribution"}


def confidence(pred, calibration: str) -> dict:
    """The calibrated reading of a prediction (see docs/server.md#confidence)."""
    d = pred.details
    return {"fake_probability": to_jsonable(pred.fake_prob), "verdict": VERDICTS[tuple(d["prediction_set"])],
            "prediction_set": d["prediction_set"], "p_real": d["p_real"], "p_fake": d["p_fake"],
            "alpha": d["alpha"], "calibration": calibration}


class CalibrateRequest(BaseModel):
    mode: Literal["full", "incremental", "refit"] = Field(
        "full", description="full: from scratch on the datasets; incremental: the base version's items plus the "
        "datasets; refit: the base version's items with new settings, nothing scored")
    datasets: list[str] = Field(default_factory=list, description="Dataset ids from POST /v1/datasets")
    dataset: str | None = Field(None, description="One dataset id (same as datasets=[id])")
    models: list[str] = Field(min_length=1)
    base_version: int | None = Field(None, description="incremental/refit: version to build on "
                                     "(default: the active one, else the latest)")
    window: int | None = Field(None, ge=2, description="incremental: max items kept per class; all new items "
                               "stay and older ones are subsampled, so the calibration follows changing data")
    only_if_drift: bool = Field(False, description="incremental: refit only if the new data fails the drift "
                                "check against the base version")
    level: float = Field(0.01, gt=0, lt=1, description="Significance level of the drift tests (only_if_drift)")
    alpha: float | None = Field(None, gt=0, lt=1, description="Target false-alarm rate on real media "
                                "(default: the base version's, else 0.05)")
    folds: int = Field(5, ge=2, le=20)
    seed: int = 0
    activate: bool = Field(True, description="Use the new calibration for predictions right away")
    score_types: dict[str, Literal["raw", "probability"]] = Field(
        default_factory=dict, description='"raw" for models whose score is not a probability (auto-detected)')

    def dataset_ids(self) -> list[str]:
        return list(dict.fromkeys(self.datasets + ([self.dataset] if self.dataset else [])))


class CheckRequest(BaseModel):
    datasets: list[str] = Field(default_factory=list)
    dataset: str | None = None
    models: list[str] | None = Field(None, description="Default: every model with an active calibration")
    level: float = Field(0.01, gt=0, lt=1, description="Significance level of the drift tests")

    def dataset_ids(self) -> list[str]:
        return list(dict.fromkeys(self.datasets + ([self.dataset] if self.dataset else [])))


class FairnessRequest(BaseModel):
    datasets: list[str] = Field(default_factory=list, description="Datasets with demographic columns "
                                "(skin_tone, age, gender, attr_*); real media at least")
    dataset: str | None = None
    models: list[str] = Field(min_length=1)
    estimate_skin_tone: bool = Field(True, description="Estimate skin tone from photos (ITA thirds) when the "
                                     "dataset has no skin_tone column")
    min_group: int = Field(30, ge=5, description="Smallest group compared in disparity tests")
    attributes: list[str] | None = Field(None, description="Default: skin_tone, age, gender, race")

    def dataset_ids(self) -> list[str]:
        return list(dict.fromkeys(self.datasets + ([self.dataset] if self.dataset else [])))


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
    telemetry = Telemetry()
    state = {"ready": False, "started": time.time()}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        for name in config.preload:
            log.info("preloading %s", name)
            await run_in_threadpool(manager.get, name)
        lazy = sorted({m for names in config.default_models.values() for m in names} - set(config.preload))
        if lazy:
            log.warning("default models not preloaded (loaded, and downloaded if needed, on first request): %s",
                        ", ".join(lazy))
        state["ready"] = True
        telemetry.start_sampler(lambda: {"models_loaded": len(manager.loaded())})
        yield
        telemetry.stop()

    app = FastAPI(title="ForensiX", version=_version(), lifespan=lifespan,
                  description="ForensiX: detect deepfakes and AI-generated images, video and voices, verify provenance, "
                              "calibrate to your false-alarm budget and audit fairness.")
    app.state.manager = manager
    app.state.telemetry = telemetry
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

    def shared_writer(request: Request) -> None:
        """Endpoints that change shared state: off on inference replicas (admin_endpoints=false)."""
        if not config.admin_endpoints:
            raise HTTPException(409, "This instance serves predictions only; datasets, calibrations and "
                                     "identities are managed by the manager instance")
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

    @app.get("/v1/server", tags=["service"], dependencies=[Depends(user)])
    async def server_info():
        """Server settings that clients and the UI need (no secrets)."""
        return {"version": _version(), "device": config.device, "max_loaded": config.max_loaded,
                "preload": config.preload, "default_models": config.default_models,
                "thresholds": config.thresholds, "fusions": sorted(config.fusions),
                "trained_weights": sorted(config.weights), "auth": bool(config.api_key),
                "admin_auth": bool(config.admin_key or config.api_key),
                "calibration_persistent": bool(config.calibration_dir),
                "admin_endpoints": config.admin_endpoints,
                "gallery_persistent": bool(config.gallery_dir),
                "limits": {"max_upload_mb": config.max_upload_mb, "max_dataset_mb": config.max_dataset_mb,
                           "max_video_seconds": config.max_video_seconds,
                           "max_video_height": config.max_video_height},
                "ready": state["ready"], "uptime_s": round(time.time() - state["started"], 1)}

    benchmarks_cache: dict = {}

    @app.get("/v1/benchmarks", tags=["service"], dependencies=[Depends(user)])
    async def benchmarks(model: str | None = None):
        """Benchmark results found in ``benchmarks_dir``: one row per (result file, model)."""
        if "rows" not in benchmarks_cache:
            benchmarks_cache["rows"] = await run_in_threadpool(read_benchmarks, config.benchmarks_dir)
        rows = benchmarks_cache["rows"]
        return {"benchmarks": [r for r in rows if model is None or r["model"] == model]}

    @app.get("/v1/telemetry", tags=["service"], dependencies=[Depends(user)])
    async def get_telemetry(window: int = Query(3600, description="Seconds: 900, 3600, 21600 or 86400"),
                            model: str | None = None):
        """Recent predictions and resource use of this instance, bucketed for the dashboard."""
        if model is not None:
            model_or_404(model)
        return to_jsonable(await run_in_threadpool(telemetry.summary, window, model))

    leaderboard_cache: dict = {}

    def leaderboard_access(request: Request) -> None:
        if not config.public_leaderboard:
            check_key(request)

    @app.get("/v1/leaderboard", tags=["leaderboard"], dependencies=[Depends(leaderboard_access)])
    async def get_leaderboard():
        """Every model on commercially usable benchmarks: AUC with 95% CIs, EER, false-alarm and detection
        rates at a fixed threshold, speed. Computed from results/ when present, else the packaged snapshot."""
        if "board" not in leaderboard_cache:
            leaderboard_cache["board"] = await run_in_threadpool(load_leaderboard)
        return to_jsonable(leaderboard_cache["board"])

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
                        details: bool, segments: bool, original: Path | None = None) -> dict:
        family = manager.family(name)
        if family == "provenance" and original is not None:
            path = original  # credentials, watermarks and metadata live in the file as uploaded
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
            result = {"status": "error", "error": str(e)[:500], "seconds": round(time.perf_counter() - start, 3)}
            telemetry.record(name, family, info.modality, result)
            return result
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
        if family == "provenance":  # evidence, not a score: the label is the check's own
            d = pred.details
            result.update(label=d.get("label"), provenance={k: to_jsonable(d.get(k)) for k in ("verdict", "summary", "findings")})
        telemetry.record(name, family, info.modality, result)
        if segments:
            result["segments"] = to_jsonable(pred.segment_fake_probs)
        if details:
            result["details"] = to_jsonable(pred.details)
        return result

    async def prepare(upload: UploadFile, tmp: Path) -> tuple[Path, str, MediaInfo, dict | None, Path]:
        """Save, hash, probe and bound an upload: ``(path, sha256, info, bounded, original)``; ``original``
        is the file as uploaded, ``path`` the one models score (re-encoded if it was too long or large)."""
        raw = await save_upload(upload, tmp)
        sha = await run_in_threadpool(sha256_file, raw)
        try:
            info = await run_in_threadpool(probe, raw)
        except ValueError as e:
            raise HTTPException(415, str(e)) from None
        original = path = raw.rename(raw.with_suffix(info.suffix(Path(upload.filename or "").suffix)))
        bounded = None
        if info.kind == "video":
            path, bounded = await run_in_threadpool(bound_video, path, info, config.max_video_seconds,
                                                    config.max_video_height, tmp)
        return path, sha, info, bounded, original

    async def detect_one(upload: UploadFile, models: list[str] | None, identity: str | None,
                         details: bool, segments: bool) -> dict:
        start = time.perf_counter()
        with tempfile.TemporaryDirectory(dir=config.tmp_dir) as tmp:
            path, sha, info, bounded, original = await prepare(upload, Path(tmp))
            names = models or config.default_models.get(info.modality, [])
            results = await asyncio.gather(*(run_model(n, path, sha, info, identity, details, segments, original)
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

    @app.post("/v1/identities/{identity}", tags=["identities"], dependencies=[Depends(shared_writer)])
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

    @app.delete("/v1/identities/{identity}", tags=["identities"], dependencies=[Depends(shared_writer)])
    async def remove_identity(identity: str, model: str = Query("id-face")):
        removed = await run_in_threadpool(manager.remove_identity, identity_model(model), identity)
        if not removed:
            raise HTTPException(404, f"{identity!r} is not enrolled in {model}")
        return {"model": model, "identity": identity, "removed": True}

    # -- calibration -------------------------------------------------------------

    @app.post("/v1/datasets", tags=["calibration"], status_code=201, dependencies=[Depends(shared_writer)])
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

    @app.delete("/v1/datasets/{dataset_id}", tags=["calibration"], dependencies=[Depends(shared_writer)])
    async def delete_dataset(dataset_id: str):
        if not manager.store.delete_dataset(dataset_id):
            raise HTTPException(404, f"Unknown dataset {dataset_id!r}")
        return {"id": dataset_id, "deleted": True}

    def check_job_inputs(datasets: list[str], models: list[str]) -> None:
        for d in datasets:
            if manager.store.dataset_info(d) is None:
                raise HTTPException(404, f"Unknown dataset {d!r}")
        for m in models:
            if model_or_404(m) == "provenance":
                raise HTTPException(422, f"{m} reads evidence in the file; provenance checks aren't calibrated")

    @app.post("/v1/calibrations", tags=["calibration"], status_code=202, dependencies=[Depends(shared_writer)])
    async def start_calibration(req: CalibrateRequest):
        """Calibrate models in the background (full, incremental or refit). Poll GET /v1/jobs/{id}."""
        datasets = req.dataset_ids()
        if req.mode in ("full", "incremental") and not datasets:
            raise HTTPException(422, f"mode {req.mode!r} needs at least one dataset")
        if req.mode == "refit" and datasets:
            raise HTTPException(422, "mode 'refit' reuses the base version's items; use 'incremental' to add datasets")
        if req.mode == "full" and (req.base_version is not None or req.window or req.only_if_drift):
            raise HTTPException(422, "base_version, window and only_if_drift apply to incremental/refit, not full")
        check_job_inputs(datasets, req.models)
        if req.mode != "full":
            for m in req.models:
                if not manager.store.versions(m):
                    raise HTTPException(409, f"{m} has no calibration version to build on; run mode 'full' first")
                if req.base_version is not None and req.base_version not in manager.store.versions(m):
                    raise HTTPException(404, f"{m} has no calibration version {req.base_version}")
        params = req.model_dump(exclude={"dataset"})
        params["datasets"] = datasets
        return jobs.submit("calibrate", params)

    @app.post("/v1/calibrations/check", tags=["calibration"], status_code=202, dependencies=[Depends(shared_writer)])
    async def start_check(req: CheckRequest):
        """Check active calibrations against new labelled data (drift), in the background."""
        models = req.models or [m for m in manager.store.models() if manager.store.active(m)]
        if not models:
            raise HTTPException(409, "No model has an active calibration")
        datasets = req.dataset_ids()
        if not datasets:
            raise HTTPException(422, "Give at least one dataset")
        check_job_inputs(datasets, models)
        return jobs.submit("check", {**req.model_dump(exclude={"dataset"}), "models": models, "datasets": datasets})

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
                             "mode": report.get("mode", "full"), "parent": report.get("parent"),
                             "datasets": [x.get("name") or x.get("dataset") for x in report.get("lineage", [])],
                             "n_real": len(cal.real_scores), "n_fake": len(cal.fake_scores),
                             "n_new": report.get("n_new"), "n_carried": report.get("n_carried"),
                             "auc": report.get("auc"), "alpha": cal.alpha, "score_type": cal.score_type,
                             "warnings": report.get("warnings", [])})
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

    @app.post("/v1/fairness", tags=["fairness"], status_code=202, dependencies=[Depends(shared_writer)])
    async def start_fairness(req: FairnessRequest):
        """Error rates by demographic group for the given models on the given datasets, in the background."""
        datasets = req.dataset_ids()
        if not datasets:
            raise HTTPException(422, "Give at least one dataset")
        for d in datasets:
            if manager.store.dataset_info(d) is None:
                raise HTTPException(404, f"Unknown dataset {d!r}")
        for m in req.models:
            if model_or_404(m) == "provenance":
                raise HTTPException(422, f"{m} reads evidence in the file; fairness reports are for detectors")
        return jobs.submit("fairness", {**req.model_dump(exclude={"dataset"}), "datasets": datasets})

    @app.get("/v1/fairness", tags=["fairness"], dependencies=[Depends(user)])
    async def list_fairness():
        return {"reports": to_jsonable(manager.store.fairness_reports())}

    @app.get("/v1/fairness/{report_id}", tags=["fairness"], dependencies=[Depends(user)])
    async def get_fairness(report_id: str, format: Literal["json", "markdown"] = "json"):
        if (rep := manager.store.fairness(report_id)) is None:
            raise HTTPException(404, f"Unknown fairness report {report_id!r}")
        if format == "markdown":
            from forensix.fairness import to_markdown

            names = ", ".join(d.get("name") or d["id"] for d in rep.get("datasets", []))
            return PlainTextResponse(to_markdown(rep, f"Fairness report {report_id} ({names}, {rep.get('created')})"),
                                     media_type="text/markdown")
        return to_jsonable(rep)

    @app.put("/v1/calibrations/{model}/active", tags=["calibration"], dependencies=[Depends(shared_writer)])
    async def set_active_calibration(model: str, req: ActiveRequest):
        """Switch the version used for predictions (roll back, or null for raw scores)."""
        model_or_404(model)
        try:
            manager.store.set_active(model, req.version)
        except KeyError as e:
            raise HTTPException(404, str(e)) from None
        return calibration_status(model)

    if config.metrics:
        @app.get("/metrics", include_in_schema=False)
        async def metrics():
            """Prometheus text format: server, per-model and job counters."""
            return PlainTextResponse(prometheus_metrics(manager, jobs, state), media_type="text/plain; version=0.0.4")

    if config.ui:
        static = Path(__file__).parent / "static"
        export_lock = asyncio.Lock()

        @app.get("/ui/studio/models/ecapa.onnx", include_in_schema=False)
        async def studio_ecapa():
            """The voice encoder for the browser studio, exported once from the id-voice checkpoint."""
            from forensix.hub import cache_dir
            from forensix.identity.anomaly import export_ecapa_onnx

            path = cache_dir() / "studio" / "ecapa.onnx"
            async with export_lock:
                if not path.exists():
                    await asyncio.to_thread(export_ecapa_onnx, path)
            return FileResponse(path, media_type="application/octet-stream",
                                headers={"Cache-Control": "public, max-age=604800"})

        app.mount("/ui", StaticFiles(directory=static, html=True), name="ui")

        @app.get("/", include_in_schema=False)
        async def root():
            return RedirectResponse("/ui/")

    return app


def prometheus_metrics(manager, jobs, state) -> str:
    """Counters and gauges in the Prometheus exposition format (text 0.0.4)."""
    lines: list[str] = []

    def metric(name: str, kind: str, help_: str, samples: list[tuple[dict, float]]) -> None:
        lines.append(f"# HELP forensix_{name} {help_}")
        lines.append(f"# TYPE forensix_{name} {kind}")
        for labels, value in samples:
            tag = ",".join(f'{k}="{str(v).replace(chr(92), chr(92) * 2).replace(chr(34), chr(92) + chr(34))}"'
                           for k, v in labels.items())
            lines.append(f"forensix_{name}{{{tag}}} {float(value):.6g}" if tag else f"forensix_{name} {float(value):.6g}")

    stats = manager.stats()
    metric("ready", "gauge", "1 when preloaded models are in memory", [({}, 1 if state["ready"] else 0)])
    metric("uptime_seconds", "gauge", "Seconds since the server started", [({}, time.time() - state["started"])])
    metric("models_loaded", "gauge", "Models in memory", [({}, len(stats))])
    per = lambda key: [({"model": n, "family": m["family"]}, m[key] or 0) for n, m in stats.items()]  # noqa: E731
    metric("requests_total", "counter", "Predictions computed per model (cache misses)", per("calls"))
    metric("errors_total", "counter", "Predictions that raised per model", per("errors"))
    metric("cache_hits_total", "counter", "Predictions served from the result cache", per("cache_hits"))
    metric("busy_seconds_total", "counter", "Seconds spent computing predictions per model", per("busy_seconds"))
    metric("load_seconds", "gauge", "Seconds the model took to load", per("load_seconds"))
    metric("batches_total", "counter", "Dynamic batches run per model", per("batches"))
    metric("batched_items_total", "counter", "Items scored in dynamic batches per model", per("batched_items"))
    by_status: dict[str, int] = {}
    for j in jobs.all():
        by_status[j["status"]] = by_status.get(j["status"], 0) + 1
    metric("jobs", "gauge", "Calibration jobs since start, by status",
           [({"status": s}, by_status.get(s, 0)) for s in ("queued", "running", "done", "failed")])
    active = [({"model": m}, a[0]) for m in manager.store.models() if (a := manager.store.active(m))]
    metric("calibration_active_version", "gauge", "Active calibration version per model", active)
    return "\n".join(lines) + "\n"


def load_leaderboard() -> dict:
    """Live from the repo's results/ (development), else the snapshot built by scripts/build_leaderboard.py."""
    from forensix import leaderboard

    root = leaderboard.default_root()
    if (root / "leaderboard.json").is_file():
        return {**leaderboard.build(root), "source": "results/ (computed live)"}
    snapshot = Path(__file__).parent / "static" / "leaderboard.json"
    if snapshot.is_file():
        return {**json.loads(snapshot.read_text()), "source": "packaged snapshot"}
    return {"title": "ForensiX leaderboard", "benchmarks": [], "excluded": [], "source": "none"}


BENCHMARK_METRICS = ("auc", "ap", "eer", "eer_percent", "balanced_acc", "real_acc", "fake_acc",
                     "accuracy_at_default_threshold", "n_real", "n_fake", "seconds")


def read_benchmarks(directory: str | None) -> list[dict]:
    """Rows ``{source, dataset, model, metrics}`` from result JSONs whose ``models`` maps
    model names to metric dicts (the format of scripts/eval_*.py)."""
    if directory == "":
        return []
    root = Path(directory) if directory else Path(forensix.__file__).resolve().parents[1] / "results"
    if not root.is_dir():
        return []
    rows = []
    for path in sorted(root.rglob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        models = data.get("models") if isinstance(data, dict) else None
        if not isinstance(models, dict):
            continue
        for name, m in models.items():
            if not isinstance(m, dict):
                continue
            metrics = {k: m[k] for k in BENCHMARK_METRICS if isinstance(m.get(k), (int, float))}
            if "auc" in metrics or "eer" in metrics or "eer_percent" in metrics:
                rows.append({"source": str(path.relative_to(root)), "dataset": data.get("dataset") or path.stem,
                             "model": name, "metrics": metrics})
    return rows


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("forensix")
    except Exception:
        return "unknown"

