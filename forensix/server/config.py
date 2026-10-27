"""Server settings: defaults, then a JSON config file, then environment variables, then CLI flags."""

import json
import os
import re
from dataclasses import dataclass, field, fields
from pathlib import Path

# models run when a request doesn't name any, by what the uploaded media contains
DEFAULT_MODELS = {
    "image": ["commfor", "c2pa", "watermark-image", "metadata"],
    "video": ["dfdet-clip", "c2pa", "watermark-image", "metadata"],
    "video+audio": ["dfdet-clip", "nes2net-x-itw", "syncnet", "c2pa", "watermark-audio", "metadata"],
    "audio": ["nes2net-x-itw", "c2pa", "watermark-audio", "metadata"],
}


@dataclass
class ServerConfig:
    device: str = "cpu"
    # loaded (and warmed up) at startup, so the first request doesn't pay for loading
    preload: list[str] = field(default_factory=list)
    warmup: bool = True
    # models kept in memory at once; the least recently used one is unloaded beyond this
    max_loaded: int = 8
    # checkpoints for models you trained, e.g. {"avcons": "/models/avcons.safetensors"}
    weights: dict[str, str] = field(default_factory=dict)
    # decision thresholds that replace a model's default, e.g. {"dfdet-clip": 0.71}
    thresholds: dict[str, float] = field(default_factory=dict)
    # calibration tables (forensix.calibration), e.g. {"syncnet": "/cal/syncnet.json"};
    # a version fitted and activated through /v1/calibrations takes precedence
    calibrations: dict[str, str] = field(default_factory=dict)
    # where calibration data and fitted versions are kept; None keeps them in memory
    calibration_dir: str | None = None
    # model options passed to forensix.load, e.g. {"aasist": {"clip_score": "first"}}
    options: dict[str, dict] = field(default_factory=dict)
    # fusion tables (forensix.fusion) served as models under these names, e.g. {"av-fusion": "/f/av.json"}
    fusions: dict[str, str] = field(default_factory=dict)
    default_models: dict[str, list[str]] = field(default_factory=lambda: {k: list(v) for k, v in DEFAULT_MODELS.items()})
    # dynamic batching of concurrent requests for image and audio models
    max_batch: int = 16
    batch_wait_ms: float = 5.0
    # results kept per (file hash, model); 0 disables the cache
    result_cache_size: int = 10000
    max_upload_mb: float = 500.0
    # labelled datasets uploaded for calibration (ZIP size, and size once extracted)
    max_dataset_mb: float = 20000.0
    # longer or larger videos are re-encoded within these limits before scoring
    max_video_seconds: float = 60.0
    max_video_height: int = 720
    # where identity galleries are saved (one .npz per identity model); None keeps them in memory
    gallery_dir: str | None = None
    # require this key in the X-API-Key header (or Authorization: Bearer); None disables auth
    api_key: str | None = None
    # model loading, unloading and gallery changes need this key when set (else api_key)
    admin_key: str | None = None
    tmp_dir: str | None = None
    # benchmark result JSONs shown in the UI (files with a "models" dict, e.g. from scripts/eval_*.py);
    # None: the repo's results/ folder if present; "" disables
    benchmarks_dir: str | None = None
    # serve the web manager at /ui
    ui: bool = True
    # serve endpoints that change shared state (datasets, calibrations, identity galleries). Turn off on
    # horizontally scaled inference replicas and let one manager instance own those writes; replicas
    # still pick up activated calibrations and gallery changes from the shared calibration_dir/gallery_dir
    admin_endpoints: bool = True
    # Prometheus metrics at /metrics (aggregate counters only, no media or keys)
    metrics: bool = True

    @classmethod
    def load(cls, path: str | Path | None = None, **overrides) -> "ServerConfig":
        """Defaults < JSON file < ``FORENSIX_SERVER_*`` environment variables < ``overrides``."""
        values: dict = {}
        if path:
            values.update(json.loads(Path(path).read_text()))
        for f in fields(cls):
            env = os.environ.get(f"FORENSIX_SERVER_{f.name.upper()}")
            if env is not None:
                values[f.name] = _parse_env(env, f.type)
        values.update({k: v for k, v in overrides.items() if v is not None})
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown server settings: {sorted(unknown)}")
        return cls(**values)


def _parse_env(value: str, type_) -> object:
    text = str(type_)
    if text.startswith(("list", "dict")):
        if text.startswith("list") and not value.lstrip().startswith("["):
            return [v for v in re.split(r"[,\s]+", value) if v]  # "a,b" or "a b"
        return json.loads(value)
    if type_ is bool:
        return value.lower() in {"1", "true", "yes", "on"}
    if type_ is int:
        return int(value)
    if type_ is float:
        return float(value)
    return value
