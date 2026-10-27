"""Model registry: every detector registers a ``ModelSpec`` under a unique name."""

from dataclasses import dataclass
from typing import Callable

from forensix.hub import WeightFile


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str  # audio, video, audiovisual, multimodal, physics, image, identity, provenance, specialized, graph
    builder: Callable[..., object]  # (weights_path, device) -> detector
    weights: WeightFile | None
    license: str
    source: str
    paper: str
    description: str
    # other files the builder fetches itself, e.g. a shared pretrained backbone
    extra_weights: tuple[WeightFile, ...] = ()


_REGISTRY: dict[str, ModelSpec] = {}


def register(spec: ModelSpec) -> ModelSpec:
    if spec.name in _REGISTRY:
        raise ValueError(f"Model {spec.name!r} is already registered")
    _REGISTRY[spec.name] = spec
    return spec


def get_spec(name: str) -> ModelSpec:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown model {name!r}. Available: {', '.join(sorted(_REGISTRY))}"
        ) from None


def list_models(family: str | None = None) -> list[str]:
    return sorted(n for n, s in _REGISTRY.items() if family in (None, s.family))
