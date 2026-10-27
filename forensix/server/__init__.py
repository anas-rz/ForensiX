"""HTTP server for ForensiX detectors: ``python -m forensix.server`` (needs ``pip install -e ".[server]"``)."""

from forensix.server.config import ServerConfig

__all__ = ["ServerConfig", "create_app"]


def create_app(config: ServerConfig | None = None):
    from forensix.server.app import create_app as _create_app  # imports FastAPI only when used

    return _create_app(config)
