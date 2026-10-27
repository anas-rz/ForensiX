"""Run the ForensiX HTTP server.

    python -m forensix.server --device cuda --preload dfdet-clip commfor nes2net-x-itw
    python -m forensix.server --config server.json --port 8080
    python -m forensix.server --preload dfdet-clip commfor --warm-only   # fill the weights cache, then exit

Settings come from defaults, then ``--config`` (JSON), then ``FORENSIX_SERVER_*``
environment variables, then these flags. ``PORT`` and ``HOST`` set the bind address
(as Cloud Run and most platforms do). One process serves one device: to use several
GPUs, run one server per GPU (``CUDA_VISIBLE_DEVICES``) behind a load balancer.
"""

import argparse
import json
import logging
import os
import sys
import time


class JsonFormatter(logging.Formatter):
    """One JSON object per line, for cloud log collectors."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
                 "level": record.levelname, "logger": record.name, "message": record.getMessage()}
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry)


def log_config(level: str, fmt: str) -> dict:
    """logging.dictConfig for our loggers and uvicorn's, in text or JSON."""
    formatter = ({"()": JsonFormatter} if fmt == "json"
                 else {"format": "%(asctime)s %(levelname)s %(name)s: %(message)s"})
    handler = {"class": "logging.StreamHandler", "formatter": "default", "stream": "ext://sys.stdout"}
    return {
        "version": 1, "disable_existing_loggers": False,
        "formatters": {"default": formatter},
        "handlers": {"default": handler},
        "root": {"handlers": ["default"], "level": level.upper()},
        "loggers": {name: {"handlers": ["default"], "level": level.upper(), "propagate": False}
                    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "forensix")},
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=os.environ.get("FORENSIX_SERVER_CONFIG"),
                        help="JSON file with ServerConfig fields (or FORENSIX_SERVER_CONFIG)")
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    parser.add_argument("--device", help="cpu, cuda or cuda:N")
    parser.add_argument("--preload", nargs="*", help="models to load at startup")
    parser.add_argument("--max-loaded", type=int, help="models kept in memory at once")
    parser.add_argument("--no-warmup", dest="warmup", action="store_false", default=None)
    parser.add_argument("--api-key", help="require this key (or set FORENSIX_SERVER_API_KEY)")
    parser.add_argument("--warm-only", action="store_true",
                        help="load (and so download and verify) the preload models, then exit; for image "
                             "builds and init containers")
    parser.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "info"))
    parser.add_argument("--log-format", choices=["text", "json"], default=os.environ.get("LOG_FORMAT", "text"))
    parser.add_argument("--graceful-timeout", type=int, default=int(os.environ.get("GRACEFUL_TIMEOUT", 30)),
                        help="seconds in-flight requests get to finish on SIGTERM")
    args = parser.parse_args(argv)

    import logging.config

    from forensix.server import ServerConfig

    logging.config.dictConfig(log_config(args.log_level, args.log_format))
    config = ServerConfig.load(args.config, device=args.device, preload=args.preload,
                               max_loaded=args.max_loaded, warmup=args.warmup, api_key=args.api_key)
    if args.warm_only:
        sys.exit(warm(config))

    import uvicorn

    from forensix.server import create_app

    # one worker process: models are loaded once and shared by all requests through threads
    uvicorn.run(create_app(config), host=args.host, port=args.port, workers=1, log_config=None,
                timeout_graceful_shutdown=args.graceful_timeout, proxy_headers=True, forwarded_allow_ips="*")


def warm(config) -> int:
    """Load every preload model once; non-zero exit if any fails."""
    from forensix.server.manager import ModelManager

    log = logging.getLogger("forensix.server")
    manager, failed = ModelManager(config), []
    for name in config.preload:
        try:
            entry = manager.get(name)
            log.info("warm: %s ready (%.1fs)", name, entry.load_seconds)
        except Exception as e:
            log.error("warm: %s failed: %s", name, e)
            failed.append(name)
    if manager.store._tmp:  # don't leave an empty calibration store behind (e.g. in an image layer)
        import shutil

        shutil.rmtree(manager.store._tmp, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    main()
