"""Run the ForensiX HTTP server.

    python -m forensix.server --device cuda --preload dfdet-clip commfor nes2net-x-itw
    python -m forensix.server --config server.json --port 8080

Settings come from defaults, then ``--config`` (JSON), then ``FORENSIX_SERVER_*``
environment variables, then these flags. One process serves one device: to use
several GPUs, run one server per GPU (``CUDA_VISIBLE_DEVICES``) behind a load balancer.
"""

import argparse
import logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", help="JSON file with ServerConfig fields")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", help="cpu, cuda or cuda:N")
    parser.add_argument("--preload", nargs="*", help="models to load at startup")
    parser.add_argument("--max-loaded", type=int, help="models kept in memory at once")
    parser.add_argument("--no-warmup", dest="warmup", action="store_false", default=None)
    parser.add_argument("--api-key", help="require this key (or set FORENSIX_SERVER_API_KEY)")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    import uvicorn

    from forensix.server import ServerConfig, create_app

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = ServerConfig.load(args.config, device=args.device, preload=args.preload,
                               max_loaded=args.max_loaded, warmup=args.warmup, api_key=args.api_key)
    # one worker process: models are loaded once and shared by all requests through threads
    uvicorn.run(create_app(config), host=args.host, port=args.port, log_level=args.log_level, workers=1)


if __name__ == "__main__":
    main()
