from __future__ import annotations

import argparse

from .config import ConfigError, load_config
from .server import serve


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m pipeline")
    parser.add_argument("--config", default=None, help="path to pipeline config JSON")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        parser.error(str(exc))
    serve(cfg, args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
