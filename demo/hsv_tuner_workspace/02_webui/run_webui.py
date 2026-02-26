#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from hsv_tuner.server import run_server


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HSV tuning WebUI")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8765, help="Bind port")
    parser.add_argument(
        "--workspace",
        type=str,
        default=str(Path(__file__).resolve().parents[0]),
        help="Workspace root containing profiles/exports/snapshots",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_server(host=args.host, port=args.port, workspace_root=Path(args.workspace))


if __name__ == "__main__":
    main()
