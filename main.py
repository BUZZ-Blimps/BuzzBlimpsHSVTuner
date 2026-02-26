#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BUZZBLIMP HSV tuner WebUI")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8765, help="Bind port")
    parser.add_argument(
        "--workspace",
        type=str,
        default="",
        help="Workspace root containing profiles/exports/snapshots/uploads",
    )
    return parser.parse_args()


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    webui_root = repo_root / "demo" / "hsv_tuner_workspace" / "02_webui"
    workspace = webui_root

    args = parse_args()
    if args.workspace:
        workspace = Path(args.workspace).resolve()

    if str(webui_root) not in sys.path:
        sys.path.insert(0, str(webui_root))

    from hsv_tuner.server import run_server

    run_server(host=args.host, port=args.port, workspace_root=workspace)


if __name__ == "__main__":
    main()
