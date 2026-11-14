# -*- coding: utf-8 -*-
"""阶段 7：Gradio 交互演示系统启动入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app.gradio_app import build_demo  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="启动 CFBL Gradio 交互演示系统")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = build_demo()
    demo.launch(server_name=args.server_name, server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
