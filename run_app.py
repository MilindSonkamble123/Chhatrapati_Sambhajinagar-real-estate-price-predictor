from __future__ import annotations

import socket
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
APP_PATH = PROJECT_ROOT / "src" / "app.py"

PREFERRED_PORTS = [8501, 8502, 8503, 8504, 8505, 8506, 8507, 8508, 8510, 9000, 9001, 9002, 9003, 9004, 9005, 9010]


def find_free_port(ports: list[int]) -> int:
    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            result = sock.connect_ex(("127.0.0.1", port))
            if result != 0:
                return port
    return 8506


if __name__ == "__main__":
    port = find_free_port(PREFERRED_PORTS)
    print(f"Starting Real Estate Predictor on http://localhost:{port}")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(APP_PATH),
            "--server.headless",
            "true",
            "--server.port",
            str(port),
        ],
        check=False,
    )
