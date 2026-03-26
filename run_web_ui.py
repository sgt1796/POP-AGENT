from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT / "frontend" / "copilotkit-app"


def _spawn_backend(env: dict[str, str], backend_port: str) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "agent_build.agent1.web.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            backend_port,
        ],
        cwd=str(ROOT),
        env=env,
    )


def _spawn_frontend(env: dict[str, str], frontend_port: str) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [
            "npm",
            "run",
            "dev",
            "--",
            "--port",
            frontend_port,
        ],
        cwd=str(FRONTEND_DIR),
        env=env,
    )


def main() -> int:
    backend_port = os.getenv("POP_AGENT_WEB_PORT", "8000")
    frontend_port = os.getenv("POP_AGENT_FRONTEND_PORT", "3001")
    backend_url = os.getenv("POP_AGENT_API_BASE_URL", f"http://127.0.0.1:{backend_port}")

    env = os.environ.copy()
    env.setdefault("POP_AGENT_API_BASE_URL", backend_url)
    env.setdefault("NEXT_PUBLIC_POP_AGENT_API_BASE_URL", backend_url)

    backend = _spawn_backend(env, backend_port)
    frontend = _spawn_frontend(env, frontend_port)
    children = [backend, frontend]

    try:
        while True:
            for child in children:
                code = child.poll()
                if code is not None:
                    return int(code)
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        for child in children:
            if child.poll() is not None:
                continue
            with contextlib.suppress(ProcessLookupError):
                child.send_signal(signal.SIGINT)
        time.sleep(0.5)
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            with contextlib.suppress(Exception):
                child.wait(timeout=5)
    return 130


if __name__ == "__main__":
    raise SystemExit(main())
