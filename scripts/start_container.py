"""Supervise the API and static frontend server in the application container."""

from __future__ import annotations

import signal
import subprocess
import sys
import time
from collections.abc import Sequence


COMMANDS: tuple[tuple[str, ...], ...] = (
    (
        sys.executable,
        "-m",
        "uvicorn",
        "api_layer.main:app",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ),
    ("nginx", "-g", "daemon off;"),
)


def stop_processes(processes: Sequence[subprocess.Popen[bytes]]) -> None:
    """Terminate both services, escalating only if a child does not stop."""

    for process in processes:
        if process.poll() is None:
            process.terminate()

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and any(
        process.poll() is None for process in processes
    ):
        time.sleep(0.1)

    for process in processes:
        if process.poll() is None:
            process.kill()


def main() -> int:
    processes: list[subprocess.Popen[bytes]] = []
    stopping = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    try:
        for command in COMMANDS:
            processes.append(subprocess.Popen(command))

        while not stopping:
            for process in processes:
                return_code = process.poll()
                if return_code is not None:
                    # Either service is essential. Stop the container if one dies.
                    return return_code or 1
            time.sleep(0.25)
        return 0
    finally:
        stop_processes(processes)


if __name__ == "__main__":
    raise SystemExit(main())
