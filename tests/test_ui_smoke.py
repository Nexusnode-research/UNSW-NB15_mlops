"""
UI smoke test — verifies the Dash app starts and serves HTTP 200 on /.

Runs the app in a subprocess, polls until ready (max 30s), then checks the
root path. Always tears down the subprocess regardless of outcome.

Skipped automatically when IDS_API_URL is not set or when running in
environments without network access to the target API (the Dash app needs
IDS_API_URL to import cleanly in some configurations).
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time

import pytest
import requests

UI_PORT = 18050  # non-standard port to avoid clashing with a real running UI
STARTUP_TIMEOUT = 30  # seconds


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


@pytest.mark.skipif(
    not os.getenv("IDS_API_URL"),
    reason="IDS_API_URL not set — UI smoke test requires a reachable API",
)
def test_dash_ui_serves_200():
    env = {
        **os.environ,
        "IDS_API_URL": os.environ.get("IDS_API_URL", "http://localhost:8000"),
        "IDS_API_TOKEN": os.environ.get("IDS_API_TOKEN", "smoke-test-token"),
    }

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "gunicorn",
            "-b",
            f"0.0.0.0:{UI_PORT}",
            "--timeout",
            "30",
            "ids_unsw.ui.app_dash:server",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        deadline = time.time() + STARTUP_TIMEOUT
        while time.time() < deadline:
            if _port_open(UI_PORT):
                break
            if proc.poll() is not None:
                out, err = proc.communicate()
                pytest.fail(
                    f"UI process exited before becoming ready.\n"
                    f"stdout: {out.decode()}\nstderr: {err.decode()}"
                )
            time.sleep(1)
        else:
            proc.terminate()
            pytest.fail(f"UI did not become ready within {STARTUP_TIMEOUT}s")

        r = requests.get(f"http://127.0.0.1:{UI_PORT}/", timeout=10)
        assert r.status_code == 200, (
            f"Dash UI returned {r.status_code}, expected 200"
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
