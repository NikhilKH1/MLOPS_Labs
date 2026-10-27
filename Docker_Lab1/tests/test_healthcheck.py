"""Run healthcheck.py the way Docker does (as a process) and check its exit code."""
import os
import socket
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest


@pytest.fixture
def run_healthcheck(app_dir):
    def run(directory, port):
        env = {**os.environ, "OUTPUT_DIR": str(directory), "STREAMLIT_SERVER_PORT": str(port)}
        return subprocess.run([sys.executable, str(app_dir / "healthcheck.py")],
                              env=env, capture_output=True, text=True, timeout=30)
    return run


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def fake_streamlit():
    """A tiny server that answers /_stcore/health with "ok", like Streamlit does."""
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200 if self.path == "/_stcore/health" else 404)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.server_address[1]
    server.shutdown()


def test_unhealthy_without_artifacts(tmp_path, run_healthcheck):
    result = run_healthcheck(tmp_path, free_port())
    assert result.returncode == 1
    assert "missing model.joblib, metrics.json" in result.stderr


def test_unhealthy_when_server_is_down(trained_dir, run_healthcheck):
    result = run_healthcheck(trained_dir, free_port())
    assert result.returncode == 1
    assert "not reachable" in result.stderr


def test_healthy_with_artifacts_and_server(trained_dir, run_healthcheck, fake_streamlit):
    result = run_healthcheck(trained_dir, fake_streamlit)
    assert result.returncode == 0, result.stderr
    assert "healthy" in result.stdout
