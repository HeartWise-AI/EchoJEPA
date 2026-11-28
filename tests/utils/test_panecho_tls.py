# tests/utils/test_panecho_tls.py

"""The PanEcho probe encoder must not turn off TLS verification for the downloads it makes.

The committed PanEcho probe configs select this encoder. An evaluation reaches it through
`evals.video_classification_frozen.models.init_module`, which the probe runner calls before it
opens a study table. `PanEcho/hubconf.py` then fetches the task list and the weight file.
These tests serve those URLs with a certificate default verification rejects. They fail while
either response is accepted, or while importing the encoder leaves certificate checks off.
"""

import json
import os
import pickle
import re
import socket
import ssl
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
HUBCONF = ROOT / "evals/video_classification_frozen/modelcustom/PanEcho/hubconf.py"
CONFIG = ROOT / "configs/eval/vitg-384/lvef/panecho_lvef.yaml"


def _remote_hosts():
    return {
        urllib.parse.urlparse(url).hostname
        for url in re.findall(r"""https://[^'"\s]+""", HUBCONF.read_text())
    }


def _code_payload(marker):
    class _Run:
        def __reduce__(self):
            return (eval, (f"open({str(marker)!r}, 'w').write('executed')",))

    return pickle.dumps(_Run())


def _benign_tasks():
    import numpy as np

    return pickle.dumps({"ef": {"task_type": "regression", "class_names": np.array(["x"]), "mean": 0.0}})


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.server.paths.append(self.path)
        if "tasks.pkl" in self.path:
            body = self.server.tasks_body
        elif "panecho.pt" in self.path:
            body = self.server.weights_body
        else:
            body = None
        if body is None:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Type", "application/octet-stream")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        return


class _Server(HTTPServer):
    def handle_error(self, request, client_address):
        self.errors += 1


def _certificate(directory):
    cert, key = directory / "cert.pem", directory / "key.pem"
    subprocess.run(
        [
            "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
            "-keyout", str(key), "-out", str(cert), "-days", "1",
            "-subj", "/CN=github.com",
            "-addext", "subjectAltName=DNS:github.com",
        ],
        check=True, capture_output=True,
    )
    return cert, key


def _child():
    port = int(sys.argv[2])
    hosts = set(filter(None, sys.argv[3].split(",")))
    marker = Path(sys.argv[4])
    result_path = Path(sys.argv[5])
    stub_backbone = sys.argv[6] == "1"
    report = {"opener_was_none": False, "unverified_context": False, "verify_mode": None, "error": "", "marker": False}
    try:
        import urllib.request

        # The probe runner imports this module, and only then builds the encoder.
        import evals.video_classification_frozen.eval  # noqa: F401
        report["opener_was_none"] = urllib.request._opener is None
        import torch
        if stub_backbone:
            import torchvision

            class _Backbone(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.classifier = torch.nn.Sequential(torch.nn.Linear(8, 4))

            # ImageNet weights are loaded before the PanEcho weight URL. This builder skips that
            # public file; `hubconf.PanEcho` still downloads panecho.pt itself.
            torchvision.models.convnext_tiny = lambda *args, **kwargs: _Backbone()
        real_connect = socket.create_connection

        def create_connection(address, *args, **kwargs):
            host = address[0]
            port_ = address[1] if len(address) > 1 else None
            if host in hosts and port_ == 443:
                address = ("127.0.0.1", port)
            elif host not in {"127.0.0.1", "::1", "localhost"}:
                raise OSError(f"blocked connection to {host}:{port_}")
            return real_connect(address, *args, **kwargs)

        socket.create_connection = create_connection
        with CONFIG.open() as handle:
            cfg = yaml.load(handle, Loader=yaml.FullLoader)
        from evals.video_classification_frozen.models import init_module

        data = cfg["experiment"]["data"]
        init_module(
            module_name=cfg["model_kwargs"]["module_name"],
            device=torch.device("cpu"),
            frames_per_clip=data["frames_per_clip"],
            resolution=data["resolution"],
            checkpoint=cfg["model_kwargs"]["checkpoint"],
            model_kwargs=cfg["model_kwargs"]["pretrain_kwargs"],
            wrapper_kwargs=cfg["model_kwargs"]["wrapper_kwargs"],
        )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    report["unverified_context"] = ssl._create_default_https_context is ssl._create_unverified_context
    report["verify_mode"] = int(ssl._create_default_https_context().verify_mode)
    report["marker"] = marker.exists()
    result_path.write_text(json.dumps(report))


class PanEchoTLSTests(unittest.TestCase):
    def _run(self, tasks_body, weights_body, stub_backbone):
        hosts = _remote_hosts()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            cert, key = _certificate(directory)
            marker = directory / "executed"
            # Payloads close over the marker path, so build them after the directory exists.
            if tasks_body is None:
                tasks_body = _code_payload(marker)
            if weights_body is None:
                weights_body = _code_payload(marker)
            result_path = directory / "result.json"
            httpd = _Server(("127.0.0.1", 0), _Handler)
            httpd.paths = []
            httpd.errors = 0
            httpd.tasks_body = tasks_body
            httpd.weights_body = weights_body
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(str(cert), str(key))
            httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
            threading.Thread(target=httpd.serve_forever, daemon=True).start()
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
            env["TORCH_HOME"] = str(directory / "torch")
            try:
                proc = subprocess.run(
                    [
                        sys.executable, __file__, "child", str(httpd.server_address[1]),
                        ",".join(sorted(hosts)), str(marker), str(result_path),
                        "1" if stub_backbone else "0",
                    ],
                    cwd=ROOT, env=env, capture_output=True, text=True, timeout=180,
                )
            finally:
                httpd.shutdown()
                httpd.server_close()
            report = json.loads(result_path.read_text())
            output = proc.stdout + proc.stderr
            with self.subTest("encoder entry"):
                self.assertIn("Initializing PanEcho encoder", output)
                self.assertTrue(report["opener_was_none"])
            with self.subTest("certificate"):
                self.assertFalse(report["unverified_context"], report["error"])
                self.assertNotEqual(report["verify_mode"], int(ssl.CERT_NONE))
                self.assertEqual(httpd.paths, [], f"untrusted certificate was accepted for {httpd.paths}")
            with self.subTest("pickle"):
                self.assertFalse(marker.exists(), "downloaded pickle ran")
                self.assertFalse(report["marker"])

    def test_task_list_from_untrusted_certificate_is_not_unpickled(self):
        self._run(tasks_body=None, weights_body=b"", stub_backbone=False)

    def test_weight_file_from_untrusted_certificate_is_not_unpickled(self):
        self._run(tasks_body=_benign_tasks(), weights_body=None, stub_backbone=True)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "child":
    _child()
elif __name__ == "__main__":
    unittest.main()
