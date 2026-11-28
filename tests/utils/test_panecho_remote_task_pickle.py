# tests/utils/test_panecho_remote_task_pickle.py

"""The PanEcho evaluation path must not run code from a task list fetched over the network.

`evals/video_classification_frozen/modelcustom/PanEcho/hubconf.py` loads its task
list with `pandas.read_pickle` from the upstream project's unpinned `main`
branch. These tests substitute that HTTP response with a pickle whose reduction
writes a file, then start the encoder the way a PanEcho eval config does.
They fail while that response is unpickled.
"""

import os
import pickle
import socket
import ssl
import tempfile
import threading
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pandas
import yaml

from evals.video_classification_frozen.models import init_module

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PANECHO_CONFIG = os.path.join(REPO, "configs", "eval", "vitg-384", "lvef", "panecho_lvef.yaml")
UPSTREAM_TASKS_URL = "https://github.com/CarDS-Yale/PanEcho/blob/main/content/tasks.pkl?raw=true"


def _marker_pickle(path):
    """Pickle that writes `path` when `pickle.load` reduces it."""

    class Marker:
        def __reduce__(self):
            return (eval, (f"__import__('pathlib').Path({path!r}).write_text('executed')",))

    return pickle.dumps(Marker(), protocol=4)


def _request_url(url):
    if isinstance(url, urllib.request.Request):
        return url.full_url
    return url


def _is_remote(value):
    return isinstance(value, str) and value.split(":", 1)[0] in ("http", "https")


class _TaskListServer:
    """Serves the same attacker-controlled body for every request, including a second fetch."""

    def __init__(self, body):
        handler = self._handler(body)
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.port = self._httpd.server_address[1]
        self.requests = 0
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def _handler(self, body):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                server.requests += 1
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, fmt, *args):
                return

        return Handler

    def close(self):
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)


class PanEchoRemoteTaskPickleTest(unittest.TestCase):
    def test_eval_config_does_not_unpickle_the_upstream_task_list(self):
        with open(PANECHO_CONFIG) as handle:
            params = yaml.load(handle, Loader=yaml.FullLoader)
        model_kwargs = params["model_kwargs"]
        data = params["experiment"]["data"]
        sentinel_dir = tempfile.mkdtemp()
        sentinel = os.path.join(sentinel_dir, "executed")
        task_urls = []
        remote_pickles = []
        server = _TaskListServer(_marker_pickle(sentinel))
        original_urlopen = urllib.request.urlopen
        original_read_pickle = pandas.read_pickle
        original_connect = socket.create_connection
        original_https = ssl._create_default_https_context

        def urlopen(url, data=None, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, *,
                    cafile=None, capath=None, cadefault=False, context=None):
            target = _request_url(url)
            if isinstance(target, str) and (target == UPSTREAM_TASKS_URL or "tasks.pkl" in target):
                task_urls.append(target)
                return original_urlopen(f"http://127.0.0.1:{server.port}/tasks.pkl", timeout=timeout)
            return original_urlopen(
                url, data, timeout, cafile=cafile, capath=capath, cadefault=cadefault, context=context)

        def read_pickle(filepath_or_buffer, *args, **kwargs):
            if _is_remote(filepath_or_buffer):
                remote_pickles.append(filepath_or_buffer)
            return original_read_pickle(filepath_or_buffer, *args, **kwargs)

        def create_connection(address, *args, **kwargs):
            host = address[0]
            if host in ("127.0.0.1", "::1", "localhost"):
                return original_connect(address, *args, **kwargs)
            raise OSError(f"blocked connection to {host}")

        urllib.request.urlopen = urlopen
        pandas.read_pickle = read_pickle
        socket.create_connection = create_connection
        try:
            try:
                init_module(
                    module_name=model_kwargs["module_name"],
                    device="cpu",
                    frames_per_clip=data["frames_per_clip"],
                    resolution=data["resolution"],
                    checkpoint=model_kwargs["checkpoint"],
                    model_kwargs=model_kwargs["pretrain_kwargs"],
                    wrapper_kwargs=model_kwargs["wrapper_kwargs"],
                )
            except (ImportError, ModuleNotFoundError):
                raise
            except Exception:
                pass
            self.assertFalse(
                os.path.exists(sentinel),
                f"unpickling the substituted task list ran code; requests={task_urls}",
            )
            self.assertEqual(task_urls, [])
            self.assertEqual(remote_pickles, [])
            self.assertEqual(server.requests, 0)
        finally:
            urllib.request.urlopen = original_urlopen
            pandas.read_pickle = original_read_pickle
            socket.create_connection = original_connect
            ssl._create_default_https_context = original_https
            server.close()


if __name__ == "__main__":
    unittest.main()
