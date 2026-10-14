"""Smoke-test a tunnelled Jupyter server from the laptop: start a kernel, run code on the GPU, shut it down.

Usage (with `cluster/jupyter.sh connect` running in another terminal):
    .venv/bin/python cluster/kernel_check.py 'http://localhost:PORT/?token=TOKEN'
"""

from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
import uuid
from typing import Any

import websocket  # pyright: ignore[reportMissingImports]  # websocket-client, from jupyter-server

CODE = "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name())"


def main() -> None:
    url = urllib.parse.urlparse(sys.argv[1])
    token = urllib.parse.parse_qs(url.query)["token"][0]
    base = f"http://{url.netloc}"
    hdr = {"Authorization": f"token {token}"}

    def api(method: str, path: str) -> Any:
        req = urllib.request.Request(
            base + path,
            method=method,
            headers=hdr,
            data=b"{}" if method == "POST" else None,
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            body = r.read()
        return json.loads(body) if body else None

    print(
        "server:",
        api("GET", "/api"),
        "| kernels before:",
        len(api("GET", "/api/kernels")),
    )
    kid = api("POST", "/api/kernels")["id"]
    print("started kernel", kid)
    try:
        ws = websocket.create_connection(
            f"ws://{url.netloc}/api/kernels/{kid}/channels?token={token}", timeout=120
        )
        msg_id = uuid.uuid4().hex
        ws.send(
            json.dumps(
                {
                    "header": {
                        "msg_id": msg_id,
                        "msg_type": "execute_request",
                        "username": "check",
                        "session": uuid.uuid4().hex,
                        "version": "5.3",
                    },
                    "parent_header": {},
                    "metadata": {},
                    "channel": "shell",
                    "content": {
                        "code": CODE,
                        "silent": False,
                        "store_history": False,
                        "user_expressions": {},
                        "allow_stdin": False,
                        "stop_on_error": True,
                    },
                }
            )
        )
        while True:
            m = json.loads(ws.recv())
            if m.get("parent_header", {}).get("msg_id") != msg_id:
                continue
            t = m["msg_type"]
            if t == "stream":
                print("kernel says:", m["content"]["text"].strip())
            elif t == "error":
                print("kernel error:", m["content"]["ename"], m["content"]["evalue"])
            elif t == "status" and m["content"]["execution_state"] == "idle":
                break
        ws.close()
    finally:
        api("DELETE", f"/api/kernels/{kid}")
        print("kernel shut down")


if __name__ == "__main__":
    main()
