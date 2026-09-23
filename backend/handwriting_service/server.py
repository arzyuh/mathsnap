from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image

PORT = 8001
MODEL_NAME = "fhswf/TrOCR_Math_handwritten"


_WRONG_INTERPRETER = sys._base_executable if hasattr(sys, "_base_executable") else None
_processor = None
_model = None
_model_ready = threading.Event()


def _kill_wrong_interpreter_duplicates() -> None:
    if not _WRONG_INTERPRETER:
        return

    my_pid = os.getpid()
    script_name = os.path.basename(__file__)

    for _ in range(20):
        time.sleep(1)
        try:
            ps_cmd = (
                f"Get-CimInstance Win32_Process | "
                f"Where-Object {{ $_.ParentProcessId -eq {my_pid} -and "
                f"$_.ExecutablePath -eq '{_WRONG_INTERPRETER}' -and "
                f"$_.CommandLine -like '*{script_name}*' }} | "
                f"ForEach-Object {{ Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue; "
                f"Write-Output $_.ProcessId }}"
            )
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_cmd],
                capture_output=True, text=True, timeout=5,
            )
            killed = result.stdout.strip()
            if killed:
                print(
                    f"[handwriting_service] Убиен дупликат процес со погрешен interpreter (PID {killed}).",
                    file=sys.stderr, flush=True,
                )
        except Exception:
            pass


def _load_model():
    global _processor, _model
    print(f"[handwriting_service] Вчитувам модел {MODEL_NAME} ...", file=sys.stderr, flush=True)

    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    _processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    _model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
    _model.eval()
    _model_ready.set()

    print("[handwriting_service] Моделот е вчитан, спремен за барања.", file=sys.stderr, flush=True)


def recognize(image_bytes: bytes) -> str:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pixel_values = _processor(images=img, return_tensors="pt").pixel_values
    generated_ids = _model.generate(pixel_values, max_length=64)
    text = _processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()


class ExclusivePortServer(ThreadingHTTPServer):

    allow_reuse_address = False


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        print(f"[handwriting_service] {format % args}", file=sys.stderr)

    def do_GET(self):
        if self.path == "/health":
            if _model_ready.is_set():
                self._send_json(200, {"status": "ok"})
            else:
                self._send_json(503, {"status": "loading"})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/recognize":
            self._send_json(404, {"error": "not found"})
            return

        if not _model_ready.is_set():
            self._send_json(503, {"error": "моделот сè уште се вчитува, пробај повторно за неколку секунди"})
            return

        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            self._send_json(422, {"error": "празно тело на барањето"})
            return

        image_bytes = self.rfile.read(length)
        try:
            latex = recognize(image_bytes)
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})
            return

        self._send_json(200, {"latex": latex})

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":

    server = ExclusivePortServer(("127.0.0.1", PORT), Handler)
    print(f"[handwriting_service] Слушам на http://127.0.0.1:{PORT} (вчитувам модел...)", file=sys.stderr, flush=True)

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    watchdog_thread = threading.Thread(target=_kill_wrong_interpreter_duplicates, daemon=True)
    watchdog_thread.start()

    _load_model()

    server_thread.join()
