"""
Мал, независен HTTP сервис за препознавање ракописна математика,
преку специјализираниот модел fhswf/TrOCR_Math_handwritten
(тренирано на MathWriting dataset - дигитално мастило од ракопис).

ЗОШТО ОДДЕЛЕН СЕРВИС (не директно во главниот backend):
Овој конкретен модел-checkpoint е зачуван со постар transformers формат
кој паѓа (RuntimeError: tensor on device meta) под новата верзија на
`transformers` инсталирана во главното Python опкружување. Наместо да
се даунгрејдира transformers глобално (ризично - можеби нешто друго на
системот зависи од понова верзија), овој сервис работи во сопствено
изолирано venv (`.venv_trocr/`, транспортирано со `transformers==4.46.0`
+ `torch==2.5.1`) и комуницира со главниот backend преку обично HTTP.

Намерно е imple - stdlib `http.server`, БЕЗ FastAPI/Flask - за да не
се внесуваат дополнителни зависности во ова веќе кревко изолирано venv.

Стартување:
    .venv_trocr\\Scripts\\python.exe backend\\handwriting_service\\server.py

Модел се вчитува ЕДНАШ при стартување (~10s на CPU), потоа секое
препознавање трае ~3-6s на CPU (0.6B параметри - побавно од EasyOCR,
затоа се користи само за "точен" повик - рачно снимање/финално
заклучување на live камерата, не за секој live preview frame).
"""
from __future__ import annotations

import io
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = 8001
MODEL_NAME = "fhswf/TrOCR_Math_handwritten"

print(f"[handwriting_service] Вчитувам модел {MODEL_NAME} ...", file=sys.stderr, flush=True)

from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

_processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
_model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
_model.eval()

print("[handwriting_service] Моделот е вчитан, серверот стартува.", file=sys.stderr, flush=True)


def recognize(image_bytes: bytes) -> str:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pixel_values = _processor(images=img, return_tensors="pt").pixel_values
    generated_ids = _model.generate(pixel_values, max_length=64)
    text = _processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        print(f"[handwriting_service] {format % args}", file=sys.stderr)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/recognize":
            self._send_json(404, {"error": "not found"})
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
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"[handwriting_service] Слушам на http://127.0.0.1:{PORT}", file=sys.stderr, flush=True)
    server.serve_forever()
