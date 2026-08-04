"""
OCR модул - препознавање на математички изрази од слика.

Главен engine: **EasyOCR** (deep learning, CRAFT детекција + CRNN
препознавање). Тестирано директно наспроти Tesseract на исти слики -
EasyOCR значително подобро се справува со фотографии од камера
(ракопис, различни агли/осветлување), додека Tesseract е тренирана
првенствено за скенирани документи со чист печатен текст.

Tesseract останува достапна како лесна fallback опција (`extract_text_tesseract`)
ако EasyOCR моделот не е достапен на системот.

Важно: EasyOCR работи многу подобро на слабо-преработена (grayscale)
слика отколку на агресивно бинаризирана - тестирано: 99% confidence на
grayscale наспроти 37% на adaptive-threshold бинарна слика. Затоа
preprocessing-от овде е намерно "лесен" (без threshold/бинаризација).
"""
from __future__ import annotations

import logging
import threading
import urllib.error
import urllib.request

import cv2
import numpy as np
import pytesseract

logger = logging.getLogger(__name__)

# Адреса на посебниот "точен" ракописен сервис (види
# backend/handwriting_service/server.py - работи во сопствено venv
# поради incompatibility на checkpoint-от со главната transformers верзија).
HANDWRITING_SERVICE_URL = "http://127.0.0.1:8001/recognize"
HANDWRITING_SERVICE_TIMEOUT_S = 20

# Дозволени карактери - го ограничува просторот на препознавање само на
# симболи релевантни за математички изрази (го намалува бројот на грешки).
MATH_CHAR_ALLOWLIST = (
    "0123456789"
    "xyzXYZabcnpqrst"
    "+-*/=()[]{}.,^"
    "<>"
)
MATH_CHAR_WHITELIST = MATH_CHAR_ALLOWLIST  # алиас, користен од Tesseract патеката

TESSERACT_CONFIG_SINGLE_LINE = (
    f'--psm 7 --oem 3 -c tessedit_char_whitelist="{MATH_CHAR_WHITELIST}"'
)
TESSERACT_CONFIG_BLOCK = (
    f'--psm 6 --oem 3 -c tessedit_char_whitelist="{MATH_CHAR_WHITELIST}"'
)

_reader_lock = threading.Lock()
_reader = None  # lazy singleton - иницијализацијата трае ~2s, прави се еднаш


def get_reader():
    """Lazy singleton за EasyOCR Reader (thread-safe). Иницијализацијата
    (вчитување на моделот) трае неколку секунди - затоа се прави само
    еднаш, идеално загреана при стартување на серверот (види main.py)."""
    global _reader
    if _reader is None:
        with _reader_lock:
            if _reader is None:
                import easyocr

                _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def _decode_image(image_bytes: bytes) -> np.ndarray:
    """Декодира bytes (пр. од upload) во OpenCV BGR слика."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Не можам да ја декодирам сликата - невалиден формат.")
    return img


def _remove_ruled_lines(gray: np.ndarray) -> np.ndarray:
    """Отстранува долги хоризонтални линии (пр. тетратка на линии) кои
    можат да го измешаат OCR-от. Детектира со морфолошко отворање со
    широк хоризонтален kernel, потоа ги "избришува" (бои во бело)."""
    inv = cv2.bitwise_not(gray)
    _, binary = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    line_width = max(gray.shape[1] // 15, 25)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_width, 1))
    detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
    detected_lines = cv2.dilate(detected_lines, np.ones((3, 3), np.uint8), iterations=1)

    result = gray.copy()
    result[detected_lines > 0] = 255
    return result


def _deskew(gray: np.ndarray) -> np.ndarray:
    """Исправа мало закривување (rotation) на текстот, ако постои."""
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if coords.shape[0] < 20:
        return gray

    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) < 0.5:
        return gray

    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        gray, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def preprocess_for_easyocr(image_bytes: bytes) -> np.ndarray:
    """Лесен preprocessing pipeline (БЕЗ бинаризација - таа му штети на
    EasyOCR, види забелешка на врвот на фајлот): grayscale -> отстрани
    линии од хартија -> исправи закривување -> зголеми ако е мала."""
    img = _decode_image(image_bytes)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    no_lines = _remove_ruled_lines(gray)
    deskewed = _deskew(no_lines)

    h, w = deskewed.shape[:2]
    if max(h, w) < 1000:
        scale = 1000 / max(h, w)
        deskewed = cv2.resize(deskewed, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    return deskewed


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Целосен (потежок) pipeline со бинаризација - користен само од
    Tesseract fallback патеката."""
    processed = preprocess_for_easyocr(image_bytes)
    binary = cv2.adaptiveThreshold(
        processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=31, C=15
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


def _assemble_text(results: list, multiline: bool) -> str:
    """EasyOCR враќа листа од (bbox, text, confidence) - по едно за секој
    детектиран "збор"/регион, без гарантиран редослед. Ги подредуваме
    лево-кон-десно (и одозгора-надолу за multiline) за да добиеме
    смислен единствен стринг наместо разбркани парчиња."""
    if not results:
        return ""

    items = []
    for bbox, text, conf in results:
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        items.append(
            {"text": text, "x": min(xs), "y": (min(ys) + max(ys)) / 2, "h": max(ys) - min(ys) or 1}
        )

    if not multiline:
        items.sort(key=lambda it: it["x"])
        return "".join(it["text"] for it in items)

    items.sort(key=lambda it: it["y"])
    rows = []
    for it in items:
        row = next((r for r in rows if abs(r["y"] - it["y"]) < it["h"] * 0.6), None)
        if row is None:
            rows.append({"y": it["y"], "items": [it]})
        else:
            row["items"].append(it)
            row["y"] = (row["y"] + it["y"]) / 2

    lines = []
    for row in rows:
        row["items"].sort(key=lambda it: it["x"])
        lines.append("".join(it["text"] for it in row["items"]))
    return "\n".join(lines)


def extract_text(image_bytes: bytes, multiline: bool = False) -> str:
    """
    Главна функција: слика (bytes) -> суров текст, преку EasyOCR.
    Паѓа назад на Tesseract ако EasyOCR не е инсталиран/не успее да се
    вчита моделот (пр. отсутни тежини на дискот).
    """
    processed = preprocess_for_easyocr(image_bytes)
    try:
        reader = get_reader()
        results = reader.readtext(processed, allowlist=MATH_CHAR_ALLOWLIST)
        text = _assemble_text(results, multiline)
        if text:
            return text.strip()
    except Exception:
        logger.exception("EasyOCR failed, falling back to Tesseract")

    return extract_text_tesseract(image_bytes, multiline)


def recognize_handwriting_accurate(image_bytes: bytes) -> str | None:
    """
    "Точна" ракописна препознавање преку специјализираниот
    TrOCR_Math_handwritten модел (посебен сервис, види
    backend/handwriting_service/server.py). Побавно (~3-6s на CPU) од
    EasyOCR, но значително попрецизно на ракопис - затоа НЕ се користи
    за секој live-preview frame, туку само за финалното "снимање"
    (рачно копче или кога live скенирањето се стабилизира).

    Враќа LaTeX стринг, или None ако сервисот не работи/е недостапен
    (повикувачот тогаш треба да падне назад на extract_text()).
    """
    try:
        req = urllib.request.Request(
            HANDWRITING_SERVICE_URL, data=image_bytes, method="POST"
        )
        with urllib.request.urlopen(req, timeout=HANDWRITING_SERVICE_TIMEOUT_S) as resp:
            import json

            payload = json.loads(resp.read().decode("utf-8"))
            return payload.get("latex", "").strip() or None
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
        logger.warning(
            "Ракописниот сервис (%s) не е достапен - паѓам назад на EasyOCR.",
            HANDWRITING_SERVICE_URL,
        )
        return None
    except Exception:
        logger.exception("Ракописниот сервис врати неочекувана грешка")
        return None


def extract_text_tesseract(image_bytes: bytes, multiline: bool = False) -> str:
    """Fallback OCR преку Tesseract (побрз, но послаб на ракопис/фотографии
    од камера отколку EasyOCR)."""
    processed = preprocess_image(image_bytes)
    config = TESSERACT_CONFIG_BLOCK if multiline else TESSERACT_CONFIG_SINGLE_LINE
    raw_text = pytesseract.image_to_string(processed, config=config)
    return raw_text.strip()


def extract_text_with_debug(image_bytes: bytes, multiline: bool = False):
    """Иста работа како extract_text, но враќа и ја преработената слика
    (како PNG bytes) за debugging/приказ во UI."""
    text = extract_text(image_bytes, multiline)
    processed = preprocess_for_easyocr(image_bytes)
    ok, buf = cv2.imencode(".png", processed)
    debug_png = buf.tobytes() if ok else b""
    return text, debug_png
