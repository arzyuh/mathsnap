from __future__ import annotations

import logging
import os
import threading
import urllib.error
import urllib.request

import cv2
import numpy as np
import pytesseract

logger = logging.getLogger(__name__)


HANDWRITING_SERVICE_URL = "http://127.0.0.1:8001/recognize"
HANDWRITING_SERVICE_TIMEOUT_S = 20
HANDWRITING_MODEL_NAME = "fhswf/TrOCR_Math_handwritten"

HANDWRITING_MODE = os.environ.get("HANDWRITING_MODE", "http")

_handwriting_processor = None
_handwriting_model = None
_handwriting_lock = threading.Lock()


def _get_inprocess_handwriting_model():
    global _handwriting_processor, _handwriting_model
    if _handwriting_model is None:
        with _handwriting_lock:
            if _handwriting_model is None:
                from transformers import TrOCRProcessor, VisionEncoderDecoderModel

                logger.info("Вчитувам ракописен модел %s (inprocess) ...", HANDWRITING_MODEL_NAME)
                _handwriting_processor = TrOCRProcessor.from_pretrained(HANDWRITING_MODEL_NAME)
                _handwriting_model = VisionEncoderDecoderModel.from_pretrained(HANDWRITING_MODEL_NAME)
                _handwriting_model.eval()
                logger.info("Ракописниот модел е вчитан.")
    return _handwriting_processor, _handwriting_model

MATH_CHAR_ALLOWLIST = (
    "0123456789"
    "xyzXYZabcinopqrst"
    "+-*/=().,^"
    "<>"
)
MATH_CHAR_WHITELIST = MATH_CHAR_ALLOWLIST

TESSERACT_CONFIG_SINGLE_LINE = (
    f'--psm 7 --oem 3 -c tessedit_char_whitelist="{MATH_CHAR_WHITELIST}"'
)
TESSERACT_CONFIG_BLOCK = (
    f'--psm 6 --oem 3 -c tessedit_char_whitelist="{MATH_CHAR_WHITELIST}"'
)

_reader_lock = threading.Lock()
_reader = None


def get_reader():
    global _reader
    if _reader is None:
        with _reader_lock:
            if _reader is None:
                import easyocr

                _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def _decode_image(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Не можам да ја декодирам сликата - невалиден формат.")
    return img


def _remove_ruled_lines(gray: np.ndarray) -> np.ndarray:

    inv = cv2.bitwise_not(gray)
    _, binary = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    line_width = max(gray.shape[1] // 15, 25)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_width, 1))
    detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
    detected_lines = cv2.dilate(detected_lines, np.ones((3, 3), np.uint8), iterations=1)

    result = gray.copy()
    result[detected_lines > 0] = 255
    return result


def crop_to_ink_content(image_bytes: bytes, padding_ratio: float = 0.25) -> bytes:

    try:
        img = _decode_image(image_bytes)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        no_lines = _remove_ruled_lines(gray)
        h_img, w_img = gray.shape[:2]

        inv = cv2.bitwise_not(no_lines)
        _, binary = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        close_size = max(int(min(h_img, w_img) * 0.015), 4)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image_bytes

        min_area = 0.0008 * (w_img * h_img)
        big_boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= min_area]
        if not big_boxes:
            return image_bytes

        x = min(bx for bx, by, bw, bh in big_boxes)
        y = min(by for bx, by, bw, bh in big_boxes)
        x2 = max(bx + bw for bx, by, bw, bh in big_boxes)
        y2 = max(by + bh for bx, by, bw, bh in big_boxes)
        w, h = x2 - x, y2 - y

        if (w * h) < 0.005 * (w_img * h_img):
            return image_bytes

        pad_x = int(w * padding_ratio)
        pad_y = int(h * padding_ratio)
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(w_img, x + w + pad_x)
        y1 = min(h_img, y + h + pad_y)

        cropped = img[y0:y1, x0:x1]
        ok, buf = cv2.imencode(".png", cropped)
        if not ok:
            return image_bytes
        return buf.tobytes()
    except Exception:
        logger.exception("crop_to_ink_content failed - користам оригинална слика")
        return image_bytes


def _deskew(gray: np.ndarray) -> np.ndarray:

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

    processed = preprocess_for_easyocr(image_bytes)
    binary = cv2.adaptiveThreshold(
        processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=31, C=15
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


def _assemble_text(results: list, multiline: bool) -> str:

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

    image_bytes = crop_to_ink_content(image_bytes)

    if HANDWRITING_MODE == "inprocess":
        try:
            import io

            from PIL import Image

            processor, model = _get_inprocess_handwriting_model()
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            pixel_values = processor(images=img, return_tensors="pt").pixel_values
            generated_ids = model.generate(pixel_values, max_length=64)
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return text.strip() or None
        except Exception:
            logger.exception("Inprocess ракописен модел не успеа")
            return None

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

    processed = preprocess_image(image_bytes)
    config = TESSERACT_CONFIG_BLOCK if multiline else TESSERACT_CONFIG_SINGLE_LINE
    raw_text = pytesseract.image_to_string(processed, config=config)
    return raw_text.strip()


def extract_text_with_debug(image_bytes: bytes, multiline: bool = False):

    text = extract_text(image_bytes, multiline)
    processed = preprocess_for_easyocr(image_bytes)
    ok, buf = cv2.imencode(".png", processed)
    debug_png = buf.tobytes() if ok else b""
    return text, debug_png
