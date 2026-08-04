"""
OCR модул - препознавање на математички изрази од слика.

Користи OpenCV за преработка на сликата (preprocessing) и Tesseract
(преку pytesseract) за самото препознавање на карактерите.

Ова е OCR за ПЕЧАТЕН текст. Препознавање на ракопис бара посебен,
трениран модел (CNN/transformer на CROHME-сличен dataset) и не е
дел од оваа верзија - виде README за понатамошни чекори.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytesseract

# Whitelist на карактери релевантни за математички изрази.
# Ова значително го намалува бројот на грешки на Tesseract бидејќи
# не дозволува да "препознае" случајни букви таму каде нема смисла.
MATH_CHAR_WHITELIST = (
    "0123456789"
    "xyzXYZabcnpqrst"
    "+-*/=()[]{}.,^"
    "<>"
)

# --psm 7 = третирај ја сликата како еден ред текст (добро за еден израз)
# --psm 6 = еден блок текст (подобро кога има повеќе редови/чекори на сликата)
TESSERACT_CONFIG_SINGLE_LINE = (
    f'--psm 7 --oem 3 -c tessedit_char_whitelist="{MATH_CHAR_WHITELIST}"'
)
TESSERACT_CONFIG_BLOCK = (
    f'--psm 6 --oem 3 -c tessedit_char_whitelist="{MATH_CHAR_WHITELIST}"'
)


def _decode_image(image_bytes: bytes) -> np.ndarray:
    """Декодира bytes (пр. од upload) во OpenCV BGR слика."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Не можам да ја декодирам сликата - невалиден формат.")
    return img


def _deskew(gray: np.ndarray) -> np.ndarray:
    """Исправа мало закривување (rotation) на текстот, ако постои."""
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if coords.shape[0] < 20:
        return gray  # премалку точки за сигурна проценка на агол

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Не ротирај за ситни агли - најчесто е шум, не вистинска ротација
    if abs(angle) < 0.5:
        return gray

    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        gray, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    OpenCV pipeline: grayscale -> denoise -> upscale -> deskew -> binarize.

    Враќа бинарна (црно-бело) слика подготвена за OCR.
    """
    img = _decode_image(image_bytes)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Отстранување шум, но чувајќи ги рабовите (важно за тенки цртички,
    # пр. знакот за минус или дропка)
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # Зголемување на резолуцијата помага значително на Tesseract
    # кога сликата од камера е мала/оддалечена
    h, w = denoised.shape[:2]
    if max(h, w) < 1200:
        scale = 1200 / max(h, w)
        denoised = cv2.resize(
            denoised, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )

    deskewed = _deskew(denoised)

    # Adaptive threshold - подобро од обичен Otsu кога осветлувањето
    # на сликата не е рамномерно (сенки од телефон/рака при слик.)
    binary = cv2.adaptiveThreshold(
        deskewed,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=15,
    )

    # Морфолошко чистење на ситен шум (изолирани пиксели)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return cleaned


def extract_text(image_bytes: bytes, multiline: bool = False) -> str:
    """
    Главна функција: слика (bytes) -> суров текст препознаен од Tesseract.
    """
    processed = preprocess_image(image_bytes)
    config = TESSERACT_CONFIG_BLOCK if multiline else TESSERACT_CONFIG_SINGLE_LINE
    raw_text = pytesseract.image_to_string(processed, config=config)
    return raw_text.strip()


def extract_text_with_debug(image_bytes: bytes, multiline: bool = False):
    """
    Иста работа како extract_text, но враќа и ја преработената слика
    (како PNG bytes) за debugging/приказ во UI - корисно за да се види
    зошто OCR евентуално погрешил.
    """
    processed = preprocess_image(image_bytes)
    config = TESSERACT_CONFIG_BLOCK if multiline else TESSERACT_CONFIG_SINGLE_LINE
    raw_text = pytesseract.image_to_string(processed, config=config).strip()

    ok, buf = cv2.imencode(".png", processed)
    debug_png = buf.tobytes() if ok else b""
    return raw_text, debug_png
