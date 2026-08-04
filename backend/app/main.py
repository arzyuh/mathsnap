"""
FastAPI backend - го поврзува целиот pipeline:

  слика (upload) --OCR--> суров текст
  текст (OCR или рачен внес) --parser--> sympy објект
  sympy објект --solver/steps--> чекори + резултат
  sympy објект --graphing--> PNG график (опционално)

Стартување (од backend/ папката):
    uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import graphing, ocr, solver
from .expression_parser import ParseError, ParsedProblem, parse, parse_latex

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
ocr_logger = logging.getLogger("photomathj.ocr")

app = FastAPI(title="PhotoMathJ API", version="0.1.0")


@app.on_event("startup")
def _warm_up_ocr_model() -> None:
    """EasyOCR-от вчитува тежини од дискот при прв повик (~2s) - го
    правиме тоа во background thread при стартување на серверот, за
    првиот корисник (пр. прв frame од live камерата) да не чека."""

    def _load():
        try:
            ocr.get_reader()
        except Exception:
            pass  # ако не успее, extract_text() сепак ќе падне на Tesseract

    threading.Thread(target=_load, daemon=True).start()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SolveRequest(BaseModel):
    input: str


class OcrResponse(BaseModel):
    raw_text: str
    normalized_text: str


def _solve_payload_from_parsed(parsed: ParsedProblem) -> dict:
    result = solver.solve(parsed)
    has_graph = graphing.build_graph(parsed, result) is not None

    return {
        "raw_text": parsed.raw_text,
        "normalized_text": parsed.normalized_text,
        "kind": result["kind"],
        "problem_type": result["problem_type"],
        "variable": result["variable"],
        "methods": [solver.method_to_dict(m) for m in result["methods"]],
        "graphable": has_graph,
    }


def _solve_payload(raw_text: str) -> dict:
    try:
        parsed = parse(raw_text)
    except ParseError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _solve_payload_from_parsed(parsed)


@app.post("/api/solve")
def solve_endpoint(req: SolveRequest) -> dict:
    """Прима текст (од рачен внес или веќе-препознаен OCR текст) и враќа
    чекор-по-чекор решение."""
    if not req.input or not req.input.strip():
        raise HTTPException(status_code=422, detail="Празен внес.")
    return _solve_payload(req.input)


@app.post("/api/ocr", response_model=OcrResponse)
async def ocr_endpoint(file: UploadFile = File(...)) -> OcrResponse:
    """Прима слика, ја преработува со OpenCV и препознава текст со Tesseract."""
    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=422, detail="Датотеката мора да биде слика.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=422, detail="Празна слика.")

    try:
        raw_text = ocr.extract_text(image_bytes)
    except Exception as exc:  # OpenCV/Tesseract runtime грешки
        raise HTTPException(status_code=500, detail=f"OCR грешка: {exc}") from exc

    ocr_logger.info("OCR raw_text=%r", raw_text)

    if not raw_text:
        raise HTTPException(
            status_code=422,
            detail="Не успеав да препознаам текст на сликата. Пробај со појасна слика.",
        )

    try:
        normalized = parse(raw_text).normalized_text
    except ParseError:
        normalized = raw_text

    return OcrResponse(raw_text=raw_text, normalized_text=normalized)


@app.post("/api/ocr-and-solve")
async def ocr_and_solve_endpoint(file: UploadFile = File(...)) -> dict:
    """Комбиниран endpoint: слика -> OCR -> решение, во еден повик."""
    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=422, detail="Датотеката мора да биде слика.")

    image_bytes = await file.read()
    try:
        raw_text = ocr.extract_text(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"OCR грешка: {exc}") from exc

    if not raw_text:
        raise HTTPException(
            status_code=422,
            detail="Не успеав да препознаам текст на сликата. Пробај со појасна слика.",
        )

    return _solve_payload(raw_text)


@app.post("/api/ocr-accurate-and-solve")
async def ocr_accurate_and_solve_endpoint(file: UploadFile = File(...)) -> dict:
    """
    "Точна" верзија на /api/ocr-and-solve - користи го специјализираниот
    ракописен модел (backend/handwriting_service/, посебен процес) наместо
    EasyOCR. Побавно (~3-6s), но значително подобро на ракопис - затоа
    се користи само за финалното снимање (рачно копче / стабилизирано
    live скенирање), не за секој live preview frame.

    Ако сервисот не работи, автоматски паѓа назад на EasyOCR
    (истото однесување како /api/ocr-and-solve).
    """
    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=422, detail="Датотеката мора да биде слика.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=422, detail="Празна слика.")

    latex = ocr.recognize_handwriting_accurate(image_bytes)

    if latex:
        ocr_logger.info("Handwriting service latex=%r", latex)
        try:
            parsed = parse_latex(latex)
        except ParseError:
            # моделот врати нешто што не парсира - падни назад на EasyOCR
            # наместо веднаш да откажеш
            latex = None
        else:
            return _solve_payload_from_parsed(parsed)

    # Fallback: ракописниот сервис не работи/не даде употреблив резултат
    try:
        raw_text = ocr.extract_text(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"OCR грешка: {exc}") from exc

    if not raw_text:
        raise HTTPException(
            status_code=422,
            detail="Не успеав да препознаам текст на сликата. Пробај со појасна слика.",
        )

    return _solve_payload(raw_text)


@app.get("/api/graph.png")
def graph_endpoint(expr: str) -> Response:
    """Враќа PNG график за дадениот израз/равенка (query param `expr`)."""
    try:
        parsed = parse(expr)
    except ParseError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    result = solver.solve(parsed)
    png_bytes = graphing.build_graph(parsed, result)
    if png_bytes is None:
        raise HTTPException(
            status_code=422, detail="Овој проблем не може да се прикаже графички."
        )
    return Response(content=png_bytes, media_type="image/png")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


# Сервирање на frontend-от (статични HTML/CSS/JS) директно преку истиот сервер,
# за да нема потреба од посебен frontend сервер во MVP.
_FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"
if _FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")
