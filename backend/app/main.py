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
from .expression_parser import ParseError, ParsedProblem, latex_to_plain, parse, parse_latex

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
ocr_logger = logging.getLogger("photomathj.ocr")

app = FastAPI(title="MathSnap API", version="0.2.0")


@app.on_event("startup")
def _warm_up_ocr_model() -> None:

    def _load():
        try:
            ocr.get_reader()
        except Exception:
            pass

    threading.Thread(target=_load, daemon=True).start()

    if ocr.HANDWRITING_MODE == "inprocess":
        def _load_handwriting():
            try:
                ocr._get_inprocess_handwriting_model()
            except Exception:
                logging.getLogger(__name__).exception("Не успеа да се вчита inprocess ракописниот модел")

        threading.Thread(target=_load_handwriting, daemon=True).start()

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

    if not req.input or not req.input.strip():
        raise HTTPException(status_code=422, detail="Празен внес.")
    return _solve_payload(req.input)


@app.post("/api/ocr", response_model=OcrResponse)
async def ocr_endpoint(file: UploadFile = File(...)) -> OcrResponse:

    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=422, detail="Датотеката мора да биде слика.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=422, detail="Празна слика.")

    try:
        raw_text = ocr.extract_text(image_bytes)
    except Exception as exc:
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


@app.post("/api/ocr-accurate", response_model=OcrResponse)
async def ocr_accurate_endpoint(file: UploadFile = File(...)) -> OcrResponse:

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
            normalized = parse_latex(latex).normalized_text
        except ParseError:

            normalized = latex_to_plain(latex)
        return OcrResponse(raw_text=latex, normalized_text=normalized)

    try:
        raw_text = ocr.extract_text(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"OCR грешка: {exc}") from exc

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

    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=422, detail="Датотеката мора да биде слика.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=422, detail="Празна слика.")

    latex = ocr.recognize_handwriting_accurate(image_bytes)

    if latex is not None:

        ocr_logger.info("Handwriting service latex=%r", latex)
        try:
            parsed = parse_latex(latex)
        except ParseError:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": f"Делумно препознаено: „{latex}“ - провери/поправи го текстот подолу.",
                    "raw_text": latex,
                },
            )
        return _solve_payload_from_parsed(parsed)

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


_FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"
if _FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")
