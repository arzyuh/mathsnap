# Docker deployment за Hugging Face Spaces - ЕДЕН container, ЕДЕН Python
# environment (за разлика од локалниот Windows dev setup кој користи
# два venv-а поради transformers version конфликт). Тоа е возможно
# бидејќи EasyOCR всушност НЕ бара `transformers` воопшто (потврдено
# со `pip show easyocr`) - конфликтот постоеше само затоа што главниот
# backend случајно имаше понова transformers верзија инсталирана
# порано. Тука инсталираме transformers==4.46.0 директно, компатибилна
# со TrOCR_Math_handwritten checkpoint-от, и EasyOCR работи fine со неа.

FROM python:3.10-slim

# tesseract-ocr - fallback OCR (backend/app/ocr.py)
# libgl1, libglib2.0-0 - потребни за opencv-python
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch/torchvision прво (посебен CPU index) - најголеми, менуваат се
# најретко, подобро keширање на Docker слоеви
COPY requirements.deploy.txt .
RUN pip install --no-cache-dir torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.deploy.txt

COPY backend/ backend/
COPY frontend/ frontend/

ENV HANDWRITING_MODE=inprocess
# HuggingFace Spaces Docker SDK очекува портата 7860 по default
ENV PORT=7860
EXPOSE 7860

WORKDIR /app/backend
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
