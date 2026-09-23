FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.deploy.txt .
RUN pip install --no-cache-dir torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.deploy.txt

COPY backend/ backend/
COPY frontend/ frontend/

ENV HANDWRITING_MODE=inprocess
ENV PORT=7860
EXPOSE 7860

WORKDIR /app/backend
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
