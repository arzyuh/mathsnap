# PhotoMathJ

Апликација слична на Photomath - скенирај (или рачно внеси) математички
проблем, добиј решение чекор-по-чекор, со повеќе методи и график.

MVP опсег: **печатен текст**, основна алгебра - линеарни равенки,
квадратни равенки (факторизација + квадратна формула), и симплификација
на изрази.

## Архитектура

```
слика (камера/upload)
        │
        ▼
[OpenCV preprocessing]  backend/app/ocr.py
  grayscale → denoise → upscale → deskew → adaptive threshold
        │
        ▼
[Tesseract OCR]          backend/app/ocr.py
        │
        ▼
[Normalizer/Parser]      backend/app/expression_parser.py
  Unicode симболи → ASCII, imlicit multiplication, → SymPy Eq/Expr
        │
        ▼
[Solver + Step engine]   backend/app/solver.py, backend/app/steps.py
  класификација (линеарна/квадратна/симплификација) → SymPy solve
  + rule-based наратив на чекорите (неколку методи кога е применливо)
        │
        ▼
[Graphing]                backend/app/graphing.py
  matplotlib PNG - функција или лева/десна страна со означени решенија
        │
        ▼
[FastAPI]                 backend/app/main.py
  /api/ocr, /api/solve, /api/ocr-and-solve, /api/graph.png
        │
        ▼
[Frontend]                 frontend/
  camera/upload таб + калкулатор таб, KaTeX за прикажување формули
```

## Стартување

Потребно:
- Python 3.10+ со: `fastapi`, `uvicorn`, `python-multipart`, `opencv-python`,
  `pytesseract`, `pillow`, `numpy`, `sympy`, `matplotlib` (`pip install -r requirements.txt`)
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) инсталиран на системот
  (веќе детектиран на `C:\Program Files\Tesseract-OCR\tesseract.exe`)

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

Отвори `http://127.0.0.1:8000` во browser - frontend-от се сервира
директно преку истиот FastAPI сервер.

### Live камера скенирање

Табот "🎥 Live камера" користи `getUserMedia` за да пушти видео од
камерата, и на секои ~1.2s зема frame (canvas snapshot) кој го праќа
до `/api/ocr`. Штом истиот текст се препознае 2 пати по ред, автоматски
се решава (или притисни "📸 Сними рачно" за веднаш).

**Важно за `getUserMedia`**: browser-от бара secure context. Тоа значи:
- работи автоматски на `http://localhost` / `http://127.0.0.1`
- НЕ работи на обичен `http://<LAN-IP>:8000` (пр. тестирање од телефон
  преку WiFi на истата мрежа) - таму треба HTTPS (self-signed сертификат,
  reverse proxy со TLS, или алатка како `ngrok`/`cloudflared` за туннел),
  или `adb reverse` (Android) за да се третира како localhost.

### Тестови

```bash
cd backend
python -m pytest tests/ -v
```

## API endpoints

| Метод | Патека | Опис |
|---|---|---|
| POST | `/api/ocr` | слика → препознаен текст |
| POST | `/api/solve` | текст → чекори + резултат |
| POST | `/api/ocr-and-solve` | слика → директно решение (комбиниран) |
| GET | `/api/graph.png?expr=...` | PNG график |
| GET | `/api/health` | health check |

## Познати ограничувања / идни чекори

- **Ракопис**: моменталната OCR (Tesseract) е за печатен текст. За
  ракопис треба посебен трениран модел (CNN/transformer на CROHME-сличен
  dataset), или Image-to-LaTeX модел - тоа е поголем, одделен проект.
- **Опсег на проблеми**: моментално линеарни/квадратни равенки и
  симплификација. Calculus (изводи, интеграли), системи равенки,
  и тригонометрија бараат дополнителни "narrator" модули во `steps.py`.
- **Мобилна апликација**: моментално е responsive web app (работи во
  мобилен browser преку камера). Нативна апликација (Flutter/React
  Native) би го користела истиот backend API.
