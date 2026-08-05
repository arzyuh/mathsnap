# MathSnap

Апликација слична на Photomath - скенирај (или рачно внеси) математички
проблем, добиј решение чекор-по-чекор, со повеќе методи и график.

**Опфат:**
- Линеарни равенки, квадратни равенки (факторизација + квадратна формула)
- Равенки со повисок степен (кубни+) - факторизација или општо решение
- Системи од 2 линеарни равенки (метод на замена)
- Основна тригонометрија (sin/cos/tan равенки, општо периодично решение)
- Деривати - `diff(израз,x)` (правило на степен, збир, познати функции)
- Интеграли - `integrate(израз,x)` (истите правила, + C)
- Симплификација на изрази
- Печатен текст + основен ракопис (цифри/оператори) преку камера

## Архитектура

```
слика (камера/upload)
        │
        ├─────────────────────────────┐
        ▼                              ▼
[Брз preview OCR]              [Точен ракописен модел]
  EasyOCR (CRAFT+CRNN)           TrOCR_Math_handwritten
  backend/app/ocr.py             backend/handwriting_service/
  ~0.3-0.8s, за live badge       ~3-6s, само за финално снимање
        │                              │
        └──────────────┬───────────────┘
                        ▼
        [Normalizer/Parser]      backend/app/expression_parser.py
          ASCII (normalize_text) или LaTeX (latex_to_plain) → SymPy Eq/Expr
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
          /api/ocr, /api/ocr-accurate-and-solve, /api/solve, /api/graph.png
                        │
                        ▼
        [Frontend]                 frontend/
          live camera / upload / калкулатор таб, KaTeX за прикажување формули
```

### Зошто два OCR "мозоци"?

Тестирано директно на ист ракопис:

| Engine | Резултат на "1+1" (европски стил, "1" со сериф горе) |
|---|---|
| Tesseract | целосно погрешно |
| EasyOCR (general-purpose) | "0+^a" / "9+5" - несигурно на овој стил |
| **TrOCR_Math_handwritten** (специјализиран) | **"1+1"** ✅ |

Специјализираниот модел е ~0.6B параметри и работи ~3-6s на CPU - предолго
за да се повикува на секој live-camera frame (~1.2s интервал). Затоа:
- **EasyOCR** - брз "preview" за live badge текст + детекција кога сликата
  се стабилизирала
- **TrOCR_Math_handwritten** - точен модел, повикан САМО кога се
  "заклучува" финалниот резултат (стабилизирано скенирање или "Сними рачно")

## Стартување

### 1. Главен backend

```bash
python -m venv .venv        # ако веќе немаш соодветно опкружување
pip install -r requirements.txt
cd backend
uvicorn app.main:app --reload --port 8000
```

Потребен е и [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)
инсталиран на системот (fallback ако EasyOCR не успее да се вчита).

### 2. Ракописен сервис (опционално, но препорачано)

Овој модел-checkpoint е зачуван со постар `transformers` формат кој паѓа
под понови верзии - затоа работи во **сопствено, изолирано venv**:

```bash
python -m venv .venv_trocr
.venv_trocr\Scripts\pip install -r backend/handwriting_service/requirements.txt
.venv_trocr\Scripts\python backend/handwriting_service/server.py
```

Слуша на `http://127.0.0.1:8001`. Првото стартување презема ~2.4GB тежини
од HuggingFace (еднократно, потоа се кешираат локално).

**Ако овој сервис не работи** - `/api/ocr-accurate-and-solve` автоматски
паѓа назад на EasyOCR, апликацијата продолжува да работи (само без
подобрената точност на ракопис).

Отвори `http://127.0.0.1:8000` во browser откако главниот backend работи.

### Live камера скенирање

Табот "🎥 Live камера" користи `getUserMedia` за да пушти видео од
камерата. На секои ~1.2s зема frame и го праќа до `/api/ocr` (брз preview,
за badge текст). Штом истиот текст се препознае 2 пати по ред - или
притиснеш "📸 Сними рачно" - тој frame се праќа до
`/api/ocr-accurate-and-solve` (точниот модел) за финалното решение.

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
| POST | `/api/ocr` | слика → брз препознаен текст (EasyOCR) |
| POST | `/api/ocr-accurate-and-solve` | слика → точно решение (специјализиран ракописен модел, fallback на EasyOCR) |
| POST | `/api/solve` | текст → чекори + резултат |
| POST | `/api/ocr-and-solve` | слика → директно решение (EasyOCR патека) |
| GET | `/api/graph.png?expr=...` | PNG график |
| GET | `/api/health` | health check |

## Познати ограничувања / идни чекори

- **Ракопис**: сега поддржан преку `fhswf/TrOCR_Math_handwritten`, но само
  за релативно кратки/едноставни изрази (тестирано на цифри + основни
  оператори). Комплексни изрази (дропки, повеќе редови) не се детално
  тестирани.
- **Опсег на проблеми**: тригонометријата покрива само едноставни равенки
  од обликот `sin(x)=a` (без сложен аргумент, пр. `sin(2x)` е вон опфат).
  Системи равенки: само 2 равенки со 2 непознати имаат детален наратив
  (повеќе паѓа на општо SymPy solve). Кубни/квартични равенки без убава
  факторизација прикажуваат само конечен резултат (без чекори - формулите
  се премногу комплексни за учебнички приказ).
- **Два Python venv-а**: непријатност за deployment (два процеси, два
  опкружувања). Подобро долгорочно решение: consolidate во еден
  compatible transformers verzija, или контејнеризирај го ракописниот
  сервис одделно (Docker).
- **Мобилна апликација**: моментално е responsive web app (работи во
  мобилен browser преку камера). Нативна апликација (Flutter/React
  Native) би го користела истиот backend API.
