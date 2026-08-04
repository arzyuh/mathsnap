// PhotoMathJ - frontend логика.
// Комуницира со FastAPI backend преку /api/* (истиот origin бидејќи
// backend-от го сервира и frontend-от статички).

const API_BASE = "";

// ---------- Tabs ----------
document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
    // камерата троши батерија/ресурс - гаси се штом се напушти скенирај-табот
    if (btn.dataset.tab !== "scan") stopLiveCamera();
  });
});

// ---------- Scan mode toggle (live camera vs upload) ----------
const modeLiveBtn = document.getElementById("mode-live-btn");
const modeUploadBtn = document.getElementById("mode-upload-btn");
const liveModeEl = document.getElementById("live-mode");
const uploadModeEl = document.getElementById("upload-mode");

modeLiveBtn.addEventListener("click", () => {
  modeLiveBtn.classList.add("active");
  modeUploadBtn.classList.remove("active");
  liveModeEl.hidden = false;
  uploadModeEl.hidden = true;
});

modeUploadBtn.addEventListener("click", () => {
  modeUploadBtn.classList.add("active");
  modeLiveBtn.classList.remove("active");
  uploadModeEl.hidden = false;
  liveModeEl.hidden = true;
  stopLiveCamera();
});

// ---------- Scan tab ----------
const imageInput = document.getElementById("image-input");
const preview = document.getElementById("preview");
const scanBtn = document.getElementById("scan-btn");
const scanStatus = document.getElementById("scan-status");
const dropLabel = document.getElementById("drop-label");
const scanRecognizedCard = document.getElementById("scan-recognized");
const recognizedText = document.getElementById("recognized-text");
const resolveRecognizedBtn = document.getElementById("resolve-recognized-btn");

let selectedFile = null;

imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];
  if (!file) return;
  selectedFile = file;
  preview.src = URL.createObjectURL(file);
  preview.hidden = false;
  dropLabel.textContent = file.name;
  scanBtn.disabled = false;
  scanStatus.textContent = "";
});

scanBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  setStatus(scanStatus, "Препознавам текст од сликата...", "");
  scanBtn.disabled = true;

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const res = await fetch(`${API_BASE}/api/ocr`, { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "OCR грешка.");

    recognizedText.value = data.normalized_text || data.raw_text;
    scanRecognizedCard.hidden = false;
    setStatus(scanStatus, "Препознаено! Провери го текстот подолу и притисни 'Реши повторно'.", "ok");

    // автоматски пробај да го решиш веднаш
    await solveAndRender(recognizedText.value, scanStatus);
  } catch (err) {
    setStatus(scanStatus, err.message, "error");
  } finally {
    scanBtn.disabled = false;
  }
});

resolveRecognizedBtn.addEventListener("click", () => {
  solveAndRender(recognizedText.value, scanStatus);
});

// ---------- Live camera mode ----------
const videoEl = document.getElementById("camera-video");
const liveBadge = document.getElementById("live-badge");
const liveStartBtn = document.getElementById("live-start-btn");
const liveCaptureBtn = document.getElementById("live-capture-btn");
const liveStopBtn = document.getElementById("live-stop-btn");
const liveStatus = document.getElementById("live-status");
const captureCanvas = document.getElementById("capture-canvas");

const LIVE_SCAN_INTERVAL_MS = 1200;
const STABLE_FRAMES_REQUIRED = 2; // ист текст, во X последователни рамки -> решавај

let cameraStream = null;
let liveTimer = null;
let liveBusy = false; // спречи преклопување на барања додека претходното не заврши
let lastSeenText = null;
let stableCount = 0;

async function startLiveCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setStatus(liveStatus, "Овој browser не поддржува пристап до камера (getUserMedia).", "error");
    return;
  }
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
  } catch (err) {
    setStatus(liveStatus, "Не можам да пристапам до камерата: " + err.message, "error");
    return;
  }

  videoEl.srcObject = cameraStream;
  liveStartBtn.hidden = true;
  liveCaptureBtn.hidden = false;
  liveStopBtn.hidden = false;
  liveBadge.hidden = false;
  liveBadge.textContent = "👀 Барам израз...";
  setStatus(liveStatus, "", "");

  lastSeenText = null;
  stableCount = 0;
  liveTimer = setInterval(() => captureAndRecognize(false), LIVE_SCAN_INTERVAL_MS);
}

function stopLiveCamera() {
  if (liveTimer) {
    clearInterval(liveTimer);
    liveTimer = null;
  }
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
    cameraStream = null;
  }
  videoEl.srcObject = null;
  liveStartBtn.hidden = false;
  liveStartBtn.textContent = "🎥 Вклучи камера";
  liveCaptureBtn.hidden = true;
  liveStopBtn.hidden = true;
  liveBadge.hidden = true;
}

function captureFrameBlob() {
  return new Promise((resolve) => {
    if (!videoEl.videoWidth) {
      resolve(null);
      return;
    }
    captureCanvas.width = videoEl.videoWidth;
    captureCanvas.height = videoEl.videoHeight;
    const ctx = captureCanvas.getContext("2d");
    ctx.drawImage(videoEl, 0, 0, captureCanvas.width, captureCanvas.height);
    captureCanvas.toBlob((blob) => resolve(blob), "image/jpeg", 0.85);
  });
}

async function captureAndRecognize(forceSolve) {
  if (liveBusy || !cameraStream) return;
  liveBusy = true;
  try {
    const blob = await captureFrameBlob();
    if (!blob) return;

    const formData = new FormData();
    formData.append("file", blob, "frame.jpg");

    const res = await fetch(`${API_BASE}/api/ocr`, { method: "POST", body: formData });
    if (!res.ok) {
      // ништо не е препознаено на овој frame - тивко продолжи, без грешка на екран
      liveBadge.textContent = "👀 Барам израз...";
      lastSeenText = null;
      stableCount = 0;
      return;
    }

    const data = await res.json();
    const text = (data.normalized_text || data.raw_text || "").trim();
    if (!text) return;

    liveBadge.textContent = `Гледам: ${text}`;

    if (forceSolve) {
      await lockAndSolve(text);
      return;
    }

    if (text === lastSeenText) {
      stableCount += 1;
    } else {
      lastSeenText = text;
      stableCount = 1;
    }

    if (stableCount >= STABLE_FRAMES_REQUIRED) {
      await lockAndSolve(text);
    }
  } catch (err) {
    // мрежна/друга грешка на еден frame - следниот интервал ќе пробa повторно
  } finally {
    liveBusy = false;
  }
}

async function lockAndSolve(text) {
  if (liveTimer) {
    clearInterval(liveTimer);
    liveTimer = null;
  }
  liveBadge.textContent = `✅ Препознаено: ${text}`;
  setStatus(liveStatus, "Решавам...", "");

  const ok = await solveAndRender(text, liveStatus);

  if (ok) {
    stopLiveCamera();
  } else {
    // не се решило (пр. половина прочитан израз) - продолжи да скенираш
    lastSeenText = null;
    stableCount = 0;
    if (cameraStream) {
      liveTimer = setInterval(() => captureAndRecognize(false), LIVE_SCAN_INTERVAL_MS);
    }
  }
}

liveStartBtn.addEventListener("click", startLiveCamera);
liveStopBtn.addEventListener("click", stopLiveCamera);
liveCaptureBtn.addEventListener("click", () => captureAndRecognize(true));

// прекини ја камерата ако корисникот ја напушти/минимизира страницата
document.addEventListener("visibilitychange", () => {
  if (document.hidden) stopLiveCamera();
});

// ---------- Calculator tab ----------
const calcInput = document.getElementById("calc-input");
const calcSolveBtn = document.getElementById("calc-solve-btn");
const calcStatus = document.getElementById("calc-status");
const calcClearBtn = document.getElementById("calc-clear");
const calcBackBtn = document.getElementById("calc-back");

document.querySelectorAll(".keypad button[data-insert]").forEach((btn) => {
  btn.addEventListener("click", () => {
    const insert = btn.dataset.insert;
    const cursorAdjust = insert.endsWith("()") ? 1 : 0;
    const pos = calcInput.selectionStart ?? calcInput.value.length;
    calcInput.value =
      calcInput.value.slice(0, pos) + insert + calcInput.value.slice(pos);
    const newPos = pos + insert.length - cursorAdjust;
    calcInput.focus();
    calcInput.setSelectionRange(newPos, newPos);
  });
});

calcClearBtn.addEventListener("click", () => {
  calcInput.value = "";
  calcInput.focus();
});

calcBackBtn.addEventListener("click", () => {
  const pos = calcInput.selectionStart ?? calcInput.value.length;
  if (pos === 0) return;
  calcInput.value = calcInput.value.slice(0, pos - 1) + calcInput.value.slice(pos);
  calcInput.focus();
  calcInput.setSelectionRange(pos - 1, pos - 1);
});

calcSolveBtn.addEventListener("click", () => solveAndRender(calcInput.value, calcStatus));
calcInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") solveAndRender(calcInput.value, calcStatus);
});

// ---------- Shared solve + render ----------
const resultsSection = document.getElementById("results");
const methodsContainer = document.getElementById("methods-container");
const graphContainer = document.getElementById("graph-container");
const graphImg = document.getElementById("graph-img");

async function solveAndRender(inputText, statusEl) {
  if (!inputText || !inputText.trim()) {
    setStatus(statusEl, "Внеси или скенирај израз прво.", "error");
    return false;
  }
  setStatus(statusEl, "Решавам...", "");

  try {
    const res = await fetch(`${API_BASE}/api/solve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ input: inputText }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Грешка при решавање.");

    renderResult(data);
    setStatus(statusEl, "Готово!", "ok");
    return true;
  } catch (err) {
    setStatus(statusEl, err.message, "error");
    resultsSection.hidden = true;
    return false;
  }
}

function renderResult(data) {
  methodsContainer.innerHTML = "";

  data.methods.forEach((method) => {
    const methodEl = document.createElement("div");
    methodEl.className = "method";

    const nameEl = document.createElement("div");
    nameEl.className = "method-name";
    nameEl.textContent = method.name;
    methodEl.appendChild(nameEl);

    method.steps.forEach((step, i) => {
      const stepEl = document.createElement("div");
      stepEl.className = "step";
      stepEl.innerHTML = `
        <div class="step-num">${i + 1}</div>
        <div class="step-text">
          <div class="step-explanation">${escapeHtml(step.explanation)}</div>
          <div class="step-expr" data-latex="${escapeHtml(step.expr_latex)}"></div>
        </div>`;
      methodEl.appendChild(stepEl);
    });

    const resultEl = document.createElement("div");
    resultEl.className = "result-final";
    resultEl.setAttribute("data-latex", method.result_latex || method.result_text || "");
    methodEl.appendChild(resultEl);

    methodsContainer.appendChild(methodEl);
  });

  renderAllLatex();

  if (data.graphable) {
    graphImg.src = `${API_BASE}/api/graph.png?expr=${encodeURIComponent(data.normalized_text)}&_=${Date.now()}`;
    graphContainer.hidden = false;
  } else {
    graphContainer.hidden = true;
  }

  resultsSection.hidden = false;
  resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderAllLatex() {
  document.querySelectorAll("[data-latex]").forEach((el) => {
    const latex = el.getAttribute("data-latex");
    if (!latex) return;
    try {
      window.katex.render(latex, el, { throwOnError: false, displayMode: false });
    } catch (e) {
      el.textContent = latex;
    }
  });
}

function setStatus(el, text, kind) {
  el.textContent = text;
  el.className = "status" + (kind ? ` ${kind}` : "");
}

function escapeHtml(str) {
  return String(str)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}
