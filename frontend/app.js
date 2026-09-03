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

// Автоматски обиди со точниот (побавен, ~3-6s) модел - без посреден
// "брз preview" чекор. Интервалот мора да е поголем од времето на
// инференца за да нема преклопувачки барања (liveBusy flag дополнително
// штити од тоа).
const AUTO_SCAN_INTERVAL_MS = 4000;

let cameraStream = null;
let liveTimer = null;
let liveBusy = false; // спречи преклопување на барања додека претходното не заврши
let lastCandidateText = null; // за stability проверка (2 последователни исти читања)

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
  liveBadge.textContent = "🔎 Скенирам...";
  setStatus(liveStatus, "", "");

  lastCandidateText = null;
  attemptCount = 0;
  liveTimer = setInterval(() => captureAndCheckStability(), AUTO_SCAN_INTERVAL_MS);
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

// Ако по овој број обиди сè уште нема 2 совпаѓања, реши со последното
// читање и онака - подобро "најдобар обид" отколку бесконечно чекање
// без никаква повратна информација (реален случај - loop замрзнуваше
// "Скенирам..." засекогаш кога моделот никогаш не даваше исто читање
// двапати по ред).
const MAX_ATTEMPTS_BEFORE_FORCE = 5;

let attemptCount = 0;

/**
 * Автоматскиот live-loop: препознај (без веднаш да решаваш) и барај 2
 * ПОСЛЕДОВАТЕЛНИ читања што се СОВПАЃААТ пред да го прикажеш резултатот
 * (доверба дека не е еднократна халуцинација). По MAX_ATTEMPTS_BEFORE_FORCE
 * неуспешни обиди без совпаѓање, сепак реши со последното читање - за да
 * не изгледа апликацијата замрзната.
 */
async function captureAndCheckStability() {
  if (liveBusy || !cameraStream) return;
  liveBusy = true;
  attemptCount += 1;
  liveBadge.textContent = `🔎 Скенирам... (${attemptCount})`;

  try {
    const blob = await captureFrameBlob();
    if (!blob) return;

    const formData = new FormData();
    formData.append("file", blob, "frame.jpg");

    const res = await fetch(`${API_BASE}/api/ocr-accurate`, { method: "POST", body: formData });
    if (!res.ok) {
      lastCandidateText = null; // неуспешно читање - reset, пробај повторно
      return;
    }
    const data = await res.json();
    const text = (data.normalized_text || data.raw_text || "").trim();
    if (!text) return;

    const confirmed = text === lastCandidateText;
    const forcedByAttemptLimit = !confirmed && attemptCount >= MAX_ATTEMPTS_BEFORE_FORCE;

    if (confirmed || forcedByAttemptLimit) {
      liveBadge.textContent = forcedByAttemptLimit ? "🔎 Решавам (најдобар обид)..." : "🔎 Потврдено, решавам...";
      const ok = await solveAndRender(text, liveStatus);
      if (ok) {
        liveBadge.textContent = `✅ ${text}`;
        stopLiveCamera();
        liveStartBtn.textContent = "🎥 Скенирај повторно";
        liveStartBtn.hidden = false;
      } else {
        attemptCount = 0; // не успеа - почни бројач одново
      }
      lastCandidateText = null;
    } else {
      lastCandidateText = text;
    }
  } catch (err) {
    lastCandidateText = null;
  } finally {
    liveBusy = false;
  }
}

/**
 * Рачно снимање ("📸 Сними рачно") - веднаш, БЕЗ stability проверка
 * (корисникот експлицитно бара единечен обид сега).
 * Земи frame и директно праќај кон точниот модел -> резултат на екран.
 * Нема посреден "Гледам: ..." текст - само статус додека чека, потоа
 * директно резултатот (или тивко продолжи ако не успее, нема да те
 * прекинува со грешки на секои неуспешни обиди).
 */
async function captureAndSolve() {
  if (liveBusy || !cameraStream) return;
  liveBusy = true;
  liveBadge.textContent = "🔎 Скенирам...";
  liveCaptureBtn.disabled = true;

  try {
    const blob = await captureFrameBlob();
    if (!blob) return;

    const formData = new FormData();
    formData.append("file", blob, "frame.jpg");

    const res = await fetch(`${API_BASE}/api/ocr-accurate-and-solve`, {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
    if (!res.ok) {
      // "Делумно препознаено" - серверот врати нешто конкретно што не
      // успеа целосно да се разбере (пр. "6+6..."). Подобро е веднаш да
      // му го покажеме на корисникот за поправка, отколку тивко да
      // продолжиме (или да прикажеме случајно погрешен резултат од
      // послаб fallback engine).
      const detail = data.detail;
      if (detail && typeof detail === "object" && detail.raw_text) {
        stopLiveCamera();
        liveStartBtn.textContent = "🎥 Скенирај повторно";
        liveStartBtn.hidden = false;
        correctionInput.value = detail.raw_text;
        methodsContainer.innerHTML = "";
        graphContainer.hidden = true;
        resultsSection.hidden = false;
        setStatus(correctionStatus, detail.message || "Делумно препознаено - провери/поправи го текстот.", "error");
        resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
      }
      return; // обична грешка (пр. сервисот недостапен) - тивко пробај повторно
    }

    liveBadge.textContent = `✅ ${data.normalized_text}`;
    renderResult(data);
    setStatus(liveStatus, "Готово!", "ok");
    stopLiveCamera();
    liveStartBtn.textContent = "🎥 Скенирај повторно";
    liveStartBtn.hidden = false;
  } catch (err) {
    // мрежна грешка на еден обид - следниот интервал ќе пробa повторно
  } finally {
    liveBusy = false;
    liveCaptureBtn.disabled = false;
  }
}

liveStartBtn.addEventListener("click", startLiveCamera);
liveStopBtn.addEventListener("click", stopLiveCamera);
liveCaptureBtn.addEventListener("click", () => captureAndSolve());

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
    const cursorAdjust = btn.dataset.cursorOffset
      ? parseInt(btn.dataset.cursorOffset, 10)
      : insert.endsWith("()")
      ? 1
      : 0;
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
const correctionInput = document.getElementById("result-correction-input");
const correctionBtn = document.getElementById("result-correction-btn");
const correctionStatus = document.getElementById("correction-status");

correctionBtn.addEventListener("click", () => {
  solveAndRender(correctionInput.value, correctionStatus);
});
correctionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") solveAndRender(correctionInput.value, correctionStatus);
});

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
  correctionInput.value = data.normalized_text || "";
  setStatus(correctionStatus, "", "");
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
