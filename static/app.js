const toggleBtn = document.getElementById("toggle");
const clearBtn = document.getElementById("clear");
const statusEl = document.getElementById("status");
const transcriptEl = document.getElementById("transcript");
const translationEl = document.getElementById("translation");
const prevCardEl = document.getElementById("caption-prev");
const prevTranscriptEl = document.getElementById("prev-transcript");
const prevTranslationEl = document.getElementById("prev-translation");
const historyEl = document.getElementById("history");
const settingsToggleBtn = document.getElementById("settings-toggle");
const settingsPanel = document.getElementById("settings-panel");
const targetLangInput = document.getElementById("target-lang");
const sourceLangSel = document.getElementById("source-lang-select");
const sourceLangInput = document.getElementById("source-lang-input");
const sceneSeedInput = document.getElementById("scene-seed");
const sceneInput = document.getElementById("scene");
const glossaryInput = document.getElementById("glossary");
const inputSourceSel = document.getElementById("input-source");
const asrBackendSel = document.getElementById("asr-backend");
const translateBackendSel = document.getElementById("translate-backend");
const hintGenerateBtn = document.getElementById("hint-generate");
const hintStatusEl = document.getElementById("hint-status");
const vuFillEl = document.getElementById("vu-fill");
const badgeMicEl = document.getElementById("badge-mic");
const badgeAsrEl = document.getElementById("badge-asr");
const badgeTranslateEl = document.getElementById("badge-translate");
const debugModeInput = document.getElementById("debug-mode");
const fontLargeInput = document.getElementById("font-large");

const LS_LANG = "livesub.targetLang";
const LS_SRC = "livesub.sourceLang";
const LS_INPUT = "livesub.inputSource";
const LS_ASR = "livesub.asrBackend";
const LS_TR = "livesub.translateBackend";
const LS_SCENE_SEED = "livesub.sceneSeed";
const LS_SCENE = "livesub.scene";
const LS_GLOSSARY = "livesub.glossary";
const LS_DEBUG = "livesub.debugMode";
const LS_FONT_LARGE = "livesub.fontLarge";

targetLangInput.value = localStorage.getItem(LS_LANG) || "Chinese (Simplified)";
// Source-lang has two visible elements (see HTML): a <select> for normal
// mode and a free-text <input> for debug mode. Both bind to the same
// localStorage key. If the persisted value isn't one of the select's
// intersection-language options (e.g. user previously typed "Cantonese" in
// debug mode), keep it in the input but default the visible select to auto
// so the user isn't staring at a blank dropdown.
const savedSrc = localStorage.getItem(LS_SRC) || "auto";
sourceLangInput.value = savedSrc;
sourceLangSel.value = [...sourceLangSel.options].some(o => o.value === savedSrc)
  ? savedSrc : "auto";

function currentSourceLang() {
  // Whichever element is currently visible is the source of truth at the
  // moment Start is pressed (or hints are generated).
  const v = document.body.classList.contains("debug-mode")
    ? sourceLangInput.value : sourceLangSel.value;
  return (v || "auto").trim();
}
inputSourceSel.value = localStorage.getItem(LS_INPUT) || "mic";
sceneSeedInput.value = localStorage.getItem(LS_SCENE_SEED) || "";
sceneInput.value = localStorage.getItem(LS_SCENE) || "";
glossaryInput.value = localStorage.getItem(LS_GLOSSARY) || "";
// Debug mode shows the diagnostic badges (mic peak / asr session / tr
// counter). Off by default; toggling it just adds/removes a body class
// since CSS does the work (.debug-mode reveals .badge).
const debugOn = localStorage.getItem(LS_DEBUG) === "1";
debugModeInput.checked = debugOn;
document.body.classList.toggle("debug-mode", debugOn);
debugModeInput.addEventListener("change", () => {
  const on = debugModeInput.checked;
  document.body.classList.toggle("debug-mode", on);
  localStorage.setItem(LS_DEBUG, on ? "1" : "0");
});

// Large captions toggle. First-run default picked from UA (mobile → on,
// desktop → off); after that the localStorage value sticks regardless of
// device. Spoofing the UA via "request desktop site" flips the default
// for that visit, which is by design — same logic on the other dimension.
const savedFontLarge = localStorage.getItem(LS_FONT_LARGE);
const fontLargeDefault = /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
const fontLargeOn = savedFontLarge !== null ? savedFontLarge === "1" : fontLargeDefault;
fontLargeInput.checked = fontLargeOn;
document.body.classList.toggle("font-large", fontLargeOn);
fontLargeInput.addEventListener("change", () => {
  const on = fontLargeInput.checked;
  document.body.classList.toggle("font-large", on);
  localStorage.setItem(LS_FONT_LARGE, on ? "1" : "0");
});

// Backend dropdowns are populated dynamically from /api/backends so the user
// only sees options that are actually configured server-side (have an API
// key or a local binary). Saved choice restored if still present; otherwise
// the first available wins (browser-default for an empty value).
function populateBackendSelect(sel, items, savedValue) {
  sel.innerHTML = "";
  if (items.length === 0) {
    // No backend configured for this kind on the server side. Show a clear
    // disabled placeholder so the user sees they need to set a key/binary,
    // instead of getting a mysterious WS-close error after pressing Start.
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "(none configured)";
    opt.disabled = true;
    sel.appendChild(opt);
    sel.disabled = true;
    return;
  }
  sel.disabled = false;
  for (const it of items) {
    const opt = document.createElement("option");
    opt.value = it.id;
    opt.textContent = it.label;
    sel.appendChild(opt);
  }
  if (savedValue && [...sel.options].some(o => o.value === savedValue)) {
    sel.value = savedValue;
  }
}

async function loadBackends() {
  let data;
  try {
    const res = await fetch("api/backends");
    data = await res.json();
  } catch (e) {
    console.error("failed to fetch /api/backends", e);
    return;
  }
  populateBackendSelect(asrBackendSel, data.asr || [], localStorage.getItem(LS_ASR));
  // Migrate old single "claude" value (pre-variant split) → claude-haiku.
  let savedTr = localStorage.getItem(LS_TR);
  if (savedTr === "claude") savedTr = "claude-haiku";
  populateBackendSelect(translateBackendSel, data.translate || [], savedTr);
  // Hide the topic/Research row entirely if no hints backend is configured.
  // Manually-filled Scene + Glossary fields stay — they work independently.
  if (!data.hints_available) {
    const row = document.getElementById("scene-seed-row");
    if (row) row.hidden = true;
  }
}
loadBackends();

sceneSeedInput.addEventListener("input", () => {
  localStorage.setItem(LS_SCENE_SEED, sceneSeedInput.value);
});
sceneInput.addEventListener("input", () => {
  localStorage.setItem(LS_SCENE, sceneInput.value);
});
glossaryInput.addEventListener("input", () => {
  localStorage.setItem(LS_GLOSSARY, glossaryInput.value);
});

async function generateHintsFromScene() {
  const desc = sceneSeedInput.value.trim();
  if (!desc) {
    hintStatusEl.textContent = "Type a scene seed first.";
    hintStatusEl.className = "muted err";
    return;
  }
  hintGenerateBtn.disabled = true;
  hintStatusEl.textContent = "Researching (may take 5–15s)…";
  hintStatusEl.className = "muted busy";
  try {
    const res = await fetch("api/hints", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        description: desc,
        // Pin glossary language to source_lang. The glossary is fed into
        // qwen-asr's --prompt as a token-level bias; if it's in a different
        // language than the audio, the bias is wasted (and may distort
        // recognition). "auto" leaves Claude to infer from description.
        source_lang: currentSourceLang(),
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
    const scene = (data.scene || "").trim();
    const glossary = (data.glossary || "").trim();
    if (!scene && !glossary) throw new Error("empty response");
    sceneInput.value = scene;
    glossaryInput.value = glossary;
    localStorage.setItem(LS_SCENE, scene);
    localStorage.setItem(LS_GLOSSARY, glossary);
    const tail = data.searches_used
      ? ` (${data.searches_used} web search${data.searches_used === 1 ? "" : "es"})`
      : "";
    const via = data.backend
      ? ` via ${data.backend.charAt(0).toUpperCase() + data.backend.slice(1)}`
      : "";
    hintStatusEl.textContent =
      `Generated${via}: scene (${scene.length}ch) + glossary (${glossary.length}ch)${tail}. Edit below as needed.`;
    hintStatusEl.className = "muted ok";
  } catch (e) {
    console.error(e);
    hintStatusEl.textContent = "Failed: " + e.message;
    hintStatusEl.className = "muted err";
  } finally {
    hintGenerateBtn.disabled = false;
  }
}

hintGenerateBtn.addEventListener("click", generateHintsFromScene);

inputSourceSel.addEventListener("change", () => {
  localStorage.setItem(LS_INPUT, inputSourceSel.value);
});
asrBackendSel.addEventListener("change", () => {
  localStorage.setItem(LS_ASR, asrBackendSel.value);
});
translateBackendSel.addEventListener("change", () => {
  localStorage.setItem(LS_TR, translateBackendSel.value);
});

targetLangInput.addEventListener("input", () => {
  localStorage.setItem(LS_LANG, targetLangInput.value);
});
sourceLangSel.addEventListener("change", () => {
  localStorage.setItem(LS_SRC, sourceLangSel.value);
  // Keep input synced so flipping into debug mode doesn't lose the choice.
  sourceLangInput.value = sourceLangSel.value;
});
sourceLangInput.addEventListener("input", () => {
  localStorage.setItem(LS_SRC, sourceLangInput.value);
  // Mirror into select only when the typed value is one of its options —
  // exotic values (Cantonese / Thai / ...) leave the select on auto.
  if ([...sourceLangSel.options].some(o => o.value === sourceLangInput.value)) {
    sourceLangSel.value = sourceLangInput.value;
  }
});

settingsToggleBtn.addEventListener("click", () => {
  settingsPanel.hidden = !settingsPanel.hidden;
  document.body.classList.toggle("settings-open", !settingsPanel.hidden);
});

function lockSettings(locked) {
  targetLangInput.disabled = locked;
  sourceLangSel.disabled = locked;
  sourceLangInput.disabled = locked;
  sceneSeedInput.disabled = locked;
  sceneInput.disabled = locked;
  glossaryInput.disabled = locked;
  inputSourceSel.disabled = locked;
  asrBackendSel.disabled = locked;
  translateBackendSel.disabled = locked;
  hintGenerateBtn.disabled = locked;
}

let ws = null;
let audioCtx = null;
let stream = null;
let workletNode = null;
let sourceNode = null;
let muteGain = null;
let analyserNode = null;
let vuRafId = null;
let recording = false;

let asrSessionCount = 0;
let translationsDone = 0;
let translationsPending = 0;
let lastServerPeak = 0;

// Sentence map: sid -> {sid, src, dst, srcDone, dstDone, archived}
const sentences = new Map();
// Which sid is currently shown in the PREV zone. Sentences live in three
// places linearly: CURRENT zone (in-flight) → PREV zone (just finalized,
// stable, animated slide-in) → HISTORY list (older). prev_revised events
// target one of these.
let prevSlotSid = null;

function setStatus(s) {
  statusEl.textContent = s;
}

function setToggleState(rec) {
  toggleBtn.querySelector(".icon").textContent = rec ? "■" : "▶";
  toggleBtn.querySelector(".label").textContent = rec ? "Stop" : "Start";
  toggleBtn.classList.toggle("recording", rec);
}

function updateBadges() {
  badgeMicEl.textContent = `mic ${lastServerPeak || "—"}`;
  badgeMicEl.classList.toggle("active", lastServerPeak > 1000);
  badgeMicEl.classList.toggle("warn", lastServerPeak > 0 && lastServerPeak <= 500);
  badgeAsrEl.textContent = `asr ${asrSessionCount}`;
  badgeAsrEl.classList.toggle("active", asrSessionCount > 0 && recording);
  badgeTranslateEl.textContent = `tr ${translationsDone}/${translationsPending + translationsDone}`;
  badgeTranslateEl.classList.toggle("active", translationsPending > 0);
}

function startVU() {
  if (!analyserNode) return;
  const buf = new Float32Array(analyserNode.fftSize);
  let smoothed = 0;
  const tick = () => {
    if (!analyserNode) return;
    analyserNode.getFloatTimeDomainData(buf);
    let peak = 0;
    for (let i = 0; i < buf.length; i++) {
      const v = Math.abs(buf[i]);
      if (v > peak) peak = v;
    }
    // Log-scaled meter (speech amplitudes are tiny on a linear scale). Fast
    // rise (α=0.9) so spikes are visible; slow decay (α=0.15) so the eye can
    // read them. Both branches must compare and blend in the same scale —
    // mixing linear `peak` into the rise branch with log `smoothed` made
    // the rise branch fire only on very loud sounds and then yank the bar
    // *down* to the linear value.
    const target = Math.min(1, Math.max(0, (Math.log10(peak + 0.001) + 3) / 3));
    smoothed = target > smoothed
      ? target * 0.9 + smoothed * 0.1
      : smoothed * 0.85 + target * 0.15;
    vuFillEl.style.width = `${Math.min(100, smoothed * 100)}%`;
    vuRafId = requestAnimationFrame(tick);
  };
  vuRafId = requestAnimationFrame(tick);
}

function stopVU() {
  if (vuRafId) cancelAnimationFrame(vuRafId);
  vuRafId = null;
  vuFillEl.style.width = "0%";
}

function getOrCreate(sid) {
  let s = sentences.get(sid);
  if (!s) {
    s = {
      sid, src: "", dst: "",
      srcDone: false, dstDone: false, archived: false,
      currentRev: 0,    // latest revision known for this sentence's translation
      // Two-phase revision rendering: streaming chunks for the FIRST
      // revision go straight into `dst` so the user gets feedback as
      // soon as Claude starts emitting tokens. Once `dst` has any
      // content, subsequent revisions buffer their chunks into
      // `pendingDst` (invisible) and atomically swap on `translation_done`.
      // This eliminates the "clear → typewriter → clear → typewriter"
      // flicker we used to see every PARTIAL_INTERVAL_SEC.
      pendingDst: "",
      usePending: false,
    };
    sentences.set(sid, s);
  }
  return s;
}

function render() {
  // CURRENT zone shows whatever sentence is "live" right now. With the
  // prev zone broken out, current is the lowest sid that is unarchived
  // AND not currently in the prev slot (so we don't render the same
  // sentence in two zones). Falls back to the newest unarchived.
  let current = null;
  let newestUnarchived = null;
  for (const [, s] of sentences) {
    if (s.archived) continue;
    if (s.sid === prevSlotSid) continue;
    if (!s.srcDone || !s.dstDone) {
      if (current === null || s.sid < current.sid) current = s;
    }
    if (newestUnarchived === null || s.sid > newestUnarchived.sid) newestUnarchived = s;
  }
  const display = current || newestUnarchived;
  transcriptEl.textContent = display ? display.src : "";
  translationEl.textContent = display ? display.dst : "";
}

// Promote a finalized sentence into the PREV zone. The sentence currently
// in PREV (if any) gets pushed to history (no animation — user said the
// prev → history transition can be instant). PREV gets the new content
// with a slide-up-from-below animation looking like the LIVE card (which
// sits at the very bottom) scrolled up into the prev slot just above it.
function promoteToPrev(s) {
  if (prevSlotSid !== null && prevSlotSid !== s.sid) {
    const oldPrev = sentences.get(prevSlotSid);
    if (oldPrev) {
      appendToHistory(oldPrev);
      oldPrev.archived = true;
    }
  }
  prevSlotSid = s.sid;
  prevTranscriptEl.textContent = s.src.trim();
  prevTranslationEl.textContent = s.dst.trim();
  prevCardEl.dataset.sid = String(s.sid);
  prevCardEl.hidden = false;
  // Restart slide-in animation by toggling the class with a forced reflow.
  prevCardEl.classList.remove("entering");
  void prevCardEl.offsetWidth;
  prevCardEl.classList.add("entering");
}

function appendToHistory(s) {
  const item = document.createElement("div");
  item.className = "history-item";
  item.dataset.sid = String(s.sid);
  const srcDiv = document.createElement("div");
  srcDiv.className = "src";
  srcDiv.textContent = s.src.trim();
  const dstDiv = document.createElement("div");
  dstDiv.className = "dst";
  dstDiv.textContent = s.dst.trim();
  item.append(srcDiv, dstDiv);
  // Append (not prepend) so DOM order is chronological — copy-paste then
  // gives oldest→newest, matching the visual top-to-bottom reading order.
  // Anchor scroll: if the user was already at the bottom, follow the
  // newest item; otherwise leave their scroll position alone so they can
  // keep reading older entries without being yanked away.
  const wasAtBottom =
    historyEl.scrollHeight - historyEl.clientHeight - historyEl.scrollTop < 4;
  historyEl.append(item);
  if (wasAtBottom) historyEl.scrollTop = historyEl.scrollHeight;
}

function applyPrevRevision(sid, text) {
  const s = sentences.get(sid);
  const before = s ? s.dst : "(unknown)";
  if (s) s.dst = text;
  console.log(
    `[prev_revised] sid=${sid}\n  before: ${before}\n  after:  ${text}`
  );
  // Where does this sid live? Three possibilities, in order of likelihood:
  // (1) PREV zone — most common (paired-translation revises the most
  //     recently finalized sentence, which is exactly what's in PREV).
  // (2) HISTORY list — if the sentence has already been pushed out of
  //     PREV by a newer sentence finalizing.
  // (3) Still in CURRENT zone — rare but possible if the revision races
  //     ahead of translation_done.
  if (prevSlotSid === sid) {
    prevTranslationEl.textContent = text.trim();
    prevTranslationEl.classList.add("revised");
    setTimeout(() => {
      prevTranslationEl.classList.remove("revised");
      prevTranslationEl.classList.add("revised-faded");
      setTimeout(() => prevTranslationEl.classList.remove("revised-faded"), 4000);
    }, 1500);
    return;
  }
  const item = historyEl.querySelector(`.history-item[data-sid="${sid}"]`);
  if (item) {
    const dstEl = item.querySelector(".dst");
    if (dstEl) {
      dstEl.textContent = text.trim();
      dstEl.classList.add("revised");
      setTimeout(() => {
        dstEl.classList.remove("revised");
        dstEl.classList.add("revised-faded");
        setTimeout(() => dstEl.classList.remove("revised-faded"), 4000);
      }, 1500);
    }
    return;
  }
  // Fall through: still in CURRENT — re-render picks up s.dst.
  render();
}

async function acquireStream(source) {
  if (source === "screen") {
    if (!navigator.mediaDevices.getDisplayMedia) {
      throw new Error("getDisplayMedia not supported in this browser");
    }
    setStatus("pick a tab/window with audio…");
    // Chrome requires video:true; many implementations refuse audio-only.
    const captured = await navigator.mediaDevices.getDisplayMedia({
      video: true,
      audio: {
        // Tab audio is already a clean digital signal — disable processing
        // that would distort it.
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });
    // Diagnostic: log everything the browser returned, before we touch it.
    const allTracks = captured.getTracks().map(t => ({
      kind: t.kind,
      label: t.label,
      readyState: t.readyState,
      muted: t.muted,
      enabled: t.enabled,
      settings: t.getSettings?.(),
    }));
    console.log("getDisplayMedia returned tracks:", allTracks);

    // Stop the video tracks we don't need, then build a fresh audio-only
    // MediaStream. Chrome's createMediaStreamSource can stall if the original
    // stream still has stopped video tracks attached.
    captured.getVideoTracks().forEach(t => t.stop());
    const audioTracks = captured.getAudioTracks();
    if (audioTracks.length === 0) {
      captured.getTracks().forEach(t => t.stop());
      throw new Error(
        "No audio track delivered. " +
        (navigator.userAgent.includes("Firefox") && navigator.userAgent.includes("Linux")
          ? "Firefox on Linux Wayland does not implement audio capture via " +
            "getDisplayMedia — try Chrome/Chromium, or route system audio " +
            "through a PipeWire virtual mic and use Mic mode."
          : "Pick a TAB (not 'Entire screen' or 'Window') and check 'Share tab audio'.")
      );
    }
    console.log("tab audio acquired:",
      audioTracks.map(t => ({label: t.label, settings: t.getSettings?.()})));
    return new MediaStream(audioTracks);
  }
  setStatus("requesting mic…");
  return navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      // For low-SNR scenarios (phone speaker captured by laptop mic)
      // browser-side processing helps a lot more than it hurts.
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  });
}

async function start() {
  try {
    const inputSource = inputSourceSel.value || "mic";
    stream = await acquireStream(inputSource);
    // If the user stops sharing from the browser's picker (Tab mode), the
    // tracks end and we should stop too.
    stream.getAudioTracks().forEach(t => {
      t.addEventListener("ended", () => {
        if (recording) stop();
      });
    });

    // Each ASR backend has its own preferred input rate. OpenAI Realtime's
    // minimum is 24kHz; everyone else (Gemini Live, Qwen, Voxtral) wants
    // 16kHz natively. Sending the right rate from the browser avoids
    // server-side resampling and — more importantly — gives 24kHz-only
    // backends real high-frequency content from the mic instead of fake
    // interpolated upsampling.
    const targetRate =
      (asrBackendSel.value === "openai-realtime") ? 24000 : 16000;
    try {
      audioCtx = new AudioContext({ sampleRate: targetRate });
    } catch {
      audioCtx = new AudioContext();
    }
    console.log(`AudioContext: ${audioCtx.sampleRate}Hz (worklet will resample to ${targetRate}Hz)`);
    await audioCtx.audioWorklet.addModule("static/pcm-worklet.js");
    sourceNode = audioCtx.createMediaStreamSource(stream);
    workletNode = new AudioWorkletNode(audioCtx, "pcm-worklet", {
      processorOptions: { targetRate },
    });
    sourceNode.connect(workletNode);
    muteGain = audioCtx.createGain();
    muteGain.gain.value = 0;
    workletNode.connect(muteGain).connect(audioCtx.destination);
    // VU meter taps the mic source directly (independent of worklet path).
    analyserNode = audioCtx.createAnalyser();
    analyserNode.fftSize = 1024;
    analyserNode.smoothingTimeConstant = 0.5;
    sourceNode.connect(analyserNode);
    startVU();

    setStatus("connecting…");
    // Resolve the /ws endpoint relative to the page's base URI so the app works
    // under a non-root path (reverse-proxy subpath), not just at the domain root.
    const wsBase = new URL("ws", document.baseURI);
    const wsProto = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(`${wsProto}//${wsBase.host}${wsBase.pathname}`);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
      ws.send(JSON.stringify({
        type: "config",
        target_lang: (targetLangInput.value || "Chinese (Simplified)").trim(),
        source_lang: currentSourceLang(),
        scene: sceneInput.value.trim(),
        glossary: glossaryInput.value.trim(),
        asr_backend: asrBackendSel.value || "gemini",
        translate_backend: translateBackendSel.value || "gemini",
      }));
      // Tag <body> so CSS can hide the redundant transcript line in
      // passthrough mode (when we mirror src into the translation slot).
      document.body.dataset.translateBackend =
          translateBackendSel.value || "gemini";
      workletNode.port.onmessage = (e) => {
        if (ws && ws.readyState === WebSocket.OPEN) ws.send(e.data);
      };
    };

    ws.onmessage = (e) => {
      let msg;
      try { msg = JSON.parse(e.data); } catch { return; }
      switch (msg.type) {
        case "ready":
          setStatus("listening");
          break;
        case "transcript": {
          const s = getOrCreate(msg.sid);
          s.src += msg.text;
          render();
          break;
        }
        case "transcript_replace": {
          // Used by backends that stream a cumulative transcript that
          // revises itself mid-segment (DashScope qwen3-asr-flash-realtime).
          // Overwrite src wholesale instead of appending.
          const s = getOrCreate(msg.sid);
          s.src = msg.text;
          render();
          break;
        }
        case "transcript_done": {
          const s = getOrCreate(msg.sid);
          s.srcDone = true;
          render();
          break;
        }
        case "translation_revision": {
          // A new revision (partial or final) starts.
          //   First revision (dst still empty): stream chunks directly so
          //   the user sees translation tokens flow in as Claude emits them.
          //   Subsequent revisions: buffer in pendingDst, atomically swap on
          //   translation_done — keeps the visible dst stable, no flicker.
          const s = getOrCreate(msg.sid);
          if (msg.rev >= s.currentRev) {
            s.currentRev = msg.rev;
            if (s.dst) {
              s.usePending = true;
              s.pendingDst = "";
            } else {
              s.usePending = false;
            }
            render();
          }
          break;
        }
        case "translation": {
          const s = getOrCreate(msg.sid);
          // Drop chunks from a stale revision that was cancelled/superseded.
          if (msg.rev != null && msg.rev !== s.currentRev) break;
          if (s.usePending) {
            // Invisible accumulation — no render() until done-swap.
            s.pendingDst += msg.text;
          } else {
            s.dst += msg.text;
            render();
          }
          break;
        }
        case "translation_done": {
          const s = getOrCreate(msg.sid);
          // Stale done — ignore.
          if (msg.rev != null && msg.rev !== s.currentRev) break;
          // Atomic swap: replace the visible dst with the buffered version,
          // but ONLY if the upstream stream succeeded. msg.ok === false
          // means Claude / Gemini raised mid-stream (502, network, etc.) —
          // the pendingDst is empty or half-formed and would clobber the
          // previously-shown good translation. Discard it instead.
          if (s.usePending) {
            if (msg.ok !== false) {
              s.dst = s.pendingDst;
              render();
            } else {
              console.log(
                `[translation_done] sid=${msg.sid} rev=${msg.rev} failed; keeping previous dst`
              );
            }
            s.pendingDst = "";
            s.usePending = false;
          }
          // Only the FINAL revision locks the sentence. Partial revs keep
          // the sentence alive for more updates.
          if (msg.final) {
            s.dstDone = true;
            translationsDone++;
            translationsPending = Math.max(0, translationsPending - 1);
            updateBadges();
            // Promote: this sentence becomes the new PREV (with slide-in
            // animation), the old PREV gets pushed to history.
            promoteToPrev(s);
            render();
          }
          break;
        }
        case "translation_start":
          translationsPending++;
          updateBadges();
          break;
        case "prev_revised":
          // Claude (paired-translation) revised the previous sentence's
          // translation in light of the new sentence. Update wherever it
          // currently lives (history list or live captions).
          if (msg.sid != null && typeof msg.text === "string") {
            applyPrevRevision(msg.sid, msg.text);
          }
          break;
        case "asr_session": {
          // asr_session events: {state: "open"|"reset"|"go_away"}.
          // Status bar stays at "listening" — the per-turn counter was
          // diagnostic noise for normal users (Gemini Live increments it
          // every sentence). The debug badge still reflects the count.
          if (msg.state === "open") asrSessionCount++;
          updateBadges();
          break;
        }
        case "audio_stats":
          // {peak, chunks_per_sec}
          lastServerPeak = msg.peak || 0;
          updateBadges();
          break;
        case "error":
          setStatus("error: " + msg.message);
          badgeAsrEl.classList.add("error");
          console.error("server error:", msg.message);
          break;
      }
    };

    ws.onerror = (e) => console.error("ws error", e);
    ws.onclose = () => {
      setStatus("disconnected");
      if (recording) stop();
    };

    recording = true;
    setToggleState(true);
    lockSettings(true);
  } catch (err) {
    console.error(err);
    setStatus("error: " + err.message);
    await stop();
  }
}

async function stop() {
  recording = false;
  stopVU();
  if (workletNode) {
    workletNode.port.onmessage = null;
    try { workletNode.disconnect(); } catch {}
    workletNode = null;
  }
  if (analyserNode) {
    try { analyserNode.disconnect(); } catch {}
    analyserNode = null;
  }
  if (sourceNode) {
    try { sourceNode.disconnect(); } catch {}
    sourceNode = null;
  }
  if (muteGain) {
    try { muteGain.disconnect(); } catch {}
    muteGain = null;
  }
  if (audioCtx) {
    try { await audioCtx.close(); } catch {}
    audioCtx = null;
  }
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  if (ws) {
    try { ws.close(); } catch {}
    ws = null;
  }
  // Push the PREV card's content into history, then drop all sentence
  // state. The server resets `next_sid` to 0 on every new WS connection,
  // so without this clear the next session's transcripts for sid=0/1/...
  // would `getOrCreate` and resurrect the previously-archived entries —
  // new content gets routed into stale `srcDone:true` records, render()
  // can't promote them as "current", and the UI shows empty captions
  // with old prev+history below. Strip data-sid off existing history
  // items for the same reason: the next session's `prev_revised` events
  // (sids start at 0 again) would otherwise target the wrong DOM nodes.
  if (prevSlotSid !== null) {
    const oldPrev = sentences.get(prevSlotSid);
    if (oldPrev) appendToHistory(oldPrev);
    prevCardEl.hidden = true;
    prevCardEl.classList.remove("entering");
    prevTranscriptEl.textContent = "";
    prevTranslationEl.textContent = "";
    prevSlotSid = null;
  }
  sentences.clear();
  for (const item of historyEl.querySelectorAll(".history-item[data-sid]")) {
    item.removeAttribute("data-sid");
  }
  transcriptEl.textContent = "";
  translationEl.textContent = "";
  setToggleState(false);
  lockSettings(false);
  setStatus("idle");
  asrSessionCount = 0;
  translationsDone = 0;
  translationsPending = 0;
  lastServerPeak = 0;
  badgeAsrEl.classList.remove("error", "active", "warn");
  updateBadges();
}

toggleBtn.addEventListener("click", () => {
  if (recording) stop(); else start();
});

clearBtn.addEventListener("click", () => {
  historyEl.innerHTML = "";
  sentences.clear();
  transcriptEl.textContent = "";
  translationEl.textContent = "";
  prevTranscriptEl.textContent = "";
  prevTranslationEl.textContent = "";
  prevCardEl.hidden = true;
  prevCardEl.classList.remove("entering");
  prevSlotSid = null;
});
