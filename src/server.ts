// HTTP + WebSocket entry point — the TS port of the FastAPI routes, the /ws
// endpoint, and the __main__ startup guard in server.py. Uses node:http + ws
// directly (no framework), matching the project's "one server file" ethos.

import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { createReadStream } from "node:fs";
import { statSync } from "node:fs";
import { extname, join, normalize } from "node:path";
import { WebSocketServer, WebSocket, type RawData } from "ws";
import { GoogleGenAI } from "@google/genai";

import {
  DEFAULT_TARGET_LANG,
  GEMINI_API_KEY,
  STATIC_DIR,
  listAsrBackends,
  listTranslateBackends,
  pickHintsBackend,
} from "./config.js";
import { generateHintsClaude, generateHintsGemini } from "./hints.js";
import { Pipeline } from "./pipeline.js";
import { log } from "./util.js";

const PORT = 8000;
const HOST = "0.0.0.0";

const MIME: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
  ".webmanifest": "application/manifest+json",
  ".woff2": "font/woff2",
};

function sendJson(res: ServerResponse, status: number, obj: unknown): void {
  const body = JSON.stringify(obj);
  res.writeHead(status, { "Content-Type": "application/json; charset=utf-8" });
  res.end(body);
}

// Serve a file from STATIC_DIR. `rel` is the path relative to STATIC_DIR;
// normalized and confined to the directory to prevent traversal.
function serveStatic(res: ServerResponse, rel: string): void {
  const safeRel = normalize(rel).replace(/^(\.\.[/\\])+/, "");
  const filePath = join(STATIC_DIR, safeRel);
  if (!filePath.startsWith(STATIC_DIR)) {
    res.writeHead(403).end("forbidden");
    return;
  }
  let st;
  try {
    st = statSync(filePath);
  } catch {
    res.writeHead(404).end("not found");
    return;
  }
  if (!st.isFile()) {
    res.writeHead(404).end("not found");
    return;
  }
  res.writeHead(200, { "Content-Type": MIME[extname(filePath)] ?? "application/octet-stream" });
  const stream = createReadStream(filePath);
  // A read error after headers are sent can't become a 500 — abort the response
  // instead of letting an unhandled 'error' crash the process.
  stream.on("error", () => res.destroy());
  stream.pipe(res);
}

function readBody(req: IncomingMessage, maxBytes = 1 << 20): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let total = 0;
    req.on("data", (c: Buffer) => {
      total += c.length;
      if (total > maxBytes) {
        reject(new Error("body too large"));
        req.destroy();
        return;
      }
      chunks.push(c);
    });
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf-8")));
    req.on("error", reject);
  });
}

// ---------- /api/hints ----------

async function handleHints(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const backend = pickHintsBackend();
  if (backend === null) {
    sendJson(res, 500, {
      error:
        "Hint generation requires GEMINI_API_KEY or ANTHROPIC_API_KEY. Set one in .env and restart, " +
        "or fill Scene/Glossary manually below.",
    });
    return;
  }
  let body: Record<string, unknown>;
  try {
    body = JSON.parse(await readBody(req));
  } catch {
    sendJson(res, 400, { error: "invalid JSON body" });
    return;
  }
  const description = String(body.description ?? "").trim();
  if (!description) {
    sendJson(res, 400, { error: "description required" });
    return;
  }
  if (description.length > 500) {
    sendJson(res, 400, { error: "description too long (>500 chars)" });
    return;
  }
  const sourceLang = String(body.source_lang ?? "").trim();

  log.info(`generating hints via ${backend} for scene: ${JSON.stringify(description.slice(0, 100))} (source_lang=${JSON.stringify(sourceLang || "auto")})`);
  try {
    const result = backend === "gemini" ? await generateHintsGemini(description, sourceLang) : await generateHintsClaude(description, sourceLang);
    log.info(
      `hints generated via ${backend}: scene=${result.scene.length}ch glossary=${result.glossary.length}ch ` +
        `${result.searchesUsed} web searches stop=${result.stopReason}`,
    );
    sendJson(res, 200, {
      scene: result.scene,
      glossary: result.glossary,
      searches_used: result.searchesUsed,
      model: result.model,
      backend,
    });
  } catch (e) {
    log.exception(`hint generation via ${backend} failed`, e);
    sendJson(res, 502, { error: e instanceof Error ? e.message : String(e), backend });
  }
}

// ---------- HTTP request router ----------

const httpServer = createServer((req, res) => {
  const url = req.url ?? "/";
  const path = url.split("?")[0];

  if (req.method === "GET" && path === "/") {
    serveStatic(res, "index.html");
    return;
  }
  if (req.method === "GET" && path === "/api/backends") {
    sendJson(res, 200, {
      asr: listAsrBackends(),
      translate: listTranslateBackends(),
      // Whether /api/hints can serve a request (any usable hints backend).
      hints_available: pickHintsBackend() !== null,
    });
    return;
  }
  if (req.method === "POST" && path === "/api/hints") {
    handleHints(req, res).catch((e) => {
      log.exception("hints handler error", e);
      if (!res.headersSent) sendJson(res, 500, { error: "internal error" });
    });
    return;
  }
  if (req.method === "GET" && path.startsWith("/static/")) {
    serveStatic(res, path.slice("/static/".length));
    return;
  }
  res.writeHead(404).end("not found");
});

// ---------- /ws websocket endpoint ----------

const wss = new WebSocketServer({ server: httpServer, path: "/ws" });

function toBuffer(data: RawData): Buffer {
  if (Buffer.isBuffer(data)) return data;
  if (Array.isArray(data)) return Buffer.concat(data);
  return Buffer.from(data as ArrayBuffer);
}

wss.on("connection", (ws: WebSocket) => {
  log.info("client connected");
  let pipeline: Pipeline | null = null;
  let firstSeen = false;
  // Binary frames that arrive in the same synchronous parse batch as config
  // (app.js starts streaming audio immediately, not gated on `ready`) — buffered
  // so none are dropped in the window before the pipeline exists.
  const preConfigAudio: Buffer[] = [];

  // Attach an error handler up front. During the config handshake an unhandled
  // 'error' (e.g. a malformed frame) would otherwise take down the whole
  // process rather than just this session.
  ws.on("error", (e) => {
    log.warning(`client ws error: ${e instanceof Error ? e.message : String(e)}`);
    pipeline?.onClose();
  });

  ws.on("message", (data: RawData, isBinary: boolean) => {
    if (pipeline) {
      if (isBinary) pipeline.onAudio(toBuffer(data));
      // Text frames after config are ignored (client_pump only reads bytes).
      return;
    }
    if (!firstSeen) {
      firstSeen = true;
      if (isBinary) {
        ws.close(1003, "expected config message first");
        return;
      }
      // startSession is synchronous, so `pipeline` is set before the next queued
      // 'message' fires — no handshake race.
      pipeline = startSession(ws, toBuffer(data), preConfigAudio);
      return;
    }
    // Config attempt made but pipeline not yet running (rare): buffer audio.
    if (isBinary) preConfigAudio.push(toBuffer(data));
  });

  ws.on("close", () => pipeline?.onClose());
});

// Parse + validate the config frame, construct the pipeline, flush any buffered
// pre-config audio, and drive it. Synchronous up to the fire-and-forget run();
// returns the pipeline (or null when the handshake was rejected).
function startSession(ws: WebSocket, firstData: Buffer, preConfigAudio: Buffer[]): Pipeline | null {
  let cfg: Record<string, unknown>;
  try {
    cfg = JSON.parse(firstData.toString("utf-8"));
  } catch {
    ws.close(1003, "expected config message first");
    return null;
  }
  if (typeof cfg !== "object" || cfg === null || cfg.type !== "config") {
    ws.close(1003, "first message must be {type:'config'}");
    return null;
  }

  const targetLang = String(cfg.target_lang ?? "").trim() || DEFAULT_TARGET_LANG;
  const sourceLang = String(cfg.source_lang ?? "auto").trim();
  const scene = String(cfg.scene ?? "").trim();
  const glossary = String(cfg.glossary ?? "").trim();

  // Validate backend choices against the live availability list; fall back to
  // the first available (or fail closed for ASR when nothing is configured).
  let asrBackend = String(cfg.asr_backend ?? "").trim().toLowerCase();
  const asrAvailable = new Set(listAsrBackends().map((b) => b.id));
  if (!asrAvailable.has(asrBackend)) asrBackend = asrAvailable.values().next().value ?? "";
  if (!asrBackend) {
    ws.close(1003, "no ASR backend available — set at least one API key or local binary");
    return null;
  }
  let translateBackend = String(cfg.translate_backend ?? "").trim().toLowerCase();
  const trAvailable = new Set(listTranslateBackends().map((b) => b.id));
  // `none` is always present, so this fallback always lands somewhere.
  if (!trAvailable.has(translateBackend)) translateBackend = trAvailable.values().next().value ?? "none";

  log.info(`config: asr=${asrBackend} tr=${translateBackend} src=${JSON.stringify(sourceLang)} dst=${JSON.stringify(targetLang)} scene=${scene.length}ch gloss=${glossary.length}ch`);

  // Only construct the Gemini client when a chosen backend needs it.
  let gemini: GoogleGenAI | null = null;
  if ((asrBackend === "gemini" || translateBackend === "gemini") && GEMINI_API_KEY) {
    gemini = new GoogleGenAI({ apiKey: GEMINI_API_KEY, httpOptions: { apiVersion: "v1beta" } });
  }

  const pipeline = new Pipeline({ clientWs: ws, gemini, targetLang, sourceLang, scene, glossary, asrBackend, translateBackend });

  // Flush any audio that arrived before the pipeline existed, then run.
  for (const buf of preConfigAudio) pipeline.onAudio(buf);
  preConfigAudio.length = 0;

  void pipeline
    .run()
    .catch((e) => {
      log.exception("pipeline error", e);
      try {
        if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "error", message: "pipeline error" }));
      } catch {
        /* ignore */
      }
    })
    // Pipeline ended (client gone, ASR done, or crash): close the socket so the
    // client stops streaming into a dead session.
    .finally(() => {
      try {
        ws.close();
      } catch {
        /* ignore */
      }
    });

  return pipeline;
}

// ---------- startup ----------

// Fail fast if no ASR backend is configured (mirrors the __main__ guard).
const asrAvail = listAsrBackends();
if (asrAvail.length === 0) {
  log.error(
    "no ASR backend configured — set at least one of GEMINI_API_KEY / OPENAI_API_KEY / DASHSCOPE_API_KEY, " +
      "or point QWEN_ASR_BIN+QWEN_ASR_MODEL_DIR_SMALL (or _LARGE) / VOXTRAL_BIN+VOXTRAL_MODEL_DIR at a local binary",
  );
  process.exit(1);
}
log.info(`ASR backends available: ${asrAvail.map((b) => b.id).join(", ")}`);

httpServer.listen(PORT, HOST, () => {
  log.info(`livesub (ts) listening on http://${HOST}:${PORT}`);
});
