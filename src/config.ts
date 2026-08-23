// Configuration + backend registry — env constants, the translate-backend
// registry (built-ins + livesub.toml), and per-backend availability checks.
// Two path anchors: user config (.env / livesub.toml) resolves against the
// working directory, bundled assets (static/) against the package root — see
// ROOT and PKG_DIR below.

import { accessSync, constants as fsConstants, existsSync, readFileSync, statSync } from "node:fs";
import { dirname, isAbsolute, join } from "node:path";
import { fileURLToPath } from "node:url";
import { config as loadDotenv } from "dotenv";
import { parse as parseToml } from "smol-toml";
import { log } from "./util.js";

const HERE = dirname(fileURLToPath(import.meta.url)); // .../src (dev) or .../dist (built)
// Bundled assets (static/) sit next to the code — repo root in dev, package root
// once installed (src/ and dist/ are each one level below their root).
const PKG_DIR = dirname(HERE);
// User config (.env / livesub.toml / relative model paths) resolves from the
// current working directory, so an installed `livesub` reads the .env where it's
// launched — not from inside node_modules. Overridable via LIVESUB_CONFIG_DIR.
export const ROOT = process.env.LIVESUB_CONFIG_DIR || process.cwd();
loadDotenv({ path: join(ROOT, ".env") });

function env(key: string, def = ""): string {
  const v = process.env[key];
  return v === undefined ? def : v;
}

// Fail fast on a non-numeric env value, matching Python's float(os.environ[...])
// which raises at import rather than silently yielding NaN at runtime.
function envFloat(key: string, def: string): number {
  const raw = env(key, def);
  const n = Number(raw);
  if (raw.trim() === "" || !Number.isFinite(n)) {
    throw new Error(`env ${key} must be a number, got ${JSON.stringify(raw)}`);
  }
  return n;
}

// Resolve a possibly-relative path against ROOT (the config dir). Relative
// model/dir paths in .env are taken relative to where livesub is launched —
// the same directory .env itself is read from — not the package install dir.
export function resolveRel(p: string): string {
  if (!p) return p;
  return isAbsolute(p) ? p : join(ROOT, p);
}

export const GEMINI_API_KEY = env("GEMINI_API_KEY");
export const ASR_MODEL = env("GEMINI_ASR_MODEL", "gemini-2.5-flash-native-audio-latest");
export const TRANSLATE_MODEL = env("GEMINI_TRANSLATE_MODEL", "gemini-2.5-flash-lite");

export const ANTHROPIC_API_KEY = env("ANTHROPIC_API_KEY");
export const DEEPSEEK_API_KEY = env("DEEPSEEK_API_KEY");
export const OPENAI_API_KEY = env("OPENAI_API_KEY");
export const OPENAI_REALTIME_MODEL = env("OPENAI_REALTIME_MODEL", "gpt-realtime-whisper");
export const DASHSCOPE_API_KEY = env("DASHSCOPE_API_KEY");
// DashScope Qwen3-ASR-Flash-Realtime — Alibaba's cloud-hosted Qwen ASR. Two
// regions: international (Singapore) at dashscope-intl, mainland China at
// dashscope. Default to international; override via env if needed.
export const DASHSCOPE_BASE_URL = env(
  "DASHSCOPE_BASE_URL",
  "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime",
);
export const DASHSCOPE_REALTIME_MODEL = env("DASHSCOPE_REALTIME_MODEL", "qwen3-asr-flash-realtime");
// Server VAD threshold for DashScope (range 0-1, lower = catch quieter speech,
// higher = need louder audio). Alibaba documents 0.2; 0.5 is our empirical
// default — tune per environment.
export const DASHSCOPE_VAD_THRESHOLD = envFloat("DASHSCOPE_VAD_THRESHOLD", "0.5");

// ISO-639-1 codes for the source-language hint passed to cloud ASR backends
// (openai-realtime, qwen-cloud). Covers the 11 languages that
// qwen3-asr-flash-realtime supports — the smallest superset across
// hint-accepting backends. Keys are lowercased natural-language names matching
// the frontend <select> values. Unknown keys fall through (pass-through).
export const SOURCE_LANG_ISO: Record<string, string> = {
  chinese: "zh",
  english: "en",
  japanese: "ja",
  korean: "ko",
  spanish: "es",
  french: "fr",
  german: "de",
  italian: "it",
  portuguese: "pt",
  arabic: "ar",
  russian: "ru",
};

export const DEEPSEEK_BASE_URL = env("DEEPSEEK_BASE_URL", "https://api.deepseek.com/anthropic");

export type TranslateSdk = "anthropic" | "openai" | "gemini" | "none";

export interface TranslateBackend {
  id: string;
  label: string;
  sdk: TranslateSdk;
  model: string | null;
  api_key_env?: string;
  base_url?: string;
}

// Translator backends. Single source of truth for the UI dropdown, which SDK to
// talk to, what model, and where to point. Users add their own via livesub.toml
// (same schema).
export const TRANSLATE_BACKENDS_BUILTIN: TranslateBackend[] = [
  { id: "claude-haiku", label: "Claude Haiku", sdk: "anthropic", model: "claude-haiku-4-5", api_key_env: "ANTHROPIC_API_KEY" },
  { id: "claude-sonnet", label: "Claude Sonnet", sdk: "anthropic", model: "claude-sonnet-4-6", api_key_env: "ANTHROPIC_API_KEY" },
  { id: "claude-opus", label: "Claude Opus", sdk: "anthropic", model: "claude-opus-4-7", api_key_env: "ANTHROPIC_API_KEY" },
  {
    id: "deepseek-flash",
    label: "DeepSeek Flash",
    sdk: "anthropic",
    model: env("DEEPSEEK_TRANSLATE_MODEL", "deepseek-v4-flash"),
    api_key_env: "DEEPSEEK_API_KEY",
    base_url: DEEPSEEK_BASE_URL,
  },
  { id: "gemini", label: "Gemini", sdk: "gemini", model: null, api_key_env: "GEMINI_API_KEY" },
  { id: "none", label: "None (transcript only)", sdk: "none", model: null },
];

// Read user-defined backends from livesub.toml (config dir, see ROOT) if present. Skipped
// silently when the file doesn't exist. Each entry must have id/label/sdk/model.
function loadCustomTranslateBackends(): TranslateBackend[] {
  const cfgPath = join(ROOT, "livesub.toml");
  if (!existsSync(cfgPath)) return [];
  let data: unknown;
  try {
    data = parseToml(readFileSync(cfgPath, "utf-8"));
  } catch (e) {
    log.error(`livesub.toml parse failed: ${e instanceof Error ? e.message : String(e)}`);
    return [];
  }
  const out: TranslateBackend[] = [];
  const entries = (data as { translate?: unknown }).translate;
  const list = Array.isArray(entries) ? entries : [];
  for (const entry of list) {
    if (typeof entry !== "object" || entry === null) continue;
    const e = entry as Record<string, unknown>;
    const missing = (["id", "label", "sdk", "model"] as const).filter((k) => !(k in e));
    if (missing.length) {
      log.warning(`livesub.toml: skipping entry missing ${JSON.stringify(missing)}: ${JSON.stringify(e)}`);
      continue;
    }
    if (e.sdk !== "anthropic" && e.sdk !== "openai") {
      log.warning(`livesub.toml: sdk must be 'anthropic' or 'openai', got ${JSON.stringify(e.sdk)} (entry ${JSON.stringify(e.id)})`);
      continue;
    }
    out.push(e as unknown as TranslateBackend);
  }
  return out;
}

// Built-ins first, then user-defined. Order = dropdown order.
export function allTranslateBackends(): TranslateBackend[] {
  return [...TRANSLATE_BACKENDS_BUILTIN, ...loadCustomTranslateBackends()];
}

// Available iff (a) no api_key_env declared (keyless / always-on), or (b) the
// env var is set non-empty.
export function backendAvailable(b: TranslateBackend): boolean {
  if (!b.api_key_env) return true;
  return Boolean(process.env[b.api_key_env]);
}

export function getTranslateBackend(backendId: string): TranslateBackend | null {
  for (const b of allTranslateBackends()) {
    if (b.id === backendId) return b;
  }
  return null;
}

// Disable extended thinking on every translation call. Claude variants default
// to no-thinking already; DeepSeek-V4-flash defaults to thinking ON, which
// inflates output tokens / latency. Passing explicitly keeps behaviour
// consistent across backends.
export const ANTHROPIC_THINKING_DISABLED = { type: "disabled" as const };

// /api/hints (scene-seed -> glossary research) supports two backends. Gemini is
// preferred when GEMINI_API_KEY is set. Claude is the fallback.
export const HINTS_BACKEND = env("HINTS_BACKEND", "auto").trim().toLowerCase();
export const HINTS_CLAUDE_MODEL = "claude-sonnet-4-6";
export const HINTS_GEMINI_MODEL = env("HINTS_GEMINI_MODEL", "gemini-2.5-flash");

export const DEFAULT_TARGET_LANG = "Chinese (Simplified)";
export const HISTORY_PAIRS = 5;

// Sentence-boundary fallbacks when no .!?。！？ shows up:
export const SENTENCE_MAX_CHARS = 50;
export const SENTENCE_IDLE_FLUSH_SEC = 5.0;
// Intra-sentence partial translation pacing (anthropic-/openai-SDK backends).
export const PARTIAL_HARD_FLOOR_SEC = 0.4;
export const PARTIAL_MIN_NEW_BYTES = 24;

// Audio gate: skip chunks whose absolute peak is below this (synthetic silence).
export const AUDIO_GATE_PEAK = 200;
// Pre-buffer length (in 40ms chunks) replayed on silent->speech transition.
export const AUDIO_PREBUFFER_CHUNKS = 12;

// qwen-asr — local C inference subprocess.
export const QWEN_BIN = env("QWEN_ASR_BIN", "qwen_asr");

// voxtral.c — local C inference of Mistral Voxtral Realtime 4B.
export const VOXTRAL_BIN = env("VOXTRAL_BIN", "voxtral");
export const VOXTRAL_MODEL_DIR = env("VOXTRAL_MODEL_DIR", "voxtral-realtime-4b");
export const VOXTRAL_INTERVAL_SEC = envFloat("VOXTRAL_INTERVAL_SEC", "2.0");

// Two qwen3-asr variants ship in the same binary; only the model dir differs.
export const QWEN_MODEL_DIR_SMALL = env("QWEN_ASR_MODEL_DIR_SMALL", "");
export const QWEN_MODEL_DIR_LARGE = env("QWEN_ASR_MODEL_DIR_LARGE", "");

// Frontend assets ship with the package (next to the code). Overridable via
// LIVESUB_STATIC_DIR.
export const STATIC_DIR = process.env.LIVESUB_STATIC_DIR || join(PKG_DIR, "static");

// Hard punctuation: cut here whenever it appears (sentence terminators).
export const SENTENCE_PUNCT = ".!?。！？\n";
// Soft punctuation: cut here only when the buffer is already getting long.
export const SENTENCE_PUNCT_SOFT = ",;、，；";
// Once buffer reaches this length, allow soft-punct cuts.
export const SOFT_CUT_THRESHOLD = 30;

// ---------- availability helpers ----------

function isExecFile(p: string): boolean {
  try {
    if (!statSync(p).isFile()) return false;
    accessSync(p, fsConstants.X_OK);
    return true;
  } catch {
    return false;
  }
}

function isFile(p: string): boolean {
  try {
    return statSync(p).isFile();
  } catch {
    return false;
  }
}

// In-process equivalent of shutil.which — no shell fork (the old execSync
// blocked the event loop on every /api/backends request). A bare name is
// searched on PATH; a path-like value is resolved (abs, or ROOT-relative) and
// checked directly.
export function which(bin: string): string | null {
  if (bin.includes("/")) {
    const abs = resolveRel(bin);
    return isExecFile(abs) ? abs : null;
  }
  for (const dir of (process.env.PATH ?? "").split(":")) {
    if (!dir) continue;
    if (isExecFile(join(dir, bin))) return join(dir, bin);
  }
  return null;
}

// `shutil.which(bin) or bin`, with relative paths resolved against ROOT.
export function resolveBin(bin: string): string {
  const w = which(bin);
  if (w) return w;
  if (bin.includes("/") && !isAbsolute(bin)) return join(ROOT, bin);
  return bin;
}

// A local ASR backend is available iff its binary is callable (on PATH or a
// resolvable file) AND its model directory exists.
export function localAsrAvailable(binPath: string, modelDir: string): boolean {
  if (!modelDir) return false;
  const binOk = which(binPath) !== null || isFile(resolveRel(binPath));
  return binOk && existsSync(resolveRel(modelDir));
}

// Map qwen-small/qwen-large backend id -> the configured model dir.
export function qwenModelDirFor(backendId: string): string {
  if (backendId === "qwen-large") return QWEN_MODEL_DIR_LARGE;
  return QWEN_MODEL_DIR_SMALL;
}

// Translate dropdown contents — registry filtered by env availability.
export function listTranslateBackends(): Array<{ id: string; label: string }> {
  return allTranslateBackends()
    .filter(backendAvailable)
    .map((b) => ({ id: b.id, label: b.label }));
}

// ASR dropdown contents — cloud gated by API key, local gated by binary + model
// dir presence. Local backends listed first so they're the default on a fresh
// install when both local and cloud are available.
export function listAsrBackends(): Array<{ id: string; label: string }> {
  const out: Array<{ id: string; label: string }> = [];
  if (localAsrAvailable(QWEN_BIN, QWEN_MODEL_DIR_SMALL)) out.push({ id: "qwen-small", label: "Qwen (small, local)" });
  if (localAsrAvailable(QWEN_BIN, QWEN_MODEL_DIR_LARGE)) out.push({ id: "qwen-large", label: "Qwen (large, local)" });
  if (localAsrAvailable(VOXTRAL_BIN, VOXTRAL_MODEL_DIR)) out.push({ id: "voxtral", label: "Voxtral (local)" });
  if (DASHSCOPE_API_KEY) out.push({ id: "qwen-cloud", label: "Qwen ASR (Cloud)" });
  if (OPENAI_API_KEY) out.push({ id: "openai-realtime", label: "OpenAI Realtime" });
  if (process.env.GEMINI_API_KEY) out.push({ id: "gemini", label: "Gemini Live" });
  return out;
}

// Return 'gemini' / 'claude' / null based on env override and keys. Default
// priority gemini > claude.
export function pickHintsBackend(): "gemini" | "claude" | null {
  if (HINTS_BACKEND === "gemini") return GEMINI_API_KEY ? "gemini" : null;
  if (HINTS_BACKEND === "claude") return ANTHROPIC_API_KEY ? "claude" : null;
  if (GEMINI_API_KEY) return "gemini";
  if (ANTHROPIC_API_KEY) return "claude";
  return null;
}
