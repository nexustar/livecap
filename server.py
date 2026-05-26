import asyncio
import base64
import codecs
import collections
import json
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types

try:
    import anthropic  # optional: only needed when translate sdk=anthropic
except ImportError:
    anthropic = None

try:
    import openai  # optional: only needed when translate sdk=openai
except ImportError:
    openai = None

try:
    import websockets  # optional: only needed when asr_backend=openai-realtime
except ImportError:
    websockets = None

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("livesub")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
ASR_MODEL = os.environ.get("GEMINI_ASR_MODEL", "gemini-2.5-flash-native-audio-latest")
TRANSLATE_MODEL = os.environ.get("GEMINI_TRANSLATE_MODEL", "gemini-2.5-flash-lite")

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_REALTIME_MODEL = os.environ.get(
    "OPENAI_REALTIME_MODEL", "gpt-realtime-whisper"
)
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY")
# DashScope Qwen3-ASR-Flash-Realtime — Alibaba's cloud-hosted Qwen ASR.
# Two regions: international (Singapore) at dashscope-intl, mainland China
# at dashscope. Default to international; override via env if needed.
DASHSCOPE_BASE_URL = os.environ.get(
    "DASHSCOPE_BASE_URL",
    "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime"
)
DASHSCOPE_REALTIME_MODEL = os.environ.get(
    "DASHSCOPE_REALTIME_MODEL", "qwen3-asr-flash-realtime"
)
# Server VAD threshold for DashScope (range 0-1, lower = catch quieter
# speech, higher = need louder audio). Alibaba documents 0.2; 0.5 is our
# empirical default — tune per environment.
DASHSCOPE_VAD_THRESHOLD = float(
    os.environ.get("DASHSCOPE_VAD_THRESHOLD", "0.5")
)

# ISO-639-1 codes for the source-language hint passed to cloud ASR backends
# (openai-realtime, qwen-cloud). Covers the 11 languages that qwen3-asr-flash-
# realtime supports — the smallest superset across hint-accepting backends, so
# anything in this map works on every cloud backend. openai-realtime is
# Whisper under the hood and accepts ~70 ISO codes if you pass them directly;
# qwen-local takes natural names verbatim and ignores this map. Keys are
# lowercased natural-language names matching the frontend <select> values.
# Unknown keys fall through (pass-through) so debug-mode users typing freeform
# values still reach the upstream API.
SOURCE_LANG_ISO = {
    "chinese": "zh", "english": "en", "japanese": "ja", "korean": "ko",
    "spanish": "es", "french": "fr", "german": "de", "italian": "it",
    "portuguese": "pt", "arabic": "ar", "russian": "ru",
}

DEEPSEEK_BASE_URL = os.environ.get(
    "DEEPSEEK_BASE_URL", "https://api.deepseek.com/anthropic"
)

# Translator backends. Single source of truth: what shows up in the UI
# dropdown, which SDK to talk to it with, what model, where to point. Add
# new built-ins here; users add their own via livesub.toml (same schema).
#
# Fields:
#   id           — stable identifier used in WS config and localStorage
#   label        — display name in the dropdown
#   sdk          — "anthropic" | "openai" | "gemini" | "none"
#   model        — model id passed to the SDK; ignored for "gemini"/"none"
#   api_key_env  — env var that must be set for this backend to appear
#                  (omit for keyless backends — local OpenAI-compat servers,
#                   "none", "gemini" which has its own module-level client)
#   base_url     — optional. Overrides the SDK's default endpoint. Required
#                  for OpenAI-compat servers (ollama, lm-studio, OpenRouter,
#                  DeepSeek's anthropic-compat).
TRANSLATE_BACKENDS_BUILTIN: list[dict] = [
    {"id": "claude-haiku",   "label": "Claude Haiku",
     "sdk": "anthropic", "model": "claude-haiku-4-5",
     "api_key_env": "ANTHROPIC_API_KEY"},
    {"id": "claude-sonnet",  "label": "Claude Sonnet",
     "sdk": "anthropic", "model": "claude-sonnet-4-6",
     "api_key_env": "ANTHROPIC_API_KEY"},
    {"id": "claude-opus",    "label": "Claude Opus",
     "sdk": "anthropic", "model": "claude-opus-4-7",
     "api_key_env": "ANTHROPIC_API_KEY"},
    {"id": "deepseek-flash", "label": "DeepSeek Flash",
     "sdk": "anthropic",
     "model": os.environ.get("DEEPSEEK_TRANSLATE_MODEL", "deepseek-v4-flash"),
     "api_key_env": "DEEPSEEK_API_KEY",
     "base_url": DEEPSEEK_BASE_URL},
    {"id": "gemini", "label": "Gemini",
     "sdk": "gemini", "model": None,
     "api_key_env": "GEMINI_API_KEY"},
    {"id": "none", "label": "None (transcript only)",
     "sdk": "none", "model": None},
]


def _load_custom_translate_backends() -> list[dict]:
    """Read user-defined backends from livesub.toml (next to .env / cwd) if
    present. Skipped silently when the file doesn't exist. Each entry must
    have id/label/sdk/model; api_key_env and base_url are optional.

    Example livesub.toml:
        [[translate]]
        id = "local-qwen"
        label = "Local Qwen 14B (Ollama)"
        sdk = "openai"
        model = "qwen2.5:14b"
        base_url = "http://localhost:11434/v1"
        # api_key_env omitted — ollama doesn't require auth by default
    """
    cfg_path = Path("livesub.toml")
    if not cfg_path.exists():
        return []
    try:
        import tomllib
        data = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception as e:
        log.error("livesub.toml parse failed: %s", e)
        return []
    out: list[dict] = []
    for entry in (data.get("translate") or []):
        if not isinstance(entry, dict):
            continue
        missing = [k for k in ("id", "label", "sdk", "model") if k not in entry]
        if missing:
            log.warning(
                "livesub.toml: skipping entry missing %s: %r", missing, entry,
            )
            continue
        if entry["sdk"] not in ("anthropic", "openai"):
            log.warning(
                "livesub.toml: sdk must be 'anthropic' or 'openai', got %r (entry %r)",
                entry["sdk"], entry.get("id"),
            )
            continue
        out.append(entry)
    return out


def _all_translate_backends() -> list[dict]:
    """Built-ins first, then user-defined. Order = dropdown order."""
    return TRANSLATE_BACKENDS_BUILTIN + _load_custom_translate_backends()


def _backend_available(b: dict) -> bool:
    """Available iff (a) no api_key_env declared (keyless / always-on), or
    (b) the env var is set non-empty."""
    env = b.get("api_key_env")
    if not env:
        return True
    return bool(os.environ.get(env))


def _get_translate_backend(backend_id: str) -> dict | None:
    """Look up a backend by id across built-ins + user-defined."""
    for b in _all_translate_backends():
        if b["id"] == backend_id:
            return b
    return None

# Disable extended thinking on every translation call. Claude variants
# default to no-thinking already (opt-in only); DeepSeek-V4-flash defaults
# to thinking ON, which inflates output tokens 10× / latency 5× and
# sometimes exhausts the budget before any text is emitted. Passing
# explicitly keeps behavior consistent across backends.
ANTHROPIC_EXTRA_BODY = {"thinking": {"type": "disabled"}}

# /api/hints (scene-seed → glossary research) supports two backends.
# Gemini is preferred when GEMINI_API_KEY is set: google_search grounding is
# backed by actual Google Search, has the best long-tail coverage for
# concrete events/people/anime, and Gemini Flash is cheaper than Claude.
# Claude is the fallback when only ANTHROPIC_API_KEY is set. The prompt
# template (build_hints_system_prompt) is backend-agnostic — same [SCENE]/
# [GLOSSARY] format for both. HINTS_BACKEND=gemini|claude env var forces
# a specific backend; default "auto" picks per available key.
HINTS_BACKEND = os.environ.get("HINTS_BACKEND", "auto").strip().lower()
HINTS_CLAUDE_MODEL = "claude-sonnet-4-6"
# Full-size Flash supports google_search grounding; -lite variants don't
# reliably. Override only if you know the target model supports it.
HINTS_GEMINI_MODEL = os.environ.get("HINTS_GEMINI_MODEL", "gemini-2.5-flash")

DEFAULT_TARGET_LANG = "Chinese (Simplified)"
HISTORY_PAIRS = 5

# Sentence-boundary fallbacks when no .!?。！？ shows up:
#   hard cap on buffer length before a forced cut at last whitespace
SENTENCE_MAX_CHARS = 50
#   force-flush if no new transcript chunk arrives for this many seconds.
#   5s (vs original 3s) gives speech with frequent short pauses (singing,
#   anime narration, hesitation-heavy speech) more time to accumulate into
#   meaningful chunks before being dispatched as a fragment.
SENTENCE_IDLE_FLUSH_SEC = 5.0
# Intra-sentence partial translation (used by Claude / OpenAI-SDK
# backends — Claude variants, DeepSeek, OpenAI direct, etc.).
# Trigger a fresh partial translation of the current buffer when BOTH:
#   - this many seconds have passed since the last partial fired
#   - this many new UTF-8 bytes have been appended since the last partial
# Why bytes, not characters: ASCII is 1 byte, CJK is 3 bytes in UTF-8.
# Speech rate in "bytes/word" is roughly comparable across English / CJK
# (~5–8 B/word), while "chars/word" differs by ~5×. Using bytes gives a
# consistent "fire on ~a phrase of new content" semantics regardless of
# source language. 24 ≈ 5 English words or 8 CJK characters.
# HARD_FLOOR is the only wall-clock gate left: it exists to bound API call
# rate (and prevent runaway cost on a chatty cloud ASR), nothing else. The
# real throttle is now "previous partial finished" — see _maybe_trigger_partial.
# Why this matters: a fixed 2s interval was prone to phase-locking against
# bursty ASR backends (qwen-small / voxtral / openai-realtime whisper all
# emit text in ~2s batches). When the two periods aligned within jitter,
# every other partial got skipped — effective interval doubled to ~4s.
# The vsync-style gate (drop if last partial still in flight) makes the
# partial cadence track whichever is slower of (ASR batch rate, translator
# round-trip), with no fixed periodicity to beat against.
PARTIAL_HARD_FLOOR_SEC = 0.4
PARTIAL_MIN_NEW_BYTES = 24

# Skip audio chunks whose absolute peak is below this. Catches the case
# where the input is synthetic silence (peak=0, e.g. between songs / video
# transitions / muted source). Real speech / ambient mic noise is always
# well above this. Currently applied to qwen-asr only — Gemini Live has its
# own server-side VAD that benefits from seeing continuous audio.
AUDIO_GATE_PEAK = 200
# Pre-buffer length (in 40ms chunks) — when transitioning silent→speech,
# replay this many recent gated chunks to qwen first. Catches the leading
# consonant of words (plosives /p/t/k/, fricatives /s/sh/f/) whose onset
# energy can fall under the gate threshold even though the word starts
# there. Worth ~half a phoneme of context. 12 chunks = 480ms.
AUDIO_PREBUFFER_CHUNKS = 12

# qwen-asr (https://github.com/antirez/qwen-asr) — local C inference
# subprocess. Stock antirez/qwen-asr works fine since we no longer pass
# --repeat-penalty; our fork just has the flag as a no-op.
QWEN_BIN = os.environ.get("QWEN_ASR_BIN", "qwen_asr")

# voxtral.c (https://github.com/antirez/voxtral.c) — local C inference of
# Mistral Voxtral Realtime 4B. Larger and more accurate than Qwen-0.6B but
# 8.9 GB model and ~2.5x realtime on Apple M3 Max. No prompt / language /
# past-text / repeat-penalty flags — voxtral is a much simpler CLI than
# qwen-asr. Only knobs: model dir + processing interval.
VOXTRAL_BIN = os.environ.get("VOXTRAL_BIN", "voxtral")
VOXTRAL_MODEL_DIR = os.environ.get(
    "VOXTRAL_MODEL_DIR", "voxtral-realtime-4b"
)
# `-I <secs>`: latency / efficiency knob, voxtral's own default is 2.0.
VOXTRAL_INTERVAL_SEC = float(os.environ.get("VOXTRAL_INTERVAL_SEC", "2.0"))

# Two qwen3-asr variants ship in the same binary; only the model dir
# differs. Users wire whichever (or both) they downloaded with
# `./download_model.sh --model small|large` in qwen-asr. Each one that
# resolves to an existing directory shows up as its own ASR option in the
# UI — small is faster + lighter, large is more accurate.
QWEN_MODEL_DIR_SMALL = os.environ.get("QWEN_ASR_MODEL_DIR_SMALL", "")
QWEN_MODEL_DIR_LARGE = os.environ.get("QWEN_ASR_MODEL_DIR_LARGE", "")

STATIC_DIR = Path(__file__).parent / "static"
# Hard punctuation: cut here whenever it appears (sentence terminators).
SENTENCE_PUNCT = ".!?。！？\n"
# Soft punctuation: cut here only when the buffer is already getting long
# (commas, semicolons in EN/JP/ZH). Prevents the 80-char fallback from
# cutting in the middle of a CJK word.
SENTENCE_PUNCT_SOFT = ",;、，；"
# Once buffer reaches this length, allow soft-punct cuts.
SOFT_CUT_THRESHOLD = 30

app = FastAPI()


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


# ---------- /api/backends: tell the frontend which backends are usable ----------

def _local_asr_available(bin_path: str, model_dir: str) -> bool:
    """A local ASR backend is available iff its binary is callable (on PATH
    or an absolute file) AND its model directory exists. Both have to be
    true — a binary without weights crashes on first audio chunk."""
    if not model_dir:
        return False
    bin_ok = bool(shutil.which(bin_path)) or Path(bin_path).is_file()
    return bin_ok and Path(model_dir).exists()


def _qwen_model_dir_for(backend_id: str) -> str:
    """Map qwen-small/qwen-large backend id → the configured model dir."""
    if backend_id == "qwen-large":
        return QWEN_MODEL_DIR_LARGE
    return QWEN_MODEL_DIR_SMALL


def _list_translate_backends() -> list[dict]:
    """Translate dropdown contents — registry filtered by env availability.
    Built-ins first, user-defined (from livesub.toml) appended after."""
    return [
        {"id": b["id"], "label": b["label"]}
        for b in _all_translate_backends()
        if _backend_available(b)
    ]


def _list_asr_backends() -> list[dict]:
    """ASR dropdown contents — cloud backends gated by API key, local ones
    gated by binary + model dir presence. Local backends listed first so
    they're the default on a fresh install when both local and cloud are
    available."""
    out: list[dict] = []
    if _local_asr_available(QWEN_BIN, QWEN_MODEL_DIR_SMALL):
        out.append({"id": "qwen-small", "label": "Qwen (small, local)"})
    if _local_asr_available(QWEN_BIN, QWEN_MODEL_DIR_LARGE):
        out.append({"id": "qwen-large", "label": "Qwen (large, local)"})
    if _local_asr_available(VOXTRAL_BIN, VOXTRAL_MODEL_DIR):
        out.append({"id": "voxtral", "label": "Voxtral (local)"})
    if DASHSCOPE_API_KEY:
        out.append({"id": "qwen-cloud", "label": "Qwen ASR (Cloud)"})
    if OPENAI_API_KEY:
        out.append({"id": "openai-realtime", "label": "OpenAI Realtime"})
    if os.environ.get("GEMINI_API_KEY"):
        out.append({"id": "gemini", "label": "Gemini Live"})
    return out


@app.get("/api/backends")
async def list_backends():
    return {
        "asr": _list_asr_backends(),
        "translate": _list_translate_backends(),
        # Whether /api/hints can serve a request (any usable hints backend
        # configured). Frontend uses this to hide the scene-seed row when
        # neither GEMINI_API_KEY nor ANTHROPIC_API_KEY is set, since the
        # Research button would just error.
        "hints_available": _pick_hints_backend() is not None,
    }


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------- /api/hints: research a scene description and produce a keyword list ----------

def build_hints_system_prompt(source_lang: str) -> str:
    """System prompt for the Generate-from-scene-seed flow. The glossary
    LANGUAGE matters: it's fed verbatim into qwen-asr's --prompt, which is
    a token-level bias for the AUDIO language. A Chinese / English glossary
    won't bias Japanese audio recognition. So if the user has set a
    source_lang (other than auto), we pin the glossary to that language."""
    src = (source_lang or "").strip()
    if not src or src.lower() == "auto":
        glossary_lang_rule = (
            "Use each entry's NATIVE script (Japanese in kanji/kana, "
            "Chinese in hanzi, Korean in hangul, etc.). Don't transliterate."
        )
        glossary_lang_hint = ""
    else:
        glossary_lang_rule = (
            f"ALL entries must be in {src}. The audio source language is "
            f"{src}; the glossary is consumed by an ASR system that biases "
            f"{src} tokens. Entries in any other language (including the "
            f"target translation language) are useless and may distort "
            f"recognition. If you only know an entry's English / Chinese "
            f"transliteration, search the web to find the {src} original "
            f"(or omit it)."
        )
        glossary_lang_hint = f", all in {src}"
    return (
        "You are an assistant for a real-time ASR + translation system. "
        "Given brief user input — a topic, URL, keyword, or short "
        "description — you produce TWO outputs:\n\n"
        "1. SCENE: a 1–3 sentence summary describing the scenario, genre, "
        "speakers, and tone. This is given to the translator as background "
        "context so it can pick appropriate register and word choice. "
        "Mention any cultural conventions or fan-slang that affect how "
        "things should be translated (e.g. \"use casual Chinese fan-slang; "
        "preserve idol terminology like 推し / ペンライト\").\n\n"
        "2. GLOSSARY: a comma-separated list of proper nouns, names, and "
        "specialty terms that may appear. "
        f"{glossary_lang_rule} "
        "Used as a soft prompt to bias the speech recognizer at the token "
        "level — the prompt language MUST match the audio language for the "
        "bias to work.\n\n"
        "Rules:\n"
        "- Search the web when the input refers to a specific real-world "
        "thing (concert, show, person, event, anime, sports, technical "
        "topic). Don't fabricate; only list what you verify or know.\n"
        "- For generic scenarios where research adds nothing, just answer "
        "from training.\n"
        "- GLOSSARY: 30–100 entries, most distinctive proper nouns first.\n"
        "- SCENE: concise. Skip generic stuff. Include things that affect "
        "translation register.\n\n"
        "Output format (use the exact delimiters, both blocks required, no "
        "other text):\n"
        "[SCENE]\n<1–3 sentence scene summary>\n[/SCENE]\n"
        f"[GLOSSARY]\n<comma-separated terms{glossary_lang_hint}>\n[/GLOSSARY]\n"
    )


def _parse_hints_response(raw: str) -> tuple[str, str]:
    """Extract [SCENE] and [GLOSSARY] blocks from a hints model response.
    Falls back to treating the whole text as glossary when neither delimiter
    block parses (some models occasionally skip the wrappers under brief
    inputs). Same regex for Claude and Gemini paths — template is shared."""
    scene_m = re.search(r"\[SCENE\]\s*(.*?)\s*\[/SCENE\]", raw, re.DOTALL)
    gloss_m = re.search(r"\[GLOSSARY\]\s*(.*?)\s*\[/GLOSSARY\]", raw, re.DOTALL)
    scene = scene_m.group(1).strip() if scene_m else ""
    glossary = gloss_m.group(1).strip() if gloss_m else ""
    if not scene and not glossary:
        glossary = raw
    return scene, glossary


def _pick_hints_backend() -> str | None:
    """Return 'gemini' / 'claude' / None based on env override and keys.
    Default priority gemini > claude — Gemini grounding uses real Google
    Search (better long-tail coverage for concrete events/names) and Flash
    is cheaper than Haiku."""
    if HINTS_BACKEND == "gemini":
        return "gemini" if GEMINI_API_KEY else None
    if HINTS_BACKEND == "claude":
        return "claude" if (ANTHROPIC_API_KEY and anthropic is not None) else None
    # auto / anything-else: try gemini first, claude second.
    if GEMINI_API_KEY:
        return "gemini"
    if ANTHROPIC_API_KEY and anthropic is not None:
        return "claude"
    return None


async def _generate_hints_claude(description: str, source_lang: str) -> dict:
    """Run the hints flow via Claude Haiku + Anthropic's web_search tool."""
    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    msg = await client.messages.create(
        model=HINTS_CLAUDE_MODEL,
        max_tokens=1024,
        system=build_hints_system_prompt(source_lang),
        tools=[{
            # Anthropic's server-side web search tool. Claude decides when to
            # call it; max_uses caps cost.
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 3,
        }],
        messages=[{"role": "user", "content": f"Scene description: {description}"}],
    )
    # Concatenate all text blocks; skip tool_use blocks.
    text_parts: list[str] = []
    searches_used = 0
    for block in msg.content:
        btype = getattr(block, "type", None)
        if btype == "text":
            text_parts.append(block.text)
        elif btype == "server_tool_use" and getattr(block, "name", "") == "web_search":
            searches_used += 1
    scene, glossary = _parse_hints_response("".join(text_parts).strip())
    return {
        "scene": scene, "glossary": glossary,
        "searches_used": searches_used,
        "model": HINTS_CLAUDE_MODEL,
        "stop_reason": msg.stop_reason,
    }


async def _generate_hints_gemini(description: str, source_lang: str) -> dict:
    """Run the hints flow via Gemini Flash + google_search grounding. SDK
    shape parallels the existing _translate_one_gemini call path — the only
    new piece is the tools=[Tool(google_search=GoogleSearch())] config and
    pulling grounding_metadata for the search-count log."""
    client = genai.Client(api_key=GEMINI_API_KEY)
    config = types.GenerateContentConfig(
        system_instruction=build_hints_system_prompt(source_lang),
        tools=[types.Tool(google_search=types.GoogleSearch())],
    )
    resp = await client.aio.models.generate_content(
        model=HINTS_GEMINI_MODEL,
        contents=f"Scene description: {description}",
        config=config,
    )
    raw = (resp.text or "").strip()
    # web_search_queries is the list of actual queries Gemini issued — its
    # length is the analogue of Claude's server_tool_use count.
    searches_used = 0
    finish_reason: str | None = None
    try:
        cand = resp.candidates[0]
        finish_reason = str(getattr(cand, "finish_reason", "") or "") or None
        gm = getattr(cand, "grounding_metadata", None)
        if gm is not None:
            queries = getattr(gm, "web_search_queries", None) or []
            searches_used = len(queries)
    except (AttributeError, IndexError):
        pass
    scene, glossary = _parse_hints_response(raw)
    return {
        "scene": scene, "glossary": glossary,
        "searches_used": searches_used,
        "model": HINTS_GEMINI_MODEL,
        "stop_reason": finish_reason,
    }


@app.post("/api/hints")
async def generate_hints(request: Request):
    backend = _pick_hints_backend()
    if backend is None:
        return JSONResponse({
            "error": (
                "Hint generation requires GEMINI_API_KEY or "
                "ANTHROPIC_API_KEY. Set one in .env and restart, "
                "or fill Scene/Glossary manually below."
            ),
        }, status_code=500)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)
    description = (body.get("description") or "").strip()
    if not description:
        return JSONResponse({"error": "description required"}, status_code=400)
    if len(description) > 500:
        return JSONResponse({"error": "description too long (>500 chars)"}, status_code=400)
    source_lang = (body.get("source_lang") or "").strip()

    log.info(
        "generating hints via %s for scene: %r (source_lang=%r)",
        backend, description[:100], source_lang or "auto",
    )
    try:
        if backend == "gemini":
            result = await _generate_hints_gemini(description, source_lang)
        else:
            result = await _generate_hints_claude(description, source_lang)
    except Exception as e:
        log.exception("hint generation via %s failed", backend)
        return JSONResponse(
            {"error": str(e), "backend": backend}, status_code=502,
        )

    log.info(
        "hints generated via %s: scene=%dch glossary=%dch %d web searches stop=%s",
        backend, len(result["scene"]), len(result["glossary"]),
        result["searches_used"], result.get("stop_reason"),
    )
    return JSONResponse({
        "scene": result["scene"],
        "glossary": result["glossary"],
        "searches_used": result["searches_used"],
        "model": result["model"],
        "backend": backend,
    })


# ---------- prompts ----------

def build_asr_prompt(source_lang: str, scene: str, glossary: str) -> str:
    """Gemini Live ASR system prompt. Native-audio Live models don't accept a
    language_code field, so we steer language detection via the prompt.
    Scene + glossary together provide background context for biasing.
    """
    src = (source_lang or "").strip().lower()
    if src in ("", "auto"):
        lang_hint = (
            "The audio may be in any language; detect it automatically."
        )
    else:
        lang_hint = (
            f"The audio is primarily in {source_lang}. Transcribe spoken "
            f"{source_lang} accurately, even if there is background music, "
            "applause, sound effects, or multiple voices."
        )

    base = (
        "You are a passive listener. Your only job is to allow accurate input "
        "transcription of the audio you receive. "
        f"{lang_hint} "
        "Stay completely silent. Do not respond. Do not generate any audio "
        "output. Just listen."
    )
    if scene:
        base += f"\n\nScene: {scene}"
    if glossary:
        base += f"\n\nGlossary (proper nouns / specialty terms): {glossary}"
    return base


def build_qwen_prompt(glossary: str) -> str:
    """qwen-asr --prompt: terminology / spelling bias for the decoder.
    Glossary only — NOT scene. Two reasons:
      1. The flag is designed for term lists ("Preserve spelling: CPU, CUDA…"),
         not narrative context. Models use it to nudge token probability;
         a paragraph of natural-language scene description gives it nothing
         to bias toward.
      2. Scene is typically generated in the user's target language (Chinese/
         English), but the audio is in source_lang (Japanese/etc). Mixing
         languages in the prompt confuses qwen's language detection — at
         session start, qwen sometimes emits a few tokens of the prompt
         language before settling onto audio. We saw this as "English
         hallucinations at the very beginning of a Japanese transcription".
    Scene is still used as system instruction for the translator path."""
    return glossary[:500] if glossary else ""


def build_translation_system_instruction(
    target_lang: str, source_lang: str, scene: str
) -> str:
    parts = [
        f"You are a translator. Translate the user's utterance into {target_lang}. "
        "Output ONLY the translation. No preamble, no quotes, no commentary. "
        "Keep technical terms, brand names, and proper nouns in their original "
        "language when natural."
    ]
    src = (source_lang or "").strip()
    if src and src.lower() != "auto":
        parts.append(f"Source language: {src}.")
    if scene:
        parts.append(f"Scene: {scene}")
    return "\n\n".join(parts)


# Paired-translation system instruction (Claude only). Per-turn the model
# also reconsiders the IMMEDIATELY PREVIOUS sentence's translation given the
# new context — when the new utterance reveals a wrong word sense, completes
# a mid-clause cut, or otherwise makes the prev translation misleading.
# 3-shot prompt validated 10/10 against a 10-case battery on Haiku 4.5.
def build_paired_translation_system_instruction(
    target_lang: str, source_lang: str, scene: str
) -> str:
    head = (
        f"Real-time {target_lang} interpreter. Each turn, also reconsider "
        "the previous translation given the new sentence.\n\n"
        "Default KEEP — [PREV] must be byte-identical to the input. REVISE "
        "only when the new sentence reveals: a wrong word sense, a "
        "mid-clause/number cut completed by new, or actual ambiguity. "
        "Stylistic tweaks are NOT a reason — they cause UI flicker.\n\n"
        "[CURR] must NEVER be empty. Even if the new sentence is "
        "fragmentary, garbled, a song-lyric piece, or unclear, output a "
        "best-effort translation in [CURR]. The UI shows [CURR] as the "
        "live caption for this turn — empty [CURR] = blank caption = bug."
    )
    src = (source_lang or "").strip()
    if src and src.lower() != "auto":
        head += f"\n\nSource language: {src}."
    if scene:
        head += f"\n\nScene: {scene}"
    return (
        head
        + "\n\nOutput ONLY the three blocks:\n"
        "[D]keep[/D] or [D]revise[/D]\n"
        "[CURR]<translation of new sentence>[/CURR]\n"
        "[PREV]<verbatim or revised>[/PREV]\n\n"
        "Examples:\n\n"
        "(KEEP — prev complete, new unrelated)\n"
        "prev_src: 今日は本当に楽しかったです。\n"
        "prev_dst: 今天真的很开心。\n"
        "new_src:  では、次に行きましょう。\n"
        "→ [D]keep[/D]\n"
        "  [CURR]那么，我们继续下一个吧。[/CURR]\n"
        "  [PREV]今天真的很开心。[/PREV]\n\n"
        "(REVISE — mid-clause cut, new completes thought)\n"
        "prev_src: 私たちは慎重にこの問題を扱う必要があると\n"
        "prev_dst: 我们需要慎重处理这个问题，\n"
        "new_src:  思っていますが、時間がかかっても価値があります。\n"
        "→ [D]revise[/D]\n"
        "  [CURR]虽然会花时间，但这是值得的。[/CURR]\n"
        "  [PREV]我们认为需要慎重处理这个问题，[/PREV]\n\n"
        "(REVISE — figurative meaning revealed)\n"
        "prev_src: このチームは本当に熱いですね。\n"
        "prev_dst: 这支队伍真的很热。\n"
        "new_src:  試合を諦めずに最後まで戦い抜きました。\n"
        "→ [D]revise[/D]\n"
        "  [CURR]他们没有放弃，战斗到了最后一刻。[/CURR]\n"
        "  [PREV]这支队伍真的很有热情。[/PREV]\n"
    )


def build_translation_prompt(
    sentence: str, history: list[tuple[str, str]], is_partial: bool = False
) -> str:
    parts = []
    if history:
        ctx = "\n".join(f"  {src} → {dst}" for src, dst in history)
        parts.append(
            "Previous translations in this session (for terminology and "
            f"pronoun consistency, do not re-translate them):\n{ctx}\n\n"
        )
    if is_partial:
        parts.append(
            "PARTIAL utterance — speaker is still mid-sentence, more text "
            "will arrive. Translate what is given so far. Output only the "
            f"translation:\n{sentence}"
        )
    else:
        parts.append(f"Now translate this new utterance:\n{sentence}")
    return "".join(parts)


# ---------- helpers ----------


def find_last_punct(text: str, include_soft: bool = False) -> int:
    """Last index of any sentence-ending punctuation in text, or -1.
    With include_soft=True, commas/semicolons (EN/JP/ZH) also count."""
    chars = SENTENCE_PUNCT + (SENTENCE_PUNCT_SOFT if include_soft else "")
    last = -1
    for p in chars:
        i = text.rfind(p)
        if i > last:
            last = i
    return last


def chunk_peak(chunk: bytes) -> int:
    """Absolute peak sample value in a 16-bit s16le PCM chunk (0..32767)."""
    if len(chunk) < 2:
        return 0
    return max(
        abs(int.from_bytes(chunk[i:i + 2], "little", signed=True))
        for i in range(0, len(chunk), 2)
    )


class PairedStreamParser:
    """Incrementally parses Claude's [D][/D][CURR][/CURR][PREV][/PREV]
    paired-translation response. Streams CURR's inner text as soon as it
    arrives (chars that COULD be the start of `[/CURR]` are held back so
    we never emit a partial closing tag). [D] and [PREV] are extracted
    from the full buffer in finalize()."""

    _CLOSE_CURR = "[/CURR]"

    def __init__(self):
        self.full = ""           # everything fed
        self.cursor = 0          # next position to consider for CURR text
        self.state = "before_curr"   # before_curr → in_curr → done

    def feed(self, chunk: str) -> str:
        """Returns whatever CURR text became safe to emit during this feed."""
        self.full += chunk
        out = []
        while True:
            if self.state == "before_curr":
                idx = self.full.find("[CURR]", self.cursor)
                if idx < 0:
                    break
                self.cursor = idx + len("[CURR]")
                # skip leading whitespace inside the block
                while (self.cursor < len(self.full)
                       and self.full[self.cursor] in " \t\n"):
                    self.cursor += 1
                self.state = "in_curr"
            if self.state == "in_curr":
                end = self.full.find(self._CLOSE_CURR, self.cursor)
                if end >= 0:
                    text = self.full[self.cursor:end].rstrip()
                    if text:
                        out.append(text)
                    self.cursor = end + len(self._CLOSE_CURR)
                    self.state = "done"
                    continue
                # No closing tag yet. Hold back chars that could be the
                # start of "[/CURR]" — emit everything else.
                tail = len(self.full)
                for i in range(min(len(self._CLOSE_CURR) - 1,
                                   tail - self.cursor), 0, -1):
                    if self.full.endswith(self._CLOSE_CURR[:i]):
                        tail -= i
                        break
                if tail > self.cursor:
                    out.append(self.full[self.cursor:tail])
                    self.cursor = tail
                break
            if self.state == "done":
                break
        return "".join(out)

    def finalize(self) -> dict:
        d_m = re.search(r"\[D\]\s*(\w+)\s*\[/D\]", self.full, re.IGNORECASE)
        p_m = re.search(r"\[PREV\]\s*(.*?)\s*\[/PREV\]", self.full, re.DOTALL)
        c_m = re.search(r"\[CURR\]\s*(.*?)\s*\[/CURR\]", self.full, re.DOTALL)
        return {
            "decision": d_m.group(1).strip().lower() if d_m else None,
            "prev": p_m.group(1).strip() if p_m else None,
            "curr": c_m.group(1).strip() if c_m else "",
        }


_RETRY_RE = re.compile(r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s?")


def _extract_retry_delay(msg: str, default: float = 5.0) -> float:
    """Pull retryDelay out of a Google API 429 error message and clamp it."""
    m = _RETRY_RE.search(msg)
    if not m:
        return default
    try:
        v = float(m.group(1))
        return min(max(v, 1.0), 30.0)
    except ValueError:
        return default


# ---------- pipeline ----------


@dataclass
class Pipeline:
    client_ws: WebSocket
    # gemini client is only created lazily when an active backend actually
    # needs it (ASR=gemini or translate=gemini). None means "no Gemini key
    # configured and no Gemini backend selected".
    gemini: "genai.Client | None"
    target_lang: str
    source_lang: str
    scene: str = ""
    glossary: str = ""
    asr_backend: str = "gemini"
    translate_backend: str = "gemini"
    claude: "anthropic.AsyncAnthropic | None" = None
    openai_client: "openai.AsyncOpenAI | None" = None

    audio_queue: "asyncio.Queue[bytes | None]" = field(default_factory=asyncio.Queue)
    sentence_queue: "asyncio.Queue[tuple[int, str, int] | None]" = field(default_factory=asyncio.Queue)
    history: list[tuple[str, str]] = field(default_factory=list)
    sentence_buffer: str = ""
    last_chunk_time: float = 0.0
    next_sid: int = 0
    resumption_handle: str | None = None
    stop_requested: bool = False
    stats_chunks: int = 0
    stats_bytes: int = 0
    # Per-segment timing
    current_seg_start: float = 0.0     # wall time when current sentence began accumulating
    current_seg_first_chunk: float = 0.0  # wall time when first transcript chunk for current sentence arrived
    # Per-ASR-session perf tracking
    asr_started_at: float = 0.0       # wall time when ASR session opened
    chunks_fed_to_asr: int = 0        # chunks actually sent to ASR (excludes silence-gated)
    # Partial-translation state (for Claude backend)
    global_rev: int = 0
    last_partial_time: float = 0.0
    last_partial_buf_len: int = 0
    in_flight_partial: "asyncio.Task | None" = None
    # Paired-translation state (Claude backend only): the most recently
    # FINALIZED sentence is sent as the "prev" for the next sentence's
    # paired translation, giving Claude a chance to revise it. None on
    # first turn (no prev exists yet).
    paired_prev_sid: int | None = None
    paired_prev_src: str = ""
    paired_prev_dst: str = ""

    # ----- audio ingest from client websocket -----

    async def client_pump(self):
        try:
            while True:
                msg = await self.client_ws.receive()
                if msg["type"] == "websocket.disconnect":
                    return
                audio = msg.get("bytes")
                if audio:
                    await self.audio_queue.put(audio)
        except WebSocketDisconnect:
            pass
        finally:
            self.stop_requested = True
            await self.audio_queue.put(None)

    # ----- ASR worker: continuous Live API for input transcription only -----

    def _build_asr_config(self) -> types.LiveConnectConfig:
        return types.LiveConnectConfig(
            # Native-audio Live models require AUDIO modality. We instruct the
            # model to stay silent via the system prompt and ignore any audio
            # bytes it does emit. Only input_audio_transcription matters here.
            response_modalities=["AUDIO"],
            system_instruction=types.Content(
                parts=[types.Part(text=build_asr_prompt(self.source_lang, self.scene, self.glossary))]
            ),
            input_audio_transcription=types.AudioTranscriptionConfig(),
            context_window_compression=types.ContextWindowCompressionConfig(
                sliding_window=types.SlidingWindow(),
            ),
            session_resumption=types.SessionResumptionConfig(handle=self.resumption_handle),
        )

    async def asr_worker(self):
        try:
            if self.asr_backend in ("qwen-small", "qwen-large"):
                await self._qwen_asr_worker()
            elif self.asr_backend == "voxtral":
                await self._voxtral_asr_worker()
            elif self.asr_backend == "openai-realtime":
                await self._openai_realtime_asr_worker()
            elif self.asr_backend == "qwen-cloud":
                await self._dashscope_asr_worker()
            else:
                while not self.stop_requested:
                    try:
                        await self._run_one_asr_session()
                    except Exception:
                        log.exception("ASR session crashed; will reopen")
                        await asyncio.sleep(0.5)
        finally:
            # flush whatever's left and signal translator to stop
            await self._finalize_pending_sentence()
            await self.sentence_queue.put(None)
            log.info(
                "ASR worker exiting (chunks=%d bytes=%d)",
                self.stats_chunks, self.stats_bytes,
            )

    # ----- qwen-asr backend (local subprocess) -----

    async def _qwen_asr_worker(self):
        bin_path = shutil.which(QWEN_BIN) or QWEN_BIN
        if not Path(bin_path).is_file():
            await self._safe_send_json({
                "type": "error",
                "message": (
                    f"qwen_asr binary not found (looked for {QWEN_BIN!r}). "
                    "Build from https://github.com/antirez/qwen-asr and set "
                    "QWEN_ASR_BIN env var."
                ),
            })
            return
        model_dir = _qwen_model_dir_for(self.asr_backend)
        env_name = (
            "QWEN_ASR_MODEL_DIR_LARGE" if self.asr_backend == "qwen-large"
            else "QWEN_ASR_MODEL_DIR_SMALL"
        )
        if not model_dir or not Path(model_dir).exists():
            await self._safe_send_json({
                "type": "error",
                "message": (
                    f"qwen-asr model dir not found for {self.asr_backend} "
                    f"(checked {model_dir!r}). Run ./download_model.sh in "
                    f"the qwen-asr repo and set {env_name} env var."
                ),
            })
            return

        # NOTE: don't pass --silent. In qwen-asr's main.c that flag also nulls
        # out the streaming token callback (qwen_set_token_callback(ctx,NULL))
        # — i.e. it disables per-token fflush'ing to stdout, so we'd never see
        # output until EOF. We discard stderr separately below.
        # 4s beats the binary's 2s default for speed and latency on
        # mainstream hardware; 0.6B is fine on 2s, so don't override.
        chunk_sec = "4" if self.asr_backend == "qwen-large" else ""
        qwen_cmd = [
            bin_path,
            "-d", model_dir,
            "--stdin",
            "--stream",
            "--stream-max-new-tokens", "32",
            # Drop long silent spans before inference. Without this, qwen
            # tends to "transcribe" the --prompt content during silence
            # (autoregressive model + no audio signal = it continues
            # generating from the conditioning context, which is the prompt).
            # Same class of bug as Whisper's initial_prompt echo.
            "--skip-silence",
            # Feed previously decoded text back as conditioning for the
            # next chunk — improves cross-chunk name / sentence continuity.
            # Requires the patched fork of antirez/qwen-asr that fixes the
            # weak-audio conditioning lock failure (whole-phrase repetition
            # across many segments).
            "--past-text", "yes",
        ]
        # Repetition penalty was tried (our fork added a --repeat-penalty
        # flag) but turned out to make things worse: with past_text=no the
        # decoder doesn't rut, and any penalty > 1 introduces forced token
        # alternation (ABAB-style artifacts). 1.0 wins in every scenario
        # we tested, so we stopped passing the flag. The C patch in the
        # fork stays as a no-op — if you ever want to re-test, see
        # TECH_DOC for the history.
        src = (self.source_lang or "").strip().lower()
        if src and src != "auto":
            # qwen-asr expects the language name (e.g. "Japanese", "Chinese")
            qwen_cmd.extend(["--language", self.source_lang])
        qwen_prompt = build_qwen_prompt(self.glossary)
        if qwen_prompt:
            qwen_cmd.extend(["--prompt", qwen_prompt])
        if chunk_sec:
            qwen_cmd.extend(["--stream-chunk-sec", chunk_sec])

        # qwen-asr block-buffers stdout when piped (no TTY), so transcript
        # tokens don't reach us until ~4KB accumulate. Wrap with stdbuf to
        # force line buffering. Falls back to raw command if stdbuf missing.
        stdbuf = shutil.which("stdbuf")
        cmd = [stdbuf, "-oL"] + qwen_cmd if stdbuf else qwen_cmd
        log.info("starting qwen-asr: %s", " ".join(cmd))
        await self._safe_send_json({"type": "asr_session", "state": "open"})

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except FileNotFoundError as e:
            await self._safe_send_json({
                "type": "error",
                "message": f"failed to spawn qwen_asr: {e}",
            })
            return

        if self.asr_started_at == 0.0:
            self.asr_started_at = time.monotonic()

        # Rolling buffer of recent chunks used as pre-buffer at speech onset.
        # When we transition silent→speech, we replay these so qwen sees the
        # leading edge of the utterance (consonants whose energy is below
        # the gate threshold).
        prebuffer: collections.deque[bytes] = collections.deque(
            maxlen=AUDIO_PREBUFFER_CHUNKS
        )
        prev_gated = True   # treat session start as if we were just silent

        # Track audio peaks per ~500ms window for the UI level meter so the
        # mic badge updates 2 Hz with the loudest chunk in that window. The
        # 10-s perf log uses its own snapshot independently.
        window_peak = 0
        window_count = 0

        async def feed():
            nonlocal prev_gated, window_peak, window_count
            try:
                while True:
                    chunk = await self.audio_queue.get()
                    if chunk is None:
                        break
                    self.stats_chunks += 1
                    self.stats_bytes += len(chunk)
                    peak = chunk_peak(chunk)
                    # Audio gate: skip silent chunks. Without this qwen tends
                    # to emit hallucinated repetition during long quiet spans
                    # (autoregressive model conditioning on its own past
                    # output continues generating the same filler tokens).
                    gated = peak < AUDIO_GATE_PEAK
                    prebuffer.append(chunk)  # always — used at next onset
                    if not gated:
                        self.chunks_fed_to_asr += 1
                    # UI level meter: emit max-peak over a rolling ~500 ms
                    # window so the badge updates 2 Hz instead of every 10 s
                    # and so it shows the LOUDEST chunk in the window (which
                    # is what speech actually peaks at), not the random one
                    # that happened to land on a stats sample.
                    if peak > window_peak:
                        window_peak = peak
                    window_count += 1
                    if window_count >= 12:    # 12 × 40 ms ≈ 500 ms
                        await self._safe_send_json({
                            "type": "audio_stats",
                            "peak": window_peak,
                            "chunks_per_sec": int(window_count / 0.5),
                        })
                        window_peak = 0
                        window_count = 0
                    if self.stats_chunks in (1, 25, 100) or self.stats_chunks % 250 == 0:
                        elapsed = time.monotonic() - self.asr_started_at
                        audio_fed_sec = self.chunks_fed_to_asr * 0.04
                        rt = audio_fed_sec / elapsed if elapsed > 0 else 0
                        queue_lag = self.audio_queue.qsize() * 0.04
                        log.info(
                            "qwen perf chunks=%d fed=%d audio=%.1fs elapsed=%.1fs "
                            "realtime=%.2fx queue_lag=%.1fs peak=%d",
                            self.stats_chunks, self.chunks_fed_to_asr,
                            audio_fed_sec, elapsed, rt, queue_lag, peak,
                        )
                    if gated:
                        prev_gated = True
                        continue
                    if proc.stdin is None or proc.stdin.is_closing():
                        continue
                    # Speech onset (silent→speech): flush pre-buffer first to
                    # give qwen 0–480ms of context leading up to the onset,
                    # so word-initial consonants don't get clipped.
                    if prev_gated:
                        # The current chunk is at the END of the prebuffer
                        # (we appended it above). Replay everything in the
                        # buffer in order; that includes this chunk, so
                        # don't write it again afterwards.
                        for buf in list(prebuffer):
                            proc.stdin.write(buf)
                        try:
                            await proc.stdin.drain()
                        except (BrokenPipeError, ConnectionResetError):
                            return
                    else:
                        proc.stdin.write(chunk)
                        try:
                            await proc.stdin.drain()
                        except (BrokenPipeError, ConnectionResetError):
                            return
                    prev_gated = False
            finally:
                if proc.stdin and not proc.stdin.is_closing():
                    try:
                        proc.stdin.close()
                    except Exception:
                        pass

        async def receive():
            assert proc.stdout is not None
            # qwen-asr emits UTF-8 byte stream; CJK chars are 3 bytes, our
            # 256-byte read can split a char mid-byte. Use an incremental
            # decoder so partial bytes are buffered until the next read
            # completes the codepoint.
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            while True:
                buf = await proc.stdout.read(256)
                if not buf:
                    # flush any trailing partial bytes
                    tail = decoder.decode(b"", final=True)
                    if tail:
                        await self._on_transcript_chunk(tail)
                    return
                text = decoder.decode(buf)
                if text:
                    await self._on_transcript_chunk(text)

        feed_task = asyncio.create_task(feed())
        recv_task = asyncio.create_task(receive())
        try:
            await asyncio.wait(
                {feed_task, recv_task}, return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            for t in (feed_task, recv_task):
                if not t.done():
                    t.cancel()
                    try:
                        await t
                    except (asyncio.CancelledError, Exception):
                        pass
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
        log.info("qwen-asr subprocess exited (rc=%s)", proc.returncode)

    # ----- voxtral.c backend (local subprocess) -----

    async def _voxtral_asr_worker(self):
        """Spawn the antirez/voxtral.c binary, pump audio in, stream
        transcripts out. Voxtral's CLI is much simpler than qwen-asr —
        no language / prompt / penalty / past-text knobs, only the
        processing interval -I. The audio gate, prebuffer-on-onset, and
        UTF-8 incremental-decode pieces are identical to the qwen path."""
        bin_path = shutil.which(VOXTRAL_BIN) or VOXTRAL_BIN
        if not Path(bin_path).is_file():
            await self._safe_send_json({
                "type": "error",
                "message": (
                    f"voxtral binary not found (looked for {VOXTRAL_BIN!r}). "
                    "Build from https://github.com/antirez/voxtral.c and set "
                    "VOXTRAL_BIN env var."
                ),
            })
            return
        if not Path(VOXTRAL_MODEL_DIR).exists():
            await self._safe_send_json({
                "type": "error",
                "message": (
                    f"voxtral model dir not found at {VOXTRAL_MODEL_DIR!r}. "
                    "Run ./download_model.sh in the voxtral.c repo and set "
                    "VOXTRAL_MODEL_DIR env var."
                ),
            })
            return

        vox_cmd = [
            bin_path,
            "-d", VOXTRAL_MODEL_DIR,
            "--stdin",
            "-I", f"{VOXTRAL_INTERVAL_SEC:g}",
            "--silent",
        ]
        # Wrap with stdbuf for line-buffered stdout (qwen-asr needed this;
        # voxtral flushes per-token so probably fine without, but it's a
        # cheap belt-and-suspenders).
        stdbuf = shutil.which("stdbuf")
        cmd = [stdbuf, "-oL"] + vox_cmd if stdbuf else vox_cmd
        log.info("starting voxtral: %s", " ".join(cmd))
        await self._safe_send_json({"type": "asr_session", "state": "open"})

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except FileNotFoundError as e:
            await self._safe_send_json({
                "type": "error",
                "message": f"failed to spawn voxtral: {e}",
            })
            return

        if self.asr_started_at == 0.0:
            self.asr_started_at = time.monotonic()

        prebuffer: collections.deque[bytes] = collections.deque(
            maxlen=AUDIO_PREBUFFER_CHUNKS
        )
        prev_gated = True

        # 500ms windowed peak for UI mic badge (mirrors qwen path).
        window_peak = 0
        window_count = 0

        async def feed():
            nonlocal prev_gated, window_peak, window_count
            try:
                while True:
                    chunk = await self.audio_queue.get()
                    if chunk is None:
                        break
                    self.stats_chunks += 1
                    self.stats_bytes += len(chunk)
                    peak = chunk_peak(chunk)
                    gated = peak < AUDIO_GATE_PEAK
                    prebuffer.append(chunk)
                    if not gated:
                        self.chunks_fed_to_asr += 1
                    if peak > window_peak:
                        window_peak = peak
                    window_count += 1
                    if window_count >= 12:
                        await self._safe_send_json({
                            "type": "audio_stats",
                            "peak": window_peak,
                            "chunks_per_sec": int(window_count / 0.5),
                        })
                        window_peak = 0
                        window_count = 0
                    if self.stats_chunks in (1, 25, 100) or self.stats_chunks % 250 == 0:
                        elapsed = time.monotonic() - self.asr_started_at
                        audio_fed_sec = self.chunks_fed_to_asr * 0.04
                        rt = audio_fed_sec / elapsed if elapsed > 0 else 0
                        queue_lag = self.audio_queue.qsize() * 0.04
                        log.info(
                            "voxtral perf chunks=%d fed=%d audio=%.1fs "
                            "elapsed=%.1fs realtime=%.2fx queue_lag=%.1fs peak=%d",
                            self.stats_chunks, self.chunks_fed_to_asr,
                            audio_fed_sec, elapsed, rt, queue_lag, peak,
                        )
                    if gated:
                        prev_gated = True
                        continue
                    if proc.stdin is None or proc.stdin.is_closing():
                        continue
                    if prev_gated:
                        for buf in list(prebuffer):
                            proc.stdin.write(buf)
                        try:
                            await proc.stdin.drain()
                        except (BrokenPipeError, ConnectionResetError):
                            return
                    else:
                        proc.stdin.write(chunk)
                        try:
                            await proc.stdin.drain()
                        except (BrokenPipeError, ConnectionResetError):
                            return
                    prev_gated = False
            finally:
                if proc.stdin and not proc.stdin.is_closing():
                    try:
                        proc.stdin.close()
                    except Exception:
                        pass

        async def receive():
            assert proc.stdout is not None
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            while True:
                buf = await proc.stdout.read(256)
                if not buf:
                    tail = decoder.decode(b"", final=True)
                    if tail:
                        await self._on_transcript_chunk(tail)
                    return
                text = decoder.decode(buf)
                if text:
                    await self._on_transcript_chunk(text)

        feed_task = asyncio.create_task(feed())
        recv_task = asyncio.create_task(receive())
        try:
            await asyncio.wait(
                {feed_task, recv_task}, return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            for t in (feed_task, recv_task):
                if not t.done():
                    t.cancel()
                    try:
                        await t
                    except (asyncio.CancelledError, Exception):
                        pass
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
        log.info("voxtral subprocess exited (rc=%s)", proc.returncode)

    # ----- OpenAI Realtime ASR backend (gpt-realtime-whisper) -----

    async def _openai_realtime_asr_worker(self):
        """Stream PCM to OpenAI's Realtime WebSocket in transcription-only
        mode. Server VAD detects turn boundaries; deltas come back as
        conversation.item.input_audio_transcription.delta events. No local
        audio gate / prebuffer / idle flush — VAD handles silence."""
        if websockets is None:
            await self._safe_send_json({
                "type": "error",
                "message": "websockets package not installed",
            })
            return
        if not OPENAI_API_KEY:
            await self._safe_send_json({
                "type": "error",
                "message": "OPENAI_API_KEY env var not set.",
            })
            return

        # `intent=transcription` is what makes this a transcription session
        # rather than the default conversational realtime session — without
        # it, the server rejects session.update with type=transcription. The
        # specific model goes inside transcription.model in session.update.
        url = "wss://api.openai.com/v1/realtime?intent=transcription"
        headers = [("Authorization", f"Bearer {OPENAI_API_KEY}")]

        # Outer reconnect loop. Any exception inside the WS context (network
        # blip, server hangup, auth refresh failure) drops us back here; we
        # reopen unless the parent worker asked us to stop.
        while not self.stop_requested:
            try:
                # `additional_headers` is the websockets≥13 name; older
                # releases used `extra_headers`. We pin a recent version.
                async with websockets.connect(
                    url, additional_headers=headers, max_size=2**24,
                ) as ws:
                    # Configure transcription-only session. OpenAI Realtime
                    # rejects rates below 24kHz, so we upsample 16k → 24k in
                    # `feed()` below before sending. Server VAD handles turn
                    # boundaries; bump silence_duration_ms if you find the
                    # model is cutting mid-clause.
                    src = (self.source_lang or "").strip().lower()
                    transcription_cfg: dict = {"model": OPENAI_REALTIME_MODEL}
                    # OpenAI expects ISO-639-1. Normal-mode UI sends one of
                    # the SOURCE_LANG_ISO keys; debug-mode free-text falls
                    # through unchanged (Whisper accepts ~70 codes natively).
                    if src and src != "auto":
                        transcription_cfg["language"] = SOURCE_LANG_ISO.get(src, src)
                    # NOTE: gpt-realtime-whisper rejects `turn_detection` —
                    # it streams deltas continuously without turn boundaries.
                    # livecap's own punctuation/length-based sentence cutting
                    # in `_on_transcript_chunk` handles segmentation; idle
                    # flush is the fallback when no punctuation appears.
                    await ws.send(json.dumps({
                        "type": "session.update",
                        "session": {
                            "type": "transcription",
                            "audio": {
                                "input": {
                                    "format": {
                                        "type": "audio/pcm",
                                        "rate": 24000,
                                    },
                                    "transcription": transcription_cfg,
                                }
                            },
                        },
                    }))
                    log.info(
                        "openai-realtime: connected, model=%s lang=%s",
                        OPENAI_REALTIME_MODEL,
                        transcription_cfg.get("language") or "auto",
                    )
                    await self._safe_send_json(
                        {"type": "asr_session", "state": "open"}
                    )
                    if self.asr_started_at == 0.0:
                        self.asr_started_at = time.monotonic()

                    window_peak = 0
                    window_count = 0

                    async def feed():
                        nonlocal window_peak, window_count
                        while True:
                            chunk = await self.audio_queue.get()
                            if chunk is None:
                                # Mark end-of-stream so the server can flush
                                # any pending transcription. Don't close the
                                # WS here — let the recv loop's break exit.
                                try:
                                    await ws.send(json.dumps(
                                        {"type": "input_audio_buffer.commit"}
                                    ))
                                except Exception:
                                    pass
                                return
                            self.stats_chunks += 1
                            self.stats_bytes += len(chunk)
                            self.chunks_fed_to_asr += 1  # no audio gate; server VAD handles it
                            peak = chunk_peak(chunk)
                            if peak > window_peak:
                                window_peak = peak
                            window_count += 1
                            if window_count >= 12:
                                await self._safe_send_json({
                                    "type": "audio_stats",
                                    "peak": window_peak,
                                    "chunks_per_sec": int(window_count / 0.5),
                                })
                                window_peak = 0
                                window_count = 0
                            if (
                                self.stats_chunks in (1, 25, 100)
                                or self.stats_chunks % 250 == 0
                            ):
                                elapsed = time.monotonic() - self.asr_started_at
                                audio_fed_sec = self.chunks_fed_to_asr * 0.04
                                rt = audio_fed_sec / elapsed if elapsed > 0 else 0
                                queue_lag = self.audio_queue.qsize() * 0.04
                                log.info(
                                    "openai-realtime perf chunks=%d audio=%.1fs "
                                    "elapsed=%.1fs realtime=%.2fx queue_lag=%.1fs",
                                    self.stats_chunks, audio_fed_sec,
                                    elapsed, rt, queue_lag,
                                )
                            # Browser worklet emits 24kHz when this backend
                            # is selected — no server-side resample needed.
                            await ws.send(json.dumps({
                                "type": "input_audio_buffer.append",
                                "audio": base64.b64encode(chunk).decode("ascii"),
                            }))

                    async def receive():
                        async for raw in ws:
                            try:
                                evt = json.loads(raw)
                            except json.JSONDecodeError:
                                log.warning("openai-realtime: non-JSON frame: %r", raw[:120])
                                continue
                            t = evt.get("type", "")
                            if t == "conversation.item.input_audio_transcription.delta":
                                delta = evt.get("delta") or ""
                                if delta:
                                    await self._on_transcript_chunk(delta)
                            elif t == "conversation.item.input_audio_transcription.completed":
                                # Server VAD says this turn ended. Flush any
                                # buffered text as a finalized sentence so
                                # paired translation has a clean boundary.
                                await self._finalize_pending_sentence()
                            elif t == "input_audio_buffer.speech_stopped":
                                # Optional signal — we already act on
                                # transcription.completed. Logged for diagnosis.
                                pass
                            elif t == "error":
                                err = evt.get("error", {})
                                log.error(
                                    "openai-realtime error: %s — %s",
                                    err.get("type"), err.get("message"),
                                )
                                await self._safe_send_json({
                                    "type": "error",
                                    "message": (
                                        f"OpenAI Realtime: {err.get('message') or err}"
                                    ),
                                })
                            elif t in ("session.created", "session.updated"):
                                # Confirm setup; nothing to do.
                                pass

                    feed_task = asyncio.create_task(feed())
                    recv_task = asyncio.create_task(receive())
                    done, pending = await asyncio.wait(
                        {feed_task, recv_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for t in pending:
                        t.cancel()
                        try:
                            await t
                        except (asyncio.CancelledError, Exception):
                            pass
                    # Surface any unexpected exception from the completed task.
                    for t in done:
                        exc = t.exception()
                        if exc is not None:
                            raise exc
                    # If feed exited because audio_queue got None (client
                    # disconnect / stop), break out of the outer reconnect loop.
                    if self.stop_requested:
                        return
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("openai-realtime: session crashed; will reopen")
                await asyncio.sleep(0.5)
                continue

    # ----- DashScope Qwen3-ASR-Flash-Realtime backend (cloud Qwen ASR) -----

    async def _dashscope_asr_worker(self):
        """Stream PCM to Alibaba's DashScope cloud at qwen3-asr-flash-realtime.
        Protocol is OpenAI-Realtime-shaped but with flatter session.update
        (input_audio_format/sample_rate at session top level, not nested) and
        slightly different event names (.text for intermediate, vs OpenAI's
        .delta). Server VAD is supported here (unlike gpt-realtime-whisper),
        so livesub's own audio gate / prebuffer stay out of the way.
        16kHz PCM accepted natively — no resampling needed."""
        if websockets is None:
            await self._safe_send_json({
                "type": "error",
                "message": "websockets package not installed",
            })
            return
        if not DASHSCOPE_API_KEY:
            await self._safe_send_json({
                "type": "error",
                "message": "DASHSCOPE_API_KEY env var not set.",
            })
            return

        url = f"{DASHSCOPE_BASE_URL}?model={DASHSCOPE_REALTIME_MODEL}"
        headers = [
            ("Authorization", f"Bearer {DASHSCOPE_API_KEY}"),
            # DashScope copies OpenAI's beta-header convention verbatim.
            ("OpenAI-Beta", "realtime=v1"),
        ]

        while not self.stop_requested:
            try:
                async with websockets.connect(
                    url, additional_headers=headers, max_size=2**24,
                ) as ws:
                    # Build session config. DashScope's shape differs from
                    # OpenAI's: input_audio_format and sample_rate are flat
                    # fields on `session`, not nested under audio.input.
                    src = (self.source_lang or "").strip().lower()
                    session_cfg: dict = {
                        "modalities": ["text"],
                        "input_audio_format": "pcm",
                        "sample_rate": 16000,
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": DASHSCOPE_VAD_THRESHOLD,
                            "silence_duration_ms": 600,
                        },
                    }
                    if src and src != "auto":
                        session_cfg["input_audio_transcription"] = {
                            "language": SOURCE_LANG_ISO.get(src, src),
                        }
                    await ws.send(json.dumps({
                        "type": "session.update",
                        "session": session_cfg,
                    }))
                    log.info(
                        "dashscope: connected, model=%s lang=%s",
                        DASHSCOPE_REALTIME_MODEL,
                        (session_cfg.get("input_audio_transcription") or {}).get("language") or "auto",
                    )
                    await self._safe_send_json(
                        {"type": "asr_session", "state": "open"}
                    )
                    if self.asr_started_at == 0.0:
                        self.asr_started_at = time.monotonic()

                    window_peak = 0
                    window_count = 0
                    # DashScope's `stash` is cumulative AND revises itself
                    # mid-segment (adds/removes trailing punctuation,
                    # rewrites words as more audio arrives). Computing
                    # incremental deltas from the cumulative stream is
                    # fragile because revisions break the "new is a prefix
                    # extension of old" assumption. Instead, we send
                    # replace-style `transcript_replace` events to the
                    # client (frontend overwrites the visible caption each
                    # time) and rely on `.completed` for the canonical
                    # final transcript / sentence boundary.

                    async def feed():
                        nonlocal window_peak, window_count
                        while True:
                            chunk = await self.audio_queue.get()
                            if chunk is None:
                                try:
                                    await ws.send(json.dumps(
                                        {"type": "input_audio_buffer.commit"}
                                    ))
                                except Exception:
                                    pass
                                return
                            self.stats_chunks += 1
                            self.stats_bytes += len(chunk)
                            self.chunks_fed_to_asr += 1
                            peak = chunk_peak(chunk)
                            if peak > window_peak:
                                window_peak = peak
                            window_count += 1
                            if window_count >= 12:
                                await self._safe_send_json({
                                    "type": "audio_stats",
                                    "peak": window_peak,
                                    "chunks_per_sec": int(window_count / 0.5),
                                })
                                window_peak = 0
                                window_count = 0
                            if (
                                self.stats_chunks in (1, 25, 100)
                                or self.stats_chunks % 250 == 0
                            ):
                                elapsed = time.monotonic() - self.asr_started_at
                                audio_fed_sec = self.chunks_fed_to_asr * 0.04
                                rt = audio_fed_sec / elapsed if elapsed > 0 else 0
                                queue_lag = self.audio_queue.qsize() * 0.04
                                log.info(
                                    "dashscope perf chunks=%d audio=%.1fs "
                                    "elapsed=%.1fs realtime=%.2fx queue_lag=%.1fs",
                                    self.stats_chunks, audio_fed_sec,
                                    elapsed, rt, queue_lag,
                                )
                            await ws.send(json.dumps({
                                "type": "input_audio_buffer.append",
                                "audio": base64.b64encode(chunk).decode("ascii"),
                            }))

                    async def receive():
                        async for raw in ws:
                            try:
                                evt = json.loads(raw)
                            except json.JSONDecodeError:
                                log.warning(
                                    "dashscope: non-JSON frame: %r", raw[:120]
                                )
                                continue
                            t = evt.get("type", "")
                            if t == "conversation.item.input_audio_transcription.text":
                                # DashScope splits the live transcript into
                                # two fields:
                                #   `text`  — confirmed/stable cumulative
                                #             transcript (what the model is
                                #             committed to so far).
                                #   `stash` — provisional tail still being
                                #             refined (the last word or two
                                #             that may yet be revised).
                                # The official Alibaba samples display
                                # `text + stash` as the live caption — that
                                # is the model's current best guess. Reading
                                # only `stash` (an earlier bug here) drops
                                # the entire confirmed prefix and shows only
                                # the wobbling tail — symptom was a tiny
                                # source caption that "snaps back" to the
                                # full sentence at `.completed`.
                                # We don't run livesub's own punctuation /
                                # length boundary detection (DashScope's VAD
                                # draws boundaries via `.completed`), but we
                                # DO mirror the live text into
                                # `sentence_buffer` and invoke
                                # `_maybe_trigger_partial` so long VAD turns
                                # get intra-segment partial translations
                                # instead of waiting the full turn.
                                text_confirmed = evt.get("text") or ""
                                stash_pending = evt.get("stash") or ""
                                live_text = text_confirmed + stash_pending
                                if live_text:
                                    if self.current_seg_first_chunk == 0.0:
                                        self.current_seg_first_chunk = time.monotonic()
                                    await self._safe_send_json({
                                        "type": "transcript_replace",
                                        "sid": self.next_sid,
                                        "text": live_text,
                                    })
                                    self.sentence_buffer = live_text
                                    await self._maybe_trigger_partial()
                            elif t == "conversation.item.input_audio_transcription.completed":
                                # DashScope's VAD draws the segment boundary
                                # and gives us a canonical `transcript`.
                                # Dispatch directly; buffer accumulation /
                                # idle flush isn't needed for this backend.
                                final = (evt.get("transcript") or "").strip()
                                self.sentence_buffer = ""
                                self.last_chunk_time = 0.0
                                if final:
                                    # Sync the client's visible src to the
                                    # canonical final before dispatch. The
                                    # last `.text` stash is sometimes shorter
                                    # / less complete than `.completed`'s
                                    # transcript (DashScope rewrites at
                                    # segment close). Without this, the
                                    # translator gets the full sentence but
                                    # the caption shows only the truncated
                                    # stash — visible bug: complete
                                    # translation paired with partial source.
                                    await self._safe_send_json({
                                        "type": "transcript_replace",
                                        "sid": self.next_sid,
                                        "text": final,
                                    })
                                    await self._dispatch_sentence(final)
                            elif t == "session.finished":
                                # Server closed the session — exit loop and
                                # let the outer reconnect logic decide what's
                                # next.
                                return
                            elif t == "error":
                                err = evt.get("error", {})
                                log.error(
                                    "dashscope error: %s — %s",
                                    err.get("type"), err.get("message"),
                                )
                                await self._safe_send_json({
                                    "type": "error",
                                    "message": (
                                        f"DashScope: {err.get('message') or err}"
                                    ),
                                })
                            elif t in (
                                "session.created", "session.updated",
                                "input_audio_buffer.speech_started",
                                "input_audio_buffer.speech_stopped",
                            ):
                                pass  # informational; no action needed

                    feed_task = asyncio.create_task(feed())
                    recv_task = asyncio.create_task(receive())
                    done, pending = await asyncio.wait(
                        {feed_task, recv_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for t in pending:
                        t.cancel()
                        try:
                            await t
                        except (asyncio.CancelledError, Exception):
                            pass
                    for t in done:
                        exc = t.exception()
                        if exc is not None:
                            raise exc
                    if self.stop_requested:
                        return
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("dashscope: session crashed; will reopen")
                await asyncio.sleep(0.5)
                continue

    async def _run_one_asr_session(self):
        config = self._build_asr_config()
        async with self.gemini.aio.live.connect(model=ASR_MODEL, config=config) as session:
            log.info(
                "ASR session opened (handle=%s)",
                "resume" if self.resumption_handle else "new",
            )
            await self._safe_send_json({"type": "asr_session", "state": "open"})

            # Track audio peaks per ~500ms window for the UI level meter.
            window_peak = 0
            window_count = 0

            async def feed():
                nonlocal window_peak, window_count
                if self.asr_started_at == 0.0:
                    self.asr_started_at = time.monotonic()
                while True:
                    chunk = await self.audio_queue.get()
                    if chunk is None:
                        return
                    self.stats_chunks += 1
                    self.stats_bytes += len(chunk)
                    self.chunks_fed_to_asr += 1   # no audio gate on Gemini path
                    peak = chunk_peak(chunk)
                    if peak > window_peak:
                        window_peak = peak
                    window_count += 1
                    # ~12 chunks @ 40ms = ~500ms; emit a stats ping
                    if window_count >= 12:
                        await self._safe_send_json({
                            "type": "audio_stats",
                            "peak": window_peak,
                            "chunks_per_sec": int(window_count / 0.5),
                        })
                        window_peak = 0
                        window_count = 0
                    if self.stats_chunks in (1, 25, 100) or self.stats_chunks % 250 == 0:
                        elapsed = time.monotonic() - self.asr_started_at
                        audio_fed_sec = self.chunks_fed_to_asr * 0.04
                        rt = audio_fed_sec / elapsed if elapsed > 0 else 0
                        queue_lag = self.audio_queue.qsize() * 0.04
                        log.info(
                            "gemini perf chunks=%d audio=%.1fs elapsed=%.1fs "
                            "realtime=%.2fx queue_lag=%.1fs",
                            self.stats_chunks, audio_fed_sec, elapsed, rt, queue_lag,
                        )
                    await session.send_realtime_input(
                        audio=types.Blob(data=chunk, mime_type="audio/pcm;rate=16000")
                    )

            async def receive():
                async for response in session.receive():
                    sru = getattr(response, "session_resumption_update", None)
                    if sru is not None:
                        if getattr(sru, "resumable", False) and getattr(sru, "new_handle", None):
                            self.resumption_handle = sru.new_handle

                    go_away = getattr(response, "go_away", None)
                    if go_away is not None:
                        log.info(
                            "go_away time_left=%s; will resume",
                            getattr(go_away, "time_left", None),
                        )
                        await self._safe_send_json(
                            {"type": "asr_session", "state": "go_away"}
                        )
                        return

                    sc = getattr(response, "server_content", None)
                    if sc is None:
                        continue

                    it = getattr(sc, "input_transcription", None)
                    if it is not None and getattr(it, "text", None):
                        await self._on_transcript_chunk(it.text)

                    if getattr(sc, "turn_complete", False):
                        await self._finalize_pending_sentence()
                        log.info("ASR turn_complete (resetting session)")
                        await self._safe_send_json(
                            {"type": "asr_session", "state": "reset"}
                        )
                        return

            feed_task = asyncio.create_task(feed())
            recv_task = asyncio.create_task(receive())
            done, pending = await asyncio.wait(
                {feed_task, recv_task}, return_when=asyncio.FIRST_COMPLETED
            )
            for t in pending:
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass

    async def _on_transcript_chunk(self, text: str):
        """A new chunk of transcript arrived from ASR. Stream it to the
        client tagged with the current sentence id, and check for sentence
        boundaries to dispatch translations."""
        now = time.monotonic()
        await self._safe_send_json(
            {"type": "transcript", "sid": self.next_sid, "text": text}
        )
        # First chunk after the previous dispatch — set first-chunk time
        # regardless of whether buffer has leftover content from a punct cut.
        if self.current_seg_first_chunk == 0.0:
            self.current_seg_first_chunk = now
        self.sentence_buffer += text
        # qwen-asr emits each token as " text" with a leading space. Without
        # lstrip, the buffer stays " 五月十一日…" indefinitely; the length
        # fallback's `rfind(" ", 0, MAX)` then matches index 0, dispatches
        # nothing, and cuts no progress — buffer grows until idle flush.
        if self.sentence_buffer[:1].isspace():
            self.sentence_buffer = self.sentence_buffer.lstrip()
        self.last_chunk_time = now
        # 1) hard-punctuation cut — preferred boundary. Any non-empty
        #    sentence ending in .!?。！？ gets dispatched immediately,
        #    even short interjections like "嗯。" / "OK." / "そう。" —
        #    those are real conversational content, not noise.
        idx = find_last_punct(self.sentence_buffer, include_soft=False)
        if idx >= 0:
            sentence = self.sentence_buffer[: idx + 1].strip()
            if sentence:
                self.sentence_buffer = self.sentence_buffer[idx + 1:]
                await self._dispatch_sentence(sentence)
                return
        # 2) soft-punctuation cut — once the buffer is long enough, comma/
        #    semicolon makes a much better translation unit than a hard 80-char
        #    word cut. Especially important for CJK where there are no spaces
        #    to fall back to.
        if len(self.sentence_buffer) >= SOFT_CUT_THRESHOLD:
            idx = find_last_punct(self.sentence_buffer, include_soft=True)
            if idx >= 0:
                sentence = self.sentence_buffer[: idx + 1].strip()
                if sentence:
                    self.sentence_buffer = self.sentence_buffer[idx + 1:]
                    await self._dispatch_sentence(sentence)
                    return
        # 3) length fallback — last resort. Hard cap buffer growth so
        #    translation actually fires even when neither hard nor soft punct
        #    appears (continuous chant/rap/announcement).
        if len(self.sentence_buffer) >= SENTENCE_MAX_CHARS:
            cut = self.sentence_buffer.rfind(" ", 0, SENTENCE_MAX_CHARS)
            # cut <= 0 (not just < 0): if the only space in the window is at
            # position 0, slicing [:0] would dispatch nothing and stall.
            if cut <= 0:
                cut = SENTENCE_MAX_CHARS
            sentence = self.sentence_buffer[:cut].strip()
            self.sentence_buffer = self.sentence_buffer[cut:]
            if sentence:
                await self._dispatch_sentence(sentence)
            return
        # 3) live partial translation (Claude only): translate the buffer
        #    so far so the user sees a draft translation while still speaking.
        await self._maybe_trigger_partial()

    def _next_rev(self) -> int:
        self.global_rev += 1
        return self.global_rev

    async def _maybe_trigger_partial(self):
        """Possibly fire an intra-sentence partial translation. Enabled for
        anthropic- and openai-SDK backends (Claude variants, DeepSeek,
        OpenAI direct, OpenRouter/ollama/etc.) where the per-call cost is
        low enough to spam mid-sentence. Skipped for Gemini (uses its own
        streaming flow) and `none` (no translation).

        Pacing model is vsync-like: gate on "previous partial finished",
        not on a fixed wall-clock interval. This avoids phase-locking
        against bursty ASR backends — see PARTIAL_HARD_FLOOR_SEC comment
        for the failure mode the fixed-interval version had."""
        backend_cfg = _get_translate_backend(self.translate_backend) or {}
        if backend_cfg.get("sdk") not in ("anthropic", "openai"):
            return
        text_now = self.sentence_buffer.strip()
        if not text_now:
            return
        now = time.monotonic()
        # Hard floor: bound API call rate even if both ASR and translator
        # are absurdly fast. Anything below ~400ms is also imperceptible to
        # the user anyway (caption flashing faster than they can read).
        if now - self.last_partial_time < PARTIAL_HARD_FLOOR_SEC:
            return
        # vsync gate: if the previous partial hasn't finished translating,
        # drop this trigger. The next ASR event after completion will pick
        # it up. This is the "consumer paces to producer" property — partial
        # frequency = min(ASR rate, 1 / translator RTT), with no fixed
        # period for ASR cadence to beat against.
        # Was: cancel-and-restart. Removed because on continuous fast ASR
        # (gemini token-by-token) a slow translator could be cancelled
        # forever — partial never completed, user saw nothing until final.
        if self.in_flight_partial and not self.in_flight_partial.done():
            return
        # UTF-8 bytes (not code points) — see PARTIAL_MIN_NEW_BYTES comment.
        buf_bytes = len(self.sentence_buffer.encode("utf-8"))
        if buf_bytes - self.last_partial_buf_len < PARTIAL_MIN_NEW_BYTES:
            return
        self.last_partial_time = now
        self.last_partial_buf_len = buf_bytes
        sid = self.next_sid
        rev = self._next_rev()
        snapshot = text_now
        history_snapshot = list(self.history)
        self.in_flight_partial = asyncio.create_task(
            self._run_partial(sid, rev, snapshot, history_snapshot)
        )

    async def _run_partial(self, sid: int, rev: int, text: str, history: list):
        try:
            await self._translate_streaming(
                sid, rev, text, history, is_partial=True, is_final=False,
            )
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("partial translation failed sid=%d rev=%d", sid, rev)

    async def _finalize_pending_sentence(self):
        """ASR turn ended (or idle timeout). Dispatch whatever's in the
        buffer as a sentence. Short utterances ("嗯。" / "OK." / "うん。")
        are real content too, so no length filtering."""
        rest = self.sentence_buffer.strip()
        self.sentence_buffer = ""
        self.last_chunk_time = 0.0
        if rest:
            await self._dispatch_sentence(rest)

    async def idle_flush_loop(self):
        """Background task: if no transcript chunk has arrived for
        SENTENCE_IDLE_FLUSH_SEC and the buffer has content, dispatch it.
        Catches the 'speaker paused but no punctuation' case so translation
        still fires periodically."""
        try:
            while not self.stop_requested:
                await asyncio.sleep(0.5)
                if self.last_chunk_time == 0:
                    continue
                if not self.sentence_buffer.strip():
                    continue
                idle = time.monotonic() - self.last_chunk_time
                if idle >= SENTENCE_IDLE_FLUSH_SEC:
                    log.info(
                        "idle flush: %.1fs since last chunk, buffer=%r",
                        idle, self.sentence_buffer[:60],
                    )
                    await self._finalize_pending_sentence()
        except asyncio.CancelledError:
            pass

    async def _dispatch_sentence(self, sentence: str):
        now = time.monotonic()
        sid = self.next_sid
        self.next_sid += 1
        # Per-segment timing log:
        #   wall: time from prev sentence dispatch (or start) to now — the
        #         "wall clock duration" of this segment, including ASR latency
        #   asr_lag: time from FIRST transcript chunk to LAST one — how long
        #            the ASR streamed text for this sentence
        #   ttf_chunk: time from start of segment to FIRST transcript chunk —
        #              latency from speech start to first ASR output
        wall = now - self.current_seg_start if self.current_seg_start else 0.0
        ttf = (
            self.current_seg_first_chunk - self.current_seg_start
            if self.current_seg_first_chunk and self.current_seg_start else 0.0
        )
        asr_span = (
            now - self.current_seg_first_chunk
            if self.current_seg_first_chunk else 0.0
        )
        rate_str = (
            f"{len(sentence) / wall:.1f}c/s" if wall >= 0.1 else "—"
        )
        log.info(
            "ASR seg sid=%d chars=%d wall=%.2fs ttf=%.2fs asr_span=%.2fs rate=%s | %r",
            sid, len(sentence), wall, ttf, asr_span, rate_str,
            sentence[:60] + ("…" if len(sentence) > 60 else ""),
        )
        self.current_seg_start = now
        self.current_seg_first_chunk = 0.0
        # cancel any in-flight partial; the final translation will supersede it.
        if self.in_flight_partial and not self.in_flight_partial.done():
            self.in_flight_partial.cancel()
        self.in_flight_partial = None
        self.last_partial_time = 0.0
        self.last_partial_buf_len = 0
        rev = self._next_rev()
        await self._safe_send_json({"type": "transcript_done", "sid": sid})
        await self._safe_send_json({"type": "translation_start", "sid": sid})
        await self.sentence_queue.put((sid, sentence, rev))

    async def _safe_send_json(self, obj):
        try:
            await self.client_ws.send_json(obj)
        except Exception:
            pass

    # ----- translator worker -----

    async def translator_worker(self):
        try:
            while True:
                item = await self.sentence_queue.get()
                if item is None:
                    return
                sid, sentence, rev = item
                # _translate_streaming has its own try/finally that always
                # emits translation_done — exceptions are surfaced as a
                # placeholder dst there, no need to re-emit done here.
                try:
                    await self._translate_final(sid, rev, sentence)
                except Exception:
                    log.exception("final translation failed for sid=%d", sid)
        finally:
            log.info("translator worker exiting")

    async def _translate_final(self, sid: int, rev: int, sentence: str):
        # Passthrough: user opted out of translation. Mirror the source as
        # the translation so the rest of the pipeline (translation_revision
        # / translation / translation_done events, promoteToPrev, history
        # bookkeeping) keeps working without a backend-specific branch in
        # the frontend.
        if self.translate_backend == "none":
            await self._safe_send_json({
                "type": "translation_revision", "sid": sid, "rev": rev,
            })
            await self._safe_send_json({
                "type": "translation", "sid": sid, "rev": rev, "text": sentence,
            })
            await self._safe_send_json({
                "type": "translation_done",
                "sid": sid, "rev": rev,
                "final": True, "ok": True,
            })
            self.history.append((sentence, sentence))
            self.history = self.history[-HISTORY_PAIRS:]
            self.paired_prev_sid = sid
            self.paired_prev_src = sentence
            self.paired_prev_dst = sentence
            log.info(
                "TR none-passthrough sid=%d rev=%d chars=%d",
                sid, rev, len(sentence),
            )
            return

        # Paired translation: revise the previous sentence's translation
        # in light of the new sentence. Enabled only for anthropic-SDK
        # backends — Claude Haiku/Sonnet/Opus follow the [D]/[CURR]/[PREV]
        # format natively; DeepSeek-V4-flash also passes a 10/10 format
        # battery once thinking is disabled (see ANTHROPIC_EXTRA_BODY).
        # OpenAI-SDK backends fall back to plain translation — paired prompt
        # works in principle but isn't validated against local LLMs yet.
        # Requires at least one prior turn (no paired on first sentence).
        backend_cfg = _get_translate_backend(self.translate_backend) or {}
        use_paired = (
            backend_cfg.get("sdk") == "anthropic"
            and self.paired_prev_sid is not None
            and bool(self.paired_prev_dst)
        )

        translated = ""
        if use_paired:
            # Pass the older context as history; the most-recent pair (which
            # IS our prev) goes in the user message body instead so the model
            # can revise it. Don't double-count it.
            older_history = list(self.history[:-1])
            result = await self._translate_paired_streaming(
                sid, rev, sentence, older_history,
                self.paired_prev_src, self.paired_prev_dst,
            )
            translated = result["curr"]
            prev_revised = result["prev_revised"]
            if prev_revised:
                log.info(
                    "  prev_revised sid=%d:\n    before: %r\n    after:  %r",
                    self.paired_prev_sid,
                    self.paired_prev_dst,
                    prev_revised,
                )
                # Update history's last entry to the revised translation so
                # downstream context (and the next turn's paired prev) sees
                # the corrected version.
                if (self.history
                        and self.history[-1] == (self.paired_prev_src, self.paired_prev_dst)):
                    self.history[-1] = (self.paired_prev_src, prev_revised)
                await self._safe_send_json({
                    "type": "prev_revised",
                    "sid": self.paired_prev_sid,
                    "text": prev_revised,
                })
        else:
            translated = await self._translate_streaming(
                sid, rev, sentence, list(self.history),
                is_partial=False, is_final=True,
            )
        if translated:
            self.history.append((sentence, translated))
            self.history = self.history[-HISTORY_PAIRS:]
            # Cache the just-finalized pair as the prev for the next turn.
            self.paired_prev_sid = sid
            self.paired_prev_src = sentence
            self.paired_prev_dst = translated

    async def _translate_paired_streaming(
        self,
        sid: int,
        rev: int,
        text: str,
        history: list[tuple[str, str]],
        prev_src: str,
        prev_dst: str,
    ) -> dict:
        """Final-translation path with paired-revision capability.
        Returns {'curr': str, 'prev_revised': str|None}."""
        await self._safe_send_json({
            "type": "translation_revision", "sid": sid, "rev": rev,
        })
        result = {"curr": "", "prev_revised": None}
        failed = False
        t_start = time.monotonic()
        metrics: dict = {"first_chunk_at": None}
        try:
            result = await self._translate_one_claude_paired(
                sid, rev, text, history, prev_src, prev_dst, metrics,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            failed = True
            log.exception("paired translate sid=%d rev=%d failed", sid, rev)
        finally:
            t_done = time.monotonic()
            ttft_ms = (
                int((metrics["first_chunk_at"] - t_start) * 1000)
                if metrics["first_chunk_at"] else None
            )
            total_ms = int((t_done - t_start) * 1000)
            log.info(
                "TR final-paired sid=%d rev=%d in_chars=%d out_chars=%d "
                "decision=%s ttft=%s total=%dms%s",
                sid, rev, len(text), len(result["curr"]),
                "revise" if result["prev_revised"] else "keep",
                f"{ttft_ms}ms" if ttft_ms is not None else "—",
                total_ms,
                " FAILED" if failed else "",
            )
            if failed and not result["curr"]:
                placeholder = "[translation failed — rate limit or network]"
                await self._safe_send_json({
                    "type": "translation", "sid": sid, "rev": rev,
                    "text": placeholder,
                })
                result["curr"] = placeholder
            await self._safe_send_json({
                "type": "translation_done",
                "sid": sid, "rev": rev,
                "final": True,
                "ok": not failed,
            })
        return result

    async def _translate_streaming(
        self,
        sid: int,
        rev: int,
        text: str,
        history: list[tuple[str, str]],
        is_partial: bool,
        is_final: bool,
    ) -> str:
        """Run one translation revision (partial or final). Emits a
        translation_revision marker (so client clears the dst), streams text
        chunks, then translation_done."""
        await self._safe_send_json({
            "type": "translation_revision", "sid": sid, "rev": rev,
        })
        full = ""
        failed = False
        t_start = time.monotonic()
        # mutable dict so inner translator can stash first-chunk time
        metrics: dict = {"first_chunk_at": None}
        backend_cfg = _get_translate_backend(self.translate_backend) or {}
        sdk = backend_cfg.get("sdk", "gemini")
        try:
            if sdk == "anthropic":
                full = await self._translate_one_claude(
                    sid, rev, text, history, is_partial, metrics
                )
            elif sdk == "openai":
                full = await self._translate_one_openai(
                    sid, rev, text, history, is_partial, metrics
                )
            else:  # "gemini" (and any unknown — fail soft to Gemini)
                full = await self._translate_one_gemini(
                    sid, rev, text, history, is_partial, metrics
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            failed = True
            log.exception("translate sid=%d rev=%d failed", sid, rev)
        finally:
            t_done = time.monotonic()
            ttft_ms = (
                int((metrics["first_chunk_at"] - t_start) * 1000)
                if metrics["first_chunk_at"] else None
            )
            total_ms = int((t_done - t_start) * 1000)
            kind = "partial" if is_partial else "final"
            log.info(
                "TR %s sid=%d rev=%d backend=%s in_chars=%d out_chars=%d "
                "ttft=%s total=%dms%s",
                kind, sid, rev, self.translate_backend,
                len(text), len(full),
                f"{ttft_ms}ms" if ttft_ms is not None else "—",
                total_ms,
                " FAILED" if failed else "",
            )
            # Final translation must not leave the client with an empty dst
            # (would render as "src=full sentence, dst=" in history). Emit a
            # placeholder so the row at least visibly shows what failed.
            if is_final and failed and not full:
                placeholder = "[translation failed — rate limit or network]"
                await self._safe_send_json({
                    "type": "translation", "sid": sid, "rev": rev, "text": placeholder,
                })
                full = placeholder
            await self._safe_send_json({
                "type": "translation_done",
                "sid": sid, "rev": rev,
                "final": is_final,
                # ok=False signals the inner stream raised. Client uses this
                # to decide whether to swap a buffered partial into the
                # visible dst — a failed partial must NOT clobber the
                # previously-shown good text with empty/half pendingDst.
                "ok": not failed,
            })
        return full

    async def _translate_one_gemini(
        self,
        sid: int,
        rev: int,
        text: str,
        history: list[tuple[str, str]],
        is_partial: bool,
        metrics: dict,
    ) -> str:
        prompt = build_translation_prompt(text, history, is_partial)
        sys = build_translation_system_instruction(self.target_lang, self.source_lang, self.scene)
        config = types.GenerateContentConfig(
            system_instruction=sys,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            temperature=0.2,
        )

        attempts = 0
        max_attempts = 2
        while True:
            attempts += 1
            full = []
            try:
                async for chunk in await self.gemini.aio.models.generate_content_stream(
                    model=TRANSLATE_MODEL,
                    contents=prompt,
                    config=config,
                ):
                    chunk_text = getattr(chunk, "text", None)
                    if chunk_text:
                        if metrics["first_chunk_at"] is None:
                            metrics["first_chunk_at"] = time.monotonic()
                        full.append(chunk_text)
                        await self._safe_send_json({
                            "type": "translation", "sid": sid, "rev": rev, "text": chunk_text,
                        })
                break
            except Exception as e:
                msg = str(e)
                is_429 = "429" in msg or "RESOURCE_EXHAUSTED" in msg
                if is_429 and attempts < max_attempts:
                    delay = _extract_retry_delay(msg, default=5.0)
                    log.warning(
                        "translate sid=%d hit rate limit, retrying in %.1fs",
                        sid, delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise
        return "".join(full).strip()

    async def _translate_one_claude(
        self,
        sid: int,
        rev: int,
        text: str,
        history: list[tuple[str, str]],
        is_partial: bool,
        metrics: dict,
    ) -> str:
        if anthropic is None:
            await self._safe_send_json({
                "type": "error",
                "message": "anthropic SDK not installed (uv add anthropic).",
            })
            return ""
        backend_cfg = _get_translate_backend(self.translate_backend)
        if backend_cfg is None or backend_cfg.get("sdk") != "anthropic":
            return ""  # validator should prevent this
        model = backend_cfg["model"]
        if self.claude is None:
            api_key_env = backend_cfg.get("api_key_env")
            api_key = os.environ.get(api_key_env) if api_key_env else None
            if api_key_env and not api_key:
                await self._safe_send_json({
                    "type": "error",
                    "message": f"{api_key_env} env var not set.",
                })
                return ""
            client_kwargs: dict = {"api_key": api_key or "dummy"}
            if backend_cfg.get("base_url"):
                client_kwargs["base_url"] = backend_cfg["base_url"]
            self.claude = anthropic.AsyncAnthropic(**client_kwargs)

        sys = build_translation_system_instruction(self.target_lang, self.source_lang, self.scene)
        # Claude messages: history goes as user/assistant turns, current
        # text is the latest user turn. This matches the model's training
        # and gives it cleaner context than embedding history in one prompt.
        messages: list[dict] = []
        for src, dst in history:
            messages.append({"role": "user", "content": f"Translate: {src}"})
            messages.append({"role": "assistant", "content": dst})
        if is_partial:
            user_msg = (
                "PARTIAL utterance — speaker is still mid-sentence, more text "
                f"will arrive. Translate what is given so far:\n{text}"
            )
        else:
            user_msg = f"Translate: {text}"
        messages.append({"role": "user", "content": user_msg})

        attempts = 0
        max_attempts = 2
        while True:
            attempts += 1
            full = []
            try:
                async with self.claude.messages.stream(
                    model=model,
                    max_tokens=512,
                    system=sys,
                    messages=messages,
                    extra_body=ANTHROPIC_EXTRA_BODY,
                    cache_control={"type": "ephemeral"},
                ) as stream:
                    async for chunk_text in stream.text_stream:
                        if chunk_text:
                            if metrics["first_chunk_at"] is None:
                                metrics["first_chunk_at"] = time.monotonic()
                            full.append(chunk_text)
                            await self._safe_send_json({
                                "type": "translation", "sid": sid, "rev": rev, "text": chunk_text,
                            })
                break
            except Exception as e:
                msg = str(e)
                # Retry on transient upstream errors. 429/overloaded/rate_limit
                # are explicit Anthropic backpressure; 502/503/504 are gateway
                # blips that resolve in 1-2 s. Without this the inner stream
                # raises, the partial completes "ok" with empty pendingDst,
                # and the client's atomic-swap clears the previously good
                # visible dst — captions go blank mid-sentence.
                ml = msg.lower()
                is_retryable = (
                    "429" in msg or "rate_limit" in ml or "overloaded" in ml
                    or "502" in msg or "503" in msg or "504" in msg
                    or "bad gateway" in ml or "service unavailable" in ml
                    or "gateway timeout" in ml
                )
                if is_retryable and attempts < max_attempts:
                    log.warning(
                        "claude sid=%d transient error, retrying in 3s: %s",
                        sid, msg[:120],
                    )
                    await asyncio.sleep(3.0)
                    continue
                raise
        return "".join(full).strip()

    async def _translate_one_openai(
        self,
        sid: int,
        rev: int,
        text: str,
        history: list[tuple[str, str]],
        is_partial: bool,
        metrics: dict,
    ) -> str:
        """Translate via OpenAI-compatible chat completions (openai SDK).
        Used for OpenAI itself, local OpenAI-compat servers (ollama,
        lm-studio, vLLM), and routing services (OpenRouter, Together, etc.).
        Same prompt shape as the Claude path, just on a different SDK.
        Paired-translation isn't wired here yet — falls back to plain mode."""
        if openai is None:
            await self._safe_send_json({
                "type": "error",
                "message": "openai SDK not installed (uv add openai).",
            })
            return ""
        backend_cfg = _get_translate_backend(self.translate_backend)
        if backend_cfg is None or backend_cfg.get("sdk") != "openai":
            return ""
        model = backend_cfg["model"]
        if self.openai_client is None:
            api_key_env = backend_cfg.get("api_key_env")
            api_key = os.environ.get(api_key_env) if api_key_env else None
            if api_key_env and not api_key:
                await self._safe_send_json({
                    "type": "error",
                    "message": f"{api_key_env} env var not set.",
                })
                return ""
            client_kwargs: dict = {"api_key": api_key or "dummy"}
            if backend_cfg.get("base_url"):
                client_kwargs["base_url"] = backend_cfg["base_url"]
            self.openai_client = openai.AsyncOpenAI(**client_kwargs)

        sys = build_translation_system_instruction(self.target_lang, self.source_lang, self.scene)
        # OpenAI chat.completions has a `system` role message rather than
        # Anthropic's top-level `system` param. Otherwise message shape is
        # the same: alternating user/assistant turns from history, current
        # text as the last user turn.
        messages: list[dict] = [{"role": "system", "content": sys}]
        for src, dst in history:
            messages.append({"role": "user", "content": f"Translate: {src}"})
            messages.append({"role": "assistant", "content": dst})
        if is_partial:
            user_msg = (
                "PARTIAL utterance — speaker is still mid-sentence, more text "
                f"will arrive. Translate what is given so far:\n{text}"
            )
        else:
            user_msg = f"Translate: {text}"
        messages.append({"role": "user", "content": user_msg})

        attempts = 0
        max_attempts = 2
        while True:
            attempts += 1
            full: list[str] = []
            try:
                stream = await self.openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=512,
                    stream=True,
                )
                async for event in stream:
                    if not event.choices:
                        continue
                    delta = event.choices[0].delta.content or ""
                    if delta:
                        if metrics["first_chunk_at"] is None:
                            metrics["first_chunk_at"] = time.monotonic()
                        full.append(delta)
                        await self._safe_send_json({
                            "type": "translation", "sid": sid, "rev": rev, "text": delta,
                        })
                break
            except Exception as e:
                msg = str(e)
                ml = msg.lower()
                # Same retry policy as the Claude path — local OpenAI-compat
                # servers (ollama, vLLM) occasionally return 502/503 during
                # model warmup, and OpenAI-direct returns 429 under burst.
                is_retryable = (
                    "429" in msg or "rate_limit" in ml or "overloaded" in ml
                    or "502" in msg or "503" in msg or "504" in msg
                    or "bad gateway" in ml or "service unavailable" in ml
                    or "gateway timeout" in ml
                )
                if is_retryable and attempts < max_attempts:
                    log.warning(
                        "openai sid=%d transient error, retrying in 3s: %s",
                        sid, msg[:120],
                    )
                    await asyncio.sleep(3.0)
                    continue
                raise
        return "".join(full).strip()

    async def _translate_one_claude_paired(
        self,
        sid: int,
        rev: int,
        text: str,
        history: list[tuple[str, str]],
        prev_src: str,
        prev_dst: str,
        metrics: dict,
    ) -> dict:
        """Paired translation: also reconsider the previous sentence's
        translation in light of the new sentence. Streams CURR text inline
        as translation events; returns {'curr': str, 'prev_revised': str|None}
        once finalized. prev_revised is None when Claude opted to KEEP."""
        if anthropic is None or self.claude is None:
            return {"curr": "", "prev_revised": None}
        backend_cfg = _get_translate_backend(self.translate_backend)
        if backend_cfg is None or backend_cfg.get("sdk") != "anthropic":
            return {"curr": "", "prev_revised": None}
        model = backend_cfg["model"]

        sys = build_paired_translation_system_instruction(
            self.target_lang, self.source_lang, self.scene
        )
        messages: list[dict] = []
        for src, dst in history:
            messages.append({"role": "user", "content": f"Translate: {src}"})
            messages.append({"role": "assistant", "content": dst})
        user_msg = (
            f"Previous source: {prev_src}\n"
            f"Previous translation: {prev_dst}\n"
            f"New source: {text}"
        )
        messages.append({"role": "user", "content": user_msg})

        attempts = 0
        max_attempts = 2
        while True:
            attempts += 1
            parser = PairedStreamParser()
            try:
                async with self.claude.messages.stream(
                    model=model,
                    max_tokens=512,
                    system=sys,
                    messages=messages,
                    extra_body=ANTHROPIC_EXTRA_BODY,
                    cache_control={"type": "ephemeral"},
                ) as stream:
                    async for chunk_text in stream.text_stream:
                        if not chunk_text:
                            continue
                        emit = parser.feed(chunk_text)
                        if emit:
                            if metrics["first_chunk_at"] is None:
                                metrics["first_chunk_at"] = time.monotonic()
                            await self._safe_send_json({
                                "type": "translation", "sid": sid, "rev": rev,
                                "text": emit,
                            })
                break
            except Exception as e:
                msg = str(e)
                ml = msg.lower()
                is_retryable = (
                    "429" in msg or "rate_limit" in ml or "overloaded" in ml
                    or "502" in msg or "503" in msg or "504" in msg
                    or "bad gateway" in ml or "service unavailable" in ml
                    or "gateway timeout" in ml
                )
                if is_retryable and attempts < max_attempts:
                    log.warning(
                        "claude paired sid=%d transient error, retrying in 3s: %s",
                        sid, msg[:120],
                    )
                    await asyncio.sleep(3.0)
                    continue
                raise

        parsed = parser.finalize()
        curr = parsed["curr"].strip()
        prev_out = (parsed["prev"] or "").strip()
        decision = parsed["decision"] or ""

        # Fallback when paired returned no usable CURR. This happens on
        # garbled / fragment input (e.g. song lyric pieces) where the model
        # follows the format but emits an empty [CURR][/CURR] block. Retry
        # with the simpler non-paired prompt so the user always gets some
        # translation. We forfeit the prev-revision opportunity for this
        # turn but that's a fair trade vs. blank output.
        if not curr:
            log.warning(
                "paired sid=%d returned empty CURR — falling back to "
                "non-paired. raw response head: %r",
                sid, parser.full[:300],
            )
            curr = await self._translate_one_claude(
                sid, rev, text, history, is_partial=False, metrics=metrics,
            )
            # Fallback path can't reason about prev — leave it untouched.
            return {"curr": curr.strip(), "prev_revised": None}

        # Treat as a revision only when the model explicitly says revise AND
        # the text actually differs. KEEP + accidentally-different text is
        # treated as no-op (avoid spurious flicker).
        prev_revised: str | None = None
        if decision == "revise" and prev_out and prev_out != prev_dst:
            prev_revised = prev_out
        elif decision != "revise" and prev_out and prev_out != prev_dst:
            # Decision says keep but text drifted — the contract says verbatim.
            # Log and ignore the drift; UI keeps the original prev.
            log.info(
                "paired sid=%d decision=keep but PREV drifted; ignoring "
                "(orig=%r got=%r)",
                sid, prev_dst[:40], prev_out[:40],
            )
        return {"curr": curr, "prev_revised": prev_revised}


# ---------- websocket endpoint ----------


@app.websocket("/ws")
async def ws_endpoint(client_ws: WebSocket):
    await client_ws.accept()
    log.info("client connected")

    try:
        config_msg = await client_ws.receive_json()
    except Exception:
        await client_ws.close(code=1003, reason="expected config message first")
        return

    if not isinstance(config_msg, dict) or config_msg.get("type") != "config":
        await client_ws.close(code=1003, reason="first message must be {type:'config'}")
        return

    target_lang = (config_msg.get("target_lang") or "").strip() or DEFAULT_TARGET_LANG
    source_lang = (config_msg.get("source_lang") or "auto").strip()
    scene = (config_msg.get("scene") or "").strip()
    glossary = (config_msg.get("glossary") or "").strip()
    # Validate backend choices against the live availability list. If the
    # client picked something that isn't enabled in this deployment (missing
    # API key, missing local binary, stale localStorage value), fall back
    # to the first available — or fail closed when nothing's configured.
    asr_backend = (config_msg.get("asr_backend") or "").strip().lower()
    asr_available = {b["id"] for b in _list_asr_backends()}
    if asr_backend not in asr_available:
        asr_backend = next(iter(asr_available), "")
    if not asr_backend:
        await client_ws.close(
            code=1003,
            reason="no ASR backend available — set at least one API key or local binary"
        )
        return
    translate_backend = (config_msg.get("translate_backend") or "").strip().lower()
    tr_available = {b["id"] for b in _list_translate_backends()}
    if translate_backend not in tr_available:
        # `none` is always present, so this fallback chain always lands somewhere.
        translate_backend = next(iter(tr_available), "none")
    log.info(
        "config: asr=%s tr=%s src=%r dst=%r scene=%dch gloss=%dch",
        asr_backend, translate_backend, source_lang, target_lang,
        len(scene), len(glossary),
    )

    # Only construct the Gemini client when the chosen backends actually
    # need it. This lets a user run livesub with just ANTHROPIC_API_KEY
    # (Claude translate) + a local ASR, no Gemini key required.
    gemini = None
    if (asr_backend == "gemini" or translate_backend == "gemini") and GEMINI_API_KEY:
        gemini = genai.Client(
            api_key=GEMINI_API_KEY, http_options={"api_version": "v1beta"}
        )
    pipeline = Pipeline(
        client_ws=client_ws,
        gemini=gemini,
        target_lang=target_lang,
        source_lang=source_lang,
        scene=scene,
        glossary=glossary,
        asr_backend=asr_backend,
        translate_backend=translate_backend,
    )

    pump_task = asyncio.create_task(pipeline.client_pump())
    asr_task = asyncio.create_task(pipeline.asr_worker())
    tr_task = asyncio.create_task(pipeline.translator_worker())
    idle_task = asyncio.create_task(pipeline.idle_flush_loop())

    try:
        await client_ws.send_json({"type": "ready"})
        await asyncio.gather(pump_task, asr_task, tr_task)
    except WebSocketDisconnect:
        log.info("client disconnected")
    except Exception:
        log.exception("pipeline error")
        try:
            await client_ws.send_json({"type": "error", "message": "pipeline error"})
        except Exception:
            pass
    finally:
        for t in (pump_task, asr_task, tr_task, idle_task):
            if not t.done():
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
        log.info("session closed")


if __name__ == "__main__":
    import sys
    import uvicorn

    # Fail fast at startup if no ASR backend is configured. Without this the
    # server starts fine and only rejects clients at WS-connect time with a
    # cryptic close code — much less obvious than a startup error.
    _asr_avail = _list_asr_backends()
    if not _asr_avail:
        log.error(
            "no ASR backend configured — set at least one of "
            "GEMINI_API_KEY / OPENAI_API_KEY / DASHSCOPE_API_KEY, or point "
            "QWEN_ASR_BIN+QWEN_ASR_MODEL_DIR_SMALL (or _LARGE) / "
            "VOXTRAL_BIN+VOXTRAL_MODEL_DIR at a local binary"
        )
        sys.exit(1)
    log.info(
        "ASR backends available: %s",
        ", ".join(b["id"] for b in _asr_avail),
    )

    uvicorn.run(app, host="0.0.0.0", port=8000)
