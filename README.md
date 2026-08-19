# livesub

**Local-first** real-time bilingual live captions. Audio capture and
speech-to-text run fully offline with the recommended Qwen3-ASR 0.6B —
which uniquely handles transcription **over background music** — while
translation is your choice: local LLM, cloud API, or skip it entirely.

Built for Japanese livestreams (concerts, anime events, podcasts) with
Chinese / English subtitles, but works for any language pair the
underlying pipeline supports.

```
[browser tab audio | mic]
        │  16 kHz s16le, 40 ms chunks
        ▼
    [ Speech-to-text ] ← Qwen3-ASR (local, BGM-capable) | Qwen Cloud | OpenAI Realtime | Voxtral (local) | Gemini Live
        ▼
    [ Translator ]     ← Local LLM (Ollama, lm-studio) | DeepSeek | Claude Haiku/Sonnet/Opus | Gemini | none
        ▼
   live captions + history (browser UI)
```

## Setup

**Prereqs:** Node 20+, npm, and a modern browser. Tested on macOS and
Linux; Windows is untested (Path C likely works as-is; Path A/B need WSL
for the local C builds).

Three paths depending on what you want. Pick one and follow it top to
bottom, then start the server.

### 🔌 Path A: Local ASR + cloud translation (recommended for most users)

Audio stays on your machine; only translated text leaves the network.
Combines the best local speech-to-text (BGM-capable Qwen 0.6B) with the
cheapest practical translation (DeepSeek).

Install qwen-asr:

```bash
git clone -b livesub https://github.com/nexustar/qwen-asr
cd qwen-asr && make blas && ./download_model.sh --model small
```

`.env`:

```bash
QWEN_ASR_BIN=/abs/path/to/qwen-asr/qwen_asr
QWEN_ASR_MODEL_DIR_SMALL=/abs/path/to/qwen-asr/qwen3-asr-0.6b
DEEPSEEK_API_KEY=sk-...                # or ANTHROPIC_API_KEY for Claude
```

### 🏠 Path B: Fully offline

No cloud, no API keys. Translation requires a local LLM (Ollama or
lm-studio) registered in `livesub.toml`; alternatively, skip translation
by picking "None (transcript only)" in Settings.

Install qwen-asr (same commands as Path A), then:

`.env`:

```bash
QWEN_ASR_BIN=/abs/path/to/qwen-asr/qwen_asr
QWEN_ASR_MODEL_DIR_SMALL=/abs/path/to/qwen-asr/qwen3-asr-0.6b
```

For translation, see [Custom translate backends](#custom-translate-backends)
below — `livesub.toml.example` ships with a ready-to-use Ollama snippet.

### ☁️ Path C: Fully cloud

Quickest to set up; nothing local to install. Pick speech-to-text by
your audio content:

- **Audio with background music** (concerts, anime, gaming):
  **Qwen Cloud** (DashScope) — best quality on this content, ~$0.005/min.
- **Voice-dominant** (podcasts, talks, interviews):
  **OpenAI Realtime** — better accuracy when the foreground is clean
  speech, ~$0.017/min.

`.env`:

```bash
# Speech-to-text — pick one:
DASHSCOPE_API_KEY=sk-...               # Qwen Cloud (recommended for BGM)
OPENAI_API_KEY=sk-...                  # OpenAI Realtime (recommended for voice)

# Translation — DeepSeek is the cost/quality sweet spot:
DEEPSEEK_API_KEY=sk-...
# OR for highest translation quality:
ANTHROPIC_API_KEY=sk-ant-...
```

### Start the server

```bash
git clone <repo> && cd livesub
cp .env.example .env       # then edit per your chosen path above
npm install
npm start                  # http://0.0.0.0:8000
```

### Higher-quality local alternatives

For Path A / B users on a powerful machine: Qwen3-ASR **1.7B** or
**Voxtral 4B** are more accurate than the 0.6B default but much heavier
on RAM/VRAM.

Qwen 1.7B (same binary as the 0.6B, just a different model):

```bash
cd qwen-asr && ./download_model.sh --model large
```

```bash
QWEN_ASR_BIN=/abs/path/to/qwen-asr/qwen_asr
QWEN_ASR_MODEL_DIR_LARGE=/abs/path/to/qwen-asr/qwen3-asr-1.7b
```

Both `_SMALL` and `_LARGE` can be set at the same time — each shows up
as its own option in the Settings → Speech-to-text dropdown.

Voxtral install:

```bash
git clone https://github.com/antirez/voxtral.c
cd voxtral.c && make mps && ./download_model.sh   # mps on Apple Silicon
```

`.env`:

```bash
VOXTRAL_BIN=/abs/path/to/voxtral.c/voxtral
VOXTRAL_MODEL_DIR=/abs/path/to/voxtral.c/voxtral-realtime-4b
```

## Custom translate backends

Copy `livesub.toml.example` to `livesub.toml` to add your own translation
backends — local LLMs via Ollama / lm-studio, routing services like
OpenRouter, or OpenAI directly. The file is optional; without it you get
the built-ins (Claude variants, DeepSeek, Gemini, None).

## Usage

Open `http://localhost:8000`, pick **🎤 Mic** or **📺 Tab**, click
**Start**, choose the source tab in the browser dialog. Settings (gear
icon) let you switch speech-to-text / translation backends, source /
target language, and feed Claude a "scene seed" to auto-generate a
glossary.

UI layout, top to bottom (newest at the bottom — like a chat log):

1. **Controls** — Start / Stop, source picker, level meter, status.
2. **History** — older finalized sentences, smaller, recessed; oldest at
   the top, newest at the bottom. Auto-scrolls to keep newest visible
   unless you've scrolled up to read older content.
3. **Prev caption** — just-finalized previous sentence, same size as
   current but dim. Slides up from the LIVE slot when current finalizes;
   revisable by paired-translation (brief blue flash).
4. **Live captions** — current sentence, blue tint + LIVE badge. Anchored
   at the very bottom.
