// Prompt builders, sentence-boundary helpers, and the streaming paired-response
// parser — a direct port of the "prompts" and "helpers" sections of server.py.
// The prompt text is byte-for-byte identical so model behaviour is unchanged.

import { SENTENCE_PUNCT, SENTENCE_PUNCT_SOFT } from "./config.js";

// ---------- /api/hints prompt ----------

export function buildHintsSystemPrompt(sourceLang: string): string {
  const src = (sourceLang || "").trim();
  let glossaryLangRule: string;
  let glossaryLangHint: string;
  if (!src || src.toLowerCase() === "auto") {
    glossaryLangRule =
      "Use each entry's NATIVE script (Japanese in kanji/kana, " +
      "Chinese in hanzi, Korean in hangul, etc.). Don't transliterate.";
    glossaryLangHint = "";
  } else {
    glossaryLangRule =
      `ALL entries must be in ${src}. The audio source language is ` +
      `${src}; the glossary is consumed by an ASR system that biases ` +
      `${src} tokens. Entries in any other language (including the ` +
      `target translation language) are useless and may distort ` +
      `recognition. If you only know an entry's English / Chinese ` +
      `transliteration, search the web to find the ${src} original ` +
      `(or omit it).`;
    glossaryLangHint = `, all in ${src}`;
  }
  return (
    "You are an assistant for a real-time ASR + translation system. " +
    "Given brief user input — a topic, URL, keyword, or short " +
    "description — you produce TWO outputs:\n\n" +
    "1. SCENE: a 1–3 sentence summary describing the scenario, genre, " +
    "speakers, and tone. This is given to the translator as background " +
    "context so it can pick appropriate register and word choice. " +
    "Mention any cultural conventions or fan-slang that affect how " +
    'things should be translated (e.g. "use casual Chinese fan-slang; ' +
    'preserve idol terminology like 推し / ペンライト").\n\n' +
    "2. GLOSSARY: a comma-separated list of proper nouns, names, and " +
    "specialty terms that may appear. " +
    `${glossaryLangRule} ` +
    "Used as a soft prompt to bias the speech recognizer at the token " +
    "level — the prompt language MUST match the audio language for the " +
    "bias to work.\n\n" +
    "Rules:\n" +
    "- Search the web when the input refers to a specific real-world " +
    "thing (concert, show, person, event, anime, sports, technical " +
    "topic). Don't fabricate; only list what you verify or know.\n" +
    "- For generic scenarios where research adds nothing, just answer " +
    "from training.\n" +
    "- GLOSSARY: 30–100 entries, most distinctive proper nouns first.\n" +
    "- SCENE: concise. Skip generic stuff. Include things that affect " +
    "translation register.\n\n" +
    "Output format (use the exact delimiters, both blocks required, no " +
    "other text):\n" +
    "[SCENE]\n<1–3 sentence scene summary>\n[/SCENE]\n" +
    `[GLOSSARY]\n<comma-separated terms${glossaryLangHint}>\n[/GLOSSARY]\n`
  );
}

// Extract [SCENE] and [GLOSSARY] blocks from a hints model response. Falls back
// to treating the whole text as glossary when neither delimiter block parses.
export function parseHintsResponse(raw: string): { scene: string; glossary: string } {
  const sceneM = /\[SCENE\]\s*([\s\S]*?)\s*\[\/SCENE\]/.exec(raw);
  const glossM = /\[GLOSSARY\]\s*([\s\S]*?)\s*\[\/GLOSSARY\]/.exec(raw);
  let scene = sceneM ? sceneM[1].trim() : "";
  let glossary = glossM ? glossM[1].trim() : "";
  if (!scene && !glossary) glossary = raw;
  return { scene, glossary };
}

// ---------- ASR / translation prompts ----------

export function buildAsrPrompt(sourceLang: string, scene: string, glossary: string): string {
  const src = (sourceLang || "").trim().toLowerCase();
  let langHint: string;
  if (src === "" || src === "auto") {
    langHint = "The audio may be in any language; detect it automatically.";
  } else {
    langHint =
      `The audio is primarily in ${sourceLang}. Transcribe spoken ` +
      `${sourceLang} accurately, even if there is background music, ` +
      "applause, sound effects, or multiple voices.";
  }
  let base =
    "You are a passive listener. Your only job is to allow accurate input " +
    "transcription of the audio you receive. " +
    `${langHint} ` +
    "Stay completely silent. Do not respond. Do not generate any audio " +
    "output. Just listen.";
  if (scene) base += `\n\nScene: ${scene}`;
  if (glossary) base += `\n\nGlossary (proper nouns / specialty terms): ${glossary}`;
  return base;
}

// qwen-asr --prompt: terminology / spelling bias for the decoder. Glossary
// only — NOT scene (mixing target-language scene text confuses qwen's language
// detection). Capped at 500 chars.
export function buildQwenPrompt(glossary: string): string {
  return glossary ? glossary.slice(0, 500) : "";
}

export function buildTranslationSystemInstruction(targetLang: string, sourceLang: string, scene: string): string {
  const parts: string[] = [
    `You are a translator. Translate the user's utterance into ${targetLang}. ` +
      "Output ONLY the translation. No preamble, no quotes, no commentary. " +
      "Keep technical terms, brand names, and proper nouns in their original " +
      "language when natural.",
  ];
  const src = (sourceLang || "").trim();
  if (src && src.toLowerCase() !== "auto") parts.push(`Source language: ${src}.`);
  if (scene) parts.push(`Scene: ${scene}`);
  return parts.join("\n\n");
}

// Paired-translation system instruction (Claude only). Per-turn the model also
// reconsiders the IMMEDIATELY PREVIOUS sentence's translation given the new
// context. 3-shot prompt validated 10/10 against a 10-case battery on Haiku 4.5.
export function buildPairedTranslationSystemInstruction(targetLang: string, sourceLang: string, scene: string): string {
  let head =
    `Real-time ${targetLang} interpreter. Each turn, also reconsider ` +
    "the previous translation given the new sentence.\n\n" +
    "Default KEEP — [PREV] must be byte-identical to the input. REVISE " +
    "only when the new sentence reveals: a wrong word sense, a " +
    "mid-clause/number cut completed by new, or actual ambiguity. " +
    "Stylistic tweaks are NOT a reason — they cause UI flicker.\n\n" +
    "[CURR] must NEVER be empty. Even if the new sentence is " +
    "fragmentary, garbled, a song-lyric piece, or unclear, output a " +
    "best-effort translation in [CURR]. The UI shows [CURR] as the " +
    "live caption for this turn — empty [CURR] = blank caption = bug.";
  const src = (sourceLang || "").trim();
  if (src && src.toLowerCase() !== "auto") head += `\n\nSource language: ${src}.`;
  if (scene) head += `\n\nScene: ${scene}`;
  return (
    head +
    "\n\nOutput ONLY the three blocks:\n" +
    "[D]keep[/D] or [D]revise[/D]\n" +
    "[CURR]<translation of new sentence>[/CURR]\n" +
    "[PREV]<verbatim or revised>[/PREV]\n\n" +
    "Examples:\n\n" +
    "(KEEP — prev complete, new unrelated)\n" +
    "prev_src: 今日は本当に楽しかったです。\n" +
    "prev_dst: 今天真的很开心。\n" +
    "new_src:  では、次に行きましょう。\n" +
    "→ [D]keep[/D]\n" +
    "  [CURR]那么，我们继续下一个吧。[/CURR]\n" +
    "  [PREV]今天真的很开心。[/PREV]\n\n" +
    "(REVISE — mid-clause cut, new completes thought)\n" +
    "prev_src: 私たちは慎重にこの問題を扱う必要があると\n" +
    "prev_dst: 我们需要慎重处理这个问题，\n" +
    "new_src:  思っていますが、時間がかかっても価値があります。\n" +
    "→ [D]revise[/D]\n" +
    "  [CURR]虽然会花时间，但这是值得的。[/CURR]\n" +
    "  [PREV]我们认为需要慎重处理这个问题，[/PREV]\n\n" +
    "(REVISE — figurative meaning revealed)\n" +
    "prev_src: このチームは本当に熱いですね。\n" +
    "prev_dst: 这支队伍真的很热。\n" +
    "new_src:  試合を諦めずに最後まで戦い抜きました。\n" +
    "→ [D]revise[/D]\n" +
    "  [CURR]他们没有放弃，战斗到了最后一刻。[/CURR]\n" +
    "  [PREV]这支队伍真的很有热情。[/PREV]\n"
  );
}

export function buildTranslationPrompt(sentence: string, history: Array<[string, string]>, isPartial = false): string {
  const parts: string[] = [];
  if (history.length) {
    const ctx = history.map(([src, dst]) => `  ${src} → ${dst}`).join("\n");
    parts.push(
      "Previous translations in this session (for terminology and " +
        `pronoun consistency, do not re-translate them):\n${ctx}\n\n`,
    );
  }
  if (isPartial) {
    parts.push(
      "PARTIAL utterance — speaker is still mid-sentence, more text " +
        "will arrive. Translate what is given so far. Output only the " +
        `translation:\n${sentence}`,
    );
  } else {
    parts.push(`Now translate this new utterance:\n${sentence}`);
  }
  return parts.join("");
}

// ---------- helpers ----------

// Last index of any sentence-ending punctuation in text, or -1. With
// includeSoft, commas/semicolons (EN/JP/ZH) also count.
export function findLastPunct(text: string, includeSoft = false): number {
  const chars = SENTENCE_PUNCT + (includeSoft ? SENTENCE_PUNCT_SOFT : "");
  let last = -1;
  for (const p of chars) {
    const i = text.lastIndexOf(p);
    if (i > last) last = i;
  }
  return last;
}

// Incrementally parses Claude's [D][/D][CURR][/CURR][PREV][/PREV] paired
// response. Streams CURR's inner text as soon as it arrives (chars that COULD
// be the start of `[/CURR]` are held back so we never emit a partial closing
// tag). [D] and [PREV] are extracted from the full buffer in finalize().
export class PairedStreamParser {
  private static readonly CLOSE_CURR = "[/CURR]";
  full = "";
  private cursor = 0;
  private state: "before_curr" | "in_curr" | "done" = "before_curr";

  // Returns whatever CURR text became safe to emit during this feed.
  feed(chunk: string): string {
    this.full += chunk;
    const out: string[] = [];
    const CLOSE = PairedStreamParser.CLOSE_CURR;
    for (;;) {
      if (this.state === "before_curr") {
        const idx = this.full.indexOf("[CURR]", this.cursor);
        if (idx < 0) break;
        this.cursor = idx + "[CURR]".length;
        // skip leading whitespace inside the block
        while (this.cursor < this.full.length && " \t\n".includes(this.full[this.cursor])) {
          this.cursor += 1;
        }
        this.state = "in_curr";
      }
      if (this.state === "in_curr") {
        const end = this.full.indexOf(CLOSE, this.cursor);
        if (end >= 0) {
          const text = this.full.slice(this.cursor, end).replace(/\s+$/u, ""); // rstrip
          if (text) out.push(text);
          this.cursor = end + CLOSE.length;
          this.state = "done";
          continue;
        }
        // No closing tag yet. Hold back chars that could be the start of
        // "[/CURR]" — emit everything else.
        let tail = this.full.length;
        const maxHold = Math.min(CLOSE.length - 1, tail - this.cursor);
        for (let i = maxHold; i > 0; i--) {
          if (this.full.endsWith(CLOSE.slice(0, i))) {
            tail -= i;
            break;
          }
        }
        if (tail > this.cursor) {
          out.push(this.full.slice(this.cursor, tail));
          this.cursor = tail;
        }
        break;
      }
      if (this.state === "done") break;
    }
    return out.join("");
  }

  finalize(): { decision: string | null; prev: string | null; curr: string } {
    const dM = /\[D\]\s*(\w+)\s*\[\/D\]/i.exec(this.full);
    const pM = /\[PREV\]\s*([\s\S]*?)\s*\[\/PREV\]/.exec(this.full);
    const cM = /\[CURR\]\s*([\s\S]*?)\s*\[\/CURR\]/.exec(this.full);
    return {
      decision: dM ? dM[1].trim().toLowerCase() : null,
      prev: pM ? pM[1].trim() : null,
      curr: cM ? cM[1].trim() : "",
    };
  }
}

const RETRY_RE = /retryDelay['"]?\s*:\s*['"]?(\d+(?:\.\d+)?)s?/;

// Pull retryDelay out of a Google API 429 error message and clamp it.
export function extractRetryDelay(msg: string, def = 5.0): number {
  const m = RETRY_RE.exec(msg);
  if (!m) return def;
  const v = parseFloat(m[1]);
  if (Number.isNaN(v)) return def;
  return Math.min(Math.max(v, 1.0), 30.0);
}
