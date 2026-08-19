// /api/hints — research a scene description and produce a keyword list. Port of
// _generate_hints_claude / _generate_hints_gemini from server.py. The backend
// choice (pickHintsBackend) lives in config.ts so /api/backends can reuse it.

import Anthropic from "@anthropic-ai/sdk";
import { GoogleGenAI } from "@google/genai";
import { ANTHROPIC_API_KEY, GEMINI_API_KEY, HINTS_CLAUDE_MODEL, HINTS_GEMINI_MODEL } from "./config.js";
import { buildHintsSystemPrompt, parseHintsResponse } from "./prompts.js";

export interface HintsResult {
  scene: string;
  glossary: string;
  searchesUsed: number;
  model: string;
  stopReason: string | null;
}

// Run the hints flow via Claude Sonnet + Anthropic's web_search tool.
export async function generateHintsClaude(description: string, sourceLang: string): Promise<HintsResult> {
  const client = new Anthropic({ apiKey: ANTHROPIC_API_KEY });
  const msg = await client.messages.create({
    model: HINTS_CLAUDE_MODEL,
    max_tokens: 1024,
    system: buildHintsSystemPrompt(sourceLang),
    tools: [
      {
        // Anthropic's server-side web search tool. Claude decides when to call
        // it; max_uses caps cost.
        type: "web_search_20250305",
        name: "web_search",
        max_uses: 3,
      },
    ],
    messages: [{ role: "user", content: `Scene description: ${description}` }],
  });
  // Concatenate all text blocks; count web_search server_tool_use blocks.
  const textParts: string[] = [];
  let searchesUsed = 0;
  for (const block of msg.content) {
    if (block.type === "text") textParts.push(block.text);
    else if (block.type === "server_tool_use" && block.name === "web_search") searchesUsed += 1;
  }
  const { scene, glossary } = parseHintsResponse(textParts.join("").trim());
  return { scene, glossary, searchesUsed, model: HINTS_CLAUDE_MODEL, stopReason: msg.stop_reason ?? null };
}

// Run the hints flow via Gemini Flash + google_search grounding.
export async function generateHintsGemini(description: string, sourceLang: string): Promise<HintsResult> {
  const client = new GoogleGenAI({ apiKey: GEMINI_API_KEY });
  const resp = await client.models.generateContent({
    model: HINTS_GEMINI_MODEL,
    contents: `Scene description: ${description}`,
    config: {
      systemInstruction: buildHintsSystemPrompt(sourceLang),
      tools: [{ googleSearch: {} }],
    },
  });
  const raw = (resp.text ?? "").trim();
  // web_search_queries is the list of actual queries Gemini issued — its length
  // is the analogue of Claude's server_tool_use count.
  let searchesUsed = 0;
  let finishReason: string | null = null;
  const cand = resp.candidates?.[0];
  if (cand) {
    finishReason = cand.finishReason ? String(cand.finishReason) : null;
    const queries = cand.groundingMetadata?.webSearchQueries ?? [];
    searchesUsed = queries.length;
  }
  const { scene, glossary } = parseHintsResponse(raw);
  return { scene, glossary, searchesUsed, model: HINTS_GEMINI_MODEL, stopReason: finishReason };
}
