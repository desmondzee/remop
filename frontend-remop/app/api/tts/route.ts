import { NextRequest, NextResponse } from "next/server";

/** Cheapest OpenAI speech model. */
const OPENAI_TTS_MODEL = "tts-1";

const ALLOWED_VOICES = new Set([
  "alloy",
  "ash",
  "ballad",
  "coral",
  "echo",
  "fable",
  "nova",
  "onyx",
  "sage",
  "shimmer",
  "verse",
]);

const MAX_INPUT_CHARS = 4096;

const API_LOG = "[remop:tts:api]";

export async function POST(req: NextRequest) {
  const key = process.env.OPENAI_API_KEY?.trim();
  if (!key) {
    console.error(API_LOG, "reject: OPENAI_API_KEY not set");
    return NextResponse.json(
      { error: "Server missing OPENAI_API_KEY" },
      { status: 503 }
    );
  }

  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  if (!body || typeof body !== "object") {
    return NextResponse.json({ error: "Expected JSON object" }, { status: 400 });
  }

  const o = body as Record<string, unknown>;
  const text =
    typeof o.text === "string" ? o.text.trim().replace(/\s+/g, " ") : "";
  if (!text) {
    return NextResponse.json({ error: "Missing or empty text" }, { status: 400 });
  }
  if (text.length > MAX_INPUT_CHARS) {
    return NextResponse.json(
      { error: `Text exceeds ${MAX_INPUT_CHARS} characters` },
      { status: 400 }
    );
  }

  const rawVoice =
    typeof o.voice === "string" ? o.voice.trim().toLowerCase() : "alloy";
  const voice = ALLOWED_VOICES.has(rawVoice) ? rawVoice : "alloy";

  console.log(API_LOG, "POST received → OpenAI speech", {
    textLength: text.length,
    voice,
    preview: text.slice(0, 100),
  });

  const t0 = Date.now();
  const upstream = await fetch("https://api.openai.com/v1/audio/speech", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${key}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: OPENAI_TTS_MODEL,
      voice,
      input: text,
      response_format: "mp3",
    }),
  });

  if (!upstream.ok) {
    const raw = await upstream.text();
    console.error(API_LOG, "OpenAI upstream error", {
      status: upstream.status,
      ms: Date.now() - t0,
      body: raw.slice(0, 400),
    });
    let detail = raw.slice(0, 500) || upstream.statusText;
    try {
      const errJson = JSON.parse(raw) as {
        error?: { message?: string } | string;
      };
      if (errJson?.error) {
        if (typeof errJson.error === "string") detail = errJson.error;
        else if (typeof errJson.error.message === "string") {
          detail = errJson.error.message;
        }
      }
    } catch {
      /* use raw slice */
    }
    return NextResponse.json(
      { error: detail || `OpenAI error (${upstream.status})` },
      { status: 502 }
    );
  }

  const buf = await upstream.arrayBuffer();
  console.log(API_LOG, "OpenAI OK → returning MP3", {
    byteLength: buf.byteLength,
    ms: Date.now() - t0,
  });
  return new NextResponse(buf, {
    status: 200,
    headers: {
      "Content-Type": "audio/mpeg",
      "Cache-Control": "private, no-store",
    },
  });
}
