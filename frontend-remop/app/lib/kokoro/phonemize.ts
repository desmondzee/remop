/**
 * Phonemization pipeline from hexgrad/kokoro (kokoro.js/src/phonemize.js).
 * eSpeak-NG runs via the `phonemizer` package, loaded dynamically only after
 * stream polyfills (see kokoroBrowserRuntime.ts) so gzip decompression works in WebKit.
 */

import {
  ensureKokoroStreamPolyfills,
  maskProcessNodeVersionForBrowserImport,
} from "./kokoroBrowserRuntime";

type EspeakPhonemize = (text: string, lang: string) => Promise<string[]>;

let espeakPhonemize: EspeakPhonemize | null = null;

async function ensureEspeakLoaded(): Promise<EspeakPhonemize> {
  if (espeakPhonemize) return espeakPhonemize;
  await ensureKokoroStreamPolyfills();
  const unmask = maskProcessNodeVersionForBrowserImport();
  try {
    const mod = await import("phonemizer");
    espeakPhonemize = mod.phonemize;
  } finally {
    unmask();
  }
  if (!espeakPhonemize) {
    throw new Error("phonemizer failed to load");
  }
  return espeakPhonemize;
}

function splitKeepingDelimiters(
  text: string,
  regex: RegExp
): { match: boolean; text: string }[] {
  const result: { match: boolean; text: string }[] = [];
  let prev = 0;
  for (const match of text.matchAll(regex)) {
    const fullMatch = match[0];
    const idx = match.index ?? 0;
    if (prev < idx) {
      result.push({ match: false, text: text.slice(prev, idx) });
    }
    if (fullMatch.length > 0) {
      result.push({ match: true, text: fullMatch });
    }
    prev = idx + fullMatch.length;
  }
  if (prev < text.length) {
    result.push({ match: false, text: text.slice(prev) });
  }
  return result;
}

function splitNum(match: string): string {
  if (match.includes(".")) {
    return match;
  }
  if (match.includes(":")) {
    const [h, m] = match.split(":").map(Number);
    if (m === 0) {
      return `${h} o'clock`;
    }
    if (m < 10) {
      return `${h} oh ${m}`;
    }
    return `${h} ${m}`;
  }
  const year = parseInt(match.slice(0, 4), 10);
  if (year < 1100 || year % 1000 < 10) {
    return match;
  }
  const left = match.slice(0, 2);
  const right = parseInt(match.slice(2, 4), 10);
  const suffix = match.endsWith("s") ? "s" : "";
  if (year % 1000 >= 100 && year % 1000 <= 999) {
    if (right === 0) {
      return `${left} hundred${suffix}`;
    }
    if (right < 10) {
      return `${left} oh ${right}${suffix}`;
    }
  }
  return `${left} ${right}${suffix}`;
}

function flipMoney(match: string): string {
  const bill = match[0] === "$" ? "dollar" : "pound";
  if (isNaN(Number(match.slice(1)))) {
    return `${match.slice(1)} ${bill}s`;
  }
  if (!match.includes(".")) {
    const suffix = match.slice(1) === "1" ? "" : "s";
    return `${match.slice(1)} ${bill}${suffix}`;
  }
  const [b, c] = match.slice(1).split(".");
  const d = parseInt(c.padEnd(2, "0"), 10);
  const coins =
    match[0] === "$" ? (d === 1 ? "cent" : "cents") : d === 1 ? "penny" : "pence";
  return `${b} ${bill}${b === "1" ? "" : "s"} and ${d} ${coins}`;
}

function pointNum(match: string): string {
  const [a, b] = match.split(".");
  return `${a} point ${b.split("").join(" ")}`;
}

function normalizeText(text: string): string {
  return (
    text
      .replace(/[‘’]/g, "'")
      .replace(/«/g, "“")
      .replace(/»/g, "”")
      .replace(/[“”]/g, '"')
      .replace(/\(/g, "«")
      .replace(/\)/g, "»")
      .replace(/、/g, ", ")
      .replace(/。/g, ". ")
      .replace(/！/g, "! ")
      .replace(/，/g, ", ")
      .replace(/：/g, ": ")
      .replace(/；/g, "; ")
      .replace(/？/g, "? ")
      .replace(/[^\S \n]/g, " ")
      .replace(/  +/g, " ")
      .replace(/(?<=\n) +(?=\n)/g, "")
      .replace(/\bD[Rr]\.(?= [A-Z])/g, "Doctor")
      .replace(/\b(?:Mr\.|MR\.(?= [A-Z]))/g, "Mister")
      .replace(/\b(?:Ms\.|MS\.(?= [A-Z]))/g, "Miss")
      .replace(/\b(?:Mrs\.|MRS\.(?= [A-Z]))/g, "Mrs")
      .replace(/\betc\.(?! [A-Z])/gi, "etc")
      .replace(/\b(y)eah?\b/gi, "$1e'a")
      .replace(
        /\d*\.\d+|\b\d{4}s?\b|(?<!:)\b(?:[1-9]|1[0-2]):[0-5]\d\b(?!:)/g,
        splitNum
      )
      .replace(/(?<=\d),(?=\d)/g, "")
      .replace(
        /[$£]\d+(?:\.\d+)?(?: hundred| thousand| (?:[bm]|tr)illion)*\b|[$£]\d+\.\d\d?\b/gi,
        flipMoney
      )
      .replace(/\d*\.\d+/g, pointNum)
      .replace(/(?<=\d)-(?=\d)/g, " to ")
      .replace(/(?<=\d)S/g, " S")
      .replace(/(?<=[BCDFGHJ-NP-TV-Z])'?s\b/g, "'S")
      .replace(/(?<=X')S\b/g, "s")
      .replace(/(?:[A-Za-z]\.){2,} [a-z]/g, (m) => m.replace(/\./g, "-"))
      .replace(/(?<=[A-Z])\.(?=[A-Z])/gi, "-")
      .trim()
  );
}

function escapeRegExp(string: string): string {
  return string.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

const PUNCTUATION = ';:,.!?¡¿—…"«»“”(){}[]';
const PUNCTUATION_PATTERN = new RegExp(
  `(\\s*[${escapeRegExp(PUNCTUATION)}]+\\s*)+`,
  "g"
);

export async function phonemize(
  text: string,
  language: "a" | "b" = "a",
  norm = true
): Promise<string> {
  if (norm) {
    text = normalizeText(text);
  }

  const sections = splitKeepingDelimiters(text, PUNCTUATION_PATTERN);
  const lang = language === "a" ? "en-us" : "en";
  const espeak = await ensureEspeakLoaded();

  const ps = (
    await Promise.all(
      sections.map(async ({ match, text: chunk }) =>
        match ? chunk : (await espeak(chunk, lang)).join(" ")
      )
    )
  ).join("");

  let processed = ps
    .replace(/kəkˈoːɹoʊ/g, "kˈoʊkəɹoʊ")
    .replace(/kəkˈɔːɹəʊ/g, "kˈəʊkəɹəʊ")
    .replace(/ʲ/g, "j")
    .replace(/r/g, "ɹ")
    .replace(/x/g, "k")
    .replace(/ɬ/g, "l")
    .replace(/(?<=[a-zɹː])(?=hˈʌndɹɪd)/g, " ")
    .replace(/ z(?=[;:,.!?¡¿—…"«»“” ]|$)/g, "z");

  if (language === "a") {
    processed = processed.replace(/(?<=nˈaɪn)ti(?!ː)/g, "di");
  }
  return processed.trim();
}
