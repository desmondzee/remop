/**
 * Browser fixes before loading Kokoro stack (hexgrad/kokoro.js style) in Next.js:
 * 1. phonemizer decompresses embedded data with Blob#stream().pipeThrough(DecompressionStream)
 *    and `for await` — needs ReadableStream[Symbol.asyncIterator] (not always present in WebKit).
 * 2. DecompressionStream / CompressionStream polyfill for older engines + Apple WebKit quirks.
 * 3. Next polyfills `process`; phonemizer treats string process.versions.node as Node — mask during import.
 */

export function maskProcessNodeVersionForBrowserImport(): () => void {
  if (typeof window === "undefined" || typeof process === "undefined") {
    return () => {};
  }
  const v = process.versions;
  if (!v || typeof v !== "object") {
    return () => {};
  }
  const vo = v as Record<string, unknown> & { node?: unknown };
  const hadOwn = Object.prototype.hasOwnProperty.call(vo, "node");
  const prev = hadOwn ? vo.node : undefined;
  const prevDesc = hadOwn ? Object.getOwnPropertyDescriptor(vo, "node") : undefined;

  try {
    Object.defineProperty(vo, "node", {
      value: undefined,
      configurable: true,
      enumerable: true,
      writable: true,
    });
  } catch {
    try {
      Reflect.deleteProperty(vo, "node");
    } catch {
      /* ignore */
    }
  }

  return () => {
    try {
      if (prevDesc) {
        Object.defineProperty(vo, "node", prevDesc);
      } else if (hadOwn) {
        vo.node = prev;
      } else {
        Reflect.deleteProperty(vo, "node");
      }
    } catch {
      /* ignore */
    }
  };
}

/**
 * phonemizer: `for await (const chunk of readable)` requires async iteration on ReadableStream.
 */
export function ensureReadableStreamAsyncIterator(): void {
  if (typeof ReadableStream === "undefined") return;
  const proto = ReadableStream.prototype as ReadableStream & {
    [Symbol.asyncIterator]?: () => AsyncIterableIterator<Uint8Array>;
  };
  if (typeof proto[Symbol.asyncIterator] === "function") return;

  proto[Symbol.asyncIterator] = async function* (this: ReadableStream) {
    const reader = this.getReader();
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        if (value) yield value;
      }
    } finally {
      reader.releaseLock();
    }
  };
}

export async function ensureCompressionStreamsPolyfill(): Promise<void> {
  if (typeof window === "undefined") return;

  await import("compression-streams-polyfill");

  const ua = typeof navigator !== "undefined" ? navigator.userAgent : "";
  const isAppleWebKit =
    /AppleWebKit\//.test(ua) &&
    !/(?:Chrome|Chromium|CriOS|EdgiOS|EdgA)\//.test(ua);

  if (!isAppleWebKit) return;

  const TS = globalThis.TransformStream;
  if (typeof TS !== "function") return;

  const { makeDecompressionStream, makeCompressionStream } = await import(
    "compression-streams-polyfill/ponyfill"
  );
  const win = globalThis as typeof globalThis & {
    DecompressionStream: new (format: string) => TransformStream;
    CompressionStream: new (format: string) => TransformStream;
  };
  win.DecompressionStream = makeDecompressionStream(TS) as typeof win.DecompressionStream;
  win.CompressionStream = makeCompressionStream(TS) as typeof win.CompressionStream;
}

/** Call before loading @huggingface/transformers / Kokoro and before first phonemize(). */
export async function ensureKokoroStreamPolyfills(): Promise<void> {
  if (typeof window === "undefined") return;
  ensureReadableStreamAsyncIterator();
  await ensureCompressionStreamsPolyfill();
}
