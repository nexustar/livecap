// Small runtime helpers that stand in for Python stdlib pieces server.py leans
// on: asyncio.Queue, time.monotonic(), asyncio.sleep(), and the logging module.
// Kept deliberately tiny — no framework, mirroring the "one server file" ethos.

// Monotonic clock in SECONDS (float), matching Python's time.monotonic().
export function monotonic(): number {
  return performance.now() / 1000;
}

// asyncio.sleep(seconds)
export function sleep(sec: number): Promise<void> {
  return new Promise((r) => setTimeout(r, sec * 1000));
}

// Thrown by AsyncQueue.get() when its AbortSignal fires. Callers that pass a
// signal are expected to catch this and unwind (mirrors task.cancel()).
export class AbortError extends Error {
  constructor() {
    super("aborted");
    this.name = "AbortError";
  }
}

// Unbounded FIFO with async get(), the analogue of asyncio.Queue. The sentinel
// value used by the pipeline is `null` (never `undefined`), so an empty queue
// is distinguishable from a queued null. get() accepts an AbortSignal so a
// blocked consumer can be unwound at teardown (mirrors task.cancel()).
export class AsyncQueue<T> {
  private items: T[] = [];
  private waiters: Array<(v: T) => void> = [];

  put(item: T): void {
    const deliver = this.waiters.shift();
    if (deliver) deliver(item);
    else this.items.push(item);
  }

  get(signal?: AbortSignal): Promise<T> {
    if (this.items.length > 0) return Promise.resolve(this.items.shift() as T);
    return new Promise<T>((resolve, reject) => {
      if (signal?.aborted) {
        reject(new AbortError());
        return;
      }
      let onAbort: (() => void) | undefined;
      const deliver = (v: T) => {
        if (signal && onAbort) signal.removeEventListener("abort", onAbort);
        resolve(v);
      };
      if (signal) {
        onAbort = () => {
          const i = this.waiters.indexOf(deliver);
          if (i >= 0) this.waiters.splice(i, 1);
          reject(new AbortError());
        };
        signal.addEventListener("abort", onAbort, { once: true });
      }
      this.waiters.push(deliver);
    });
  }

  qsize(): number {
    return this.items.length;
  }
}

// Minimal logger matching Python's "%(asctime)s %(levelname)s %(message)s"
// output. Call sites build the message string with template literals rather
// than %-style args.
function stamp(): string {
  return new Date().toISOString();
}

export const log = {
  info(msg: string): void {
    console.log(`${stamp()} INFO ${msg}`);
  },
  warning(msg: string): void {
    console.log(`${stamp()} WARNING ${msg}`);
  },
  error(msg: string): void {
    console.error(`${stamp()} ERROR ${msg}`);
  },
  // Mirrors log.exception(): message plus the traceback of the active error.
  exception(msg: string, err?: unknown): void {
    const trace = err instanceof Error ? `\n${err.stack ?? err.message}` : err !== undefined ? `\n${String(err)}` : "";
    console.error(`${stamp()} ERROR ${msg}${trace}`);
  },
};

// Absolute peak sample value in a 16-bit s16le PCM chunk (0..32767).
export function chunkPeak(chunk: Buffer): number {
  if (chunk.length < 2) return 0;
  let peak = 0;
  for (let i = 0; i + 1 < chunk.length; i += 2) {
    const v = Math.abs(chunk.readInt16LE(i));
    if (v > peak) peak = v;
  }
  return peak;
}
