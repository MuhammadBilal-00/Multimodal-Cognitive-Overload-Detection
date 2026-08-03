const FRAME = 13;
const WINDOW = 30;

export class RingBuffer {
  private buf = new Float32Array(WINDOW * FRAME);
  count = 0;

  push(f: Float32Array): void {
    if (f.length !== FRAME) throw new Error(`expected ${FRAME} features, got ${f.length}`);
    this.buf.copyWithin(0, FRAME);
    this.buf.set(f, (WINDOW - 1) * FRAME);
    this.count++;
  }

  isFull(): boolean {
    return this.count >= WINDOW;
  }

  window(): Float32Array {
    return this.buf;
  }
}
