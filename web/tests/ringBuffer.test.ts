import { describe, it, expect } from 'vitest';
import { RingBuffer } from '../lib/ringBuffer';

const frame = (v: number) => new Float32Array(13).fill(v);

describe('RingBuffer', () => {
  it('is not full until 30 frames pushed', () => {
    const rb = new RingBuffer();
    for (let i = 0; i < 29; i++) rb.push(frame(i));
    expect(rb.isFull()).toBe(false);
    rb.push(frame(29));
    expect(rb.isFull()).toBe(true);
  });

  it('window is oldest->newest and evicts', () => {
    const rb = new RingBuffer();
    for (let i = 0; i < 31; i++) rb.push(frame(i)); // frame 0 evicted
    const w = rb.window();
    expect(w.length).toBe(390);
    expect(w[0]).toBe(1);        // oldest remaining
    expect(w[389]).toBe(30);     // newest
  });
});
