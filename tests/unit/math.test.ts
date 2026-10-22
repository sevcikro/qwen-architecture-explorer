import { describe, expect, it } from 'vitest';
import { QWEN3_30B_A3B, QWEN3_8B, TOY_DENSE, TOY_MOE } from '../../src/models/profiles';
import { kvBytesPerToken, kvCacheBytes, GiB, KiB } from '../../src/math/memory';
import { applyRope, rmsnorm, ropeInvFreq, routeFromLogits, silu, stableSoftmax, swiglu } from '../../src/math/ops';
import { dot, linear } from '../../src/math/linalg';
import { makeRng, nextNormal } from '../../src/math/rng';
import { bf16Parts, fromBf16Bits, isExactBf16, toBf16Bits } from '../../src/math/bf16';

describe('reference operators against hand-computed fixtures', () => {
  it('the brief example: x=[1,2,3], W=[[0.2,-0.5,0.1],[0.4,0.3,-0.2]] gives y=[-0.5,0.4]', () => {
    const y = linear([1, 2, 3], [[0.2, -0.5, 0.1], [0.4, 0.3, -0.2]]);
    expect(y[0]).toBeCloseTo(-0.5, 12);
    expect(y[1]).toBeCloseTo(0.4, 12);
  });

  it('RMSNorm: no mean subtraction, elementwise gamma', () => {
    // x=[1,2,3,4]: mean(x^2)=7.5, rsqrt(7.5+0)=0.365148...
    const out = rmsnorm([1, 2, 3, 4], [1, 1, 2, 0.5], 0);
    const r = 1 / Math.sqrt(7.5);
    expect(out).toEqual([1 * r, 2 * r, 6 * r, 2 * r].map((v) => expect.closeTo(v, 12)));
    // a constant vector is NOT mapped to zero (LayerNorm would)
    expect(rmsnorm([2, 2], [1, 1], 0)).toEqual([expect.closeTo(1, 12), expect.closeTo(1, 12)]);
    // epsilon is inside the square root
    expect(rmsnorm([0, 0], [1, 1], 1e-6)).toEqual([0, 0]);
  });

  it('SiLU and SwiGLU match a hand fixture', () => {
    expect(silu(0)).toBe(0);
    expect(silu(1)).toBeCloseTo(1 / (1 + Math.exp(-1)), 14);
    expect(silu(-20)).toBeCloseTo(-20 * Math.exp(-20) / (1 + Math.exp(-20)), 20);
    const x = [1, -1];
    const Wg = [[1, 0], [0.5, 0.5]], Wu = [[2, 1], [0, -3]], Wd = [[1, 2], [-1, 0]];
    const r = swiglu(x, Wg, Wu, Wd);
    const g = [silu(1), silu(0)], u = [1, 3];
    expect(r.z).toEqual([g[0] * u[0], g[1] * u[1]]);
    expect(r.y[0]).toBeCloseTo(1 * r.z[0] + 2 * r.z[1], 14);
    expect(r.y[1]).toBeCloseTo(-r.z[0], 14);
  });

  it('stable softmax: normalized, masked entries exactly zero, extreme logits safe, all-masked policy', () => {
    const s = stableSoftmax([1000, 1001, 999], [false, false, true]);
    expect(s.probs[2]).toBe(0);
    expect(s.probs[0] + s.probs[1]).toBeCloseTo(1, 14);
    expect(s.probs.every(Number.isFinite)).toBe(true);
    const all = stableSoftmax([1, 2], [true, true]);
    expect(all.policy).toBe('all-masked');
    expect(all.probs).toEqual([0, 0]);
    expect(() => stableSoftmax([NaN])).toThrow();
  });

  it('RoPE uses half-split pairing and preserves norms; q.k depends only on relative offset', () => {
    const inv = ropeInvFreq(8, 10000);
    expect(inv[0]).toBe(1);
    expect(inv[3]).toBeCloseTo(10000 ** (-6 / 8), 14);
    const v = [1, 2, 3, 4, 5, 6, 7, 8];
    const r = applyRope(v, 3, inv);
    // pair (0,4) rotated by 3 rad
    expect(r[0]).toBeCloseTo(1 * Math.cos(3) - 5 * Math.sin(3), 12);
    expect(r[4]).toBeCloseTo(5 * Math.cos(3) + 1 * Math.sin(3), 12);
    expect(dot(r, r)).toBeCloseTo(dot(v, v), 10);
    const q = [0.3, -1, 2, 0.5, 1, 1, -0.2, 0.7], k = [1, 0.1, -0.4, 2, 0.3, -1, 0.5, 0.2];
    const a = dot(applyRope(q, 7, inv), applyRope(k, 4, inv));
    const b = dot(applyRope(q, 13, inv), applyRope(k, 10, inv));
    expect(a).toBeCloseTo(b, 10);
  });

  it('official-size router fixture: 128 experts, top-8, unique, coefficients sum to 1', () => {
    const rng = makeRng(7);
    for (let trial = 0; trial < 50; trial++) {
      const logits = Array.from({ length: QWEN3_30B_A3B.moe!.numExperts }, () => nextNormal(rng) * 2);
      const r = routeFromLogits(logits, QWEN3_30B_A3B.moe!.topK, true);
      expect(r.probs.length).toBe(128);
      expect(r.probs.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
      expect(new Set(r.selected).size).toBe(8);
      expect(r.alpha.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
      const minSel = Math.min(...r.selected.map((e) => r.probs[e]));
      expect(r.probs.filter((p, i) => !r.selected.includes(i) && p > minSel).length).toBe(0);
    }
  });

  it('toy top-2 routing is separate from the official top-8 fixture', () => {
    const r = routeFromLogits([0.1, 2, -1, 1.5], TOY_MOE.moe!.topK, true);
    expect(r.selected).toEqual([1, 3]);
    expect(r.alpha[0] + r.alpha[1]).toBeCloseTo(1, 14);
    expect(r.alpha[0]).toBeCloseTo(r.probs[1] / (r.probs[1] + r.probs[3]), 14);
  });
});

describe('KV cache formulas', () => {
  it('reproduces 144 KiB/token (8B) and 96 KiB/token (30B-A3B) at BF16, batch 1', () => {
    expect(kvBytesPerToken(QWEN3_8B)).toBe(147_456);
    expect(kvBytesPerToken(QWEN3_8B) / KiB).toBe(144);
    expect(kvBytesPerToken(QWEN3_30B_A3B)).toBe(98_304);
    expect(kvBytesPerToken(QWEN3_30B_A3B) / KiB).toBe(96);
  });
  it('reproduces 4.5 GiB and 3 GiB at 32,768 tokens and uses K/V heads, not query heads', () => {
    expect(kvCacheBytes(QWEN3_8B, { batch: 1, tokens: 32768, bytesPerScalar: 2 }) / GiB).toBe(4.5);
    expect(kvCacheBytes(QWEN3_30B_A3B, { batch: 1, tokens: 32768, bytesPerScalar: 2 }) / GiB).toBe(3);
    expect(kvBytesPerToken(QWEN3_8B)).toBe(2 * 36 * 8 * 128 * 2);
    expect(TOY_DENSE.hkv).toBe(1);
  });
});

describe('BF16 representation', () => {
  it('encodes and decodes exactly for representable values', () => {
    expect(fromBf16Bits(toBf16Bits(1))).toBe(1);
    expect(toBf16Bits(1)).toBe(0x3f80);
    const p = bf16Parts(0xc0a0); // -5
    expect(p.value).toBe(-5);
    expect(p.sign).toBe(1);
    expect(p.unbiasedExponent).toBe(2);
    expect(isExactBf16(0.000568389892578125)).toBe(true);
    expect(isExactBf16(0.1)).toBe(false);
  });
});
