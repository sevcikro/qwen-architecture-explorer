import { describe, expect, it } from 'vitest';
import { TOY_DENSE, TOY_MOE, QWEN3_8B } from '../../src/models/profiles';
import {
  CACHE_TOLERANCE, DEFAULT_PROMPT, DEFAULT_SEED, compareLogits, decodeStep, emptyCache, forwardFull, initToyWeights, modelStateKey,
  syncCache, toyTensor, toyTensorNames, withEditedWeight,
} from '../../src/math/toy';
import { maxAbsDiff } from '../../src/math/linalg';
import { swiglu } from '../../src/math/ops';
import { drawToken, samplerRng, samplingStages } from '../../src/math/sampling';

describe.each([TOY_DENSE, TOY_MOE])('toy simulator %s', (p) => {
  const w = initToyWeights(p, DEFAULT_SEED);

  it('is deterministic for a seed and changes with the seed', () => {
    expect(initToyWeights(p, DEFAULT_SEED)).toEqual(w);
    expect(initToyWeights(p, DEFAULT_SEED + 1)).not.toEqual(w);
  });

  it('every named tensor exists with the Qwen [out, in] orientation', () => {
    for (const n of toyTensorNames(p)) expect(() => toyTensor(w, n)).not.toThrow();
    expect((toyTensor(w, 'model.layers.0.self_attn.q_proj.weight') as number[][]).length).toBe(p.hq * p.headDim);
    expect((toyTensor(w, 'model.layers.0.self_attn.q_proj.weight') as number[][])[0].length).toBe(p.d);
  });

  it('causal attention rows are normalized and future positions have exactly zero probability', () => {
    const tr = forwardFull(p, w, DEFAULT_PROMPT);
    for (const pos of tr) for (const L of pos.layers) for (const row of L.attn.probs) {
      expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
      row.forEach((pr, s) => (s > pos.pos ? expect(pr).toBe(0) : expect(pr).toBeGreaterThan(0)));
    }
  });

  it('changing a future token does not alter earlier-position logits', () => {
    const a = forwardFull(p, w, DEFAULT_PROMPT);
    const edited = DEFAULT_PROMPT.slice();
    edited[4] = 9;
    const b = forwardFull(p, w, edited);
    for (let t = 0; t < 4; t++) expect(maxAbsDiff(a[t].logits, b[t].logits)).toBe(0);
    expect(maxAbsDiff(a[4].logits, b[4].logits)).toBeGreaterThan(0);
  });

  it('cached incremental decoding equals full-prefix recomputation (RoPE offsets, GQA cache shapes)', () => {
    const full = forwardFull(p, w, DEFAULT_PROMPT);
    let cache = emptyCache(p, 'k');
    DEFAULT_PROMPT.forEach((tok, t) => {
      const r = decodeStep(p, w, cache, tok);
      cache = r.cache;
      expect(r.trace.pos).toBe(t);
      expect(compareLogits(r.trace.logits, full[t].logits).diff).toBeLessThanOrEqual(CACHE_TOLERANCE);
    });
    for (const L of cache.layers) {
      expect(L.k.length).toBe(DEFAULT_PROMPT.length);
      expect(L.k[0].length).toBe(p.hkv); // cache stores K/V heads, not query heads
      expect(L.k[0][0].length).toBe(p.headDim);
    }
  });

  it('query heads sharing one K/V head still produce separate outputs', () => {
    const L = forwardFull(p, w, DEFAULT_PROMPT)[5].layers[0].attn;
    expect(p.hq / p.hkv).toBe(2);
    expect(maxAbsDiff(L.headOut[0], L.headOut[1])).toBeGreaterThan(1e-6);
    expect(maxAbsDiff(L.probs[0], L.probs[1])).toBeGreaterThan(1e-9);
  });

  it('weight edits and prefix edits invalidate incompatible caches; decoding afterwards still matches recomputation', () => {
    const key0 = modelStateKey(p, DEFAULT_SEED, 0);
    let s = syncCache(p, w, emptyCache(p, key0), key0, DEFAULT_PROMPT);
    expect(s.reason).toBe('extended');
    // inspection-equivalent: same state + same tokens -> nothing recomputed
    const again = syncCache(p, w, s.cache, key0, DEFAULT_PROMPT);
    expect(again.reason).toBe('none');
    expect(again.traces.length).toBe(0);
    // prefix edit at position 2 truncates from there
    const toks = DEFAULT_PROMPT.slice();
    toks[2] = 7;
    s = syncCache(p, w, s.cache, key0, toks);
    expect(s.reason).toBe('prefix-edited');
    expect(s.reusedPositions).toBe(2);
    expect(s.invalidatedPositions).toBe(DEFAULT_PROMPT.length - 2);
    const full = forwardFull(p, w, toks);
    expect(compareLogits(s.traces[s.traces.length - 1].logits, full[full.length - 1].logits).ok).toBe(true);
    // weight edit -> whole cache dropped
    const w2 = withEditedWeight(w, 'model.layers.1.self_attn.v_proj.weight', 0, 0, 0.5);
    const key1 = modelStateKey(p, DEFAULT_SEED, 1);
    const s2 = syncCache(p, w2, s.cache, key1, toks);
    expect(s2.reason).toBe('model-state-changed');
    expect(s2.reusedPositions).toBe(0);
    const full2 = forwardFull(p, w2, toks);
    expect(compareLogits(s2.traces[s2.traces.length - 1].logits, full2[full2.length - 1].logits).ok).toBe(true);
    expect(maxAbsDiff(full2[5].logits, full[5].logits)).toBeGreaterThan(0);
    // original weights untouched
    expect((toyTensor(w, 'model.layers.1.self_attn.v_proj.weight') as number[][])[0][0]).not.toBe(0.5);
  });

  it('rejects toy weights for official profiles and invalid tokens', () => {
    expect(() => initToyWeights(QWEN3_8B, 1)).toThrow();
    expect(() => forwardFull(p, w, [99])).toThrow();
    expect(() => withEditedWeight(w, 'model.norm.weight', 0, null, NaN)).toThrow();
  });
});

describe('MoE toy routing invariants', () => {
  const p = TOY_MOE;
  const w = initToyWeights(p, DEFAULT_SEED);
  it('changing an UNSELECTED expert leaves the token output unchanged; changing a selected one changes it', () => {
    const tr = forwardFull(p, w, DEFAULT_PROMPT);
    const t = 3, l = 0;
    const sel = tr[t].layers[l].moe!.route.selected;
    const unselected = [0, 1, 2, 3].find((e) => !sel.includes(e))!;
    const wU = withEditedWeight(w, `model.layers.${l}.mlp.experts.${unselected}.down_proj.weight`, 0, 0, 9);
    const trU = forwardFull(p, wU, DEFAULT_PROMPT);
    expect(trU[t].layers[l].moe!.route.selected).toEqual(sel);
    expect(maxAbsDiff(trU[t].layers[l].ffnOut, tr[t].layers[l].ffnOut)).toBe(0);
    const wS = withEditedWeight(w, `model.layers.${l}.mlp.experts.${sel[0]}.down_proj.weight`, 0, 0, 9);
    expect(maxAbsDiff(forwardFull(p, wS, DEFAULT_PROMPT)[t].layers[l].ffnOut, tr[t].layers[l].ffnOut)).toBeGreaterThan(0);
  });
  it('MoE output is the alpha-weighted sum of the selected experts', () => {
    const L = forwardFull(p, w, DEFAULT_PROMPT)[2].layers[1];
    const m = L.moe!;
    const manual = new Array(p.d).fill(0);
    m.route.selected.forEach((e, i) => {
      const ex = w.layers[1].experts![e];
      swiglu(L.ln2, ex.gate, ex.up, ex.down).y.forEach((v, j) => (manual[j] += m.route.alpha[i] * v));
    });
    expect(maxAbsDiff(manual, m.y)).toBeLessThan(1e-14);
    expect(m.route.probs.length).toBe(4);
    expect(m.route.selected.length).toBe(2);
  });
});

describe('sampling', () => {
  const logits = [2, 1, 0.5, -1, 3];
  it('temperature 0 is explicit greedy', () => {
    const s = samplingStages(logits, { temperature: 0, topK: 0, topP: 1 });
    expect(s.greedy).toBe(true);
    expect(s.final).toEqual([0, 0, 0, 0, 1]);
  });
  it('top-k then top-p then renormalization', () => {
    const s = samplingStages(logits, { temperature: 1, topK: 3, topP: 0.7 });
    expect(s.afterTopK.filter(Boolean).length).toBe(3);
    expect(s.final.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    expect(s.final[3]).toBe(0);
    expect(s.afterTopP.filter(Boolean).length).toBeLessThanOrEqual(3);
    expect(s.afterTopP[4]).toBe(true);
  });
  it('seeded draws are reproducible and do not mutate the caller RNG', () => {
    const rng = samplerRng(42);
    const before = rng.state;
    const final = samplingStages(logits, { temperature: 1, topK: 0, topP: 1 }).final;
    const a = drawToken(final, rng), b = drawToken(final, rng);
    expect(rng.state).toBe(before);
    expect(a.token).toBe(b.token);
    expect(a.rng.state).not.toBe(before);
    expect(() => samplingStages(logits, { temperature: -1, topK: 0, topP: 1 })).toThrow();
  });
});
