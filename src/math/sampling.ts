import { stableSoftmax } from './ops';
import { makeRng, nextFloat, type Rng } from './rng';

// Sampling is a decoding procedure applied to logits. It is NOT a trained neural-network layer.
export interface SamplingSettings {
  temperature: number; // 0 => explicit greedy mode
  topK: number; // 0 => disabled
  topP: number; // 1 => disabled
}

export interface SamplingStages {
  greedy: boolean;
  logits: number[];
  scaled: number[] | null;
  probsAll: number[];
  afterTopK: boolean[]; // kept flags
  afterTopP: boolean[];
  final: number[]; // renormalized distribution over kept tokens
  order: number[]; // token ids by descending probability
}

export function samplingStages(logits: number[], s: SamplingSettings): SamplingStages {
  if (!logits.length) throw new Error('empty logits');
  if (s.temperature < 0 || !Number.isFinite(s.temperature)) throw new Error('temperature must be >= 0');
  const greedy = s.temperature === 0;
  const order = logits.map((l, i) => [l, i] as const).sort((a, b) => b[0] - a[0] || a[1] - b[1]).map((x) => x[1]);
  if (greedy) {
    const final = logits.map((_, i) => (i === order[0] ? 1 : 0));
    const keep = logits.map((_, i) => i === order[0]);
    return { greedy, logits, scaled: null, probsAll: stableSoftmax(logits).probs, afterTopK: keep, afterTopP: keep, final, order };
  }
  const scaled = logits.map((l) => l / s.temperature);
  const probsAll = stableSoftmax(scaled).probs;
  // 1) top-k on the temperature-scaled distribution
  const k = s.topK > 0 ? Math.min(s.topK, logits.length) : logits.length;
  const inTopK = new Set(order.slice(0, k));
  const afterTopK = logits.map((_, i) => inTopK.has(i));
  // 2) top-p (nucleus) over the renormalized top-k distribution: smallest prefix with cumulative mass >= p
  const zk = order.filter((i) => afterTopK[i]).reduce((a, i) => a + probsAll[i], 0);
  const afterTopP = logits.map(() => false);
  let cum = 0;
  for (const i of order) {
    if (!afterTopK[i]) continue;
    afterTopP[i] = true;
    cum += probsAll[i] / zk;
    if (cum >= s.topP - 1e-12) break;
  }
  const z = order.filter((i) => afterTopP[i]).reduce((a, i) => a + probsAll[i], 0);
  const final = logits.map((_, i) => (afterTopP[i] ? probsAll[i] / z : 0));
  return { greedy, logits, scaled, probsAll, afterTopK, afterTopP, final, order };
}

/** Draw one token from a final distribution with an explicit RNG; returns the advanced RNG (input is not mutated). */
export function drawToken(final: number[], rng: Rng): { token: number; u: number; rng: Rng } {
  const r = { ...rng };
  const u = nextFloat(r);
  let cum = 0;
  let last = -1;
  for (let i = 0; i < final.length; i++) {
    if (final[i] <= 0) continue;
    last = i;
    cum += final[i];
    if (u < cum) return { token: i, u, rng: r };
  }
  if (last < 0) throw new Error('empty distribution');
  return { token: last, u, rng: r };
}

export const samplerRng = (seed: number) => makeRng(seed ^ 0x5a5a5a5a);
