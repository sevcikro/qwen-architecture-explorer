import { dot, type Mat, type Vec } from './linalg';

// ---------------------------------------------------------------------------
// Pure operators matching the pinned Qwen3 implementation conventions.
// ---------------------------------------------------------------------------

/** RMSNorm(x) = gamma * x / sqrt(mean(x^2) + eps). No mean subtraction. */
export function rmsnorm(x: Vec, gamma: Vec, eps: number): Vec {
  const ms = x.reduce((s, v) => s + v * v, 0) / x.length;
  const inv = 1 / Math.sqrt(ms + eps);
  return x.map((v, i) => gamma[i] * v * inv);
}

export const sigmoid = (a: number) => (a >= 0 ? 1 / (1 + Math.exp(-a)) : Math.exp(a) / (1 + Math.exp(a)));
/** SiLU(a) = a * sigmoid(a). */
export const silu = (a: number) => a * sigmoid(a);

export interface SoftmaxResult {
  probs: Vec;
  max: number;
  validCount: number;
  policy: 'normal' | 'all-masked';
}

/**
 * Numerically stable softmax with an explicit masking policy.
 * masked[i] = true excludes entry i (probability exactly 0).
 * Policy for an all-masked row: return all zeros and flag it, never NaN and never a silently normalized empty row.
 */
export function stableSoftmax(logits: Vec, masked?: boolean[]): SoftmaxResult {
  let max = -Infinity;
  let validCount = 0;
  for (let i = 0; i < logits.length; i++) {
    if (masked?.[i]) continue;
    if (!Number.isFinite(logits[i])) throw new Error('non-finite logit');
    validCount++;
    if (logits[i] > max) max = logits[i];
  }
  if (validCount === 0) return { probs: logits.map(() => 0), max: -Infinity, validCount: 0, policy: 'all-masked' };
  const e = logits.map((v, i) => (masked?.[i] ? 0 : Math.exp(v - max)));
  const z = e.reduce((s, v) => s + v, 0);
  return { probs: e.map((v) => v / z), max, validCount, policy: 'normal' };
}

/** Qwen3 RoPE inverse frequencies: theta^(-2i/dh) for i in [0, dh/2). */
export function ropeInvFreq(headDim: number, theta: number): Vec {
  const out: Vec = [];
  for (let i = 0; i < headDim / 2; i++) out.push(1 / Math.pow(theta, (2 * i) / headDim));
  return out;
}

/**
 * Apply RoPE with the half-split pairing used by transformers' rotate_half:
 * coordinate i is paired with i + dh/2 and the pair is rotated by angle pos * invFreq[i].
 */
export function applyRope(v: Vec, pos: number, invFreq: Vec): Vec {
  const h = v.length / 2;
  const out = v.slice();
  for (let i = 0; i < h; i++) {
    const a = pos * invFreq[i];
    const c = Math.cos(a), s = Math.sin(a);
    out[i] = v[i] * c - v[i + h] * s;
    out[i + h] = v[i + h] * c + v[i] * s;
  }
  return out;
}

/** SwiGLU FFN, returning every intermediate so the UI can show them. */
export function swiglu(x: Vec, Wg: Mat, Wu: Mat, Wd: Mat) {
  const gPre = Wg.map((r) => dot(x, r));
  const g = gPre.map(silu);
  const u = Wu.map((r) => dot(x, r));
  const z = g.map((v, j) => v * u[j]);
  const y = Wd.map((r) => dot(z, r));
  return { gPre, g, u, z, y };
}

export interface RouteResult {
  logits: Vec;
  probs: Vec;
  selected: number[]; // expert ids, descending probability
  selectedProbs: Vec;
  alpha: Vec; // mixture coefficients for selected experts
}

/** Qwen3-MoE router: logits = x W^T, softmax over all experts, top-k, optional renormalization. Ties broken by lower id. */
export function route(x: Vec, Wr: Mat, k: number, normTopk: boolean): RouteResult {
  const logits = Wr.map((r) => dot(x, r));
  return routeFromLogits(logits, k, normTopk);
}

export function routeFromLogits(logits: Vec, k: number, normTopk: boolean): RouteResult {
  if (k < 1 || k > logits.length) throw new Error('invalid top-k');
  const { probs } = stableSoftmax(logits);
  const order = probs.map((p, i) => [p, i] as const).sort((a, b) => b[0] - a[0] || a[1] - b[1]);
  const selected = order.slice(0, k).map((o) => o[1]);
  const selectedProbs = selected.map((e) => probs[e]);
  const s = selectedProbs.reduce((a, b) => a + b, 0);
  const alpha = normTopk ? selectedProbs.map((p) => p / s) : selectedProbs.slice();
  return { logits, probs, selected, selectedProbs, alpha };
}
