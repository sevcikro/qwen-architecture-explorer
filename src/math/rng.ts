// Deterministic, explicit-state PRNG (mulberry32) so seeds can be shared and replayed.
export interface Rng {
  state: number;
}

export const makeRng = (seed: number): Rng => ({ state: seed >>> 0 });

/** Advances the given state and returns a float in [0, 1). Pure w.r.t. everything except `rng.state`. */
export function nextFloat(rng: Rng): number {
  let t = (rng.state = (rng.state + 0x6d2b79f5) >>> 0);
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}

/** Standard normal via Box-Muller (uses two uniforms). */
export function nextNormal(rng: Rng): number {
  let u = nextFloat(rng);
  if (u < 1e-12) u = 1e-12;
  const v = nextFloat(rng);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** Derive an independent stream from a seed and a label (string hash -> seed). */
export function deriveSeed(seed: number, label: string): number {
  let h = (seed ^ 0x9e3779b9) >>> 0;
  for (let i = 0; i < label.length; i++) {
    h = Math.imul(h ^ label.charCodeAt(i), 0x85ebca6b) >>> 0;
    h ^= h >>> 13;
  }
  return h >>> 0;
}
