import type { CoreProfile } from '../models/profiles';
import { add, dot, linear, maxAbsDiff, type Mat, type Vec } from './linalg';
import { applyRope, rmsnorm, route, ropeInvFreq, stableSoftmax, swiglu, type RouteResult } from './ops';
import { deriveSeed, makeRng, nextNormal } from './rng';

// ---------------------------------------------------------------------------
// Toy simulator: a structurally faithful, tiny Qwen3-style decoder with seeded synthetic weights.
// Same operations and tensor names as Qwen3; NOT Qwen's dimensions and NOT Qwen's weights.
// ---------------------------------------------------------------------------

export const TOY_VOCAB = ['<bos>', 'the', 'cat', 'sat', 'on', 'mat', 'a', 'dog', 'ran', 'to', 'park', '.'];
export const DEFAULT_SEED = 20260422;
export const DEFAULT_PROMPT = [0, 1, 2, 3, 4, 1];

export interface FfnWeights { gate: Mat; up: Mat; down: Mat }
export interface LayerWeights {
  ln1: Vec; q: Mat; k: Mat; v: Mat; o: Mat; qn: Vec; kn: Vec; ln2: Vec;
  ffn?: FfnWeights; router?: Mat; experts?: FfnWeights[];
}
export interface ToyWeights { embed: Mat; layers: LayerWeights[]; norm: Vec; head: Mat }

const gauss = (seed: number, label: string, rows: number, cols: number, std: number): Mat => {
  const rng = makeRng(deriveSeed(seed, label));
  return Array.from({ length: rows }, () => Array.from({ length: cols }, () => nextNormal(rng) * std));
};
const gain = (seed: number, label: string, n: number): Vec => {
  const rng = makeRng(deriveSeed(seed, label));
  return Array.from({ length: n }, () => 1 + 0.15 * nextNormal(rng));
};

export function initToyWeights(p: CoreProfile, seed: number): ToyWeights {
  if (p.status !== 'toy' && p.status !== 'custom') throw new Error('toy weights are only for toy/custom profiles');
  const { d, hq, hkv, headDim: dh, vocabRows: V } = p;
  const layers: LayerWeights[] = [];
  for (let l = 0; l < p.layers; l++) {
    const pre = `model.layers.${l}`;
    const lw: LayerWeights = {
      ln1: gain(seed, `${pre}.input_layernorm.weight`, d),
      q: gauss(seed, `${pre}.self_attn.q_proj.weight`, hq * dh, d, 1 / Math.sqrt(d)),
      k: gauss(seed, `${pre}.self_attn.k_proj.weight`, hkv * dh, d, 1 / Math.sqrt(d)),
      v: gauss(seed, `${pre}.self_attn.v_proj.weight`, hkv * dh, d, 1 / Math.sqrt(d)),
      o: gauss(seed, `${pre}.self_attn.o_proj.weight`, d, hq * dh, 1 / Math.sqrt(hq * dh)),
      qn: gain(seed, `${pre}.self_attn.q_norm.weight`, dh),
      kn: gain(seed, `${pre}.self_attn.k_norm.weight`, dh),
      ln2: gain(seed, `${pre}.post_attention_layernorm.weight`, d),
    };
    if (p.kind === 'dense') {
      const f = p.ffnWidth!;
      lw.ffn = {
        gate: gauss(seed, `${pre}.mlp.gate_proj.weight`, f, d, 1 / Math.sqrt(d)),
        up: gauss(seed, `${pre}.mlp.up_proj.weight`, f, d, 1 / Math.sqrt(d)),
        down: gauss(seed, `${pre}.mlp.down_proj.weight`, d, f, 1 / Math.sqrt(f)),
      };
    } else {
      const m = p.moe!;
      lw.router = gauss(seed, `${pre}.mlp.gate.weight`, m.numExperts, d, 1 / Math.sqrt(d));
      lw.experts = Array.from({ length: m.numExperts }, (_, e) => ({
        gate: gauss(seed, `${pre}.mlp.experts.${e}.gate_proj.weight`, m.expertWidth, d, 1 / Math.sqrt(d)),
        up: gauss(seed, `${pre}.mlp.experts.${e}.up_proj.weight`, m.expertWidth, d, 1 / Math.sqrt(d)),
        down: gauss(seed, `${pre}.mlp.experts.${e}.down_proj.weight`, d, m.expertWidth, 1 / Math.sqrt(m.expertWidth)),
      }));
    }
    layers.push(lw);
  }
  return {
    embed: gauss(seed, 'model.embed_tokens.weight', V, d, 1),
    layers,
    norm: gain(seed, 'model.norm.weight', d),
    head: gauss(seed, 'lm_head.weight', V, d, 2.2 / Math.sqrt(d)),
  };
}

/** Every toy tensor addressable by its Qwen-style checkpoint name. */
export function toyTensor(w: ToyWeights, name: string): Mat | Vec {
  if (name === 'model.embed_tokens.weight') return w.embed;
  if (name === 'model.norm.weight') return w.norm;
  if (name === 'lm_head.weight') return w.head;
  const m = name.match(/^model\.layers\.(\d+)\.(.+)\.weight$/);
  if (!m) throw new Error(`unknown toy tensor ${name}`);
  const L = w.layers[Number(m[1])];
  if (!L) throw new Error(`no layer in ${name}`);
  const rest = m[2];
  const map: Record<string, Mat | Vec | undefined> = {
    input_layernorm: L.ln1, post_attention_layernorm: L.ln2,
    'self_attn.q_proj': L.q, 'self_attn.k_proj': L.k, 'self_attn.v_proj': L.v, 'self_attn.o_proj': L.o,
    'self_attn.q_norm': L.qn, 'self_attn.k_norm': L.kn,
    'mlp.gate_proj': L.ffn?.gate, 'mlp.up_proj': L.ffn?.up, 'mlp.down_proj': L.ffn?.down, 'mlp.gate': L.router,
  };
  if (rest in map) {
    const t = map[rest];
    if (!t) throw new Error(`tensor ${name} absent in this toy variant`);
    return t;
  }
  const e = rest.match(/^mlp\.experts\.(\d+)\.(gate|up|down)_proj$/);
  if (e && L.experts?.[Number(e[1])]) return L.experts[Number(e[1])][e[2] as 'gate' | 'up' | 'down'];
  throw new Error(`unknown toy tensor ${name}`);
}

export function toyTensorNames(p: CoreProfile): string[] {
  const out = ['model.embed_tokens.weight'];
  for (let l = 0; l < p.layers; l++) {
    const pre = `model.layers.${l}`;
    out.push(`${pre}.input_layernorm.weight`, `${pre}.self_attn.q_proj.weight`, `${pre}.self_attn.k_proj.weight`, `${pre}.self_attn.v_proj.weight`, `${pre}.self_attn.q_norm.weight`, `${pre}.self_attn.k_norm.weight`, `${pre}.self_attn.o_proj.weight`, `${pre}.post_attention_layernorm.weight`);
    if (p.kind === 'dense') out.push(`${pre}.mlp.gate_proj.weight`, `${pre}.mlp.up_proj.weight`, `${pre}.mlp.down_proj.weight`);
    else {
      out.push(`${pre}.mlp.gate.weight`);
      for (let e = 0; e < p.moe!.numExperts; e++) out.push(`${pre}.mlp.experts.${e}.gate_proj.weight`, `${pre}.mlp.experts.${e}.up_proj.weight`, `${pre}.mlp.experts.${e}.down_proj.weight`);
    }
  }
  out.push('model.norm.weight', 'lm_head.weight');
  return out;
}

/** Returns a deep-copied weight set with one scalar changed. */
export function withEditedWeight(w: ToyWeights, name: string, row: number, col: number | null, value: number): ToyWeights {
  if (!Number.isFinite(value)) throw new Error('weight must be a finite number');
  const copy: ToyWeights = structuredClone(w);
  const t = toyTensor(copy, name);
  if (col === null) (t as Vec)[row] = value;
  else (t as Mat)[row][col] = value;
  return copy;
}

// ---------------------------------------------------------------------------
// Traces
// ---------------------------------------------------------------------------
export interface AttnTrace {
  qPre: Vec[]; kPre: Vec[]; v: Vec[]; // per head, before head-norm (v has no norm)
  qNorm: Vec[]; kNorm: Vec[];
  qRot: Vec[]; kRot: Vec[];
  /** per query head: raw dot products with every key position 0..T-1 (future positions flagged masked) */
  rawScores: Vec[]; masked: boolean[];
  scaled: Vec[]; probs: Vec[];
  headOut: Vec[]; concat: Vec; out: Vec;
}
export interface MoeTrace { route: RouteResult; expertOut: Vec[]; y: Vec }
export interface LayerTrace {
  xIn: Vec; ln1: Vec; attn: AttnTrace; h: Vec; ln2: Vec;
  ffn?: ReturnType<typeof swiglu>; moe?: MoeTrace; ffnOut: Vec; out: Vec;
}
export interface PositionTrace { token: number; pos: number; embed: Vec; layers: LayerTrace[]; finalNorm: Vec; logits: Vec }

function splitHeads(v: Vec, n: number, dh: number): Vec[] {
  return Array.from({ length: n }, (_, h) => v.slice(h * dh, (h + 1) * dh));
}

function ffnForward(p: CoreProfile, L: LayerWeights, x: Vec): { ffn?: ReturnType<typeof swiglu>; moe?: MoeTrace; y: Vec } {
  if (p.kind === 'dense') {
    const r = swiglu(x, L.ffn!.gate, L.ffn!.up, L.ffn!.down);
    return { ffn: r, y: r.y };
  }
  const m = p.moe!;
  const rt = route(x, L.router!, m.topK, m.normTopkProb);
  const expertOut = rt.selected.map((e) => swiglu(x, L.experts![e].gate, L.experts![e].up, L.experts![e].down).y);
  const y = new Array(p.d).fill(0);
  expertOut.forEach((o, i) => o.forEach((v, j) => (y[j] += rt.alpha[i] * v)));
  return { moe: { route: rt, expertOut, y }, y };
}

/**
 * FULL-PREFIX RECOMPUTATION (prefill style): layer-major over all positions with an explicit causal mask.
 * Independent code path from `decodeStep` below so their agreement is a real correctness check.
 */
export function forwardFull(p: CoreProfile, w: ToyWeights, tokens: number[]): PositionTrace[] {
  const T = tokens.length;
  const { hq, hkv, headDim: dh, rmsEps: eps } = p;
  const group = hq / hkv;
  const inv = ropeInvFreq(dh, p.ropeTheta);
  let X: Vec[] = tokens.map((t) => {
    if (t < 0 || t >= p.vocabRows) throw new Error(`token id ${t} out of range`);
    return w.embed[t].slice();
  });
  const embeds = X.map((x) => x.slice());
  const perLayer: LayerTrace[][] = Array.from({ length: T }, () => []);
  for (let l = 0; l < p.layers; l++) {
    const L = w.layers[l];
    const ln1 = X.map((x) => rmsnorm(x, L.ln1, eps));
    const qPre = ln1.map((x) => splitHeads(linear(x, L.q), hq, dh));
    const kPre = ln1.map((x) => splitHeads(linear(x, L.k), hkv, dh));
    const vH = ln1.map((x) => splitHeads(linear(x, L.v), hkv, dh));
    const qNorm = qPre.map((hs) => hs.map((q) => rmsnorm(q, L.qn, eps)));
    const kNorm = kPre.map((hs) => hs.map((k) => rmsnorm(k, L.kn, eps)));
    const qRot = qNorm.map((hs, t) => hs.map((q) => applyRope(q, t, inv)));
    const kRot = kNorm.map((hs, t) => hs.map((k) => applyRope(k, t, inv)));
    const Xn: Vec[] = [];
    for (let t = 0; t < T; t++) {
      const masked = Array.from({ length: T }, (_, s) => s > t);
      const rawScores: Vec[] = [], scaled: Vec[] = [], probs: Vec[] = [], headOut: Vec[] = [];
      for (let h = 0; h < hq; h++) {
        const kv = Math.floor(h / group);
        const raw = Array.from({ length: T }, (_, s) => dot(qRot[t][h], kRot[s][kv]));
        const sc = raw.map((r) => r / Math.sqrt(dh));
        const pr = stableSoftmax(sc, masked).probs;
        const o = new Array(dh).fill(0);
        for (let s = 0; s <= t; s++) for (let i = 0; i < dh; i++) o[i] += pr[s] * vH[s][kv][i];
        rawScores.push(raw); scaled.push(sc); probs.push(pr); headOut.push(o);
      }
      const concat = headOut.flat();
      const out = linear(concat, L.o);
      const hRes = add(X[t], out);
      const ln2 = rmsnorm(hRes, L.ln2, eps);
      const f = ffnForward(p, L, ln2);
      const y = add(hRes, f.y);
      perLayer[t].push({
        xIn: X[t], ln1: ln1[t],
        attn: { qPre: qPre[t], kPre: kPre[t], v: vH[t], qNorm: qNorm[t], kNorm: kNorm[t], qRot: qRot[t], kRot: kRot[t], rawScores, masked, scaled, probs, headOut, concat, out },
        h: hRes, ln2, ffn: f.ffn, moe: f.moe, ffnOut: f.y, out: y,
      });
      Xn.push(y);
    }
    X = Xn;
  }
  return X.map((x, t) => {
    const fn = rmsnorm(x, w.norm, eps);
    return { token: tokens[t], pos: t, embed: embeds[t], layers: perLayer[t], finalNorm: fn, logits: linear(fn, w.head) };
  });
}

// ---------------------------------------------------------------------------
// KV cache and incremental decoding
// ---------------------------------------------------------------------------
export interface KvCache {
  /** model-state fingerprint the cache was computed under */
  stateKey: string;
  tokens: number[];
  /** layers[l].k[s][kvHead] = post-norm, post-RoPE key at absolute position s */
  layers: { k: Vec[][]; v: Vec[][] }[];
}

export const emptyCache = (p: CoreProfile, stateKey: string): KvCache => ({
  stateKey,
  tokens: [],
  layers: Array.from({ length: p.layers }, () => ({ k: [], v: [] })),
});

/** Process ONE new token at absolute position = cache length, reading earlier K/V from the cache and appending its own. */
export function decodeStep(p: CoreProfile, w: ToyWeights, cache: KvCache, token: number): { cache: KvCache; trace: PositionTrace } {
  const pos = cache.tokens.length;
  const { hq, hkv, headDim: dh, rmsEps: eps } = p;
  const group = hq / hkv;
  const inv = ropeInvFreq(dh, p.ropeTheta);
  const next: KvCache = { stateKey: cache.stateKey, tokens: [...cache.tokens, token], layers: cache.layers.map((c) => ({ k: c.k.slice(), v: c.v.slice() })) };
  let x = w.embed[token].slice();
  const embed = x.slice();
  const layers: LayerTrace[] = [];
  for (let l = 0; l < p.layers; l++) {
    const L = w.layers[l];
    const ln1 = rmsnorm(x, L.ln1, eps);
    const qPre = splitHeads(linear(ln1, L.q), hq, dh);
    const kPre = splitHeads(linear(ln1, L.k), hkv, dh);
    const v = splitHeads(linear(ln1, L.v), hkv, dh);
    const qNorm = qPre.map((q) => rmsnorm(q, L.qn, eps));
    const kNorm = kPre.map((k) => rmsnorm(k, L.kn, eps));
    const qRot = qNorm.map((q) => applyRope(q, pos, inv));
    const kRot = kNorm.map((k) => applyRope(k, pos, inv));
    next.layers[l].k.push(kRot);
    next.layers[l].v.push(v);
    const Tk = pos + 1;
    const rawScores: Vec[] = [], scaled: Vec[] = [], probs: Vec[] = [], headOut: Vec[] = [];
    for (let h = 0; h < hq; h++) {
      const kv = Math.floor(h / group);
      const raw = next.layers[l].k.map((ks) => dot(qRot[h], ks[kv]));
      const sc = raw.map((r) => r / Math.sqrt(dh));
      const pr = stableSoftmax(sc).probs;
      const o = new Array(dh).fill(0);
      for (let s = 0; s < Tk; s++) for (let i = 0; i < dh; i++) o[i] += pr[s] * next.layers[l].v[s][kv][i];
      rawScores.push(raw); scaled.push(sc); probs.push(pr); headOut.push(o);
    }
    const concat = headOut.flat();
    const out = linear(concat, L.o);
    const hRes = add(x, out);
    const ln2 = rmsnorm(hRes, L.ln2, eps);
    const f = ffnForward(p, L, ln2);
    const y = add(hRes, f.y);
    layers.push({
      xIn: x, ln1,
      attn: { qPre, kPre, v, qNorm, kNorm, qRot, kRot, rawScores, masked: new Array(Tk).fill(false), scaled, probs, headOut, concat, out },
      h: hRes, ln2, ffn: f.ffn, moe: f.moe, ffnOut: f.y, out: y,
    });
    x = y;
  }
  const fn = rmsnorm(x, w.norm, eps);
  return { cache: next, trace: { token, pos, embed, layers, finalNorm: fn, logits: linear(fn, w.head) } };
}

export interface CacheSync {
  cache: KvCache;
  reusedPositions: number;
  invalidatedPositions: number;
  reason: 'none' | 'model-state-changed' | 'prefix-edited' | 'extended';
  traces: PositionTrace[]; // traces for newly computed positions
}

/**
 * Make the cache consistent with (model state, tokens): drop everything if the model state changed,
 * truncate from the first edited prefix position, then decode the remaining positions incrementally.
 */
export function syncCache(p: CoreProfile, w: ToyWeights, cache: KvCache, stateKey: string, tokens: number[]): CacheSync {
  let c = cache;
  let reason: CacheSync['reason'] = 'none';
  let invalidated = 0;
  if (c.stateKey !== stateKey) {
    invalidated = c.tokens.length;
    c = emptyCache(p, stateKey);
    reason = 'model-state-changed';
  } else {
    let i = 0;
    while (i < c.tokens.length && i < tokens.length && c.tokens[i] === tokens[i]) i++;
    if (i < c.tokens.length) {
      invalidated = c.tokens.length - i;
      c = { stateKey, tokens: c.tokens.slice(0, i), layers: c.layers.map((L) => ({ k: L.k.slice(0, i), v: L.v.slice(0, i) })) };
      reason = 'prefix-edited';
    }
  }
  const reused = c.tokens.length;
  const traces: PositionTrace[] = [];
  for (let t = c.tokens.length; t < tokens.length; t++) {
    const r = decodeStep(p, w, c, tokens[t]);
    c = r.cache;
    traces.push(r.trace);
  }
  if (reason === 'none' && traces.length > 0) reason = 'extended';
  return { cache: c, reusedPositions: reused, invalidatedPositions: invalidated, reason, traces };
}

export function modelStateKey(p: CoreProfile, seed: number, editsVersion: number): string {
  return `${p.id}|${p.d}|${p.layers}|${p.hq}|${p.hkv}|${p.headDim}|${p.ropeTheta}|${p.moe ? `${p.moe.numExperts}/${p.moe.topK}` : 'dense'}|seed=${seed}|edits=${editsVersion}`;
}

export const CACHE_TOLERANCE = 1e-9;
export function compareLogits(a: Vec, b: Vec) {
  const diff = maxAbsDiff(a, b);
  return { diff, ok: diff <= CACHE_TOLERANCE };
}
