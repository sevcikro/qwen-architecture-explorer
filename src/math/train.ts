import { Value, vdot, vsum } from './autograd';
import { dot, type Mat, type Vec } from './linalg';
import { rmsnorm, stableSoftmax, swiglu } from './ops';
import { deriveSeed, makeRng, nextNormal } from './rng';

// ---------------------------------------------------------------------------
// Training toy: one dense Qwen-style residual FFN block with teacher-forced next-token loss.
//   e = E[token];  h = e + SwiGLU(RMSNorm(e; g1));  logits = W_out RMSNorm(h; g2);  loss = mean CE(next token)
// Deliberately small and attention-free so every gradient can be shown and finite-difference checked.
// This is NOT the Qwen training recipe.
// ---------------------------------------------------------------------------

export interface TrainParams {
  embed: Mat; // [V, d]
  g1: Vec; // [d]
  gate: Mat; // [f, d]
  up: Mat; // [f, d]
  down: Mat; // [d, f]
  g2: Vec; // [d]
  head: Mat; // [V, d]
}
export const TRAIN_DIMS = { V: 12, d: 4, f: 6 };
export const TRAIN_EPS = 1e-6;
export const TRAIN_SEQUENCE = [1, 2, 3, 4, 1, 5, 11]; // "the cat sat on the mat ."

export function initTrainParams(seed: number): TrainParams {
  const { V, d, f } = TRAIN_DIMS;
  const g = (label: string, r: number, c: number, std: number) => {
    const rng = makeRng(deriveSeed(seed, `train.${label}`));
    return Array.from({ length: r }, () => Array.from({ length: c }, () => nextNormal(rng) * std));
  };
  const gain = (label: string, n: number) => {
    const rng = makeRng(deriveSeed(seed, `train.${label}`));
    return Array.from({ length: n }, () => 1 + 0.1 * nextNormal(rng));
  };
  return {
    embed: g('embed', V, d, 1),
    g1: gain('g1', d),
    gate: g('gate', f, d, 0.5),
    up: g('up', f, d, 0.5),
    down: g('down', d, f, 0.4),
    g2: gain('g2', d),
    head: g('head', V, d, 0.5),
  };
}

export type ParamName = keyof TrainParams;
export interface ParamRef { tensor: ParamName; row: number; col: number | null }

export const PARAM_SHAPES: Record<ParamName, [number, number] | [number]> = {
  embed: [TRAIN_DIMS.V, TRAIN_DIMS.d], g1: [TRAIN_DIMS.d], gate: [TRAIN_DIMS.f, TRAIN_DIMS.d], up: [TRAIN_DIMS.f, TRAIN_DIMS.d],
  down: [TRAIN_DIMS.d, TRAIN_DIMS.f], g2: [TRAIN_DIMS.d], head: [TRAIN_DIMS.V, TRAIN_DIMS.d],
};

export const getParam = (p: TrainParams, r: ParamRef) => (r.col === null ? (p[r.tensor] as Vec)[r.row] : (p[r.tensor] as Mat)[r.row][r.col]);
export function withParam(p: TrainParams, r: ParamRef, v: number): TrainParams {
  const c: TrainParams = structuredClone(p);
  if (r.col === null) (c[r.tensor] as Vec)[r.row] = v;
  else (c[r.tensor] as Mat)[r.row][r.col] = v;
  return c;
}

/** Plain-number forward (independent of the autograd path). Returns per-position details. */
export function trainForward(p: TrainParams, seq: number[]) {
  const positions = [];
  let total = 0;
  for (let t = 0; t < seq.length - 1; t++) {
    const tok = seq[t], target = seq[t + 1];
    const e = p.embed[tok];
    const n1 = rmsnorm(e, p.g1, TRAIN_EPS);
    const f = swiglu(n1, p.gate, p.up, p.down);
    const h = e.map((v, i) => v + f.y[i]);
    const n2 = rmsnorm(h, p.g2, TRAIN_EPS);
    const logits = p.head.map((r) => dot(n2, r));
    const probs = stableSoftmax(logits).probs;
    // cross-entropy via log-sum-exp so a confident wrong prediction gives a large finite loss, not -log(0)
    const m = Math.max(...logits);
    const loss = Math.log(logits.reduce((s, l) => s + Math.exp(l - m), 0)) + m - logits[target];
    total += loss;
    positions.push({ t, tok, target, e, n1, ffn: f, h, n2, logits, probs, loss, pTarget: probs[target] });
  }
  return { positions, loss: total / (seq.length - 1) };
}

/** Autograd forward + backward. Returns loss, a gradient with the same structure as the params, and labeled intermediate grads. */
export function lossAndGrad(p: TrainParams, seq: number[]) {
  const wrapM = (m: Mat, name: string) => m.map((r, i) => r.map((v, j) => new Value(v, [], '', `${name}[${i},${j}]`)));
  const wrapV = (v: Vec, name: string) => v.map((x, i) => new Value(x, [], '', `${name}[${i}]`));
  const P = { embed: wrapM(p.embed, 'embed'), g1: wrapV(p.g1, 'g1'), gate: wrapM(p.gate, 'gate'), up: wrapM(p.up, 'up'), down: wrapM(p.down, 'down'), g2: wrapV(p.g2, 'g2'), head: wrapM(p.head, 'head') };
  const rms = (x: Value[], g: Value[]) => {
    const ms = vsum(x.map((v) => v.mul(v))).mul(1 / x.length);
    const inv = ms.add(TRAIN_EPS).rsqrt();
    return x.map((v, i) => g[i].mul(v.mul(inv)));
  };
  const inter: { t: number; logits: Value[]; n2: Value[]; h: Value[]; z: Value[]; g: Value[]; u: Value[]; n1: Value[] }[] = [];
  const losses: Value[] = [];
  for (let t = 0; t < seq.length - 1; t++) {
    const e = P.embed[seq[t]];
    const n1 = rms(e, P.g1);
    const gPre = P.gate.map((r) => vdot(n1, r));
    const g = gPre.map((a) => a.silu());
    const u = P.up.map((r) => vdot(n1, r));
    const z = g.map((a, j) => a.mul(u[j]));
    const y = P.down.map((r) => vdot(z, r));
    const h = e.map((v, i) => v.add(y[i]));
    const n2 = rms(h, P.g2);
    const logits = P.head.map((r) => vdot(n2, r));
    const m = Math.max(...logits.map((l) => l.data));
    const lse = vsum(logits.map((l) => l.sub(m).exp())).log().add(m);
    losses.push(lse.sub(logits[seq[t + 1]]));
    inter.push({ t, logits, n2, h, z, g, u, n1 });
  }
  const loss = vsum(losses).mul(1 / losses.length);
  loss.backward();
  const gM = (m: Value[][]) => m.map((r) => r.map((v) => v.grad));
  const gV = (v: Value[]) => v.map((x) => x.grad);
  const grad: TrainParams = { embed: gM(P.embed), g1: gV(P.g1), gate: gM(P.gate), up: gM(P.up), down: gM(P.down), g2: gV(P.g2), head: gM(P.head) };
  const norm = (vs: Value[]) => Math.sqrt(vs.reduce((s, v) => s + v.grad * v.grad, 0));
  const chain = inter.map((x) => ({
    t: x.t,
    dLogits: norm(x.logits), dFinalNorm: norm(x.n2), dResidual: norm(x.h), dGated: norm(x.z), dGateBranch: norm(x.g), dUpBranch: norm(x.u), dFfnInput: norm(x.n1),
  }));
  return { loss: loss.data, grad, chain };
}

export function sgdStep(p: TrainParams, grad: TrainParams, lr: number): TrainParams {
  const m = (a: Mat, g: Mat) => a.map((r, i) => r.map((v, j) => v - lr * g[i][j]));
  const v = (a: Vec, g: Vec) => a.map((x, i) => x - lr * g[i]);
  return { embed: m(p.embed, grad.embed), g1: v(p.g1, grad.g1), gate: m(p.gate, grad.gate), up: m(p.up, grad.up), down: m(p.down, grad.down), g2: v(p.g2, grad.g2), head: m(p.head, grad.head) };
}

/** Central finite difference on the plain-number forward. */
export function finiteDifference(p: TrainParams, seq: number[], r: ParamRef, h = 1e-5): number {
  const w = getParam(p, r);
  const lp = trainForward(withParam(p, r, w + h), seq).loss;
  const lm = trainForward(withParam(p, r, w - h), seq).loss;
  return (lp - lm) / (2 * h);
}

export const paramCount = (): number => Object.values(PARAM_SHAPES).reduce((s, sh) => s + sh.reduce((a, b) => a * b, 1), 0);
