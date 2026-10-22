import { dot, type Vec } from './linalg';
import { sigmoid } from './ops';
import { deriveSeed, makeRng, nextNormal } from './rng';

// ---------------------------------------------------------------------------
// TOY ANALOGUES of computations whose exact form is verified in the Qwen3.8-Flash-Next implementation
// (transformers modeling_qwen4_exp.py). Tiny synthetic dimensions; not a Qwen3.8 run.
// ---------------------------------------------------------------------------

const l2norm = (v: Vec, eps = 1e-6) => {
  const n = Math.sqrt(v.reduce((s, x) => s + x * x, 0) + eps);
  return v.map((x) => x / n);
};

export interface GdnStep {
  t: number;
  decay: number; // exp(g_t), g_t = -exp(A_log) * softplus(a_t + dt_bias)
  beta: number; // sigmoid(b_t)
  kvMem: Vec; // S^T k after decay (what the state currently returns for key k)
  delta: Vec; // beta * (v - kvMem)
  stateAfter: number[][]; // [dk][dv]
  out: Vec; // S^T q
}

/**
 * torch_recurrent_gated_delta_rule for ONE head:
 *   S <- S * exp(g_t)
 *   kv_mem = S^T k_t ;  delta = beta_t * (v_t - kv_mem)
 *   S <- S + k_t delta^T ;  o_t = S^T q_t
 * with q, k L2-normalized and q scaled by 1/sqrt(dk), as in the implementation.
 */
export function gatedDeltaRule(q: Vec[], k: Vec[], v: Vec[], g: number[], beta: number[]): GdnStep[] {
  const dk = k[0].length, dv = v[0].length;
  let S = Array.from({ length: dk }, () => new Array(dv).fill(0));
  const steps: GdnStep[] = [];
  for (let t = 0; t < q.length; t++) {
    const qt = l2norm(q[t]).map((x) => x / Math.sqrt(dk));
    const kt = l2norm(k[t]);
    const decay = Math.exp(g[t]);
    S = S.map((r) => r.map((x) => x * decay));
    const kvMem = Array.from({ length: dv }, (_, j) => S.reduce((s, r, i) => s + r[j] * kt[i], 0));
    const delta = v[t].map((x, j) => beta[t] * (x - kvMem[j]));
    S = S.map((r, i) => r.map((x, j) => x + kt[i] * delta[j]));
    const out = Array.from({ length: dv }, (_, j) => S.reduce((s, r, i) => s + r[j] * qt[i], 0));
    steps.push({ t, decay, beta: beta[t], kvMem, delta, stateAfter: S.map((r) => r.slice()), out });
  }
  return steps;
}

export const softplus = (x: number) => (x > 20 ? x : Math.log1p(Math.exp(x)));
export function gdnGate(aLog: number, a: number, dtBias: number): number {
  return -Math.exp(aLog) * softplus(a + dtBias);
}

export function toyGdnInputs(seed: number, T: number, dk = 4, dv = 4) {
  const rng = makeRng(deriveSeed(seed, 'gdn-toy'));
  const r = (n: number) => Array.from({ length: n }, () => nextNormal(rng));
  return {
    q: Array.from({ length: T }, () => r(dk)),
    k: Array.from({ length: T }, () => r(dk)),
    v: Array.from({ length: T }, () => r(dv)),
    a: r(T),
    b: r(T),
  };
}

export interface QsaSelection {
  blocks: { start: number; tokens: number[]; score: number; selected: boolean }[];
  tail: number[];
  selectedTokens: number[];
  blockBudget: number;
}

/**
 * QSA indexer selection for one query (Qwen4ExpTextQSAIndexer, simplified to toy size):
 * visible keys are grouped into complete blocks of `ratio` tokens, keys are mean-pooled per block,
 * score = sum over indexer heads of ReLU(q_h . k_block) / sqrt(dim), the top (budget / ratio) blocks are kept,
 * and the incomplete tail block is always kept. (The toy omits RoPE and the key RMSNorm.)
 */
export function qsaSelect(queryHeads: Vec[], keys: Vec[], visible: number, budgetTokens: number, ratio: number): QsaSelection {
  const nBlocks = Math.floor(visible / ratio);
  const blockBudget = Math.floor(budgetTokens / ratio);
  const dim = keys[0]?.length ?? 1;
  const blocks = Array.from({ length: nBlocks }, (_, b) => {
    const tokens = Array.from({ length: ratio }, (_, i) => b * ratio + i);
    const pooled = keys[0].map((_, j) => tokens.reduce((s, t) => s + keys[t][j], 0) / ratio);
    const score = queryHeads.reduce((s, qh) => s + Math.max(0, dot(qh, pooled)), 0) / Math.sqrt(dim);
    return { start: b * ratio, tokens, score, selected: false };
  });
  const order = blocks.map((b, i) => [b.score, i] as const).sort((a, b) => b[0] - a[0] || a[1] - b[1]);
  order.slice(0, Math.min(blockBudget, nBlocks)).forEach(([, i]) => (blocks[i].selected = true));
  const tail = Array.from({ length: visible - nBlocks * ratio }, (_, i) => nBlocks * ratio + i);
  const selectedTokens = [...blocks.filter((b) => b.selected).flatMap((b) => b.tokens), ...tail].sort((a, b) => a - b);
  return { blocks, tail, selectedTokens, blockBudget };
}

/**
 * Gated residual (Qwen4ExpTextGatedResidual) for one token at toy size, with zero-centred norm gains (weight 1 + 0):
 * read:  x_in = mean over branches of ( sigmoid(W_up silu(W_down n / n_br)) * n_b )
 * write: branch_b += 2 * sigmoid(w_b . n / n_br) * y
 */
export function gatedResidualToy(branches: Vec[], down: number[][], up: number[][], inject: number[][], block: (x: Vec) => Vec) {
  const nb = branches.length;
  const d = branches[0].length;
  const normed = branches.map((b) => {
    const inv = 1 / Math.sqrt(b.reduce((s, x) => s + x * x, 0) / d + 1e-6);
    return b.map((x) => x * inv);
  });
  const flat = normed.flat();
  const hidden = down.map((r) => {
    const a = dot(r, flat) / nb;
    return a * sigmoid(a);
  });
  const readGate = up.map((r) => sigmoid(dot(r, hidden)));
  const gated = normed.map((b, bi) => b.map((x, j) => readGate[bi * d + j] * x));
  const xin = Array.from({ length: d }, (_, j) => gated.reduce((s, b) => s + b[j], 0) / nb);
  const y = block(xin);
  const writeGate = inject.map((r) => 2 * sigmoid(dot(r, flat) / nb));
  const after = branches.map((b, bi) => b.map((x, j) => x + writeGate[bi] * y[j]));
  return { normed, readGate, xin, y, writeGate, after };
}
