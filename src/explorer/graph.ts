import type { CoreProfile } from '../models/profiles';
import { accounting, prod } from '../models/shapes';
import { fmtB, fmtInt } from '../components/ui';
import type { ExplorerState } from '../state/route';
import type { Role } from './describe';

export interface NodeSpec {
  id: string;
  x: number;
  y: number;
  w: number;
  h: number;
  label: string;
  sub?: string[]; // revealed progressively as the node gets larger on screen
  role: Role;
  open?: Partial<ExplorerState>; // deeper semantic level, if any
  shape?: 'rect' | 'circle';
}
export interface EdgeSpec { id: string; pts: [number, number][]; label?: string; resid?: boolean }
export interface GraphSpec { width: number; height: number; nodes: NodeSpec[]; edges: EdgeSpec[]; title: string; description: string }

const bottom = (n: NodeSpec): [number, number] => [n.x + n.w / 2, n.y + n.h];
const top = (n: NodeSpec): [number, number] => [n.x + n.w / 2, n.y];
const link = (a: NodeSpec, b: NodeSpec, label?: string): EdgeSpec => ({ id: `${a.id}->${b.id}`, pts: [bottom(a), top(b)], label });

export function modelGraph(p: CoreProfile, opts: { compressed: boolean; selectedBlock?: number }): GraphSpec {
  const a = accounting(p);
  const W = 760;
  const cx = W / 2;
  const nodes: NodeSpec[] = [];
  const edges: EdgeSpec[] = [];
  const tok: NodeSpec = { id: 'tokenizer', x: cx - 170, y: 16, w: 340, h: 50, label: 'Chat template → tokenizer', sub: ['text → token ids [B, T]', 'no learned weights in the checkpoint'], role: 'op' };
  const emb: NodeSpec = { id: 'embedding', x: cx - 170, y: 100, w: 340, h: 56, label: 'Input embedding', sub: [`[${fmtInt(p.vocabRows)}, ${fmtInt(p.d)}] · ${fmtB(a.embedding)} params`, 'model.embed_tokens.weight'], role: 'param', open: { m: p.id, lv: 3, mod: 'embedding' } };
  nodes.push(tok, emb);
  edges.push(link(tok, emb, 'ids [B,T]'));
  const blockSub = (i: number) => [p.kind === 'dense' ? 'attn + FFN' : `attn + MoE ${p.moe!.topK}/${p.moe!.numExperts}`, `${fmtB(a.perBlock)} params`, `model.layers.${i}.*`];
  const stackTop = 196;
  let stackBottom: number;
  const blocks: NodeSpec[] = [];
  if (!opts.compressed) {
    const cols = 6, bw = 106, bh = 50, gap = 10;
    const gx = cx - (cols * bw + (cols - 1) * gap) / 2;
    for (let i = 0; i < p.layers; i++) {
      const r = Math.floor(i / cols), c = i % cols;
      blocks.push({ id: `block:${i}`, x: gx + c * (bw + gap), y: stackTop + 34 + r * (bh + gap), w: bw, h: bh, label: `Block ${i}`, sub: blockSub(i), role: 'group', open: { m: p.id, lv: 4, b: i } });
    }
    stackBottom = stackTop + 34 + Math.ceil(p.layers / cols) * (bh + gap) + 6;
  } else {
    const sel = opts.selectedBlock ?? 0;
    const keep = [...new Set([0, 1, sel - 1, sel, sel + 1, p.layers - 2, p.layers - 1].filter((i) => i >= 0 && i < p.layers))].sort((x, y) => x - y);
    let y = stackTop + 34;
    let prev = -1;
    for (const i of keep) {
      if (i > prev + 1) {
        const n = i - prev - 1;
        blocks.push({ id: `gap:${prev + 1}-${i - 1}`, x: cx - 160, y, w: 320, h: 34, label: `… blocks ${prev + 1}–${i - 1} (${n} more, each with distinct weights)`, role: 'op' });
        y += 44;
      }
      blocks.push({ id: `block:${i}`, x: cx - 160, y, w: 320, h: 46, label: `Block ${i}`, sub: blockSub(i), role: 'group', open: { m: p.id, lv: 4, b: i } });
      y += 56;
      prev = i;
    }
    stackBottom = y + 4;
  }
  const stack: NodeSpec = { id: 'stack', x: 40, y: stackTop, w: W - 80, h: stackBottom - stackTop, label: `Decoder stack · ${p.layers} blocks`, sub: [`${fmtB(a.perBlock * p.layers)} params in blocks`], role: 'group' };
  nodes.push(stack, ...blocks);
  edges.push(link(emb, stack, `X [B,T,${p.d}]`));
  const fn: NodeSpec = { id: 'final_norm', x: cx - 170, y: stackBottom + 34, w: 340, h: 46, label: 'Final RMSNorm', sub: [`γ [${fmtInt(p.d)}]`, 'model.norm.weight'], role: 'norm', open: { m: p.id, lv: 3, mod: 'final_norm' } };
  const head: NodeSpec = { id: 'lm_head', x: cx - 170, y: fn.y + 80, w: 340, h: 56, label: 'Vocabulary projection (untied)', sub: [`[${fmtInt(p.vocabRows)}, ${fmtInt(p.d)}] · ${fmtB(a.head)} params`, 'lm_head.weight'], role: 'param', open: { m: p.id, lv: 3, mod: 'lm_head' } };
  const samp: NodeSpec = { id: 'sampling', x: cx - 170, y: head.y + 90, w: 340, h: 50, label: 'Sampling → next token', sub: ['temperature · top-k · top-p · draw', 'no learned weights'], role: 'op', open: { m: p.id, lv: 3, mod: 'sampling' } };
  nodes.push(fn, head, samp);
  edges.push(link(stack, fn, `[B,T,${p.d}]`), link(fn, head), link(head, samp, `logits [${fmtInt(p.vocabRows)}]`));
  edges.push({ id: 'loop', pts: [[samp.x + samp.w, samp.y + samp.h / 2], [W - 12, samp.y + samp.h / 2], [W - 12, tok.y + tok.h / 2], [tok.x + tok.w, tok.y + tok.h / 2]], label: 'append token, repeat' });
  return { width: W, height: samp.y + samp.h + 24, nodes, edges, title: `${p.name}: full model`, description: `Full ${p.name} pipeline: tokenizer, embedding, ${p.layers} decoder blocks, final norm, untied vocabulary projection and sampling.` };
}

export function blockGraph(p: CoreProfile, b: number): GraphSpec {
  const a = accounting(p);
  const W = 640, cx = 360, w = 300;
  const mk = (id: string, y: number, h: number, label: string, role: Role, sub?: string[], open?: Partial<ExplorerState>, shape: 'rect' | 'circle' = 'rect'): NodeSpec => ({ id, x: shape === 'circle' ? cx - h / 2 : cx - w / 2, y, w: shape === 'circle' ? h : w, h, label, role, sub, open, shape });
  const xin = mk('x_in', 16, 44, `x (block ${b} input)`, 'act', [`[B, T, ${fmtInt(p.d)}]`]);
  const n1 = mk('attn_norm', 96, 50, 'RMSNorm (attention)', 'norm', [`γ [${fmtInt(p.d)}] · input_layernorm`], { m: p.id, lv: 3, b, mod: 'attn_norm' });
  const at = mk('attention', 182, 74, 'Self-attention', 'param', [`GQA ${p.hq} Q / ${p.hkv} KV heads · d_h ${p.headDim}`, `${fmtB(a.attentionPerBlock + 2 * p.headDim)} params · QK-norm · RoPE`], { m: p.id, lv: 3, b, mod: 'attention' });
  const r1 = mk('residual1', 292, 44, '+', 'op', ['h = x + attn'], undefined, 'circle');
  const n2 = mk('ffn_norm', 372, 50, 'RMSNorm (FFN)', 'norm', [`γ [${fmtInt(p.d)}] · post_attention_layernorm`], { m: p.id, lv: 3, b, mod: 'ffn_norm' });
  const ff = p.kind === 'dense'
    ? mk('ffn', 458, 74, 'SwiGLU FFN', 'param', [`d ${fmtInt(p.d)} → f ${fmtInt(p.ffnWidth!)} → d`, `${fmtB(a.ffnPerBlock)} params (3·d·f)`], { m: p.id, lv: 3, b, mod: 'ffn' })
    : mk('moe', 458, 74, `MoE: ${p.moe!.numExperts} experts, top-${p.moe!.topK}`, 'param', [`router [${p.moe!.numExperts}, ${fmtInt(p.d)}] · expert width ${p.moe!.expertWidth}`, `${fmtB(a.ffnPerBlock + a.routerPerBlock)} stored · no shared expert`], { m: p.id, lv: 3, b, mod: 'moe' });
  const r2 = mk('residual2', 568, 44, '+', 'op', ['y = h + ffn'], undefined, 'circle');
  const xout = mk('x_out', 648, 44, `y (to ${b === p.layers - 1 ? 'final norm' : `block ${b + 1}`})`, 'act', [`[B, T, ${fmtInt(p.d)}]`]);
  const nodes = [xin, n1, at, r1, n2, ff, r2, xout];
  const sh = `[B,T,${p.d}]`;
  const edges: EdgeSpec[] = [link(xin, n1, sh), link(n1, at, sh), link(at, r1, sh), link(r1, n2, sh), link(n2, ff, sh), link(ff, r2, sh), link(r2, xout, sh)];
  const lx = 70;
  edges.push({ id: 'skip1', resid: true, pts: [[xin.x, xin.y + xin.h / 2], [lx, xin.y + xin.h / 2], [lx, r1.y + r1.h / 2], [r1.x, r1.y + r1.h / 2]], label: 'residual x' });
  edges.push({ id: 'skip2', resid: true, pts: [[r1.x, r1.y + r1.h / 2 + 6], [lx + 24, r1.y + r1.h / 2 + 6], [lx + 24, r2.y + r2.h / 2], [r2.x, r2.y + r2.h / 2]], label: 'residual h' });
  return { width: W, height: 712, nodes, edges, title: `Decoder block ${b}`, description: `Pre-norm decoder block ${b}: RMSNorm, attention, residual add, RMSNorm, ${p.kind === 'dense' ? 'SwiGLU FFN' : 'MoE'}, residual add.` };
}

export function moduleGraph(p: CoreProfile, mod: string, b: number, selectedExpert?: number): GraphSpec {
  const qw = p.hq * p.headDim, kvw = p.hkv * p.headDim;
  const nodes: NodeSpec[] = [];
  const edges: EdgeSpec[] = [];
  const n = (id: string, x: number, y: number, w: number, h: number, label: string, role: Role, sub?: string[], open?: Partial<ExplorerState>): NodeSpec => {
    const node = { id, x, y, w, h, label, role, sub, open };
    nodes.push(node);
    return node;
  };
  const T = (key: string, extra: Partial<ExplorerState> = {}): Partial<ExplorerState> => ({ m: p.id, lv: 2, b, t: key, ...extra });
  const pc = (s: number[]) => `${fmtInt(prod(s))} params`;
  if (mod === 'attention') {
    const W = 760;
    const inp = n('x_in', 280, 14, 200, 40, 'normalized x', 'act', [`[B, T, ${p.d}]`]);
    const q = n('tensor:q_proj', 40, 96, 200, 56, 'q_proj  W_Q', 'param', [`[${qw}, ${p.d}]`, pc([qw, p.d])], T('q_proj'));
    const k = n('tensor:k_proj', 280, 96, 200, 56, 'k_proj  W_K', 'param', [`[${kvw}, ${p.d}]`, pc([kvw, p.d])], T('k_proj'));
    const v = n('tensor:v_proj', 520, 96, 200, 56, 'v_proj  W_V', 'param', [`[${kvw}, ${p.d}]`, pc([kvw, p.d])], T('v_proj'));
    const rs = n('reshape', 40, 190, 680, 36, `reshape into heads: Q ${p.hq}×${p.headDim} · K ${p.hkv}×${p.headDim} · V ${p.hkv}×${p.headDim}`, 'op');
    n('tensor:q_norm', 40, 260, 200, 50, 'q_norm (per head)', 'norm', [`γ [${p.headDim}] shared by ${p.hq} heads`], T('q_norm'));
    n('tensor:k_norm', 280, 260, 200, 50, 'k_norm (per head)', 'norm', [`γ [${p.headDim}] shared by ${p.hkv} heads`], T('k_norm'));
    n('vpass', 520, 260, 200, 50, 'V: no norm, no RoPE', 'act', [`[B, ${p.hkv}, T, ${p.headDim}]`]);
    n('rope', 40, 344, 440, 44, `RoPE on Q and K (θ = ${fmtInt(p.ropeTheta)}, absolute positions)`, 'op', ['pairs (i, i + d_h/2) rotated · no weights']);
    const sd = n('sdpa', 40, 424, 680, 60, `causal softmax(QKᵀ/√${p.headDim}) V · GQA: ${p.hq / p.hkv} query heads per K/V head`, 'op', [`logical scores [B, ${p.hq}, Tq, Tk] · future masked`, 'kernels may never store the full matrix']);
    const cc = n('concat', 160, 520, 440, 36, `concatenate heads → [B, T, ${qw}]`, 'op');
    const o = n('tensor:o_proj', 260, 590, 240, 56, 'o_proj  W_O', 'param', [`[${p.d}, ${qw}]`, pc([p.d, qw])], T('o_proj'));
    const out = n('x_out', 280, 680, 200, 40, 'to residual add', 'act', [`[B, T, ${p.d}]`]);
    edges.push(link(inp, q), link(inp, k), link(inp, v), link(q, rs), link(k, rs), link(v, rs));
    edges.push({ id: 'rs-qn', pts: [[140, 226], [140, 260]] }, { id: 'rs-kn', pts: [[380, 226], [380, 260]] }, { id: 'rs-v', pts: [[620, 226], [620, 260]] });
    edges.push({ id: 'qn-r', pts: [[140, 310], [140, 344]] }, { id: 'kn-r', pts: [[380, 310], [380, 344]] }, { id: 'r-s', pts: [[260, 388], [260, 424]], label: 'Q, K rotated' }, { id: 'v-s', pts: [[620, 310], [620, 424]], label: 'V' });
    edges.push(link(sd, cc), link(cc, o), link(o, out));
    return { width: W, height: 740, nodes, edges, title: `Attention · block ${b}`, description: 'Attention module: Q/K/V projections, head reshape, Q/K RMS norms, RoPE, causal grouped-query attention, concatenation, output projection.' };
  }
  if (mod === 'ffn') {
    const f = p.ffnWidth!;
    const inp = n('x_in', 240, 14, 240, 40, 'normalized h', 'act', [`[B, T, ${p.d}]`]);
    const g = n('tensor:gate_proj', 40, 100, 280, 60, 'gate_proj  W_gate', 'param', [`[${fmtInt(f)}, ${fmtInt(p.d)}]`, pc([f, p.d])], T('gate_proj'));
    const u = n('tensor:up_proj', 400, 100, 280, 60, 'up_proj  W_up', 'param', [`[${fmtInt(f)}, ${fmtInt(p.d)}]`, pc([f, p.d])], T('up_proj'));
    const s = n('silu', 80, 204, 200, 44, 'SiLU(a) = a·σ(a)', 'op', [`g [B, T, ${fmtInt(f)}]`]);
    const m = n('mul', 260, 290, 200, 44, 'z = g ⊙ u', 'op', [`[B, T, ${fmtInt(f)}] gated channels`]);
    const d = n('tensor:down_proj', 220, 376, 280, 60, 'down_proj  W_down', 'param', [`[${fmtInt(p.d)}, ${fmtInt(f)}]`, pc([p.d, f])], T('down_proj'));
    const out = n('x_out', 240, 476, 240, 40, 'to residual add', 'act', [`[B, T, ${p.d}]`]);
    edges.push(link(inp, g), link(inp, u), link(g, s, 'pre-activation'), link(s, m, 'g'), { id: 'u-m', pts: [[540, 160], [540, 312], [460, 312]], label: 'u' }, link(m, d, 'z'), link(d, out));
    return { width: 720, height: 540, nodes, edges, title: `SwiGLU FFN · block ${b}`, description: 'SwiGLU feed-forward: gate and up projections, SiLU, elementwise gate, down projection.' };
  }
  if (mod === 'moe') {
    const m = p.moe!;
    const inp = n('x_in', 260, 14, 240, 40, 'normalized h (one token)', 'act', [`[${p.d}]`]);
    const r = n('tensor:router', 240, 86, 280, 56, 'router  mlp.gate.weight', 'param', [`[${m.numExperts}, ${p.d}] · ${fmtInt(m.numExperts * p.d)} params`], T('router'));
    const sm = n('softmax', 240, 170, 280, 36, `softmax over all ${m.numExperts} logits`, 'op');
    const tk = n('topk', 240, 230, 280, 36, `top-${m.topK} experts`, 'op');
    const rn = n('renorm', 240, 290, 280, 36, `renormalize: α sums to 1 (norm_topk_prob=${m.normTopkProb})`, 'op');
    const ex = n('experts', 30, 350, 700, 34 + Math.ceil(m.numExperts / 16) * 34 + 10, `${m.numExperts} experts (${m.topK} selected per token in each block; unselected still stored)`, 'group');
    const cols = 16, ew = 40, eh = 28;
    for (let e = 0; e < m.numExperts; e++) {
      const rr = Math.floor(e / cols), cc = e % cols;
      nodes.push({ id: `expert:${e}`, x: 46 + cc * (ew + 2), y: 384 + rr * (eh + 6), w: ew, h: eh, label: String(e), role: e === selectedExpert ? 'act' : 'param', open: { m: p.id, lv: 2, b, t: 'expert_gate_proj', e } });
    }
    const ws = n('wsum', 240, ex.y + ex.h + 30, 280, 40, 'y = Σ α_e · Expert_e(x)', 'op');
    const out = n('x_out', 260, ws.y + 76, 240, 40, 'to residual add', 'act', [`[${p.d}]`]);
    edges.push(link(inp, r), link(r, sm, `${m.numExperts} logits`), link(sm, tk), link(tk, rn), link(rn, ex, `${m.topK} ids + α`), link(ex, ws), link(ws, out));
    edges.push({ id: 'x-ex', pts: [[260, 34], [16, 34], [16, 360], [30, 360]], label: 'x to selected experts', resid: false });
    return { width: 760, height: out.y + 60, nodes, edges, title: `MoE · block ${b}`, description: `Mixture of experts: router, softmax, top-${m.topK}, renormalization, ${m.numExperts} expert FFNs, weighted sum.` };
  }
  if (mod === 'attn_norm' || mod === 'ffn_norm' || mod === 'final_norm') {
    const key = mod === 'attn_norm' ? 'input_layernorm' : mod === 'ffn_norm' ? 'post_attention_layernorm' : 'norm';
    const inp = n('x_in', 200, 14, 280, 40, mod === 'ffn_norm' ? 'h = x + attn' : mod === 'final_norm' ? 'last block output' : 'block input x', 'act', [`[B, T, ${p.d}]`]);
    const sq = n('sq', 200, 90, 280, 40, 'mean(x²) over features', 'op', ['no mean subtraction']);
    const rq = n('rsqrt', 200, 160, 280, 40, `1/√(mean + ε), ε = ${p.rmsEps}`, 'op');
    const sx = n('scalex', 200, 230, 280, 40, 'x · scale', 'op');
    const gm = n(`tensor:${key}`, 200, 300, 280, 56, `× γ (elementwise)`, 'norm', [`γ [${fmtInt(p.d)}] · ${key}.weight`], { m: p.id, lv: 2, b: mod === 'final_norm' ? undefined : b, t: key });
    const out = n('x_out', 200, 392, 280, 40, 'normalized output', 'act', [`[B, T, ${p.d}]`]);
    edges.push(link(inp, sq), link(sq, rq), link(rq, sx), { id: 'x-sx', pts: [[200, 34], [150, 34], [150, 250], [200, 250]], label: 'x' }, link(sx, gm), link(gm, out));
    return { width: 680, height: 450, nodes, edges, title: 'RMSNorm', description: 'RMSNorm: mean of squares, reciprocal square root, scale, multiply by learned gain.' };
  }
  if (mod === 'embedding') {
    const ids = n('ids', 200, 14, 280, 40, 'token ids [B, T]', 'act');
    const tb = n('tensor:embed_tokens', 140, 90, 400, 70, 'Embedding table E', 'param', [`[${fmtInt(p.vocabRows)}, ${fmtInt(p.d)}] · ${fmtB(prod([p.vocabRows, p.d]))} params`, 'model.embed_tokens.weight'], { m: p.id, lv: 2, t: 'embed_tokens' });
    const lk = n('lookup', 200, 196, 280, 40, 'row lookup x_t = E[id_t]', 'op');
    const out = n('x_out', 200, 270, 280, 40, `X [B, T, ${p.d}] → block 0`, 'act');
    edges.push(link(ids, tb), link(tb, lk), link(lk, out));
    return { width: 680, height: 330, nodes, edges, title: 'Embedding', description: 'Embedding lookup: one table row per token id.' };
  }
  if (mod === 'lm_head') {
    const inp = n('x_in', 200, 14, 280, 40, 'final normalized h', 'act', [`[B, T, ${p.d}]`]);
    const hd = n('tensor:lm_head', 140, 90, 400, 70, 'lm_head W_vocab (untied)', 'param', [`[${fmtInt(p.vocabRows)}, ${fmtInt(p.d)}] · ${fmtB(prod([p.vocabRows, p.d]))} params`, 'lm_head.weight'], { m: p.id, lv: 2, t: 'lm_head' });
    const lg = n('logits', 200, 196, 280, 40, `logits [${fmtInt(p.vocabRows)}]`, 'act');
    const sp = n('sampling', 200, 270, 280, 40, 'to sampling', 'op', [], { m: p.id, lv: 3, mod: 'sampling' });
    edges.push(link(inp, hd), link(hd, lg), link(lg, sp));
    return { width: 680, height: 330, nodes, edges, title: 'Vocabulary projection', description: 'Output head: one logit per vocabulary row.' };
  }
  if (mod === 'sampling') {
    const lg = n('logits', 200, 14, 280, 40, 'logits (last position)', 'act');
    const tp = n('temp', 200, 84, 280, 40, 'divide by temperature (0 = greedy)', 'op');
    const fl = n('filter', 200, 154, 280, 40, 'top-k → top-p → renormalize', 'op');
    const dr = n('draw', 200, 224, 280, 40, 'seeded random draw', 'op');
    const out = n('x_out', 200, 294, 280, 40, 'next token id → append', 'act');
    edges.push(link(lg, tp), link(tp, fl), link(fl, dr), link(dr, out));
    return { width: 680, height: 350, nodes, edges, title: 'Sampling', description: 'Decoding procedure: temperature, filtering, renormalization, random draw. No learned parameters.' };
  }
  // residual
  const xa = n('x_in', 60, 30, 240, 44, mod === 'residual1' ? 'x (block input)' : 'h', 'act', [`[B, T, ${p.d}]`]);
  const xb = n('branch', 380, 30, 240, 44, mod === 'residual1' ? 'attention output' : `${p.kind === 'dense' ? 'FFN' : 'MoE'} output`, 'act', [`[B, T, ${p.d}]`]);
  const ad = n(mod, 290, 140, 100, 60, '+', 'op', ['no weights']);
  const out = n('x_out', 220, 250, 240, 44, mod === 'residual1' ? 'h' : 'y', 'act', [`[B, T, ${p.d}]`]);
  edges.push({ id: 'a', pts: [bottom(xa), [340, 140]] }, { id: 'b', pts: [bottom(xb), [340, 140]] }, link(ad, out));
  return { width: 680, height: 320, nodes, edges, title: 'Residual addition', description: 'Elementwise addition of two activation vectors. No learned parameters.' };
}
