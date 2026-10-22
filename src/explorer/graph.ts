import type { CoreProfile } from '../models/profiles';
import { accounting, prod } from '../models/shapes';
import { fmtB, fmtInt } from '../components/ui';
import type { ExplorerState } from '../state/route';
import type { Role } from './describe';
import type { Lang } from '../i18n/i18n';

const lk = (lang: Lang) => (en: string, sk: string) => (lang === 'sk' ? sk : en);

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

export function modelGraph(p: CoreProfile, opts: { compressed: boolean; selectedBlock?: number }, lang: Lang = 'en'): GraphSpec {
  const L = lk(lang);
  const P = L('params', 'parametrov');
  const a = accounting(p);
  const W = 760;
  const cx = W / 2;
  const nodes: NodeSpec[] = [];
  const edges: EdgeSpec[] = [];
  const tok: NodeSpec = { id: 'tokenizer', x: cx - 170, y: 16, w: 340, h: 50, label: L('Chat template → tokenizer', 'Šablóna rozhovoru → tokenizér'), sub: [L('text → token ids [B, T]', 'text → ID tokenov [B, T]'), L('no learned weights in the checkpoint', 'v checkpointe bez naučených váh')], role: 'op' };
  const emb: NodeSpec = { id: 'embedding', x: cx - 170, y: 100, w: 340, h: 56, label: L('Input embedding', 'Vstupný embedding'), sub: [`[${fmtInt(p.vocabRows)}, ${fmtInt(p.d)}] · ${fmtB(a.embedding)} ${P}`, 'model.embed_tokens.weight'], role: 'param', open: { m: p.id, lv: 3, mod: 'embedding' } };
  nodes.push(tok, emb);
  edges.push(link(tok, emb, 'ids [B,T]'));
  const blockSub = (i: number) => [p.kind === 'dense' ? 'attn + FFN' : `attn + MoE ${p.moe!.topK}/${p.moe!.numExperts}`, `${fmtB(a.perBlock)} ${P}`, `model.layers.${i}.*`];
  const stackTop = 196;
  let stackBottom: number;
  const blocks: NodeSpec[] = [];
  if (!opts.compressed) {
    const cols = 6, bw = 106, bh = 50, gap = 10;
    const gx = cx - (cols * bw + (cols - 1) * gap) / 2;
    for (let i = 0; i < p.layers; i++) {
      const r = Math.floor(i / cols), c = i % cols;
      blocks.push({ id: `block:${i}`, x: gx + c * (bw + gap), y: stackTop + 34 + r * (bh + gap), w: bw, h: bh, label: `${L('Block', 'Blok')} ${i}`, sub: blockSub(i), role: 'group', open: { m: p.id, lv: 4, b: i } });
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
        blocks.push({ id: `gap:${prev + 1}-${i - 1}`, x: cx - 160, y, w: 320, h: 34, label: L(`… blocks ${prev + 1}–${i - 1} (${n} more, each with distinct weights)`, `… bloky ${prev + 1}–${i - 1} (ďalších ${n}, každý s vlastnými váhami)`), role: 'op' });
        y += 44;
      }
      blocks.push({ id: `block:${i}`, x: cx - 160, y, w: 320, h: 46, label: `${L('Block', 'Blok')} ${i}`, sub: blockSub(i), role: 'group', open: { m: p.id, lv: 4, b: i } });
      y += 56;
      prev = i;
    }
    stackBottom = y + 4;
  }
  const stack: NodeSpec = { id: 'stack', x: 40, y: stackTop, w: W - 80, h: stackBottom - stackTop, label: L(`Decoder stack · ${p.layers} blocks`, `Zásobník dekodéra · ${p.layers} blokov`), sub: [L(`${fmtB(a.perBlock * p.layers)} params in blocks`, `${fmtB(a.perBlock * p.layers)} parametrov v blokoch`)], role: 'group' };
  nodes.push(stack, ...blocks);
  edges.push(link(emb, stack, `X [B,T,${p.d}]`));
  const fn: NodeSpec = { id: 'final_norm', x: cx - 170, y: stackBottom + 34, w: 340, h: 46, label: L('Final RMSNorm', 'Záverečná RMSNorm'), sub: [`γ [${fmtInt(p.d)}]`, 'model.norm.weight'], role: 'norm', open: { m: p.id, lv: 3, mod: 'final_norm' } };
  const head: NodeSpec = { id: 'lm_head', x: cx - 170, y: fn.y + 80, w: 340, h: 56, label: L('Vocabulary projection (untied)', 'Projekcia do slovníka (neviazaná)'), sub: [`[${fmtInt(p.vocabRows)}, ${fmtInt(p.d)}] · ${fmtB(a.head)} ${P}`, 'lm_head.weight'], role: 'param', open: { m: p.id, lv: 3, mod: 'lm_head' } };
  const samp: NodeSpec = { id: 'sampling', x: cx - 170, y: head.y + 90, w: 340, h: 50, label: L('Sampling → next token', 'Vzorkovanie → ďalší token'), sub: [L('temperature · top-k · top-p · draw', 'teplota · top-k · top-p · výber'), L('no learned weights', 'bez naučených váh')], role: 'op', open: { m: p.id, lv: 3, mod: 'sampling' } };
  nodes.push(fn, head, samp);
  edges.push(link(stack, fn, `[B,T,${p.d}]`), link(fn, head), link(head, samp, `logits [${fmtInt(p.vocabRows)}]`));
  edges.push({ id: 'loop', pts: [[samp.x + samp.w, samp.y + samp.h / 2], [W - 12, samp.y + samp.h / 2], [W - 12, tok.y + tok.h / 2], [tok.x + tok.w, tok.y + tok.h / 2]], label: L('append token, repeat', 'pripojiť token, opakovať') });
  return { width: W, height: samp.y + samp.h + 24, nodes, edges, title: L(`${p.name}: full model`, `${p.name}: celý model`), description: L(`Full ${p.name} pipeline: tokenizer, embedding, ${p.layers} decoder blocks, final norm, untied vocabulary projection and sampling.`, `Celý postup ${p.name}: tokenizér, embedding, ${p.layers} blokov dekodéra, záverečná normalizácia, neviazaná projekcia do slovníka a vzorkovanie.`) };
}

export function blockGraph(p: CoreProfile, b: number, lang: Lang = 'en'): GraphSpec {
  const L = lk(lang);
  const P = L('params', 'parametrov');
  const a = accounting(p);
  const W = 640, cx = 360, w = 300;
  const mk = (id: string, y: number, h: number, label: string, role: Role, sub?: string[], open?: Partial<ExplorerState>, shape: 'rect' | 'circle' = 'rect'): NodeSpec => ({ id, x: shape === 'circle' ? cx - h / 2 : cx - w / 2, y, w: shape === 'circle' ? h : w, h, label, role, sub, open, shape });
  const xin = mk('x_in', 16, 44, L(`x (block ${b} input)`, `x (vstup bloku ${b})`), 'act', [`[B, T, ${fmtInt(p.d)}]`]);
  const n1 = mk('attn_norm', 96, 50, L('RMSNorm (attention)', 'RMSNorm (pozornosť)'), 'norm', [`γ [${fmtInt(p.d)}] · input_layernorm`], { m: p.id, lv: 3, b, mod: 'attn_norm' });
  const at = mk('attention', 182, 74, L('Self-attention', 'Pozornosť'), 'param', [L(`GQA ${p.hq} Q / ${p.hkv} KV heads · d_h ${p.headDim}`, `GQA ${p.hq} Q / ${p.hkv} KV hláv · d_h ${p.headDim}`), `${fmtB(a.attentionPerBlock + 2 * p.headDim)} ${P} · QK-norm · RoPE`], { m: p.id, lv: 3, b, mod: 'attention' });
  const r1 = mk('residual1', 292, 44, '+', 'op', ['h = x + attn'], undefined, 'circle');
  const n2 = mk('ffn_norm', 372, 50, 'RMSNorm (FFN)', 'norm', [`γ [${fmtInt(p.d)}] · post_attention_layernorm`], { m: p.id, lv: 3, b, mod: 'ffn_norm' });
  const ff = p.kind === 'dense'
    ? mk('ffn', 458, 74, 'SwiGLU FFN', 'param', [`d ${fmtInt(p.d)} → f ${fmtInt(p.ffnWidth!)} → d`, `${fmtB(a.ffnPerBlock)} ${P} (3·d·f)`], { m: p.id, lv: 3, b, mod: 'ffn' })
    : mk('moe', 458, 74, L(`MoE: ${p.moe!.numExperts} experts, top-${p.moe!.topK}`, `MoE: ${p.moe!.numExperts} expertov, top-${p.moe!.topK}`), 'param', [L(`router [${p.moe!.numExperts}, ${fmtInt(p.d)}] · expert width ${p.moe!.expertWidth}`, `smerovač [${p.moe!.numExperts}, ${fmtInt(p.d)}] · šírka experta ${p.moe!.expertWidth}`), L(`${fmtB(a.ffnPerBlock + a.routerPerBlock)} stored · no shared expert`, `${fmtB(a.ffnPerBlock + a.routerPerBlock)} uložených · bez zdieľaného experta`)], { m: p.id, lv: 3, b, mod: 'moe' });
  const r2 = mk('residual2', 568, 44, '+', 'op', ['y = h + ffn'], undefined, 'circle');
  const xout = mk('x_out', 648, 44, L(`y (to ${b === p.layers - 1 ? 'final norm' : `block ${b + 1}`})`, `y (do ${b === p.layers - 1 ? 'záverečnej normy' : `bloku ${b + 1}`})`), 'act', [`[B, T, ${fmtInt(p.d)}]`]);
  const nodes = [xin, n1, at, r1, n2, ff, r2, xout];
  const sh = `[B,T,${p.d}]`;
  const edges: EdgeSpec[] = [link(xin, n1, sh), link(n1, at, sh), link(at, r1, sh), link(r1, n2, sh), link(n2, ff, sh), link(ff, r2, sh), link(r2, xout, sh)];
  const lx = 70;
  edges.push({ id: 'skip1', resid: true, pts: [[xin.x, xin.y + xin.h / 2], [lx, xin.y + xin.h / 2], [lx, r1.y + r1.h / 2], [r1.x, r1.y + r1.h / 2]], label: L('residual x', 'reziduálne x') });
  edges.push({ id: 'skip2', resid: true, pts: [[r1.x, r1.y + r1.h / 2 + 6], [lx + 24, r1.y + r1.h / 2 + 6], [lx + 24, r2.y + r2.h / 2], [r2.x, r2.y + r2.h / 2]], label: L('residual h', 'reziduálne h') });
  return { width: W, height: 712, nodes, edges, title: L(`Decoder block ${b}`, `Blok dekodéra ${b}`), description: L(`Pre-norm decoder block ${b}: RMSNorm, attention, residual add, RMSNorm, ${p.kind === 'dense' ? 'SwiGLU FFN' : 'MoE'}, residual add.`, `Blok dekodéra ${b} s normalizáciou na vstupe: RMSNorm, pozornosť, reziduálne sčítanie, RMSNorm, ${p.kind === 'dense' ? 'SwiGLU FFN' : 'MoE'}, reziduálne sčítanie.`) };
}

export function moduleGraph(p: CoreProfile, mod: string, b: number, selectedExpert?: number, lang: Lang = 'en'): GraphSpec {
  const L = lk(lang);
  const qw = p.hq * p.headDim, kvw = p.hkv * p.headDim;
  const nodes: NodeSpec[] = [];
  const edges: EdgeSpec[] = [];
  const n = (id: string, x: number, y: number, w: number, h: number, label: string, role: Role, sub?: string[], open?: Partial<ExplorerState>): NodeSpec => {
    const node = { id, x, y, w, h, label, role, sub, open };
    nodes.push(node);
    return node;
  };
  const T = (key: string, extra: Partial<ExplorerState> = {}): Partial<ExplorerState> => ({ m: p.id, lv: 2, b, t: key, ...extra });
  const pc = (s: number[]) => `${fmtInt(prod(s))} ${L('params', 'parametrov')}`;
  const toRes = L('to residual add', 'do reziduálneho sčítania');
  if (mod === 'attention') {
    const W = 760;
    const inp = n('x_in', 280, 14, 200, 40, L('normalized x', 'normalizované x'), 'act', [`[B, T, ${p.d}]`]);
    const q = n('tensor:q_proj', 40, 96, 200, 56, 'q_proj  W_Q', 'param', [`[${qw}, ${p.d}]`, pc([qw, p.d])], T('q_proj'));
    const k = n('tensor:k_proj', 280, 96, 200, 56, 'k_proj  W_K', 'param', [`[${kvw}, ${p.d}]`, pc([kvw, p.d])], T('k_proj'));
    const v = n('tensor:v_proj', 520, 96, 200, 56, 'v_proj  W_V', 'param', [`[${kvw}, ${p.d}]`, pc([kvw, p.d])], T('v_proj'));
    const rs = n('reshape', 40, 190, 680, 36, L(`reshape into heads: Q ${p.hq}×${p.headDim} · K ${p.hkv}×${p.headDim} · V ${p.hkv}×${p.headDim}`, `preskupenie do hláv: Q ${p.hq}×${p.headDim} · K ${p.hkv}×${p.headDim} · V ${p.hkv}×${p.headDim}`), 'op');
    n('tensor:q_norm', 40, 260, 200, 50, L('q_norm (per head)', 'q_norm (pre hlavu)'), 'norm', [L(`γ [${p.headDim}] shared by ${p.hq} heads`, `γ [${p.headDim}] spoločné pre ${p.hq} hláv`)], T('q_norm'));
    n('tensor:k_norm', 280, 260, 200, 50, L('k_norm (per head)', 'k_norm (pre hlavu)'), 'norm', [L(`γ [${p.headDim}] shared by ${p.hkv} heads`, `γ [${p.headDim}] spoločné pre ${p.hkv} hláv`)], T('k_norm'));
    n('vpass', 520, 260, 200, 50, L('V: no norm, no RoPE', 'V: bez normy, bez RoPE'), 'act', [`[B, ${p.hkv}, T, ${p.headDim}]`]);
    n('rope', 40, 344, 440, 44, L(`RoPE on Q and K (θ = ${fmtInt(p.ropeTheta)}, absolute positions)`, `RoPE na Q a K (θ = ${fmtInt(p.ropeTheta)}, absolútne pozície)`), 'op', [L('pairs (i, i + d_h/2) rotated · no weights', 'otáčajú sa dvojice (i, i + d_h/2) · bez váh')]);
    const sd = n('sdpa', 40, 424, 680, 60, L(`causal softmax(QKᵀ/√${p.headDim}) V · GQA: ${p.hq / p.hkv} query heads per K/V head`, `kauzálny softmax(QKᵀ/√${p.headDim}) V · GQA: ${p.hq / p.hkv} hlavy dopytov na hlavu K/V`), 'op', [L(`logical scores [B, ${p.hq}, Tq, Tk] · future masked`, `logické skóre [B, ${p.hq}, Tq, Tk] · budúcnosť zamaskovaná`), L('kernels may never store the full matrix', 'jadrá nemusia celú maticu nikdy uložiť')]);
    const cc = n('concat', 160, 520, 440, 36, L(`concatenate heads → [B, T, ${qw}]`, `spojenie hláv → [B, T, ${qw}]`), 'op');
    const o = n('tensor:o_proj', 260, 590, 240, 56, 'o_proj  W_O', 'param', [`[${p.d}, ${qw}]`, pc([p.d, qw])], T('o_proj'));
    const out = n('x_out', 280, 680, 200, 40, toRes, 'act', [`[B, T, ${p.d}]`]);
    edges.push(link(inp, q), link(inp, k), link(inp, v), link(q, rs), link(k, rs), link(v, rs));
    edges.push({ id: 'rs-qn', pts: [[140, 226], [140, 260]] }, { id: 'rs-kn', pts: [[380, 226], [380, 260]] }, { id: 'rs-v', pts: [[620, 226], [620, 260]] });
    edges.push({ id: 'qn-r', pts: [[140, 310], [140, 344]] }, { id: 'kn-r', pts: [[380, 310], [380, 344]] }, { id: 'r-s', pts: [[260, 388], [260, 424]], label: L('Q, K rotated', 'Q, K otočené') }, { id: 'v-s', pts: [[620, 310], [620, 424]], label: 'V' });
    edges.push(link(sd, cc), link(cc, o), link(o, out));
    return { width: W, height: 740, nodes, edges, title: L(`Attention · block ${b}`, `Pozornosť · blok ${b}`), description: L('Attention module: Q/K/V projections, head reshape, Q/K RMS norms, RoPE, causal grouped-query attention, concatenation, output projection.', 'Modul pozornosti: projekcie Q/K/V, preskupenie do hláv, RMS normy Q/K, RoPE, kauzálna pozornosť so skupinovými dopytmi, spojenie, výstupná projekcia.') };
  }
  if (mod === 'ffn') {
    const f = p.ffnWidth!;
    const inp = n('x_in', 240, 14, 240, 40, L('normalized h', 'normalizované h'), 'act', [`[B, T, ${p.d}]`]);
    const g = n('tensor:gate_proj', 40, 100, 280, 60, 'gate_proj  W_gate', 'param', [`[${fmtInt(f)}, ${fmtInt(p.d)}]`, pc([f, p.d])], T('gate_proj'));
    const u = n('tensor:up_proj', 400, 100, 280, 60, 'up_proj  W_up', 'param', [`[${fmtInt(f)}, ${fmtInt(p.d)}]`, pc([f, p.d])], T('up_proj'));
    const s = n('silu', 80, 204, 200, 44, 'SiLU(a) = a·σ(a)', 'op', [`g [B, T, ${fmtInt(f)}]`]);
    const m = n('mul', 260, 290, 200, 44, 'z = g ⊙ u', 'op', [L(`[B, T, ${fmtInt(f)}] gated channels`, `[B, T, ${fmtInt(f)}] hradlovaných kanálov`)]);
    const d = n('tensor:down_proj', 220, 376, 280, 60, 'down_proj  W_down', 'param', [`[${fmtInt(p.d)}, ${fmtInt(f)}]`, pc([p.d, f])], T('down_proj'));
    const out = n('x_out', 240, 476, 240, 40, toRes, 'act', [`[B, T, ${p.d}]`]);
    edges.push(link(inp, g), link(inp, u), link(g, s, L('pre-activation', 'pred-aktivácia')), link(s, m, 'g'), { id: 'u-m', pts: [[540, 160], [540, 312], [460, 312]], label: 'u' }, link(m, d, 'z'), link(d, out));
    return { width: 720, height: 540, nodes, edges, title: L(`SwiGLU FFN · block ${b}`, `SwiGLU FFN · blok ${b}`), description: L('SwiGLU feed-forward: gate and up projections, SiLU, elementwise gate, down projection.', 'Dopredná sieť SwiGLU: projekcie gate a up, SiLU, hradlo po prvkoch, projekcia down.') };
  }
  if (mod === 'moe') {
    const m = p.moe!;
    const inp = n('x_in', 260, 14, 240, 40, L('normalized h (one token)', 'normalizované h (jeden token)'), 'act', [`[${p.d}]`]);
    const r = n('tensor:router', 240, 86, 280, 56, 'router  mlp.gate.weight', 'param', [`[${m.numExperts}, ${p.d}] · ${fmtInt(m.numExperts * p.d)} params`], T('router'));
    const sm = n('softmax', 240, 170, 280, 36, L(`softmax over all ${m.numExperts} logits`, `softmax cez všetkých ${m.numExperts} logitov`), 'op');
    const tk = n('topk', 240, 230, 280, 36, L(`top-${m.topK} experts`, `top-${m.topK} expertov`), 'op');
    const rn = n('renorm', 240, 290, 280, 36, L(`renormalize: α sums to 1 (norm_topk_prob=${m.normTopkProb})`, `prepočet: α dáva spolu 1 (norm_topk_prob=${m.normTopkProb})`), 'op');
    const ex = n('experts', 30, 350, 700, 34 + Math.ceil(m.numExperts / 16) * 34 + 10, L(`${m.numExperts} experts (${m.topK} selected per token in each block; unselected still stored)`, `${m.numExperts} expertov (${m.topK} vybraných na token v každom bloku; nevybraní sú stále uložení)`), 'group');
    const cols = 16, ew = 40, eh = 28;
    for (let e = 0; e < m.numExperts; e++) {
      const rr = Math.floor(e / cols), cc = e % cols;
      nodes.push({ id: `expert:${e}`, x: 46 + cc * (ew + 2), y: 384 + rr * (eh + 6), w: ew, h: eh, label: String(e), role: e === selectedExpert ? 'act' : 'param', open: { m: p.id, lv: 2, b, t: 'expert_gate_proj', e } });
    }
    const ws = n('wsum', 240, ex.y + ex.h + 30, 280, 40, 'y = Σ α_e · Expert_e(x)', 'op');
    const out = n('x_out', 260, ws.y + 76, 240, 40, toRes, 'act', [`[${p.d}]`]);
    edges.push(link(inp, r), link(r, sm, L(`${m.numExperts} logits`, `${m.numExperts} logitov`)), link(sm, tk), link(tk, rn), link(rn, ex, L(`${m.topK} ids + α`, `${m.topK} čísel + α`)), link(ex, ws), link(ws, out));
    edges.push({ id: 'x-ex', pts: [[260, 34], [16, 34], [16, 360], [30, 360]], label: L('x to selected experts', 'x k vybraným expertom'), resid: false });
    return { width: 760, height: out.y + 60, nodes, edges, title: L(`MoE · block ${b}`, `MoE · blok ${b}`), description: L(`Mixture of experts: router, softmax, top-${m.topK}, renormalization, ${m.numExperts} expert FFNs, weighted sum.`, `Zmes expertov: smerovač, softmax, top-${m.topK}, prepočet, ${m.numExperts} expertných sietí FFN, vážený súčet.`) };
  }
  if (mod === 'attn_norm' || mod === 'ffn_norm' || mod === 'final_norm') {
    const key = mod === 'attn_norm' ? 'input_layernorm' : mod === 'ffn_norm' ? 'post_attention_layernorm' : 'norm';
    const inp = n('x_in', 200, 14, 280, 40, mod === 'ffn_norm' ? 'h = x + attn' : mod === 'final_norm' ? L('last block output', 'výstup posledného bloku') : L('block input x', 'vstup bloku x'), 'act', [`[B, T, ${p.d}]`]);
    const sq = n('sq', 200, 90, 280, 40, L('mean(x²) over features', 'mean(x²) cez vlastnosti'), 'op', [L('no mean subtraction', 'bez odčítania priemeru')]);
    const rq = n('rsqrt', 200, 160, 280, 40, L(`1/√(mean + ε), ε = ${p.rmsEps}`, `1/√(priemer + ε), ε = ${p.rmsEps}`), 'op');
    const sx = n('scalex', 200, 230, 280, 40, L('x · scale', 'x · faktor'), 'op');
    const gm = n(`tensor:${key}`, 200, 300, 280, 56, L('× γ (elementwise)', '× γ (po prvkoch)'), 'norm', [`γ [${fmtInt(p.d)}] · ${key}.weight`], { m: p.id, lv: 2, b: mod === 'final_norm' ? undefined : b, t: key });
    const out = n('x_out', 200, 392, 280, 40, L('normalized output', 'normalizovaný výstup'), 'act', [`[B, T, ${p.d}]`]);
    edges.push(link(inp, sq), link(sq, rq), link(rq, sx), { id: 'x-sx', pts: [[200, 34], [150, 34], [150, 250], [200, 250]], label: 'x' }, link(sx, gm), link(gm, out));
    return { width: 680, height: 450, nodes, edges, title: 'RMSNorm', description: L('RMSNorm: mean of squares, reciprocal square root, scale, multiply by learned gain.', 'RMSNorm: priemer druhých mocnín, prevrátená odmocnina, škálovanie, násobenie naučeným ziskom.') };
  }
  if (mod === 'embedding') {
    const ids = n('ids', 200, 14, 280, 40, L('token ids [B, T]', 'ID tokenov [B, T]'), 'act');
    const tb = n('tensor:embed_tokens', 140, 90, 400, 70, L('Embedding table E', 'Tabuľka embeddingov E'), 'param', [`[${fmtInt(p.vocabRows)}, ${fmtInt(p.d)}] · ${fmtB(prod([p.vocabRows, p.d]))} ${L('params', 'parametrov')}`, 'model.embed_tokens.weight'], { m: p.id, lv: 2, t: 'embed_tokens' });
    const lk = n('lookup', 200, 196, 280, 40, L('row lookup x_t = E[id_t]', 'vyhľadanie riadku x_t = E[id_t]'), 'op');
    const out = n('x_out', 200, 270, 280, 40, L(`X [B, T, ${p.d}] → block 0`, `X [B, T, ${p.d}] → blok 0`), 'act');
    edges.push(link(ids, tb), link(tb, lk), link(lk, out));
    return { width: 680, height: 330, nodes, edges, title: 'Embedding', description: L('Embedding lookup: one table row per token id.', 'Vyhľadanie embeddingu: jeden riadok tabuľky pre každé ID tokenu.') };
  }
  if (mod === 'lm_head') {
    const inp = n('x_in', 200, 14, 280, 40, L('final normalized h', 'záverečné normalizované h'), 'act', [`[B, T, ${p.d}]`]);
    const hd = n('tensor:lm_head', 140, 90, 400, 70, L('lm_head W_vocab (untied)', 'lm_head W_vocab (neviazaná)'), 'param', [`[${fmtInt(p.vocabRows)}, ${fmtInt(p.d)}] · ${fmtB(prod([p.vocabRows, p.d]))} ${L('params', 'parametrov')}`, 'lm_head.weight'], { m: p.id, lv: 2, t: 'lm_head' });
    const lg = n('logits', 200, 196, 280, 40, L(`logits [${fmtInt(p.vocabRows)}]`, `logity [${fmtInt(p.vocabRows)}]`), 'act');
    const sp = n('sampling', 200, 270, 280, 40, L('to sampling', 'na vzorkovanie'), 'op', [], { m: p.id, lv: 3, mod: 'sampling' });
    edges.push(link(inp, hd), link(hd, lg), link(lg, sp));
    return { width: 680, height: 330, nodes, edges, title: L('Vocabulary projection', 'Projekcia do slovníka'), description: L('Output head: one logit per vocabulary row.', 'Výstupná hlava: jeden logit pre každý riadok slovníka.') };
  }
  if (mod === 'sampling') {
    const lg = n('logits', 200, 14, 280, 40, L('logits (last position)', 'logity (posledná pozícia)'), 'act');
    const tp = n('temp', 200, 84, 280, 40, L('divide by temperature (0 = greedy)', 'delenie teplotou (0 = najpravdepodobnejší)'), 'op');
    const fl = n('filter', 200, 154, 280, 40, L('top-k → top-p → renormalize', 'top-k → top-p → prepočet'), 'op');
    const dr = n('draw', 200, 224, 280, 40, L('seeded random draw', 'náhodný výber s osivom'), 'op');
    const out = n('x_out', 200, 294, 280, 40, L('next token id → append', 'ID ďalšieho tokenu → pripojiť'), 'act');
    edges.push(link(lg, tp), link(tp, fl), link(fl, dr), link(dr, out));
    return { width: 680, height: 350, nodes, edges, title: L('Sampling', 'Vzorkovanie'), description: L('Decoding procedure: temperature, filtering, renormalization, random draw. No learned parameters.', 'Postup dekódovania: teplota, filtrovanie, prepočet, náhodný výber. Bez naučených parametrov.') };
  }
  // residual
  const xa = n('x_in', 60, 30, 240, 44, mod === 'residual1' ? L('x (block input)', 'x (vstup bloku)') : 'h', 'act', [`[B, T, ${p.d}]`]);
  const xb = n('branch', 380, 30, 240, 44, mod === 'residual1' ? L('attention output', 'výstup pozornosti') : L(`${p.kind === 'dense' ? 'FFN' : 'MoE'} output`, `výstup ${p.kind === 'dense' ? 'FFN' : 'MoE'}`), 'act', [`[B, T, ${p.d}]`]);
  const ad = n(mod, 290, 140, 100, 60, '+', 'op', [L('no weights', 'bez váh')]);
  const out = n('x_out', 220, 250, 240, 44, mod === 'residual1' ? 'h' : 'y', 'act', [`[B, T, ${p.d}]`]);
  edges.push({ id: 'a', pts: [bottom(xa), [340, 140]] }, { id: 'b', pts: [bottom(xb), [340, 140]] }, link(ad, out));
  return { width: 680, height: 320, nodes, edges, title: L('Residual addition', 'Reziduálne sčítanie'), description: L('Elementwise addition of two activation vectors. No learned parameters.', 'Sčítanie dvoch vektorov aktivácií po prvkoch. Bez naučených parametrov.') };
}
