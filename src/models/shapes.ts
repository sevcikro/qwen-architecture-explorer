import type { CoreProfile } from './profiles';

// One consistent convention: activations feature-axis-last, stored linear weights [out, in], Y = X W^T.

export type ModuleId =
  | 'embedding'
  | 'attn_norm'
  | 'attention'
  | 'ffn_norm'
  | 'ffn'
  | 'router'
  | 'experts'
  | 'final_norm'
  | 'lm_head';

export interface TensorFamily {
  key: string; // stable id, e.g. "q_proj"
  label: string;
  module: ModuleId;
  scope: 'model' | 'block' | 'expert';
  kind: 'matrix' | 'vector' | 'table';
  shape: number[];
  axes: string[];
  multiplicity: number; // how many such tensors exist in the whole model
  pathTemplate: string; // checkpoint tensor name with {layer} / {expert}
  symbol: string; // implementation symbol
  role: string;
  labelSk: string;
  roleSk: string;
}

type RawFamily = Omit<TensorFamily, 'labelSk' | 'roleSk'>;

/** Slovak label and plain-language role for every family (same keys as tensorFamilies). */
function slovak(p: CoreProfile): Record<string, [string, string]> {
  return {
    embed_tokens: ['Tabuľka vstupných embeddingov', 'Jeden naučený riadok s d číslami pre každé ID tokenu. Priechod modelom riadky vyhľadáva; celú tabuľku nenásobí.'],
    input_layernorm: ['Zisk γ normalizácie pred pozornosťou', 'Naučený zisk pre každú vlastnosť, použitý po RMS normalizácii na vstupe bloku.'],
    q_proj: ['Projekcia dopytov W_Q', 'Premení normalizovaný vektor reziduálneho prúdu na všetky hlavy dopytov.'],
    k_proj: ['Projekcia kľúčov W_K', 'Vytvorí hlavy kľúčov, s ktorými sa neskoršie dopyty porovnávajú.'],
    v_proj: ['Projekcia hodnôt W_V', 'Vytvorí hlavy hodnôt, ktoré pozornosť mieša.'],
    q_norm: ['Zisk normy hláv dopytov', `Jeden zisk pre každú súradnicu hlavy, spoločný pre všetkých ${p.hq} hláv dopytov (nenásobí sa počtom hláv).`],
    k_norm: ['Zisk normy hláv kľúčov', `Jeden zisk pre každú súradnicu hlavy, spoločný pre všetky ${p.hkv} hlavy kľúčov.`],
    o_proj: ['Výstupná projekcia W_O', 'Zmieša spojené výstupy hláv späť na šírku reziduálneho prúdu.'],
    post_attention_layernorm: ['Zisk γ normalizácie pred FFN', 'Naučený zisk RMSNorm pred vetvou FFN / MoE (nazýva sa „post-attention“, lebo nasleduje za prvým reziduálnym sčítaním).'],
    gate_proj: ['Hradlo FFN W_gate', 'Riadok j vypočíta pred-aktiváciu hradlového kanála j; na ňu sa použije SiLU.'],
    up_proj: ['FFN up W_up', 'Riadok j vypočíta lineárnu hodnotu „up“, ktorú hradlo násobí.'],
    down_proj: ['FFN down W_down', 'Stĺpec j prenesie hradlovaný kanál j späť do každej vlastnosti reziduálneho prúdu.'],
    router: ['Smerovač W_router', 'Naučené parametre smerovača: jeden riadok skóre pre každého experta. (V checkpointe sa volá „gate“; so SwiGLU hradlom nesúvisí.)'],
    expert_gate_proj: ['Hradlo experta W_gate[e]', 'Vlastná matica hradla SwiGLU experta e.'],
    expert_up_proj: ['Up experta W_up[e]', 'Vlastná matica up SwiGLU experta e.'],
    expert_down_proj: ['Down experta W_down[e]', 'Vlastná projekcia down experta e späť na šírku reziduálneho prúdu.'],
    norm: ['Zisk γ záverečnej normy', 'Zisk poslednej RMSNorm pred projekciou do slovníka.'],
    lm_head: ['Výstupná hlava W_vocab', p.tiedEmbeddings ? 'Zviazaná so vstupným embeddingom.' : 'Neviazaná: vlastná matica, oddelená od vstupnej tabuľky embeddingov. Vytvorí jeden logit pre každý riadok slovníka.'],
  };
}


export const prod = (s: number[]) => s.reduce((a, b) => a * b, 1);

export function tensorFamilies(p: CoreProfile): TensorFamily[] {
  const qw = p.hq * p.headDim;
  const kvw = p.hkv * p.headDim;
  const L = p.layers;
  const fam: RawFamily[] = [
    { key: 'embed_tokens', label: 'Input embedding table', module: 'embedding', scope: 'model', kind: 'table', shape: [p.vocabRows, p.d], axes: ['row = token id', 'column = residual feature'], multiplicity: 1, pathTemplate: 'model.embed_tokens.weight', symbol: 'nn.Embedding', role: 'One learned row of d numbers per token id. A forward pass looks up rows; it does not multiply the whole table.' },
    { key: 'input_layernorm', label: 'Attention pre-norm scale γ', module: 'attn_norm', scope: 'block', kind: 'vector', shape: [p.d], axes: ['feature'], multiplicity: L, pathTemplate: 'model.layers.{layer}.input_layernorm.weight', symbol: 'Qwen3RMSNorm.weight', role: 'Learned per-feature gain applied after RMS normalization at the block entrance.' },
    { key: 'q_proj', label: 'Query projection W_Q', module: 'attention', scope: 'block', kind: 'matrix', shape: [qw, p.d], axes: [`out = Hq·dh = ${qw}`, `in = d = ${p.d}`], multiplicity: L, pathTemplate: 'model.layers.{layer}.self_attn.q_proj.weight', symbol: 'self_attn.q_proj (nn.Linear, bias=False)', role: 'Turns the normalized residual vector into all query heads.' },
    { key: 'k_proj', label: 'Key projection W_K', module: 'attention', scope: 'block', kind: 'matrix', shape: [kvw, p.d], axes: [`out = Hkv·dh = ${kvw}`, `in = d = ${p.d}`], multiplicity: L, pathTemplate: 'model.layers.{layer}.self_attn.k_proj.weight', symbol: 'self_attn.k_proj (nn.Linear, bias=False)', role: 'Produces the key heads that later queries compare against.' },
    { key: 'v_proj', label: 'Value projection W_V', module: 'attention', scope: 'block', kind: 'matrix', shape: [kvw, p.d], axes: [`out = Hkv·dh = ${kvw}`, `in = d = ${p.d}`], multiplicity: L, pathTemplate: 'model.layers.{layer}.self_attn.v_proj.weight', symbol: 'self_attn.v_proj (nn.Linear, bias=False)', role: 'Produces the value heads that attention mixes.' },
    { key: 'q_norm', label: 'Query head-norm scale', module: 'attention', scope: 'block', kind: 'vector', shape: [p.headDim], axes: ['head coordinate'], multiplicity: L, pathTemplate: 'model.layers.{layer}.self_attn.q_norm.weight', symbol: 'self_attn.q_norm (RMSNorm over head_dim)', role: `One gain per head coordinate, shared by all ${p.hq} query heads (not multiplied by the head count).` },
    { key: 'k_norm', label: 'Key head-norm scale', module: 'attention', scope: 'block', kind: 'vector', shape: [p.headDim], axes: ['head coordinate'], multiplicity: L, pathTemplate: 'model.layers.{layer}.self_attn.k_norm.weight', symbol: 'self_attn.k_norm (RMSNorm over head_dim)', role: `One gain per head coordinate, shared by all ${p.hkv} key heads.` },
    { key: 'o_proj', label: 'Output projection W_O', module: 'attention', scope: 'block', kind: 'matrix', shape: [p.d, qw], axes: [`out = d = ${p.d}`, `in = Hq·dh = ${qw}`], multiplicity: L, pathTemplate: 'model.layers.{layer}.self_attn.o_proj.weight', symbol: 'self_attn.o_proj (nn.Linear, bias=False)', role: 'Mixes the concatenated head outputs back to residual width.' },
    { key: 'post_attention_layernorm', label: 'FFN pre-norm scale γ', module: 'ffn_norm', scope: 'block', kind: 'vector', shape: [p.d], axes: ['feature'], multiplicity: L, pathTemplate: 'model.layers.{layer}.post_attention_layernorm.weight', symbol: 'post_attention_layernorm (RMSNorm)', role: 'Learned gain of the RMSNorm before the FFN / MoE branch (named "post-attention" because it follows the first residual add).' },
  ];
  if (p.kind === 'dense') {
    const f = p.ffnWidth!;
    fam.push(
      { key: 'gate_proj', label: 'FFN gate W_gate', module: 'ffn', scope: 'block', kind: 'matrix', shape: [f, p.d], axes: [`out = f = ${f}`, `in = d = ${p.d}`], multiplicity: L, pathTemplate: 'model.layers.{layer}.mlp.gate_proj.weight', symbol: 'mlp.gate_proj', role: 'Row j computes the pre-activation of gate channel j; SiLU is applied to it.' },
      { key: 'up_proj', label: 'FFN up W_up', module: 'ffn', scope: 'block', kind: 'matrix', shape: [f, p.d], axes: [`out = f = ${f}`, `in = d = ${p.d}`], multiplicity: L, pathTemplate: 'model.layers.{layer}.mlp.up_proj.weight', symbol: 'mlp.up_proj', role: 'Row j computes the linear "up" value that the gate multiplies.' },
      { key: 'down_proj', label: 'FFN down W_down', module: 'ffn', scope: 'block', kind: 'matrix', shape: [p.d, f], axes: [`out = d = ${p.d}`, `in = f = ${f}`], multiplicity: L, pathTemplate: 'model.layers.{layer}.mlp.down_proj.weight', symbol: 'mlp.down_proj', role: 'Column j carries gated channel j back into every residual feature.' },
    );
  } else {
    const m = p.moe!;
    fam.push(
      { key: 'router', label: 'Router W_router', module: 'router', scope: 'block', kind: 'matrix', shape: [m.numExperts, p.d], axes: [`out = experts = ${m.numExperts}`, `in = d = ${p.d}`], multiplicity: L, pathTemplate: 'model.layers.{layer}.mlp.gate.weight', symbol: 'mlp.gate (TopKRouter.weight)', role: 'Learned router parameters: one score row per expert. (Called "gate" in the checkpoint; unrelated to the SwiGLU gate.)' },
      { key: 'expert_gate_proj', label: 'Expert gate W_gate[e]', module: 'experts', scope: 'expert', kind: 'matrix', shape: [m.expertWidth, p.d], axes: [`out = expert width = ${m.expertWidth}`, `in = d = ${p.d}`], multiplicity: L * m.numExperts, pathTemplate: 'model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight', symbol: 'experts.gate_up_proj[e, :I, :] at runtime', role: 'Expert e\'s own SwiGLU gate matrix.' },
      { key: 'expert_up_proj', label: 'Expert up W_up[e]', module: 'experts', scope: 'expert', kind: 'matrix', shape: [m.expertWidth, p.d], axes: [`out = expert width = ${m.expertWidth}`, `in = d = ${p.d}`], multiplicity: L * m.numExperts, pathTemplate: 'model.layers.{layer}.mlp.experts.{expert}.up_proj.weight', symbol: 'experts.gate_up_proj[e, I:, :] at runtime', role: 'Expert e\'s own SwiGLU up matrix.' },
      { key: 'expert_down_proj', label: 'Expert down W_down[e]', module: 'experts', scope: 'expert', kind: 'matrix', shape: [p.d, m.expertWidth], axes: [`out = d = ${p.d}`, `in = expert width = ${m.expertWidth}`], multiplicity: L * m.numExperts, pathTemplate: 'model.layers.{layer}.mlp.experts.{expert}.down_proj.weight', symbol: 'experts.down_proj[e] at runtime', role: 'Expert e\'s own down projection back to residual width.' },
    );
  }
  fam.push(
    { key: 'norm', label: 'Final norm scale γ', module: 'final_norm', scope: 'model', kind: 'vector', shape: [p.d], axes: ['feature'], multiplicity: 1, pathTemplate: 'model.norm.weight', symbol: 'model.norm (RMSNorm)', role: 'Gain of the last RMSNorm before the vocabulary projection.' },
    { key: 'lm_head', label: 'Output head W_vocab', module: 'lm_head', scope: 'model', kind: 'matrix', shape: [p.vocabRows, p.d], axes: [`out = vocabulary rows = ${p.vocabRows}`, `in = d = ${p.d}`], multiplicity: 1, pathTemplate: 'lm_head.weight', symbol: 'lm_head (nn.Linear, bias=False)', role: p.tiedEmbeddings ? 'Tied to the input embedding.' : 'Untied: its own matrix, separate from the input embedding table. Produces one logit per vocabulary row.' },
  );
  const sk = slovak(p);
  return fam.map((f) => ({ ...f, labelSk: sk[f.key]?.[0] ?? f.label, roleSk: sk[f.key]?.[1] ?? f.role }));
}

export const familyParams = (f: TensorFamily) => prod(f.shape) * f.multiplicity;

export function tensorName(f: TensorFamily, layer = 0, expert = 0): string {
  return f.pathTemplate.replace('{layer}', String(layer)).replace('{expert}', String(expert));
}

export interface Accounting {
  embedding: number;
  head: number;
  attentionPerBlock: number;
  qkvoPerBlock: { q: number; k: number; v: number; o: number };
  oneExpert: number | null;
  ffnPerBlock: number;
  routerPerBlock: number;
  normsPerBlock: number;
  perBlock: number;
  finalNorm: number;
  total: number;
  nonEmbedding: number; // model-card convention: total minus input embedding AND output head
  selectedExpertsAllLayers: number | null;
  activeIncludingFullEmbedding: number | null;
  activeEmbeddingRowOnly: number | null;
}

/** All counts come from the same shape table, so graph labels, catalog and treemap cannot drift. */
export function accounting(p: CoreProfile): Accounting {
  const fams = tensorFamilies(p);
  const get = (k: string) => fams.find((f) => f.key === k);
  const one = (k: string) => (get(k) ? prod(get(k)!.shape) : 0);
  const q = one('q_proj'), k = one('k_proj'), v = one('v_proj'), o = one('o_proj');
  const attentionPerBlock = q + k + v + o;
  const normsPerBlock = one('input_layernorm') + one('post_attention_layernorm') + one('q_norm') + one('k_norm');
  let ffnPerBlock: number;
  let oneExpert: number | null = null;
  let routerPerBlock = 0;
  if (p.kind === 'dense') {
    ffnPerBlock = one('gate_proj') + one('up_proj') + one('down_proj');
  } else {
    oneExpert = one('expert_gate_proj') + one('expert_up_proj') + one('expert_down_proj');
    ffnPerBlock = oneExpert * p.moe!.numExperts;
    routerPerBlock = one('router');
  }
  const perBlock = attentionPerBlock + normsPerBlock + ffnPerBlock + routerPerBlock;
  const embedding = one('embed_tokens');
  const head = p.tiedEmbeddings ? 0 : one('lm_head');
  const finalNorm = one('norm');
  const total = embedding + head + finalNorm + perBlock * p.layers;
  const totalFromFamilies = fams.reduce((s, f) => s + familyParams(f), 0) - (p.tiedEmbeddings ? one('lm_head') : 0);
  if (totalFromFamilies !== total) throw new Error('internal accounting mismatch');
  let selectedExpertsAllLayers: number | null = null;
  let activeIncludingFullEmbedding: number | null = null;
  let activeEmbeddingRowOnly: number | null = null;
  if (p.kind === 'moe') {
    const m = p.moe!;
    selectedExpertsAllLayers = p.layers * m.topK * oneExpert!;
    activeIncludingFullEmbedding = total - p.layers * (m.numExperts - m.topK) * oneExpert!;
    activeEmbeddingRowOnly = activeIncludingFullEmbedding - embedding + p.d;
  }
  return {
    embedding, head, attentionPerBlock, qkvoPerBlock: { q, k, v, o }, oneExpert, ffnPerBlock, routerPerBlock, normsPerBlock, perBlock, finalNorm, total,
    nonEmbedding: total - embedding - head,
    selectedExpertsAllLayers, activeIncludingFullEmbedding, activeEmbeddingRowOnly,
  };
}

/** Logical-to-physical expert mapping for the runtime packed layout used by current transformers. */
export function packedExpertMapping(p: CoreProfile, expert: number) {
  if (p.kind !== 'moe') return null;
  const I = p.moe!.expertWidth, H = p.d, E = p.moe!.numExperts;
  return {
    checkpoint: {
      gate: `model.layers.{layer}.mlp.experts.${expert}.gate_proj.weight  [${I}, ${H}]`,
      up: `model.layers.{layer}.mlp.experts.${expert}.up_proj.weight  [${I}, ${H}]`,
      down: `model.layers.{layer}.mlp.experts.${expert}.down_proj.weight  [${H}, ${I}]`,
    },
    runtime: {
      gate: `mlp.experts.gate_up_proj[${expert}, 0:${I}, :]  of [${E}, ${2 * I}, ${H}]`,
      up: `mlp.experts.gate_up_proj[${expert}, ${I}:${2 * I}, :]  of [${E}, ${2 * I}, ${H}]`,
      down: `mlp.experts.down_proj[${expert}]  of [${E}, ${H}, ${I}]`,
    },
    note: 'The packed runtime tensors are a re-layout of the same scalars (MergeModulelist over experts, then Concatenate gate and up along dim 1). Parameter totals count the scalars once.',
  };
}

/** Matrix-multiply operation estimate per token (2 ops per multiply-accumulate), excluding attention-score work. */
export function matmulOpsPerToken(p: CoreProfile): number {
  const a = accounting(p);
  const perBlockActive = a.attentionPerBlock + (p.kind === 'dense' ? a.ffnPerBlock : a.routerPerBlock + p.moe!.topK * a.oneExpert!);
  return 2 * (perBlockActive * p.layers + a.head);
}

/** Attention score + weighted-sum ops for one new token attending to T earlier positions (logical, unfused). */
export function attentionOpsPerToken(p: CoreProfile, contextT: number): number {
  return 2 * 2 * p.layers * p.hq * p.headDim * contextT;
}
