import type { EvidenceClass } from '../evidence/types';
import type { CoreProfile } from '../models/profiles';
import { accounting, prod, tensorFamilies, tensorName, type TensorFamily } from '../models/shapes';
import { fmtInt, shapeStr } from '../components/ui';
import { explorerHref, type ExplorerState } from '../state/route';

export type Role = 'param' | 'norm' | 'act' | 'state' | 'op' | 'group';

export interface InspectorInfo {
  id: string;
  title: string;
  role: Role;
  what: string;
  where: string;
  inputs: string;
  outputs: string;
  params: { count: number; formula: string } | null;
  persistence: string;
  equation?: string;
  checkpoint?: string[];
  implementation?: string;
  evidence: { kind: EvidenceClass; note: string }[];
  sources: string[];
  below: { label: string; href: string }[];
  above?: { label: string; href: string };
  notes?: string[];
}

const V = (p: CoreProfile) => (p.status === 'official' ? 'verified' : 'simulation') as EvidenceClass;
const evStruct = (p: CoreProfile): { kind: EvidenceClass; note: string } =>
  p.status === 'official'
    ? { kind: 'verified', note: `Structure and shapes from ${p.repo} config.json + checkpoint headers @ ${p.revision!.slice(0, 12)} and the pinned Transformers implementation.` }
    : { kind: 'simulation', note: 'Toy simulator: Qwen3 structure at reduced size with seeded synthetic weights. Not Qwen dimensions or values.' };
const srcs = (p: CoreProfile, extra: string[] = []) => [...(p.status === 'official' ? p.sources : ['tf-qwen3']), ...extra];

export function famByKey(p: CoreProfile, key: string): TensorFamily | undefined {
  return tensorFamilies(p).find((f) => f.key === key);
}

function blockLabel(p: CoreProfile, b?: number) {
  return b === undefined ? `every decoder block (×${p.layers})` : `decoder block ${b} of ${p.layers} (code index ${b}, human count ${b + 1})`;
}

export function describeTensor(p: CoreProfile, f: TensorFamily, s: Partial<ExplorerState>): InspectorInfo {
  const b = s.b ?? 0;
  const name = tensorName(f, b, s.e ?? 0);
  const n = prod(f.shape);
  const isMat = f.shape.length === 2;
  const eq: Record<string, string> = {
    embed_tokens: 'x_t = E[\\text{id}_t,\\,:]',
    q_proj: 'Q = X\\,W_Q^{\\top}',
    k_proj: 'K = X\\,W_K^{\\top}',
    v_proj: 'V = X\\,W_V^{\\top}',
    o_proj: 'Y = Z\\,W_O^{\\top}',
    q_norm: '\\hat q_h = \\gamma_q \\odot q_h / \\sqrt{\\operatorname{mean}(q_h^2)+\\epsilon}',
    k_norm: '\\hat k_h = \\gamma_k \\odot k_h / \\sqrt{\\operatorname{mean}(k_h^2)+\\epsilon}',
    input_layernorm: '\\operatorname{RMSNorm}(x)=\\gamma \\odot x/\\sqrt{\\operatorname{mean}(x^2)+\\epsilon}',
    post_attention_layernorm: '\\operatorname{RMSNorm}(h)=\\gamma \\odot h/\\sqrt{\\operatorname{mean}(h^2)+\\epsilon}',
    gate_proj: 'g = \\operatorname{SiLU}(X\\,W_{gate}^{\\top})',
    up_proj: 'u = X\\,W_{up}^{\\top}',
    down_proj: 'Y = (g \\odot u)\\,W_{down}^{\\top}',
    router: 'r = x\\,W_{router}^{\\top},\\; p = \\operatorname{softmax}(r)',
    expert_gate_proj: 'g_e = \\operatorname{SiLU}(x\\,W_{gate,e}^{\\top})',
    expert_up_proj: 'u_e = x\\,W_{up,e}^{\\top}',
    expert_down_proj: '\\operatorname{Expert}_e(x) = (g_e \\odot u_e)\\,W_{down,e}^{\\top}',
    norm: '\\operatorname{RMSNorm}(h_L)=\\gamma \\odot h_L/\\sqrt{\\operatorname{mean}(h_L^2)+\\epsilon}',
    lm_head: '\\text{logits} = \\hat h\\,W_{vocab}^{\\top}',
  };
  const inOut: Record<string, [string, string]> = {
    embed_tokens: ['token ids [B, T] (integers)', `X [B, T, ${p.d}]`],
    q_proj: [`normalized X [B, T, ${p.d}]`, `Q [B, T, ${p.hq * p.headDim}] → reshaped [B, ${p.hq}, T, ${p.headDim}]`],
    k_proj: [`normalized X [B, T, ${p.d}]`, `K [B, T, ${p.hkv * p.headDim}] → reshaped [B, ${p.hkv}, T, ${p.headDim}]`],
    v_proj: [`normalized X [B, T, ${p.d}]`, `V [B, T, ${p.hkv * p.headDim}] → reshaped [B, ${p.hkv}, T, ${p.headDim}]`],
    o_proj: [`concatenated head outputs Z [B, T, ${p.hq * p.headDim}]`, `attention branch output [B, T, ${p.d}]`],
    q_norm: [`each query head vector [${p.headDim}]`, `normalized query head [${p.headDim}]`],
    k_norm: [`each key head vector [${p.headDim}]`, `normalized key head [${p.headDim}]`],
    input_layernorm: [`block input x [B, T, ${p.d}]`, `normalized x [B, T, ${p.d}]`],
    post_attention_layernorm: [`h = x + attention [B, T, ${p.d}]`, `normalized h [B, T, ${p.d}]`],
    norm: [`last block output [B, T, ${p.d}]`, `normalized [B, T, ${p.d}]`],
    lm_head: [`normalized final hidden state [B, T, ${p.d}]`, `logits [B, T, ${fmtInt(p.vocabRows)}]`],
    router: [`normalized h, one token [${p.d}]`, `${p.moe?.numExperts} router logits`],
  };
  const io = inOut[f.key] ?? (f.key.startsWith('expert_') ? (f.key === 'expert_down_proj' ? [`gated channels [${p.moe!.expertWidth}]`, `expert output [${p.d}]`] : [`normalized h, one routed token [${p.d}]`, `[${p.moe!.expertWidth}] channels`]) : f.key === 'down_proj' ? [`gated channels z [B, T, ${p.ffnWidth}]`, `FFN output [B, T, ${p.d}]`] : [`normalized h [B, T, ${p.d}]`, `[B, T, ${p.ffnWidth}]`]);
  const notes: string[] = [];
  if (f.key === 'embed_tokens' && p.tokenizerVocab) notes.push(`${fmtInt(p.vocabRows)} rows is the configured table size. The tokenizer defines ${fmtInt(p.tokenizerVocab)} token ids (151,643 regular + 26 added); the remaining ${fmtInt(p.vocabRows - p.tokenizerVocab)} rows are padding that no token id maps to.`);
  if (f.key === 'router') notes.push('Naming trap: the checkpoint calls this tensor "mlp.gate". It is the router, not the SwiGLU gate_proj inside each expert.');
  if (f.key.startsWith('expert_')) notes.push('Checkpoint stores each expert as its own tensor. Current Transformers packs all experts of a layer into gate_up_proj [E, 2I, H] and down_proj [E, H, I] at load time; the parameter count is the same scalars counted once.');
  if (f.key === 'q_norm' || f.key === 'k_norm') notes.push(`One ${p.headDim}-value gain vector shared by every ${f.key === 'q_norm' ? 'query' : 'key'} head in the block: ${p.headDim} parameters, not ${p.headDim} × heads.`);
  if (f.key === 'lm_head') notes.push('Untied: this is a separate matrix from the input embedding table, so the two together hold 2 × vocab × d parameters.');
  if (p.kind === 'moe' && f.key === 'q_proj') notes.push(`Here d = ${p.d} but Hq·dh = ${p.hq * p.headDim}: the query projection expands width, and W_O brings it back to ${p.d}. head_dim is a config field (${p.headDim}), not d / Hq (= ${p.d / p.hq}).`);
  const level2: ExplorerState = { m: p.id, lv: 2, b: f.scope === 'model' ? undefined : b, t: f.key, e: f.scope === 'expert' ? s.e ?? 0 : undefined };
  return {
    id: `tensor:${f.key}`,
    title: f.label,
    role: f.kind === 'vector' ? 'norm' : 'param',
    what: f.role,
    where: `${f.scope === 'model' ? 'Model-level component (one copy)' : f.scope === 'expert' ? `Expert ${s.e ?? 0} of ${p.moe!.numExperts} in ${blockLabel(p, s.b)}` : `Inside ${blockLabel(p, s.b)}`}. Logical shape ${shapeStr(f.shape)} (${f.axes.join(' × ')}).`,
    inputs: io[0],
    outputs: io[1],
    params: { count: n, formula: `${f.shape.map(fmtInt).join(' × ')} = ${fmtInt(n)}${f.multiplicity > 1 ? `  (× ${fmtInt(f.multiplicity)} such tensors = ${fmtInt(n * f.multiplicity)} in the whole model)` : ''}` },
    persistence: `Persistent learned weights (${p.status === 'official' ? `stored as ${p.dtype} in the checkpoint` : 'toy float64 values'}). The ${isMat ? 'products computed with it' : 'normalized values it scales'} are runtime activations that change with every input.`,
    equation: eq[f.key],
    checkpoint: [name],
    implementation: f.symbol,
    evidence: [evStruct(p), { kind: 'derived', note: `Parameter count = product of the logical shape. ${isMat ? 'PyTorch nn.Linear stores [out, in] and computes X W^T.' : ''}` }],
    sources: srcs(p, isMat ? ['torch-linear'] : ['rmsnorm']),
    below: [
      { label: 'Open this tensor (Level 2: matrix tile)', href: explorerHref(level2) },
      { label: isMat ? 'Open output row 0 (Level 1: unit)' : 'Open element 0 (Level 0: scalar)', href: explorerHref({ ...level2, lv: isMat ? 1 : 0, r: 0 }) },
    ],
    notes,
  };
}

export function describeElement(p: CoreProfile, id: string, s: Partial<ExplorerState>): InspectorInfo {
  const a = accounting(p);
  const b = s.b;
  const base = { evidence: [evStruct(p)], sources: srcs(p), below: [] as { label: string; href: string }[] };
  const blockHref = (bb: number, extra: Partial<ExplorerState> = {}) => explorerHref({ m: p.id, lv: 4, b: bb, ...extra });
  const modHref = (mod: string, extra: Partial<ExplorerState> = {}) => explorerHref({ m: p.id, lv: 3, b: b ?? 0, mod, ...extra });
  if (id.startsWith('tensor:')) {
    const f = famByKey(p, id.slice(7));
    if (f) return describeTensor(p, f, s);
  }
  if (id.startsWith('block:')) {
    const bb = Number(id.slice(6));
    return {
      ...base,
      id,
      title: `Decoder block ${bb}`,
      role: 'group',
      what: `One of ${p.layers} stacked decoder blocks. Each block owns its own separate matrices and norm gains; block ${bb}'s weights are different numbers from block ${bb + 1}'s, even though the shapes are identical. Within the block the same weights are reused for every token position.`,
      where: `${blockLabel(p, bb)}. Checkpoint prefix model.layers.${bb}.*`,
      inputs: `x [B, T, ${p.d}] from ${bb === 0 ? 'the embedding lookup' : `block ${bb - 1}`}`,
      outputs: `y [B, T, ${p.d}] to ${bb === p.layers - 1 ? 'the final RMSNorm' : `block ${bb + 1}`}`,
      params: { count: a.perBlock, formula: `attention ${fmtInt(a.attentionPerBlock)} + norms ${fmtInt(a.normsPerBlock)} + ${p.kind === 'dense' ? `FFN ${fmtInt(a.ffnPerBlock)}` : `experts ${fmtInt(a.ffnPerBlock)} + router ${fmtInt(a.routerPerBlock)}`} = ${fmtInt(a.perBlock)}` },
      persistence: 'Weights are persistent; the vectors flowing through are activations recomputed for each input. During generation the block also keeps a K/V cache (runtime state, not trained parameters).',
      equation: `h = x + \\operatorname{Attn}(\\operatorname{RMSNorm}_{attn}(x)),\\quad y = h + \\operatorname{${p.kind === 'dense' ? 'FFN' : 'MoE'}}(\\operatorname{RMSNorm}_{ffn}(h))`,
      checkpoint: [`model.layers.${bb}.*`],
      implementation: p.kind === 'dense' ? 'Qwen3DecoderLayer' : 'Qwen3MoeDecoderLayer',
      below: [{ label: `Open block ${bb} (Level 4)`, href: blockHref(bb) }],
      above: { label: 'Back to the full model', href: explorerHref({ m: p.id, lv: 5, b: bb }) },
    };
  }
  const common: Record<string, Omit<InspectorInfo, 'evidence' | 'sources' | 'below'> & { below?: InspectorInfo['below']; sources?: string[]; evidence?: InspectorInfo['evidence'] }> = {
    model: {
      id, title: p.name, role: 'group',
      what: p.status === 'official' ? `${p.kind === 'dense' ? 'Dense' : 'Mixture-of-experts'} decoder-only language model. All of its learned parameters are numbers stored in the tensors listed in the catalog; nothing else in the diagram holds learned values.` : 'The toy simulator: a tiny model with the same operations and tensor names as Qwen3, filled with seeded synthetic weights.',
      where: p.repo ? `${p.repo} @ ${p.revision!.slice(0, 12)}` : 'Runs entirely in your browser.',
      inputs: 'text → chat template → token ids',
      outputs: `logits over ${fmtInt(p.vocabRows)} vocabulary rows → sampled next token`,
      params: { count: a.total, formula: `embedding ${fmtInt(a.embedding)} + ${p.layers} × ${fmtInt(a.perBlock)} + final norm ${fmtInt(a.finalNorm)} + head ${fmtInt(a.head)} = ${fmtInt(a.total)}` },
      persistence: 'Learned parameters are fixed during inference. Activations, the K/V cache and sampled tokens are runtime data.',
    },
    tokenizer: {
      id, title: 'Chat template + tokenizer', role: 'op',
      what: 'Turns messages into text with role markers (<|im_start|>user …) and then into integer token ids using byte-level BPE. It is a preprocessing program, not part of the neural network, and has no trainable weights in the checkpoint.',
      where: 'Before the model. Files tokenizer.json / tokenizer_config.json in the repository.',
      inputs: 'messages / text', outputs: 'token ids [B, T]', params: null,
      persistence: 'Fixed vocabulary and merge rules (data files, not parameters).',
      sources: ['q8-tokenizer', 'qwen3-report'],
      below: [{ label: 'See real Qwen3 tokenizer outputs', href: '#/pipeline' }],
    },
    stack: {
      id, title: `Decoder stack (${p.layers} blocks)`, role: 'group',
      what: `${p.layers} blocks applied one after another. The output of block i is the input of block i+1, so blocks are a sequential dependency; within a block all prompt positions can be computed in parallel during prefill.`,
      where: 'Between the embedding lookup and the final norm.', inputs: `X [B, T, ${p.d}]`, outputs: `[B, T, ${p.d}]`,
      params: { count: a.perBlock * p.layers, formula: `${p.layers} × ${fmtInt(a.perBlock)}` },
      persistence: 'Distinct weights per block; reused across token positions.',
    },
    sampling: {
      id, title: 'Next-token selection (sampling)', role: 'op',
      what: 'A decoding procedure, not a neural-network layer: temperature, top-k and top-p filtering, renormalization and a random draw (or greedy argmax) pick one token id, which is appended and fed back in.',
      where: 'After the vocabulary projection.', inputs: `logits [${fmtInt(p.vocabRows)}] for the last position`, outputs: 'one token id', params: null,
      persistence: 'No learned parameters. Settings (e.g. temperature 0.6, top-k 20, top-p 0.95 by default for these models) live in generation_config.json.',
      sources: ['q8-gen', 'q8-card'],
      below: [{ label: 'Open the sampling lab', href: '#/labs/sampling' }],
    },
    attention: {
      id, title: 'Self-attention (GQA, QK-norm, RoPE)', role: 'param',
      what: `Lets each position read from earlier positions. ${p.hq} query heads compare against ${p.hkv} key heads (${p.hq / p.hkv} query heads share each K/V head), then mix value vectors with the resulting causal attention weights.`,
      where: `${blockLabel(p, b)}, after the first RMSNorm.`,
      inputs: `normalized x [B, T, ${p.d}]`, outputs: `[B, T, ${p.d}] added to the residual stream`,
      params: { count: a.attentionPerBlock + 2 * p.headDim, formula: `W_Q ${fmtInt(a.qkvoPerBlock.q)} + W_K ${fmtInt(a.qkvoPerBlock.k)} + W_V ${fmtInt(a.qkvoPerBlock.v)} + W_O ${fmtInt(a.qkvoPerBlock.o)} + q/k norm gains 2 × ${p.headDim}` },
      persistence: 'W_Q/W_K/W_V/W_O and the q/k norm gains are learned. Attention weights (softmax probabilities) are computed activations, recomputed per input. Keys/values of past tokens are cached at runtime.',
      equation: 'A = \\operatorname{softmax}\\!\\left(\\tfrac{QK^{\\top}}{\\sqrt{d_h}} + M_{causal}\\right),\\; Z = AV',
      implementation: p.kind === 'dense' ? 'Qwen3Attention' : 'Qwen3MoeAttention',
      sources: [...srcs(p), 'vaswani', 'gqa', 'rope'],
      below: [{ label: 'Open the attention module (Level 3)', href: modHref('attention') }, { label: 'Attention lab (numbers)', href: '#/labs/attention' }],
    },
    attn_norm: {
      id, title: 'RMSNorm before attention', role: 'norm',
      what: 'Rescales each token vector to unit root-mean-square, then multiplies elementwise by a learned gain γ. Unlike LayerNorm it does not subtract the mean.',
      where: `${blockLabel(p, b)} entrance (input_layernorm).`, inputs: `x [B, T, ${p.d}]`, outputs: `[B, T, ${p.d}]`,
      params: { count: p.d, formula: `γ has ${fmtInt(p.d)} values` },
      persistence: 'γ is learned; the RMS of each token is computed at runtime. Transformers computes the normalization in float32 even when weights are stored in BF16.',
      equation: '\\operatorname{RMSNorm}(x)=\\gamma\\odot\\frac{x}{\\sqrt{\\tfrac1d\\sum_i x_i^2+\\epsilon}}', sources: [...srcs(p), 'rmsnorm'],
    },
    ffn_norm: {
      id, title: 'RMSNorm before the FFN / MoE', role: 'norm',
      what: 'Same operation as the first norm with its own learned γ (post_attention_layernorm).',
      where: `${blockLabel(p, b)}, after the first residual addition.`, inputs: `h [B, T, ${p.d}]`, outputs: `[B, T, ${p.d}]`,
      params: { count: p.d, formula: `γ has ${fmtInt(p.d)} values` }, persistence: 'γ learned; normalization computed at runtime.',
      equation: '\\operatorname{RMSNorm}(h)=\\gamma\\odot h/\\sqrt{\\operatorname{mean}(h^2)+\\epsilon}', sources: [...srcs(p), 'rmsnorm'],
    },
    residual1: {
      id, title: 'Residual addition after attention', role: 'op',
      what: 'Adds the attention branch output to the unchanged block input: h = x + attention(...). The addition itself has no learned weights. The values being added are both activations.',
      where: blockLabel(p, b), inputs: `x [B, T, ${p.d}] and attention output [B, T, ${p.d}]`, outputs: `h [B, T, ${p.d}]`, params: { count: 0, formula: 'no parameters' },
      persistence: 'Pure runtime arithmetic.', equation: 'h = x + \\operatorname{Attn}(\\operatorname{RMSNorm}(x))',
      below: [{ label: 'See the two vectors being added (toy simulator)', href: '#/labs/layers' }],
    },
    residual2: {
      id, title: 'Residual addition after the FFN / MoE', role: 'op',
      what: 'Adds the FFN (or MoE) branch output to h: y = h + FFN(...). No learned weights.',
      where: blockLabel(p, b), inputs: `h and FFN output, both [B, T, ${p.d}]`, outputs: `y [B, T, ${p.d}]`, params: { count: 0, formula: 'no parameters' },
      persistence: 'Pure runtime arithmetic.', equation: `y = h + \\operatorname{${p.kind === 'dense' ? 'FFN' : 'MoE'}}(\\operatorname{RMSNorm}(h))`,
      below: [{ label: 'See the two vectors being added (toy simulator)', href: '#/labs/layers' }],
    },
    ffn: {
      id, title: 'SwiGLU feed-forward network', role: 'param',
      what: `Applied to each token separately. ${fmtInt(p.ffnWidth ?? 0)} gated channels: each channel j is SiLU(gate row j · x) × (up row j · x); the down projection mixes all channels back to ${fmtInt(p.d)} features.`,
      where: `${blockLabel(p, b)}, after the second RMSNorm.`, inputs: `normalized h [B, T, ${p.d}]`, outputs: `[B, T, ${p.d}]`,
      params: { count: a.ffnPerBlock, formula: `3 × d × f = 3 × ${fmtInt(p.d)} × ${fmtInt(p.ffnWidth ?? 0)} = ${fmtInt(a.ffnPerBlock)}` },
      persistence: 'Three learned matrices; g, u, z are activations.', equation: 'Y = \\big(\\operatorname{SiLU}(XW_{gate}^{\\top})\\odot XW_{up}^{\\top}\\big)W_{down}^{\\top}',
      implementation: 'Qwen3MLP', sources: [...srcs(p), 'swiglu', 'torch-silu'],
      below: [{ label: 'Open the FFN module (Level 3)', href: modHref('ffn') }, { label: 'Unit microscope', href: '#/labs/unit' }],
    },
    moe: {
      id, title: `Routed mixture of experts (${p.moe?.numExperts} experts, top-${p.moe?.topK})`, role: 'param',
      what: `Replaces the dense FFN. For every token in this block the router scores all ${p.moe?.numExperts} experts, keeps the ${p.moe?.topK} most probable, renormalizes their probabilities to sum to 1, and adds those experts' outputs weighted by the renormalized coefficients. ${p.moe?.sharedExperts === 0 ? 'There is no shared expert in this model.' : ''}`,
      where: `${blockLabel(p, b)}, after the second RMSNorm. Attention stays shared (not routed).`, inputs: `normalized h, per token [${p.d}]`, outputs: `[${p.d}] per token`,
      params: { count: a.ffnPerBlock + a.routerPerBlock, formula: `${p.moe?.numExperts} experts × ${fmtInt(a.oneExpert ?? 0)} + router ${fmtInt(a.routerPerBlock)}` },
      persistence: 'All experts are stored (and must be in memory or reachable) even though only a few run per token. Router logits, probabilities, the selected ids and mixture coefficients are per-token activations.',
      equation: 'y=\\sum_{e\\in\\operatorname{TopK}(p,k)}\\frac{p_e}{\\sum_{j\\in S}p_j}\\operatorname{Expert}_e(x)', implementation: 'Qwen3MoeSparseMoeBlock (TopKRouter + Experts)',
      sources: [...srcs(p), 'switch'],
      below: [{ label: 'Open the MoE module (Level 3)', href: modHref('moe') }, { label: 'Router lab', href: '#/labs/router' }],
    },
    embedding: {
      id, title: 'Input embedding lookup', role: 'param',
      what: 'For each token id, copy that row of the embedding table. Only one row per token is read; the rest of the table is untouched for that token.',
      where: 'Model entrance (model.embed_tokens.weight).', inputs: 'token ids [B, T]', outputs: `X [B, T, ${p.d}]`,
      params: { count: a.embedding, formula: `${fmtInt(p.vocabRows)} × ${fmtInt(p.d)}` },
      persistence: 'The table is learned; the looked-up rows become the first activations. There is no learned position vector added here: position enters later through RoPE inside attention.',
      equation: 'x_t = E[\\text{id}_t]',
    },
    final_norm: {
      id, title: 'Final RMSNorm', role: 'norm', what: 'Normalizes the last block output before the vocabulary projection.', where: 'After the last block (model.norm.weight).',
      inputs: `[B, T, ${p.d}]`, outputs: `[B, T, ${p.d}]`, params: { count: a.finalNorm, formula: `γ has ${fmtInt(p.d)} values` }, persistence: 'γ learned.',
      equation: '\\operatorname{RMSNorm}(h_L)', sources: [...srcs(p), 'rmsnorm'],
    },
    lm_head: {
      id, title: 'Vocabulary projection (LM head)', role: 'param',
      what: `Computes one score (logit) for each of the ${fmtInt(p.vocabRows)} vocabulary rows by a dot product with that row of its own weight matrix. Logits are not probabilities; softmax turns them into probabilities.`,
      where: 'Model exit (lm_head.weight).', inputs: `[B, T, ${p.d}]`, outputs: `logits [B, T, ${fmtInt(p.vocabRows)}]`,
      params: { count: a.head, formula: `${fmtInt(p.vocabRows)} × ${fmtInt(p.d)} (untied)` }, persistence: 'Learned and untied from the input embedding in this model.',
      equation: '\\text{logits}=\\hat h W_{vocab}^{\\top}',
    },
  };
  const ops: Record<string, Omit<InspectorInfo, 'evidence' | 'sources' | 'below'> & { sources?: string[] }> = {
    reshape: { id, title: 'Reshape into heads', role: 'op', what: `Splits the ${p.hq * p.headDim}-wide query vector into ${p.hq} heads of ${p.headDim}, and the ${p.hkv * p.headDim}-wide key/value vectors into ${p.hkv} heads. A view change with no arithmetic and no weights.`, where: blockLabel(p, b), inputs: `[B, T, ${p.hq * p.headDim}] / [B, T, ${p.hkv * p.headDim}]`, outputs: `[B, ${p.hq}, T, ${p.headDim}] / [B, ${p.hkv}, T, ${p.headDim}]`, params: null, persistence: 'No storage of its own.' },
    rope: { id, title: 'Rotary position embedding (RoPE)', role: 'op', what: `Rotates pairs of coordinates (i, i + ${p.headDim / 2}) of every query and key head by an angle proportional to the token's absolute position, with frequencies θ^(−2i/d_h), θ = ${fmtInt(p.ropeTheta)}. Applied to Q and K only, not to V. It has no learned weights and is not a vector added at the embedding.`, where: `${blockLabel(p, b)}, after q/k norm.`, inputs: 'normalized q, k heads + absolute positions', outputs: 'rotated q, k', params: null, persistence: 'cos/sin tables are computed (non-persistent buffers), not trained.', equation: "q'_i = q_i\\cos(m\\omega_i) - q_{i+d/2}\\sin(m\\omega_i)", sources: ['rope', 'tf-qwen3'] },
    sdpa: { id, title: 'Causal scaled dot-product attention (GQA)', role: 'op', what: `Each query head h uses K/V head ⌊h / ${p.hq / p.hkv}⌋. Scores q·k/√${p.headDim}, future positions masked, softmax per row, weighted sum of V. Optimized kernels (e.g. FlashAttention) compute the same result in tiles without storing the full T×T matrix.`, where: blockLabel(p, b), inputs: `Q [B, ${p.hq}, Tq, ${p.headDim}], K/V [B, ${p.hkv}, Tk, ${p.headDim}]`, outputs: `Z [B, ${p.hq}, Tq, ${p.headDim}] (logical scores [B, ${p.hq}, Tq, Tk])`, params: null, persistence: 'Attention probabilities are activations; K/V are cached during generation.', equation: '\\operatorname{softmax}(QK^{\\top}/\\sqrt{d_h}+M)V', sources: ['vaswani', 'gqa', 'flashattn', 'tf-qwen3'] },
    concat: { id, title: 'Concatenate head outputs', role: 'op', what: `Puts the ${p.hq} head outputs side by side into one ${p.hq * p.headDim}-wide vector per token.`, where: blockLabel(p, b), inputs: `[B, ${p.hq}, T, ${p.headDim}]`, outputs: `[B, T, ${p.hq * p.headDim}]`, params: null, persistence: 'No storage of its own.' },
    silu: { id, title: 'SiLU nonlinearity', role: 'op', what: 'SiLU(a) = a·sigmoid(a), applied to each gate pre-activation separately.', where: 'Inside the FFN / each expert.', inputs: 'gate pre-activations', outputs: 'g', params: null, persistence: 'No weights.', equation: '\\operatorname{SiLU}(a)=a\\,\\sigma(a)', sources: ['torch-silu', 'swiglu'] },
    mul: { id, title: 'Multiplicative gate (⊙)', role: 'op', what: 'Elementwise product z_j = g_j × u_j. This multiplication is why a SwiGLU channel is not the same as a textbook "neuron" with a single weighted sum and activation.', where: 'Inside the FFN / each expert.', inputs: 'g and u', outputs: 'z', params: null, persistence: 'No weights.' },
    softmax: { id, title: 'Router softmax over all experts', role: 'op', what: `Turns the ${p.moe?.numExperts} router logits of a token into probabilities that sum to 1 (computed in float32).`, where: 'MoE router', inputs: `${p.moe?.numExperts} logits`, outputs: `${p.moe?.numExperts} probabilities`, params: null, persistence: 'Per-token activation.', sources: ['tf-qwen3moe'] },
    topk: { id, title: `Top-${p.moe?.topK} selection`, role: 'op', what: `Keeps the ${p.moe?.topK} experts with the highest probability. A discrete choice: it has no gradient with respect to which experts are chosen.`, where: 'MoE router', inputs: 'probabilities', outputs: `${p.moe?.topK} expert ids + their probabilities`, params: null, persistence: 'Per-token activation.', sources: ['tf-qwen3moe'] },
    renorm: { id, title: 'Renormalize selected probabilities', role: 'op', what: `norm_topk_prob = ${p.moe?.normTopkProb}: divide the ${p.moe?.topK} kept probabilities by their sum, so the mixture coefficients α sum to 1.`, where: 'MoE router', inputs: `${p.moe?.topK} probabilities`, outputs: 'α', params: null, persistence: 'Per-token activation.', equation: '\\alpha_e = p_e / \\sum_{j\\in S} p_j', sources: ['q30-config', 'tf-qwen3moe'] },
    wsum: { id, title: 'Weighted sum of expert outputs', role: 'op', what: 'Adds the selected experts’ outputs, each scaled by its α. Unselected experts contribute nothing for this token in this block.', where: 'MoE output', inputs: `${p.moe?.topK} expert outputs`, outputs: `[${p.d}]`, params: null, persistence: 'No weights.' },
    sq: { id, title: 'Square and mean', role: 'op', what: 'Computes mean(x²) over the feature axis for each token.', where: 'Inside RMSNorm', inputs: `[${p.d}]`, outputs: 'one number per token', params: null, persistence: 'Runtime.' },
    rsqrt: { id, title: '1 / √(mean + ε)', role: 'op', what: `Adds ε = ${p.rmsEps} for numerical safety and takes the reciprocal square root.`, where: 'Inside RMSNorm', inputs: 'mean(x²)', outputs: 'scale factor', params: null, persistence: 'Runtime.' },
    scalex: { id, title: 'Scale x', role: 'op', what: 'Multiplies every feature by the per-token scale factor.', where: 'Inside RMSNorm', inputs: 'x and scale', outputs: 'normalized x', params: null, persistence: 'Runtime.' },
    lookup: { id, title: 'Row lookup', role: 'op', what: 'Indexing, not a matrix multiply: row id_t of the table becomes x_t.', where: 'Embedding', inputs: 'token id', outputs: `[${p.d}]`, params: null, persistence: 'Runtime.' },
    logits: { id, title: 'Logits', role: 'act', what: 'Unnormalized scores, one per vocabulary row. Large positive = more likely after softmax.', where: 'Model output', inputs: '', outputs: `[${fmtInt(p.vocabRows)}]`, params: null, persistence: 'Runtime activation.' },
    temp: { id, title: 'Temperature', role: 'op', what: 'Divides logits by T. T < 1 sharpens, T > 1 flattens; T = 0 is treated as explicit greedy argmax.', where: 'Sampling', inputs: 'logits', outputs: 'scaled logits', params: null, persistence: 'A setting, not a parameter.', sources: ['q8-gen'] },
    filter: { id, title: 'Top-k / top-p filtering + renormalization', role: 'op', what: 'Removes unlikely tokens, then renormalizes the remaining probabilities to sum to 1.', where: 'Sampling', inputs: 'probabilities', outputs: 'filtered distribution', params: null, persistence: 'Setting.' },
    draw: { id, title: 'Random draw', role: 'op', what: 'Samples one token id from the filtered distribution with a pseudo-random number (seeded in the simulator).', where: 'Sampling', inputs: 'distribution', outputs: 'token id', params: null, persistence: 'Advances the RNG state.' },
    x_in: { id, title: 'Block input x', role: 'act', what: 'The residual-stream vector entering this block for each token: an activation.', where: blockLabel(p, b), inputs: '', outputs: `[B, T, ${p.d}]`, params: null, persistence: 'Runtime activation.' },
    x_out: { id, title: 'Block output y', role: 'act', what: 'The residual-stream vector leaving this block.', where: blockLabel(p, b), inputs: '', outputs: `[B, T, ${p.d}]`, params: null, persistence: 'Runtime activation.' },
    experts: { id, title: `${p.moe?.numExperts ?? 0} expert FFNs`, role: 'param', what: 'Each expert is an ordinary SwiGLU FFN with its own gate, up and down matrices. An expert is not a separate language model. No human-readable job description is assigned to any expert id: specialization is learned and may be distributed.', where: blockLabel(p, b), inputs: `token vector [${p.d}] (only for tokens routed to it)`, outputs: `[${p.d}]`, params: { count: a.ffnPerBlock, formula: `${p.moe?.numExperts} × 3 × ${p.d} × ${p.moe?.expertWidth}` }, persistence: 'All stored; a few computed per token.' },
  };
  const c = common[id] ?? ops[id];
  if (c) {
    const src = (c as { sources?: string[] }).sources ?? srcs(p);
    return {
      evidence: (c as { evidence?: InspectorInfo['evidence'] }).evidence ?? [evStruct(p)],
      below: (c as { below?: InspectorInfo['below'] }).below ?? [],
      ...c,
      sources: src,
      above: b !== undefined && (BLOCKISH.has(id) || ops[id]) ? { label: `Back to block ${b}`, href: blockHref(b) } : { label: 'Back to the full model', href: explorerHref({ m: p.id, lv: 5 }) },
    } as InspectorInfo;
  }
  if (id.startsWith('expert:')) {
    const e = Number(id.slice(7));
    return {
      id, title: `Expert ${e}`, role: 'param',
      what: `Expert ${e} of ${p.moe!.numExperts} in this block: a SwiGLU FFN of width ${p.moe!.expertWidth}. Its id is just an index; no specialization label is claimed.`,
      where: `${blockLabel(p, b)}, mlp.experts.${e}`, inputs: `[${p.d}] for routed tokens`, outputs: `[${p.d}]`,
      params: { count: a.oneExpert!, formula: `3 × ${p.d} × ${p.moe!.expertWidth} = ${fmtInt(a.oneExpert!)}` },
      persistence: 'Stored whether or not it is selected.',
      checkpoint: ['gate', 'up', 'down'].map((w) => `model.layers.${b ?? 0}.mlp.experts.${e}.${w}_proj.weight`),
      evidence: [evStruct(p)], sources: srcs(p),
      below: ['expert_gate_proj', 'expert_up_proj', 'expert_down_proj'].map((k) => ({ label: `Open ${k.replace('expert_', '')} matrix`, href: explorerHref({ m: p.id, lv: 2, b: b ?? 0, t: k, e }) })),
      above: { label: 'Back to the MoE module', href: explorerHref({ m: p.id, lv: 3, b: b ?? 0, mod: 'moe' }) },
    };
  }
  return {
    id, title: id, role: 'op', what: 'Unknown element.', where: '', inputs: '', outputs: '', params: null, persistence: '', evidence: [{ kind: 'unknown', note: 'No description available.' }], sources: [], below: [],
  };
}

const BLOCKISH = new Set(['attention', 'attn_norm', 'ffn_norm', 'residual1', 'residual2', 'ffn', 'moe']);

export const isEvidenceSim = (p: CoreProfile) => V(p) === 'simulation';
