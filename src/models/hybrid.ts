import type { HybridStateSpec } from '../math/memory';

export type Coverage = 'full-toy' | 'structural' | 'overview';
export const COVERAGE_LABEL: Record<Coverage, string> = {
  'full-toy': 'Full numerical toy analogue + verified structural view',
  structural: 'Verified structural view (config, checkpoint headers, implementation)',
  overview: 'Explanatory overview',
};

export interface HybridProfile {
  id: string;
  name: string;
  repo: string;
  revision: string;
  modelType: string;
  coverage: Coverage;
  extraToys: string[];
  layers: number;
  layout: string; // model-card wording
  layerTypes: ('gdn' | 'attn')[];
  attnLabel: string; // what the "attention" slot really is
  d: number;
  residual: string;
  gdn: { vHeads: number; qkHeads: number; headDim: number; convKernel: number };
  attn: { qHeads: number; kvHeads: number; headDim: number; rotaryDim: number; outputGate: boolean };
  moe: { experts: number; routed: number; shared: number; expertWidth: number; sharedWidth: number };
  vocabRows: number;
  context: string;
  advertised: string;
  extras: string[];
  sources: string[];
}

const pattern = (blocks: number): ('gdn' | 'attn')[] => Array.from({ length: blocks * 4 }, (_, i) => (i % 4 === 3 ? 'attn' : 'gdn'));

export const QWEN3_NEXT: HybridProfile = {
  id: 'qwen3-next-80b-a3b',
  name: 'Qwen3-Next-80B-A3B',
  repo: 'Qwen/Qwen3-Next-80B-A3B-Instruct',
  revision: '9c7f2fbe84465e40164a94cc16cd30b6999b0cc7',
  modelType: 'qwen3_next',
  coverage: 'structural',
  extraToys: ['Gated DeltaNet recurrence (shared with Qwen3.8 toy)'],
  layers: 48,
  layout: '12 × (3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE))',
  layerTypes: pattern(12),
  attnLabel: 'Gated attention (ordinary causal softmax attention with a sigmoid output gate)',
  d: 2048,
  residual: 'Ordinary single residual stream of width 2048',
  gdn: { vHeads: 32, qkHeads: 16, headDim: 128, convKernel: 4 },
  attn: { qHeads: 16, kvHeads: 2, headDim: 256, rotaryDim: 64, outputGate: true },
  moe: { experts: 512, routed: 10, shared: 1, expertWidth: 512, sharedWidth: 512 },
  vocabRows: 151936,
  context: '262,144 native (model card); extensible to 1,010,000',
  advertised: '80B total, 3B activated (model card)',
  extras: ['Multi-token prediction (MTP)', 'Zero-centred, weight-decayed layernorm (Transformers docs)'],
  sources: ['qn-card'],
};

export const QWEN3_5_35B: HybridProfile = {
  id: 'qwen3.5-35b-a3b',
  name: 'Qwen3.5-35B-A3B',
  repo: 'Qwen/Qwen3.5-35B-A3B',
  revision: '59d61f3ce65a6d9863b86d2e96597125219dc754',
  modelType: 'qwen3_5_moe',
  coverage: 'structural',
  extraToys: [],
  layers: 40,
  layout: '10 × (3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE))',
  layerTypes: pattern(10),
  attnLabel: 'Gated attention',
  d: 2048,
  residual: 'Ordinary single residual stream of width 2048',
  gdn: { vHeads: 32, qkHeads: 16, headDim: 128, convKernel: 4 },
  attn: { qHeads: 16, kvHeads: 2, headDim: 256, rotaryDim: 64, outputGate: true },
  moe: { experts: 256, routed: 8, shared: 1, expertWidth: 512, sharedWidth: 512 },
  vocabRows: 248320,
  context: '262,144 native (model card); extensible to 1,010,000',
  advertised: '35B total, 3B activated (model card); vision encoder included',
  extras: ['Vision encoder (early-fusion vision-language model)', 'MTP trained with multi-steps'],
  sources: ['q35-card'],
};

export const QWEN3_8_FLASH_NEXT: HybridProfile = {
  id: 'qwen3.8-flash-next',
  name: 'Qwen3.8-Flash-Next',
  repo: 'Qwen/Qwen3.8-Flash-Next',
  revision: 'de4b8e4d43b917e7706784d8bb445c9af86a3540',
  modelType: 'qwen4_exp',
  coverage: 'structural',
  extraToys: ['Gated DeltaNet recurrence', 'QSA block selection', 'Gated residual read/write'],
  layers: 48,
  layout: '12 × (3 × (Gated DeltaNet → MoE) → 1 × (Qwen Sparse Attention → MoE))',
  layerTypes: pattern(12),
  attnLabel: 'Qwen Sparse Attention (QSA): gated softmax attention restricted to indexer-selected 4-token micro-blocks plus the tail block. The config calls these layers "full_attention", but the implementation adds a selection mask.',
  d: 2560,
  residual: '4 residual branches of width 2560 each, carried as one [B, T, 10240] tensor (hc_count = 4)',
  gdn: { vHeads: 48, qkHeads: 16, headDim: 128, convKernel: 4 },
  attn: { qHeads: 24, kvHeads: 2, headDim: 256, rotaryDim: 64, outputGate: true },
  moe: { experts: 512, routed: 10, shared: 1, expertWidth: 640, sharedWidth: 640 },
  vocabRows: 248320,
  context: '262,144 native, extensible to 1,000,000 (model card); max_position_embeddings 262,144 (config)',
  advertised: '125B with 6B activated, plus 51B n-gram embedding and 4B MTP (model card)',
  extras: ['Hashed bigram/trigram tables at "layer 2" (config ple_layer_ids=[2]; code block index 1)', 'MTP module (1 layer)', 'Vision encoder (27 blocks)'],
  sources: ['q38-card', 'q38-config', 'q38-index', 'tf-qwen4exp', 'q38-report'],
};

export const HYBRIDS = [QWEN3_NEXT, QWEN3_5_35B, QWEN3_8_FLASH_NEXT];

/** Stored elements by component, summed from the pinned Qwen3.8-Flash-Next safetensors headers (derived). */
export const QWEN38_COMPONENTS: { key: string; label: string; elements: number; group: 'backbone' | 'ngram' | 'mtp' | 'vision' }[] = [
  { key: 'routed_experts', label: 'Routed experts (48 × 512)', elements: 120_795_955_200, group: 'backbone' },
  { key: 'gated_deltanet', label: 'Gated DeltaNet mixers (36 blocks)', elements: 2_086_510_464, group: 'backbone' },
  { key: 'gated_residual', label: 'Gated residual read/write (2 per block + final mixer)', elements: 640_624_640, group: 'backbone' },
  { key: 'embed', label: 'Input embedding [248320, 2560]', elements: 635_699_200, group: 'backbone' },
  { key: 'lm_head', label: 'Output head [248320, 2560]', elements: 635_699_200, group: 'backbone' },
  { key: 'qsa_attention', label: 'QSA attention projections + norms (12 blocks)', elements: 597_694_464, group: 'backbone' },
  { key: 'shared_expert', label: 'Shared experts + shared-expert gates (48)', elements: 236_052_480, group: 'backbone' },
  { key: 'router', label: 'Routers (48 × [512, 2560])', elements: 62_914_560, group: 'backbone' },
  { key: 'qsa_indexer', label: 'QSA indexers (12)', elements: 19_663_872, group: 'backbone' },
  { key: 'ngram_tables', label: 'N-gram tables (128 shards, 320,001,536 rows × 160)', elements: 51_200_245_760, group: 'ngram' },
  { key: 'ple_other', label: 'N-gram injection projections, norms, conv (+35 int64 buffer entries)', elements: 32_839_715, group: 'ngram' },
  { key: 'mtp', label: 'MTP module tensors (mtp.*)', elements: 2_607_150_848, group: 'mtp' },
  { key: 'vision', label: 'Vision encoder (model.visual.*)', elements: 448_931_056, group: 'vision' },
];

export const QWEN38_STATE: HybridStateSpec = {
  qsaLayers: 12,
  qsaKvHeads: 2,
  qsaHeadDim: 256,
  indexerKeyDim: 128,
  gdnLayers: 36,
  gdnValueHeads: 48,
  gdnKeyDim: 128,
  gdnValueDim: 128,
  gdnConvChannels: 2 * 16 * 128 + 48 * 128,
  gdnConvKernel: 4,
  pleLayers: 1,
  pleConvChannels: 4 * 2560,
  pleConvStateLen: 9,
  ngramContext: 2,
};

export interface FamilyEntry {
  name: string;
  date: string; // HF repo creation date of the named reference repository
  repo: string;
  architecture: string;
  relation: string;
  inAtlas: string;
}

export const FAMILY: FamilyEntry[] = [
  { name: 'Qwen (first generation)', date: '2023-08-03', repo: 'Qwen/Qwen-7B', architecture: 'Dense decoder', relation: 'Original series', inAtlas: 'Timeline only' },
  { name: 'Qwen1.5', date: '2024-01-22', repo: 'Qwen/Qwen1.5-7B', architecture: 'Dense decoder (and Qwen1.5-MoE variants)', relation: 'Predecessor', inAtlas: 'Timeline only' },
  { name: 'Qwen2', date: '2024-06-04', repo: 'Qwen/Qwen2-7B', architecture: 'Dense + MoE; GQA; QKV bias (per the Qwen3 report, which removes it)', relation: 'Predecessor', inAtlas: 'Timeline only' },
  { name: 'Qwen2.5', date: '2024-09-15', repo: 'Qwen/Qwen2.5-7B', architecture: 'Dense; Qwen2.5-MoE used shared experts (per the Qwen3 report)', relation: 'Direct predecessor of Qwen3', inAtlas: 'Timeline only' },
  { name: 'Qwen3 (dense + MoE)', date: '2025-04-27', repo: 'Qwen/Qwen3-8B, Qwen/Qwen3-30B-A3B', architecture: 'Pre-norm decoder, GQA, QK-Norm, RoPE, SwiGLU; MoE with 128 experts / top-8, no shared expert', relation: 'Reference presets of this atlas', inAtlas: 'Deepest coverage: full toy analogue + verified structure' },
  { name: 'Qwen3-Next', date: '2025-09-09', repo: 'Qwen/Qwen3-Next-80B-A3B-Instruct', architecture: 'Hybrid 3:1 Gated DeltaNet / gated attention, 512 experts (10 routed + 1 shared), MTP', relation: 'First hybrid Qwen line', inAtlas: 'Verified structural view' },
  { name: 'Qwen3.5', date: '2026-02-24', repo: 'Qwen/Qwen3.5-35B-A3B', architecture: 'Hybrid 3:1 Gated DeltaNet / gated attention, MoE with shared expert, vision encoder', relation: 'Hybrid, vision-language', inAtlas: 'Verified structural view' },
  { name: 'Qwen3.6', date: '2026-04-15', repo: 'Qwen/Qwen3.6-35B-A3B', architecture: 'Same Transformers architecture class as Qwen3.5 (Qwen3_5MoeForConditionalGeneration; 40 layers, 256 experts, top-8)', relation: 'Hybrid, vision-language', inAtlas: 'Listed only (config class checked; not analyzed separately)' },
  { name: 'Qwen3.8-Flash-Next', date: '2026-08-24', repo: 'Qwen/Qwen3.8-Flash-Next', architecture: 'model_type qwen4_exp: Gated DeltaNet + Qwen Sparse Attention, 4-branch gated residual, hashed n-gram tables, 512 experts (10 + 1 shared), MTP, vision encoder', relation: '"Experimental preview of the architecture that will underpin Qwen4" (model card)', inAtlas: 'Verified structural view + toy analogues of three mechanisms' },
];
