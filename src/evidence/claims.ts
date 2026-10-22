import type { EvidenceClass } from './types';

export interface Claim {
  id: string;
  scope: string;
  claim: string;
  evidence: EvidenceClass;
  sources: string[];
  note?: string;
}

// Claim/evidence registry. Discrepancies between sources are recorded explicitly rather than flattened.
export const CLAIMS: Claim[] = [
  { id: 'c-8b-dims', scope: 'Qwen3-8B', claim: '36 decoder blocks; d = 4096; 32 query / 8 K/V heads; head_dim 128; FFN width 12,288; vocab rows 151,936; untied; no linear biases; RMSNorm eps 1e-6; rope_theta 1e6.', evidence: 'verified', sources: ['q8-config', 'tf-qwen3'] },
  { id: 'c-30b-dims', scope: 'Qwen3-30B-A3B', claim: '48 decoder blocks; d = 2048; 32 / 4 heads; head_dim 128 (so Hq·dh = 4096 ≠ d); 128 experts, top-8, expert width 768, norm_topk_prob = true; no shared expert.', evidence: 'verified', sources: ['q30-config', 'tf-qwen3moe', 'qwen3-report'] },
  { id: 'c-30b-intermediate', scope: 'Qwen3-30B-A3B', claim: 'intermediate_size = 6144 is present in the config but no block uses a dense FFN: mlp_only_layers = [] and decoder_sparse_step = 1, so all 48 blocks are MoE with expert width 768.', evidence: 'verified', sources: ['q30-config', 'tf-qwen3moe'] },
  { id: 'c-totals', scope: 'both', claim: 'Exact parameter totals: 8,190,735,360 (Qwen3-8B) and 30,532,122,624 (Qwen3-30B-A3B).', evidence: 'derived', sources: ['q8-config', 'q30-config', 'q8-index', 'q30-index'], note: 'Computed from config shapes; independently equal to the element totals of the pinned safetensors headers and to index total_size / 2 bytes.' },
  { id: 'c-advertised', scope: 'both', claim: 'Model cards round these to 8.2B (6.95B non-embedding) and 30.5B (29.9B non-embedding, 3.3B activated).', evidence: 'verified', sources: ['q8-card', 'q30-card'], note: 'The non-embedding figures match total − input embedding − output head: 6,946,075,648 ≈ 6.95B and 29,909,792,768 ≈ 29.9B. Subtracting only the input table (7.57B / 30.22B) does not match.' },
  { id: 'c-active', scope: 'Qwen3-30B-A3B', claim: 'Selected experts over all 48 blocks: 1,811,939,328 parameters; with all non-expert parameters (incl. the whole embedding table): 3,353,032,704.', evidence: 'derived', sources: ['q30-config', 'q30-card'], note: 'The second convention reproduces the card’s ≈3.3B but counts an entire embedding table that a lookup does not read. Not a count of FLOPs, traffic or latency.' },
  { id: 'c-context', scope: 'both', claim: 'Native context 32,768 tokens; 131,072 with YaRN (factor 4, opt-in rope_scaling); config max_position_embeddings 40,960.', evidence: 'verified', sources: ['q8-card', 'q8-config', 'q30-card'], note: 'The Qwen3 report table lists 128K context; the model card explains 40,960 = 32,768 output + 8,192 prompt reserve. These are different quantities and are shown separately.' },
  { id: 'c-tokenizer', scope: 'Qwen3', claim: 'Tokenizer: byte-level BPE with 151,643 regular + 26 added tokens = 151,669 ids; embedding/head have 151,936 rows (267 unused padding rows).', evidence: 'verified', sources: ['q8-tokenizer', 'qwen3-report'] },
  { id: 'c-qknorm', scope: 'Qwen3', claim: 'Q and K are RMS-normalized per head with one head_dim-sized gain each per block (shared across heads), before RoPE.', evidence: 'verified', sources: ['tf-qwen3', 'qwen3-report'] },
  { id: 'c-rope', scope: 'Qwen3', claim: 'RoPE pairs coordinate i with i + head_dim/2 (rotate_half), frequencies θ^(−2i/d), applied to Q and K only, no learned parameters.', evidence: 'verified', sources: ['tf-qwen3', 'rope'] },
  { id: 'c-rmsnorm', scope: 'Qwen3', claim: 'RMSNorm computes in float32: weight * (x * rsqrt(mean(x²) + eps)), no mean subtraction.', evidence: 'verified', sources: ['tf-qwen3', 'rmsnorm'] },
  { id: 'c-experts-layout', scope: 'Qwen3-30B-A3B', claim: 'The checkpoint stores 128 × 3 separate expert tensors per block; current Transformers packs them at load into gate_up_proj [128, 1536, 2048] (gate first) and down_proj [128, 2048, 768].', evidence: 'verified', sources: ['q30-index', 'tf-qwen3moe', 'tf-convmap'] },
  { id: 'c-router', scope: 'Qwen3-30B-A3B', claim: 'Router: logits = x W_routerᵀ; softmax over all 128 experts in float32; top-8; renormalize the 8 probabilities.', evidence: 'verified', sources: ['tf-qwen3moe', 'q30-config'] },
  { id: 'c-lb', scope: 'Qwen3-30B-A3B', claim: 'Training used a global-batch load-balancing loss; router_aux_loss_coef = 0.001 in the config.', evidence: 'verified', sources: ['qwen3-report', 'q30-config'] },
  { id: 'c-kv', scope: 'both', claim: 'K/V cache at BF16, batch 1: 147,456 B/token (8B), 98,304 B/token (30B-A3B); 4.5 GiB and 3 GiB at 32,768 tokens (K/V only).', evidence: 'derived', sources: ['q8-config', 'q30-config', 'hf-cache'] },
  { id: 'c-train', scope: 'Qwen3', claim: 'Pre-training on over 36T tokens (119 languages) in three stages; 8B and 30B-A3B post-trained by strong-to-weak distillation.', evidence: 'verified', sources: ['qwen3-report'] },
  { id: 'c-train-unknown', scope: 'Qwen3', claim: 'Exact data mixtures, per-model optimizer/learning-rate settings and meanings of individual neurons or experts.', evidence: 'unknown', sources: ['qwen3-report'], note: 'Not reproduced on this site; nothing fills these gaps.' },
  { id: 'c-thinking', scope: 'Qwen3', claim: 'Thinking / non-thinking is selected through the chat template (enable_thinking) or /think, /no_think; the weights and layers are identical.', evidence: 'verified', sources: ['q8-tokenizer', 'qwen3-report', 'q8-card'] },
  { id: 'c-38-layout', scope: 'Qwen3.8-Flash-Next', claim: '48 blocks = 12 × (3 Gated DeltaNet + 1 Qwen Sparse Attention), each followed by MoE with 512 experts (10 routed + 1 shared, width 640).', evidence: 'verified', sources: ['q38-card', 'q38-config', 'tf-qwen4exp'] },
  { id: 'c-38-qsa', scope: 'Qwen3.8-Flash-Next', claim: 'Layers labelled full_attention in the config are QSA: keys pooled into 4-token blocks, indexer keeps the top 2048/4 = 512 blocks plus the incomplete tail, applied as an extra mask on gated softmax attention.', evidence: 'verified', sources: ['tf-qwen4exp', 'q38-config', 'q38-report'] },
  { id: 'c-38-residual', scope: 'Qwen3.8-Flash-Next', claim: 'Residual state is 4 branches × 2560 = [B, T, 10240]; per sub-layer: elementwise sigmoid read gate (rank-320 bottleneck), mean over branches, per-branch 2·sigmoid write gate.', evidence: 'verified', sources: ['tf-qwen4exp', 'q38-config', 'q38-card'] },
  { id: 'c-38-ngram', scope: 'Qwen3.8-Flash-Next', claim: 'Hashed bigram/trigram tables at config "layer 2" = code block index 1 (layers.1.ple.*): 128 shards, 320,001,536 rows × 160 = 51,200,245,760 parameters.', evidence: 'verified', sources: ['q38-config', 'q38-index', 'tf-qwen4exp'], note: 'ple_layer_ids = [2] is 1-based: the implementation tests layer_idx + 1 ∈ ple_layer_ids.' },
  { id: 'c-38-accounting', scope: 'Qwen3.8-Flash-Next', claim: 'Header-derived: text backbone 125,710,814,080; n-gram 51,233,085,475 (incl. injection); MTP 2,607,150,848; vision 448,931,056; total stored elements 179,999,981,459.', evidence: 'derived', sources: ['q38-index'] },
  { id: 'c-38-mtp', scope: 'Qwen3.8-Flash-Next', claim: 'How the model card arrives at “4B MTP”.', evidence: 'unknown', sources: ['q38-card', 'q38-index'], note: 'mtp.* tensors total 2.61B; adding the shared embedding and head gives 3.88B ≈ 4B, which may be the convention, but no source checked states it.' },
  { id: 'c-38-gdn', scope: 'Qwen3.8-Flash-Next', claim: 'Gated DeltaNet recurrence: S ← S·exp(g); δ = β(v − Sᵀk); S ← S + kδᵀ; o = Sᵀq, with g = −exp(A_log)·softplus(a + dt_bias), β = σ(b), L2-normalized q/k.', evidence: 'verified', sources: ['tf-qwen4exp', 'gdn'] },
  { id: 'c-toy', scope: 'Toy simulator', claim: 'All toy values (weights, activations, routes, attention maps, generated toy tokens) are illustrative simulations with seeded synthetic weights.', evidence: 'simulation', sources: [] },
  { id: 'c-samples', scope: 'Qwen3-8B / 30B-A3B', claim: 'Seven small weight tiles were read from the pinned checkpoint shards with verified HTTP 206 range responses; each has a SHA-256 of its raw bytes.', evidence: 'sample', sources: ['q8-index', 'q30-index', 'safetensors'] },
];
