import { explorerHref } from '../state/route';
import { QWEN3_8B } from '../models/profiles';
import { accounting } from '../models/shapes';

const A8 = accounting(QWEN3_8B);
const pct = (x: number) => `${((x / A8.total) * 100).toFixed(0)}%`;
const SHARES = `about ${pct(A8.ffnPerBlock * QWEN3_8B.layers)} in FFN matrices, ${pct(A8.attentionPerBlock * QWEN3_8B.layers)} in attention projections, ${pct(A8.embedding + A8.head)} in the two vocabulary-sized tables, and under 0.01% in norm gains`;

export interface TourStep { title: string; text: string; href: string; cta: string }
export interface Tour { id: string; title: string; summary: string; steps: TourStep[] }

export const TOURS: Tour[] = [
  {
    id: 'weights',
    title: 'Where are the weights?',
    summary: 'From one real stored number in Qwen3-8B out to the 8.19 billion of them.',
    steps: [
      { title: 'One stored number', text: 'This is element [0, 0] of the query projection of block 0: a single BF16 value actually read from the checkpoint file, with its exact byte position. A parameter is exactly this: one learned scalar.', href: explorerHref({ m: 'qwen3-8b', lv: 0, b: 0, t: 'q_proj', r: 0, c: 0 }), cta: 'See the number' },
      { title: 'Its row', text: 'Row 0 of W_Q holds the 4,096 weights that produce query coordinate 0. The output is a weighted sum of the 4,096 normalized input features; the toy panel shows each product.', href: explorerHref({ m: 'qwen3-8b', lv: 1, b: 0, t: 'q_proj', r: 0 }), cta: 'Open the unit' },
      { title: 'Its matrix', text: 'W_Q is a [4,096 × 4,096] matrix: 16.8 million parameters, stored once, applied to every token position.', href: explorerHref({ m: 'qwen3-8b', lv: 2, b: 0, t: 'q_proj' }), cta: 'Open the matrix' },
      { title: 'Its module', text: 'Attention owns four matrices (Q, K, V, O) plus two 128-value norm gains. RoPE, the softmax and the head reshapes own no weights.', href: explorerHref({ m: 'qwen3-8b', lv: 3, b: 0, mod: 'attention' }), cta: 'Open attention' },
      { title: 'Its block', text: 'A block adds two RMSNorms and the SwiGLU FFN (three 50-million-parameter matrices). The residual additions have no weights.', href: explorerHref({ m: 'qwen3-8b', lv: 4, b: 0 }), cta: 'Open block 0' },
      { title: 'All of them', text: `Where the ${A8.total.toLocaleString('en-US')} parameters sit: ${SHARES}.`, href: '#/parameters?m=qwen3-8b', cta: 'See the treemap' },
    ],
  },
  {
    id: 'neuron',
    title: 'What is a neuron here?',
    summary: 'Why “8B parameters” is not 8 billion neurons, and what a SwiGLU channel computes.',
    steps: [
      { title: 'Weights vs activations', text: 'Change the input and the outputs change while the six weights stay the same; change a weight and you have changed the model.', href: '#/labs/weight', cta: 'Try the tiny example' },
      { title: 'A gated channel', text: 'In Qwen3 the FFN “neuron” j uses two rows (gate and up), a SiLU, a multiplication, and a column of W_down. Qwen3-8B has 12,288 such channels per block.', href: '#/labs/unit', cta: 'Open the unit microscope' },
      { title: 'Many widths', text: 'There are 4,096 residual features, 32 query heads × 128 coordinates, 8 K/V heads × 128, and 12,288 FFN channels per block. Which of these you call “neurons” is a convention.', href: explorerHref({ m: 'qwen3-8b', lv: 4, b: 0 }), cta: 'See the widths in a block' },
      { title: 'No fixed meanings', text: 'Features are often distributed across many coordinates, and this site does not label any unit as “the English neuron”. Attention maps alone are not causal explanations.', href: '#/glossary', cta: 'Glossary' },
    ],
  },
  {
    id: 'token',
    title: 'How does one token move?',
    summary: 'From text to token id to embedding, through every block, to the next token.',
    steps: [
      { title: 'Text to ids', text: 'The chat template adds role markers, then byte-level BPE produces ids. These are real Qwen3 tokenizer outputs.', href: '#/pipeline', cta: 'See real tokenization' },
      { title: 'Embedding row', text: 'Each id selects one 4,096-number row of the embedding table. No position vector is added here.', href: explorerHref({ m: 'qwen3-8b', lv: 3, mod: 'embedding' }), cta: 'Open the embedding' },
      { title: 'Through the blocks', text: 'Walk a toy token through each block and see the exact vectors added at the two residual connections.', href: '#/labs/layers', cta: 'Layer traversal' },
      { title: 'Attention reads the past', text: 'The query compares with cached keys of earlier positions; future positions have probability exactly zero.', href: '#/labs/attention', cta: 'Attention lab' },
      { title: 'Prefill and decode', text: 'Prefill processes the prompt block by block; decoding then adds one token at a time using the K/V cache, and the result equals recomputation.', href: '#/labs/cache', cta: 'Cache lab' },
      { title: 'Choosing the next token', text: 'Logits → temperature → top-k → top-p → renormalize → seeded draw → append → repeat.', href: '#/labs/sampling', cta: 'Sampling lab' },
    ],
  },
  {
    id: 'moe',
    title: 'What changes in MoE?',
    summary: 'Qwen3-30B-A3B: same attention, 128 expert FFNs per block, 8 used per token.',
    steps: [
      { title: 'Same block, different FFN', text: 'Attention, norms and residuals are as in the dense model; the FFN sub-layer is replaced by a router and 128 experts.', href: explorerHref({ m: 'qwen3-30b-a3b', lv: 4, b: 0 }), cta: 'Open a MoE block' },
      { title: 'The router', text: 'All 128 router logits are computed for every token in every block; softmax, keep the top 8, renormalize to α.', href: '#/labs/router', cta: 'Router lab' },
      { title: 'An expert is an FFN', text: 'Each expert is an ordinary SwiGLU of width 768 with its own three matrices. No job description is attached to any expert id.', href: explorerHref({ m: 'qwen3-30b-a3b', lv: 3, b: 0, mod: 'moe' }), cta: 'See all 128' },
      { title: 'Stored vs used', text: '30.5B parameters stored, about 3.3B “activated” per token: a convention, not the number of scalars read or FLOPs.', href: '#/parameters?m=qwen3-30b-a3b', cta: 'The accounting' },
      { title: 'A fair comparison', text: 'The two reference models differ in width and depth too; the dimension-matched toy comparison isolates the FFN change.', href: '#/labs/compare', cta: 'Dense vs MoE' },
    ],
  },
];
