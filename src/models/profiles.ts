import { z } from 'zod';

// ---------------------------------------------------------------------------
// Core decoder profiles (ordinary Qwen3 dense / MoE family and the toy simulator)
// ---------------------------------------------------------------------------

export const MoeSchema = z.object({
  numExperts: z.number().int().positive(),
  topK: z.number().int().positive(),
  expertWidth: z.number().int().positive(),
  sharedExperts: z.number().int().min(0),
  normTopkProb: z.boolean(),
});

export const CoreProfileSchema = z
  .object({
    id: z.string().min(1),
    name: z.string().min(1),
    kind: z.enum(['dense', 'moe']),
    status: z.enum(['official', 'toy', 'custom']),
    repo: z.string().optional(),
    revision: z.string().optional(),
    hfClass: z.string().optional(),
    d: z.number().int().positive(),
    layers: z.number().int().positive(),
    hq: z.number().int().positive(),
    hkv: z.number().int().positive(),
    headDim: z.number().int().positive(),
    vocabRows: z.number().int().positive(),
    ffnWidth: z.number().int().positive().nullable(), // dense intermediate_size actually used by blocks
    configIntermediateSize: z.number().int().positive().nullable(), // raw config field (may be unused)
    moe: MoeSchema.nullable(),
    tiedEmbeddings: z.boolean(),
    linearBias: z.boolean(),
    rmsEps: z.number().positive(),
    ropeTheta: z.number().positive(),
    context: z.object({ native: z.number().int(), yarn: z.number().int().nullable(), maxPositionEmbeddings: z.number().int() }),
    tokenizerVocab: z.number().int().nullable(),
    advertised: z.object({ total: z.string(), activated: z.string().nullable(), nonEmbedding: z.string().nullable() }).nullable(),
    dtype: z.string(),
    sources: z.array(z.string()),
  })
  .refine((p) => p.hq % p.hkv === 0, { message: 'query heads must be a multiple of K/V heads' })
  .refine((p) => (p.kind === 'dense') === (p.moe === null), { message: 'dense profiles have no MoE block and vice versa' })
  .refine((p) => p.kind === 'moe' || p.ffnWidth !== null, { message: 'dense profiles need an FFN width' })
  .refine((p) => p.moe === null || p.moe.topK <= p.moe.numExperts, { message: 'top-k cannot exceed expert count' });

export type CoreProfile = z.infer<typeof CoreProfileSchema>;

export const QWEN3_8B: CoreProfile = CoreProfileSchema.parse({
  id: 'qwen3-8b',
  name: 'Qwen3-8B',
  kind: 'dense',
  status: 'official',
  repo: 'Qwen/Qwen3-8B',
  revision: 'b968826d9c46dd6066d109eabc6255188de91218',
  hfClass: 'Qwen3ForCausalLM',
  d: 4096,
  layers: 36,
  hq: 32,
  hkv: 8,
  headDim: 128,
  vocabRows: 151936,
  ffnWidth: 12288,
  configIntermediateSize: 12288,
  moe: null,
  tiedEmbeddings: false,
  linearBias: false,
  rmsEps: 1e-6,
  ropeTheta: 1_000_000,
  context: { native: 32768, yarn: 131072, maxPositionEmbeddings: 40960 },
  tokenizerVocab: 151669,
  advertised: { total: '8.2B', activated: null, nonEmbedding: '6.95B' },
  dtype: 'BF16',
  sources: ['q8-config', 'q8-card', 'q8-index', 'tf-qwen3', 'qwen3-report'],
});

export const QWEN3_30B_A3B: CoreProfile = CoreProfileSchema.parse({
  id: 'qwen3-30b-a3b',
  name: 'Qwen3-30B-A3B',
  kind: 'moe',
  status: 'official',
  repo: 'Qwen/Qwen3-30B-A3B',
  revision: 'ad44e777bcd18fa416d9da3bd8f70d33ebb85d39',
  hfClass: 'Qwen3MoeForCausalLM',
  d: 2048,
  layers: 48,
  hq: 32,
  hkv: 4,
  headDim: 128,
  vocabRows: 151936,
  ffnWidth: null,
  configIntermediateSize: 6144,
  moe: { numExperts: 128, topK: 8, expertWidth: 768, sharedExperts: 0, normTopkProb: true },
  tiedEmbeddings: false,
  linearBias: false,
  rmsEps: 1e-6,
  ropeTheta: 1_000_000,
  context: { native: 32768, yarn: 131072, maxPositionEmbeddings: 40960 },
  tokenizerVocab: 151669,
  advertised: { total: '30.5B', activated: '3.3B', nonEmbedding: '29.9B' },
  dtype: 'BF16',
  sources: ['q30-config', 'q30-card', 'q30-index', 'tf-qwen3moe', 'tf-convmap', 'qwen3-report'],
});

// Toy simulator: deliberately tiny. Its counts are NOT Qwen's (4 experts / top-2, not 128 / top-8).
export const TOY_DENSE: CoreProfile = CoreProfileSchema.parse({
  id: 'toy-dense',
  name: 'Toy simulator (dense)',
  kind: 'dense',
  status: 'toy',
  d: 8,
  layers: 2,
  hq: 2,
  hkv: 1,
  headDim: 4,
  vocabRows: 12,
  ffnWidth: 16,
  configIntermediateSize: 16,
  moe: null,
  tiedEmbeddings: false,
  linearBias: false,
  rmsEps: 1e-6,
  ropeTheta: 10000,
  context: { native: 16, yarn: null, maxPositionEmbeddings: 16 },
  tokenizerVocab: 12,
  advertised: null,
  dtype: 'float64 (simulator)',
  sources: ['tf-qwen3'],
});

export const TOY_MOE: CoreProfile = CoreProfileSchema.parse({
  ...TOY_DENSE,
  id: 'toy-moe',
  name: 'Toy simulator (MoE)',
  kind: 'moe',
  ffnWidth: null,
  configIntermediateSize: null,
  // 4 experts of width 8 with top-2: active FFN width 16 = the dense toy's FFN width (compute-matched).
  moe: { numExperts: 4, topK: 2, expertWidth: 8, sharedExperts: 0, normTopkProb: true },
  sources: ['tf-qwen3moe'],
});

export const CORE_PROFILES: CoreProfile[] = [QWEN3_8B, QWEN3_30B_A3B];
export const ALL_CORE: CoreProfile[] = [QWEN3_8B, QWEN3_30B_A3B, TOY_DENSE, TOY_MOE];

export function coreById(id: string): CoreProfile | undefined {
  return ALL_CORE.find((p) => p.id === id);
}

/** A slider change never mutates an official preset: it produces a clearly named custom configuration. */
export function customFrom(base: CoreProfile, patch: Partial<CoreProfile>): CoreProfile {
  const merged = { ...base, ...patch } as CoreProfile;
  const changed = (Object.keys(patch) as (keyof CoreProfile)[]).some((k) => JSON.stringify(patch[k]) !== JSON.stringify(base[k]));
  if (!changed) return base;
  return CoreProfileSchema.parse({
    ...merged,
    id: `custom-from-${base.id}`,
    name: `Custom educational configuration (based on ${base.name})`,
    status: 'custom',
    repo: undefined,
    revision: undefined,
    advertised: null,
  });
}
