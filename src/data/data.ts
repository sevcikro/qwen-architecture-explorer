import { useEffect, useState } from 'react';
import { z } from 'zod';

// Lazily loaded, schema-validated static data. The core explorer never waits for these.

export const SampleSchema = z.object({
  id: z.string(),
  evidence: z.literal('actual-checkpoint-sample'),
  modelKey: z.string(),
  model: z.string(),
  revision: z.string().regex(/^[0-9a-f]{40}$/),
  shard: z.string(),
  tensor: z.string(),
  logicalShape: z.array(z.number().int().positive()),
  dtype: z.literal('BF16'),
  index: z.object({ rows: z.tuple([z.number(), z.number()]).optional(), cols: z.tuple([z.number(), z.number()]) }),
  conversion: z.string(),
  values: z.array(z.array(z.number())),
  rawSha256: z.string().regex(/^[0-9a-f]{64}$/),
  rawBytes: z.number().int().positive(),
  byteRange: z.string(),
  retrieved: z.string(),
  license: z.string(),
  note: z.string(),
});
export type CheckpointSample = z.infer<typeof SampleSchema>;
export const SamplesFileSchema = z.object({ budgetBytes: z.number(), bytesUsed: z.number(), samples: z.array(SampleSchema) });

export const TokExampleSchema = z.object({
  id: z.string(),
  kind: z.enum(['raw', 'chat']),
  text: z.string().optional(),
  messages: z.array(z.object({ role: z.string(), content: z.string() })).optional(),
  enable_thinking: z.boolean().optional(),
  renderedText: z.string(),
  ids: z.array(z.number().int()),
  tokens: z.array(z.string()),
  decodedPieces: z.array(z.string()),
  offsets: z.array(z.tuple([z.number().int(), z.number().int()])).optional(),
  special: z.array(z.boolean()),
});
export const TokFileSchema = z.object({
  evidence: z.literal('actual-tokenizer-output'),
  model: z.string(),
  revision: z.string(),
  tokenizerJsonSha256: z.string(),
  library: z.string(),
  chatTemplateSource: z.string(),
  generated: z.string(),
  baseVocabSize: z.number(),
  vocabWithAddedTokens: z.number(),
  addedTokens: z.array(z.object({ id: z.number(), content: z.string(), special: z.boolean() })),
  note: z.string(),
  examples: z.array(TokExampleSchema),
});
export type TokFile = z.infer<typeof TokFileSchema>;

export const OffsetsSchema = z.object({
  shards: z.array(z.string()),
  headerBytes: z.array(z.number().int()),
  t: z.record(z.string(), z.tuple([z.number().int(), z.number().int(), z.string(), z.array(z.number().int())])),
});
export type Offsets = z.infer<typeof OffsetsSchema>;

export const CatalogSchema = z.object({
  model: z.string(),
  revision: z.string(),
  retrieved: z.string(),
  method: z.string(),
  shards: z.array(z.object({ file: z.string(), headerBytes: z.number(), tensors: z.number() }).passthrough()),
  tensorCount: z.number(),
  headerTotalElements: z.number(),
  families: z.array(z.object({ family: z.string(), count: z.number(), dtype: z.array(z.string()), shapes: z.array(z.array(z.number())), elements: z.number(), shards: z.array(z.string()) })),
  layerShards: z.record(z.string(), z.array(z.number())),
  expertShardPlacement: z.record(z.string(), z.object({ gate: z.array(z.number()), up: z.array(z.number()), down: z.array(z.number()) })).optional(),
}).passthrough();
export type Catalog = z.infer<typeof CatalogSchema>;

const cache = new Map<string, Promise<unknown>>();

export function loadJson<T>(file: string, schema: z.ZodType<T>): Promise<T> {
  if (!cache.has(file)) {
    const url = `${import.meta.env.BASE_URL}data/${file}`;
    const p = fetch(url)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status} for ${file}`);
        return r.json();
      })
      .then((j) => {
        const parsed = schema.safeParse(j);
        if (!parsed.success) throw new Error(`${file} failed schema validation: ${parsed.error.issues[0]?.message}`);
        return parsed.data;
      });
    p.catch(() => cache.delete(file));
    cache.set(file, p);
  }
  return cache.get(file) as Promise<T>;
}

export type Loadable<T> = { status: 'idle' | 'loading' } | { status: 'ok'; data: T } | { status: 'error'; error: string };

export function useJson<T>(file: string | null, schema: z.ZodType<T>): Loadable<T> {
  const [s, set] = useState<Loadable<T>>({ status: file ? 'loading' : 'idle' });
  useEffect(() => {
    if (!file) {
      set({ status: 'idle' });
      return;
    }
    let live = true;
    set({ status: 'loading' });
    loadJson(file, schema).then(
      (data) => live && set({ status: 'ok', data }),
      (e: Error) => live && set({ status: 'error', error: e.message }),
    );
    return () => {
      live = false;
    };
  }, [file, schema]);
  return s;
}

export const useSamples = () => useJson('checkpoint-samples.json', SamplesFileSchema);
export const useTokenizerExamples = () => useJson('tokenizer-examples.json', TokFileSchema);
export const useOffsets = (modelId: string | null) => useJson(modelId === 'qwen3-8b' || modelId === 'qwen3-30b-a3b' ? `offsets-${modelId}.json` : null, OffsetsSchema);
export const useCatalog = (modelId: string | null) => useJson(modelId ? `catalog-${modelId}.json` : null, CatalogSchema);

/** Real sample covering a coordinate, if any. */
export function sampleFor(samples: CheckpointSample[] | undefined, tensor: string, row: number | undefined, col: number | undefined): { sample: CheckpointSample; value: number | null } | null {
  if (!samples) return null;
  const s = samples.find((x) => x.tensor === tensor);
  if (!s) return null;
  if (row === undefined) return { sample: s, value: null };
  if (s.logicalShape.length === 1) {
    const [c0, c1] = s.index.cols;
    return row >= c0 && row <= c1 ? { sample: s, value: s.values[0][row - c0] } : { sample: s, value: null };
  }
  const [r0, r1] = s.index.rows!;
  const [c0, c1] = s.index.cols;
  if (row < r0 || row > r1 || col === undefined || col < c0 || col > c1) return { sample: s, value: null };
  return { sample: s, value: s.values[row - r0][col - c0] };
}

/** File byte offset of element (r, c) inside its shard (NOT a GPU memory address). */
export function fileOffset(off: Offsets, tensor: string, r: number, c: number | null): { shard: string; byte: number; dtype: string; shape: number[] } | null {
  const t = off.t[tensor];
  if (!t) return null;
  const [si, start, dtype, shape] = t;
  const bytes = dtype === 'BF16' ? 2 : 4;
  const idx = shape.length === 1 ? r : r * shape[1] + (c ?? 0);
  return { shard: off.shards[si], byte: 8 + off.headerBytes[si] + start + idx * bytes, dtype, shape };
}
