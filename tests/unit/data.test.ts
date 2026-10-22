import { describe, expect, it } from 'vitest';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { OffsetsSchema, SamplesFileSchema, TokFileSchema, sampleFor } from '../../src/data/data';
import { QWEN3_30B_A3B, QWEN3_8B } from '../../src/models/profiles';
import { toBf16Bits, fromBf16Bits } from '../../src/math/bf16';
import { locateTensor } from '../../src/explorer/locate';

const json = (f: string) => JSON.parse(readFileSync(`public/data/${f}`, 'utf8'));

describe('real checkpoint samples: provenance, schema and checksum', () => {
  const file = SamplesFileSchema.parse(json('checkpoint-samples.json'));
  it('stayed within the 100 MB extraction budget', () => {
    expect(file.budgetBytes).toBe(100_000_000);
    expect(file.bytesUsed).toBeLessThan(file.budgetBytes);
    expect(file.samples.length).toBeGreaterThan(0);
  });
  it.each(file.samples.map((s) => [s.id, s] as const))('%s re-hashes to its recorded SHA-256 and matches the header', (_id, s) => {
    const p = s.modelKey === 'qwen3-8b' ? QWEN3_8B : QWEN3_30B_A3B;
    expect(s.revision).toBe(p.revision);
    // values were widened exactly from BF16, so the raw little-endian bytes can be rebuilt bit-for-bit
    const buf = Buffer.alloc(s.rawBytes);
    let o = 0;
    for (const row of s.values) for (const v of row) {
      const bits = toBf16Bits(v);
      expect(fromBf16Bits(bits)).toBe(v);
      buf.writeUInt16LE(bits, o);
      o += 2;
    }
    expect(o).toBe(s.rawBytes);
    expect(createHash('sha256').update(buf).digest('hex')).toBe(s.rawSha256);
    const off = OffsetsSchema.parse(json(`offsets-${s.modelKey}.json`));
    const [si, , dtype, shape] = off.t[s.tensor];
    expect(off.shards[si]).toBe(s.shard);
    expect(dtype).toBe(s.dtype);
    expect(shape).toEqual(s.logicalShape);
    expect(locateTensor(p, s.tensor)).not.toBeNull();
  });
  it('a sample only answers for its own tensor and covered indices', () => {
    const q = file.samples.find((s) => s.tensor === 'model.layers.0.self_attn.q_proj.weight' && s.modelKey === 'qwen3-8b')!;
    expect(sampleFor([q], q.tensor, 0, 0)!.value).toBe(0.000568389892578125);
    expect(sampleFor([q], q.tensor, 16, 0)!.value).toBeNull();
    expect(sampleFor([q], 'model.layers.1.self_attn.q_proj.weight', 0, 0)).toBeNull();
  });
});

describe('tokenizer examples', () => {
  const t = TokFileSchema.parse(json('tokenizer-examples.json'));
  it('vocabulary arithmetic matches the report and the config', () => {
    expect(t.baseVocabSize + t.addedTokens.length).toBe(t.vocabWithAddedTokens);
    expect(t.vocabWithAddedTokens).toBe(151_669);
    expect(QWEN3_8B.vocabRows - t.vocabWithAddedTokens).toBe(267);
    for (const e of t.examples) expect(e.ids.length).toBe(e.decodedPieces.length);
  });
});
