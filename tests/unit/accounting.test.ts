import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { QWEN3_30B_A3B, QWEN3_8B, TOY_DENSE, TOY_MOE, customFrom } from '../../src/models/profiles';
import { accounting, familyParams, packedExpertMapping, tensorFamilies, prod } from '../../src/models/shapes';

// Independent fixture: hand-written formulas from config numbers only (does not call tensorFamilies).
function independentTotal(c: { d: number; L: number; hq: number; hkv: number; dh: number; V: number; f?: number; E?: number; fe?: number }) {
  const attn = c.d * c.hq * c.dh + 2 * c.d * c.hkv * c.dh + c.hq * c.dh * c.d;
  const norms = 2 * c.d + 2 * c.dh;
  const ffn = c.f ? 3 * c.d * c.f : c.E! * 3 * c.d * c.fe! + c.E! * c.d;
  return 2 * c.V * c.d + c.d + c.L * (attn + norms + ffn);
}

describe('exact parameter accounting', () => {
  it('Qwen3-8B reproduces the brief table and 8,190,735,360 total', () => {
    const a = accounting(QWEN3_8B);
    expect(a.embedding).toBe(622_329_856);
    expect(a.head).toBe(622_329_856);
    expect(a.attentionPerBlock).toBe(41_943_040);
    expect(a.qkvoPerBlock).toEqual({ q: 16_777_216, k: 4_194_304, v: 4_194_304, o: 16_777_216 });
    expect(a.ffnPerBlock).toBe(150_994_944);
    expect(a.ffnPerBlock / 3).toBe(50_331_648);
    expect(a.routerPerBlock).toBe(0);
    expect(a.normsPerBlock).toBe(8_448);
    expect(a.perBlock).toBe(192_946_432);
    expect(a.finalNorm).toBe(4_096);
    expect(a.total).toBe(8_190_735_360);
    // model-card 'non-embedding 6.95B' excludes both vocabulary matrices
    expect(a.nonEmbedding).toBe(6_946_075_648);
    expect(independentTotal({ d: 4096, L: 36, hq: 32, hkv: 8, dh: 128, V: 151936, f: 12288 })).toBe(a.total);
  });

  it('Qwen3-30B-A3B reproduces the brief table and 30,532,122,624 total', () => {
    const a = accounting(QWEN3_30B_A3B);
    expect(a.embedding).toBe(311_164_928);
    expect(a.head).toBe(311_164_928);
    expect(a.attentionPerBlock).toBe(18_874_368);
    expect(a.oneExpert).toBe(4_718_592);
    expect(a.ffnPerBlock).toBe(603_979_776);
    expect(a.routerPerBlock).toBe(262_144);
    expect(a.normsPerBlock).toBe(4_352);
    expect(a.perBlock).toBe(623_120_640);
    expect(a.finalNorm).toBe(2_048);
    expect(a.total).toBe(30_532_122_624);
    expect(a.nonEmbedding).toBe(29_909_792_768); // card: 29.9B
    expect(a.selectedExpertsAllLayers).toBe(1_811_939_328);
    expect(a.activeIncludingFullEmbedding).toBe(3_353_032_704);
    expect(independentTotal({ d: 2048, L: 48, hq: 32, hkv: 4, dh: 128, V: 151936, E: 128, fe: 768 })).toBe(a.total);
  });

  it('matches the element counts read from the real pinned safetensors headers', () => {
    for (const [p, file] of [[QWEN3_8B, 'catalog-qwen3-8b.json'], [QWEN3_30B_A3B, 'catalog-qwen3-30b-a3b.json']] as const) {
      const cat = JSON.parse(readFileSync(`public/data/${file}`, 'utf8'));
      expect(cat.revision).toBe(p.revision);
      expect(cat.headerTotalElements).toBe(accounting(p).total);
      expect(cat.indexTotalSize).toBe(accounting(p).total * 2); // BF16 = 2 bytes
      expect(cat.indexMissingFromHeaders).toEqual([]);
      // every logical family maps to a real checkpoint family with identical shape and count
      for (const f of tensorFamilies(p)) {
        const famKey = f.pathTemplate.replace('{layer}', '{i}').replace('{expert}', '{i}');
        const real = cat.families.find((x: { family: string }) => x.family === famKey);
        expect(real, famKey).toBeTruthy();
        expect(real.shapes).toEqual([f.shape]);
        expect(real.count).toBe(f.multiplicity);
        expect(real.elements).toBe(familyParams(f));
        expect(real.dtype).toEqual(['BF16']);
      }
      expect(cat.families.length).toBe(tensorFamilies(p).length);
    }
  });

  it('30B-A3B maps residual 2048 -> Q width 4096 -> back to 2048 (head_dim is not d / Hq)', () => {
    const f = tensorFamilies(QWEN3_30B_A3B);
    const q = f.find((x) => x.key === 'q_proj')!;
    const o = f.find((x) => x.key === 'o_proj')!;
    expect(q.shape).toEqual([4096, 2048]);
    expect(o.shape).toEqual([2048, 4096]);
    expect(QWEN3_30B_A3B.d / QWEN3_30B_A3B.hq).toBe(64);
    expect(QWEN3_30B_A3B.headDim).toBe(128);
  });

  it('no projection biases, untied embeddings, q/k norms counted once per block (not per head)', () => {
    for (const p of [QWEN3_8B, QWEN3_30B_A3B]) {
      const fams = tensorFamilies(p);
      expect(fams.some((f) => f.pathTemplate.includes('bias'))).toBe(false);
      expect(p.linearBias).toBe(false);
      expect(p.tiedEmbeddings).toBe(false);
      expect(fams.find((f) => f.key === 'lm_head')).toBeTruthy();
      expect(prod(fams.find((f) => f.key === 'q_norm')!.shape)).toBe(128);
      expect(prod(fams.find((f) => f.key === 'k_norm')!.shape)).toBe(128);
    }
  });

  it('dense FFN and expert FFN obey 3*d*f', () => {
    expect(accounting(QWEN3_8B).ffnPerBlock).toBe(3 * 4096 * 12288);
    expect(accounting(QWEN3_30B_A3B).oneExpert).toBe(3 * 2048 * 768);
    expect(QWEN3_30B_A3B.configIntermediateSize).toBe(6144); // dense fallback field, not used by MoE blocks
  });

  it('packed runtime expert layout is a re-layout, counted once', () => {
    const m = packedExpertMapping(QWEN3_30B_A3B, 5)!;
    expect(m.runtime.gate).toContain('[5, 0:768, :]');
    expect(m.runtime.up).toContain('[5, 768:1536, :]');
    expect(m.runtime.down).toContain('of [128, 2048, 768]');
    const total = accounting(QWEN3_30B_A3B).total;
    // packed tensor sizes equal the sum of the logical per-expert tensors
    expect(128 * 1536 * 2048 + 128 * 2048 * 768).toBe(accounting(QWEN3_30B_A3B).ffnPerBlock);
    expect(total).toBe(30_532_122_624);
  });

  it('custom configurations never keep an official name', () => {
    const c = customFrom(QWEN3_8B, { layers: 12 });
    expect(c.status).toBe('custom');
    expect(c.name).not.toBe('Qwen3-8B');
    expect(c.repo).toBeUndefined();
    expect(accounting(c).total).toBe(2 * 622_329_856 + 4096 + 12 * 192_946_432);
    expect(customFrom(QWEN3_8B, { layers: 36 })).toBe(QWEN3_8B);
  });

  it('toy profiles are compute-matched: MoE active FFN width equals dense FFN width', () => {
    expect(TOY_MOE.moe!.topK * TOY_MOE.moe!.expertWidth).toBe(TOY_DENSE.ffnWidth);
    expect(TOY_MOE.d).toBe(TOY_DENSE.d);
  });
});
