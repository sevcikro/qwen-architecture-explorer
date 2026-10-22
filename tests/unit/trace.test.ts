import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { JourneySchema, TraceSchema } from '../../src/data/data';
import { QWEN3_0_6B } from '../../src/models/profiles';
import { accounting } from '../../src/models/shapes';

const trace = TraceSchema.parse(JSON.parse(readFileSync('public/data/trace-qwen3-0.6b.json', 'utf8')));

describe('measured Qwen3-0.6B trace', () => {
  it('provenance matches the pinned profile and the checkpoint header', () => {
    expect(trace.revision).toBe(QWEN3_0_6B.revision);
    expect(trace.fileSha256['model.safetensors']).toBe('f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b'); // = Hub LFS hash
    const a = accounting(QWEN3_0_6B);
    expect(a.total).toBe(596_049_920);
    expect(a.total - a.embedding).toBe(440_467_456); // card: 0.44B non-embedding
    // the file additionally stores lm_head.weight, a byte-identical copy of the tied embedding
    expect(trace.checkpointHeader.has_lm_head).toBe(true);
    expect(trace.checkpointHeader.elements).toBe(a.total + a.embedding);
    expect(trace.config).toMatchObject({ num_hidden_layers: 28, hidden_size: 1024, num_attention_heads: 16, num_key_value_heads: 8, head_dim: 128, intermediate_size: 3072, tie_word_embeddings: true });
  });

  it('every prompt has internally consistent shapes', () => {
    for (const p of trace.prompts) {
      const T = p.ids.length;
      expect(p.pieces.length).toBe(T);
      expect(p.residualRms.blocks.length).toBe(28);
      p.residualRms.blocks.forEach((b) => expect(b.length).toBe(T));
      expect(p.residualFeaturesLastPos.blocks.length).toBe(28);
      p.residualFeaturesLastPos.blocks.forEach((b) => expect(b.length).toBe(32));
      for (const a of p.attention) {
        expect(a.kvHead).toBe(Math.floor(a.head / 2));
        expect(a.probs.length).toBe(T);
        a.probs.forEach((row) => expect(row.length).toBe(T));
      }
      const s = p.topNext.reduce((x, y) => x + y.p, 0);
      expect(s).toBeLessThanOrEqual(1.000001);
    }
  });

  it('attention rows are probability distributions with exactly zero on future positions', () => {
    for (const p of trace.prompts) for (const a of p.attention) a.probs.forEach((row, t) => {
      expect(row.reduce((x, y) => x + y, 0)).toBeCloseTo(1, 4);
      row.forEach((v, s) => (s > t ? expect(v).toBe(0) : expect(v).toBeGreaterThanOrEqual(0)));
    });
  });

  it('the statements printed on the trace page hold for the data', () => {
    const cap = trace.prompts.find((p) => p.id === 'raw-capital')!;
    expect(cap.topNext[0].piece).toBe(' Paris');
    expect(cap.topNext[0].p).toBeCloseTo(0.657, 3);
    expect(trace.prompts.find((p) => p.id === 'raw-slovak')!.topNext[0].piece).toBe(' Br');
    for (const p of trace.prompts) {
      const T = p.ids.length;
      // attention sink: in blocks 13 and 27 every recorded head gives most attention to token 0
      for (const a of p.attention.filter((x) => x.block !== 0)) expect(a.probs.slice(1).reduce((s, r) => s + r[0], 0) / (T - 1)).toBeGreaterThan(0.5);
      // massive activation: a feature in the last block reaches 140–270 while the RMS stays 12–21
      const last = p.residualFeaturesLastPos.blocks[27];
      const mx = Math.max(...last.map(Math.abs));
      expect(mx).toBeGreaterThan(130);
      expect(mx).toBeLessThan(280);
      const rms = p.residualRms.blocks[27][T - 1];
      expect(rms).toBeGreaterThan(11);
      expect(rms).toBeLessThan(22);
      // gates are not mostly closed; SiLU minimum is about −0.278
      for (const g of p.gateLastPos) {
        expect(g.silu.filter((v) => Math.abs(v) < 0.05).length / g.silu.length).toBeLessThan(0.5);
        g.silu.forEach((v) => expect(v).toBeGreaterThan(-0.2785));
      }
    }
  });
});

describe('recorded chat turn for the message-journey simulation', () => {
  const j = JourneySchema.parse(JSON.parse(readFileSync('public/data/journey-qwen3-0.6b.json', 'utf8')));
  const tok = JSON.parse(readFileSync('public/data/tokenizer-examples.json', 'utf8'));
  it('uses the pinned model, the same file as the trace, and the same prompt ids as the tokenizer example', () => {
    expect(j.revision).toBe('c1899de289a04d12100db370d81485cdf75e47ca');
    expect(j.modelSha256.startsWith('f47f7117')).toBe(true);
    const ex = tok.examples.find((e: { id: string }) => e.id === 'chat-nothink');
    expect(j.promptIds).toEqual(ex.ids);
    expect(j.promptPieces.length).toBe(j.promptIds.length);
  });
  it('is a greedy run that ends on <|im_end|>, and every step lists the chosen token first', () => {
    expect(j.stoppedOnEnd).toBe(true);
    expect(j.steps[j.steps.length - 1].id).toBe(151645);
    for (const s of j.steps) {
      expect(s.top.length).toBe(5);
      expect(s.top[0].piece).toBe(s.piece);
      expect(s.top[0].p).toBe(s.p);
      for (let i = 1; i < 5; i++) expect(s.top[i].p).toBeLessThanOrEqual(s.top[i - 1].p);
    }
    expect(j.answer.startsWith('A **parameter** is')).toBe(true);
  });
});

