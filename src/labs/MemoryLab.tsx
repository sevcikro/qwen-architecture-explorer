import { useState } from 'react';
import { Cite, Evidence, NumberField, Section, Select, Slider, Tex, fmtB, fmtInt } from '../components/ui';
import { CORE_PROFILES, TOY_DENSE } from '../models/profiles';
import { accounting, attentionOpsPerToken, matmulOpsPerToken } from '../models/shapes';
import { GB, GiB, fmtBytes, hybridStateBytes, kvCacheBytes, quantizedEstimate, trainingMemory, weightBytes } from '../math/memory';
import { QWEN38_COMPONENTS, QWEN38_STATE } from '../models/hybrid';
import { forwardFull, initToyWeights, DEFAULT_PROMPT } from '../math/toy';

export function MemoryLab() {
  const [mid, setMid] = useState(CORE_PROFILES[0].id);
  const p = CORE_PROFILES.find((x) => x.id === mid)!;
  const a = accounting(p);
  const [bits, setBits] = useState(16);
  const [group, setGroup] = useState(64);
  const [scaleBits, setScaleBits] = useState(16);
  const [zp, setZp] = useState(0);
  const [keepEmb, setKeepEmb] = useState(true);
  const [batch, setBatch] = useState(1);
  const [ctx, setCtx] = useState(32768);
  const [kvBytes, setKvBytes] = useState(2);
  const raw = weightBytes(a.total, bits);
  const q = quantizedEstimate(p, { bits, groupSize: group, scaleBits, zeroPointBits: zp, keepUnquantized: { embedding: keepEmb, head: keepEmb, norms: true }, unquantizedBits: 16 });
  const kv = kvCacheBytes(p, { batch, tokens: ctx, bytesPerScalar: kvBytes });
  const tm = trainingMemory(a.total, { weightBytes: 2, gradBytes: 2, adamStateBytes: 4, masterCopyBytes: 4 });
  const [measure, setMeasure] = useState<string | null>(null);
  const hybrid = hybridStateBytes(QWEN38_STATE, { tokens: ctx, batch, kvBytes, stateBytes: 4 });
  const q38total = QWEN38_COMPONENTS.reduce((s, c) => s + c.elements, 0);
  return (
    <>
      <Section title="12 · Memory lab" badges={<><Evidence kind="derived" /></>}>
        <Select label="Model" value={mid} onChange={setMid} options={CORE_PROFILES.map((x) => ({ value: x.id, label: x.name }))} />
        <div className="grid2" style={{ marginTop: 10 }}>
          <div className="card">
            <h3>Weights</h3>
            <Tex block>{'\\text{weight bytes} = N_{params} \\times \\text{bits} / 8'}</Tex>
            <div className="row">
              <Select label="bits per weight" value={bits} onChange={setBits} options={[{ value: 32, label: '32 (FP32)' }, { value: 16, label: '16 (BF16 — as published)' }, { value: 8, label: '8 (FP8 / INT8)' }, { value: 4, label: '4 (4-bit)' }]} />
            </div>
            <p className="mono">{fmtInt(a.total)} × {bits} / 8 = <strong>{fmtBytes(raw)}</strong></p>
            {bits < 16 && (
              <>
                <p className="faint" style={{ fontSize: '0.86rem' }}>Ideal payload only. Real quantized formats add scales (and sometimes zero-points) per group, often keep some tensors at higher precision, and have container overhead. Quantization changes how numbers are stored, not how many parameters the architecture has.</p>
                <div className="row">
                  <NumberField label="group size" value={group} min={8} step={8} width={90} onChange={(v) => setGroup(Math.max(8, Math.round(v)))} />
                  <Select label="scale bits" value={scaleBits} onChange={setScaleBits} options={[{ value: 16, label: '16' }, { value: 32, label: '32' }, { value: 8, label: '8' }]} />
                  <Select label="zero-point bits" value={zp} onChange={setZp} options={[{ value: 0, label: '0 (symmetric)' }, { value: bits, label: `${bits} (asymmetric)` }]} />
                  <label className="field inline"><input type="checkbox" checked={keepEmb} onChange={(e) => setKeepEmb(e.target.checked)} /> keep embedding + head at 16-bit</label>
                </div>
                <dl className="kv" style={{ marginTop: 6 }}>
                  <dt>quantized payload</dt><dd>{fmtBytes(q.payloadBytes)} ({fmtInt(q.quantizedParams)} params)</dd>
                  <dt>scale/zero metadata</dt><dd>{fmtBytes(q.metadataBytes)}</dd>
                  <dt>unquantized tensors</dt><dd>{fmtBytes(q.unquantizedBytes)} ({fmtInt(q.unquantizedParams)} params at 16-bit)</dd>
                  <dt>estimate</dt><dd><strong>{fmtBytes(q.totalBytes)}</strong> — not a guaranteed GGUF or Safetensors file size</dd>
                </dl>
                <p className="faint" style={{ fontSize: '0.82rem' }}>GGUF is a container format; one GGUF file can hold many different quantization types.<Cite ids={['gguf']} /></p>
              </>
            )}
            {p.kind === 'moe' && <div className="callout">Only ~3.3B parameters are “activated” per token, but all {fmtB(a.total)} must be stored and reachable, because different tokens and blocks pick different experts.</div>}
          </div>
          <div className="card">
            <h3>K/V cache (ordinary full-attention Qwen3 blocks)</h3>
            <Tex block>{'\\text{KV bytes} = 2 \\cdot B \\cdot L \\cdot T \\cdot H_{kv} \\cdot d_h \\cdot \\text{bytes}'}</Tex>
            <div className="row">
              <Slider label="batch B" value={batch} min={1} max={64} onChange={setBatch} />
              <Select label="context T" value={ctx} onChange={setCtx} options={[1024, 4096, 8192, 32768, 40960, 131072].map((v) => ({ value: v, label: fmtInt(v) }))} />
              <Select label="cache precision" value={kvBytes} onChange={setKvBytes} options={[{ value: 2, label: 'BF16 / FP16 (2 B)' }, { value: 1, label: 'FP8 (1 B)' }, { value: 4, label: 'FP32 (4 B)' }]} />
            </div>
            <p className="mono">2 × {batch} × {p.layers} × {fmtInt(ctx)} × {p.hkv} × {p.headDim} × {kvBytes} = <strong>{fmtBytes(kv)}</strong></p>
            <p className="faint" style={{ fontSize: '0.86rem' }}>K/V only: excludes allocator overhead, paging granularity, activations and kernel workspaces. Uses K/V heads ({p.hkv}), not query heads ({p.hq}). Weight precision and cache precision are independent settings.{ctx > p.context.native && ` ${fmtInt(ctx)} exceeds the native ${fmtInt(p.context.native)}; the model card describes YaRN (factor 4) for up to ${fmtInt(p.context.yarn!)}.`}</p>
            <h3>Training (persistent state only)</h3>
            <p className="mono" style={{ fontSize: '0.86rem' }}>BF16 weights {fmtBytes(tm.weights)} + BF16 grads {fmtBytes(tm.gradients)} + Adam m, v in FP32 {fmtBytes(tm.optimizerStates)} + FP32 master copy {fmtBytes(tm.masterWeights)} = <strong>{fmtBytes(tm.total)}</strong></p>
            <p className="faint" style={{ fontSize: '0.84rem' }}>One common mixed-precision layout (16 bytes/parameter); others exist. Activation memory for backpropagation depends on batch, sequence length and recomputation strategy and is not included.</p>
          </div>
        </div>
      </Section>
      <Section title="Compute, traffic and what is architecture vs serving choice" badges={<Evidence kind="derived" />}>
        <div className="table-wrap"><table className="t"><tbody>
          <tr><td>Matrix-multiply ops per generated token (2 × active matmul params, excl. attention scores)</td><td className="num">{fmtB(matmulOpsPerToken(p))}</td></tr>
          <tr><td>Attention score + value ops for one token over T = {fmtInt(ctx)} earlier tokens</td><td className="num">{fmtB(attentionOpsPerToken(p, ctx))}</td></tr>
          <tr><td>Weight bytes that must be read per decode step (BF16, batch 1, dense reading of active weights)</td><td className="num">{p.kind === 'dense' ? fmtBytes(a.total * 2) : `${fmtBytes((a.activeEmbeddingRowOnly!) * 2)} (selected experts only)`}</td></tr>
        </tbody></table></div>
        <ul style={{ paddingLeft: 18, fontSize: '0.92rem' }}>
          <li>Parameter counts, operation counts, memory traffic, measured latency and hardware utilization are different quantities; this page shows only the first two plus a traffic estimate. No tokens-per-second figure is claimed.</li>
          <li><strong>Tiling and accumulation.</strong> GPU kernels split matrices into tiles that fit fast on-chip memory and usually accumulate BF16 products in FP32. FlashAttention applies the same idea to attention with an online softmax.<Cite ids={['flashattn']} /></li>
          <li><strong>Parallelism.</strong> Tensor parallelism splits individual matrices across devices; pipeline parallelism splits blocks; expert parallelism places experts on different devices and moves token activations to them (all-to-all).<Cite ids={['switch']} /> These are serving/training implementation choices, not part of the model architecture.</li>
        </ul>
        <div className="row">
          <button onClick={() => {
            const w = initToyWeights(TOY_DENSE, 1);
            const runs = 200;
            const t0 = performance.now();
            for (let i = 0; i < runs; i++) forwardFull(TOY_DENSE, w, DEFAULT_PROMPT);
            const ms = (performance.now() - t0) / runs;
            setMeasure(`Measured just now in this browser: toy dense forward pass over ${DEFAULT_PROMPT.length} tokens took ${ms.toFixed(3)} ms on average (${runs} runs). Conditions: ${navigator.userAgent}; ${navigator.hardwareConcurrency ?? '?'} logical CPUs reported. A tiny float64 JavaScript simulator; says nothing about Qwen speed on any hardware.`);
          }}>Measure the toy simulator in this browser</button>
        </div>
        {measure && <p className="callout" style={{ fontSize: '0.86rem' }}>{measure}</p>}
      </Section>
      <Section title="Hybrid model state: Qwen3.8-Flash-Next" badges={<><Evidence kind="derived" /><Evidence kind="unknown" title="State dtype is a runtime choice" /></>}>
        <p>Derived from the pinned implementation and config: only the 12 QSA blocks keep per-token K/V (2 K/V heads × 256) plus 128-wide indexer keys; the 36 Gated DeltaNet blocks keep a fixed-size recurrent state [48 heads × 128 × 128] plus short convolution state; the n-gram tables (51.2B parameters) are weights, not state. Uses the context and batch above; KV precision as selected; recurrent state assumed FP32 (config mamba_ssm_dtype is float32; a runtime may differ).</p>
        <div className="table-wrap"><table className="t"><tbody>
          <tr><td>QSA K/V per token (12 blocks)</td><td className="num">{fmtInt(hybrid.kvPerToken)} B</td></tr>
          <tr><td>QSA indexer keys per token</td><td className="num">{fmtInt(hybrid.indexerPerToken)} B</td></tr>
          <tr><td>Growing state at T = {fmtInt(ctx)}, batch {batch}</td><td className="num">{fmtBytes(hybrid.growing)}</td></tr>
          <tr><td>Fixed recurrent state (36 GDN blocks)</td><td className="num">{fmtBytes(hybrid.recurrent)}</td></tr>
          <tr><td>Convolution states (approximate)</td><td className="num">{fmtBytes(hybrid.conv)}</td></tr>
          <tr><td>Stored checkpoint elements (all components)</td><td className="num">{fmtInt(q38total)} ≈ {(q38total * 2 / GB).toFixed(1)} GB = {(q38total * 2 / GiB).toFixed(1)} GiB at BF16</td></tr>
        </tbody></table></div>
        <p className="faint" style={{ fontSize: '0.84rem' }}>So total state still grows with context (attention blocks), just more slowly than if all 48 blocks were full attention; the model as a whole is not constant-memory. MTP and vision components are excluded from the per-token state above.</p>
      </Section>
    </>
  );
}
