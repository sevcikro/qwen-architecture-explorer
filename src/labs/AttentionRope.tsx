import { useState } from 'react';
import { useToy } from '../state/toyStore';
import { Evidence, NumberField, Section, Select, Slider, Tex, VectorView, fmtInt, fmtNum } from '../components/ui';
import { ToyControls, PromptEditor, tok } from './common';
import { CORE_PROFILES, QWEN3_8B } from '../models/profiles';
import { applyRope, ropeInvFreq } from '../math/ops';
import { dot } from '../math/linalg';

export function AttentionLab() {
  const { profile: p, trace, state } = useToy();
  const T = state.tokens.length;
  const [layer, setLayer] = useState(0);
  const [qpos, setQpos] = useState(T - 1);
  const [kpos, setKpos] = useState(0);
  const [head, setHead] = useState(0);
  const [ref, setRef] = useState(QWEN3_8B.id);
  const t = Math.min(qpos, T - 1), s = Math.min(kpos, T - 1), h = Math.min(head, p.hq - 1);
  const group = p.hq / p.hkv;
  const kv = Math.floor(h / group);
  const A = trace[t].layers[layer].attn;
  const raw = A.rawScores[h], scaled = A.scaled[h], probs = A.probs[h];
  const validMax = Math.max(...scaled.filter((_, i) => i <= t));
  const R = CORE_PROFILES.find((x) => x.id === ref)!;
  const kRot = (pos: number) => trace[pos].layers[layer].attn.kRot[kv];
  const vOf = (pos: number) => trace[pos].layers[layer].attn.v[kv];
  return (
    <>
      <Section title="4 · Attention lab" badges={<Evidence kind="simulation" />}>
        <ToyControls />
        <PromptEditor />
        <div className="row">
          <NumberField label="block" value={layer} min={0} max={p.layers - 1} step={1} onChange={(v) => setLayer(Math.round(v))} width={70} />
          <NumberField label="query position" value={t} min={0} max={T - 1} step={1} onChange={(v) => setQpos(Math.round(v))} width={80} />
          <NumberField label="key position" value={s} min={0} max={T - 1} step={1} onChange={(v) => setKpos(Math.round(v))} width={80} />
          <NumberField label={`query head (0–${p.hq - 1})`} value={h} min={0} max={p.hq - 1} step={1} onChange={(v) => setHead(Math.round(v))} width={90} />
        </div>
        <p className="muted" style={{ fontSize: '0.9rem' }}>Query head {h} uses K/V head {kv} (⌊{h} / {group}⌋). Selecting positions and heads is inspection only.</p>
        <div className="grid2">
          <div>
            <h3>Per key position for query “{tok(state.tokens[t])}” (position {t})</h3>
            <div className="table-wrap">
              <table className="t">
                <thead><tr><th>key pos</th><th>token</th><th className="num">q·k</th><th className="num">÷ √{p.headDim}</th><th>mask</th><th className="num">exp(s − max)</th><th className="num">probability</th></tr></thead>
                <tbody>
                  {raw.map((r, i) => {
                    const masked = i > t;
                    return (
                      <tr key={i} style={{ background: i === s ? 'color-mix(in srgb, var(--focus) 14%, transparent)' : undefined }}>
                        <td>{i}</td><td className="mono">{tok(state.tokens[i])}</td>
                        <td className="num">{fmtNum(r)}</td><td className="num">{fmtNum(scaled[i])}</td>
                        <td>{masked ? <strong>future · −∞</strong> : 'visible'}</td>
                        <td className="num">{masked ? '0' : fmtNum(Math.exp(scaled[i] - validMax))}</td>
                        <td className="num"><strong>{probs[i] === 0 ? '0' : fmtNum(probs[i])}</strong></td>
                      </tr>
                    );
                  })}
                  <tr><td colSpan={6}>sum over visible keys</td><td className="num">{fmtNum(probs.reduce((a, b) => a + b, 0))}</td></tr>
                </tbody>
              </table>
            </div>
            <p className="faint" style={{ fontSize: '0.84rem' }}>Stable softmax subtracts the largest visible score ({fmtNum(validMax)}) before exponentiating. Future positions get probability exactly 0. Policy for a row with no visible key: all zeros, flagged; never NaN, never renormalized out of nothing (cannot happen in causal self-attention, where a token always sees itself).</p>
          </div>
          <div>
            <h3>q · k for key position {s}</h3>
            {s > t ? <div className="callout warn">Key {s} is in the future of query {t}: masked, so this product is never used.</div> : (
              <>
                <VectorView label={`q (head ${h}, after q_norm and RoPE at position ${t})`} values={A.qRot[h]} />
                <VectorView label={`k (K/V head ${kv}, after k_norm and RoPE at position ${s})`} values={kRot(s)} />
                <VectorView label="elementwise products" values={A.qRot[h].map((x, i) => x * kRot(s)[i])} />
                <p className="mono" style={{ fontSize: '0.88rem' }}>sum = {fmtNum(dot(A.qRot[h], kRot(s)))} → ÷ √{p.headDim} = {fmtNum(scaled[s])} → probability {fmtNum(probs[s])}</p>
              </>
            )}
            <h3>Weighted sum of V</h3>
            {Array.from({ length: t + 1 }, (_, i) => <VectorView key={i} label={`${fmtNum(probs[i], 3)} × v[${i}] (“${tok(state.tokens[i])}”)`} values={vOf(i)} />)}
            <VectorView label={`head ${h} output z = Σ p_i v_i`} values={A.headOut[h]} />
            {p.hq > 1 && <p className="faint" style={{ fontSize: '0.84rem' }}>Query heads 0 and 1 share K/V head 0 but have different queries, so their probabilities and outputs differ: head 0 out {A.headOut[0].map((x) => fmtNum(x, 2)).join(', ')} vs head 1 out {A.headOut[1].map((x) => fmtNum(x, 2)).join(', ')}.</p>}
          </div>
        </div>
      </Section>
      <Section title={`Computed attention probabilities, head ${h}, block ${layer}`} badges={<Evidence kind="simulation" />}>
        <div className="grid2">
          <div>
            <p className="muted" style={{ fontSize: '0.88rem' }}>Actual computed values of this toy run (rows = query positions, columns = key positions). This is <strong>not</strong> an observed Qwen heatmap, and attention strength alone is not a complete causal explanation of model behavior.</p>
            <AttnMatrix T={T} values={(i, j) => trace[i].layers[layer].attn.probs[h][j]} sel={[t, s]} />
          </div>
          <div>
            <p className="muted" style={{ fontSize: '0.88rem' }}>Causal dependency structure (the same for every model, head and input): which keys a query may read at all.</p>
            <AttnMatrix T={T} values={(i, j) => (j <= i ? 1 : 0)} sel={[t, s]} binary />
          </div>
        </div>
      </Section>
      <Section title="The same operation at real size" badges={<><Evidence kind="verified" /><Evidence kind="derived" /></>}>
        <Select label="Reference model" value={ref} onChange={setRef} options={CORE_PROFILES.map((x) => ({ value: x.id, label: x.name }))} />
        <div className="table-wrap" style={{ marginTop: 8 }}>
          <table className="t">
            <tbody>
              <tr><td>W_Q</td><td className="mono">[Hq·dh, d] = [{fmtInt(R.hq * R.headDim)}, {fmtInt(R.d)}]</td></tr>
              <tr><td>W_K, W_V</td><td className="mono">[Hkv·dh, d] = [{fmtInt(R.hkv * R.headDim)}, {fmtInt(R.d)}]</td></tr>
              <tr><td>Q</td><td className="mono">[B, {R.hq}, Tq, {R.headDim}]</td></tr>
              <tr><td>K, V</td><td className="mono">[B, {R.hkv}, Tk, {R.headDim}]</td></tr>
              <tr><td>Logical scores</td><td className="mono">[B, {R.hq}, Tq, Tk]</td></tr>
              <tr><td>W_O</td><td className="mono">[d, Hq·dh] = [{fmtInt(R.d)}, {fmtInt(R.hq * R.headDim)}]</td></tr>
              <tr><td>GQA</td><td>{R.hq / R.hkv} query heads share each of the {R.hkv} K/V heads</td></tr>
              <tr><td>q/k norm gains</td><td>{R.headDim} + {R.headDim} per block, shared across heads</td></tr>
            </tbody>
          </table>
        </div>
        <GqaMap hq={R.hq} hkv={R.hkv} />
        <Tex block>{'A = \\operatorname{softmax}\\!\\Big(\\frac{QK^{\\top}}{\\sqrt{d_h}} + M_{\\text{causal}}\\Big),\\qquad Z = AV'}</Tex>
        <p className="muted" style={{ fontSize: '0.9rem' }}>Optimized kernels such as FlashAttention compute exactly this result in tiles with an online softmax, without storing the full Tq × Tk matrix; the square matrix shown above is the logical object, not a literal allocation every runtime makes.</p>
      </Section>
    </>
  );
}

function AttnMatrix({ T, values, sel, binary = false }: { T: number; values: (i: number, j: number) => number; sel: [number, number]; binary?: boolean }) {
  const cell = Math.min(34, Math.floor(300 / T));
  return (
    <svg width={cell * T + 30} height={cell * T + 30} role="img" aria-label={binary ? 'Causal mask: lower triangle allowed' : 'Attention probability matrix'}>
      {Array.from({ length: T }, (_, i) => (
        <g key={i}>
          <text x={22} y={30 + i * cell + cell / 2 + 4} fontSize={10} textAnchor="end" fill="var(--text-3)">{i}</text>
          <text x={30 + i * cell + cell / 2} y={20} fontSize={10} textAnchor="middle" fill="var(--text-3)">{i}</text>
          {Array.from({ length: T }, (_, j) => {
            const v = values(i, j);
            const future = j > i;
            return (
              <g key={j}>
                <rect x={30 + j * cell} y={26 + i * cell} width={cell - 2} height={cell - 2} rx={2}
                  fill={future ? 'var(--bg-2)' : binary ? 'color-mix(in srgb, var(--attn) 45%, transparent)' : `color-mix(in srgb, var(--attn) ${Math.round(v * 100)}%, var(--bg-2))`}
                  stroke={i === sel[0] && j === sel[1] ? 'var(--focus)' : 'var(--line-soft)'} strokeWidth={i === sel[0] && j === sel[1] ? 2 : 1} />
                {future && <text x={30 + j * cell + cell / 2 - 1} y={26 + i * cell + cell / 2 + 3} fontSize={9} textAnchor="middle" fill="var(--text-3)">×</text>}
                {!future && !binary && cell >= 30 && <text x={30 + j * cell + cell / 2 - 1} y={26 + i * cell + cell / 2 + 3} fontSize={8} textAnchor="middle" fill="var(--text)">{v.toFixed(2)}</text>}
              </g>
            );
          })}
        </g>
      ))}
    </svg>
  );
}

function GqaMap({ hq, hkv }: { hq: number; hkv: number }) {
  const g = hq / hkv;
  const w = Math.min(18, Math.floor(560 / hq));
  return (
    <svg width={hq * w + 20} height={86} role="img" aria-label={`${hq} query heads in ${hkv} groups of ${g}, each group sharing one key/value head`} style={{ maxWidth: '100%' }}>
      {Array.from({ length: hq }, (_, h) => (
        <g key={h}>
          <rect x={10 + h * w} y={6} width={w - 3} height={18} rx={3} fill="color-mix(in srgb, var(--attn) 30%, transparent)" stroke="var(--attn)" />
          <line x1={10 + h * w + (w - 3) / 2} y1={24} x2={10 + (Math.floor(h / g) * g + g / 2) * w - 1.5} y2={56} stroke="var(--text-3)" />
        </g>
      ))}
      {Array.from({ length: hkv }, (_, k) => (
        <rect key={k} x={10 + k * g * w} y={56} width={g * w - 3} height={18} rx={3} fill="color-mix(in srgb, var(--state) 30%, transparent)" stroke="var(--state)" />
      ))}
      <text x={10} y={84} fontSize={10} fill="var(--text-3)">top: {hq} query heads · bottom: {hkv} shared K/V heads (cached)</text>
    </svg>
  );
}

export function RopeLab() {
  const [dh, setDh] = useState(128);
  const [theta, setTheta] = useState(1_000_000);
  const [pair, setPair] = useState(0);
  const [m, setM] = useState(5);
  const [n, setN] = useState(2);
  const [shift, setShift] = useState(10);
  const inv = ropeInvFreq(dh, theta);
  const i = Math.min(pair, dh / 2 - 1);
  const w = inv[i];
  const q = [1, 0.35], k = [0.6, -0.8];
  const rot = (v: number[], pos: number) => { const a = pos * w; return [v[0] * Math.cos(a) - v[1] * Math.sin(a), v[1] * Math.cos(a) + v[0] * Math.sin(a)]; };
  const qm = rot(q, m), kn = rot(k, n), qm2 = rot(q, m + shift), kn2 = rot(k, n + shift);
  const d1 = qm[0] * kn[0] + qm[1] * kn[1], d2 = qm2[0] * kn2[0] + qm2[1] * kn2[1];
  // full-vector check with the implemented half-split convention
  const qv = Array.from({ length: dh }, (_, j) => Math.sin(j * 0.37 + 1)), kv = Array.from({ length: dh }, (_, j) => Math.cos(j * 0.21));
  const full1 = dot(applyRope(qv, m, inv), applyRope(kv, n, inv)), full2 = dot(applyRope(qv, m + shift, inv), applyRope(kv, n + shift, inv));
  const S = 110;
  const arrow = (v: number[], color: string, label: string) => (
    <g>
      <line x1={S} y1={S} x2={S + v[0] * 80} y2={S - v[1] * 80} stroke={color} strokeWidth={3} markerEnd="url(#ra)" />
      <text x={S + v[0] * 92} y={S - v[1] * 92} fill={color} fontSize={12} textAnchor="middle">{label}</text>
    </g>
  );
  return (
    <Section title="5 · RoPE lab: rotary position embedding" badges={<><Evidence kind="verified" title="Convention: half-split pairing and θ^(−2i/d) frequencies as implemented" /><Evidence kind="simulation" title="The 2-D vectors are illustrative" /></>}>
      <p>Qwen3 applies RoPE to each query and key head <strong>after</strong> q/k RMSNorm. Coordinate i is paired with i + d<sub>h</sub>/2 (the Transformers <code>rotate_half</code> convention, not adjacent pairs) and the pair is rotated by angle position × ω<sub>i</sub>, with ω<sub>i</sub> = θ<sup>−2i/d<sub>h</sub></sup>. V is not rotated, and nothing is added at the embedding: RoPE has no learned weights.</p>
      <Tex block>{"\\begin{pmatrix} q'_i \\\\ q'_{i+d/2} \\end{pmatrix} = \\begin{pmatrix} \\cos m\\omega_i & -\\sin m\\omega_i \\\\ \\sin m\\omega_i & \\cos m\\omega_i \\end{pmatrix} \\begin{pmatrix} q_i \\\\ q_{i+d/2} \\end{pmatrix}"}</Tex>
      <div className="row">
        <Select label="head dim" value={dh} onChange={(v) => { setDh(v); setPair(0); }} options={[{ value: 128, label: '128 (Qwen3-8B / 30B-A3B)' }, { value: 4, label: '4 (toy simulator)' }]} />
        <Select label="θ (rope_theta)" value={theta} onChange={setTheta} options={[{ value: 1_000_000, label: '1,000,000 (Qwen3 config)' }, { value: 10000, label: '10,000 (toy; original RoPE)' }]} />
        <Slider label={`pair i (coords ${i} & ${i + dh / 2})`} value={i} min={0} max={dh / 2 - 1} onChange={setPair} />
      </div>
      <div className="row">
        <Slider label="query position m" value={m} min={0} max={64} onChange={setM} />
        <Slider label="key position n" value={n} min={0} max={64} onChange={setN} />
        <Slider label="shift both by" value={shift} min={0} max={500} onChange={setShift} />
      </div>
      <div className="grid2">
        <svg viewBox="0 0 220 220" style={{ maxWidth: 300 }} role="img" aria-label="Rotation of a query pair and a key pair">
          <defs><marker id="ra" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" fill="var(--text-2)" /></marker></defs>
          <circle cx={S} cy={S} r={80} fill="none" stroke="var(--line)" />
          <line x1={20} y1={S} x2={200} y2={S} stroke="var(--line-soft)" /><line x1={S} y1={20} x2={S} y2={200} stroke="var(--line-soft)" />
          {arrow(q, 'var(--text-3)', 'q')}
          {arrow(qm, 'var(--attn)', `q@${m}`)}
          {arrow(kn, 'var(--router)', `k@${n}`)}
        </svg>
        <div>
          <p className="mono" style={{ fontSize: '0.88rem' }}>
            ω_{i} = {theta}^(−{2 * i}/{dh}) = {w.toExponential(4)} rad/position<br />
            wavelength = 2π/ω = {(2 * Math.PI / w).toExponential(3)} positions<br />
            angle at m={m}: {(m * w).toFixed(4)} rad
          </p>
          <p>2-D illustration: q·k at (m, n) = ({m}, {n}) is <strong>{d1.toFixed(6)}</strong>; shifting both by {shift} gives <strong>{d2.toFixed(6)}</strong>. The score depends only on the offset m − n = {m - n}.</p>
          <p className="faint" style={{ fontSize: '0.86rem' }}>Full {dh}-dimensional check with the implemented operator (synthetic vectors): {full1.toFixed(8)} vs {full2.toFixed(8)} (difference {Math.abs(full1 - full2).toExponential(2)}, floating-point rounding).</p>
          <div className="callout">During cached decoding the new token is rotated with its <strong>absolute</strong> position (= number of tokens already in the cache), while the cached keys keep the rotations they received at their own positions. The Prefill/decode lab tests that this reproduces full recomputation.</div>
          <p className="faint" style={{ fontSize: '0.85rem' }}>Context extension (YaRN, factor 4 to 131,072 tokens per the model card) rescales these frequencies at inference time; it is a runtime setting (rope_scaling), not enabled in the published config (rope_scaling: null).</p>
        </div>
      </div>
    </Section>
  );
}
