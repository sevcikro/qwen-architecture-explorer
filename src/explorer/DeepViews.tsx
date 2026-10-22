import { useMemo, useState } from 'react';
import type { CoreProfile } from '../models/profiles';
import { packedExpertMapping, prod, tensorName, type TensorFamily } from '../models/shapes';
import { locateTensor } from './locate';
export { locateTensor };
import { Evidence, ExportButton, NumberField, SimBanner, Tex, VectorView, fmtInt, fmtNum, shapeStr } from '../components/ui';
import { fileOffset, sampleFor, useCatalog, useOffsets, useSamples } from '../data/data';
import { useToy } from '../state/toyStore';
import { explorerHref, type ExplorerState } from '../state/route';
import { toyCounterpart, toyTrace, toyValue, unitInput } from './toyCounterpart';
import { bf16Parts, toBf16Bits, isExactBf16 } from '../math/bf16';
import { forwardFull, toyTensor, withEditedWeight, TOY_VOCAB } from '../math/toy';
import { silu, sigmoid } from '../math/ops';
import { dot, maxAbsDiff } from '../math/linalg';
import { kvBytesPerToken, fmtBytes } from '../math/memory';
import { accounting } from '../models/shapes';

const TILE = 16;



interface DeepProps { p: CoreProfile; f: TensorFamily; s: ExplorerState; go: (s: Partial<ExplorerState>) => void }

function useRealValues(p: CoreProfile, name: string) {
  const samples = useSamples();
  const list = samples.status === 'ok' ? samples.data.samples : undefined;
  return { samples, list, has: !!list?.some((x) => x.tensor === name && x.modelKey === p.id) };
}

/** LEVEL 2 — matrix / vector tile with exact indices and an extent indicator. Never renders one element per real parameter. */
export function MatrixView({ p, f, s, go }: DeepProps) {
  const toyApi = useToy();
  const rows = f.shape[0], cols = f.shape.length > 1 ? f.shape[1] : 1;
  const isVec = f.shape.length === 1;
  const name = tensorName(f, s.b ?? 0, s.e ?? 0);
  const selR = s.r ?? 0, selC = s.c ?? 0;
  const [r0, setR0] = useState(() => Math.floor(selR / TILE) * TILE);
  const [c0, setC0] = useState(() => Math.floor((isVec ? selR : selC) / TILE) * TILE);
  const real = useRealValues(p, name);
  const isToy = p.status === 'toy';
  const tt = useMemo(() => (isToy ? toyTrace(p, toyApi.state.seed, toyApi.state.tokens, { variant: toyApi.state.variant, weights: toyApi.weights }) : null), [isToy, p, toyApi.state.seed, toyApi.state.tokens, toyApi.state.variant, toyApi.weights]);
  const catalog = useCatalog(p.status === 'official' ? p.id : null);
  const offsets = useOffsets(p.status === 'official' ? p.id : null);
  const value = (r: number, c: number): { v: number | null; kind: 'toy' | 'sample' | 'none' } => {
    if (isToy && tt) {
      const t = toyTensor(tt.w, name);
      return { v: isVec ? (t as number[])[c] : (t as number[][])[r][c], kind: 'toy' };
    }
    const hit = sampleFor(real.list?.filter((x) => x.modelKey === p.id), name, isVec ? c : r, isVec ? undefined : c);
    return hit && hit.value !== null ? { v: hit.value, kind: 'sample' } : { v: null, kind: 'none' };
  };
  const nr = isVec ? 1 : Math.min(TILE, rows - r0), nc = Math.min(TILE, cols - c0);
  const fo = offsets.status === 'ok' ? fileOffset(offsets.data, name, isVec ? c0 : r0, isVec ? null : c0) : null;
  const mapping = f.scope === 'expert' ? packedExpertMapping(p, s.e ?? 0) : null;
  const shard = catalog.status === 'ok' && f.scope !== 'model' ? catalog.data.layerShards[String(s.b ?? 0)] : null;
  const anyReal = !isToy && Array.from({ length: nr }, (_, i) => Array.from({ length: nc }, (_, j) => value(r0 + i, c0 + j).kind === 'sample')).flat().some(Boolean);
  const mm = 140;
  const bw = Math.max(3, (nc / cols) * mm), bh = isVec ? 10 : Math.max(3, (nr / rows) * mm);
  return (
    <div className="page wide" style={{ paddingTop: 12 }}>
      <div className="row" style={{ justifyContent: 'space-between' }}>
        <h2 style={{ margin: 0 }}>{f.label} <span className="faint mono" style={{ fontSize: '0.85rem' }}>{name}</span></h2>
        <div className="badge-row">
          <Evidence kind={isToy ? 'simulation' : 'verified'} title="Shape, dtype and name" />
          {anyReal && <Evidence kind="sample" />}
          {!isToy && !anyReal && (real.samples.status === 'loading' ? <span className="coverage">checking real samples…</span> : <Evidence kind="unknown" title="Numerical values in this window are not loaded" />)}
        </div>
      </div>
      {isToy && <SimBanner />}
      <div className="grid2" style={{ marginTop: 10 }}>
        <div className="card">
          <dl className="kv">
            <dt>Logical shape</dt><dd className="mono">{shapeStr(f.shape)} — {f.axes.join(' × ')}</dd>
            <dt>Parameters</dt><dd className="mono">{f.shape.map(fmtInt).join(' × ')} = {fmtInt(prod(f.shape))}</dd>
            <dt>Storage dtype</dt><dd>{p.status === 'official' ? `${p.dtype} (2 bytes per value) in the checkpoint` : 'float64 in the browser simulator'}</dd>
            <dt>Orientation</dt><dd>{isVec ? 'vector: one gain per feature' : `PyTorch [out, in]: row = output coordinate, column = input feature; Y = X Wᵀ`}</dd>
            {p.status === 'official' && <><dt>Checkpoint</dt><dd className="mono">{fo ? `${fo.shard} · bytes ${fmtInt(fo.byte)}…` : shard ? `shard(s) ${shard.join(', ')}` : offsets.status === 'loading' ? 'loading byte map…' : 'see catalog'}</dd></>}
            <dt>Implementation</dt><dd className="mono">{f.symbol}</dd>
          </dl>
          {!isVec && (
            <p className="muted" style={{ fontSize: '0.88rem' }}>
              Batched use: for input X of shape [B, T, {fmtInt(f.shape[1])}] the same matrix is applied at every one of the B·T token positions, producing [B, T, {fmtInt(f.shape[0])}]. The weights are stored once; the products are activations.
            </p>
          )}
          {mapping && (
            <details>
              <summary>Logical expert matrices ↔ physical tensors</summary>
              <table className="t"><tbody>
                <tr><th>Logical</th><th>Checkpoint file tensor</th><th>Runtime (Transformers, packed)</th></tr>
                {(['gate', 'up', 'down'] as const).map((k) => (
                  <tr key={k}><td>{k}</td><td className="mono" style={{ fontSize: '0.78rem' }}>{mapping.checkpoint[k].replace('{layer}', String(s.b ?? 0))}</td><td className="mono" style={{ fontSize: '0.78rem' }}>{mapping.runtime[k]}</td></tr>
                ))}
              </tbody></table>
              <p className="faint" style={{ fontSize: '0.82rem' }}>{mapping.note}</p>
            </details>
          )}
        </div>
        <div className="card">
          <div className="row" style={{ justifyContent: 'space-between' }}>
            <strong>Tile window</strong>
            <span className="faint" style={{ fontSize: '0.82rem' }}>showing {isVec ? `elements ${c0}–${c0 + nc - 1}` : `rows ${r0}–${r0 + nr - 1}, cols ${c0}–${c0 + nc - 1}`} of {shapeStr(f.shape)}</span>
          </div>
          <div className="row" style={{ marginTop: 6 }}>
            {!isVec && <NumberField label="first row" value={r0} min={0} max={Math.max(0, rows - 1)} step={TILE} onChange={(v) => setR0(Math.min(Math.floor(v / TILE) * TILE, Math.max(0, rows - 1)))} width={100} />}
            <NumberField label={isVec ? 'first element' : 'first column'} value={c0} min={0} max={Math.max(0, cols - 1)} step={TILE} onChange={(v) => setC0(Math.min(Math.floor(v / TILE) * TILE, Math.max(0, cols - 1)))} width={100} />
            <svg width={mm + 4} height={(isVec ? 12 : mm) + 4} role="img" aria-label={`Extent indicator: the window covers ${nr * nc} of ${fmtInt(prod(f.shape))} values`}>
              <rect x={2} y={2} width={mm} height={isVec ? 10 : mm} fill="var(--bg-2)" stroke="var(--line)" />
              <rect x={2 + (c0 / cols) * mm} y={2 + (isVec ? 0 : (r0 / rows) * mm)} width={bw} height={bh} fill="var(--focus)" />
            </svg>
          </div>
          <p className="faint" style={{ fontSize: '0.8rem', margin: '4px 0' }}>Rendering limit: at most {TILE}×{TILE} cells at a time (no element is created per real parameter).</p>
        </div>
      </div>
      <div className="card" style={{ marginTop: 12, overflowX: 'auto' }}>
        <div className="matrix-grid" style={{ gridTemplateColumns: `52px repeat(${nc}, minmax(34px, 1fr))` }} role="grid" aria-label={`${f.label} values, window`}>
          <div className="cell hdr" role="columnheader">{isVec ? 'i' : 'r \\ c'}</div>
          {Array.from({ length: nc }, (_, j) => <div key={j} className="cell hdr" role="columnheader">{c0 + j}</div>)}
          {Array.from({ length: nr }, (_, i) => {
            const r = isVec ? 0 : r0 + i;
            return [
              <div key={`h${i}`} className="cell hdr" role="rowheader">{isVec ? '' : r}</div>,
              ...Array.from({ length: nc }, (_, j) => {
                const c = c0 + j;
                const v = value(r, c);
                const isSel = isVec ? s.r === c : s.r === r && s.c === c;
                const isRow = !isVec && s.r === r;
                return (
                  <button key={`${i}-${j}`} role="gridcell" className={`cell${v.kind === 'none' ? ' na' : ''}${isSel ? ' sel' : ''}${isRow && !isSel ? ' hl' : ''}`}
                    aria-label={`${isVec ? `element ${c}` : `row ${r}, column ${c}`}: ${v.v === null ? 'value not loaded' : v.v}`}
                    title={v.v === null ? 'Actual value not loaded (outside the bounded real sample)' : `${v.kind === 'sample' ? 'Real checkpoint value' : 'Toy value'}: ${v.v}`}
                    onClick={() => go(isVec ? { lv: 0, r: c, c: undefined } : { lv: 0, r, c })}>
                    {v.v === null ? '·' : fmtNum(v.v, 2)}
                  </button>
                );
              }),
            ];
          })}
        </div>
        <p className="faint" style={{ fontSize: '0.82rem' }}>
          Click a cell to open that single stored number (Level 0). {!isVec && 'Use “Open row as unit” to see how one output coordinate is computed (Level 1).'}
          {!isToy && !anyReal && ' Hatched cells: the actual checkpoint values here were not loaded. Only small, provenance-checked tiles are shipped (see Sources). The toy microscope shows the same computation with simulated values.'}
        </p>
        <div className="row">
          {!isVec && <button onClick={() => go({ lv: 1, r: s.r ?? r0 })}>Open row {s.r ?? r0} as unit (Level 1)</button>}
          <ExportButton filename={`${p.id}-${f.key}-tile.json`} data={() => ({
            model: p.repo ?? p.name, revision: p.revision ?? null, tensor: name, logicalShape: f.shape, dtype: p.status === 'official' ? p.dtype : 'float64 (toy)',
            window: { rows: isVec ? null : [r0, r0 + nr - 1], cols: [c0, c0 + nc - 1] },
            evidence: isToy ? 'illustrative-simulation' : anyReal ? 'actual-checkpoint-sample' : 'verified-structure; values not loaded',
            values: Array.from({ length: nr }, (_, i) => Array.from({ length: nc }, (_, j) => value(isVec ? 0 : r0 + i, c0 + j).v)),
          })} />
        </div>
      </div>
    </div>
  );
}

/** LEVEL 1 — one output coordinate (unit / gated channel): every multiplication, the sum, the nonlinearity. */
export function UnitView({ p, f, s, go }: DeepProps) {
  const toyApi = useToy();
  const r = s.r ?? 0;
  const cp = toyCounterpart(p, f.key, s.b ?? 0, s.e ?? 0, r, 0);
  const tt = useMemo(() => (cp ? toyTrace(cp.toy, toyApi.state.seed, toyApi.state.tokens, { variant: toyApi.state.variant, weights: toyApi.weights }) : null), [cp?.toy, toyApi.state.seed, toyApi.state.tokens, toyApi.state.variant, toyApi.weights]); // eslint-disable-line react-hooks/exhaustive-deps
  const [pos, setPos] = useState(toyApi.state.tokens.length - 1);
  const inCount = f.shape.length > 1 ? f.shape[1] : 1;
  const ffnLike = ['gate_proj', 'up_proj', 'down_proj', 'expert_gate_proj', 'expert_up_proj', 'expert_down_proj'].includes(f.key);
  const isDown = f.key.endsWith('down_proj');
  if (!cp || !tt) return <div className="page">No unit view for this tensor.</div>;
  const inp = unitInput(cp, tt.trace, tt.w, pos);
  const W = toyTensor(tt.w, cp.name) as number[][];
  const row = W[cp.row];
  const products = inp ? inp.x.map((x, i) => x * row[i]) : [];
  const sum = products.reduce((a, b) => a + b, 0);
  // gated channel view for FFN-like tensors
  const ch = cp.row;
  const L = tt.trace[Math.min(pos, tt.trace.length - 1)].layers[cp.layer];
  let gated: { gPre: number; g: number; u: number; z: number; downCol: number[] } | null = null;
  if (ffnLike) {
    const fw = cp.family.scope === 'expert' ? tt.w.layers[cp.layer].experts![cp.expert] : tt.w.layers[cp.layer].ffn!;
    const chan = isDown ? 0 : ch;
    const gPre = dot(L.ln2, fw.gate[chan]);
    const u = dot(L.ln2, fw.up[chan]);
    gated = { gPre, g: silu(gPre), u, z: silu(gPre) * u, downCol: fw.down.map((rr) => rr[chan]) };
  }
  return (
    <div className="page wide" style={{ paddingTop: 12 }}>
      <h2 style={{ marginTop: 0 }}>{isDown ? `Output coordinate ${r} of ${f.label}` : ffnLike ? `Gated channel ${r}` : `Output coordinate ${r} of ${f.label}`}</h2>
      <div className="card">
        <p>
          In <strong>{p.name}</strong>, output {r} of <code>{tensorName(f, s.b ?? 0, s.e ?? 0)}</code> is a weighted sum of <strong>{fmtInt(inCount)}</strong> input features using row {r} ({fmtInt(inCount)} learned weights):
        </p>
        <Tex block>{`y_{${r}} = \\sum_{i=0}^{${inCount - 1}} x_i \\, W[${r}, i]`}</Tex>
        {ffnLike && !isDown && (
          <p>
            In SwiGLU a “neuron” is really a <strong>gated channel</strong>: channel j uses row j of W<sub>gate</sub> <em>and</em> row j of W<sub>up</sub> (two weighted sums), applies SiLU to the gate, multiplies the two, and then column j of W<sub>down</sub> carries the result into every residual feature. The classic circle-with-arrows neuron picture is only an analogy for this.
          </p>
        )}
        <p className="faint" style={{ fontSize: '0.86rem' }}>Incoming weights: row {r} (persistent). Pre-activation: the sum (runtime). Activation / gate: computed from it (runtime). Outgoing weights: {ffnLike && !isDown ? `column ${r} of W_down` : 'the next layer’s matrix'}.</p>
      </div>
      <div className="card" style={{ marginTop: 12 }}>
        <SimBanner>
          {cp.mapped ? <>Computed in the <strong>{cp.toy.name}</strong> on the same tensor role: <code>{cp.name}</code> row {cp.row} ({cp.toy.d}-dimensional instead of {fmtInt(inCount)}). Indices were folded into the toy's smaller extent.</> : <>Toy simulator values.</>}
          {!tt.shared && ' Uses the unedited seeded toy weights.'}
        </SimBanner>
        <div className="row" style={{ marginTop: 8 }}>
          <label className="field inline">Token position
            <select value={pos} onChange={(e) => setPos(Number(e.target.value))}>
              {toyApi.state.tokens.map((t, i) => <option key={i} value={i}>{i}: {TOY_VOCAB[t]}</option>)}
            </select>
          </label>
          <span className="faint" style={{ fontSize: '0.85rem' }}>Changing the input recomputes activations; the weights stay fixed.</span>
        </div>
        {inp ? (
          <>
            <p className="faint" style={{ fontSize: '0.85rem', marginBottom: 2 }}>Input x = {inp.label}</p>
            <div className="table-wrap">
              <table className="t" style={{ maxWidth: 720 }}>
                <thead><tr><th>i</th><th className="num">x_i (activation)</th><th className="num">W[{cp.row}, i] (weight)</th><th className="num">product</th></tr></thead>
                <tbody>
                  {inp.x.map((x, i) => (
                    <tr key={i}><td>{i}</td><td className="num">{fmtNum(x)}</td><td className="num">{fmtNum(row[i])}</td><td className="num">{fmtNum(products[i])}</td></tr>
                  ))}
                  <tr><td colSpan={3}><strong>sum = pre-activation y<sub>{cp.row}</sub></strong></td><td className="num"><strong>{fmtNum(sum)}</strong></td></tr>
                </tbody>
              </table>
            </div>
            {gated && !isDown && (
              <div className="callout" style={{ marginTop: 10 }}>
                <div>gate pre-activation a = {fmtNum(gated.gPre)} → SiLU(a) = a·σ(a) = {fmtNum(gated.gPre)} × {fmtNum(sigmoid(gated.gPre))} = <strong>{fmtNum(gated.g)}</strong></div>
                <div>up value u = {fmtNum(gated.u)}</div>
                <div>gated channel z = g × u = <strong>{fmtNum(gated.z)}</strong></div>
                <VectorView label={`column ${ch} of W_down (outgoing weights)`} values={gated.downCol} kind="par" />
                <VectorView label={`contribution of this channel to every output feature: z × W_down[:, ${ch}]`} values={gated.downCol.map((v) => v * gated!.z)} />
              </div>
            )}
          </>
        ) : (
          <p className="muted">This tensor is applied elementwise or by lookup, so there is no weighted sum; see Level 0 or the module view.</p>
        )}
        <div className="row" style={{ marginTop: 8 }}>
          <button onClick={() => go({ lv: 0, c: s.c ?? 0 })}>Open weight W[{r}, {s.c ?? 0}] (Level 0)</button>
          <button className="ghost" onClick={() => go({ lv: 2 })}>Back to the matrix (Level 2)</button>
        </div>
      </div>
    </div>
  );
}

/** LEVEL 0 — one stored scalar: value, sign, precision, coordinate, checkpoint location and provenance. */
export function ScalarView({ p, f, s, go }: DeepProps) {
  const toyApi = useToy();
  const name = tensorName(f, s.b ?? 0, s.e ?? 0);
  const isVec = f.shape.length === 1;
  const r = s.r ?? 0, c = isVec ? null : s.c ?? 0;
  const real = useRealValues(p, name);
  const offsets = useOffsets(p.status === 'official' ? p.id : null);
  const hit = p.status === 'official' ? sampleFor(real.list?.filter((x) => x.modelKey === p.id), name, r, c ?? undefined) : null;
  const realVal = hit?.value ?? null;
  const cp = toyCounterpart(p, f.key, s.b ?? 0, s.e ?? 0, r, c);
  const tt = useMemo(() => (cp ? toyTrace(cp.toy, toyApi.state.seed, toyApi.state.tokens, { variant: toyApi.state.variant, weights: toyApi.weights }) : null), [cp?.toy, toyApi.state.seed, toyApi.state.tokens, toyApi.state.variant, toyApi.weights]); // eslint-disable-line react-hooks/exhaustive-deps
  const toyVal = cp && tt ? toyValue(tt.w, cp) : null;
  const [draft, setDraft] = useState<number | null>(null);
  const fo = offsets.status === 'ok' ? fileOffset(offsets.data, name, r, c) : null;
  const shown = p.status === 'toy' ? toyVal : realVal;
  const bits = shown !== null ? toBf16Bits(shown) : null;
  const parts = bits !== null ? bf16Parts(bits) : null;
  const preview = useMemo(() => {
    if (!cp || !tt || draft === null || !Number.isFinite(draft)) return null;
    const w2 = withEditedWeight(tt.w, cp.name, cp.row, cp.col, draft);
    const a = tt.trace[tt.trace.length - 1].logits;
    const b = forwardFull(cp.toy, w2, toyApi.state.tokens);
    const bl = b[b.length - 1].logits;
    return { before: a, after: bl, diff: maxAbsDiff(a, bl) };
  }, [cp, tt, draft, toyApi.state.tokens]);
  const canCommit = cp && toyApi.state.variant === cp.toy.id;
  return (
    <div className="page wide" style={{ paddingTop: 12 }}>
      <h2 style={{ marginTop: 0 }}>One stored number: {f.label} [{r}{c === null ? '' : `, ${c}`}]</h2>
      <div className="grid2">
        <div className="card">
          <p>A <strong>parameter</strong> is one learned scalar. This one sits at coordinate <code>[{r}{c === null ? '' : `, ${c}`}]</code> of <code>{name}</code> {isVec ? '' : `(row ${r} = output coordinate, column ${c} = input feature)`}.</p>
          {p.status === 'official' ? (
            real.samples.status === 'loading' ? (
              <p className="muted">Checking the shipped real samples for this coordinate…</p>
            ) : realVal !== null ? (
              <>
                <div className="row"><Evidence kind="sample" /><span className="big-num">{realVal}</span></div>
                <p className="muted" style={{ fontSize: '0.88rem' }}>Actual value read from {hit!.sample.shard} of {hit!.sample.model} @ {hit!.sample.revision.slice(0, 12)} (raw-bytes SHA-256 {hit!.sample.rawSha256.slice(0, 16)}…), widened exactly from BF16. {hit!.sample.note}</p>
              </>
            ) : (
              <div className="callout warn">
                <div className="row"><Evidence kind="unknown" /><strong>Actual value not loaded.</strong></div>
                <p style={{ marginBottom: 0 }}>Only small real tiles are shipped (e.g. rows 0–15 × cols 0–15 of selected tensors), to avoid downloading multi-gigabyte checkpoints. The structure, location and dtype below are verified; the number itself is not shown rather than invented. {real.samples.status === 'error' && `(Sample file failed to load: ${real.samples.error})`}</p>
                <div className="row" style={{ marginTop: 6 }}>
                  {(real.list ?? []).filter((x) => x.modelKey === p.id).map((x) => {
                    const loc = locateTensor(p, x.tensor);
                    return loc ? (
                      <a key={x.id} className="btn small" href={explorerHref({ m: p.id, lv: 0, t: loc.key, b: loc.b, e: loc.e, r: 0, c: x.logicalShape.length > 1 ? 0 : undefined })}>Real sample: {x.tensor.replace('model.', '')}</a>
                    ) : null;
                  })}
                </div>
              </div>
            )
          ) : (
            <><div className="row"><Evidence kind="simulation" /><span className="big-num">{toyVal?.toFixed(6)}</span></div><p className="faint">Toy simulator value (seeded synthetic weight, float64).</p></>
          )}
          {parts && (
            <div style={{ marginTop: 10 }}>
              <strong>BF16 representation</strong> {p.status === 'official' && realVal !== null ? <Evidence kind="sample" title="Exact bits of the stored value" /> : <Evidence kind="derived" title="What this toy float64 value would round to in BF16" />}
              <div className="bits" aria-label={`bits ${bits!.toString(2).padStart(16, '0')}`} style={{ margin: '6px 0' }}>
                {bits!.toString(2).padStart(16, '0').split('').map((b, i) => <span key={i} className={i === 0 ? 's' : i < 9 ? 'e' : 'f'}>{b}</span>)}
              </div>
              <div className="faint" style={{ fontSize: '0.84rem' }}>
                sign {parts.sign} ({parts.sign ? 'negative' : 'positive'}) · exponent bits {parts.exponentBits} {parts.unbiasedExponent !== null && `(2^${parts.unbiasedExponent})`} · 7 fraction bits {parts.fractionBits} · class {parts.kind} · value {parts.value}
                {p.status === 'toy' && shown !== null && !isExactBf16(shown) && ` (the toy keeps full float64 precision; BF16 would round it to ${parts.value})`}
              </div>
              <p className="faint" style={{ fontSize: '0.8rem' }}>BF16 keeps float32's 8-bit exponent (same range) with only 7 fraction bits (about 2–3 significant decimal digits). Arithmetic usually accumulates in float32. This bit picture describes the number format; it is not a dump of file bytes (files store it little-endian).</p>
            </div>
          )}
        </div>
        <div className="card">
          <h3>Where it lives</h3>
          <dl className="kv">
            <dt>Model</dt><dd>{p.repo ? `${p.repo} @ ${p.revision!.slice(0, 12)}` : p.name}</dd>
            <dt>Tensor</dt><dd className="mono">{name}</dd>
            <dt>Coordinate</dt><dd className="mono">[{r}{c === null ? '' : `, ${c}`}] of {shapeStr(f.shape)}</dd>
            <dt>Flat index</dt><dd className="mono">{fmtInt(isVec ? r : r * f.shape[1] + (c ?? 0))} (row-major)</dd>
            {p.status === 'official' && <><dt>File location</dt><dd className="mono">{fo ? `${fo.shard}, byte ${fmtInt(fo.byte)} (2 bytes, little-endian)` : offsets.status === 'loading' ? 'loading byte map…' : offsets.status === 'error' ? `byte map unavailable: ${offsets.error}` : '—'}</dd></>}
            <dt>Runtime</dt><dd>After loading, the tensor becomes an array in RAM/VRAM. Its memory address is chosen by the allocator at run time and is unrelated to the file offset or the coordinate.</dd>
          </dl>
          {p.status === 'official' && <p className="faint" style={{ fontSize: '0.8rem' }}>File offset = 8 (length prefix) + header length + tensor data offset + flat index × 2, from the pinned safetensors header. <Evidence kind="derived" /></p>}
          <h3 style={{ marginTop: 14 }}>Change one weight {cp?.mapped ? '(toy analogue)' : ''}</h3>
          {cp && toyVal !== null ? (
            <>
              <p className="faint" style={{ fontSize: '0.85rem' }}>Edits apply to the <strong>{cp.toy.name}</strong> tensor <code>{cp.name}</code>[{cp.row}{cp.col === null ? '' : `, ${cp.col}`}] (current value {toyVal.toFixed(5)}), never to a real checkpoint.</p>
              <div className="row">
                <NumberField label="new value" value={draft ?? toyVal} onChange={setDraft} step={0.05} width={120} />
                <button onClick={() => setDraft(toyVal + 0.5)}>+0.5</button>
                <button onClick={() => setDraft(null)} className="ghost">clear</button>
              </div>
              {preview && (
                <div className="banner-edit" style={{ marginTop: 8 }}>
                  Preview (not yet applied): last-position logits change by up to <strong>{preview.diff.toExponential(3)}</strong>.
                  <VectorView label="logits before" values={preview.before} />
                  <VectorView label="logits after" values={preview.after} />
                  {canCommit ? (
                    <button className="small" style={{ marginTop: 6 }} onClick={() => { toyApi.dispatch({ type: 'edit', name: cp.name, row: cp.row, col: cp.col, value: draft! }); setDraft(null); }}>Apply to the shared toy simulator (marks it “edited”, invalidates its cache)</button>
                  ) : (
                    <button className="small" style={{ marginTop: 6 }} onClick={() => toyApi.dispatch({ type: 'variant', v: cp.toy.id as 'toy-dense' | 'toy-moe' })}>Switch shared simulator to {cp.toy.name} first</button>
                  )}
                </div>
              )}
              {toyApi.edited && <div className="banner-edit" style={{ marginTop: 8 }}>The shared toy simulator is in an <strong>edited state</strong> ({toyApi.state.edits.length} change{toyApi.state.edits.length > 1 ? 's' : ''}). <button className="small" onClick={() => toyApi.dispatch({ type: 'resetWeights' })}>Reset to seeded fixture</button></div>}
            </>
          ) : <p className="muted">No toy analogue for this tensor.</p>}
          <div className="row" style={{ marginTop: 10 }}>
            {!isVec && <button className="ghost" onClick={() => go({ lv: 1 })}>Up: the unit using this weight (Level 1)</button>}
            <button className="ghost" onClick={() => go({ lv: 2 })}>Up: the whole matrix (Level 2)</button>
            <ExportButton filename={`${p.id}-scalar.json`} data={() => ({ model: p.repo ?? p.name, revision: p.revision ?? null, tensor: name, coordinate: c === null ? [r] : [r, c], dtype: p.status === 'official' ? p.dtype : 'float64 (toy)', value: shown, evidence: p.status === 'toy' ? 'illustrative-simulation' : realVal !== null ? 'actual-checkpoint-sample' : 'unknown (value not loaded)', fileLocation: fo, sampleSha256: hit?.sample.rawSha256 ?? null })} />
          </div>
        </div>
      </div>
    </div>
  );
}

/** LEVEL 6 — execution & learning for the selected model. */
export function ExecutionView({ p }: { p: CoreProfile }) {
  const a = accounting(p);
  const kv = kvBytesPerToken(p);
  const steps = [
    ['Messages → chat template', 'Role markers and (optionally) an empty <think></think> block are inserted as text.', '#/pipeline'],
    ['Tokenizer → token ids', `Byte-level BPE; ${p.tokenizerVocab ? fmtInt(p.tokenizerVocab) : p.vocabRows} ids defined.`, '#/pipeline'],
    ['Embedding row lookup', `${fmtInt(p.d)} numbers per token copied from the table.`, explorerHref({ m: p.id, lv: 3, mod: 'embedding' })],
    ['Prefill', `All prompt positions go through block 0, then block 1, … block ${p.layers - 1}. Positions run in parallel inside a block; blocks run in sequence. Each block stores K and V for every position.`, '#/labs/cache'],
    ['Decode', `One new token at a time passes through all ${p.layers} blocks, attends to cached K/V, and appends its own K/V (${fmtBytes(kv)} per token at BF16 for this model, K/V only).`, '#/labs/cache'],
    ['Final norm → logits', `${fmtInt(p.vocabRows)} logits from the untied head.`, explorerHref({ m: p.id, lv: 3, mod: 'lm_head' })],
    ['Sampling → next token', 'Temperature / top-k / top-p / draw; the token is appended and the loop repeats.', '#/labs/sampling'],
    ['Training path (separate)', 'Targets = inputs shifted by one; cross-entropy loss; gradients by the chain rule; optimizer updates weights. Inference never changes weights.', '#/labs/training'],
    ['Memory & compute', `${fmtBytes(a.total * 2)} of BF16 weights before any cache or workspace.`, '#/labs/memory'],
  ];
  return (
    <div className="page" style={{ paddingTop: 12 }}>
      <h2 style={{ marginTop: 0 }}>Execution and learning · {p.name}</h2>
      <ol style={{ paddingLeft: 20 }}>
        {steps.map(([t, d, h]) => (
          <li key={t} style={{ marginBottom: 10 }}><a href={h}><strong>{t}</strong></a> — <span className="muted">{d}</span></li>
        ))}
      </ol>
      <div className="callout">Each step links to a working laboratory driven by the toy simulator, or to real tokenizer data. A real {p.name} is not running in your browser.</div>
    </div>
  );
}
