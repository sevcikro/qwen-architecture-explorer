import { useState } from 'react';
import { Evidence, Tex, fmtInt } from '../components/ui';
import { linear } from '../math/linalg';
import { explorerHref } from '../state/route';

export const FIXTURE_X = [1, 2, 3];
export const FIXTURE_W = [
  [0.2, -0.5, 0.1],
  [0.4, 0.3, -0.2],
];

/** x = [1,2,3], W [2,3]: y = x W^T = [-0.5, 0.4]. Teaching values chosen for the example, not learned, not Qwen's. */
export function TinyExample() {
  const [x, setX] = useState(FIXTURE_X.slice());
  const [W, setW] = useState(FIXTURE_W.map((r) => r.slice()));
  const [sel, setSel] = useState<[number, number]>([0, 1]);
  const y = linear(x, W);
  const weightsEdited = W.some((r, i) => r.some((v, j) => v !== FIXTURE_W[i][j]));
  const inputEdited = x.some((v, i) => v !== FIXTURE_X[i]);
  const clean = (v: number) => Math.round(v * 1e10) / 1e10;
  return (
    <div>
      <p className="muted">
        Six weights, two outputs, no bias. <Evidence kind="simulation" title="Teaching values chosen for this example" /> These values were chosen for the example; they were not learned from data and are not extracted from Qwen. The operation is exactly what every Qwen3 linear projection does, at a much larger size.
      </p>
      <Tex block>{`y = x\\,W^{\\top},\\quad x \\in \\mathbb{R}^{3},\\; W \\in \\mathbb{R}^{2\\times 3}\\;\\text{(stored [out, in])}`}</Tex>
      <div className="grid2">
        <div>
          <div className="faint" style={{ fontSize: '0.85rem' }}>Input x (activations) — <strong>Change input</strong>:</div>
          <div className="row">
            {x.map((v, i) => (
              <label key={i} className="field" style={{ width: 80 }}>x[{i}]
                <input type="number" step="0.5" value={v} aria-label={`input x ${i}`} onChange={(e) => Number.isFinite(e.target.valueAsNumber) && setX(x.map((o, k) => (k === i ? e.target.valueAsNumber : o)))} />
              </label>
            ))}
          </div>
          <div className="faint" style={{ fontSize: '0.85rem', marginTop: 8 }}>Stored W (learned parameters in a real model) — <strong>Change one weight</strong>: click a cell, edit it.</div>
          <div className="matrix-grid" style={{ gridTemplateColumns: '40px repeat(3, 64px)', marginTop: 4 }}>
            <div className="cell hdr">out\in</div>{[0, 1, 2].map((j) => <div key={j} className="cell hdr">{j}</div>)}
            {W.map((row, i) => [
              <div key={`h${i}`} className="cell hdr">{i}</div>,
              ...row.map((v, j) => (
                <button key={`${i}${j}`} className={`cell${sel[0] === i && sel[1] === j ? ' sel' : ''}${sel[0] === i ? ' hl' : ''}`} onClick={() => setSel([i, j])} aria-label={`W[${i}][${j}] = ${v}`}>{v}</button>
              )),
            ])}
          </div>
          <div className="row" style={{ marginTop: 6 }}>
            <label className="field inline">W[{sel[0]}][{sel[1]}] =
              <input type="number" step="0.1" value={W[sel[0]][sel[1]]} aria-label="selected weight value" style={{ width: 90 }} onChange={(e) => Number.isFinite(e.target.valueAsNumber) && setW(W.map((r, i) => r.map((v, j) => (i === sel[0] && j === sel[1] ? e.target.valueAsNumber : v))))} />
            </label>
            <button onClick={() => { setX(FIXTURE_X.slice()); setW(FIXTURE_W.map((r) => r.slice())); }}>Reset to the exact fixture</button>
          </div>
          {weightsEdited && <div className="banner-edit" style={{ marginTop: 6 }}>Edited model state: W differs from the seeded fixture. (Changing a weight changes the model; changing the input only changes activations.)</div>}
          {!weightsEdited && inputEdited && <div className="faint" style={{ marginTop: 6, fontSize: '0.85rem' }}>Same weights, new input: only the activations y changed.</div>}
        </div>
        <div>
          {y.map((yi, i) => (
            <div key={i} className={`card tight${sel[0] === i ? '' : ''}`} style={{ marginBottom: 8 }}>
              <div className="mono" style={{ fontSize: '0.9rem' }}>
                y[{i}] = {x.map((xv, j) => `${xv}×${W[i][j]}`).join(' + ')} = <strong>{clean(yi)}</strong>
              </div>
            </div>
          ))}
          <p className="muted" style={{ fontSize: '0.9rem' }}>
            With the fixture, y = [−0.5, 0.4]. In <strong>Qwen3-8B</strong> the query projection W_Q has the same [out, in] layout with shape [4,096, 4,096]: each of its 4,096 outputs is a sum of 4,096 products, and the matrix holds {fmtInt(4096 * 4096)} parameters, in each of 36 blocks, with no bias.{' '}
            <a href={explorerHref({ m: 'qwen3-8b', lv: 2, b: 0, t: 'q_proj' })}>Open the real W_Q →</a>
          </p>
        </div>
      </div>
    </div>
  );
}
