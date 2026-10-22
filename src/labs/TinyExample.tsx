import { useState } from 'react';
import { Evidence, Tex } from '../components/ui';
import { linear } from '../math/linalg';
import { explorerHref } from '../state/route';
import { useLang } from '../i18n/i18n';

export const FIXTURE_X = [1, 2, 3];
export const FIXTURE_W = [
  [0.2, -0.5, 0.1],
  [0.4, 0.3, -0.2],
];

/** x = [1,2,3], W [2,3]: y = x W^T = [-0.5, 0.4]. Teaching values chosen for the example, not learned, not Qwen's. */
export function TinyExample() {
  const { lang, num } = useLang();
  const T = (en: string, sk: string) => (lang === 'sk' ? sk : en);
  const [x, setX] = useState(FIXTURE_X.slice());
  const [W, setW] = useState(FIXTURE_W.map((r) => r.slice()));
  const [sel, setSel] = useState<[number, number]>([0, 1]);
  const y = linear(x, W);
  const weightsEdited = W.some((r, i) => r.some((v, j) => v !== FIXTURE_W[i][j]));
  const inputEdited = x.some((v, i) => v !== FIXTURE_X[i]);
  const clean = (v: number) => num(Math.round(v * 1e10) / 1e10);
  return (
    <div>
      <p className="muted">
        {T('Six weights, two outputs, no bias.', 'Šesť váh, dva výstupy, bez biasu.')} <Evidence kind="simulation" /> {T('These values were chosen for the example; they were not learned from data and are not extracted from Qwen. The operation is exactly what every Qwen3 linear projection does, at a much larger size.', 'Tieto hodnoty boli zvolené pre príklad; neboli naučené z dát ani prevzaté z modelu Qwen. Operácia je presne tá, ktorú robí každá lineárna projekcia v Qwen3, len v oveľa väčšom rozsahu.')}
      </p>
      <Tex block>{`y = x\\,W^{\\top},\\quad x \\in \\mathbb{R}^{3},\\; W \\in \\mathbb{R}^{2\\times 3}\\;\\text{(${T('stored [out, in]', 'uložené [výstup, vstup]')})}`}</Tex>
      <div className="grid2">
        <div>
          <div className="faint" style={{ fontSize: '0.85rem' }}>{T('Input x (activations)', 'Vstup x (aktivácie)')} — <strong>{T('Change input', 'Zmeniť vstup')}</strong>:</div>
          <div className="row">
            {x.map((v, i) => (
              <label key={i} className="field" style={{ width: 80 }}>x[{i}]
                <input type="number" step="0.5" value={v} aria-label={`${T('input', 'vstup')} x ${i}`} onChange={(e) => Number.isFinite(e.target.valueAsNumber) && setX(x.map((o, k) => (k === i ? e.target.valueAsNumber : o)))} />
              </label>
            ))}
          </div>
          <div className="faint" style={{ fontSize: '0.85rem', marginTop: 8 }}>{T('Stored W (learned parameters in a real model)', 'Uložená W (v skutočnom modeli naučené parametre)')} — <strong>{T('Change one weight', 'Zmeniť jednu váhu')}</strong>: {T('click a cell, edit it.', 'kliknite na bunku a upravte ju.')}</div>
          <div className="matrix-grid" style={{ gridTemplateColumns: '40px repeat(3, 64px)', marginTop: 4 }}>
            <div className="cell hdr">{T('out\\in', 'výst\\vst')}</div>{[0, 1, 2].map((j) => <div key={j} className="cell hdr">{j}</div>)}
            {W.map((row, i) => [
              <div key={`h${i}`} className="cell hdr">{i}</div>,
              ...row.map((v, j) => (
                <button key={`${i}${j}`} className={`cell${sel[0] === i && sel[1] === j ? ' sel' : ''}${sel[0] === i ? ' hl' : ''}`} onClick={() => setSel([i, j])} aria-label={`W[${i}][${j}] = ${v}`}>{num(v)}</button>
              )),
            ])}
          </div>
          <div className="row" style={{ marginTop: 6 }}>
            <label className="field inline">W[{sel[0]}][{sel[1]}] =
              <input type="number" step="0.1" value={W[sel[0]][sel[1]]} aria-label={T('selected weight value', 'hodnota vybranej váhy')} style={{ width: 90 }} onChange={(e) => Number.isFinite(e.target.valueAsNumber) && setW(W.map((r, i) => r.map((v, j) => (i === sel[0] && j === sel[1] ? e.target.valueAsNumber : v))))} />
            </label>
            <button onClick={() => { setX(FIXTURE_X.slice()); setW(FIXTURE_W.map((r) => r.slice())); }}>{T('Reset to the exact fixture', 'Obnoviť presné pôvodné hodnoty')}</button>
          </div>
          {weightsEdited && <div className="banner-edit" style={{ marginTop: 6 }}>{T('Edited model state: W differs from the seeded fixture. (Changing a weight changes the model; changing the input only changes activations.)', 'Upravený stav modelu: W sa líši od pôvodných hodnôt. (Zmena váhy mení model; zmena vstupu mení len aktivácie.)')}</div>}
          {!weightsEdited && inputEdited && <div className="faint" style={{ marginTop: 6, fontSize: '0.85rem' }}>{T('Same weights, new input: only the activations y changed.', 'Rovnaké váhy, nový vstup: zmenili sa len aktivácie y.')}</div>}
        </div>
        <div>
          {y.map((yi, i) => (
            <div key={i} className="card tight" style={{ marginBottom: 8 }}>
              <div className="mono" style={{ fontSize: '0.9rem' }}>
                y[{i}] = {x.map((xv, j) => `${num(xv)}×${num(W[i][j])}`).join(' + ')} = <strong>{clean(yi)}</strong>
              </div>
            </div>
          ))}
          <p className="muted" style={{ fontSize: '0.9rem' }}>
            {T(`With the fixture, y = [−0.5, 0.4]. In Qwen3-8B the query projection W_Q has the same [out, in] layout with shape [4,096, 4,096]: each of its 4,096 outputs is a sum of 4,096 products, and the matrix holds ${(4096 * 4096).toLocaleString('en-US')} parameters, in each of 36 blocks, with no bias.`, `S pôvodnými hodnotami je y = [−0,5; 0,4]. V Qwen3-8B má projekcia dopytov W_Q rovnaké usporiadanie [výstup, vstup] s tvarom [4 096, 4 096]: každý zo 4 096 výstupov je súčet 4 096 súčinov a matica obsahuje ${(4096 * 4096).toLocaleString('sk-SK')} parametrov, v každom z 36 blokov, bez biasu.`)}{' '}
            <a href={explorerHref({ m: 'qwen3-8b', lv: 2, b: 0, t: 'q_proj' })}>{T('Open the real W_Q →', 'Otvoriť skutočnú W_Q →')}</a>
          </p>
        </div>
      </div>
    </div>
  );
}
