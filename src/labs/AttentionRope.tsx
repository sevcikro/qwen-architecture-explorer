import { useState } from 'react';
import { useToy } from '../state/toyStore';
import { Evidence, NumberField, Section, Select, Slider, Tex, VectorView } from '../components/ui';
import { ToyControls, PromptEditor, tok } from './common';
import { CORE_PROFILES, QWEN3_8B } from '../models/profiles';
import { applyRope, ropeInvFreq } from '../math/ops';
import { dot } from '../math/linalg';
import { useLang } from '../i18n/i18n';

export function AttentionLab() {
  const { profile: p, trace, state } = useToy();
  const { lang, num, fnum } = useLang();
  const L = (en: string, sk: string) => (lang === 'sk' ? sk : en);
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
      <Section title={L('4 · Attention lab', '4 · Laboratórium pozornosti')} badges={<Evidence kind="simulation" />}>
        <ToyControls />
        <PromptEditor />
        <div className="row">
          <NumberField label={L('block', 'blok')} value={layer} min={0} max={p.layers - 1} step={1} onChange={(v) => setLayer(Math.round(v))} width={70} />
          <NumberField label={L('query position', 'pozícia dopytu')} value={t} min={0} max={T - 1} step={1} onChange={(v) => setQpos(Math.round(v))} width={80} />
          <NumberField label={L('key position', 'pozícia kľúča')} value={s} min={0} max={T - 1} step={1} onChange={(v) => setKpos(Math.round(v))} width={80} />
          <NumberField label={L(`query head (0–${p.hq - 1})`, `hlava dopytov (0–${p.hq - 1})`)} value={h} min={0} max={p.hq - 1} step={1} onChange={(v) => setHead(Math.round(v))} width={90} />
        </div>
        <p className="muted" style={{ fontSize: '0.9rem' }}>{L(`Query head ${h} uses K/V head ${kv} (⌊${h} / ${group}⌋). Selecting positions and heads is inspection only.`, `Hlava dopytov ${h} používa hlavu K/V ${kv} (⌊${h} / ${group}⌋). Výber pozícií a hláv je len prezeranie.`)}</p>
        <div className="grid2">
          <div>
            <h3>{L(`Per key position for query “${tok(state.tokens[t])}” (position ${t})`, `Pre každú pozíciu kľúča, dopyt „${tok(state.tokens[t])}“ (pozícia ${t})`)}</h3>
            <div className="table-wrap" tabIndex={0}>
              <table className="t">
                <thead><tr><th>{L('key pos', 'poz. kľúča')}</th><th>{L('token', 'token')}</th><th className="num">q·k</th><th className="num">÷ √{p.headDim}</th><th>{L('mask', 'maska')}</th><th className="num">exp(s − max)</th><th className="num">{L('probability', 'pravdepodobnosť')}</th></tr></thead>
                <tbody>
                  {raw.map((r, i) => {
                    const masked = i > t;
                    return (
                      <tr key={i} style={{ background: i === s ? 'color-mix(in srgb, var(--focus) 14%, transparent)' : undefined }}>
                        <td>{i}</td><td className="mono">{tok(state.tokens[i])}</td>
                        <td className="num">{fnum(r)}</td><td className="num">{fnum(scaled[i])}</td>
                        <td>{masked ? <strong>{L('future · −∞', 'budúcnosť · −∞')}</strong> : L('visible', 'viditeľné')}</td>
                        <td className="num">{masked ? '0' : fnum(Math.exp(scaled[i] - validMax))}</td>
                        <td className="num"><strong>{probs[i] === 0 ? '0' : fnum(probs[i])}</strong></td>
                      </tr>
                    );
                  })}
                  <tr><td colSpan={6}>{L('sum over visible keys', 'súčet cez viditeľné kľúče')}</td><td className="num">{fnum(probs.reduce((a, b) => a + b, 0))}</td></tr>
                </tbody>
              </table>
            </div>
            <p className="faint" style={{ fontSize: '0.84rem' }}>{L(`Stable softmax subtracts the largest visible score (${fnum(validMax)}) before exponentiating. Future positions get probability exactly 0. Policy for a row with no visible key: all zeros, flagged; never NaN, never renormalized out of nothing (cannot happen in causal self-attention, where a token always sees itself).`, `Stabilný softmax pred umocnením odčíta najväčšie viditeľné skóre (${fnum(validMax)}). Budúce pozície dostanú pravdepodobnosť presne 0. Pravidlo pre riadok bez viditeľného kľúča: samé nuly s označením; nikdy NaN a nikdy prepočet z ničoho (v kauzálnej pozornosti sa to nestane, lebo token vždy vidí sám seba).`)}</p>
          </div>
          <div>
            <h3>{L(`q · k for key position ${s}`, `q · k pre pozíciu kľúča ${s}`)}</h3>
            {s > t ? <div className="callout warn">{L(`Key ${s} is in the future of query ${t}: masked, so this product is never used.`, `Kľúč ${s} je v budúcnosti dopytu ${t}: je zamaskovaný, takže tento súčin sa nikdy nepoužije.`)}</div> : (
              <>
                <VectorView label={L(`q (head ${h}, after q_norm and RoPE at position ${t})`, `q (hlava ${h}, po q_norm a RoPE na pozícii ${t})`)} values={A.qRot[h]} />
                <VectorView label={L(`k (K/V head ${kv}, after k_norm and RoPE at position ${s})`, `k (hlava K/V ${kv}, po k_norm a RoPE na pozícii ${s})`)} values={kRot(s)} />
                <VectorView label={L('elementwise products', 'súčiny po prvkoch')} values={A.qRot[h].map((x, i) => x * kRot(s)[i])} />
                <p className="mono" style={{ fontSize: '0.88rem' }}>{L('sum', 'súčet')} = {fnum(dot(A.qRot[h], kRot(s)))} → ÷ √{p.headDim} = {fnum(scaled[s])} → {L('probability', 'pravdepodobnosť')} {fnum(probs[s])}</p>
              </>
            )}
            <h3>{L('Weighted sum of V', 'Vážený súčet V')}</h3>
            {Array.from({ length: t + 1 }, (_, i) => <VectorView key={i} label={`${fnum(probs[i], 3)} × v[${i}] (“${tok(state.tokens[i])}”)`} values={vOf(i)} />)}
            <VectorView label={L(`head ${h} output z = Σ p_i v_i`, `výstup hlavy ${h} z = Σ p_i v_i`)} values={A.headOut[h]} />
            {p.hq > 1 && <p className="faint" style={{ fontSize: '0.84rem' }}>{L(`Query heads 0 and 1 share K/V head 0 but have different queries, so their probabilities and outputs differ: head 0 out ${A.headOut[0].map((x) => fnum(x, 2)).join(', ')} vs head 1 out ${A.headOut[1].map((x) => fnum(x, 2)).join(', ')}.`, `Hlavy dopytov 0 a 1 zdieľajú hlavu K/V 0, ale majú iné dopyty, preto sa ich pravdepodobnosti aj výstupy líšia: výstup hlavy 0 je ${A.headOut[0].map((x) => fnum(x, 2)).join('; ')} a hlavy 1 ${A.headOut[1].map((x) => fnum(x, 2)).join('; ')}.`)}</p>}
          </div>
        </div>
      </Section>
      <Section title={L(`Computed attention probabilities, head ${h}, block ${layer}`, `Vypočítané pravdepodobnosti pozornosti, hlava ${h}, blok ${layer}`)} badges={<Evidence kind="simulation" />}>
        <div className="grid2">
          <div>
            <p className="muted" style={{ fontSize: '0.88rem' }}>{lang === 'sk' ? <>Skutočne vypočítané hodnoty tohto hračkového behu (riadky = pozície dopytov, stĺpce = pozície kľúčov). <strong>Nie je</strong> to pozorovaná mapa modelu Qwen a samotná sila pozornosti nie je úplným príčinným vysvetlením správania modelu. Skutočné mapy z Qwen3-0.6B: <a href="#/trace">Skutočný beh</a>.</> : <>Actual computed values of this toy run (rows = query positions, columns = key positions). This is <strong>not</strong> an observed Qwen heatmap, and attention strength alone is not a complete causal explanation of model behavior. Real maps from Qwen3-0.6B: <a href="#/trace">Real run</a>.</>}</p>
            <AttnMatrix T={T} values={(i, j) => trace[i].layers[layer].attn.probs[h][j]} sel={[t, s]} />
          </div>
          <div>
            <p className="muted" style={{ fontSize: '0.88rem' }}>{L('Causal dependency structure (the same for every model, head and input): which keys a query may read at all.', 'Štruktúra kauzálnych závislostí (rovnaká pre každý model, hlavu a vstup): ktoré kľúče môže dopyt vôbec čítať.')}</p>
            <AttnMatrix T={T} values={(i, j) => (j <= i ? 1 : 0)} sel={[t, s]} binary />
          </div>
        </div>
      </Section>
      <Section title={L('The same operation at real size', 'Rovnaká operácia v skutočnej veľkosti')} badges={<><Evidence kind="verified" /><Evidence kind="derived" /></>}>
        <Select label={L('Reference model', 'Referenčný model')} value={ref} onChange={setRef} options={CORE_PROFILES.map((x) => ({ value: x.id, label: x.name }))} />
        <div className="table-wrap" tabIndex={0} style={{ marginTop: 8 }}>
          <table className="t">
            <tbody>
              <tr><td>W_Q</td><td className="mono">[Hq·dh, d] = [{num(R.hq * R.headDim)}, {num(R.d)}]</td></tr>
              <tr><td>W_K, W_V</td><td className="mono">[Hkv·dh, d] = [{num(R.hkv * R.headDim)}, {num(R.d)}]</td></tr>
              <tr><td>Q</td><td className="mono">[B, {R.hq}, Tq, {R.headDim}]</td></tr>
              <tr><td>K, V</td><td className="mono">[B, {R.hkv}, Tk, {R.headDim}]</td></tr>
              <tr><td>{L('Logical scores', 'Logické skóre')}</td><td className="mono">[B, {R.hq}, Tq, Tk]</td></tr>
              <tr><td>W_O</td><td className="mono">[d, Hq·dh] = [{num(R.d)}, {num(R.hq * R.headDim)}]</td></tr>
              <tr><td>GQA</td><td>{L(`${R.hq / R.hkv} query heads share each of the ${R.hkv} K/V heads`, `${R.hq / R.hkv} hláv dopytov zdieľa každú z ${R.hkv} hláv K/V`)}</td></tr>
              <tr><td>{L('q/k norm gains', 'zisky q/k noriem')}</td><td>{L(`${R.headDim} + ${R.headDim} per block, shared across heads`, `${R.headDim} + ${R.headDim} v bloku, spoločné pre všetky hlavy`)}</td></tr>
            </tbody>
          </table>
        </div>
        <GqaMap hq={R.hq} hkv={R.hkv} />
        <Tex block>{'A = \\operatorname{softmax}\\!\\Big(\\frac{QK^{\\top}}{\\sqrt{d_h}} + M_{\\text{causal}}\\Big),\\qquad Z = AV'}</Tex>
        <p className="muted" style={{ fontSize: '0.9rem' }}>{L('Optimized kernels such as FlashAttention compute exactly this result in tiles with an online softmax, without storing the full Tq × Tk matrix; the square matrix shown above is the logical object, not a literal allocation every runtime makes.', 'Optimalizované jadrá ako FlashAttention vypočítajú presne tento výsledok po blokoch s priebežným softmaxom bez uloženia celej matice Tq × Tk; štvorcová matica vyššie je logický objekt, nie doslovná alokácia, ktorú robí každé prostredie.')}</p>
      </Section>
    </>
  );
}

function AttnMatrix({ T, values, sel, binary = false }: { T: number; values: (i: number, j: number) => number; sel: [number, number]; binary?: boolean }) {
  const { lang, fnum } = useLang();
  const cell = Math.min(34, Math.floor(300 / T));
  return (
    <svg width={cell * T + 30} height={cell * T + 30} role="img" aria-label={binary ? (lang === 'sk' ? 'Kauzálna maska: povolený je dolný trojuholník' : 'Causal mask: lower triangle allowed') : (lang === 'sk' ? 'Matica pravdepodobností pozornosti' : 'Attention probability matrix')}>
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
                {!future && !binary && cell >= 30 && <text x={30 + j * cell + cell / 2 - 1} y={26 + i * cell + cell / 2 + 3} fontSize={8} textAnchor="middle" fill="var(--text)">{fnum(v, 2).replace(/^0[.,]/, lang === 'sk' ? '0,' : '0.')}</text>}
              </g>
            );
          })}
        </g>
      ))}
    </svg>
  );
}

function GqaMap({ hq, hkv }: { hq: number; hkv: number }) {
  const { lang } = useLang();
  const g = hq / hkv;
  const w = Math.min(18, Math.floor(560 / hq));
  return (
    <svg width={hq * w + 20} height={86} role="img" aria-label={lang === 'sk' ? `${hq} hláv dopytov v ${hkv} skupinách po ${g}, každá skupina zdieľa jednu hlavu kľúčov a hodnôt` : `${hq} query heads in ${hkv} groups of ${g}, each group sharing one key/value head`} style={{ maxWidth: '100%' }}>
      {Array.from({ length: hq }, (_, h) => (
        <g key={h}>
          <rect x={10 + h * w} y={6} width={w - 3} height={18} rx={3} fill="color-mix(in srgb, var(--attn) 30%, transparent)" stroke="var(--attn)" />
          <line x1={10 + h * w + (w - 3) / 2} y1={24} x2={10 + (Math.floor(h / g) * g + g / 2) * w - 1.5} y2={56} stroke="var(--text-3)" />
        </g>
      ))}
      {Array.from({ length: hkv }, (_, k) => (
        <rect key={k} x={10 + k * g * w} y={56} width={g * w - 3} height={18} rx={3} fill="color-mix(in srgb, var(--state) 30%, transparent)" stroke="var(--state)" />
      ))}
      <text x={10} y={84} fontSize={10} fill="var(--text-3)">{lang === 'sk' ? `hore: ${hq} hláv dopytov · dole: ${hkv} zdieľaných hláv K/V (v cache)` : `top: ${hq} query heads · bottom: ${hkv} shared K/V heads (cached)`}</text>
    </svg>
  );
}

export function RopeLab() {
  const { lang, num, fnum } = useLang();
  const L = (en: string, sk: string) => (lang === 'sk' ? sk : en);
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
    <Section title={L('5 · RoPE lab: rotary position embedding', '5 · Laboratórium RoPE: rotačné kódovanie pozície')} badges={<><Evidence kind="verified" title="Convention: half-split pairing and θ^(−2i/d) frequencies as implemented" /><Evidence kind="simulation" title="The 2-D vectors are illustrative" /></>}>
      {lang === 'sk'
        ? <p>Qwen3 používa RoPE na každú hlavu dopytov a kľúčov <strong>po</strong> q/k RMSNorm. Súradnica i sa páruje so súradnicou i + d<sub>h</sub>/2 (konvencia <code>rotate_half</code> v Transformers, nie susedné dvojice) a dvojica sa otočí o uhol pozícia × ω<sub>i</sub>, kde ω<sub>i</sub> = θ<sup>−2i/d<sub>h</sub></sup>. V sa neotáča a pri embeddingu sa nič nepridáva: RoPE nemá naučené váhy.</p>
        : <p>Qwen3 applies RoPE to each query and key head <strong>after</strong> q/k RMSNorm. Coordinate i is paired with i + d<sub>h</sub>/2 (the Transformers <code>rotate_half</code> convention, not adjacent pairs) and the pair is rotated by angle position × ω<sub>i</sub>, with ω<sub>i</sub> = θ<sup>−2i/d<sub>h</sub></sup>. V is not rotated, and nothing is added at the embedding: RoPE has no learned weights.</p>}
      <Tex block>{"\\begin{pmatrix} q'_i \\\\ q'_{i+d/2} \\end{pmatrix} = \\begin{pmatrix} \\cos m\\omega_i & -\\sin m\\omega_i \\\\ \\sin m\\omega_i & \\cos m\\omega_i \\end{pmatrix} \\begin{pmatrix} q_i \\\\ q_{i+d/2} \\end{pmatrix}"}</Tex>
      <div className="row">
        <Select label={L('head dim', 'šírka hlavy')} value={dh} onChange={(v) => { setDh(v); setPair(0); }} options={[{ value: 128, label: '128 (Qwen3-8B / 30B-A3B)' }, { value: 4, label: L('4 (toy simulator)', '4 (hračkový simulátor)') }]} />
        <Select label="θ (rope_theta)" value={theta} onChange={setTheta} options={[{ value: 1_000_000, label: L('1,000,000 (Qwen3 config)', '1 000 000 (konfigurácia Qwen3)') }, { value: 10000, label: L('10,000 (toy; original RoPE)', '10 000 (hračka; pôvodné RoPE)') }]} />
        <Slider label={L(`pair i (coords ${i} & ${i + dh / 2})`, `dvojica i (súradnice ${i} a ${i + dh / 2})`)} value={i} min={0} max={dh / 2 - 1} onChange={setPair} />
      </div>
      <div className="row">
        <Slider label={L('query position m', 'pozícia dopytu m')} value={m} min={0} max={64} onChange={setM} />
        <Slider label={L('key position n', 'pozícia kľúča n')} value={n} min={0} max={64} onChange={setN} />
        <Slider label={L('shift both by', 'posunúť obe o')} value={shift} min={0} max={500} onChange={setShift} />
      </div>
      <div className="grid2">
        <svg viewBox="0 0 220 220" style={{ maxWidth: 300 }} role="img" aria-label={L('Rotation of a query pair and a key pair', 'Otočenie dvojice dopytu a dvojice kľúča')}>
          <defs><marker id="ra" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" fill="var(--text-2)" /></marker></defs>
          <circle cx={S} cy={S} r={80} fill="none" stroke="var(--line)" />
          <line x1={20} y1={S} x2={200} y2={S} stroke="var(--line-soft)" /><line x1={S} y1={20} x2={S} y2={200} stroke="var(--line-soft)" />
          {arrow(q, 'var(--text-3)', 'q')}
          {arrow(qm, 'var(--attn)', `q@${m}`)}
          {arrow(kn, 'var(--router)', `k@${n}`)}
        </svg>
        <div>
          <p className="mono" style={{ fontSize: '0.88rem' }}>
            ω_{i} = {num(theta)}^(−{2 * i}/{dh}) = {fnum(w, 5)} {L('rad/position', 'rad/pozícia')}<br />
            {L('wavelength', 'vlnová dĺžka')} = 2π/ω = {fnum(2 * Math.PI / w, 4)} {L('positions', 'pozícií')}<br />
            {L('angle at', 'uhol pri')} m={m}: {fnum(m * w, 4)} rad
          </p>
          <p>{L(`2-D illustration: q·k at (m, n) = (${m}, ${n}) is ${fnum(d1, 6)}; shifting both by ${shift} gives ${fnum(d2, 6)}. The score depends only on the offset m − n = ${m - n}.`, `2-D ilustrácia: q·k pri (m, n) = (${m}, ${n}) je ${fnum(d1, 6)}; posun oboch o ${shift} dáva ${fnum(d2, 6)}. Skóre závisí len od vzdialenosti m − n = ${m - n}.`)}</p>
          <p className="faint" style={{ fontSize: '0.86rem' }}>{L(`Full ${dh}-dimensional check with the implemented operator (synthetic vectors): ${fnum(full1, 8)} vs ${fnum(full2, 8)} (difference ${fnum(Math.abs(full1 - full2), 3)}, floating-point rounding).`, `Úplná ${dh}-rozmerná kontrola s implementovaným operátorom (syntetické vektory): ${fnum(full1, 8)} vs ${fnum(full2, 8)} (rozdiel ${fnum(Math.abs(full1 - full2), 3)}, zaokrúhľovanie s pohyblivou čiarkou).`)}</p>
          <div className="callout">{lang === 'sk' ? <>Pri dekódovaní s cache sa nový token otočí podľa svojej <strong>absolútnej</strong> pozície (= počet tokenov, ktoré už sú v cache), kým uložené kľúče si ponechajú otočenie zo svojich pozícií. Laboratórium Prefill/decode overuje, že to dáva rovnaký výsledok ako úplný prepočet.</> : <>During cached decoding the new token is rotated with its <strong>absolute</strong> position (= number of tokens already in the cache), while the cached keys keep the rotations they received at their own positions. The Prefill/decode lab tests that this reproduces full recomputation.</>}</div>
          <p className="faint" style={{ fontSize: '0.85rem' }}>{L('Context extension (YaRN, factor 4 to 131,072 tokens per the model card) rescales these frequencies at inference time; it is a runtime setting (rope_scaling), not enabled in the published config (rope_scaling: null).', 'Rozšírenie kontextu (YaRN, faktor 4 až na 131 072 tokenov podľa karty modelu) pri inferencii preškáluje tieto frekvencie; je to nastavenie počas behu (rope_scaling), ktoré v zverejnenej konfigurácii nie je zapnuté (rope_scaling: null).')}</p>
        </div>
      </div>
    </Section>
  );
}
