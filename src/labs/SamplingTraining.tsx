import { useMemo, useState } from 'react';
import { useToy } from '../state/toyStore';
import { Cite, Evidence, NumberField, ProbRow, Section, Select, SimBanner, Slider, Tex, VectorView } from '../components/ui';
import { ToyControls, PromptEditor, tok } from './common';
import { drawToken, samplingStages } from '../math/sampling';
import { TOY_VOCAB } from '../math/toy';
import { PARAM_SHAPES, TRAIN_SEQUENCE, finiteDifference, getParam, initTrainParams, lossAndGrad, paramCount, sgdStep, trainForward, type ParamName, type ParamRef } from '../math/train';
import { useLang } from '../i18n/i18n';

export function SamplingLab() {
  const { trace, state, dispatch } = useToy();
  const { lang, fnum } = useLang();
  const L = (en: string, sk: string) => (lang === 'sk' ? sk : en);
  const dec = (s: string) => (lang === 'sk' ? s.replace('.', ',') : s);
  const [src, setSrc] = useState<'toy' | 'custom'>('toy');
  const [custom, setCustom] = useState<number[]>(() => TOY_VOCAB.map((_, i) => [2.1, 1.2, 0.3, -0.4, 1.8, 0.9, -1, 0.2, 0.5, -0.7, 0.1, 1.0][i]));
  const [T, setT] = useState(0.6);
  const [topK, setTopK] = useState(20);
  const [topP, setTopP] = useState(0.95);
  const [last, setLast] = useState<{ token: number; u: number } | null>(null);
  const logits = src === 'toy' ? trace[trace.length - 1].logits : custom;
  const st = useMemo(() => samplingStages(logits, { temperature: T, topK, topP }), [logits, T, topK, topP]);
  const draw = () => {
    const r = drawToken(st.final, state.rng);
    dispatch({ type: 'advanceRng', rng: r.rng });
    setLast({ token: r.token, u: r.u });
  };
  const kept = L('kept', 'ponechaný');
  return (
    <>
      <Section title={L('10 · Sampling lab: from logits to the next token', '10 · Laboratórium vzorkovania: od logitov k ďalšiemu tokenu')} badges={<Evidence kind="simulation" />}>
        <ToyControls />
        <PromptEditor />
        <div className="row">
          <Select label={L('Logits from', 'Logity z')} value={src} onChange={setSrc} options={[{ value: 'toy', label: L('toy simulator (last position)', 'hračkový simulátor (posledná pozícia)') }, { value: 'custom', label: L('editable vector', 'upraviteľný vektor') }]} />
          <button onClick={() => { setT(0.6); setTopK(20); setTopP(0.95); }}>{L('Qwen3 default (generation_config: T 0.6, k 20, p 0.95)', 'Predvoľba Qwen3 (generation_config: T 0,6, k 20, p 0,95)')}</button>
          <button onClick={() => { setT(0.7); setTopK(20); setTopP(0.8); }}>{L('Model-card non-thinking suggestion (T 0.7, k 20, p 0.8)', 'Odporúčanie karty modelu bez premýšľania (T 0,7, k 20, p 0,8)')}</button>
          <button onClick={() => setT(0)}>{L('Greedy (T = 0)', 'Hladové (T = 0)')}</button>
        </div>
        <div className="row">
          <Slider label={L('temperature', 'teplota')} value={T} min={0} max={2} step={0.05} onChange={setT} format={(v) => (v === 0 ? L('0 (greedy)', '0 (hladové)') : dec(v.toFixed(2)))} />
          <Slider label={L('top-k (0 = off)', 'top-k (0 = vypnuté)')} value={topK} min={0} max={24} onChange={setTopK} />
          <Slider label="top-p" value={topP} min={0.05} max={1} step={0.01} onChange={setTopP} format={(v) => dec(v.toFixed(2))} />
        </div>
        {src === 'custom' && (
          <div className="row" style={{ gap: 4 }}>
            {custom.map((v, i) => <NumberField key={i} label={TOY_VOCAB[i]} value={v} step={0.1} width={70} onChange={(x) => setCustom(custom.map((o, k) => (k === i ? x : o)))} />)}
          </div>
        )}
        <p className="muted" style={{ fontSize: '0.88rem' }}>{L(
          'Order of operations (as in Hugging Face generation): divide logits by temperature → softmax → keep the top-k → keep the smallest set reaching cumulative probability top-p → renormalize → draw. ',
          'Poradie krokov (ako pri generovaní v Hugging Face): logity vydeliť teplotou → softmax → ponechať top-k → ponechať najmenšiu množinu, ktorá dosiahne kumulatívnu pravdepodobnosť top-p → prepočítať → vylosovať. ',
        )}{topK > logits.length && L(`top-k ${topK} exceeds this toy vocabulary of ${logits.length}, so it removes nothing.`, `top-k ${topK} presahuje tento hračkový slovník s ${logits.length} slovami, takže nič neodstráni.`)} {L('Sampling is a decoding procedure, not a trained layer.', 'Vzorkovanie je postup dekódovania, nie naučená vrstva.')}</p>
        <div className="grid2">
          <div>
            <h3>{L('Distribution after each step', 'Rozdelenie po každom kroku')}</h3>
            <div className="table-wrap" tabIndex={0}><table className="t">
              <thead><tr><th>{L('token', 'token')}</th><th className="num">{L('logit', 'logit')}</th><th className="num">÷T</th><th className="num">{L('p (all)', 'p (všetky)')}</th><th>top-k</th><th>top-p</th><th className="num">{L('final p', 'výsledné p')}</th></tr></thead>
              <tbody>{st.order.map((i) => (
                <tr key={i}><td className="mono">{TOY_VOCAB[i]}</td><td className="num">{fnum(logits[i], 3)}</td><td className="num">{st.scaled ? fnum(st.scaled[i], 3) : '—'}</td><td className="num">{fnum(st.probsAll[i], 3)}</td><td>{st.afterTopK[i] ? kept : '✗'}</td><td>{st.afterTopP[i] ? kept : '✗'}</td><td className="num"><strong>{fnum(st.final[i], 3)}</strong></td></tr>
              ))}</tbody>
            </table></div>
          </div>
          <div>
            <h3>{L('Final distribution', 'Výsledné rozdelenie')} {st.greedy && L('(greedy: argmax gets probability 1)', '(hladové: argmax dostane pravdepodobnosť 1)')}</h3>
            {st.order.map((i) => <ProbRow key={i} label={TOY_VOCAB[i]} p={st.final[i]} off={st.final[i] === 0} />)}
            <div className="row" style={{ marginTop: 10 }}>
              <button className="primary" onClick={draw}>{L('Draw next token (seeded)', 'Vylosovať ďalší token (s osivom)')}</button>
              <button disabled={!last || state.tokens.length >= 12} onClick={() => { if (last) { dispatch({ type: 'tokens', tokens: [...state.tokens, last.token] }); setLast(null); } }}>{L('Append it and continue', 'Pripojiť ho a pokračovať')}</button>
              <NumberField label={L('sampler seed', 'osivo vzorkovača')} value={state.samplerSeed} min={0} step={1} width={90} onChange={(v) => dispatch({ type: 'samplerSeed', seed: Math.round(v) })} />
            </div>
            <p className="mono" style={{ fontSize: '0.86rem' }}>{L('draws so far', 'doterajšie losovania')}: {state.draws} · {L('RNG state', 'stav RNG')} {state.rng.state}{last && <> · u = {dec(last.u.toFixed(6))} → {lang === 'sk' ? `„${tok(last.token)}“` : `“${tok(last.token)}”`}</>}</p>
            <p className="faint" style={{ fontSize: '0.82rem' }}>{L('Moving sliders or selecting rows never advances the RNG; only “Draw” does. Same seed + same distribution ⇒ same draw.', 'Posúvanie jazdcov ani výber riadkov generátor náhodných čísel neposúva; robí to len „Vylosovať“. Rovnaké osivo + rovnaké rozdelenie ⇒ rovnaký výsledok.')}</p>
          </div>
        </div>
      </Section>
      <Section title={L('Thinking vs non-thinking mode (Qwen3)', 'Režim s premýšľaním a bez neho (Qwen3)')} badges={<Evidence kind="verified" />}>
        {lang === 'sk' ? (
          <>
            <p>Modely Qwen3 sa prepínajú medzi režimom uvažovania („premýšľania“) a priamym režimom cez <strong>šablónu chatu</strong> (napr. <code>enable_thinking=False</code> vloží prázdny blok <code>&lt;think&gt;&lt;/think&gt;</code>) alebo cez mäkké prepínače <code>/think</code>, <code>/no_think</code> v správe. Checkpoint je ten istý: nepridávajú sa žiadne vrstvy a nemenia sa žiadne váhy. Rozdiel je v texte, na ktorom je model podmienený, a v tokenoch, ktoré potom generuje (správa to opisuje ako natrénované „zlúčením režimov premýšľania“).<Cite ids={['qwen3-report', 'q8-card']} /></p>
            <p>Karta modelu neodporúča hladové dekódovanie v režime premýšľania (opakovanie). Skutočné ID tokenov pre obe šablóny nájdete na stránke <a href="#/pipeline">Tokeny a generovanie</a>.</p>
          </>
        ) : (
          <>
            <p>Qwen3 models switch between a reasoning (“thinking”) and a direct mode through the <strong>chat template</strong> (e.g. <code>enable_thinking=False</code> inserts an empty <code>&lt;think&gt;&lt;/think&gt;</code> block) or soft switches <code>/think</code>, <code>/no_think</code> in a message. The checkpoint is the same: no extra layers are inserted and no weights change. The difference is the text the model is conditioned on and the tokens it then generates (the report describes this as trained in by “thinking mode fusion”).<Cite ids={['qwen3-report', 'q8-card']} /></p>
            <p>The model card recommends against greedy decoding in thinking mode (repetition). See real token ids for both templates on the <a href="#/pipeline">Tokens &amp; generation</a> page.</p>
          </>
        )}
      </Section>
    </>
  );
}

const TENSORS: ParamName[] = ['embed', 'g1', 'gate', 'up', 'down', 'g2', 'head'];
const TLABEL: Record<ParamName, string> = { embed: 'embedding E [12, 4]', g1: 'RMSNorm γ₁ [4]', gate: 'W_gate [6, 4]', up: 'W_up [6, 4]', down: 'W_down [4, 6]', g2: 'final RMSNorm γ₂ [4]', head: 'output head [12, 4]' };
const TLABEL_SK: Record<ParamName, string> = { embed: 'embedding E [12, 4]', g1: 'RMSNorm γ₁ [4]', gate: 'W_gate [6, 4]', up: 'W_up [6, 4]', down: 'W_down [4, 6]', g2: 'záverečná RMSNorm γ₂ [4]', head: 'výstupná hlava [12, 4]' };

export function TrainingLab() {
  const { lang, fnum } = useLang();
  const T = (en: string, sk: string) => (lang === 'sk' ? sk : en);
  const ex = (x: number, d: number) => (lang === 'sk' ? x.toExponential(d).replace('.', ',') : x.toExponential(d));
  const q = (s: string) => (lang === 'sk' ? `„${s}“` : `“${s}”`);
  const [seed, setSeed] = useState(11);
  const [params, setParams] = useState(() => initTrainParams(11));
  const [steps, setSteps] = useState(0);
  const [history, setHistory] = useState<{ step: number; lr: number; before: number; after: number }[]>([]);
  const [lr, setLr] = useState(0.5);
  const [ref, setRef] = useState<ParamRef>({ tensor: 'gate', row: 2, col: 1 });
  const [posSel, setPosSel] = useState(0);
  const fwd = useMemo(() => trainForward(params, TRAIN_SEQUENCE), [params]);
  const lg = useMemo(() => lossAndGrad(params, TRAIN_SEQUENCE), [params]);
  const sh = PARAM_SHAPES[ref.tensor];
  const g = getParam(lg.grad, ref);
  const fd = useMemo(() => finiteDifference(params, TRAIN_SEQUENCE, ref), [params, ref]);
  const lrs = [0.01, 0.05, 0.1, 0.3, 1, 3, 10, 30];
  const sweep = useMemo(() => lrs.map((x) => ({ lr: x, after: trainForward(sgdStep(params, lg.grad, x), TRAIN_SEQUENCE).loss })), [params, lg]); // eslint-disable-line react-hooks/exhaustive-deps
  const step = () => {
    const before = lg.loss;
    const next = sgdStep(params, lg.grad, lr);
    const after = trainForward(next, TRAIN_SEQUENCE).loss;
    setParams(next);
    setSteps(steps + 1);
    setHistory([{ step: steps + 1, lr, before, after }, ...history].slice(0, 12));
  };
  const reset = () => { setParams(initTrainParams(seed)); setSteps(0); setHistory([]); };
  const pos = fwd.positions[Math.min(posSel, fwd.positions.length - 1)];
  const chain = lg.chain[Math.min(posSel, lg.chain.length - 1)];
  const lrStr = (x: number) => (lang === 'sk' ? String(x).replace('.', ',') : String(x));
  return (
    <>
      <Section title={T('11 · Training lab: how the weights change', '11 · Laboratórium trénovania: ako sa menia váhy')} badges={<Evidence kind="simulation" />}>
        <SimBanner>{lang === 'sk'
          ? <span>Malý hustý reziduálny blok (d = 4, šírka FFN 6, {paramCount()} parametrov, <strong>bez podvrstvy pozornosti</strong>) trénovaný obyčajným SGD na jednej hračkovej vete. Používa tie isté dopredné jednotky ako prieskumník (embedding, RMSNorm, SwiGLU, neprepojená hlava, softmax s krížovou entropiou). Nie je to postup trénovania Qwen.</span>
          : <span>A tiny dense residual block (d = 4, FFN width 6, {paramCount()} parameters, <strong>no attention sub-layer</strong>) trained with plain SGD on one toy sentence. It reuses the forward units of the explorer (embedding, RMSNorm, SwiGLU, untied head, softmax cross-entropy). This is not the Qwen training recipe.</span>}</SimBanner>
        <div className="row" style={{ marginTop: 8 }}>
          <NumberField label={T('seed', 'osivo')} value={seed} min={0} step={1} width={90} onChange={(v) => setSeed(Math.round(v))} />
          <button onClick={reset}>{T(`Reset to seed ${seed}`, `Obnoviť na osivo ${seed}`)}</button>
          <span className="faint">{T('steps taken', 'vykonané kroky')}: {steps}</span>
        </div>
        <h3>{T('Teacher forcing: targets are the inputs shifted by one', 'Učiteľské vedenie: ciele sú vstupy posunuté o jeden')}</h3>
        <div className="table-wrap" tabIndex={0}><table className="t">
          <thead><tr><th>{T('position', 'pozícia')}</th><th>{T('input', 'vstup')}</th><th>{T('target', 'cieľ')}</th><th className="num">p({T('target', 'cieľ')})</th><th className="num">{T('loss −log p', 'chyba −log p')}</th></tr></thead>
          <tbody>{fwd.positions.map((x) => (
            <tr key={x.t} style={{ background: x.t === posSel ? 'color-mix(in srgb, var(--focus) 10%, transparent)' : undefined }}>
              <td><button className="small ghost" onClick={() => setPosSel(x.t)}>{x.t}</button></td><td className="mono">{tok(x.tok)}</td><td className="mono">{tok(x.target)}</td><td className="num">{fnum(x.pTarget)}</td><td className="num">{fnum(x.loss)}</td>
            </tr>
          ))}
            <tr><td colSpan={4}><strong>{T('mean cross-entropy loss', 'priemerná chyba krížovej entropie')}</strong></td><td className="num"><strong>{fnum(fwd.loss)}</strong></td></tr>
          </tbody>
        </table></div>
        <Tex block>{'\\mathcal{L} = -\\frac{1}{T-1}\\sum_{t} \\log \\operatorname{softmax}(\\text{logits}_t)[\\text{target}_t]'}</Tex>
      </Section>
      <Section title={T('Backward pass: the chain rule through the same units', 'Spätný prechod: reťazové pravidlo cez tie isté jednotky')} badges={<Evidence kind="simulation" />}>
        <p className="muted" style={{ fontSize: '0.9rem' }}>{T(`Gradient magnitudes ‖∂L/∂·‖ at position ${pos.t} (“${tok(pos.tok)}” → “${tok(pos.target)}”), traversed from the loss back to the input:`, `Veľkosti gradientov ‖∂L/∂·‖ na pozícii ${pos.t} („${tok(pos.tok)}“ → „${tok(pos.target)}“), od chyby späť k vstupu:`)}</p>
        <ol style={{ paddingLeft: 20, fontFamily: 'var(--mono)', fontSize: '0.88rem' }}>
          <li>{T('softmax + cross-entropy → logits', 'softmax + krížová entropia → logity')}: {fnum(chain.dLogits)}  {T('(∂L/∂logit = p − onehot(target), averaged)', '(∂L/∂logit = p − onehot(cieľ), spriemerované)')}</li>
          <li>{T('head W_vocab → final RMSNorm output', 'hlava W_vocab → výstup záverečnej RMSNorm')}: {fnum(chain.dFinalNorm)}</li>
          <li>{T('final RMSNorm (γ₂, 1/rms) → residual h', 'záverečná RMSNorm (γ₂, 1/rms) → reziduálny h')}: {fnum(chain.dResidual)}  {T('(the residual add copies this gradient to both branches)', '(reziduálne sčítanie skopíruje tento gradient do oboch vetiev)')}</li>
          <li>{T('W_down → gated channels z', 'W_down → hradlované kanály z')}: {fnum(chain.dGated)}</li>
          <li>{T('z = g ⊙ u splits into ∂/∂g', 'z = g ⊙ u sa rozdelí na ∂/∂g')}: {fnum(chain.dGateBranch)} {T('(then × SiLU′) and ∂/∂u', '(potom × SiLU′) a ∂/∂u')}: {fnum(chain.dUpBranch)}</li>
          <li>{T('W_gate, W_up → normalized input', 'W_gate, W_up → normalizovaný vstup')}: {fnum(chain.dFfnInput)} → RMSNorm γ₁ → {T('embedding row', 'riadok embeddingu')}</li>
        </ol>
        <div className="row">
          <Select label={T('parameter tensor', 'tenzor parametrov')} value={ref.tensor} onChange={(t) => setRef({ tensor: t, row: 0, col: PARAM_SHAPES[t].length > 1 ? 0 : null })} options={TENSORS.map((t) => ({ value: t, label: (lang === 'sk' ? TLABEL_SK : TLABEL)[t] }))} />
          <NumberField label={T('row', 'riadok')} value={ref.row} min={0} max={sh[0] - 1} step={1} width={70} onChange={(v) => setRef({ ...ref, row: Math.round(v) })} />
          {sh.length > 1 && <NumberField label={T('column', 'stĺpec')} value={ref.col ?? 0} min={0} max={sh[1]! - 1} step={1} width={70} onChange={(v) => setRef({ ...ref, col: Math.round(v) })} />}
        </div>
        <div className="card tight" style={{ marginTop: 8 }}>
          <div className="mono" style={{ fontSize: '0.92rem' }}>
            {T('weight', 'váha')} {ref.tensor}[{ref.row}{ref.col === null ? '' : `, ${ref.col}`}] = {fnum(getParam(params, ref), 6)}<br />
            ∂L/∂w ({T('backprop', 'spätné šírenie')}) = <strong>{ex(g, 6)}</strong> · {T('sign', 'znamienko')} {g > 0 ? T('+ (increasing w increases loss)', '+ (zväčšenie w zväčší chybu)') : g < 0 ? T('− (increasing w decreases loss)', '− (zväčšenie w zmenší chybu)') : '0'}<br />
            ∂L/∂w ({T('central finite difference, h = 1e-5, independent forward', 'centrálna konečná diferencia, h = 1e-5, nezávislý dopredný prechod')}) = {ex(fd, 6)}<br />
            {T('relative difference', 'relatívny rozdiel')} = {ex(Math.abs(g - fd) / Math.max(1e-12, Math.abs(fd)), 2)} {Math.abs(g - fd) < 1e-6 + 1e-4 * Math.abs(fd) ? '✓' : '✗'}
          </div>
        </div>
        {ref.tensor === 'embed' && !TRAIN_SEQUENCE.slice(0, -1).includes(ref.row) && <p className="faint">{T(`Embedding row ${ref.row} (“${tok(ref.row)}”) is never looked up in this sentence, so its gradient is exactly 0.`, `Riadok embeddingu ${ref.row} („${tok(ref.row)}“) sa v tejto vete nikdy nevyhľadá, takže jeho gradient je presne 0.`)}</p>}
        <p className="faint" style={{ fontSize: '0.84rem' }}>{T('Smooth units (RMSNorm, SiLU, softmax) give well-defined derivatives here. In an MoE, the top-k choice is discrete: gradients flow through the selected experts and α, not through which experts were picked.', 'Hladké jednotky (RMSNorm, SiLU, softmax) tu dávajú dobre definované derivácie. V MoE je výber top-k diskrétny: gradienty tečú cez vybraných expertov a α, nie cez to, ktorí experti boli vybraní.')}</p>
      </Section>
      <Section title={T('One SGD step: w_new = w_old − learning rate × gradient', 'Jeden krok SGD: w_nové = w_staré − rýchlosť učenia × gradient')} badges={<Evidence kind="simulation" />}>
        <div className="row">
          <Slider label={T('learning rate', 'rýchlosť učenia')} value={lr} min={0.01} max={30} step={0.01} onChange={setLr} format={(v) => (lang === 'sk' ? v.toFixed(2).replace('.', ',') : v.toFixed(2))} />
          <button className="primary" onClick={step}>{T(`Take one step (updates all ${paramCount()} weights)`, `Urobiť jeden krok (zmení všetkých ${paramCount()} váh)`)}</button>
        </div>
        <p className="mono" style={{ fontSize: '0.88rem' }}>{T('selected weight', 'vybraná váha')}: {fnum(getParam(params, ref), 6)} − {lrStr(lr)} × {ex(g, 3)} = {fnum(getParam(params, ref) - lr * g, 6)}</p>
        {history.length > 0 && (
          <table className="t"><thead><tr><th>{T('step', 'krok')}</th><th className="num">lr</th><th className="num">{T('loss before', 'chyba pred')}</th><th className="num">{T('loss after (recomputed)', 'chyba po (prepočítaná)')}</th><th>{T('change', 'zmena')}</th></tr></thead>
            <tbody>{history.map((h) => <tr key={h.step}><td>{h.step}</td><td className="num">{lrStr(h.lr)}</td><td className="num">{fnum(h.before)}</td><td className="num">{fnum(h.after)}</td><td>{h.after < h.before ? T('decreased', 'klesla') : T('increased or unchanged', 'stúpla alebo bez zmeny')}</td></tr>)}</tbody>
          </table>
        )}
        <h3>{T('Compare learning rates (one step from the current weights, computed, not applied)', 'Porovnanie rýchlostí učenia (jeden krok z aktuálnych váh, vypočítaný, nepoužitý)')}</h3>
        <table className="t"><thead><tr><th className="num">lr</th><th className="num">{T('loss after one step', 'chyba po jednom kroku')}</th><th>{T('vs now', 'oproti teraz')} ({fnum(lg.loss)})</th></tr></thead>
          <tbody>{sweep.map((x) => <tr key={x.lr}><td className="num">{lrStr(x.lr)}</td><td className="num">{fnum(x.after)}</td><td>{x.after < lg.loss ? T('↓ lower', '↓ nižšia') : T('↑ higher', '↑ vyššia')}</td></tr>)}</tbody>
        </table>
        <p className="faint" style={{ fontSize: '0.86rem' }}>{T('A larger step is not always better: the gradient only describes the loss locally. The table shows the actual recomputed losses.', 'Väčší krok nie je vždy lepší: gradient opisuje chybu len lokálne. Tabuľka ukazuje skutočné prepočítané chyby.')}</p>
        <VectorView label={T('current output-head row for the first target (“cat”), after your steps', `aktuálny riadok výstupnej hlavy pre prvý cieľ (${q('cat')}) po vašich krokoch`)} values={params.head[2]} kind="par" />
      </Section>
      <Section title={T('Optimizers and how Qwen3 was actually trained', 'Optimalizátory a ako sa Qwen3 naozaj trénoval')} badges={<><Evidence kind="verified" /><Evidence kind="unknown" /></>}>
        {lang === 'sk' ? (
          <ul style={{ paddingLeft: 18 }}>
            <li><strong>Optimalizátory typu Adam</strong> si držia dve ďalšie čísla na každý parameter (kĺzavé priemery gradientu a jeho druhej mocniny) a nimi škálujú každú zmenu. Tieto stavy optimalizátora sú pamäť trénovania, nie parametre modelu, a v zverejnenom checkpointe nie sú.<Cite ids={['adam']} /></li>
            <li><strong>Predtrénovanie</strong> (správa o Qwen3): viac než 36 biliónov tokenov v 119 jazykoch a nárečiach, v troch fázach: všeobecná fáza (viac než 30 bil. tokenov, sekvencie s 4 096 tokenmi), fáza uvažovania (asi 5 bil. kvalitnejších tokenov s viac STEM/kódom) a fáza dlhého kontextu s 32 768 tokenmi, so základom RoPE zvýšeným z 10 000 na 1 000 000.<Cite ids={['qwen3-report']} /></li>
            <li><strong>Dotrénovanie</strong>: vlajkové modely prešli studeným štartom s dlhým reťazcom úvah, RL na uvažovanie, zlúčením režimov premýšľania a všeobecným RL. Qwen3-8B a Qwen3-30B-A3B namiesto toho vznikli <strong>destiláciou zo silného na slabý</strong> z väčších učiteľov Qwen3 (najprv off-policy, potom on-policy zarovnanie logitov).<Cite ids={['qwen3-report']} /></li>
            <li>Parametre sa menia len pri trénovaní. Zadania, šablóny chatu a režim premýšľania menia <em>kontext</em>, nie váhy.</li>
            <li><Evidence kind="unknown" /> Netvrdí sa tu: presné zloženie dát, nastavenia optimalizátora a rýchlosti učenia pre jednotlivé modely ani interpretácie jednotlivých neurónov či expertov. Overené zdroje ich neuvádzajú a táto stránka si ich nedomýšľa.</li>
          </ul>
        ) : (
          <ul style={{ paddingLeft: 18 }}>
            <li><strong>Adam-style optimizers</strong> keep two extra numbers per parameter (running averages of the gradient and of its square) and scale each update by them. These optimizer states are training memory, not model parameters, and are not in the released checkpoint.<Cite ids={['adam']} /></li>
            <li><strong>Pre-training</strong> (Qwen3 report): over 36 trillion tokens in 119 languages and dialects, in three stages: a general stage (over 30T tokens, 4,096-token sequences), a reasoning stage (about 5T higher-quality tokens with more STEM/code), and a long-context stage at 32,768 tokens, with the RoPE base raised from 10,000 to 1,000,000.<Cite ids={['qwen3-report']} /></li>
            <li><strong>Post-training</strong>: flagship models went through long chain-of-thought cold start, reasoning RL, thinking-mode fusion and general RL. Qwen3-8B and Qwen3-30B-A3B were instead produced by <strong>strong-to-weak distillation</strong> from larger Qwen3 teachers (off-policy, then on-policy logit alignment).<Cite ids={['qwen3-report']} /></li>
            <li>Parameter changes happen only in training. Prompting, chat templates and thinking mode change the <em>context</em>, not the weights.</li>
            <li><Evidence kind="unknown" /> Not claimed here: exact data mixtures, the per-model optimizer and learning-rate settings, and interpretations of individual neurons or experts. They are not reproduced from the sources checked, and nothing on this site fills them in.</li>
          </ul>
        )}
      </Section>
    </>
  );
}
