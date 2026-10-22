import { useMemo, useState } from 'react';
import { useToy } from '../state/toyStore';
import { Cite, Evidence, NumberField, ProbRow, Section, Select, SimBanner, Slider, Tex, VectorView, fmtNum } from '../components/ui';
import { ToyControls, PromptEditor, tok } from './common';
import { drawToken, samplingStages } from '../math/sampling';
import { TOY_VOCAB } from '../math/toy';
import { PARAM_SHAPES, TRAIN_SEQUENCE, finiteDifference, getParam, initTrainParams, lossAndGrad, paramCount, sgdStep, trainForward, type ParamName, type ParamRef } from '../math/train';

export function SamplingLab() {
  const { trace, state, dispatch } = useToy();
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
  return (
    <>
      <Section title="10 · Sampling lab: from logits to the next token" badges={<Evidence kind="simulation" />}>
        <ToyControls />
        <PromptEditor />
        <div className="row">
          <Select label="Logits from" value={src} onChange={setSrc} options={[{ value: 'toy', label: 'toy simulator (last position)' }, { value: 'custom', label: 'editable vector' }]} />
          <button onClick={() => { setT(0.6); setTopK(20); setTopP(0.95); }}>Qwen3 default (generation_config: T 0.6, k 20, p 0.95)</button>
          <button onClick={() => { setT(0.7); setTopK(20); setTopP(0.8); }}>Model-card non-thinking suggestion (T 0.7, k 20, p 0.8)</button>
          <button onClick={() => setT(0)}>Greedy (T = 0)</button>
        </div>
        <div className="row">
          <Slider label="temperature" value={T} min={0} max={2} step={0.05} onChange={setT} format={(v) => (v === 0 ? '0 (greedy)' : v.toFixed(2))} />
          <Slider label="top-k (0 = off)" value={topK} min={0} max={24} onChange={setTopK} />
          <Slider label="top-p" value={topP} min={0.05} max={1} step={0.01} onChange={setTopP} format={(v) => v.toFixed(2)} />
        </div>
        {src === 'custom' && (
          <div className="row" style={{ gap: 4 }}>
            {custom.map((v, i) => <NumberField key={i} label={TOY_VOCAB[i]} value={v} step={0.1} width={70} onChange={(x) => setCustom(custom.map((o, k) => (k === i ? x : o)))} />)}
          </div>
        )}
        <p className="muted" style={{ fontSize: '0.88rem' }}>Order of operations (as in Hugging Face generation): divide logits by temperature → softmax → keep the top-k → keep the smallest set reaching cumulative probability top-p → renormalize → draw. {topK > logits.length && `top-k ${topK} exceeds this toy vocabulary of ${logits.length}, so it removes nothing.`} Sampling is a decoding procedure, not a trained layer.</p>
        <div className="grid2">
          <div>
            <h3>Distribution after each step</h3>
            <div className="table-wrap"><table className="t">
              <thead><tr><th>token</th><th className="num">logit</th><th className="num">÷T</th><th className="num">p (all)</th><th>top-k</th><th>top-p</th><th className="num">final p</th></tr></thead>
              <tbody>{st.order.map((i) => (
                <tr key={i}><td className="mono">{TOY_VOCAB[i]}</td><td className="num">{fmtNum(logits[i], 3)}</td><td className="num">{st.scaled ? fmtNum(st.scaled[i], 3) : '—'}</td><td className="num">{fmtNum(st.probsAll[i], 3)}</td><td>{st.afterTopK[i] ? 'kept' : '✗'}</td><td>{st.afterTopP[i] ? 'kept' : '✗'}</td><td className="num"><strong>{fmtNum(st.final[i], 3)}</strong></td></tr>
              ))}</tbody>
            </table></div>
          </div>
          <div>
            <h3>Final distribution {st.greedy && '(greedy: argmax gets probability 1)'}</h3>
            {st.order.map((i) => <ProbRow key={i} label={TOY_VOCAB[i]} p={st.final[i]} off={st.final[i] === 0} />)}
            <div className="row" style={{ marginTop: 10 }}>
              <button className="primary" onClick={draw}>Draw next token (seeded)</button>
              <button disabled={!last || state.tokens.length >= 12} onClick={() => { if (last) { dispatch({ type: 'tokens', tokens: [...state.tokens, last.token] }); setLast(null); } }}>Append it and continue</button>
              <NumberField label="sampler seed" value={state.samplerSeed} min={0} step={1} width={90} onChange={(v) => dispatch({ type: 'samplerSeed', seed: Math.round(v) })} />
            </div>
            <p className="mono" style={{ fontSize: '0.86rem' }}>draws so far: {state.draws} · RNG state {state.rng.state}{last && <> · u = {last.u.toFixed(6)} → “{tok(last.token)}”</>}</p>
            <p className="faint" style={{ fontSize: '0.82rem' }}>Moving sliders or selecting rows never advances the RNG; only “Draw” does. Same seed + same distribution ⇒ same draw.</p>
          </div>
        </div>
      </Section>
      <Section title="Thinking vs non-thinking mode (Qwen3)" badges={<Evidence kind="verified" />}>
        <p>Qwen3 models switch between a reasoning (“thinking”) and a direct mode through the <strong>chat template</strong> (e.g. <code>enable_thinking=False</code> inserts an empty <code>&lt;think&gt;&lt;/think&gt;</code> block) or soft switches <code>/think</code>, <code>/no_think</code> in a message. The checkpoint is the same: no extra layers are inserted and no weights change. The difference is the text the model is conditioned on and the tokens it then generates (the report describes this as trained in by “thinking mode fusion”).<Cite ids={['qwen3-report', 'q8-card']} /></p>
        <p>The model card recommends against greedy decoding in thinking mode (repetition). See real token ids for both templates on the <a href="#/pipeline">Tokens &amp; generation</a> page.</p>
      </Section>
    </>
  );
}

const TENSORS: ParamName[] = ['embed', 'g1', 'gate', 'up', 'down', 'g2', 'head'];
const TLABEL: Record<ParamName, string> = { embed: 'embedding E [12, 4]', g1: 'RMSNorm γ₁ [4]', gate: 'W_gate [6, 4]', up: 'W_up [6, 4]', down: 'W_down [4, 6]', g2: 'final RMSNorm γ₂ [4]', head: 'output head [12, 4]' };

export function TrainingLab() {
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
  return (
    <>
      <Section title="11 · Training lab: how the weights change" badges={<Evidence kind="simulation" />}>
        <SimBanner><span>A tiny dense residual block (d = 4, FFN width 6, {paramCount()} parameters, <strong>no attention sub-layer</strong>) trained with plain SGD on one toy sentence. It reuses the forward units of the explorer (embedding, RMSNorm, SwiGLU, untied head, softmax cross-entropy). This is not the Qwen training recipe.</span></SimBanner>
        <div className="row" style={{ marginTop: 8 }}>
          <NumberField label="seed" value={seed} min={0} step={1} width={90} onChange={(v) => setSeed(Math.round(v))} />
          <button onClick={reset}>Reset to seed {seed}</button>
          <span className="faint">steps taken: {steps}</span>
        </div>
        <h3>Teacher forcing: targets are the inputs shifted by one</h3>
        <div className="table-wrap"><table className="t">
          <thead><tr><th>position</th><th>input</th><th>target</th><th className="num">p(target)</th><th className="num">loss −log p</th></tr></thead>
          <tbody>{fwd.positions.map((x) => (
            <tr key={x.t} style={{ background: x.t === posSel ? 'color-mix(in srgb, var(--focus) 10%, transparent)' : undefined }}>
              <td><button className="small ghost" onClick={() => setPosSel(x.t)}>{x.t}</button></td><td className="mono">{tok(x.tok)}</td><td className="mono">{tok(x.target)}</td><td className="num">{fmtNum(x.pTarget)}</td><td className="num">{fmtNum(x.loss)}</td>
            </tr>
          ))}
            <tr><td colSpan={4}><strong>mean cross-entropy loss</strong></td><td className="num"><strong>{fmtNum(fwd.loss)}</strong></td></tr>
          </tbody>
        </table></div>
        <Tex block>{'\\mathcal{L} = -\\frac{1}{T-1}\\sum_{t} \\log \\operatorname{softmax}(\\text{logits}_t)[\\text{target}_t]'}</Tex>
      </Section>
      <Section title="Backward pass: the chain rule through the same units" badges={<Evidence kind="simulation" />}>
        <p className="muted" style={{ fontSize: '0.9rem' }}>Gradient magnitudes ‖∂L/∂·‖ at position {pos.t} (“{tok(pos.tok)}” → “{tok(pos.target)}”), traversed from the loss back to the input:</p>
        <ol style={{ paddingLeft: 20, fontFamily: 'var(--mono)', fontSize: '0.88rem' }}>
          <li>softmax + cross-entropy → logits: {fmtNum(chain.dLogits)}  (∂L/∂logit = p − onehot(target), averaged)</li>
          <li>head W_vocab → final RMSNorm output: {fmtNum(chain.dFinalNorm)}</li>
          <li>final RMSNorm (γ₂, 1/rms) → residual h: {fmtNum(chain.dResidual)}  (the residual add copies this gradient to both branches)</li>
          <li>W_down → gated channels z: {fmtNum(chain.dGated)}</li>
          <li>z = g ⊙ u splits into ∂/∂g: {fmtNum(chain.dGateBranch)} (then × SiLU′) and ∂/∂u: {fmtNum(chain.dUpBranch)}</li>
          <li>W_gate, W_up → normalized input: {fmtNum(chain.dFfnInput)} → RMSNorm γ₁ → embedding row</li>
        </ol>
        <div className="row">
          <Select label="parameter tensor" value={ref.tensor} onChange={(t) => setRef({ tensor: t, row: 0, col: PARAM_SHAPES[t].length > 1 ? 0 : null })} options={TENSORS.map((t) => ({ value: t, label: TLABEL[t] }))} />
          <NumberField label="row" value={ref.row} min={0} max={sh[0] - 1} step={1} width={70} onChange={(v) => setRef({ ...ref, row: Math.round(v) })} />
          {sh.length > 1 && <NumberField label="column" value={ref.col ?? 0} min={0} max={sh[1]! - 1} step={1} width={70} onChange={(v) => setRef({ ...ref, col: Math.round(v) })} />}
        </div>
        <div className="card tight" style={{ marginTop: 8 }}>
          <div className="mono" style={{ fontSize: '0.92rem' }}>
            weight {ref.tensor}[{ref.row}{ref.col === null ? '' : `, ${ref.col}`}] = {fmtNum(getParam(params, ref), 6)}<br />
            ∂L/∂w (backprop) = <strong>{g.toExponential(6)}</strong> · sign {g > 0 ? '+ (increasing w increases loss)' : g < 0 ? '− (increasing w decreases loss)' : '0'}<br />
            ∂L/∂w (central finite difference, h = 1e-5, independent forward) = {fd.toExponential(6)}<br />
            relative difference = {(Math.abs(g - fd) / Math.max(1e-12, Math.abs(fd))).toExponential(2)} {Math.abs(g - fd) < 1e-6 + 1e-4 * Math.abs(fd) ? '✓' : '✗'}
          </div>
        </div>
        {ref.tensor === 'embed' && !TRAIN_SEQUENCE.slice(0, -1).includes(ref.row) && <p className="faint">Embedding row {ref.row} (“{tok(ref.row)}”) is never looked up in this sentence, so its gradient is exactly 0.</p>}
        <p className="faint" style={{ fontSize: '0.84rem' }}>Smooth units (RMSNorm, SiLU, softmax) give well-defined derivatives here. In an MoE, the top-k choice is discrete: gradients flow through the selected experts and α, not through which experts were picked.</p>
      </Section>
      <Section title="One SGD step: w_new = w_old − learning rate × gradient" badges={<Evidence kind="simulation" />}>
        <div className="row">
          <Slider label="learning rate" value={lr} min={0.01} max={30} step={0.01} onChange={setLr} format={(v) => v.toFixed(2)} />
          <button className="primary" onClick={step}>Take one step (updates all {paramCount()} weights)</button>
        </div>
        <p className="mono" style={{ fontSize: '0.88rem' }}>selected weight: {fmtNum(getParam(params, ref), 6)} − {lr} × {g.toExponential(3)} = {fmtNum(getParam(params, ref) - lr * g, 6)}</p>
        {history.length > 0 && (
          <table className="t"><thead><tr><th>step</th><th className="num">lr</th><th className="num">loss before</th><th className="num">loss after (recomputed)</th><th>change</th></tr></thead>
            <tbody>{history.map((h) => <tr key={h.step}><td>{h.step}</td><td className="num">{h.lr}</td><td className="num">{fmtNum(h.before)}</td><td className="num">{fmtNum(h.after)}</td><td>{h.after < h.before ? 'decreased' : 'increased or unchanged'}</td></tr>)}</tbody>
          </table>
        )}
        <h3>Compare learning rates (one step from the current weights, computed, not applied)</h3>
        <table className="t"><thead><tr><th className="num">lr</th><th className="num">loss after one step</th><th>vs now ({fmtNum(lg.loss)})</th></tr></thead>
          <tbody>{sweep.map((x) => <tr key={x.lr}><td className="num">{x.lr}</td><td className="num">{fmtNum(x.after)}</td><td>{x.after < lg.loss ? '↓ lower' : '↑ higher'}</td></tr>)}</tbody>
        </table>
        <p className="faint" style={{ fontSize: '0.86rem' }}>A larger step is not always better: the gradient only describes the loss locally. The table shows the actual recomputed losses.</p>
        <VectorView label="current output-head row for the first target (“cat”), after your steps" values={params.head[2]} kind="par" />
      </Section>
      <Section title="Optimizers and how Qwen3 was actually trained" badges={<><Evidence kind="verified" /><Evidence kind="unknown" /></>}>
        <ul style={{ paddingLeft: 18 }}>
          <li><strong>Adam-style optimizers</strong> keep two extra numbers per parameter (running averages of the gradient and of its square) and scale each update by them. These optimizer states are training memory, not model parameters, and are not in the released checkpoint.<Cite ids={['adam']} /></li>
          <li><strong>Pre-training</strong> (Qwen3 report): over 36 trillion tokens in 119 languages and dialects, in three stages: a general stage (over 30T tokens, 4,096-token sequences), a reasoning stage (about 5T higher-quality tokens with more STEM/code), and a long-context stage at 32,768 tokens, with the RoPE base raised from 10,000 to 1,000,000.<Cite ids={['qwen3-report']} /></li>
          <li><strong>Post-training</strong>: flagship models went through long chain-of-thought cold start, reasoning RL, thinking-mode fusion and general RL. Qwen3-8B and Qwen3-30B-A3B were instead produced by <strong>strong-to-weak distillation</strong> from larger Qwen3 teachers (off-policy, then on-policy logit alignment).<Cite ids={['qwen3-report']} /></li>
          <li>Parameter changes happen only in training. Prompting, chat templates and thinking mode change the <em>context</em>, not the weights.</li>
          <li><Evidence kind="unknown" /> Not claimed here: exact data mixtures, the per-model optimizer and learning-rate settings, and interpretations of individual neurons or experts. They are not reproduced from the sources checked, and nothing on this site fills them in.</li>
        </ul>
      </Section>
    </>
  );
}
