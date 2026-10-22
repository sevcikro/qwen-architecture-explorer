import { useMemo, useState } from 'react';
import { HYBRIDS, QWEN38_COMPONENTS, COVERAGE_LABEL, type HybridProfile } from '../models/hybrid';
import { Cite, Evidence, Section, SimBanner, Slider, Tex, VectorView, fmtB, fmtInt, fmtNum } from '../components/ui';
import { gatedDeltaRule, gdnGate, gatedResidualToy, qsaSelect, toyGdnInputs } from '../math/hybridToy';
import { sigmoid } from '../math/ops';
import { deriveSeed, makeRng, nextNormal } from '../math/rng';
import { EnglishOnlyNotice } from '../i18n/i18n';

function LayerStrip({ h }: { h: HybridProfile }) {
  const w = 16;
  return (
    <svg viewBox={`0 0 ${h.layers * w + 20} 70`} style={{ width: '100%', height: 'auto', maxWidth: 900 }} role="img" aria-label={`${h.layers} blocks: ${h.layerTypes.filter((t) => t === 'gdn').length} Gated DeltaNet and ${h.layerTypes.filter((t) => t === 'attn').length} attention blocks, each followed by MoE`}>
      {h.layerTypes.map((t, i) => (
        <g key={i}>
          <rect x={10 + i * w} y={8} width={w - 3} height={30} rx={3} fill={t === 'attn' ? 'color-mix(in srgb, var(--attn) 45%, transparent)' : 'color-mix(in srgb, var(--state) 35%, transparent)'} stroke={t === 'attn' ? 'var(--attn)' : 'var(--state)'} strokeDasharray={t === 'attn' ? undefined : '3 2'} />
          {i % 4 === 0 && <text x={10 + i * w} y={52} fontSize={8} fill="var(--text-3)">{i}</text>}
          {h.id === 'qwen3.8-flash-next' && i === 1 && <text x={10 + i * w + 6} y={66} fontSize={8} fill="var(--router)" textAnchor="middle">n-gram</text>}
        </g>
      ))}
    </svg>
  );
}

function GdnToy() {
  const [T, setT] = useState(6);
  const [aLog, setALog] = useState(0);
  const [step, setStep] = useState(5);
  const inp = useMemo(() => toyGdnInputs(17, 12), []);
  const g = inp.a.slice(0, T).map((a) => gdnGate(aLog, a, 0));
  const beta = inp.b.slice(0, T).map(sigmoid);
  const steps = gatedDeltaRule(inp.q.slice(0, T), inp.k.slice(0, T), inp.v.slice(0, T), g, beta);
  const s = steps[Math.min(step, T - 1)];
  const max = Math.max(1e-9, ...s.stateAfter.flat().map(Math.abs));
  return (
    <div>
      <SimBanner>Toy analogue of the verified recurrence (one head, d_k = d_v = 4, synthetic inputs). Qwen3.8 runs this per head with 48 value heads of 128 × 128 state in each of 36 blocks.</SimBanner>
      <div className="row" style={{ marginTop: 8 }}>
        <Slider label="tokens processed" value={T} min={1} max={12} onChange={(v) => { setT(v); setStep(Math.min(step, v - 1)); }} />
        <Slider label="inspect step t" value={Math.min(step, T - 1)} min={0} max={T - 1} onChange={setStep} />
        <Slider label="A_log (decay strength)" value={aLog} min={-3} max={2} step={0.1} onChange={setALog} format={(v) => v.toFixed(1)} />
      </div>
      <div className="grid2">
        <div>
          <p className="mono" style={{ fontSize: '0.86rem' }}>
            g_t = −exp(A_log)·softplus(a_t + dt_bias) = {fmtNum(g[Math.min(step, T - 1)])}<br />
            decay exp(g_t) = {fmtNum(s.decay)} (keeps this fraction of the old state)<br />
            write strength β_t = σ(b_t) = {fmtNum(s.beta)}
          </p>
          <VectorView label="what the decayed state predicts for k_t (Sᵀk)" values={s.kvMem} />
          <VectorView label="delta = β (v_t − Sᵀk): only the prediction error is written" values={s.delta} />
          <VectorView label="output o_t = Sᵀ q_t" values={s.out} />
        </div>
        <div>
          <div className="faint" style={{ fontSize: '0.84rem' }}>State S after step {s.t} (fixed size d_k × d_v, independent of how many tokens came before):</div>
          <svg viewBox="0 0 170 170" style={{ maxWidth: 220 }} role="img" aria-label="Recurrent state matrix heatmap">
            {s.stateAfter.map((row, i) => row.map((v, j) => (
              <g key={`${i}${j}`}>
                <rect x={5 + j * 40} y={5 + i * 40} width={38} height={38} rx={3} fill={v >= 0 ? `color-mix(in srgb, var(--state) ${Math.round((Math.abs(v) / max) * 90)}%, var(--bg-2))` : `color-mix(in srgb, var(--danger) ${Math.round((Math.abs(v) / max) * 90)}%, var(--bg-2))`} stroke="var(--line)" />
                <text x={24 + j * 40} y={28 + i * 40} fontSize={9} textAnchor="middle" fill="var(--text)">{v.toFixed(2)}</text>
              </g>
            )))}
          </svg>
        </div>
      </div>
      <Tex block>{'S_t = e^{g_t} S_{t-1} + k_t\\,\\big(\\beta_t\\,(v_t - (e^{g_t}S_{t-1})^{\\top}k_t)\\big)^{\\top},\\qquad o_t = S_t^{\\top} q_t'}</Tex>
      <p className="faint" style={{ fontSize: '0.84rem' }}>As implemented in <code>torch_recurrent_gated_delta_rule</code>: q and k are L2-normalized, q is scaled by 1/√d_k. Before the recurrence a short causal depthwise convolution (kernel 4) mixes the q/k/v channels; afterwards a gated RMSNorm (gate z from <code>in_proj_z</code>, sigmoid) and <code>out_proj</code> produce the block output. Prefill uses an equivalent chunked algorithm.<Cite ids={['tf-qwen4exp', 'gdn']} /></p>
    </div>
  );
}

function QsaToy() {
  const [visible, setVisible] = useState(41);
  const [budget, setBudget] = useState(16);
  const ratio = 4;
  const keys = useMemo(() => { const r = makeRng(deriveSeed(5, 'qsa')); return Array.from({ length: 64 }, () => Array.from({ length: 4 }, () => nextNormal(r))); }, []);
  const qh = useMemo(() => { const r = makeRng(deriveSeed(6, 'qsa-q')); return Array.from({ length: 4 }, () => Array.from({ length: 4 }, () => nextNormal(r))); }, []);
  const sel = qsaSelect(qh, keys, visible, budget, ratio);
  const chosen = new Set(sel.selectedTokens);
  return (
    <div>
      <SimBanner>Toy analogue of the verified selection rule (4 indexer query heads, synthetic keys, toy budget). Real QSA: 4-token micro-blocks, budget 2,048 tokens = 512 blocks, 4 indexer query heads + 1 shared key head of width 128.</SimBanner>
      <div className="row" style={{ marginTop: 8 }}>
        <Slider label="visible (earlier) tokens" value={visible} min={1} max={64} onChange={setVisible} />
        <Slider label="token budget" value={budget} min={4} max={64} step={4} onChange={setBudget} />
      </div>
      <svg viewBox={`0 0 ${64 * 13 + 10} 64`} style={{ width: '100%', height: 'auto' }} role="img" aria-label={`${sel.selectedTokens.length} of ${visible} visible tokens attended`}>
        {Array.from({ length: 64 }, (_, t) => {
          const vis = t < visible;
          const isTail = sel.tail.includes(t);
          const c = !vis ? 'var(--bg-2)' : chosen.has(t) ? (isTail ? 'var(--router)' : 'var(--attn)') : 'var(--panel-2)';
          return <rect key={t} x={5 + t * 13} y={8 + (t % 4 === 0 ? 0 : 0)} width={11} height={30} rx={2} fill={c} stroke={t % 4 === 0 ? 'var(--text-3)' : 'var(--line-soft)'} strokeDasharray={vis ? undefined : '2 2'} />;
        })}
        <text x={5} y={56} fontSize={10} fill="var(--text-3)">blue = selected blocks · orange = incomplete tail block (always kept) · grey = skipped · dashed = not yet visible (future)</text>
      </svg>
      <p className="mono" style={{ fontSize: '0.86rem' }}>{sel.blocks.length} complete blocks scored, keeping the top {Math.min(sel.blockBudget, sel.blocks.length)}; attended tokens: {sel.selectedTokens.length} of {visible}.</p>
      <p>Ordinary causal attention would read all {visible} earlier tokens. QSA first scores pooled 4-token blocks with a cheap indexer, score = Σ_heads ReLU(q_h · k̄_block)/√128, keeps the best ⌊2048/4⌋ = 512 blocks plus the tail, and then runs ordinary gated softmax attention <em>only</em> over those tokens (an extra mask on top of the causal mask). Below 2,048 visible tokens nothing is skipped.<Cite ids={['tf-qwen4exp', 'q38-report']} /></p>
    </div>
  );
}

function GatedResidualToy() {
  const d = 3, nb = 4;
  const rnd = useMemo(() => { const r = makeRng(deriveSeed(9, 'gr')); const m = (a: number, b: number, s = 0.6) => Array.from({ length: a }, () => Array.from({ length: b }, () => nextNormal(r) * s)); return { br: m(nb, d, 1), down: m(2, nb * d), up: m(nb * d, 2), inject: m(nb, nb * d) }; }, []);
  const res = gatedResidualToy(rnd.br, rnd.down, rnd.up, rnd.inject, (x) => x.map((v) => Math.tanh(2 * v)));
  return (
    <div>
      <SimBanner>Toy analogue (4 branches of width 3, synthetic weights, a stand-in block function). Real: 4 branches of width 2,560 carried as one [B, T, 10,240] tensor; bottleneck rank 320.</SimBanner>
      <div className="grid2" style={{ marginTop: 8 }}>
        <div>
          {res.normed.map((b, i) => <VectorView key={i} label={`branch ${i} (RMS-normalized)`} values={b} />)}
          <VectorView label="read gates (sigmoid, elementwise per branch)" values={res.readGate} />
          <VectorView label="block input = mean over branches of gate ⊙ branch" values={res.xin} />
        </div>
        <div>
          <VectorView label="block output y (stand-in function)" values={res.y} />
          <VectorView label="write gates 2·σ(w_b · n / 4), one scalar per branch" values={res.writeGate} />
          {res.after.map((b, i) => <VectorView key={i} label={`branch ${i} after: old + write_${i} · y`} values={b} />)}
        </div>
      </div>
      <p className="faint" style={{ fontSize: '0.84rem' }}>Every block does this twice (around its token mixer and around its MoE). The embedding is copied into all four branches at the start, and a final read-only mixer combines them before the output head. There is no separate input_layernorm: the grouped <code>hc_norm</code> inside each gated residual plays that role.<Cite ids={['tf-qwen4exp']} /></p>
    </div>
  );
}

export default function Hybrid({ id }: { id?: string }) {
  const h = HYBRIDS.find((x) => x.id === id) ?? HYBRIDS[2];
  const is38 = h.id === 'qwen3.8-flash-next';
  const total = QWEN38_COMPONENTS.reduce((s, c) => s + c.elements, 0);
  const group = (g: string) => QWEN38_COMPONENTS.filter((c) => c.group === g).reduce((s, c) => s + c.elements, 0);
  return (
    <div className="page">
      <nav className="row" style={{ marginBottom: 6 }}><a href="#/family">← Family map</a>{HYBRIDS.map((x) => <a key={x.id} href={`#/hybrid/${x.id}`} aria-current={x.id === h.id ? 'page' : undefined}>{x.name}</a>)}</nav>
      <h1>{h.name}</h1>
      <EnglishOnlyNotice />
      <div className="badge-row" style={{ marginBottom: 8 }}>
        <span className="coverage">{COVERAGE_LABEL[h.coverage]}</span>
        {h.extraToys.length > 0 && <span className="coverage">+ toy analogues: {h.extraToys.join(', ')}</span>}
        <Evidence kind="verified" />
      </div>
      <p className="muted">{h.repo} @ <span className="mono">{h.revision.slice(0, 12)}</span> · model_type <code>{h.modelType}</code> · {h.advertised}. A separate profile: nothing here applies to Qwen3-8B or Qwen3-30B-A3B.<Cite ids={h.sources} /></p>
      <Section title="Layer layout" badges={<Evidence kind="verified" />}>
        <p>{h.layout}</p>
        <LayerStrip h={h} />
        <dl className="kv">
          <dt>Width</dt><dd>{fmtInt(h.d)} · {h.residual}</dd>
          <dt>Gated DeltaNet</dt><dd>{h.gdn.vHeads} value heads, {h.gdn.qkHeads} query/key heads, head dim {h.gdn.headDim}, conv kernel {h.gdn.convKernel}</dd>
          <dt>Attention slots</dt><dd>{h.attnLabel}. {h.attn.qHeads} Q / {h.attn.kvHeads} KV heads, head dim {h.attn.headDim}, RoPE on {h.attn.rotaryDim} of {h.attn.headDim} dims, sigmoid output gate.</dd>
          <dt>MoE</dt><dd>{h.moe.experts} experts, {h.moe.routed} routed + {h.moe.shared} shared, expert width {h.moe.expertWidth}</dd>
          <dt>Vocabulary rows</dt><dd>{fmtInt(h.vocabRows)}</dd>
          <dt>Context</dt><dd>{h.context}</dd>
          <dt>Also</dt><dd>{h.extras.join(' · ')}</dd>
        </dl>
      </Section>
      {is38 && (
        <>
          <Section title="Component accounting (from the checkpoint headers)" badges={<><Evidence kind="derived" /><Evidence kind="unknown" title="How the model card counts '4B MTP'" /></>}>
            <div className="table-wrap"><table className="t">
              <thead><tr><th>component</th><th className="num">stored elements</th><th>group</th></tr></thead>
              <tbody>{QWEN38_COMPONENTS.map((c) => <tr key={c.key}><td>{c.label}</td><td className="num">{fmtInt(c.elements)}</td><td>{c.group}</td></tr>)}
                <tr><td><strong>Text backbone</strong> (everything except n-gram, MTP, vision)</td><td className="num"><strong>{fmtInt(group('backbone'))}</strong> ≈ {fmtB(group('backbone'))}</td><td>card: “125B”</td></tr>
                <tr><td><strong>N-gram tables + injection</strong></td><td className="num"><strong>{fmtInt(group('ngram'))}</strong></td><td>card: “51B n-gram embedding”</td></tr>
                <tr><td><strong>MTP module</strong> (mtp.* tensors)</td><td className="num"><strong>{fmtInt(group('mtp'))}</strong></td><td>card: “4B MTP”</td></tr>
                <tr><td>Vision encoder</td><td className="num">{fmtInt(group('vision'))}</td><td>not in the card's text count</td></tr>
                <tr><td>All stored elements</td><td className="num">{fmtInt(total)}</td><td>= header total</td></tr>
              </tbody>
            </table></div>
            <p>“125B” matches the text backbone (125.71B) and does <strong>not</strong> include the 51B n-gram tables, the MTP module or the vision encoder, so 125B is not the total of all components. <Evidence kind="unknown" /> The checkpoint's mtp.* tensors hold 2.61B elements; the card's “4B MTP” is consistent with also counting the shared embedding and output head (2.61 + 0.64 + 0.64 = 3.88B), but the sources checked do not state the convention. “6B activated” is quoted from the card and not re-derived here.<Cite ids={['q38-card', 'q38-index']} /></p>
          </Section>
          <Section title="1 · Gated DeltaNet: how the recurrent state updates" badges={<><Evidence kind="verified" /><Evidence kind="simulation" /></>}><GdnToy /></Section>
          <Section title="2 · Qwen Sparse Attention: selecting micro-blocks" badges={<><Evidence kind="verified" /><Evidence kind="simulation" /></>}><QsaToy /></Section>
          <Section title="3 · Routed and shared experts" badges={<Evidence kind="verified" />}>
            <Tex block>{'y = \\sum_{e\\in\\operatorname{TopK}(p,10)} \\alpha_e\\,\\operatorname{Expert}_e(x) \\;+\\; \\sigma\\!\\big(x\\,w_{shared}\\big)\\,\\operatorname{SharedExpert}(x)'}</Tex>
            <p>Every token always passes through the one shared expert (width 640), scaled by a learned sigmoid gate (<code>shared_expert_gate</code>, a [1, 2560] matrix), and through 10 of 512 routed experts chosen by the same softmax-top-k rule as Qwen3. Routed experts are stored packed (<code>gate_up_proj [512, 1280, 2560]</code>, <code>down_proj [512, 2560, 640]</code>).<Cite ids={['tf-qwen4exp', 'q38-index']} /> <code>norm_topk_prob</code> is absent from config.json, so the configuration-class default (<code>True</code>) applies: the 10 routed probabilities are renormalized to sum to 1.<Cite ids={['tf-qwen4exp']} /></p>
          </Section>
          <Section title="4 · Gated residual branches" badges={<><Evidence kind="verified" /><Evidence kind="simulation" /></>}><GatedResidualToy /></Section>
          <Section title="5 · N-gram tables are parameters, not state" badges={<Evidence kind="verified" />}>
            <p>At the decoder layer with code index <strong>1</strong> (config <code>ple_layer_ids = [2]</code> counts from 1; the implementation tests <code>layer_idx + 1 in ple_layer_ids</code>; tensors live at <code>model.language_model.layers.1.ple.*</code>), each token's bigram and trigram of token ids (ending at that token) are hashed (XOR of ids times fixed odd multipliers, then modulo 8 different primes ≈ 20,000,000 per n-gram order) into 16 lookups in learned tables (128 shards, 320,001,536 rows × 160 values ≈ 51.2B parameters). The concatenated vectors are projected and added to all four residual branches through a contextual sigmoid gate and a short dilated convolution.<Cite ids={['tf-qwen4exp', 'q38-config', 'q38-index']} /></p>
            <ul style={{ paddingLeft: 18 }}>
              <li><strong>Learned parameters</strong> (fixed after training): n-gram tables, key/value projections, norms, conv weights.</li>
              <li><strong>Input-dependent state</strong> (per sequence, at run time): the last 2 token ids, the convolution history, Gated DeltaNet recurrent states, QSA K/V and indexer keys.</li>
              <li>The model card motivates the tables as cheap to offload to host memory; per-token lookup cost is a few rows, not the full table.</li>
            </ul>
          </Section>
          <Section title="6 · Which components belong to what" badges={<Evidence kind="verified" />}>
            <ul style={{ paddingLeft: 18 }}>
              <li><strong>Text inference</strong>: embedding, 48 blocks (with gated residuals and the n-gram injection), final residual mixer, output head.</li>
              <li><strong>Vision encoding</strong>: <code>model.visual.*</code> (27 ViT blocks, patch size 16, output projected to 2,560) — used only for image/video inputs.</li>
              <li><strong>Multi-token prediction</strong>: <code>mtp.*</code> (one extra QSA + MoE layer) — an auxiliary multi-token-prediction module (“1 layer, trained with multi-steps” per the model card), not needed for ordinary next-token inference.</li>
            </ul>
          </Section>
          <Section title="Training recipe (as reported)" badges={<Evidence kind="verified" />}>
            <p>The report and model card describe Muon for 2-D linear weights, AdamW for embeddings, output head, router and gated-residual low-rank projections, Adam without weight decay for the n-gram tables, and no batch-size warm-up. Data mixtures and full hyperparameters are not reproduced here.<Cite ids={['q38-report', 'q38-card']} /></p>
          </Section>
        </>
      )}
      {!is38 && (
        <Section title="What is and is not covered" badges={<Evidence kind="unknown" />}>
          <p>This profile is a verified structural view from the pinned model card and config: layer layout, head counts, expert counts and widths. It shares the Gated DeltaNet recurrence and gated attention with Qwen3.8-Flash-Next (see that page's toy analogues) but has a single ordinary residual stream, no QSA and no n-gram tables. Exact parameter breakdowns from checkpoint headers were not extracted for this model.</p>
          <a className="btn" href="#/hybrid/qwen3.8-flash-next">See the Qwen3.8-Flash-Next deep dive</a>
        </Section>
      )}
      <Section title="Unknowns" badges={<Evidence kind="unknown" />}>
        <ul style={{ paddingLeft: 18 }}>
          <li>No real activation traces, routing decisions or recurrent states of this model are shown anywhere on this site.</li>
          <li>Learned feature meanings of experts, heads or n-gram rows are not known and not claimed.</li>
          {is38 && <li>Exact deployment memory depends on runtime state dtypes and kernels (see the memory lab for a derived estimate).</li>}
        </ul>
      </Section>
    </div>
  );
}
