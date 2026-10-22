import { useMemo, useState } from 'react';
import { useToy } from '../state/toyStore';
import { Cite, Evidence, NumberField, ProbRow, Section, SimBanner, Slider, Tex, VectorView, fmtB, fmtInt, fmtNum } from '../components/ui';
import { ToyControls, PromptEditor, tok } from './common';
import { QWEN3_30B_A3B, QWEN3_8B, TOY_DENSE, TOY_MOE } from '../models/profiles';
import { accounting, packedExpertMapping } from '../models/shapes';
import { forwardFull, initToyWeights } from '../math/toy';
import { routeFromLogits } from '../math/ops';
import { makeRng, nextNormal } from '../math/rng';
import { swiglu } from '../math/ops';
import { explorerHref } from '../state/route';

export function CompareLab() {
  const { state } = useToy();
  const a8 = accounting(QWEN3_8B), a30 = accounting(QWEN3_30B_A3B);
  const ad = accounting(TOY_DENSE), am = accounting(TOY_MOE);
  const dense = useMemo(() => forwardFull(TOY_DENSE, initToyWeights(TOY_DENSE, state.seed), state.tokens), [state.seed, state.tokens]);
  const moe = useMemo(() => forwardFull(TOY_MOE, initToyWeights(TOY_MOE, state.seed), state.tokens), [state.seed, state.tokens]);
  const row = (label: string, d: string, m: string) => <tr><td>{label}</td><td className="num">{d}</td><td className="num">{m}</td></tr>;
  return (
    <>
      <Section title="6 · Dense vs MoE: the two reference models" badges={<><Evidence kind="verified" /><Evidence kind="derived" /></>}>
        <div className="callout warn">Not a controlled experiment: the two models differ in residual width, depth and K/V heads, not only in the FFN. Use the toy experiment below to isolate the FFN difference.</div>
        <div className="table-wrap"><table className="t">
          <thead><tr><th /><th className="num">Qwen3-8B (dense)</th><th className="num">Qwen3-30B-A3B (MoE)</th></tr></thead>
          <tbody>
            {row('Decoder blocks', '36', '48')}
            {row('Residual width d', '4,096', '2,048')}
            {row('Query / K/V heads', '32 / 8', '32 / 4')}
            {row('Query width Hq·dh', '4,096 (= d)', '4,096 (≠ d)')}
            {row('FFN per block', 'one SwiGLU, f = 12,288', '128 experts of width 768, top-8, no shared expert')}
            {row('FFN/expert params stored per block', fmtInt(a8.ffnPerBlock), fmtInt(a30.ffnPerBlock))}
            {row('FFN/expert params used per token per block', fmtInt(a8.ffnPerBlock), `${fmtInt(8 * a30.oneExpert!)} (8 experts) + router ${fmtInt(a30.routerPerBlock)}`)}
            {row('Total stored parameters', fmtInt(a8.total), fmtInt(a30.total))}
            {row('K/V cache per token (BF16)', '144 KiB', '96 KiB')}
            {row('Advertised (model card)', '8.2B', '30.5B total, 3.3B activated')}
          </tbody>
        </table></div>
      </Section>
      <Section title="Controlled toy comparison (matched dimensions)" badges={<Evidence kind="simulation" />}>
        <SimBanner>Both toys share d = 8, 2 blocks, 2 Q / 1 K/V heads and the seed {state.seed}. Only the FFN sub-layer differs: dense width 16 vs 4 experts of width 8 with top-2 (active width 16 = dense width).</SimBanner>
        <PromptEditor />
        <div className="table-wrap" style={{ marginTop: 8 }}><table className="t">
          <thead><tr><th /><th className="num">Toy dense</th><th className="num">Toy MoE</th></tr></thead>
          <tbody>
            {row('FFN params stored per block', fmtInt(ad.ffnPerBlock), `${fmtInt(am.ffnPerBlock)} + router ${am.routerPerBlock}`)}
            {row('FFN params used per token per block', fmtInt(ad.ffnPerBlock), `${fmtInt(2 * am.oneExpert!)} + router ${am.routerPerBlock}`)}
            {row('Total parameters', fmtInt(ad.total), fmtInt(am.total))}
            {row('Attention params per block (identical)', fmtInt(ad.attentionPerBlock), fmtInt(am.attentionPerBlock))}
          </tbody>
        </table></div>
        <h3>Which experts each token used (toy MoE, computed)</h3>
        <div className="table-wrap"><table className="t">
          <thead><tr><th>pos</th><th>token</th>{moe[0].layers.map((_, l) => <th key={l}>block {l}: experts (α)</th>)}</tr></thead>
          <tbody>{moe.map((pt) => (
            <tr key={pt.pos}><td>{pt.pos}</td><td className="mono">{tok(pt.token)}</td>{pt.layers.map((L, l) => <td key={l} className="mono">{L.moe!.route.selected.map((e, i) => `${e} (${L.moe!.route.alpha[i].toFixed(2)})`).join(', ')}</td>)}</tr>
          ))}</tbody>
        </table></div>
        <p className="faint" style={{ fontSize: '0.86rem' }}>Last-position logits differ between the two toys ({fmtNum(dense[dense.length - 1].logits[0])} vs {fmtNum(moe[moe.length - 1].logits[0])} for “{tok(0)}”) because they are different models with different random weights, not because one is “better”: no training happened here.</p>
      </Section>
    </>
  );
}

export function RouterLab() {
  const { profile, trace, state, dispatch, weights } = useToy();
  const [layer, setLayer] = useState(0);
  const [pos, setPos] = useState(state.tokens.length - 1);
  const [fixSeed, setFixSeed] = useState(3);
  const [k, setK] = useState(8);
  const [nTok, setNTok] = useState(16);
  const [focusE, setFocusE] = useState(0);
  const m = QWEN3_30B_A3B.moe!;
  const fixture = useMemo(() => {
    const rng = makeRng(fixSeed);
    return Array.from({ length: nTok }, () => Array.from({ length: m.numExperts }, () => nextNormal(rng) * 1.6));
  }, [fixSeed, nTok, m.numExperts]);
  const r0 = routeFromLogits(fixture[0], k, true);
  const union = new Set(fixture.flatMap((l) => routeFromLogits(l, k, true).selected));
  const a30 = accounting(QWEN3_30B_A3B);
  const isMoe = profile.kind === 'moe';
  const P = trace[Math.min(pos, trace.length - 1)];
  const L = isMoe ? P.layers[layer] : null;
  const mapping = packedExpertMapping(QWEN3_30B_A3B, focusE)!;
  return (
    <>
      <Section title="7 · Router lab (toy simulator: 4 experts, top-2)" badges={<Evidence kind="simulation" />}>
        <ToyControls />
        {!isMoe ? (
          <div className="callout warn">The shared simulator is in its dense variant. <button className="small" onClick={() => dispatch({ type: 'variant', v: 'toy-moe' })}>Switch to the MoE toy</button></div>
        ) : (
          <>
            <PromptEditor />
            <div className="row">
              <NumberField label="block" value={layer} min={0} max={profile.layers - 1} step={1} onChange={(v) => setLayer(Math.round(v))} width={70} />
              <NumberField label="token position" value={Math.min(pos, state.tokens.length - 1)} min={0} max={state.tokens.length - 1} step={1} onChange={(v) => setPos(Math.round(v))} width={80} />
            </div>
            <div className="grid2" style={{ marginTop: 8 }}>
              <div>
                <VectorView label={`router logits r = x W_routerᵀ for “${tok(P.token)}” (raw scores, any sign)`} values={L!.moe!.route.logits} />
                <h3>Probabilities over all experts</h3>
                {L!.moe!.route.probs.map((p, e) => <ProbRow key={e} label={`expert ${e}`} p={p} off={!L!.moe!.route.selected.includes(e)} />)}
                <h3>Selected and renormalized (α)</h3>
                {L!.moe!.route.selected.map((e, i) => <ProbRow key={e} label={`expert ${e}`} p={L!.moe!.route.alpha[i]} extra={`${fmtNum(L!.moe!.route.selectedProbs[i])} → ${fmtNum(L!.moe!.route.alpha[i])}`} />)}
                <p className="faint" style={{ fontSize: '0.84rem' }}>α sums to {fmtNum(L!.moe!.route.alpha.reduce((a, b) => a + b, 0))}. Learned router parameters = the matrix W_router; per-token mixture coefficients = α. They are different objects.</p>
              </div>
              <div>
                <h3>Expert outputs and their mixture</h3>
                {L!.moe!.route.selected.map((e, i) => (
                  <VectorView key={e} label={`α = ${fmtNum(L!.moe!.route.alpha[i], 3)} × Expert ${e}(x)`} values={L!.moe!.expertOut[i]} />
                ))}
                <VectorView label="y = Σ α_e · Expert_e(x) (MoE output, before the residual add)" values={L!.moe!.y} />
                <VectorView label="an unselected expert's output, for comparison (not used)" values={swiglu(L!.ln2, ...(['gate', 'up', 'down'] as const).map((w) => weights.layers[layer].experts![[0, 1, 2, 3].find((x) => !L!.moe!.route.selected.includes(x))!][w]) as [number[][], number[][], number[][]]).y} />
              </div>
            </div>
          </>
        )}
      </Section>
      <Section title={`Official-size fixture: ${m.numExperts} experts, top-${m.topK}`} badges={<><Evidence kind="verified" title="Counts, shapes and rule from config + implementation" /><Evidence kind="simulation" title="Router logits are synthetic" /></>}>
        <SimBanner>Router logits below are synthetic (seed {fixSeed}); the counts ({m.numExperts}, top-{m.topK}), renormalization rule and expert shapes are Qwen3-30B-A3B's. No real routing decision is shown.</SimBanner>
        <div className="row" style={{ marginTop: 8 }}>
          <NumberField label="fixture seed" value={fixSeed} min={0} step={1} onChange={(v) => setFixSeed(Math.round(v))} width={90} />
          <Slider label="top-k (educational)" value={k} min={1} max={16} onChange={setK} />
          <Slider label="tokens in batch" value={nTok} min={1} max={64} onChange={setNTok} />
        </div>
        {k !== m.topK && <div className="banner-edit" style={{ margin: '6px 0' }}>Custom educational routing rule (top-{k}). The pretrained model was configured and evaluated with top-{m.topK}; this does not claim anything about running it with top-{k}.</div>}
        <div className="grid2">
          <div>
            <p className="faint" style={{ fontSize: '0.85rem' }}>Token 0: all {m.numExperts} probabilities are computed; {k} are selected (orange). Click any expert to inspect its structure.</p>
            <div className="expert-grid" role="group" aria-label="Expert overview">
              {Array.from({ length: m.numExperts }, (_, e) => (
                <button key={e} className={`${r0.selected.includes(e) ? 'sel' : ''}${focusE === e ? ' focus' : ''}`} aria-pressed={focusE === e} aria-label={`expert ${e}${r0.selected.includes(e) ? ', selected' : ''}, probability ${r0.probs[e].toFixed(4)}`} onClick={() => setFocusE(e)}>{e}</button>
              ))}
            </div>
            <p className="mono" style={{ fontSize: '0.86rem' }}>selected: {r0.selected.join(', ')}<br />Σ selected p = {fmtNum(r0.selectedProbs.reduce((a, b) => a + b, 0))} → renormalized Σ α = {fmtNum(r0.alpha.reduce((a, b) => a + b, 0))}</p>
            <p>Across a batch of {nTok} synthetic tokens, <strong>{union.size}</strong> distinct experts are used in this block, more than {k}: each token chooses separately (and each block routes separately).</p>
          </div>
          <div className="card tight">
            <h3>Expert {focusE} (block 0) — structure <Evidence kind="verified" /></h3>
            <dl className="kv" style={{ fontSize: '0.85rem' }}>
              <dt>gate</dt><dd className="mono">{mapping.checkpoint.gate.replace('{layer}', '0')}</dd>
              <dt>up</dt><dd className="mono">{mapping.checkpoint.up.replace('{layer}', '0')}</dd>
              <dt>down</dt><dd className="mono">{mapping.checkpoint.down.replace('{layer}', '0')}</dd>
              <dt>params</dt><dd>{fmtInt(a30.oneExpert!)} (3 × 2,048 × 768)</dd>
              <dt>runtime</dt><dd className="mono">{mapping.runtime.gate}</dd>
              <dt>this token</dt><dd>{r0.selected.includes(focusE) ? `selected, α = ${fmtNum(r0.alpha[r0.selected.indexOf(focusE)])}` : `not selected (p = ${fmtNum(r0.probs[focusE])}); still stored`}</dd>
            </dl>
            <a href={explorerHref({ m: 'qwen3-30b-a3b', lv: 2, b: 0, t: 'expert_gate_proj', e: focusE })}>Open its gate matrix in the explorer →</a>
          </div>
        </div>
      </Section>
      <Section title="Storage versus per-token work" badges={<Evidence kind="derived" />}>
        <div className="table-wrap"><table className="t"><tbody>
          <tr><td>Expert parameters stored per block</td><td className="num">{fmtInt(a30.ffnPerBlock)}</td></tr>
          <tr><td>Expert parameters used per token per block (8 × {fmtInt(a30.oneExpert!)})</td><td className="num">{fmtInt(8 * a30.oneExpert!)} ({((8 / 128) * 100).toFixed(2)}%)</td></tr>
          <tr><td>Selected experts summed over all 48 blocks</td><td className="num">{fmtInt(a30.selectedExpertsAllLayers!)}</td></tr>
          <tr><td>Convention “all non-expert parameters + selected experts” (includes the whole embedding table)</td><td className="num">{fmtInt(a30.activeIncludingFullEmbedding!)}</td></tr>
          <tr><td>Same, counting one embedding row per token instead of the table</td><td className="num">{fmtInt(a30.activeEmbeddingRowOnly!)}</td></tr>
          <tr><td>Model card “activated”</td><td className="num">3.3B</td></tr>
        </tbody></table></div>
        <p className="muted" style={{ fontSize: '0.9rem' }}>“A3B” names an approximate activated-parameter convention, not the stored total ({fmtB(a30.total)}). None of these numbers is a universal count of scalars read, FLOPs, memory traffic or latency.<Cite ids={['q30-card', 'q30-config']} /></p>
      </Section>
      <Section title="Advanced: serving and training MoE" badges={<Evidence kind="verified" />}>
        <ul style={{ paddingLeft: 18 }}>
          <li><strong>Expert parallelism.</strong> Experts of a layer can be spread across devices. Each token's activation vector must then be <em>dispatched</em> to the devices holding its selected experts and the results <em>gathered</em> back, an all-to-all communication step that a dense model does not need.<Cite ids={['switch']} /></li>
          <li><strong>Load balancing.</strong> Training adds an auxiliary loss that discourages the router from overloading a few experts (router_aux_loss_coef = 0.001 in the config). The Qwen3 report states that Qwen3-MoE uses a global-batch load-balancing loss.<Cite ids={['q30-config', 'qwen3-report']} /></li>
          <li><strong>Why latency is not dense latency × (active/total).</strong> All experts must be resident (memory), small per-expert matrices use hardware less efficiently than one big matrix, tokens are grouped by expert (gather/scatter), load can be uneven, and communication adds time. Attention, embeddings and the head are not sparse at all.</li>
          <li><strong>Gradients.</strong> The top-k choice is discrete: training receives gradients through the selected experts and their coefficients α (and thus the router probabilities), not through the decision of which experts were chosen.</li>
        </ul>
        <Tex block>{'y=\\sum_{e\\in S}\\alpha_e\\,\\operatorname{Expert}_e(x),\\quad S=\\operatorname{TopK}(\\operatorname{softmax}(xW_{router}^{\\top}),8),\\quad \\alpha_e=\\frac{p_e}{\\sum_{j\\in S}p_j}'}</Tex>
      </Section>
    </>
  );
}
