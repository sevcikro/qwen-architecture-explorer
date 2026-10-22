import { useMemo, useState } from 'react';
import { HYBRIDS, QWEN38_COMPONENTS, COVERAGE_LABEL, type HybridProfile } from '../models/hybrid';
import { Cite, Evidence, Section, SimBanner, Slider, Tex, VectorView } from '../components/ui';
import { gatedDeltaRule, gdnGate, gatedResidualToy, qsaSelect, toyGdnInputs } from '../math/hybridToy';
import { sigmoid } from '../math/ops';
import { deriveSeed, makeRng, nextNormal } from '../math/rng';
import { useLang } from '../i18n/i18n';
import { COMPONENT_SK, COVERAGE_LABEL_SK, HYBRID_SK } from '../i18n/dataSk';

function useL() {
  const api = useLang();
  return { ...api, L: (en: string, sk: string) => (api.lang === 'sk' ? sk : en) };
}

function LayerStrip({ h }: { h: HybridProfile }) {
  const { L } = useL();
  const w = 16;
  return (
    <svg viewBox={`0 0 ${h.layers * w + 20} 70`} style={{ width: '100%', height: 'auto', maxWidth: 900 }} role="img" aria-label={L(`${h.layers} blocks: ${h.layerTypes.filter((t) => t === 'gdn').length} Gated DeltaNet and ${h.layerTypes.filter((t) => t === 'attn').length} attention blocks, each followed by MoE`, `${h.layers} blokov: ${h.layerTypes.filter((t) => t === 'gdn').length} Gated DeltaNet a ${h.layerTypes.filter((t) => t === 'attn').length} blokov pozornosti, za každým MoE`)}>
      {h.layerTypes.map((t, i) => (
        <g key={i}>
          <rect x={10 + i * w} y={8} width={w - 3} height={30} rx={3} fill={t === 'attn' ? 'color-mix(in srgb, var(--attn) 45%, transparent)' : 'color-mix(in srgb, var(--state) 35%, transparent)'} stroke={t === 'attn' ? 'var(--attn)' : 'var(--state)'} strokeDasharray={t === 'attn' ? undefined : '3 2'} />
          {i % 4 === 0 && <text x={10 + i * w} y={52} fontSize={8} fill="var(--text-3)">{i}</text>}
          {h.id === 'qwen3.8-flash-next' && i === 1 && <text x={10 + i * w + 6} y={66} fontSize={8} fill="var(--router)" textAnchor="middle">{L('n-gram', 'n-gramy')}</text>}
        </g>
      ))}
    </svg>
  );
}

function GdnToy() {
  const { L, fnum: fmtNum, lang } = useL();
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
      <SimBanner>{L('Toy analogue of the verified recurrence (one head, d_k = d_v = 4, synthetic inputs). Qwen3.8 runs this per head with 48 value heads of 128 × 128 state in each of 36 blocks.', 'Hračková obdoba overenej rekurencie (jedna hlava, d_k = d_v = 4, syntetické vstupy). Qwen3.8 to počíta pre každú hlavu: 48 hláv hodnôt so stavom 128 × 128 v každom z 36 blokov.')}</SimBanner>
      <div className="row" style={{ marginTop: 8 }}>
        <Slider label={L('tokens processed', 'spracované tokeny')} value={T} min={1} max={12} onChange={(v) => { setT(v); setStep(Math.min(step, v - 1)); }} />
        <Slider label={L('inspect step t', 'skúmaný krok t')} value={Math.min(step, T - 1)} min={0} max={T - 1} onChange={setStep} />
        <Slider label={L('A_log (decay strength)', 'A_log (sila útlmu)')} value={aLog} min={-3} max={2} step={0.1} onChange={setALog} format={(v) => (lang === 'sk' ? v.toFixed(1).replace('.', ',') : v.toFixed(1))} />
      </div>
      <div className="grid2">
        <div>
          <p className="mono" style={{ fontSize: '0.86rem' }}>
            g_t = −exp(A_log)·softplus(a_t + dt_bias) = {fmtNum(g[Math.min(step, T - 1)])}<br />
            {L('decay', 'útlm')} exp(g_t) = {fmtNum(s.decay)} {L('(keeps this fraction of the old state)', '(ponechá tento podiel starého stavu)')}<br />
            {L('write strength', 'sila zápisu')} β_t = σ(b_t) = {fmtNum(s.beta)}
          </p>
          <VectorView label={L('what the decayed state predicts for k_t (Sᵀk)', 'čo stav po útlme predpovedá pre k_t (Sᵀk)')} values={s.kvMem} />
          <VectorView label={L('delta = β (v_t − Sᵀk): only the prediction error is written', 'delta = β (v_t − Sᵀk): zapíše sa len chyba predpovede')} values={s.delta} />
          <VectorView label={L('output o_t = Sᵀ q_t', 'výstup o_t = Sᵀ q_t')} values={s.out} />
        </div>
        <div>
          <div className="faint" style={{ fontSize: '0.84rem' }}>{L(`State S after step ${s.t} (fixed size d_k × d_v, independent of how many tokens came before):`, `Stav S po kroku ${s.t} (pevná veľkosť d_k × d_v, nezávislá od počtu predchádzajúcich tokenov):`)}</div>
          <svg viewBox="0 0 170 170" style={{ maxWidth: 220 }} role="img" aria-label={L('Recurrent state matrix heatmap', 'Teplotná mapa matice rekurentného stavu')}>
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
      {lang === 'sk' ? <p className="faint" style={{ fontSize: '0.84rem' }}>Ako v <code>torch_recurrent_gated_delta_rule</code>: q a k sú normalizované L2, q je škálované 1/√d_k. Pred rekurenciou kanály q/k/v premieša krátka kauzálna konvolúcia po kanáloch (jadro 4); potom hradlovaná RMSNorm (hradlo z z <code>in_proj_z</code>, sigmoid) a <code>out_proj</code> vytvoria výstup bloku. Predvyplnenie používa ekvivalentný algoritmus po kúskoch.<Cite ids={['tf-qwen4exp', 'gdn']} /></p> : <p className="faint" style={{ fontSize: '0.84rem' }}>As implemented in <code>torch_recurrent_gated_delta_rule</code>: q and k are L2-normalized, q is scaled by 1/√d_k. Before the recurrence a short causal depthwise convolution (kernel 4) mixes the q/k/v channels; afterwards a gated RMSNorm (gate z from <code>in_proj_z</code>, sigmoid) and <code>out_proj</code> produce the block output. Prefill uses an equivalent chunked algorithm.<Cite ids={['tf-qwen4exp', 'gdn']} /></p>}
    </div>
  );
}

function QsaToy() {
  const { L, lang } = useL();
  const [visible, setVisible] = useState(41);
  const [budget, setBudget] = useState(16);
  const ratio = 4;
  const keys = useMemo(() => { const r = makeRng(deriveSeed(5, 'qsa')); return Array.from({ length: 64 }, () => Array.from({ length: 4 }, () => nextNormal(r))); }, []);
  const qh = useMemo(() => { const r = makeRng(deriveSeed(6, 'qsa-q')); return Array.from({ length: 4 }, () => Array.from({ length: 4 }, () => nextNormal(r))); }, []);
  const sel = qsaSelect(qh, keys, visible, budget, ratio);
  const chosen = new Set(sel.selectedTokens);
  return (
    <div>
      <SimBanner>{L('Toy analogue of the verified selection rule (4 indexer query heads, synthetic keys, toy budget). Real QSA: 4-token micro-blocks, budget 2,048 tokens = 512 blocks, 4 indexer query heads + 1 shared key head of width 128.', 'Hračková obdoba overeného pravidla výberu (4 hlavy dopytov indexera, syntetické kľúče, hračkový rozpočet). Skutočné QSA: 4-tokenové mikrobloky, rozpočet 2 048 tokenov = 512 blokov, 4 hlavy dopytov indexera + 1 spoločná hlava kľúčov so šírkou 128.')}</SimBanner>
      <div className="row" style={{ marginTop: 8 }}>
        <Slider label={L('visible (earlier) tokens', 'viditeľné (skoršie) tokeny')} value={visible} min={1} max={64} onChange={setVisible} />
        <Slider label={L('token budget', 'rozpočet tokenov')} value={budget} min={4} max={64} step={4} onChange={setBudget} />
      </div>
      <svg viewBox={`0 0 ${64 * 13 + 10} 64`} style={{ width: '100%', height: 'auto' }} role="img" aria-label={L(`${sel.selectedTokens.length} of ${visible} visible tokens attended`, `pozornosť na ${sel.selectedTokens.length} z ${visible} viditeľných tokenov`)}>
        {Array.from({ length: 64 }, (_, t) => {
          const vis = t < visible;
          const isTail = sel.tail.includes(t);
          const c = !vis ? 'var(--bg-2)' : chosen.has(t) ? (isTail ? 'var(--router)' : 'var(--attn)') : 'var(--panel-2)';
          return <rect key={t} x={5 + t * 13} y={8 + (t % 4 === 0 ? 0 : 0)} width={11} height={30} rx={2} fill={c} stroke={t % 4 === 0 ? 'var(--text-3)' : 'var(--line-soft)'} strokeDasharray={vis ? undefined : '2 2'} />;
        })}
        <text x={5} y={56} fontSize={10} fill="var(--text-3)">{L('blue = selected blocks · orange = incomplete tail block (always kept) · grey = skipped · dashed = not yet visible (future)', 'modré = vybrané bloky · oranžové = neúplný koncový blok (vždy ponechaný) · sivé = preskočené · čiarkované = ešte neviditeľné (budúcnosť)')}</text>
      </svg>
      <p className="mono" style={{ fontSize: '0.86rem' }}>{L(`${sel.blocks.length} complete blocks scored, keeping the top ${Math.min(sel.blockBudget, sel.blocks.length)}; attended tokens: ${sel.selectedTokens.length} of ${visible}.`, `ohodnotených ${sel.blocks.length} úplných blokov, ponechaných najlepších ${Math.min(sel.blockBudget, sel.blocks.length)}; tokeny v pozornosti: ${sel.selectedTokens.length} z ${visible}.`)}</p>
      {lang === 'sk' ? <p>Bežná kauzálna pozornosť by čítala všetkých {visible} skorších tokenov. QSA najprv lacným indexerom ohodnotí združené 4-tokenové bloky, skóre = Σ_hlavy ReLU(q_h · k̄_blok)/√128, ponechá najlepších ⌊2048/4⌋ = 512 blokov plus koniec a potom spustí bežnú hradlovanú softmax pozornosť <em>len</em> nad týmito tokenmi (ďalšia maska nad kauzálnou maskou). Pod 2 048 viditeľnými tokenmi sa nič nepreskakuje.<Cite ids={['tf-qwen4exp', 'q38-report']} /></p> : <p>Ordinary causal attention would read all {visible} earlier tokens. QSA first scores pooled 4-token blocks with a cheap indexer, score = Σ_heads ReLU(q_h · k̄_block)/√128, keeps the best ⌊2048/4⌋ = 512 blocks plus the tail, and then runs ordinary gated softmax attention <em>only</em> over those tokens (an extra mask on top of the causal mask). Below 2,048 visible tokens nothing is skipped.<Cite ids={['tf-qwen4exp', 'q38-report']} /></p>}
    </div>
  );
}

function GatedResidualToy() {
  const { L, lang } = useL();
  const d = 3, nb = 4;
  const rnd = useMemo(() => { const r = makeRng(deriveSeed(9, 'gr')); const m = (a: number, b: number, s = 0.6) => Array.from({ length: a }, () => Array.from({ length: b }, () => nextNormal(r) * s)); return { br: m(nb, d, 1), down: m(2, nb * d), up: m(nb * d, 2), inject: m(nb, nb * d) }; }, []);
  const res = gatedResidualToy(rnd.br, rnd.down, rnd.up, rnd.inject, (x) => x.map((v) => Math.tanh(2 * v)));
  return (
    <div>
      <SimBanner>{L('Toy analogue (4 branches of width 3, synthetic weights, a stand-in block function). Real: 4 branches of width 2,560 carried as one [B, T, 10,240] tensor; bottleneck rank 320.', 'Hračková obdoba (4 vetvy so šírkou 3, syntetické váhy, zástupná funkcia bloku). Skutočnosť: 4 vetvy so šírkou 2 560 nesené ako jeden tenzor [B, T, 10 240]; úzke hrdlo hodnosti 320.')}</SimBanner>
      <div className="grid2" style={{ marginTop: 8 }}>
        <div>
          {res.normed.map((b, i) => <VectorView key={i} label={L(`branch ${i} (RMS-normalized)`, `vetva ${i} (RMS-normalizovaná)`)} values={b} />)}
          <VectorView label={L('read gates (sigmoid, elementwise per branch)', 'čítacie hradlá (sigmoid, po prvkoch pre každú vetvu)')} values={res.readGate} />
          <VectorView label={L('block input = mean over branches of gate ⊙ branch', 'vstup bloku = priemer cez vetvy z hradlo ⊙ vetva')} values={res.xin} />
        </div>
        <div>
          <VectorView label={L('block output y (stand-in function)', 'výstup bloku y (zástupná funkcia)')} values={res.y} />
          <VectorView label={L('write gates 2·σ(w_b · n / 4), one scalar per branch', 'zapisovacie hradlá 2·σ(w_b · n / 4), jedno číslo na vetvu')} values={res.writeGate} />
          {res.after.map((b, i) => <VectorView key={i} label={L(`branch ${i} after: old + write_${i} · y`, `vetva ${i} potom: staré + zápis_${i} · y`)} values={b} />)}
        </div>
      </div>
      {lang === 'sk' ? <p className="faint" style={{ fontSize: '0.84rem' }}>Každý blok to robí dvakrát (okolo svojho miešača tokenov a okolo svojho MoE). Embedding sa na začiatku skopíruje do všetkých štyroch vetiev a záverečný miešač, ktorý len číta, ich pred výstupnou hlavou spojí. Neexistuje samostatná input_layernorm: túto úlohu plní zoskupená <code>hc_norm</code> v každom hradlovanom rezíduu.<Cite ids={['tf-qwen4exp']} /></p> : <p className="faint" style={{ fontSize: '0.84rem' }}>Every block does this twice (around its token mixer and around its MoE). The embedding is copied into all four branches at the start, and a final read-only mixer combines them before the output head. There is no separate input_layernorm: the grouped <code>hc_norm</code> inside each gated residual plays that role.<Cite ids={['tf-qwen4exp']} /></p>}
    </div>
  );
}

export default function Hybrid({ id }: { id?: string }) {
  const h = HYBRIDS.find((x) => x.id === id) ?? HYBRIDS[2];
  const is38 = h.id === 'qwen3.8-flash-next';
  const total = QWEN38_COMPONENTS.reduce((s, c) => s + c.elements, 0);
  const group = (g: string) => QWEN38_COMPONENTS.filter((c) => c.group === g).reduce((s, c) => s + c.elements, 0);
  const { lang, L, num: fmtInt, big: fmtB } = useL();
  const sk = lang === 'sk';
  const hs = sk ? HYBRID_SK[h.id] : undefined;
  return (
    <div className="page">
      <nav className="row" style={{ marginBottom: 6 }}><a href="#/family">← {L('Family map', 'Mapa rodiny')}</a>{HYBRIDS.map((x) => <a key={x.id} href={`#/hybrid/${x.id}`} aria-current={x.id === h.id ? 'page' : undefined}>{x.name}</a>)}</nav>
      <h1>{h.name}</h1>
      <div className="badge-row" style={{ marginBottom: 8 }}>
        <span className="coverage">{sk ? COVERAGE_LABEL_SK[h.coverage] : COVERAGE_LABEL[h.coverage]}</span>
        {h.extraToys.length > 0 && <span className="coverage">{L('+ toy analogues', '+ hračkové obdoby')}: {(hs?.extraToys ?? h.extraToys).join(', ')}</span>}
        <Evidence kind="verified" />
      </div>
      <p className="muted">{h.repo} @ <span className="mono">{h.revision.slice(0, 12)}</span> · model_type <code>{h.modelType}</code> · {hs?.advertised ?? h.advertised}. {L('A separate profile: nothing here applies to Qwen3-8B or Qwen3-30B-A3B.', 'Samostatný profil: nič z toho neplatí pre Qwen3-8B ani Qwen3-30B-A3B.')}<Cite ids={h.sources} /></p>
      <Section title={L('Layer layout', 'Rozloženie vrstiev')} badges={<Evidence kind="verified" />}>
        <p>{h.layout}</p>
        <LayerStrip h={h} />
        <dl className="kv">
          <dt>{L('Width', 'Šírka')}</dt><dd>{fmtInt(h.d)} · {hs?.residual ?? h.residual}</dd>
          <dt>Gated DeltaNet</dt><dd>{L(`${h.gdn.vHeads} value heads, ${h.gdn.qkHeads} query/key heads, head dim ${h.gdn.headDim}, conv kernel ${h.gdn.convKernel}`, `${h.gdn.vHeads} hláv hodnôt, ${h.gdn.qkHeads} hláv dopytov/kľúčov, šírka hlavy ${h.gdn.headDim}, jadro konvolúcie ${h.gdn.convKernel}`)}</dd>
          <dt>{L('Attention slots', 'Miesta pozornosti')}</dt><dd>{hs?.attnLabel ?? h.attnLabel}. {L(`${h.attn.qHeads} Q / ${h.attn.kvHeads} KV heads, head dim ${h.attn.headDim}, RoPE on ${h.attn.rotaryDim} of ${h.attn.headDim} dims, sigmoid output gate.`, `${h.attn.qHeads} hláv Q / ${h.attn.kvHeads} hlavy KV, šírka hlavy ${h.attn.headDim}, RoPE na ${h.attn.rotaryDim} z ${h.attn.headDim} rozmerov, sigmoidové výstupné hradlo.`)}</dd>
          <dt>MoE</dt><dd>{L(`${h.moe.experts} experts, ${h.moe.routed} routed + ${h.moe.shared} shared, expert width ${h.moe.expertWidth}`, `${h.moe.experts} expertov, ${h.moe.routed} smerovaných + ${h.moe.shared} zdieľaný, šírka experta ${h.moe.expertWidth}`)}</dd>
          <dt>{L('Vocabulary rows', 'Riadky slovníka')}</dt><dd>{fmtInt(h.vocabRows)}</dd>
          <dt>{L('Context', 'Kontext')}</dt><dd>{hs?.context ?? h.context}</dd>
          <dt>{L('Also', 'Ďalej')}</dt><dd>{(hs?.extras ?? h.extras).join(' · ')}</dd>
        </dl>
      </Section>
      {is38 && (
        <>
          <Section title={L('Component accounting (from the checkpoint headers)', 'Počítanie častí (z hlavičiek checkpointu)')} badges={<><Evidence kind="derived" /><Evidence kind="unknown" title={L("How the model card counts '4B MTP'", 'Ako karta modelu počíta „4B MTP“')} /></>}>
            <div className="table-wrap" tabIndex={0}><table className="t">
              <thead><tr><th>{L('component', 'časť')}</th><th className="num">{L('stored elements', 'uložené prvky')}</th><th>{L('group', 'skupina')}</th></tr></thead>
              <tbody>{QWEN38_COMPONENTS.map((c) => <tr key={c.key}><td>{sk ? COMPONENT_SK[c.key] ?? c.label : c.label}</td><td className="num">{fmtInt(c.elements)}</td><td>{c.group}</td></tr>)}
                <tr><td><strong>{L('Text backbone', 'Textová kostra')}</strong> {L('(everything except n-gram, MTP, vision)', '(všetko okrem n-gramov, MTP a obrazu)')}</td><td className="num"><strong>{fmtInt(group('backbone'))}</strong> ≈ {fmtB(group('backbone'))}</td><td>{L('card', 'karta')}: “125B”</td></tr>
                <tr><td><strong>{L('N-gram tables + injection', 'N-gramové tabuľky + vkladanie')}</strong></td><td className="num"><strong>{fmtInt(group('ngram'))}</strong></td><td>{L('card', 'karta')}: “51B n-gram embedding”</td></tr>
                <tr><td><strong>{L('MTP module', 'Modul MTP')}</strong> {L('(mtp.* tensors)', '(tenzory mtp.*)')}</td><td className="num"><strong>{fmtInt(group('mtp'))}</strong></td><td>{L('card', 'karta')}: “4B MTP”</td></tr>
                <tr><td>{L('Vision encoder', 'Kódovač obrazu')}</td><td className="num">{fmtInt(group('vision'))}</td><td>{L("not in the card's text count", 'nie je v textovom počte karty')}</td></tr>
                <tr><td>{L('All stored elements', 'Všetky uložené prvky')}</td><td className="num">{fmtInt(total)}</td><td>{L('= header total', '= súčet z hlavičiek')}</td></tr>
              </tbody>
            </table></div>
            {sk ? <p>„125B“ zodpovedá textovej kostre (125,71 mld.) a <strong>nezahŕňa</strong> 51 mld. n-gramových tabuliek, modul MTP ani kódovač obrazu, takže 125 mld. nie je súčet všetkých častí. <Evidence kind="unknown" /> Tenzory mtp.* v checkpointe majú 2,61 mld. prvkov; „4B MTP“ na karte zodpovedá aj započítaniu spoločného embeddingu a výstupnej hlavy (2,61 + 0,64 + 0,64 = 3,88 mld.), ale overené zdroje tento spôsob neuvádzajú. „6B activated“ je citované z karty a tu sa znovu neodvodzuje.<Cite ids={['q38-card', 'q38-index']} /></p> : <p>“125B” matches the text backbone (125.71B) and does <strong>not</strong> include the 51B n-gram tables, the MTP module or the vision encoder, so 125B is not the total of all components. <Evidence kind="unknown" /> The checkpoint's mtp.* tensors hold 2.61B elements; the card's “4B MTP” is consistent with also counting the shared embedding and output head (2.61 + 0.64 + 0.64 = 3.88B), but the sources checked do not state the convention. “6B activated” is quoted from the card and not re-derived here.<Cite ids={['q38-card', 'q38-index']} /></p>}
          </Section>
          <Section title={L('1 · Gated DeltaNet: how the recurrent state updates', '1 · Gated DeltaNet: ako sa aktualizuje rekurentný stav')} badges={<><Evidence kind="verified" /><Evidence kind="simulation" /></>}><GdnToy /></Section>
          <Section title={L('2 · Qwen Sparse Attention: selecting micro-blocks', '2 · Qwen Sparse Attention: výber mikroblokov')} badges={<><Evidence kind="verified" /><Evidence kind="simulation" /></>}><QsaToy /></Section>
          <Section title={L('3 · Routed and shared experts', '3 · Smerovaní a zdieľaní experti')} badges={<Evidence kind="verified" />}>
            <Tex block>{'y = \\sum_{e\\in\\operatorname{TopK}(p,10)} \\alpha_e\\,\\operatorname{Expert}_e(x) \\;+\\; \\sigma\\!\\big(x\\,w_{shared}\\big)\\,\\operatorname{SharedExpert}(x)'}</Tex>
            {sk ? <p>Každý token vždy prejde jediným zdieľaným expertom (šírka 640), škálovaným naučeným sigmoidovým hradlom (<code>shared_expert_gate</code>, matica [1, 2560]), a 10 z 512 smerovaných expertov vybraných rovnakým pravidlom softmax-top-k ako v Qwen3. Smerovaní experti sú uložení zbalení (<code>gate_up_proj [512, 1280, 2560]</code>, <code>down_proj [512, 2560, 640]</code>).<Cite ids={['tf-qwen4exp', 'q38-index']} /> <code>norm_topk_prob</code> v config.json chýba, takže platí predvolená hodnota triedy konfigurácie (<code>True</code>): 10 smerovaných pravdepodobností sa prepočíta tak, aby dávali spolu 1.<Cite ids={['tf-qwen4exp']} /></p> : <p>Every token always passes through the one shared expert (width 640), scaled by a learned sigmoid gate (<code>shared_expert_gate</code>, a [1, 2560] matrix), and through 10 of 512 routed experts chosen by the same softmax-top-k rule as Qwen3. Routed experts are stored packed (<code>gate_up_proj [512, 1280, 2560]</code>, <code>down_proj [512, 2560, 640]</code>).<Cite ids={['tf-qwen4exp', 'q38-index']} /> <code>norm_topk_prob</code> is absent from config.json, so the configuration-class default (<code>True</code>) applies: the 10 routed probabilities are renormalized to sum to 1.<Cite ids={['tf-qwen4exp']} /></p>}
          </Section>
          <Section title={L('4 · Gated residual branches', '4 · Hradlované reziduálne vetvy')} badges={<><Evidence kind="verified" /><Evidence kind="simulation" /></>}><GatedResidualToy /></Section>
          <Section title={L('5 · N-gram tables are parameters, not state', '5 · N-gramové tabuľky sú parametre, nie stav')} badges={<Evidence kind="verified" />}>
            {sk ? <>
            <p>Vo vrstve dekodéra s indexom v kóde <strong>1</strong> (konfigurácia <code>ple_layer_ids = [2]</code> počíta od 1; implementácia testuje <code>layer_idx + 1 in ple_layer_ids</code>; tenzory sú v <code>model.language_model.layers.1.ple.*</code>) sa bigram a trigram ID tokenov každého tokenu (končiaci týmto tokenom) hašujú (XOR ID krát pevné nepárne násobitele, potom modulo 8 rôznych prvočísel ≈ 20 000 000 pre každý rád n-gramu) do 16 vyhľadaní v naučených tabuľkách (128 častí, 320 001 536 riadkov × 160 hodnôt ≈ 51,2 mld. parametrov). Spojené vektory sa premietnu a pripočítajú ku všetkým štyrom reziduálnym vetvám cez kontextové sigmoidové hradlo a krátku dilatovanú konvolúciu.<Cite ids={['tf-qwen4exp', 'q38-config', 'q38-index']} /></p>
            <ul style={{ paddingLeft: 18 }}>
              <li><strong>Naučené parametre</strong> (po trénovaní pevné): n-gramové tabuľky, projekcie kľúčov/hodnôt, normalizácie, váhy konvolúcie.</li>
              <li><strong>Stav závislý od vstupu</strong> (pre každú sekvenciu počas behu): posledné 2 ID tokenov, história konvolúcie, rekurentné stavy Gated DeltaNet, K/V QSA a kľúče indexera.</li>
              <li>Karta modelu zdôvodňuje tabuľky tým, že sa dajú lacno odložiť do pamäte hostiteľa; vyhľadanie pre token stojí niekoľko riadkov, nie celú tabuľku.</li>
            </ul>
            </> : <>
            <p>At the decoder layer with code index <strong>1</strong> (config <code>ple_layer_ids = [2]</code> counts from 1; the implementation tests <code>layer_idx + 1 in ple_layer_ids</code>; tensors live at <code>model.language_model.layers.1.ple.*</code>), each token's bigram and trigram of token ids (ending at that token) are hashed (XOR of ids times fixed odd multipliers, then modulo 8 different primes ≈ 20,000,000 per n-gram order) into 16 lookups in learned tables (128 shards, 320,001,536 rows × 160 values ≈ 51.2B parameters). The concatenated vectors are projected and added to all four residual branches through a contextual sigmoid gate and a short dilated convolution.<Cite ids={['tf-qwen4exp', 'q38-config', 'q38-index']} /></p>
            <ul style={{ paddingLeft: 18 }}>
              <li><strong>Learned parameters</strong> (fixed after training): n-gram tables, key/value projections, norms, conv weights.</li>
              <li><strong>Input-dependent state</strong> (per sequence, at run time): the last 2 token ids, the convolution history, Gated DeltaNet recurrent states, QSA K/V and indexer keys.</li>
              <li>The model card motivates the tables as cheap to offload to host memory; per-token lookup cost is a few rows, not the full table.</li>
            </ul>
            </>}
          </Section>
          <Section title={L('6 · Which components belong to what', '6 · Ktoré časti k čomu patria')} badges={<Evidence kind="verified" />}>
            {sk ? (
            <ul style={{ paddingLeft: 18 }}>
              <li><strong>Inferencia textu</strong>: embedding, 48 blokov (s hradlovanými rezíduami a vkladaním n-gramov), záverečný reziduálny miešač, výstupná hlava.</li>
              <li><strong>Kódovanie obrazu</strong>: <code>model.visual.*</code> (27 blokov ViT, veľkosť záplaty 16, výstup premietnutý na 2 560) — používa sa len pre vstupy s obrázkom/videom.</li>
              <li><strong>Predpovedanie viacerých tokenov</strong>: <code>mtp.*</code> (jedna vrstva QSA + MoE navyše) — pomocný modul MTP („1 vrstva, trénovaná vo viacerých krokoch“ podľa karty modelu), na bežnú inferenciu ďalšieho tokenu nie je potrebný.</li>
            </ul>
            ) : (
            <ul style={{ paddingLeft: 18 }}>
              <li><strong>Text inference</strong>: embedding, 48 blocks (with gated residuals and the n-gram injection), final residual mixer, output head.</li>
              <li><strong>Vision encoding</strong>: <code>model.visual.*</code> (27 ViT blocks, patch size 16, output projected to 2,560) — used only for image/video inputs.</li>
              <li><strong>Multi-token prediction</strong>: <code>mtp.*</code> (one extra QSA + MoE layer) — an auxiliary multi-token-prediction module (“1 layer, trained with multi-steps” per the model card), not needed for ordinary next-token inference.</li>
            </ul>
            )}
          </Section>
          <Section title={L('Training recipe (as reported)', 'Postup trénovania (podľa správy)')} badges={<Evidence kind="verified" />}>
            {sk ? <p>Správa a karta modelu opisujú Muon pre 2-D lineárne váhy, AdamW pre embeddingy, výstupnú hlavu, smerovač a nízkohodnostné projekcie hradlovaného rezídua, Adam bez váhového útlmu pre n-gramové tabuľky a žiadne zahrievanie veľkosti dávky. Zloženie dát a úplné hyperparametre sa tu nereprodukujú.<Cite ids={['q38-report', 'q38-card']} /></p> : <p>The report and model card describe Muon for 2-D linear weights, AdamW for embeddings, output head, router and gated-residual low-rank projections, Adam without weight decay for the n-gram tables, and no batch-size warm-up. Data mixtures and full hyperparameters are not reproduced here.<Cite ids={['q38-report', 'q38-card']} /></p>}
          </Section>
        </>
      )}
      {!is38 && (
        <Section title={L('What is and is not covered', 'Čo je a čo nie je pokryté')} badges={<Evidence kind="unknown" />}>
          {sk ? <p>Tento profil je overený štrukturálny pohľad z pripnutej karty modelu a konfigurácie: rozloženie vrstiev, počty hláv, počty a šírky expertov. S Qwen3.8-Flash-Next zdieľa rekurenciu Gated DeltaNet a hradlovanú pozornosť (pozrite hračkové obdoby na tej stránke), ale má jediný bežný reziduálny prúd, žiadne QSA a žiadne n-gramové tabuľky. Presné rozpisy parametrov z hlavičiek checkpointu sa pre tento model nezískavali.</p> : <p>This profile is a verified structural view from the pinned model card and config: layer layout, head counts, expert counts and widths. It shares the Gated DeltaNet recurrence and gated attention with Qwen3.8-Flash-Next (see that page's toy analogues) but has a single ordinary residual stream, no QSA and no n-gram tables. Exact parameter breakdowns from checkpoint headers were not extracted for this model.</p>}
          <a className="btn" href="#/hybrid/qwen3.8-flash-next">{L('See the Qwen3.8-Flash-Next deep dive', 'Pozrite podrobný rozbor Qwen3.8-Flash-Next')}</a>
        </Section>
      )}
      <Section title={L('Unknowns', 'Neznáme')} badges={<Evidence kind="unknown" />}>
        <ul style={{ paddingLeft: 18 }}>
          <li>{L('No real activation traces, routing decisions or recurrent states of this model are shown anywhere on this site.', 'Nikde na tejto stránke sa nezobrazujú skutočné aktivácie, rozhodnutia smerovania ani rekurentné stavy tohto modelu.')}</li>
          <li>{L('Learned feature meanings of experts, heads or n-gram rows are not known and not claimed.', 'Naučené významy expertov, hláv či riadkov n-gramov nie sú známe a netvrdia sa.')}</li>
          {is38 && <li>{L('Exact deployment memory depends on runtime state dtypes and kernels (see the memory lab for a derived estimate).', 'Presná pamäť pri nasadení závisí od dátových typov stavu a jadier počas behu (odvodený odhad nájdete v laboratóriu pamäte).')}</li>}
        </ul>
      </Section>
    </div>
  );
}
