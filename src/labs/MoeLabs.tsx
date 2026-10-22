import { useMemo, useState } from 'react';
import { useToy } from '../state/toyStore';
import { Cite, Evidence, NumberField, ProbRow, Section, SimBanner, Slider, Tex, VectorView, fmtB } from '../components/ui';
import { ToyControls, PromptEditor, tok } from './common';
import { QWEN3_30B_A3B, QWEN3_8B, TOY_DENSE, TOY_MOE } from '../models/profiles';
import { accounting, packedExpertMapping } from '../models/shapes';
import { forwardFull, initToyWeights } from '../math/toy';
import { routeFromLogits } from '../math/ops';
import { makeRng, nextNormal } from '../math/rng';
import { swiglu } from '../math/ops';
import { explorerHref } from '../state/route';
import { useLang } from '../i18n/i18n';

export function CompareLab() {
  const { state } = useToy();
  const { lang, num, fnum } = useLang();
  const L = (en: string, sk: string) => (lang === 'sk' ? sk : en);
  const a8 = accounting(QWEN3_8B), a30 = accounting(QWEN3_30B_A3B);
  const ad = accounting(TOY_DENSE), am = accounting(TOY_MOE);
  const dense = useMemo(() => forwardFull(TOY_DENSE, initToyWeights(TOY_DENSE, state.seed), state.tokens), [state.seed, state.tokens]);
  const moe = useMemo(() => forwardFull(TOY_MOE, initToyWeights(TOY_MOE, state.seed), state.tokens), [state.seed, state.tokens]);
  const row = (label: string, d: string, m: string) => <tr><td>{label}</td><td className="num">{d}</td><td className="num">{m}</td></tr>;
  const router = L('router', 'smerovač');
  return (
    <>
      <Section title={L('6 · Dense vs MoE: the two reference models', '6 · Hustý model vs MoE: dva referenčné modely')} badges={<><Evidence kind="verified" /><Evidence kind="derived" /></>}>
        <div className="callout warn">{L('Not a controlled experiment: the two models differ in residual width, depth and K/V heads, not only in the FFN. Use the toy experiment below to isolate the FFN difference.', 'Nie je to kontrolovaný pokus: dva modely sa líšia šírkou reziduálneho prúdu, hĺbkou aj počtom hláv K/V, nielen doprednou sieťou. Na izolovanie rozdielu FFN použite hračkový pokus nižšie.')}</div>
        <div className="table-wrap" tabIndex={0}><table className="t">
          <thead><tr><th /><th className="num">{L('Qwen3-8B (dense)', 'Qwen3-8B (hustý)')}</th><th className="num">Qwen3-30B-A3B (MoE)</th></tr></thead>
          <tbody>
            {row(L('Decoder blocks', 'Bloky dekodéra'), '36', '48')}
            {row(L('Residual width d', 'Šírka reziduálneho prúdu d'), num(4096), num(2048))}
            {row(L('Query / K/V heads', 'Hlavy dopytov / K/V'), '32 / 8', '32 / 4')}
            {row(L('Query width Hq·dh', 'Šírka dopytov Hq·dh'), `${num(4096)} (= d)`, `${num(4096)} (≠ d)`)}
            {row(L('FFN per block', 'FFN v bloku'), L('one SwiGLU, f = 12,288', 'jedna SwiGLU, f = 12 288'), L('128 experts of width 768, top-8, no shared expert', '128 expertov so šírkou 768, top-8, bez zdieľaného experta'))}
            {row(L('FFN/expert params stored per block', 'Uložené parametre FFN/expertov v bloku'), num(a8.ffnPerBlock), num(a30.ffnPerBlock))}
            {row(L('FFN/expert params used per token per block', 'Použité parametre FFN/expertov na token v bloku'), num(a8.ffnPerBlock), `${num(8 * a30.oneExpert!)} (${L('8 experts', '8 expertov')}) + ${router} ${num(a30.routerPerBlock)}`)}
            {row(L('Total stored parameters', 'Uložené parametre spolu'), num(a8.total), num(a30.total))}
            {row(L('K/V cache per token (BF16)', 'K/V cache na token (BF16)'), '144 KiB', '96 KiB')}
            {row(L('Advertised (model card)', 'Uvádzané (karta modelu)'), L('8.2B', '8,2 mld.'), L('30.5B total, 3.3B activated', '30,5 mld. spolu, 3,3 mld. aktivovaných'))}
          </tbody>
        </table></div>
      </Section>
      <Section title={L('Controlled toy comparison (matched dimensions)', 'Kontrolované hračkové porovnanie (rovnaké rozmery)')} badges={<Evidence kind="simulation" />}>
        <SimBanner>{L(`Both toys share d = 8, 2 blocks, 2 Q / 1 K/V heads and the seed ${state.seed}. Only the FFN sub-layer differs: dense width 16 vs 4 experts of width 8 with top-2 (active width 16 = dense width).`, `Obe hračky majú d = 8, 2 bloky, 2 hlavy Q / 1 hlavu K/V a osivo ${state.seed}. Líši sa len podvrstva FFN: hustá šírka 16 vs 4 experti so šírkou 8 a top-2 (aktívna šírka 16 = hustá šírka).`)}</SimBanner>
        <PromptEditor />
        <div className="table-wrap" tabIndex={0} style={{ marginTop: 8 }}><table className="t">
          <thead><tr><th /><th className="num">{L('Toy dense', 'Hustá hračka')}</th><th className="num">{L('Toy MoE', 'Hračka MoE')}</th></tr></thead>
          <tbody>
            {row(L('FFN params stored per block', 'Uložené parametre FFN v bloku'), num(ad.ffnPerBlock), `${num(am.ffnPerBlock)} + ${router} ${am.routerPerBlock}`)}
            {row(L('FFN params used per token per block', 'Použité parametre FFN na token v bloku'), num(ad.ffnPerBlock), `${num(2 * am.oneExpert!)} + ${router} ${am.routerPerBlock}`)}
            {row(L('Total parameters', 'Parametre spolu'), num(ad.total), num(am.total))}
            {row(L('Attention params per block (identical)', 'Parametre pozornosti v bloku (rovnaké)'), num(ad.attentionPerBlock), num(am.attentionPerBlock))}
          </tbody>
        </table></div>
        <h3>{L('Which experts each token used (toy MoE, computed)', 'Ktorých expertov použil každý token (hračka MoE, vypočítané)')}</h3>
        <div className="table-wrap" tabIndex={0}><table className="t">
          <thead><tr><th>{L('pos', 'poz.')}</th><th>{L('token', 'token')}</th>{moe[0].layers.map((_, l) => <th key={l}>{L(`block ${l}: experts (α)`, `blok ${l}: experti (α)`)}</th>)}</tr></thead>
          <tbody>{moe.map((pt) => (
            <tr key={pt.pos}><td>{pt.pos}</td><td className="mono">{tok(pt.token)}</td>{pt.layers.map((Ly, l) => <td key={l} className="mono">{Ly.moe!.route.selected.map((e, i) => `${e} (${fnum(Ly.moe!.route.alpha[i], 2)})`).join('; ')}</td>)}</tr>
          ))}</tbody>
        </table></div>
        <p className="faint" style={{ fontSize: '0.86rem' }}>{L(`Last-position logits differ between the two toys (${fnum(dense[dense.length - 1].logits[0])} vs ${fnum(moe[moe.length - 1].logits[0])} for “${tok(0)}”) because they are different models with different random weights, not because one is “better”: no training happened here.`, `Logity poslednej pozície sa medzi dvoma hračkami líšia (${fnum(dense[dense.length - 1].logits[0])} vs ${fnum(moe[moe.length - 1].logits[0])} pre „${tok(0)}“), lebo sú to rôzne modely s rôznymi náhodnými váhami, nie preto, že by jeden bol „lepší“: nič sa tu netrénovalo.`)}</p>
      </Section>
    </>
  );
}

export function RouterLab() {
  const { profile, trace, state, dispatch, weights } = useToy();
  const { lang, num, fnum } = useLang();
  const T = (en: string, sk: string) => (lang === 'sk' ? sk : en);
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
  const ex = T('expert', 'expert');
  return (
    <>
      <Section title={T('7 · Router lab (toy simulator: 4 experts, top-2)', '7 · Laboratórium smerovača (hračkový simulátor: 4 experti, top-2)')} badges={<Evidence kind="simulation" />}>
        <ToyControls />
        {!isMoe ? (
          <div className="callout warn">{T('The shared simulator is in its dense variant.', 'Spoločný simulátor je v hustom variante.')} <button className="small" onClick={() => dispatch({ type: 'variant', v: 'toy-moe' })}>{T('Switch to the MoE toy', 'Prepnúť na hračku MoE')}</button></div>
        ) : (
          <>
            <PromptEditor />
            <div className="row">
              <NumberField label={T('block', 'blok')} value={layer} min={0} max={profile.layers - 1} step={1} onChange={(v) => setLayer(Math.round(v))} width={70} />
              <NumberField label={T('token position', 'pozícia tokenu')} value={Math.min(pos, state.tokens.length - 1)} min={0} max={state.tokens.length - 1} step={1} onChange={(v) => setPos(Math.round(v))} width={80} />
            </div>
            <div className="grid2" style={{ marginTop: 8 }}>
              <div>
                <VectorView label={T(`router logits r = x W_routerᵀ for “${tok(P.token)}” (raw scores, any sign)`, `logity smerovača r = x W_routerᵀ pre „${tok(P.token)}“ (surové skóre, ľubovoľné znamienko)`)} values={L!.moe!.route.logits} />
                <h3>{T('Probabilities over all experts', 'Pravdepodobnosti pre všetkých expertov')}</h3>
                {L!.moe!.route.probs.map((p, e) => <ProbRow key={e} label={`${ex} ${e}`} p={p} off={!L!.moe!.route.selected.includes(e)} />)}
                <h3>{T('Selected and renormalized (α)', 'Vybraní a prepočítaní (α)')}</h3>
                {L!.moe!.route.selected.map((e, i) => <ProbRow key={e} label={`${ex} ${e}`} p={L!.moe!.route.alpha[i]} extra={`${fnum(L!.moe!.route.selectedProbs[i])} → ${fnum(L!.moe!.route.alpha[i])}`} />)}
                <p className="faint" style={{ fontSize: '0.84rem' }}>{T(`α sums to ${fnum(L!.moe!.route.alpha.reduce((a, b) => a + b, 0))}. Learned router parameters = the matrix W_router; per-token mixture coefficients = α. They are different objects.`, `α dáva spolu ${fnum(L!.moe!.route.alpha.reduce((a, b) => a + b, 0))}. Naučené parametre smerovača = matica W_router; koeficienty zmesi pre token = α. Sú to rôzne objekty.`)}</p>
              </div>
              <div>
                <h3>{T('Expert outputs and their mixture', 'Výstupy expertov a ich zmes')}</h3>
                {L!.moe!.route.selected.map((e, i) => (
                  <VectorView key={e} label={`α = ${fnum(L!.moe!.route.alpha[i], 3)} × Expert ${e}(x)`} values={L!.moe!.expertOut[i]} />
                ))}
                <VectorView label={T('y = Σ α_e · Expert_e(x) (MoE output, before the residual add)', 'y = Σ α_e · Expert_e(x) (výstup MoE pred reziduálnym sčítaním)')} values={L!.moe!.y} />
                <VectorView label={T("an unselected expert's output, for comparison (not used)", 'výstup nevybraného experta na porovnanie (nepoužíva sa)')} values={swiglu(L!.ln2, ...(['gate', 'up', 'down'] as const).map((w) => weights.layers[layer].experts![[0, 1, 2, 3].find((x) => !L!.moe!.route.selected.includes(x))!][w]) as [number[][], number[][], number[][]]).y} />
              </div>
            </div>
          </>
        )}
      </Section>
      <Section title={T(`Official-size fixture: ${m.numExperts} experts, top-${m.topK}`, `Prípad v skutočnej veľkosti: ${m.numExperts} expertov, top-${m.topK}`)} badges={<><Evidence kind="verified" title="Counts, shapes and rule from config + implementation" /><Evidence kind="simulation" title="Router logits are synthetic" /></>}>
        <SimBanner>{T(`Router logits below are synthetic (seed ${fixSeed}); the counts (${m.numExperts}, top-${m.topK}), renormalization rule and expert shapes are Qwen3-30B-A3B's. No real routing decision is shown.`, `Logity smerovača nižšie sú syntetické (osivo ${fixSeed}); počty (${m.numExperts}, top-${m.topK}), pravidlo prepočtu a tvary expertov sú z Qwen3-30B-A3B. Nezobrazuje sa žiadne skutočné rozhodnutie smerovania.`)}</SimBanner>
        <div className="row" style={{ marginTop: 8 }}>
          <NumberField label={T('fixture seed', 'osivo prípadu')} value={fixSeed} min={0} step={1} onChange={(v) => setFixSeed(Math.round(v))} width={90} />
          <Slider label={T('top-k (educational)', 'top-k (na učenie)')} value={k} min={1} max={16} onChange={setK} />
          <Slider label={T('tokens in batch', 'tokeny v dávke')} value={nTok} min={1} max={64} onChange={setNTok} />
        </div>
        {k !== m.topK && <div className="banner-edit" style={{ margin: '6px 0' }}>{T(`Custom educational routing rule (top-${k}). The pretrained model was configured and evaluated with top-${m.topK}; this does not claim anything about running it with top-${k}.`, `Vlastné pravidlo smerovania na učenie (top-${k}). Predtrénovaný model bol nastavený a hodnotený s top-${m.topK}; o jeho behu s top-${k} sa tu nič netvrdí.`)}</div>}
        <div className="grid2">
          <div>
            <p className="faint" style={{ fontSize: '0.85rem' }}>{T(`Token 0: all ${m.numExperts} probabilities are computed; ${k} are selected (orange). Click any expert to inspect its structure.`, `Token 0: vypočíta sa všetkých ${m.numExperts} pravdepodobností; vybraných je ${k} (oranžové). Kliknutím na experta zobrazíte jeho štruktúru.`)}</p>
            <div className="expert-grid" role="group" aria-label={T('Expert overview', 'Prehľad expertov')}>
              {Array.from({ length: m.numExperts }, (_, e) => (
                <button key={e} className={`${r0.selected.includes(e) ? 'sel' : ''}${focusE === e ? ' focus' : ''}`} aria-pressed={focusE === e} aria-label={T(`expert ${e}${r0.selected.includes(e) ? ', selected' : ''}, probability ${r0.probs[e].toFixed(4)}`, `expert ${e}${r0.selected.includes(e) ? ', vybraný' : ''}, pravdepodobnosť ${fnum(r0.probs[e])}`)} onClick={() => setFocusE(e)}>{e}</button>
              ))}
            </div>
            <p className="mono" style={{ fontSize: '0.86rem' }}>{T('selected', 'vybraní')}: {r0.selected.join(', ')}<br />{T('Σ selected p', 'Σ vybraných p')} = {fnum(r0.selectedProbs.reduce((a, b) => a + b, 0))} → {T('renormalized Σ α', 'prepočítané Σ α')} = {fnum(r0.alpha.reduce((a, b) => a + b, 0))}</p>
            <p>{lang === 'sk' ? <>V dávke {nTok} syntetických tokenov sa v tomto bloku použije <strong>{union.size}</strong> rôznych expertov, viac než {k}: každý token vyberá samostatne (a každý blok smeruje samostatne).</> : <>Across a batch of {nTok} synthetic tokens, <strong>{union.size}</strong> distinct experts are used in this block, more than {k}: each token chooses separately (and each block routes separately).</>}</p>
          </div>
          <div className="card tight">
            <h3>{T(`Expert ${focusE} (block 0) — structure`, `Expert ${focusE} (blok 0) — štruktúra`)} <Evidence kind="verified" /></h3>
            <dl className="kv" style={{ fontSize: '0.85rem' }}>
              <dt>gate</dt><dd className="mono">{mapping.checkpoint.gate.replace('{layer}', '0')}</dd>
              <dt>up</dt><dd className="mono">{mapping.checkpoint.up.replace('{layer}', '0')}</dd>
              <dt>down</dt><dd className="mono">{mapping.checkpoint.down.replace('{layer}', '0')}</dd>
              <dt>{T('params', 'parametre')}</dt><dd>{num(a30.oneExpert!)} (3 × {num(2048)} × 768)</dd>
              <dt>{T('runtime', 'počas behu')}</dt><dd className="mono">{mapping.runtime.gate}</dd>
              <dt>{T('this token', 'tento token')}</dt><dd>{r0.selected.includes(focusE) ? T(`selected, α = ${fnum(r0.alpha[r0.selected.indexOf(focusE)])}`, `vybraný, α = ${fnum(r0.alpha[r0.selected.indexOf(focusE)])}`) : T(`not selected (p = ${fnum(r0.probs[focusE])}); still stored`, `nevybraný (p = ${fnum(r0.probs[focusE])}); stále uložený`)}</dd>
            </dl>
            <a href={explorerHref({ m: 'qwen3-30b-a3b', lv: 2, b: 0, t: 'expert_gate_proj', e: focusE })}>{T('Open its gate matrix in the explorer →', 'Otvoriť jeho maticu gate v prieskumníku →')}</a>
          </div>
        </div>
      </Section>
      <Section title={T('Storage versus per-token work', 'Uloženie verzus práca na token')} badges={<Evidence kind="derived" />}>
        <div className="table-wrap" tabIndex={0}><table className="t"><tbody>
          <tr><td>{T('Expert parameters stored per block', 'Uložené parametre expertov v bloku')}</td><td className="num">{num(a30.ffnPerBlock)}</td></tr>
          <tr><td>{T(`Expert parameters used per token per block (8 × ${num(a30.oneExpert!)})`, `Použité parametre expertov na token v bloku (8 × ${num(a30.oneExpert!)})`)}</td><td className="num">{num(8 * a30.oneExpert!)} ({num((8 / 128) * 100, 2)} %)</td></tr>
          <tr><td>{T('Selected experts summed over all 48 blocks', 'Vybraní experti spolu cez všetkých 48 blokov')}</td><td className="num">{num(a30.selectedExpertsAllLayers!)}</td></tr>
          <tr><td>{T('Convention “all non-expert parameters + selected experts” (includes the whole embedding table)', 'Spôsob „všetky parametre mimo expertov + vybraní experti“ (zahŕňa celú tabuľku embeddingov)')}</td><td className="num">{num(a30.activeIncludingFullEmbedding!)}</td></tr>
          <tr><td>{T('Same, counting one embedding row per token instead of the table', 'To isté, ale s jedným riadkom embeddingu na token namiesto celej tabuľky')}</td><td className="num">{num(a30.activeEmbeddingRowOnly!)}</td></tr>
          <tr><td>{T('Model card “activated”', 'Karta modelu „aktivované“')}</td><td className="num">{T('3.3B', '3,3 mld.')}</td></tr>
        </tbody></table></div>
        <p className="muted" style={{ fontSize: '0.9rem' }}>{T(`“A3B” names an approximate activated-parameter convention, not the stored total (${fmtB(a30.total)}). None of these numbers is a universal count of scalars read, FLOPs, memory traffic or latency.`, `„A3B“ označuje približný spôsob počítania aktivovaných parametrov, nie uložený celkový počet (${fmtB(a30.total).replace('.', ',')}). Žiadne z týchto čísel nie je univerzálny počet prečítaných čísel, operácií, prenosov pamäte ani oneskorenia.`)}<Cite ids={['q30-card', 'q30-config']} /></p>
      </Section>
      <Section title={T('Advanced: serving and training MoE', 'Pokročilé: prevádzka a trénovanie MoE')} badges={<Evidence kind="verified" />}>
        {lang === 'sk' ? (
          <ul style={{ paddingLeft: 18 }}>
            <li><strong>Paralelizmus expertov.</strong> Expertov jednej vrstvy možno rozdeliť na viac zariadení. Vektor aktivácií každého tokenu sa potom musí <em>odoslať</em> na zariadenia s jeho vybranými expertmi a výsledky sa musia <em>zozbierať</em> späť – komunikácia „všetci so všetkými“, ktorú hustý model nepotrebuje.<Cite ids={['switch']} /></li>
            <li><strong>Vyvažovanie záťaže.</strong> Trénovanie pridáva pomocnú chybu, ktorá bráni smerovaču preťažiť niekoľko expertov (v konfigurácii router_aux_loss_coef = 0,001). Správa o Qwen3 uvádza, že Qwen3-MoE používa vyvažovanie záťaže cez celú dávku.<Cite ids={['q30-config', 'qwen3-report']} /></li>
            <li><strong>Prečo oneskorenie nie je oneskorenie hustého modelu × (aktívne/celkové).</strong> Všetci experti musia byť v pamäti, malé matice expertov využívajú hardvér horšie než jedna veľká, tokeny sa zoskupujú podľa expertov (gather/scatter), záťaž môže byť nerovnomerná a komunikácia pridáva čas. Pozornosť, embeddingy a výstupná hlava nie sú vôbec riedke.</li>
            <li><strong>Gradienty.</strong> Výber top-k je diskrétny: trénovanie dostáva gradienty cez vybraných expertov a ich koeficienty α (a tým cez pravdepodobnosti smerovača), nie cez samotné rozhodnutie, ktorí experti boli vybraní.</li>
          </ul>
        ) : (
          <ul style={{ paddingLeft: 18 }}>
            <li><strong>Expert parallelism.</strong> Experts of a layer can be spread across devices. Each token's activation vector must then be <em>dispatched</em> to the devices holding its selected experts and the results <em>gathered</em> back, an all-to-all communication step that a dense model does not need.<Cite ids={['switch']} /></li>
            <li><strong>Load balancing.</strong> Training adds an auxiliary loss that discourages the router from overloading a few experts (router_aux_loss_coef = 0.001 in the config). The Qwen3 report states that Qwen3-MoE uses a global-batch load-balancing loss.<Cite ids={['q30-config', 'qwen3-report']} /></li>
            <li><strong>Why latency is not dense latency × (active/total).</strong> All experts must be resident (memory), small per-expert matrices use hardware less efficiently than one big matrix, tokens are grouped by expert (gather/scatter), load can be uneven, and communication adds time. Attention, embeddings and the head are not sparse at all.</li>
            <li><strong>Gradients.</strong> The top-k choice is discrete: training receives gradients through the selected experts and their coefficients α (and thus the router probabilities), not through the decision of which experts were chosen.</li>
          </ul>
        )}
        <Tex block>{'y=\\sum_{e\\in S}\\alpha_e\\,\\operatorname{Expert}_e(x),\\quad S=\\operatorname{TopK}(\\operatorname{softmax}(xW_{router}^{\\top}),8),\\quad \\alpha_e=\\frac{p_e}{\\sum_{j\\in S}p_j}'}</Tex>
      </Section>
    </>
  );
}
