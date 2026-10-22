import { useState } from 'react';
import { useToy } from '../state/toyStore';
import { TOY_VOCAB, DEFAULT_SEED } from '../math/toy';
import { SimBanner } from '../components/ui';
import { useLang } from '../i18n/i18n';

/** Explicit, state-changing simulator controls (variant, seed, reset). Inspection controls live elsewhere. */
export function ToyControls({ allowVariant = true }: { allowVariant?: boolean }) {
  const { state, dispatch, edited, profile } = useToy();
  const { lang } = useLang();
  const T = (en: string, sk: string) => (lang === 'sk' ? sk : en);
  const [seedDraft, setSeedDraft] = useState(String(state.seed));
  const [copied, setCopied] = useState(false);
  const shape = T(
    `d=${profile.d}, ${profile.layers} blocks, ${profile.hq} query heads / ${profile.hkv} K/V head, head dim ${profile.headDim}${profile.moe ? `, ${profile.moe.numExperts} experts top-${profile.moe.topK} (width ${profile.moe.expertWidth})` : `, FFN width ${profile.ffnWidth}`}. Not Qwen's sizes (${profile.moe ? 'Qwen3-30B-A3B has 128 experts, top-8' : 'Qwen3-8B has d=4096'}).`,
    `d=${profile.d}, ${profile.layers} bloky, ${profile.hq} hlavy dopytov / ${profile.hkv} hlava K/V, šírka hlavy ${profile.headDim}${profile.moe ? `, ${profile.moe.numExperts} expertov top-${profile.moe.topK} (šírka ${profile.moe.expertWidth})` : `, šírka FFN ${profile.ffnWidth}`}. Nie sú to rozmery modelu Qwen (${profile.moe ? 'Qwen3-30B-A3B má 128 expertov, top-8' : 'Qwen3-8B má d=4096'}).`,
  );
  const name = lang === 'sk' ? (profile.kind === 'dense' ? 'Hračkový simulátor (hustý)' : 'Hračkový simulátor (MoE)') : profile.name;
  return (
    <div className="card tight" style={{ marginBottom: 12 }}>
      <SimBanner>
        <span><strong>{name}</strong>: {shape}</span>
      </SimBanner>
      <div className="row" style={{ marginTop: 8 }}>
        {allowVariant && (
          <div className="row" role="group" aria-label={T('Simulator variant', 'Variant simulátora')}>
            <button className="small" aria-pressed={state.variant === 'toy-dense'} onClick={() => dispatch({ type: 'variant', v: 'toy-dense' })}>{T('Dense toy', 'Hustá hračka')}</button>
            <button className="small" aria-pressed={state.variant === 'toy-moe'} onClick={() => dispatch({ type: 'variant', v: 'toy-moe' })}>{T('MoE toy', 'Hračka MoE')}</button>
          </div>
        )}
        <label className="field inline">{T('Seed', 'Osivo')}
          <input type="text" inputMode="numeric" value={seedDraft} style={{ width: 120 }} onChange={(e) => setSeedDraft(e.target.value.replace(/[^0-9]/g, ''))} aria-describedby="seed-help" />
        </label>
        <button className="small" disabled={!seedDraft || Number(seedDraft) === state.seed} onClick={() => dispatch({ type: 'seed', seed: Number(seedDraft) })}>{T('Apply seed', 'Použiť osivo')}</button>
        <button className="small" onClick={() => { navigator.clipboard?.writeText(String(state.seed)).then(() => setCopied(true), () => setCopied(false)); }}>{copied ? T('Seed copied', 'Osivo skopírované') : T('Copy seed', 'Kopírovať osivo')}</button>
        <button className="small" onClick={() => { setSeedDraft(String(DEFAULT_SEED)); dispatch({ type: 'resetAll' }); }}>{T('Reset simulator', 'Obnoviť simulátor')}</button>
        {edited && <span className="banner-edit">{T('Edited weights', 'Upravené váhy')} ({state.edits.length}) <button className="small" onClick={() => dispatch({ type: 'resetWeights' })}>{T('restore fixture', 'obnoviť pôvodné')}</button></span>}
      </div>
      <div id="seed-help" className="faint" style={{ fontSize: '0.78rem', marginTop: 4 }}>{T('The same seed always regenerates the same weights. Applying a seed, switching variant or editing a weight changes the model state and invalidates its cache.', 'To isté osivo vždy vytvorí tie isté váhy. Použitie osiva, zmena variantu alebo úprava váhy mení stav modelu a zneplatní jeho cache.')}</div>
    </div>
  );
}

export function PromptEditor({ max = 12 }: { max?: number }) {
  const { state, dispatch } = useToy();
  const { lang } = useLang();
  const T = (en: string, sk: string) => (lang === 'sk' ? sk : en);
  const toks = state.tokens;
  return (
    <div>
      <div className="faint" style={{ fontSize: '0.85rem' }}>{lang === 'sk' ? <>Hračkové zadanie (hračkový tokenizér po slovách, <strong>nie</strong> tokenizácia Qwen; slová sú anglické):</> : <>Toy prompt (toy word-level tokenizer, <strong>not</strong> Qwen tokenization):</>}</div>
      <div className="row" style={{ gap: 6, margin: '4px 0' }}>
        {toks.map((t, i) => (
          <label key={i} className="chip" style={{ padding: '0 4px' }}>
            <span className="faint">{i}</span>
            <select aria-label={T(`token at position ${i}`, `token na pozícii ${i}`)} value={t} onChange={(e) => dispatch({ type: 'tokens', tokens: toks.map((o, k) => (k === i ? Number(e.target.value) : o)) })} style={{ minHeight: 28, padding: '2px 4px', border: 'none', background: 'transparent' }}>
              {TOY_VOCAB.map((w, id) => <option key={id} value={id}>{w}</option>)}
            </select>
          </label>
        ))}
        <button className="small" disabled={toks.length >= max} onClick={() => dispatch({ type: 'tokens', tokens: [...toks, 11] })}>{T('+ token', '+ token')}</button>
        <button className="small" disabled={toks.length <= 1} onClick={() => dispatch({ type: 'tokens', tokens: toks.slice(0, -1) })}>{T('− last', '− posledný')}</button>
      </div>
    </div>
  );
}

export const tok = (id: number) => TOY_VOCAB[id] ?? `#${id}`;
