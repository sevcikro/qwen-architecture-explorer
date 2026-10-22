import { useState } from 'react';
import { useTokenizerExamples } from '../data/data';
import { Cite, Evidence, Section, SimBanner, ProbRow } from '../components/ui';
import { QWEN3_8B } from '../models/profiles';
import { useToy } from '../state/toyStore';
import { stableSoftmax } from '../math/ops';
import { TOY_VOCAB } from '../math/toy';
import { explorerHref } from '../state/route';
import { useLang } from '../i18n/i18n';

export default function Pipeline() {
  const tok = useTokenizerExamples();
  const [ex, setEx] = useState('chat-nothink');
  const [selTok, setSelTok] = useState(0);
  const { trace, state } = useToy();
  const last = trace[trace.length - 1];
  const probs = stableSoftmax(last.logits).probs;
  const order = probs.map((p, i) => [p, i] as const).sort((a, b) => b[0] - a[0]);
  const p = QWEN3_8B;
  const { lang, num: fmtInt, fnum: fmtNum } = useLang();
  const T = (en: string, sk: string) => (lang === 'sk' ? sk : en);
  const q = (x: string) => (lang === 'sk' ? `„${x}“` : `“${x}”`);
  return (
    <div className="page">
      <h1>{T('Tokens, the forward pass and generation', 'Tokeny, dopredný prechod a generovanie')}</h1>
      <ol className="muted" style={{ paddingLeft: 20 }}>
        <li>{T('messages → model-specific chat template → text', 'správy → šablóna chatu daného modelu → text')}</li>
        <li>{T('text → tokenizer → token ids', 'text → tokenizér → ID tokenov')}</li>
        <li>{T('token ids → embedding row lookup', 'ID tokenov → vyhľadanie riadkov embeddingu')} → [B, T, {fmtInt(p.d)}]</li>
        <li>{T(`${p.layers} decoder blocks → final RMSNorm`, `${p.layers} blokov dekodéra → záverečná RMSNorm`)}</li>
        <li>{T(`untied vocabulary projection → ${fmtInt(p.vocabRows)} logits per position`, `neprepojená projekcia na slovník → ${fmtInt(p.vocabRows)} logitov na pozíciu`)}</li>
        <li>{T("last position's logits → temperature / filtering → sampled next token → append → repeat", 'logity poslednej pozície → teplota / filtrovanie → vylosovaný ďalší token → pripojiť → opakovať')}</li>
      </ol>
      <Section title={T('Real Qwen3 tokenizer output', 'Skutočný výstup tokenizéra Qwen3')} badges={<Evidence kind="sample" title={T('Actual tokenizer output from the pinned tokenizer files (no model was run)', 'Skutočný výstup tokenizéra z pripnutých súborov tokenizéra (žiadny model sa nespúšťal)')} />}>
        {tok.status === 'loading' && <p className="muted">{T('Loading tokenizer examples…', 'Načítavajú sa príklady tokenizéra…')}</p>}
        {tok.status === 'error' && <div className="callout warn">{T(`Tokenizer examples unavailable (${tok.error}). Everything else on this page still works.`, `Príklady tokenizéra nie sú dostupné (${tok.error}). Všetko ostatné na tejto stránke funguje.`)}</div>}
        {tok.status === 'ok' && (() => {
          const d = tok.data;
          const e = d.examples.find((x) => x.id === ex) ?? d.examples[0];
          const i = Math.min(selTok, e.ids.length - 1);
          return (
            <>
              <p className="faint" style={{ fontSize: '0.85rem' }}>{lang === 'sk' ? <>Vygenerované offline pomocou {d.library} z {d.model} @ {d.revision.slice(0, 12)} (tokenizer.json SHA-256 {d.tokenizerJsonSha256.slice(0, 16)}…), šablóna chatu z {d.chatTemplateSource}. Hodnoty sú skutočný výstup tokenizéra; žiadny model sa nespúšťal.</> : <>Generated offline with {d.library} from {d.model} @ {d.revision.slice(0, 12)} (tokenizer.json SHA-256 {d.tokenizerJsonSha256.slice(0, 16)}…), chat template from {d.chatTemplateSource}. {d.note}</>}</p>
              <div className="row" role="group" aria-label={T('Example', 'Príklad')}>
                {d.examples.map((x) => <button key={x.id} className="small" aria-pressed={x.id === e.id} onClick={() => { setEx(x.id); setSelTok(0); }}>{x.kind === 'chat' ? `chat, enable_thinking=${String(x.enable_thinking)}` : `“${x.text}”`}</button>)}
              </div>
              <h3>{T('Text the model actually receives', 'Text, ktorý model naozaj dostane')}</h3>
              <pre className="card tight mono" style={{ whiteSpace: 'pre-wrap', overflowWrap: 'anywhere', fontSize: '0.85rem' }}>{e.renderedText}</pre>
              <h3>{T(`${e.ids.length} token ids`, `${e.ids.length} ID tokenov`)}</h3>
              <div className="row" style={{ gap: 4 }}>
                {e.ids.map((id, k) => (
                  <button key={k} className={`chip${e.special[k] ? ' special' : ''}${k === i ? ' sel' : ''}`} onClick={() => setSelTok(k)} aria-label={T(`token ${k}: id ${id}, piece ${JSON.stringify(e.decodedPieces[k])}${e.special[k] ? ', added token' : ''}`, `token ${k}: ID ${id}, kúsok ${JSON.stringify(e.decodedPieces[k])}${e.special[k] ? ', pridaný token' : ''}`)}>
                    <span>{JSON.stringify(e.decodedPieces[k]).slice(1, -1) || '∅'}</span><span className="faint">{id}</span>
                  </button>
                ))}
              </div>
              <dl className="kv" style={{ marginTop: 10 }}>
                <dt>{T('position', 'pozícia')}</dt><dd>{i}</dd>
                <dt>{T('token id', 'ID tokenu')}</dt><dd className="mono">{e.ids[i]}</dd>
                <dt>{T('byte-level BPE symbol', 'symbol BPE na úrovni bajtov')}</dt><dd className="mono">{e.tokens[i]}</dd>
                <dt>{T('decoded piece', 'dekódovaný kúsok')}</dt><dd className="mono">{JSON.stringify(e.decodedPieces[i])}</dd>
                <dt>{T('kind', 'druh')}</dt><dd>{e.special[i] ? T('added token (control / template token)', 'pridaný token (riadiaci token / token šablóny)') : T('regular vocabulary token', 'bežný token slovníka')}</dd>
                <dt>{T('embedding row', 'riadok embeddingu')}</dt><dd><a href={explorerHref({ m: 'qwen3-8b', lv: 1, t: 'embed_tokens', r: e.ids[i] })}>model.embed_tokens.weight[{e.ids[i]}, :]</a> {T('(4,096 learned numbers; values not loaded)', '(4 096 naučených čísel; hodnoty sa nenačítavajú)')}</dd>
              </dl>
              {e.decodedPieces.some((x) => x.includes('�')) && <p className="faint" style={{ fontSize: '0.85rem' }}>{T('Byte-level BPE can split one multi-byte character across tokens: here “ď” (2 bytes in UTF-8) becomes two tokens, and each of them decoded alone shows a replacement character (shown as “�”). Tokens are not words or characters.', 'BPE na úrovni bajtov môže jeden viacbajtový znak rozdeliť na viac tokenov: tu sa „ď“ (2 bajty v UTF-8) stane dvoma tokenmi a každý z nich dekódovaný samostatne ukáže náhradný znak („�“). Tokeny nie sú slová ani znaky.')}</p>}
              <h3>{T('Vocabulary facts', 'Fakty o slovníku')}</h3>
              <ul style={{ paddingLeft: 18 }}>
                {lang === 'sk' ? <>
                <li>{fmtInt(d.baseVocabSize)} bežných tokenov + {d.addedTokens.length} pridaných tokenov (ID {d.addedTokens[0].id}–{d.addedTokens[d.addedTokens.length - 1].id}) = <strong>{fmtInt(d.vocabWithAddedTokens)}</strong>, v súlade so 151 669 v správe.<Cite ids={['q8-tokenizer', 'qwen3-report']} /></li>
                <li>Tabuľka embeddingov a výstupná hlava majú <strong>{fmtInt(p.vocabRows)}</strong> riadkov (vocab_size v konfigurácii). {fmtInt(p.vocabRows - d.vocabWithAddedTokens)} riadkov navyše je výplň, na ktorú sa nemapuje žiadne ID tokenu; hlava pre ne aj tak počíta logity.</li>
                <li>Medzi pridané tokeny patria {d.addedTokens.slice(0, 3).map((t) => t.content).join(', ')} … {d.addedTokens.slice(-2).map((t) => t.content).join(', ')}.</li>
                </> : <>
                <li>{fmtInt(d.baseVocabSize)} regular tokens + {d.addedTokens.length} added tokens (ids {d.addedTokens[0].id}–{d.addedTokens[d.addedTokens.length - 1].id}) = <strong>{fmtInt(d.vocabWithAddedTokens)}</strong>, matching the report's 151,669.<Cite ids={['q8-tokenizer', 'qwen3-report']} /></li>
                <li>The embedding table and output head have <strong>{fmtInt(p.vocabRows)}</strong> rows (config vocab_size). The {fmtInt(p.vocabRows - d.vocabWithAddedTokens)} extra rows are padding no token id maps to; the head still computes logits for them.</li>
                <li>Added tokens include {d.addedTokens.slice(0, 3).map((t) => t.content).join(', ')} … {d.addedTokens.slice(-2).map((t) => t.content).join(', ')}.</li>
                </>}
              </ul>
            </>
          );
        })()}
      </Section>
      <Section title={T('Logits are not probabilities', 'Logity nie sú pravdepodobnosti')} badges={<Evidence kind="simulation" />}>
        <SimBanner>{T('Logits below come from the toy simulator with its toy word-level tokenizer (kept separate from the real Qwen tokenizer output above).', 'Logity nižšie pochádzajú z hračkového simulátora s jeho hračkovým tokenizérom po slovách (oddelene od skutočného výstupu tokenizéra Qwen vyššie).')}</SimBanner>
        <p>{T(`Prompt: “${state.tokens.map((t) => TOY_VOCAB[t]).join(' ')}”. The head produces one logit per vocabulary row (here ${TOY_VOCAB.length}; in Qwen3 all ${fmtInt(p.vocabRows)} are computed conceptually). Softmax turns them into probabilities. Showing selected entries, sorted:`, `Zadanie: ${q(state.tokens.map((t) => TOY_VOCAB[t]).join(' '))}. Hlava vytvorí jeden logit pre každý riadok slovníka (tu ${TOY_VOCAB.length}; v Qwen3 sa koncepčne počíta všetkých ${fmtInt(p.vocabRows)}). Softmax ich zmení na pravdepodobnosti. Zobrazené položky, zoradené:`)}</p>
        {order.map(([pr, i]) => <ProbRow key={i} label={TOY_VOCAB[i]} p={pr} extra={`${fmtNum(last.logits[i], 3)} → ${lang === 'sk' ? pr.toFixed(3).replace('.', ',') : pr.toFixed(3)}`} />)}
        <p><a href="#/labs/sampling">{T('Turn these into a next token in the sampling lab →', 'Premeňte ich na ďalší token v laboratóriu vzorkovania →')}</a></p>
      </Section>
      <Section title={T('Thinking and non-thinking', 'S premýšľaním a bez neho')} badges={<Evidence kind="verified" />}>
        {lang === 'sk' ? <p>Porovnajte dva príklady chatu vyššie: pri <code>enable_thinking=False</code> šablóna za <code>&lt;|im_start|&gt;assistant\n</code> pripojí prázdne <code>&lt;think&gt;\n\n&lt;/think&gt;\n\n</code> (ID 151667, 271, 151668, 271); pri zapnutom premýšľaní model najprv sám generuje svoje úvahy. Rovnaké váhy, rovnaké vrstvy; líši sa len text, na ktorom je model podmienený.<Cite ids={['q8-tokenizer', 'qwen3-report']} /></p> : <p>Compare the two chat examples above: with <code>enable_thinking=False</code> the template appends an empty <code>&lt;think&gt;\n\n&lt;/think&gt;\n\n</code> (ids 151667, 271, 151668, 271) after <code>&lt;|im_start|&gt;assistant\n</code>; with thinking enabled the model is left to generate its own reasoning first. Same weights, same layers; only the conditioning text differs.<Cite ids={['q8-tokenizer', 'qwen3-report']} /></p>}
      </Section>
    </div>
  );
}
