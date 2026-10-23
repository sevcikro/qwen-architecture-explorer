import { useEffect, useRef, useState } from 'react';
import { useLang } from '../i18n/i18n';
import { Section } from '../components/ui';
import { MODEL_ID, MODEL_REVISION, MODEL_URL, VARIANT_BYTES, type Dtype } from '../live/config';
import type { FromWorker, ToWorker } from '../live/worker';

// Optional live demo: a real Qwen3-0.6B (ONNX conversion) running in the visitor's browser.
// Nothing is downloaded until the visitor presses the start button.

type Plan = { device: 'webgpu' | 'wasm'; dtype: Dtype; bytes: number; why: string };
type Msg = { role: 'user' | 'assistant'; content: string };

async function detect(lang: 'en' | 'sk'): Promise<Plan> {
  const T = (en: string, sk: string) => (lang === 'sk' ? sk : en);
  const gpu = (navigator as unknown as { gpu?: { requestAdapter: () => Promise<{ features: Set<string> } | null> } }).gpu;
  if (gpu) {
    try {
      const ad = await gpu.requestAdapter();
      if (ad && ad.features.has('shader-f16')) return { device: 'webgpu', dtype: 'q4f16', bytes: VARIANT_BYTES.q4f16, why: T('WebGPU with 16-bit floats is available: the model runs on your graphics chip (4-bit weights).', 'WebGPU so 16-bitovými číslami je k dispozícii: model pobeží na vašom grafickom čipe (4-bitové váhy).') };
    } catch { /* fall back to WASM */ }
  }
  return { device: 'wasm', dtype: 'q4', bytes: VARIANT_BYTES.q4, why: T('No suitable WebGPU: the model runs on your processor with WebAssembly (4-bit weights). This works almost everywhere but is slow: often around one token per second on a laptop processor.', 'Vhodné WebGPU nie je k dispozícii: model pobeží na procesore cez WebAssembly (4-bitové váhy). Funguje to takmer všade, ale pomaly: na procesore notebooku často okolo jedného tokenu za sekundu.') };
}

export default function Live() {
  const { lang, num, fnum } = useLang();
  const T = (en: string, sk: string) => (lang === 'sk' ? sk : en);
  const [plan, setPlan] = useState<Plan | null>(null);
  const [state, setState] = useState<'idle' | 'loading' | 'ready' | 'busy' | 'error'>('idle');
  const [error, setError] = useState('');
  const [files, setFiles] = useState<Record<string, { loaded: number; total: number }>>({});
  const [ready, setReady] = useState<{ device: string; dtype: string; seconds: number } | null>(null);
  const [mode, setMode] = useState<'chat' | 'next'>('chat');
  const [chat, setChat] = useState<Msg[]>([]);
  const [draft, setDraft] = useState(T('What is a parameter in a neural network? Answer in two sentences.', 'Čo je parameter v neurónovej sieti? Odpovedz dvoma vetami.'));
  const [live, setLive] = useState<{ text: string; count: number; elapsed: number } | null>(null);
  const [last, setLast] = useState<{ count: number; elapsed: number; promptTokens: number; stopped: string } | null>(null);
  const [sample, setSample] = useState(false);
  const [nextText, setNextText] = useState('The capital of France is');
  const [next, setNext] = useState<Extract<FromWorker, { type: 'next' }> | null>(null);
  const worker = useRef<Worker | null>(null);
  useEffect(() => { detect(lang).then((p) => { const forced = new URLSearchParams(window.location.hash.split('?')[1] ?? '').get('dtype') as Dtype | null; setPlan(forced && forced in VARIANT_BYTES ? { ...p, dtype: forced, bytes: VARIANT_BYTES[forced] } : p); }); }, [lang]);
  useEffect(() => () => worker.current?.terminate(), []);

  const send = (m: ToWorker) => worker.current?.postMessage(m);
  const start = () => {
    if (!plan) return;
    setState('loading'); setError('');
    const w = new Worker(new URL('../live/worker.ts', import.meta.url), { type: 'module' });
    worker.current = w;
    w.onmessage = (ev: MessageEvent<FromWorker>) => {
      const m = ev.data;
      if (m.type === 'progress') setFiles((f) => ({ ...f, [m.file]: { loaded: m.loaded, total: m.total } }));
      else if (m.type === 'ready') { setReady(m); setState('ready'); }
      else if (m.type === 'token') setLive({ text: m.text, count: m.count, elapsed: m.elapsed });
      else if (m.type === 'done') { setChat((c) => [...c, { role: 'assistant', content: m.text }]); setLive(null); setLast(m); setState('ready'); }
      else if (m.type === 'next') { setNext(m); setState('ready'); }
      else if (m.type === 'error') { setError(m.message); setState(ready ? 'ready' : 'error'); }
    };
    w.onerror = (e) => { setError(e.message || 'worker error'); setState('error'); };
    send({ type: 'load', device: plan.device, dtype: plan.dtype });
  };
  const ask = () => {
    if (!draft.trim()) return;
    const msgs: Msg[] = [...chat, { role: 'user', content: draft.trim() }];
    setChat(msgs); setDraft(''); setState('busy'); setLive({ text: '', count: 0, elapsed: 0 }); setLast(null);
    send({ type: 'chat', messages: msgs, maxNewTokens: 256, sample, temperature: 0.7 });
  };
  const predict = (text: string) => { setNextText(text); setState('busy'); send({ type: 'next', text, k: 10 }); };
  const totalLoaded = Object.values(files).reduce((s, f) => s + f.loaded, 0);
  const totalBytes = Object.values(files).reduce((s, f) => s + f.total, 0);
  const busy = state === 'busy';

  return (
    <div className="page">
      <h1>{T('Run a real Qwen model in your browser', 'Spustite skutočný model Qwen vo svojom prehliadači')}</h1>
      <p className="muted" style={{ fontSize: '1.05rem' }}>{T('Everything else on this site uses recorded runs or a toy simulator. Here you can download the real Qwen3-0.6B (an ONNX conversion of the same weights) and run it on your own device: chat with it, or watch the probabilities of the next token for any text you type.', 'Všetko ostatné na tejto stránke používa zaznamenané behy alebo hračkový simulátor. Tu si môžete stiahnuť skutočný Qwen3-0.6B (konverziu tých istých váh do ONNX) a spustiť ho na vlastnom zariadení: rozprávať sa s ním alebo sledovať pravdepodobnosti ďalšieho tokenu pre ľubovoľný text.')}</p>

      <div className="callout warn">
        <strong>{T('Before you start', 'Skôr než začnete')}</strong>
        <ul style={{ paddingLeft: 18, margin: '6px 0 0' }}>
          <li>{T(`Your browser will download about ${plan ? fnum(plan.bytes / 1e6, 0) : '570–920'} MB of model files directly from huggingface.co (${MODEL_ID} at a pinned revision). Hugging Face sees that download like any other visit; this site sends it nothing else. The files are kept in your browser’s cache for next time.`, `Váš prehliadač stiahne asi ${plan ? fnum(plan.bytes / 1e6, 0) : '570–920'} MB súborov modelu priamo z huggingface.co (${MODEL_ID} v pripnutej revízii). Hugging Face toto sťahovanie vidí ako každú inú návštevu; táto stránka mu nič iné neposiela. Súbory zostanú v cache prehliadača na budúce použitie.`)}</li>
          <li>{T('Your prompts and the answers never leave your device: the model runs locally.', 'Vaše zadania a odpovede nikdy neopustia vaše zariadenie: model beží lokálne.')}</li>
          <li>{T('It needs about 1–2 GB of free memory. On phones it may be slow or fail; a computer with a recent Chrome or Edge works best.', 'Potrebuje asi 1–2 GB voľnej pamäte. Na mobiloch môže byť pomalý alebo zlyhať; najlepšie funguje počítač s aktuálnym Chrome alebo Edge.')}</li>
          <li>{T('The weights are quantized for the browser, so answers can differ slightly from the original BF16 model. It is a small model: it makes mistakes and invents facts.', 'Váhy sú pre prehliadač kvantizované, takže odpovede sa môžu mierne líšiť od pôvodného modelu v BF16. Je to malý model: robí chyby a vymýšľa si fakty.')}</li>
        </ul>
      </div>

      {plan && <p className="faint" style={{ fontSize: '0.88rem' }}>{plan.why}</p>}
      {plan && state === 'idle' && (
        <label className="field" style={{ marginBottom: 8, maxWidth: 420 }}>{T('Weight format', 'Formát váh')}
          <select style={{ maxWidth: '100%' }} value={plan.dtype} onChange={(e) => { const d = e.target.value as Dtype; setPlan({ ...plan, dtype: d, bytes: VARIANT_BYTES[d] }); }}>
            {(plan.device === 'webgpu' ? (['q4f16', 'q4'] as Dtype[]) : (['q4', 'q8'] as Dtype[])).map((d) => (
              <option key={d} value={d}>{d} · {fnum(VARIANT_BYTES[d] / 1e6, 0)} MB{d === 'q8' ? T(' (less accurate)', ' (menej presné)') : ''}</option>
            ))}
          </select>
        </label>
      )}
      <details style={{ margin: '4px 0 10px' }}>
        <summary className="faint" style={{ cursor: 'pointer' }}>{T('How close is the browser version to the original?', 'Nakoľko sa verzia v prehliadači blíži originálu?')}</summary>
        <p className="faint" style={{ fontSize: '0.86rem' }}>{T('Our own test for the text “The capital of France is”: the original float32 model in PyTorch gives “ Paris” probability 0.657. In the browser (WebAssembly) the 4-bit file (q4) gave 0.517 and still ranked “ Paris” first; the 8-bit file (q8) gave only 0.016. Quantization changes the numbers; that is why q4 is the default here.', 'Náš vlastný test pre text „The capital of France is“: pôvodný model vo float32 v PyTorch dáva „ Paris“ pravdepodobnosť 0,657. V prehliadači (WebAssembly) dal 4-bitový súbor (q4) 0,517 a „ Paris“ zostal na prvom mieste; 8-bitový súbor (q8) dal len 0,016. Kvantizácia mení čísla; preto je tu predvolený q4.')}</p>
      </details>
      {state === 'idle' && <button className="primary" disabled={!plan} onClick={start} data-testid="live-start">{T(`Download (${plan ? fnum(plan.bytes / 1e6, 0) : '…'} MB) and start the model`, `Stiahnuť (${plan ? fnum(plan.bytes / 1e6, 0) : '…'} MB) a spustiť model`)}</button>}
      {state === 'loading' && (
        <div className="card tight" role="status" aria-live="polite">
          <div>{T('Downloading and preparing the model…', 'Sťahujem a pripravujem model…')} {totalBytes > 0 && `${fnum(totalLoaded / 1e6, 0)} / ${fnum(totalBytes / 1e6, 0)} MB`}</div>
          <div className="progress" style={{ marginTop: 6 }}><span style={{ width: `${totalBytes ? (totalLoaded / totalBytes) * 100 : 2}%` }} /></div>
          <div className="faint" style={{ fontSize: '0.8rem', marginTop: 4 }}>{Object.keys(files).join(' · ')}</div>
        </div>
      )}
      {error && <div className="callout warn" role="alert">{T('Something went wrong:', 'Niečo sa pokazilo:')} <code>{error}</code> {state === 'error' && <button className="small" onClick={() => { worker.current?.terminate(); setState('idle'); setFiles({}); }}>{T('Try again', 'Skúsiť znova')}</button>}</div>}

      {ready && (
        <>
          <p className="faint" style={{ fontSize: '0.85rem' }} data-testid="live-ready">{T(`Model ready on ${ready.device === 'webgpu' ? 'WebGPU' : 'WebAssembly (CPU)'} with ${ready.dtype} weights after ${fnum(ready.seconds, 1)} s.`, `Model je pripravený na ${ready.device === 'webgpu' ? 'WebGPU' : 'WebAssembly (CPU)'} s váhami ${ready.dtype} po ${fnum(ready.seconds, 1)} s.`)}</p>
          <div className="row" role="tablist" aria-label={T('Demo mode', 'Režim ukážky')}>
            <button role="tab" aria-selected={mode === 'chat'} className="small" onClick={() => setMode('chat')}>{T('Chat', 'Rozhovor')}</button>
            <button role="tab" aria-selected={mode === 'next'} className="small" onClick={() => { setMode('next'); if (!next) predict(nextText); }}>{T('Next-token probabilities', 'Pravdepodobnosti ďalšieho tokenu')}</button>
          </div>
          {mode === 'chat' ? (
            <Section title={T('Chat with Qwen3-0.6B', 'Rozhovor s Qwen3-0.6B')}>
              <div className="live-chat" aria-live="polite">
                {chat.map((m, i) => <div key={i} className={`bubble ${m.role}`}><span className="faint">{m.role === 'user' ? T('You', 'Vy') : 'Qwen3-0.6B'}</span><div>{m.content}</div></div>)}
                {live && <div className="bubble assistant"><span className="faint">Qwen3-0.6B</span><div>{live.text}<span className="cursor" aria-hidden="true">▍</span></div></div>}
              </div>
              <label className="field" style={{ marginTop: 8 }}>{T('Your message', 'Vaša správa')}
                <textarea rows={3} value={draft} onChange={(e) => setDraft(e.target.value)} onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey && !busy) { e.preventDefault(); ask(); } }} />
              </label>
              <div className="row" style={{ marginTop: 6 }}>
                <button className="primary" disabled={busy || !draft.trim()} onClick={ask} data-testid="live-send">{T('Send', 'Odoslať')}</button>
                <button disabled={!busy} onClick={() => send({ type: 'stop' })}>{T('Stop', 'Zastaviť')}</button>
                <button disabled={busy || chat.length === 0} onClick={() => { setChat([]); setLast(null); }}>{T('New conversation', 'Nový rozhovor')}</button>
                <label className="field inline"><input type="checkbox" checked={sample} onChange={(e) => setSample(e.target.checked)} /> {T('sample (temperature 0.7, top-k 20, top-p 0.8) instead of greedy', 'vzorkovať (teplota 0,7, top-k 20, top-p 0,8) namiesto hladového výberu')}</label>
              </div>
              {(live || last) && (
                <p className="mono faint" style={{ fontSize: '0.82rem' }} data-testid="live-stats">
                  {T('tokens', 'tokeny')}: {num((live ?? last)!.count)} · {fnum((live ?? last)!.count / Math.max(0.001, (live ?? last)!.elapsed), 1)} {T('tokens/s', 'tokenov/s')}
                  {last && ` · ${T('prompt', 'zadanie')}: ${num(last.promptTokens)} ${T('tokens (the whole conversation in the chat template)', 'tokenov (celý rozhovor v šablóne chatu)')} · ${last.stopped === 'end' ? T('stopped on <|im_end|>', 'zastavené na <|im_end|>') : last.stopped === 'user' ? T('stopped by you', 'zastavili ste') : T('length limit', 'limit dĺžky')}`}
                </p>
              )}
              <p className="faint" style={{ fontSize: '0.84rem' }}>{T('Every message sends the whole conversation through the chat template again; the model has no memory of its own. See ', 'Každá správa znova pošle celý rozhovor cez šablónu chatu; model nemá vlastnú pamäť. Pozrite ')}<a href="#/learn/journey">{T('From your message to the answer', 'Od vašej správy k odpovedi')}</a>.</p>
            </Section>
          ) : (
            <Section title={T('What comes next? Real probabilities', 'Čo nasleduje? Skutočné pravdepodobnosti')}>
              <label className="field">{T('Text (no chat template; the model just continues it)', 'Text (bez šablóny chatu; model ho len pokračuje)')}
                <input type="text" value={nextText} onChange={(e) => setNextText(e.target.value)} onKeyDown={(e) => { if (e.key === 'Enter' && !busy) predict(nextText); }} />
              </label>
              <div className="row" style={{ marginTop: 6 }}><button className="primary" disabled={busy} onClick={() => predict(nextText)} data-testid="live-predict">{T('Compute', 'Vypočítať')}</button></div>
              {next && (
                <>
                  <div className="row" style={{ gap: 4, margin: '8px 0' }}>{next.pieces.map((p, i) => <span key={i} className="chip"><span>{JSON.stringify(p).slice(1, -1) || '∅'}</span><span className="faint">{next.promptIds[i]}</span></span>)}</div>
                  <div style={{ display: 'grid', gap: 4 }} data-testid="live-top">
                    {next.top.map((c) => (
                      <button key={c.id} className="prob-row ghost" style={{ gridTemplateColumns: '110px minmax(0,1fr) 64px', textAlign: 'left' }} disabled={busy} onClick={() => predict(nextText + c.piece)} title={T('Append this token and compute again', 'Pripojiť tento token a vypočítať znova')}>
                        <span className="mono">{JSON.stringify(c.piece).slice(1, -1)}</span>
                        <span className="track" aria-hidden="true"><span className="fill" style={{ width: `${c.p * 100}%`, display: 'block' }} /></span>
                        <span className="mono num">{fnum(c.p, 3)}</span>
                      </button>
                    ))}
                  </div>
                  <p className="faint" style={{ fontSize: '0.84rem' }}>{T(`One forward pass over ${next.promptIds.length} tokens took ${fnum(next.elapsed, 2)} s. Click a candidate to append it: that is exactly one step of generation, chosen by you.`, `Jeden dopredný prechod cez ${next.promptIds.length} tokenov trval ${fnum(next.elapsed, 2)} s. Kliknutím na kandidáta ho pripojíte: to je presne jeden krok generovania, ktorý vyberiete vy.`)}</p>
                </>
              )}
            </Section>
          )}
        </>
      )}
      <p className="faint" style={{ fontSize: '0.84rem', marginTop: 16 }}>{T('Model files: ', 'Súbory modelu: ')}<a href={MODEL_URL} target="_blank" rel="noreferrer">{MODEL_ID}</a> @ <span className="mono">{MODEL_REVISION.slice(0, 12)}</span>{T(', an ONNX conversion of Qwen/Qwen3-0.6B (Apache-2.0). Runtime: Transformers.js 4.3.1 and ONNX Runtime Web, served by this site.', ', konverzia Qwen/Qwen3-0.6B do ONNX (Apache-2.0). Prostredie: Transformers.js 4.3.1 a ONNX Runtime Web, poskytované touto stránkou.')}</p>
    </div>
  );
}
