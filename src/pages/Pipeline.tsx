import { useState } from 'react';
import { useTokenizerExamples } from '../data/data';
import { Cite, Evidence, Section, SimBanner, fmtInt, fmtNum, ProbRow } from '../components/ui';
import { QWEN3_8B } from '../models/profiles';
import { useToy } from '../state/toyStore';
import { stableSoftmax } from '../math/ops';
import { TOY_VOCAB } from '../math/toy';
import { explorerHref } from '../state/route';

export default function Pipeline() {
  const tok = useTokenizerExamples();
  const [ex, setEx] = useState('chat-nothink');
  const [selTok, setSelTok] = useState(0);
  const { trace, state } = useToy();
  const last = trace[trace.length - 1];
  const probs = stableSoftmax(last.logits).probs;
  const order = probs.map((p, i) => [p, i] as const).sort((a, b) => b[0] - a[0]);
  const p = QWEN3_8B;
  return (
    <div className="page">
      <h1>Tokens, the forward pass and generation</h1>
      <ol className="muted" style={{ paddingLeft: 20 }}>
        <li>messages → model-specific chat template → text</li>
        <li>text → tokenizer → token ids</li>
        <li>token ids → embedding row lookup → [B, T, {fmtInt(p.d)}]</li>
        <li>{p.layers} decoder blocks → final RMSNorm</li>
        <li>untied vocabulary projection → {fmtInt(p.vocabRows)} logits per position</li>
        <li>last position's logits → temperature / filtering → sampled next token → append → repeat</li>
      </ol>
      <Section title="Real Qwen3 tokenizer output" badges={<Evidence kind="sample" title="Actual tokenizer output from the pinned tokenizer files (no model was run)" />}>
        {tok.status === 'loading' && <p className="muted">Loading tokenizer examples…</p>}
        {tok.status === 'error' && <div className="callout warn">Tokenizer examples unavailable ({tok.error}). Everything else on this page still works.</div>}
        {tok.status === 'ok' && (() => {
          const d = tok.data;
          const e = d.examples.find((x) => x.id === ex) ?? d.examples[0];
          const i = Math.min(selTok, e.ids.length - 1);
          return (
            <>
              <p className="faint" style={{ fontSize: '0.85rem' }}>Generated offline with {d.library} from {d.model} @ {d.revision.slice(0, 12)} (tokenizer.json SHA-256 {d.tokenizerJsonSha256.slice(0, 16)}…), chat template from {d.chatTemplateSource}. {d.note}</p>
              <div className="row" role="group" aria-label="Example">
                {d.examples.map((x) => <button key={x.id} className="small" aria-pressed={x.id === e.id} onClick={() => { setEx(x.id); setSelTok(0); }}>{x.kind === 'chat' ? `chat, enable_thinking=${String(x.enable_thinking)}` : `“${x.text}”`}</button>)}
              </div>
              <h3>Text the model actually receives</h3>
              <pre className="card tight mono" style={{ whiteSpace: 'pre-wrap', overflowWrap: 'anywhere', fontSize: '0.85rem' }}>{e.renderedText}</pre>
              <h3>{e.ids.length} token ids</h3>
              <div className="row" style={{ gap: 4 }}>
                {e.ids.map((id, k) => (
                  <button key={k} className={`chip${e.special[k] ? ' special' : ''}${k === i ? ' sel' : ''}`} onClick={() => setSelTok(k)} aria-label={`token ${k}: id ${id}, piece ${JSON.stringify(e.decodedPieces[k])}${e.special[k] ? ', added token' : ''}`}>
                    <span>{JSON.stringify(e.decodedPieces[k]).slice(1, -1) || '∅'}</span><span className="faint">{id}</span>
                  </button>
                ))}
              </div>
              <dl className="kv" style={{ marginTop: 10 }}>
                <dt>position</dt><dd>{i}</dd>
                <dt>token id</dt><dd className="mono">{e.ids[i]}</dd>
                <dt>byte-level BPE symbol</dt><dd className="mono">{e.tokens[i]}</dd>
                <dt>decoded piece</dt><dd className="mono">{JSON.stringify(e.decodedPieces[i])}</dd>
                <dt>kind</dt><dd>{e.special[i] ? 'added token (control / template token)' : 'regular vocabulary token'}</dd>
                <dt>embedding row</dt><dd><a href={explorerHref({ m: 'qwen3-8b', lv: 1, t: 'embed_tokens', r: e.ids[i] })}>model.embed_tokens.weight[{e.ids[i]}, :]</a> (4,096 learned numbers; values not loaded)</dd>
              </dl>
              {e.id === 'raw-czech' && <p className="faint" style={{ fontSize: '0.85rem' }}>Byte-level BPE can split a multi-byte character such as “ž” or “á” across pieces; a piece decoded alone may then show a replacement character. Tokens are not words or characters.</p>}
              <h3>Vocabulary facts</h3>
              <ul style={{ paddingLeft: 18 }}>
                <li>{fmtInt(d.baseVocabSize)} regular tokens + {d.addedTokens.length} added tokens (ids {d.addedTokens[0].id}–{d.addedTokens[d.addedTokens.length - 1].id}) = <strong>{fmtInt(d.vocabWithAddedTokens)}</strong>, matching the report's 151,669.<Cite ids={['q8-tokenizer', 'qwen3-report']} /></li>
                <li>The embedding table and output head have <strong>{fmtInt(p.vocabRows)}</strong> rows (config vocab_size). The {fmtInt(p.vocabRows - d.vocabWithAddedTokens)} extra rows are padding no token id maps to; the head still computes logits for them.</li>
                <li>Added tokens include {d.addedTokens.slice(0, 3).map((t) => t.content).join(', ')} … {d.addedTokens.slice(-2).map((t) => t.content).join(', ')}.</li>
              </ul>
            </>
          );
        })()}
      </Section>
      <Section title="Logits are not probabilities" badges={<Evidence kind="simulation" />}>
        <SimBanner>Logits below come from the toy simulator with its toy word-level tokenizer (kept separate from the real Qwen tokenizer output above).</SimBanner>
        <p>Prompt: “{state.tokens.map((t) => TOY_VOCAB[t]).join(' ')}”. The head produces one logit per vocabulary row (here {TOY_VOCAB.length}; in Qwen3 all {fmtInt(p.vocabRows)} are computed conceptually). Softmax turns them into probabilities. Showing selected entries, sorted:</p>
        {order.map(([pr, i]) => <ProbRow key={i} label={TOY_VOCAB[i]} p={pr} extra={`${fmtNum(last.logits[i], 3)} → ${pr.toFixed(3)}`} />)}
        <p><a href="#/labs/sampling">Turn these into a next token in the sampling lab →</a></p>
      </Section>
      <Section title="Thinking and non-thinking" badges={<Evidence kind="verified" />}>
        <p>Compare the two chat examples above: with <code>enable_thinking=False</code> the template appends an empty <code>&lt;think&gt;\n\n&lt;/think&gt;\n\n</code> (ids 151667, 271, 151668, 271) after <code>&lt;|im_start|&gt;assistant\n</code>; with thinking enabled the model is left to generate its own reasoning first. Same weights, same layers; only the conditioning text differs.<Cite ids={['q8-tokenizer', 'qwen3-report']} /></p>
      </Section>
    </div>
  );
}
