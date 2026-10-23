import { useState } from 'react';
import type { Block, ExamQuestion } from '../content/chapters';
import { CHAPTERS } from '../content/chapters';
import { Rich, useLang } from '../i18n/i18n';

// Hands-on exercises (with an optional hint and a hidden worked solution) and the final test.

export function Exercise({ b, n }: { b: Extract<Block, { k: 'exercise' }>; n: number }) {
  const { lang, tr } = useLang();
  const T = (en: string, sk: string) => (lang === 'sk' ? sk : en);
  return (
    <div className="card exercise" style={{ margin: '14px 0' }}>
      <strong>✍️ {T('Exercise', 'Cvičenie')} {n}</strong>
      <p style={{ margin: '6px 0' }}><Rich text={tr(b.task)} /></p>
      {b.hint && <details><summary>{T('Hint', 'Nápoveda')}</summary><p style={{ margin: '6px 0' }}><Rich text={tr(b.hint)} /></p></details>}
      <details><summary>{T('Show the worked solution', 'Zobraziť riešenie')}</summary><p style={{ margin: '6px 0' }}><Rich text={tr(b.solution)} /></p></details>
    </div>
  );
}

const KEY = 'qa-exam-best';

export function Exam({ questions }: { questions: ExamQuestion[] }) {
  const { lang, tr } = useLang();
  const T = (en: string, sk: string) => (lang === 'sk' ? sk : en);
  const [ans, setAns] = useState<(number | null)[]>(() => questions.map(() => null));
  const [checked, setChecked] = useState(false);
  const [best, setBest] = useState<number | null>(() => { try { const v = localStorage.getItem(KEY); return v === null ? null : Number(v); } catch { return null; } });
  const score = ans.filter((a, i) => a === questions[i].correct).length;
  const answered = ans.filter((a) => a !== null).length;
  const check = () => {
    setChecked(true);
    if (best === null || score > best) { setBest(score); try { localStorage.setItem(KEY, String(score)); } catch { /* storage unavailable */ } }
  };
  return (
    <div className="exam">
      {questions.map((q, i) => {
        const ch = CHAPTERS.find((c) => c.id === q.chapter);
        return (
          <fieldset key={i} className="card" style={{ margin: '12px 0' }}>
            <legend><strong>{i + 1}.</strong> <Rich text={tr(q.q)} /></legend>
            <div className="row" style={{ flexDirection: 'column', alignItems: 'stretch', gap: 4 }}>
              {q.options.map((o, j) => {
                const right = checked && j === q.correct, wrong = checked && ans[i] === j && j !== q.correct;
                return (
                  <label key={j} className={`exam-opt${right ? ' right' : ''}${wrong ? ' wrong' : ''}`}>
                    <input type="radio" name={`exam-${i}`} checked={ans[i] === j} disabled={checked} onChange={() => setAns(ans.map((a, k) => (k === i ? j : a)))} />
                    <span>{right ? '✓ ' : wrong ? '✗ ' : ''}{tr(o)}</span>
                  </label>
                );
              })}
            </div>
            {checked && <p style={{ margin: '6px 0 0', fontSize: '0.9rem' }}><Rich text={tr(q.explain)} /> {ch && <a href={`#/learn/${ch.id}`}>{T(`Chapter ${ch.n} →`, `Kapitola ${ch.n} →`)}</a>}</p>}
          </fieldset>
        );
      })}
      <div className="row" style={{ marginTop: 10 }}>
        {!checked
          ? <button className="primary" disabled={answered === 0} onClick={check}>{T(`Check my answers (${answered} of ${questions.length} answered)`, `Skontrolovať odpovede (zodpovedaných ${answered} z ${questions.length})`)}</button>
          : <button onClick={() => { setAns(questions.map(() => null)); setChecked(false); }}>{T('Try again', 'Skúsiť znova')}</button>}
        {best !== null && <span className="faint">{T(`Best score so far: ${best} / ${questions.length}`, `Doterajšie najlepšie skóre: ${best} / ${questions.length}`)}</span>}
      </div>
      {checked && (
        <p role="status" className="callout answer" style={{ marginTop: 10 }}>
          <strong>{T(`Score: ${score} / ${questions.length}`, `Skóre: ${score} / ${questions.length}`)}</strong>{' '}
          {score === questions.length ? T('Perfect: you understand how a transformer works, from one weight to the whole model.', 'Výborne: rozumiete, ako funguje transformer, od jednej váhy po celý model.')
            : score >= questions.length * 0.75 ? T('Very good. Revisit the chapters linked under the questions you missed.', 'Veľmi dobre. Vráťte sa ku kapitolám pri otázkach, ktoré ste nezodpovedali správne.')
            : T('Read the chapters linked under the missed questions and try again.', 'Prečítajte si kapitoly pri chybných odpovediach a skúste to znova.')}
        </p>
      )}
    </div>
  );
}
