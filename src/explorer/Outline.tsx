import { useMemo, useState } from 'react';
import type { CoreProfile } from '../models/profiles';
import { tensorFamilies, tensorName } from '../models/shapes';
import { explorerHref, type ExplorerState } from '../state/route';
import { useLang } from '../i18n/i18n';
import { locateTensor } from './locate';

/** Structured, searchable alternative to the canvas: reaches every block, module, tensor, expert and coordinate. */
export function Outline({ p, s, onClose }: { p: CoreProfile; s: ExplorerState; onClose: () => void }) {
  const [q, setQ] = useState('');
  const { lang, num: fmtInt, shape: shapeStr } = useLang();
  const T = (en: string, sk: string) => (lang === 'sk' ? sk : en);
  const [openBlock, setOpenBlock] = useState<number | null>(s.b ?? null);
  const [expert, setExpert] = useState(s.e ?? 0);
  const fams = tensorFamilies(p);
  const blockFams = fams.filter((f) => f.scope !== 'model');
  const href = (x: Partial<ExplorerState>) => explorerHref({ m: p.id, ...x });
  const results = useMemo(() => {
    const t = q.trim();
    if (!t) return [];
    const exact = locateTensor(p, t) ?? locateTensor(p, t.startsWith('model.') || t === 'lm_head.weight' ? t : `model.${t}`);
    const out: { label: string; href: string }[] = [];
    if (exact) out.push({ label: `${T('Exact tensor', 'Presný tenzor')}: ${t}`, href: href({ lv: 2, t: exact.key, b: exact.b, e: exact.e }) });
    const low = t.toLowerCase();
    for (const f of fams) if (f.label.toLowerCase().includes(low) || f.labelSk.toLowerCase().includes(low) || f.pathTemplate.toLowerCase().includes(low) || f.key.includes(low)) out.push({ label: `${lang === 'sk' ? f.labelSk : f.label} — ${f.pathTemplate}`, href: href({ lv: 2, t: f.key, b: f.scope === 'model' ? undefined : s.b ?? 0, e: f.scope === 'expert' ? 0 : undefined }) });
    const m = low.match(/(?:block|blok)\s*(\d+)|layers\.(\d+)/);
    if (m) {
      const b = Number(m[1] ?? m[2]);
      if (b < p.layers) out.push({ label: T(`Decoder block ${b}`, `Blok dekodéra ${b}`), href: href({ lv: 4, b }) });
    }
    return out.slice(0, 50);
  }, [q, p, fams, s.b, lang]); // eslint-disable-line react-hooks/exhaustive-deps
  const cur = (x: Partial<ExplorerState>) => (x.b === s.b && x.t === s.t && (x.e === undefined || x.e === s.e) ? 'true' : undefined);
  return (
    <div className="outline-panel" role="region" aria-label={T('Model outline', 'Prehľad modelu')}>
      <div className="row" style={{ justifyContent: 'space-between' }}>
        <h2 style={{ margin: 0 }}>{T('Outline', 'Prehľad')} · {p.name}</h2>
        <button onClick={onClose}>{T('Back to diagram', 'Späť na diagram')}</button>
      </div>
      <label className="field" style={{ margin: '10px 0', maxWidth: 520 }}>
        {T('Search tensors, modules or blocks (e.g. “o_proj”, “block 12”, “layers.3.mlp.experts.17.up_proj.weight”)', 'Hľadať tenzory, moduly alebo bloky (napr. „o_proj“, „blok 12“, „layers.3.mlp.experts.17.up_proj.weight“)')}
        <input type="search" value={q} onChange={(e) => setQ(e.target.value)} placeholder={T('search…', 'hľadať…')} />
      </label>
      {q && (
        <ul aria-label={T('Search results', 'Výsledky hľadania')} style={{ paddingLeft: 18 }}>
          {results.length === 0 ? <li className="muted">{T('No match in', 'Žiadna zhoda v')} {p.name}.</li> : results.map((r) => <li key={r.href}><a href={r.href}>{r.label}</a></li>)}
        </ul>
      )}
      <div className="tree">
        <ul>
          <li><a href={href({ lv: 5 })}>{p.name} ({T('full model', 'celý model')})</a>
            <ul>
              <li><a href={href({ lv: 2, t: 'embed_tokens' })} aria-current={cur({ t: 'embed_tokens' })}>{T('Embedding table', 'Tabuľka embeddingov')} {shapeStr([p.vocabRows, p.d])}</a></li>
              <li>
                <span className="muted">{T(`Decoder blocks (${p.layers}) — expand one:`, `Bloky dekodéra (${p.layers}) — rozbaľte jeden:`)}</span>
                <ul>
                  {Array.from({ length: p.layers }, (_, b) => (
                    <li key={b}>
                      <button aria-expanded={openBlock === b} onClick={() => setOpenBlock(openBlock === b ? null : b)}>{openBlock === b ? '▾' : '▸'} {T('Block', 'Blok')} {b}</button>{' '}
                      <a href={href({ lv: 4, b })} style={{ fontSize: '0.85rem' }}>{T('open', 'otvoriť')}</a>
                      {openBlock === b && (
                        <ul>
                          <li><a href={href({ lv: 3, b, mod: 'attn_norm' })}>{T('RMSNorm (attention)', 'RMSNorm (pozornosť)')}</a></li>
                          <li><a href={href({ lv: 3, b, mod: 'attention' })}>{T('Attention', 'Pozornosť')}</a></li>
                          <li><a href={href({ lv: 3, b, mod: 'residual1' })}>{T('Residual add 1 (no weights)', 'Reziduálne sčítanie 1 (bez váh)')}</a></li>
                          <li><a href={href({ lv: 3, b, mod: 'ffn_norm' })}>RMSNorm (FFN)</a></li>
                          <li><a href={href({ lv: 3, b, mod: p.kind === 'dense' ? 'ffn' : 'moe' })}>{p.kind === 'dense' ? 'SwiGLU FFN' : 'MoE'}</a></li>
                          <li><a href={href({ lv: 3, b, mod: 'residual2' })}>{T('Residual add 2 (no weights)', 'Reziduálne sčítanie 2 (bez váh)')}</a></li>
                          <li><span className="muted">{T('Tensors:', 'Tenzory:')}</span>
                            <ul>
                              {blockFams.filter((f) => f.scope === 'block').map((f) => (
                                <li key={f.key}><a href={href({ lv: 2, b, t: f.key })} aria-current={cur({ b, t: f.key })}><code>{tensorName(f, b)}</code></a> <span className="faint mono" style={{ fontSize: '0.78rem' }}>{shapeStr(f.shape)}</span></li>
                              ))}
                              {p.moe && (
                                <li>
                                  <label className="field inline" style={{ display: 'inline-flex' }}>{T('Expert', 'Expert')}
                                    <input type="number" min={0} max={p.moe.numExperts - 1} value={expert} style={{ width: 80 }} onChange={(e) => setExpert(Math.max(0, Math.min(p.moe!.numExperts - 1, e.target.valueAsNumber || 0)))} />
                                  </label>{' '}
                                  <span className="faint">{T('of', 'z')} {p.moe.numExperts}:</span>
                                  <ul>
                                    {blockFams.filter((f) => f.scope === 'expert').map((f) => (
                                      <li key={f.key}><a href={href({ lv: 2, b, t: f.key, e: expert })} aria-current={cur({ b, t: f.key, e: expert })}><code>{tensorName(f, b, expert)}</code></a> <span className="faint mono" style={{ fontSize: '0.78rem' }}>{shapeStr(f.shape)}</span></li>
                                    ))}
                                  </ul>
                                </li>
                              )}
                            </ul>
                          </li>
                        </ul>
                      )}
                    </li>
                  ))}
                </ul>
              </li>
              <li><a href={href({ lv: 2, t: 'norm' })}>{T('Final RMSNorm', 'Záverečná RMSNorm')} γ [{fmtInt(p.d)}]</a></li>
              <li><a href={href({ lv: 2, t: 'lm_head' })}>{T('Output head', 'Výstupná hlava')} {shapeStr([p.vocabRows, p.d])}</a></li>
              <li><a href={href({ lv: 3, mod: 'sampling' })}>{T('Sampling (no weights)', 'Vzorkovanie (bez váh)')}</a></li>
              <li><a href={href({ lv: 6 })}>{T('Execution & learning', 'Beh a učenie')}</a></li>
            </ul>
          </li>
        </ul>
      </div>
    </div>
  );
}
