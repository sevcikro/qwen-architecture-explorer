import { useMemo, useState } from 'react';
import type { CoreProfile } from '../models/profiles';
import { tensorFamilies, tensorName } from '../models/shapes';
import { explorerHref, type ExplorerState } from '../state/route';
import { fmtInt, shapeStr } from '../components/ui';
import { locateTensor } from './locate';

/** Structured, searchable alternative to the canvas: reaches every block, module, tensor, expert and coordinate. */
export function Outline({ p, s, onClose }: { p: CoreProfile; s: ExplorerState; onClose: () => void }) {
  const [q, setQ] = useState('');
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
    if (exact) out.push({ label: `Exact tensor: ${t}`, href: href({ lv: 2, t: exact.key, b: exact.b, e: exact.e }) });
    const low = t.toLowerCase();
    for (const f of fams) if (f.label.toLowerCase().includes(low) || f.pathTemplate.toLowerCase().includes(low) || f.key.includes(low)) out.push({ label: `${f.label} — ${f.pathTemplate}`, href: href({ lv: 2, t: f.key, b: f.scope === 'model' ? undefined : s.b ?? 0, e: f.scope === 'expert' ? 0 : undefined }) });
    const m = low.match(/block\s*(\d+)|layers\.(\d+)/);
    if (m) {
      const b = Number(m[1] ?? m[2]);
      if (b < p.layers) out.push({ label: `Decoder block ${b}`, href: href({ lv: 4, b }) });
    }
    return out.slice(0, 50);
  }, [q, p, fams, s.b]); // eslint-disable-line react-hooks/exhaustive-deps
  const cur = (x: Partial<ExplorerState>) => (x.b === s.b && x.t === s.t && (x.e === undefined || x.e === s.e) ? 'true' : undefined);
  return (
    <div className="outline-panel" role="region" aria-label="Model outline">
      <div className="row" style={{ justifyContent: 'space-between' }}>
        <h2 style={{ margin: 0 }}>Outline · {p.name}</h2>
        <button onClick={onClose}>Back to diagram</button>
      </div>
      <label className="field" style={{ margin: '10px 0', maxWidth: 520 }}>
        Search tensors, modules or blocks (e.g. “o_proj”, “block 12”, “layers.3.mlp.experts.17.up_proj.weight”)
        <input type="search" value={q} onChange={(e) => setQ(e.target.value)} placeholder="search…" />
      </label>
      {q && (
        <ul aria-label="Search results" style={{ paddingLeft: 18 }}>
          {results.length === 0 ? <li className="muted">No match in {p.name}.</li> : results.map((r) => <li key={r.href}><a href={r.href}>{r.label}</a></li>)}
        </ul>
      )}
      <div className="tree">
        <ul>
          <li><a href={href({ lv: 5 })}>{p.name} (full model)</a>
            <ul>
              <li><a href={href({ lv: 2, t: 'embed_tokens' })} aria-current={cur({ t: 'embed_tokens' })}>Embedding table {shapeStr([p.vocabRows, p.d])}</a></li>
              <li>
                <span className="muted">Decoder blocks ({p.layers}) — expand one:</span>
                <ul>
                  {Array.from({ length: p.layers }, (_, b) => (
                    <li key={b}>
                      <button aria-expanded={openBlock === b} onClick={() => setOpenBlock(openBlock === b ? null : b)}>{openBlock === b ? '▾' : '▸'} Block {b}</button>{' '}
                      <a href={href({ lv: 4, b })} style={{ fontSize: '0.85rem' }}>open</a>
                      {openBlock === b && (
                        <ul>
                          <li><a href={href({ lv: 3, b, mod: 'attn_norm' })}>RMSNorm (attention)</a></li>
                          <li><a href={href({ lv: 3, b, mod: 'attention' })}>Attention</a></li>
                          <li><a href={href({ lv: 3, b, mod: 'residual1' })}>Residual add 1 (no weights)</a></li>
                          <li><a href={href({ lv: 3, b, mod: 'ffn_norm' })}>RMSNorm (FFN)</a></li>
                          <li><a href={href({ lv: 3, b, mod: p.kind === 'dense' ? 'ffn' : 'moe' })}>{p.kind === 'dense' ? 'SwiGLU FFN' : 'MoE'}</a></li>
                          <li><a href={href({ lv: 3, b, mod: 'residual2' })}>Residual add 2 (no weights)</a></li>
                          <li><span className="muted">Tensors:</span>
                            <ul>
                              {blockFams.filter((f) => f.scope === 'block').map((f) => (
                                <li key={f.key}><a href={href({ lv: 2, b, t: f.key })} aria-current={cur({ b, t: f.key })}><code>{tensorName(f, b)}</code></a> <span className="faint mono" style={{ fontSize: '0.78rem' }}>{shapeStr(f.shape)}</span></li>
                              ))}
                              {p.moe && (
                                <li>
                                  <label className="field inline" style={{ display: 'inline-flex' }}>Expert
                                    <input type="number" min={0} max={p.moe.numExperts - 1} value={expert} style={{ width: 80 }} onChange={(e) => setExpert(Math.max(0, Math.min(p.moe!.numExperts - 1, e.target.valueAsNumber || 0)))} />
                                  </label>{' '}
                                  <span className="faint">of {p.moe.numExperts}:</span>
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
              <li><a href={href({ lv: 2, t: 'norm' })}>Final RMSNorm γ [{fmtInt(p.d)}]</a></li>
              <li><a href={href({ lv: 2, t: 'lm_head' })}>Output head {shapeStr([p.vocabRows, p.d])}</a></li>
              <li><a href={href({ lv: 3, mod: 'sampling' })}>Sampling (no weights)</a></li>
              <li><a href={href({ lv: 6 })}>Execution & learning</a></li>
            </ul>
          </li>
        </ul>
      </div>
    </div>
  );
}
