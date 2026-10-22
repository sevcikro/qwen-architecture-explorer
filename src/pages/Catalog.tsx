import { useMemo, useRef, useState } from 'react';
import { CORE_PROFILES, type CoreProfile } from '../models/profiles';
import { prod, tensorFamilies, tensorName, type TensorFamily } from '../models/shapes';
import { Evidence, Section, Select } from '../components/ui';
import { fileOffset, useCatalog, useOffsets } from '../data/data';
import { explorerHref, type Route } from '../state/route';
import { QWEN3_8_FLASH_NEXT } from '../models/hybrid';
import { useLang } from '../i18n/i18n';

interface Row { f: TensorFamily; layer?: number; expert?: number; name: string }

/** Virtual enumeration of every learned tensor without materializing the list. */
function enumerator(p: CoreProfile, famFilter: string, layerFilter: number | null) {
  const fams = tensorFamilies(p).filter((f) => famFilter === 'all' || f.key === famFilter);
  const model = fams.filter((f) => f.scope === 'model');
  const block = fams.filter((f) => f.scope === 'block');
  const expert = fams.filter((f) => f.scope === 'expert');
  const E = p.moe?.numExperts ?? 0;
  const perLayer = block.length + expert.length * E;
  const layers = layerFilter === null ? Array.from({ length: p.layers }, (_, i) => i) : [layerFilter];
  const head = model.filter((f) => f.key === 'embed_tokens');
  const tail = model.filter((f) => f.key !== 'embed_tokens');
  const total = head.length + layers.length * perLayer + tail.length;
  const at = (i: number): Row => {
    if (i < head.length) return { f: head[i], name: tensorName(head[i]) };
    i -= head.length;
    if (i < layers.length * perLayer) {
      const L = layers[Math.floor(i / perLayer)];
      let j = i % perLayer;
      if (j < block.length) return { f: block[j], layer: L, name: tensorName(block[j], L) };
      j -= block.length;
      const e = Math.floor(j / expert.length), f = expert[j % expert.length];
      return { f, layer: L, expert: e, name: tensorName(f, L, e) };
    }
    i -= layers.length * perLayer;
    return { f: tail[i], name: tensorName(tail[i]) };
  };
  return { total, at };
}

const ROW_H = 34;

export default function Catalog({ route }: { route: Route }) {
  const [p, setP] = useState(CORE_PROFILES.find((x) => x.id === route.query.get('m')) ?? CORE_PROFILES[0]);
  const [fam, setFam] = useState('all');
  const [layer, setLayer] = useState<number | null>(null);
  const [top, setTop] = useState(0);
  const box = useRef<HTMLDivElement>(null);
  const cat = useCatalog(p.id);
  const off = useOffsets(p.id);
  const en = useMemo(() => enumerator(p, fam, layer), [p, fam, layer]);
  const H = 520;
  const start = Math.max(0, Math.floor(top / ROW_H) - 4);
  const count = Math.min(en.total - start, Math.ceil(H / ROW_H) + 8);
  const totalParams = tensorFamilies(p).reduce((s, f) => s + prod(f.shape) * f.multiplicity, 0);
  const q38 = useCatalog('qwen3.8-flash-next');
  const { lang, num: fmtInt, shape: shapeStr } = useLang();
  const T = (en: string, sk: string) => (lang === 'sk' ? sk : en);
  return (
    <div className="page wide">
      <h1>{T('Tensor catalog', 'Katalóg tenzorov')}</h1>
      <p className="muted">{T('Every learned tensor of the two reference checkpoints, expanded over all block and expert indices (virtualized: only the visible rows are rendered). Names, shapes and dtypes come from the pinned safetensors headers; file byte offsets are computed from them.', 'Každý naučený tenzor dvoch referenčných checkpointov, rozpísaný cez všetky indexy blokov a expertov (virtualizované: vykreslia sa len viditeľné riadky). Názvy, tvary a dátové typy pochádzajú z pripnutých hlavičiek safetensors; bajtové posuny v súboroch sú z nich vypočítané.')}</p>
      <div className="row">
        <Select label={T('Model', 'Model')} value={p.id} onChange={(id) => { setP(CORE_PROFILES.find((x) => x.id === id)!); setFam('all'); setLayer(null); setTop(0); box.current?.scrollTo(0, 0); }} options={CORE_PROFILES.map((x) => ({ value: x.id, label: x.name }))} />
        <Select label={T('Tensor family', 'Rodina tenzorov')} value={fam} onChange={(v) => { setFam(v); setTop(0); box.current?.scrollTo(0, 0); }} options={[{ value: 'all', label: T('all families', 'všetky rodiny') }, ...tensorFamilies(p).map((f) => ({ value: f.key, label: lang === 'sk' ? f.labelSk : f.label }))]} />
        <Select label={T('Block', 'Blok')} value={layer ?? -1} onChange={(v) => { setLayer(v < 0 ? null : v); setTop(0); box.current?.scrollTo(0, 0); }} options={[{ value: -1, label: T('all blocks', 'všetky bloky') }, ...Array.from({ length: p.layers }, (_, i) => ({ value: i, label: `${T('block', 'blok')} ${i}` }))]} />
        <div className="badge-row"><Evidence kind="verified" /><Evidence kind="derived" title={T('Byte offsets', 'Bajtové posuny')} /></div>
      </div>
      <p className="faint" style={{ fontSize: '0.86rem' }}>{T(`${fmtInt(en.total)} tensors shown · ${fmtInt(totalParams)} parameters in the full model · `, `${fmtInt(en.total)} zobrazených tenzorov · ${fmtInt(totalParams)} parametrov v celom modeli · `)}{cat.status === 'ok' ? T(`${cat.data.tensorCount} tensors in ${cat.data.shards.length} shards (checkpoint headers @ ${cat.data.revision.slice(0, 12)}, read ${cat.data.retrieved})`, `${fmtInt(cat.data.tensorCount)} tenzorov v ${cat.data.shards.length} častiach (hlavičky checkpointu @ ${cat.data.revision.slice(0, 12)}, načítané ${cat.data.retrieved})`) : cat.status === 'error' ? T(`catalog metadata unavailable (${cat.error}); logical shapes still shown`, `metadáta katalógu nedostupné (${cat.error}); logické tvary sa stále zobrazujú`) : T('loading checkpoint metadata…', 'načítavajú sa metadáta checkpointu…')}</p>
      <div className="card" style={{ padding: 0 }}>
        <div className="table-wrap" tabIndex={0}>
          <div style={{ minWidth: 820 }}>
            <div className="row" style={{ display: 'grid', gridTemplateColumns: 'minmax(320px, 2.4fr) 1fr 0.9fr 0.6fr 1.6fr 0.5fr', gap: 8, padding: '8px 12px', borderBottom: '1px solid var(--line)', fontWeight: 600, color: 'var(--text-2)', fontSize: '0.85rem' }}>
              <span>{T('checkpoint tensor', 'tenzor checkpointu')}</span><span>{T('logical shape', 'logický tvar')}</span><span className="num">{T('parameters', 'parametre')}</span><span>{T('dtype', 'dát. typ')}</span><span>{T('shard · file byte offset', 'časť · bajtový posun v súbore')}</span><span />
            </div>
            <div ref={box} style={{ height: H, overflowY: 'auto', position: 'relative' }} onScroll={(e) => setTop((e.target as HTMLDivElement).scrollTop)} role="table" aria-label={T('Tensor catalog (virtualized)', 'Katalóg tenzorov (virtualizovaný)')} aria-rowcount={en.total}>
              <div role="presentation" style={{ height: en.total * ROW_H, position: 'relative' }}>
                {Array.from({ length: Math.max(0, count) }, (_, k) => {
                  const i = start + k;
                  const r = en.at(i);
                  const fo = off.status === 'ok' ? fileOffset(off.data, r.name, 0, r.f.shape.length > 1 ? 0 : null) : null;
                  return (
                    <div key={i} role="row" aria-rowindex={i + 1} style={{ position: 'absolute', top: i * ROW_H, left: 0, right: 0, height: ROW_H, display: 'grid', gridTemplateColumns: 'minmax(320px, 2.4fr) 1fr 0.9fr 0.6fr 1.6fr 0.5fr', gap: 8, alignItems: 'center', padding: '0 12px', borderBottom: '1px solid var(--line-soft)', fontSize: '0.84rem' }}>
                      <span role="cell" className="mono" style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={r.name}>{r.name}</span>
                      <span role="cell" className="mono">{shapeStr(r.f.shape)}</span>
                      <span role="cell" className="num mono">{fmtInt(prod(r.f.shape))}</span>
                      <span role="cell">{fo?.dtype ?? 'BF16'}</span>
                      <span role="cell" className="mono" style={{ fontSize: '0.78rem' }}>{fo ? `${fo.shard.replace('model-', '').replace('.safetensors', '')} · ${fmtInt(fo.byte)}` : off.status === 'loading' ? '…' : '—'}</span>
                      <span role="cell"><a href={explorerHref({ m: p.id, lv: 2, t: r.f.key, b: r.layer, e: r.expert })}>{T('open', 'otvoriť')}</a></span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      </div>
      <p className="faint" style={{ fontSize: '0.82rem' }}>{T(`Rendering limit: about ${Math.ceil(H / ROW_H) + 8} rows exist in the page at any time, regardless of the ${fmtInt(en.total)} tensors listed.`, `Limit vykresľovania: na stránke je naraz len asi ${Math.ceil(H / ROW_H) + 8} riadkov bez ohľadu na ${fmtInt(en.total)} uvedených tenzorov.`)}</p>
      <Section title={T('How the checkpoint files are organized', 'Ako sú usporiadané súbory checkpointu')} badges={<Evidence kind="verified" />}>
        {lang === 'sk' ? <p>Súbor Safetensors tvorí 8-bajtová dĺžka hlavičky (little-endian), hlavička JSON so zoznamom dátového typu, tvaru a rozsahu bajtov každého tenzora a potom surové bajty tenzorov. Model je rozdelený do viacerých súborov (častí) a indexový súbor priraďuje každý názov tenzora časti, ktorá ho obsahuje: v týchto checkpointoch leží každý tenzor <strong>celý</strong> v jednej časti (žiadny tenzor nie je rozdelený medzi súbory). Prostredie behu načíta bajty každého tenzora do poľa v pamäti CPU alebo GPU; výsledná adresa v pamäti nemá nič spoločné s posunom v súbore ani so súradnicou v tenzore.</p> : <p>A Safetensors file is an 8-byte little-endian header length, a JSON header listing every tensor's dtype, shape and byte range, then the raw tensor bytes. The model is split into several shard files, and the index file maps each tensor name to the shard that holds it: in these checkpoints every tensor lies <strong>complete</strong> inside one shard (no tensor is split across files). A runtime reads the bytes of each tensor into an array in CPU or GPU memory; the resulting memory address has nothing to do with the file offset or the tensor coordinate.</p>}
        {cat.status === 'ok' && (
          <div className="table-wrap" tabIndex={0}><table className="t">
            <thead><tr><th>{T('shard', 'časť')}</th><th className="num">{T('header bytes', 'bajty hlavičky')}</th><th className="num">{T('tensors', 'tenzory')}</th></tr></thead>
            <tbody>{cat.data.shards.map((s) => <tr key={s.file}><td className="mono">{s.file}</td><td className="num">{fmtInt(s.headerBytes)}</td><td className="num">{s.tensors}</td></tr>)}</tbody>
          </table></div>
        )}
        {p.kind === 'moe' && lang === 'sk' && <p>Checkpoint Qwen3-30B-A3B ukladá každú maticu experta ako samostatný tenzor ({fmtInt(48 * 128 * 3)} tenzorov expertov). Súčasný kód Transformers ich pri načítaní zbalí do <code>mlp.experts.gate_up_proj [128, 1536, 2048]</code> (riadky gate 0–767, riadky up 768–1535) a <code>mlp.experts.down_proj [128, 2048, 768]</code> v každom bloku. Ide o preusporiadanie tých istých čísel; súčty parametrov ich počítajú raz.</p>}
        {p.kind === 'moe' && lang !== 'sk' && <p>Qwen3-30B-A3B's checkpoint stores each expert matrix as its own tensor ({fmtInt(48 * 128 * 3)} expert tensors). Current Transformers code packs them at load time into <code>mlp.experts.gate_up_proj [128, 1536, 2048]</code> (gate rows 0–767, up rows 768–1535) and <code>mlp.experts.down_proj [128, 2048, 768]</code> per block. That is a re-layout of the same scalars; parameter totals count them once.</p>}
      </Section>
      <Section title={T(`${QWEN3_8_FLASH_NEXT.name}: tensor families (structural view)`, `${QWEN3_8_FLASH_NEXT.name}: rodiny tenzorov (štrukturálny pohľad)`)} badges={<Evidence kind="verified" />}>
        {q38.status === 'ok' ? (
          <div className="table-wrap" tabIndex={0} style={{ maxHeight: 380, overflowY: 'auto' }}><table className="t">
            <thead><tr><th>{T('family', 'rodina')}</th><th className="num">{T('tensors', 'tenzory')}</th><th>{T('shape(s)', 'tvar(y)')}</th><th>{T('dtype', 'dát. typ')}</th><th className="num">{T('elements', 'prvky')}</th></tr></thead>
            <tbody>{q38.data.families.filter((f) => !/ngram_embedding\.shard_/.test(f.family) || f.family.endsWith('shard_0.weight')).map((f) => (
              <tr key={f.family}><td className="mono" style={{ fontSize: '0.78rem' }}>{f.family.replace('shard_0', 'shard_{0..127}')}</td><td className="num">{/shard_0/.test(f.family) ? 128 : f.count}</td><td className="mono" style={{ fontSize: '0.78rem' }}>{f.shapes.slice(0, 2).map((s) => shapeStr(s)).join(' ')}{f.shapes.length > 2 ? ' …' : ''}</td><td>{f.dtype.join(', ')}</td><td className="num">{/shard_0/.test(f.family) ? T('≈ 51.2B (all 128)', '≈ 51,2 mld. (všetkých 128)') : fmtInt(f.elements)}</td></tr>
            ))}</tbody>
          </table></div>
        ) : <p className="muted">{q38.status === 'error' ? `${T('Unavailable', 'Nedostupné')}: ${q38.error}` : T('Loading…', 'Načítava sa…')}</p>}
        <p className="faint" style={{ fontSize: '0.84rem' }}>{T('No numerical values of this checkpoint are shipped (its license is qwen-community-1.0); only names, shapes and dtypes from the headers.', 'Žiadne číselné hodnoty tohto checkpointu sa nedodávajú (jeho licencia je qwen-community-1.0); len názvy, tvary a dátové typy z hlavičiek.')}</p>
      </Section>
    </div>
  );
}
