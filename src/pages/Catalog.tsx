import { useMemo, useRef, useState } from 'react';
import { CORE_PROFILES, type CoreProfile } from '../models/profiles';
import { prod, tensorFamilies, tensorName, type TensorFamily } from '../models/shapes';
import { Evidence, Section, Select, fmtInt, shapeStr } from '../components/ui';
import { fileOffset, useCatalog, useOffsets } from '../data/data';
import { explorerHref, type Route } from '../state/route';
import { QWEN3_8_FLASH_NEXT } from '../models/hybrid';

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
  return (
    <div className="page wide">
      <h1>Tensor catalog</h1>
      <p className="muted">Every learned tensor of the two reference checkpoints, expanded over all block and expert indices (virtualized: only the visible rows are rendered). Names, shapes and dtypes come from the pinned safetensors headers; file byte offsets are computed from them.</p>
      <div className="row">
        <Select label="Model" value={p.id} onChange={(id) => { setP(CORE_PROFILES.find((x) => x.id === id)!); setFam('all'); setLayer(null); setTop(0); box.current?.scrollTo(0, 0); }} options={CORE_PROFILES.map((x) => ({ value: x.id, label: x.name }))} />
        <Select label="Tensor family" value={fam} onChange={(v) => { setFam(v); setTop(0); box.current?.scrollTo(0, 0); }} options={[{ value: 'all', label: 'all families' }, ...tensorFamilies(p).map((f) => ({ value: f.key, label: f.label }))]} />
        <Select label="Block" value={layer ?? -1} onChange={(v) => { setLayer(v < 0 ? null : v); setTop(0); box.current?.scrollTo(0, 0); }} options={[{ value: -1, label: 'all blocks' }, ...Array.from({ length: p.layers }, (_, i) => ({ value: i, label: `block ${i}` }))]} />
        <div className="badge-row"><Evidence kind="verified" /><Evidence kind="derived" title="Byte offsets" /></div>
      </div>
      <p className="faint" style={{ fontSize: '0.86rem' }}>{fmtInt(en.total)} tensors shown · {fmtInt(totalParams)} parameters in the full model · {cat.status === 'ok' ? `${cat.data.tensorCount} tensors in ${cat.data.shards.length} shards (checkpoint headers @ ${cat.data.revision.slice(0, 12)}, read ${cat.data.retrieved})` : cat.status === 'error' ? `catalog metadata unavailable (${cat.error}); logical shapes still shown` : 'loading checkpoint metadata…'}</p>
      <div className="card" style={{ padding: 0 }}>
        <div className="table-wrap">
          <div style={{ minWidth: 820 }}>
            <div className="row" style={{ display: 'grid', gridTemplateColumns: 'minmax(320px, 2.4fr) 1fr 0.9fr 0.6fr 1.6fr 0.5fr', gap: 8, padding: '8px 12px', borderBottom: '1px solid var(--line)', fontWeight: 600, color: 'var(--text-2)', fontSize: '0.85rem' }}>
              <span>checkpoint tensor</span><span>logical shape</span><span className="num">parameters</span><span>dtype</span><span>shard · file byte offset</span><span />
            </div>
            <div ref={box} style={{ height: H, overflowY: 'auto', position: 'relative' }} onScroll={(e) => setTop((e.target as HTMLDivElement).scrollTop)} role="table" aria-label="Tensor catalog (virtualized)" aria-rowcount={en.total}>
              <div style={{ height: en.total * ROW_H, position: 'relative' }}>
                {Array.from({ length: Math.max(0, count) }, (_, k) => {
                  const i = start + k;
                  const r = en.at(i);
                  const fo = off.status === 'ok' ? fileOffset(off.data, r.name, 0, r.f.shape.length > 1 ? 0 : null) : null;
                  return (
                    <div key={i} role="row" aria-rowindex={i + 1} style={{ position: 'absolute', top: i * ROW_H, left: 0, right: 0, height: ROW_H, display: 'grid', gridTemplateColumns: 'minmax(320px, 2.4fr) 1fr 0.9fr 0.6fr 1.6fr 0.5fr', gap: 8, alignItems: 'center', padding: '0 12px', borderBottom: '1px solid var(--line-soft)', fontSize: '0.84rem' }}>
                      <span className="mono" style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={r.name}>{r.name}</span>
                      <span className="mono">{shapeStr(r.f.shape)}</span>
                      <span className="num mono">{fmtInt(prod(r.f.shape))}</span>
                      <span>{fo?.dtype ?? 'BF16'}</span>
                      <span className="mono" style={{ fontSize: '0.78rem' }}>{fo ? `${fo.shard.replace('model-', '').replace('.safetensors', '')} · ${fmtInt(fo.byte)}` : off.status === 'loading' ? '…' : '—'}</span>
                      <a href={explorerHref({ m: p.id, lv: 2, t: r.f.key, b: r.layer, e: r.expert })}>open</a>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      </div>
      <p className="faint" style={{ fontSize: '0.82rem' }}>Rendering limit: about {Math.ceil(H / ROW_H) + 8} rows exist in the page at any time, regardless of the {fmtInt(en.total)} tensors listed.</p>
      <Section title="How the checkpoint files are organized" badges={<Evidence kind="verified" />}>
        <p>A Safetensors file is an 8-byte little-endian header length, a JSON header listing every tensor's dtype, shape and byte range, then the raw tensor bytes. The model is split into several shard files, and the index file maps each tensor name to the shard that holds it: in these checkpoints every tensor lies <strong>complete</strong> inside one shard (no tensor is split across files). A runtime reads the bytes of each tensor into an array in CPU or GPU memory; the resulting memory address has nothing to do with the file offset or the tensor coordinate.</p>
        {cat.status === 'ok' && (
          <div className="table-wrap"><table className="t">
            <thead><tr><th>shard</th><th className="num">header bytes</th><th className="num">tensors</th></tr></thead>
            <tbody>{cat.data.shards.map((s) => <tr key={s.file}><td className="mono">{s.file}</td><td className="num">{fmtInt(s.headerBytes)}</td><td className="num">{s.tensors}</td></tr>)}</tbody>
          </table></div>
        )}
        {p.kind === 'moe' && <p>Qwen3-30B-A3B's checkpoint stores each expert matrix as its own tensor ({fmtInt(48 * 128 * 3)} expert tensors). Current Transformers code packs them at load time into <code>mlp.experts.gate_up_proj [128, 1536, 2048]</code> (gate rows 0–767, up rows 768–1535) and <code>mlp.experts.down_proj [128, 2048, 768]</code> per block. That is a re-layout of the same scalars; parameter totals count them once.</p>}
      </Section>
      <Section title={`${QWEN3_8_FLASH_NEXT.name}: tensor families (structural view)`} badges={<Evidence kind="verified" />}>
        {q38.status === 'ok' ? (
          <div className="table-wrap" style={{ maxHeight: 380, overflowY: 'auto' }}><table className="t">
            <thead><tr><th>family</th><th className="num">tensors</th><th>shape(s)</th><th>dtype</th><th className="num">elements</th></tr></thead>
            <tbody>{q38.data.families.filter((f) => !/ngram_embedding\.shard_/.test(f.family) || f.family.endsWith('shard_0.weight')).map((f) => (
              <tr key={f.family}><td className="mono" style={{ fontSize: '0.78rem' }}>{f.family.replace('shard_0', 'shard_{0..127}')}</td><td className="num">{/shard_0/.test(f.family) ? 128 : f.count}</td><td className="mono" style={{ fontSize: '0.78rem' }}>{f.shapes.slice(0, 2).map((s) => shapeStr(s)).join(' ')}{f.shapes.length > 2 ? ' …' : ''}</td><td>{f.dtype.join(', ')}</td><td className="num">{/shard_0/.test(f.family) ? '≈ 51.2B (all 128)' : fmtInt(f.elements)}</td></tr>
            ))}</tbody>
          </table></div>
        ) : <p className="muted">{q38.status === 'error' ? `Unavailable: ${q38.error}` : 'Loading…'}</p>}
        <p className="faint" style={{ fontSize: '0.84rem' }}>No numerical values of this checkpoint are shipped (its license is qwen-community-1.0); only names, shapes and dtypes from the headers.</p>
      </Section>
    </div>
  );
}
