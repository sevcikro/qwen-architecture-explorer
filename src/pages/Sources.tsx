import { useState } from 'react';
import { SOURCES } from '../evidence/sources';
import { CLAIMS } from '../evidence/claims';
import { EVIDENCE_META } from '../evidence/types';
import { Evidence, EvidenceLegend, ExportButton } from '../components/ui';
import { EnglishOnlyNotice } from '../i18n/i18n';

export default function Sources() {
  const [q, setQ] = useState('');
  const low = q.toLowerCase();
  const claims = CLAIMS.filter((c) => !q || `${c.scope} ${c.claim} ${c.note ?? ''}`.toLowerCase().includes(low));
  return (
    <div className="page wide">
      <h1>Sources and evidence</h1>
      <EnglishOnlyNotice />
      <p className="muted">Every factual statement on this site is backed by one of the primary sources below, pinned to an immutable revision where the host provides one (retrieved 2026-10-22). Secondary articles were not used as evidence.</p>
      <EvidenceLegend />
      <ul style={{ paddingLeft: 18, fontSize: '0.9rem' }}>
        {Object.entries(EVIDENCE_META).map(([k, m]) => <li key={k}><strong>{m.label}</strong>: {m.description}</li>)}
      </ul>
      <h2>Claim registry</h2>
      <label className="field" style={{ maxWidth: 420 }}>Filter claims<input type="search" value={q} onChange={(e) => setQ(e.target.value)} /></label>
      <div className="table-wrap" style={{ marginTop: 8 }}>
        <table className="t">
          <thead><tr><th>Scope</th><th>Claim</th><th>Evidence</th><th>Sources</th></tr></thead>
          <tbody>{claims.map((c) => (
            <tr key={c.id}>
              <td style={{ whiteSpace: 'nowrap' }}>{c.scope}</td>
              <td>{c.claim}{c.note && <div className="faint" style={{ fontSize: '0.84rem', marginTop: 4 }}>{c.note}</div>}</td>
              <td><Evidence kind={c.evidence} /></td>
              <td style={{ fontSize: '0.82rem' }}>{c.sources.map((s) => <div key={s}><a href={`#src-${s}`} onClick={(e) => { e.preventDefault(); document.getElementById(`src-${s}`)?.scrollIntoView({ block: 'center' }); }}>{s}</a></div>)}</td>
            </tr>
          ))}</tbody>
        </table>
      </div>
      <h2 style={{ marginTop: 20 }}>Source registry</h2>
      <ExportButton filename="qwen-atlas-sources.json" label="Export registry" data={() => ({ sources: SOURCES, claims: CLAIMS })} />
      <div className="table-wrap" style={{ marginTop: 8 }}>
        <table className="t">
          <thead><tr><th>Id</th><th>Source</th><th>Kind</th><th>Revision</th><th>Supports</th></tr></thead>
          <tbody>{SOURCES.map((s) => (
            <tr key={s.id} id={`src-${s.id}`}>
              <td className="mono" style={{ fontSize: '0.8rem' }}>{s.id}</td>
              <td><a href={s.url} target="_blank" rel="noreferrer">{s.title}</a>{s.locator && <div className="faint" style={{ fontSize: '0.8rem' }}>{s.locator}</div>}</td>
              <td>{s.kind}</td>
              <td className="mono" style={{ fontSize: '0.75rem', overflowWrap: 'anywhere' }}>{s.revision ?? '—'}</td>
              <td style={{ fontSize: '0.85rem' }}>{s.supports}</td>
            </tr>
          ))}</tbody>
        </table>
      </div>
      <h2 style={{ marginTop: 20 }}>Resolved discrepancies</h2>
      <ol style={{ paddingLeft: 20 }}>
        <li><strong>Context length.</strong> Report table: 128K. Model cards: 32,768 native, 131,072 with YaRN. Config: max_position_embeddings 40,960 (a generation budget). Shown as three distinct quantities.</li>
        <li><strong>30B-A3B intermediate_size 6144.</strong> Present in config, unused: every block is MoE with moe_intermediate_size 768.</li>
        <li><strong>Vocabulary 151,669 vs 151,936.</strong> Token ids defined by the tokenizer vs rows of the embedding / head matrices (padding).</li>
        <li><strong>“Non-embedding” parameters.</strong> The cards’ 6.95B / 29.9B match total minus both vocabulary matrices (embedding and head), not minus the embedding alone.</li>
        <li><strong>Expert tensor layout.</strong> Per-expert tensors in the checkpoint; packed gate_up_proj / down_proj in current Transformers. Same scalars, counted once.</li>
        <li><strong>Qwen3.8 “layer 2”.</strong> 1-based in config and card; code block index 1.</li>
        <li><strong>Qwen3.8 “full_attention”.</strong> Implemented as QSA (indexer-selected blocks), not unrestricted attention.</li>
        <li><strong>Qwen3.8 “125B”.</strong> Equals the text backbone only (125.71B from headers); n-gram (51.2B), MTP (2.61B stored) and vision (0.45B) are separate. The card’s “4B MTP” convention is not stated (marked Unknown).</li>
      </ol>
    </div>
  );
}
