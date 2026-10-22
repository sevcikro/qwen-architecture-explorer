import { useState } from 'react';
import { SOURCES } from '../evidence/sources';
import { CLAIMS } from '../evidence/claims';
import { EVIDENCE_META, type EvidenceClass } from '../evidence/types';
import { Evidence, EvidenceLegend, ExportButton } from '../components/ui';
import { useLang } from '../i18n/i18n';
import { CLAIM_SK, SCOPE_SK, SOURCE_SK } from '../i18n/dataSk';

const EVIDENCE_SK: Record<EvidenceClass, { label: string; description: string }> = {
  verified: { label: 'Overená architektúra', description: 'Priamo podložené konkrétnym konfiguračným súborom, hlavičkou checkpointu alebo implementáciou v pripnutej revízii.' },
  derived: { label: 'Odvodený výpočet', description: 'Vypočítané z overených faktov. Vzorec a spôsob počítania sú uvedené pri výsledku.' },
  simulation: { label: 'Názorná simulácia', description: 'Reprodukovateľný výpočet so zmenšenými rozmermi a syntetickými váhami (z osiva). Platí pre tento simulátor, nie je to pozorovaný beh Qwen.' },
  sample: { label: 'Skutočná vzorka z checkpointu', description: 'Hodnoty naozaj prečítané z uvedeného súboru checkpointu v pripnutej revízii, s pôvodom až na úroveň bajtov.' },
  unknown: { label: 'Neznáme / neoverené', description: 'Skutočná medzera v informáciách. Aplikácia to povie namiesto toho, aby ju vyplnila vierohodne vyzerajúcimi údajmi.' },
};

const KIND_SK: Record<string, string> = { config: 'konfigurácia', 'model-card': 'karta modelu', checkpoint: 'checkpoint', implementation: 'implementácia', report: 'správa', paper: 'článok', docs: 'dokumentácia', tokenizer: 'tokenizér' };

export default function Sources() {
  const { lang } = useLang();
  const sk = lang === 'sk';
  const T = (en: string, s: string) => (sk ? s : en);
  const [q, setQ] = useState('');
  const low = q.toLowerCase();
  const claimText = (id: string, en: string) => (sk ? CLAIM_SK[id]?.claim ?? en : en);
  const noteText = (id: string, en?: string) => (sk ? CLAIM_SK[id]?.note ?? en : en);
  const claims = CLAIMS.filter((c) => !q || `${c.scope} ${c.claim} ${c.note ?? ''} ${claimText(c.id, '')} ${noteText(c.id) ?? ''}`.toLowerCase().includes(low));
  return (
    <div className="page wide">
      <h1>{T('Sources and evidence', 'Zdroje a dôkazy')}</h1>
      <p className="muted">{T('Every factual statement on this site is backed by one of the primary sources below, pinned to an immutable revision where the host provides one (retrieved 2026-10-22). Secondary articles were not used as evidence.', 'Každé faktické tvrdenie na tejto stránke je podložené niektorým z primárnych zdrojov nižšie, pripnutým na nemennú revíziu, ak ju hostiteľ poskytuje (načítané 2026-10-22). Sekundárne články sa ako dôkaz nepoužili.')}</p>
      <EvidenceLegend />
      <ul style={{ paddingLeft: 18, fontSize: '0.9rem' }}>
        {(Object.keys(EVIDENCE_META) as EvidenceClass[]).map((k) => <li key={k}><strong>{sk ? EVIDENCE_SK[k].label : EVIDENCE_META[k].label}</strong>: {sk ? EVIDENCE_SK[k].description : EVIDENCE_META[k].description}</li>)}
      </ul>
      <h2>{T('Claim registry', 'Register tvrdení')}</h2>
      <label className="field" style={{ maxWidth: 420 }}>{T('Filter claims', 'Filtrovať tvrdenia')}<input type="search" value={q} onChange={(e) => setQ(e.target.value)} /></label>
      <div className="table-wrap" tabIndex={0} style={{ marginTop: 8 }}>
        <table className="t">
          <thead><tr><th>{T('Scope', 'Rozsah')}</th><th>{T('Claim', 'Tvrdenie')}</th><th>{T('Evidence', 'Dôkaz')}</th><th>{T('Sources', 'Zdroje')}</th></tr></thead>
          <tbody>{claims.map((c) => {
            const note = noteText(c.id, c.note);
            return (
              <tr key={c.id}>
                <td style={{ whiteSpace: 'nowrap' }}>{sk ? SCOPE_SK[c.scope] ?? c.scope : c.scope}</td>
                <td>{claimText(c.id, c.claim)}{note && <div className="faint" style={{ fontSize: '0.84rem', marginTop: 4 }}>{note}</div>}</td>
                <td><Evidence kind={c.evidence} /></td>
                <td style={{ fontSize: '0.82rem' }}>{c.sources.map((s) => <div key={s}><a href={`#src-${s}`} onClick={(e) => { e.preventDefault(); document.getElementById(`src-${s}`)?.scrollIntoView({ block: 'center' }); }}>{s}</a></div>)}</td>
              </tr>
            );
          })}</tbody>
        </table>
      </div>
      <h2 style={{ marginTop: 20 }}>{T('Source registry', 'Register zdrojov')}</h2>
      <ExportButton filename="qwen-atlas-sources.json" label={T('Export registry', 'Exportovať register')} data={() => ({ sources: SOURCES, claims: CLAIMS })} />
      <div className="table-wrap" tabIndex={0} style={{ marginTop: 8 }}>
        <table className="t">
          <thead><tr><th>Id</th><th>{T('Source', 'Zdroj')}</th><th>{T('Kind', 'Druh')}</th><th>{T('Revision', 'Revízia')}</th><th>{T('Supports', 'Podporuje')}</th></tr></thead>
          <tbody>{SOURCES.map((s) => (
            <tr key={s.id} id={`src-${s.id}`}>
              <td className="mono" style={{ fontSize: '0.8rem' }}>{s.id}</td>
              <td><a href={s.url} target="_blank" rel="noreferrer">{s.title}</a>{s.locator && <div className="faint" style={{ fontSize: '0.8rem' }}>{s.locator}</div>}</td>
              <td>{sk ? KIND_SK[s.kind] ?? s.kind : s.kind}</td>
              <td className="mono" style={{ fontSize: '0.75rem', overflowWrap: 'anywhere' }}>{s.revision ?? '—'}</td>
              <td style={{ fontSize: '0.85rem' }}>{sk ? SOURCE_SK[s.id] ?? s.supports : s.supports}</td>
            </tr>
          ))}</tbody>
        </table>
      </div>
      <h2 style={{ marginTop: 20 }}>{T('Resolved discrepancies', 'Vyriešené nezrovnalosti')}</h2>
      {sk ? (
        <ol style={{ paddingLeft: 20 }}>
          <li><strong>Dĺžka kontextu.</strong> Tabuľka v správe: 128K. Karty modelov: 32 768 natívne, 131 072 s YaRN. Konfigurácia: max_position_embeddings 40 960 (rozpočet na generovanie). Zobrazené ako tri rôzne veličiny.</li>
          <li><strong>30B-A3B intermediate_size 6144.</strong> V konfigurácii je, ale nepoužíva sa: každý blok je MoE s moe_intermediate_size 768.</li>
          <li><strong>Slovník 151 669 vs 151 936.</strong> ID tokenov definované tokenizérom vs riadky matíc embeddingov / hlavy (výplň).</li>
          <li><strong>Parametre „bez embeddingov“.</strong> 6,95 mld. / 29,9 mld. z kariet zodpovedajú súčtu mínus obe matice slovníka (embedding aj hlava), nie mínus samotný embedding.</li>
          <li><strong>Rozloženie tenzorov expertov.</strong> V checkpointe samostatné tenzory pre každého experta; v súčasných Transformers zbalené gate_up_proj / down_proj. Tie isté čísla, počítané raz.</li>
          <li><strong>Qwen3.8 „vrstva 2“.</strong> V konfigurácii a na karte sa počíta od 1; v kóde index bloku 1.</li>
          <li><strong>Qwen3.8 „full_attention“.</strong> Implementované ako QSA (bloky vybrané indexerom), nie neobmedzená pozornosť.</li>
          <li><strong>Qwen3.8 „125B“.</strong> Rovná sa len textovej kostre (125,71 mld. z hlavičiek); n-gramy (51,2 mld.), MTP (2,61 mld. uložených) a obraz (0,45 mld.) sú samostatné. Spôsob počítania „4B MTP“ na karte nie je uvedený (označené ako Neznáme).</li>
        </ol>
      ) : (
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
      )}
    </div>
  );
}
