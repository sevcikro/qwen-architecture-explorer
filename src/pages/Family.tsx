import { FAMILY, HYBRIDS, COVERAGE_LABEL } from '../models/hybrid';
import { Cite, Evidence, Section } from '../components/ui';
import { useLang } from '../i18n/i18n';
import { COVERAGE_LABEL_SK, FAMILY_SK, HYBRID_SK } from '../i18n/dataSk';

export default function Family() {
  const { lang } = useLang();
  const sk = lang === 'sk';
  const T = (en: string, s: string) => (sk ? s : en);
  return (
    <div className="page">
      <h1>{T('The Qwen family and newer hybrid architectures', 'Rodina Qwen a novšie hybridné architektúry')}</h1>
      <p className="muted">{T('Dates are the creation dates of the named Hugging Face repositories (Hub API), used as a consistent, checkable proxy for release order. Different model versions are kept separate: properties of the hybrid lines are never imported into the Qwen3 reference presets.', 'Dátumy sú dátumy vytvorenia uvedených repozitárov na Hugging Face (Hub API), použité ako jednotný a overiteľný ukazovateľ poradia vydaní. Rôzne verzie modelov sa držia oddelene: vlastnosti hybridných línií sa nikdy neprenášajú do referenčných predvolieb Qwen3.')}<Cite ids={['hf-api']} /></p>
      <Section title={T('Timeline', 'Časová os')} badges={<Evidence kind="verified" />}>
        <div className="table-wrap" tabIndex={0}>
          <svg viewBox="0 0 900 150" style={{ width: '100%', minWidth: 640, height: 'auto' }} role="img" aria-label={T('Timeline from Qwen 2023 to Qwen3.8-Flash-Next 2026, branching into ordinary decoders and hybrid models', 'Časová os od Qwen 2023 po Qwen3.8-Flash-Next 2026, s vetvením na bežné dekodéry a hybridné modely')}>
            <line x1={20} y1={60} x2={880} y2={60} stroke="var(--line)" strokeWidth={2} />
            {FAMILY.map((f, i) => {
              const x = 40 + i * 102;
              const hybrid = i >= 5;
              return (
                <g key={f.name}>
                  <circle cx={x} cy={60} r={hybrid ? 9 : 7} fill={i === 4 ? 'var(--accent)' : hybrid ? 'var(--state)' : 'var(--panel-2)'} stroke={i === 4 ? 'var(--focus)' : 'var(--text-3)'} strokeWidth={2} />
                  <text x={x} y={i % 2 ? 98 : 34} textAnchor="middle" fontSize={12} fill="var(--text)" fontWeight={600}>{f.name.replace(' (first generation)', '').replace(' (dense + MoE)', '')}</text>
                  <text x={x} y={i % 2 ? 114 : 18} textAnchor="middle" fontSize={10} fill="var(--text-3)">{f.date}</text>
                </g>
              );
            })}
            <text x={460} y={140} fontSize={11} fill="var(--text-3)" textAnchor="middle">{T('filled circles at right = hybrid Gated DeltaNet lines · highlighted = Qwen3 reference presets of this atlas', 'plné kruhy vpravo = hybridné línie Gated DeltaNet · zvýraznené = referenčné predvoľby Qwen3 v tomto atlase')}</text>
          </svg>
        </div>
        <div className="table-wrap" tabIndex={0}><table className="t">
          <thead><tr><th>{T('Generation', 'Generácia')}</th><th>{T('Reference repo', 'Referenčný repozitár')}</th><th>{T('Architecture (as documented)', 'Architektúra (podľa dokumentácie)')}</th><th>{T('Coverage in this atlas', 'Pokrytie v tomto atlase')}</th></tr></thead>
          <tbody>{FAMILY.map((f) => {
            const s = sk ? FAMILY_SK[f.name] : undefined;
            return <tr key={f.name}><td><strong>{f.name}</strong><br /><span className="faint">{f.date}</span></td><td className="mono" style={{ fontSize: '0.8rem' }}>{f.repo}</td><td>{s?.architecture ?? f.architecture}<br /><span className="faint">{s?.relation ?? f.relation}</span></td><td>{s?.inAtlas ?? f.inAtlas}</td></tr>;
          })}</tbody>
        </table></div>
        <p className="faint" style={{ fontSize: '0.84rem' }}>{T('Earlier generations are listed for orientation only; their internals are not analyzed here. A “Qwen 4” has been described by the Qwen3.8-Flash-Next model card only as the future generation this architecture previews; no Qwen4 specification is claimed.', 'Skoršie generácie sú uvedené len na orientáciu; ich vnútro sa tu neanalyzuje. „Qwen 4“ opisuje karta modelu Qwen3.8-Flash-Next len ako budúcu generáciu, ktorú táto architektúra predznamenáva; žiadna špecifikácia Qwen4 sa netvrdí.')}</p>
      </Section>
      <Section title={T('Hybrid profiles', 'Hybridné profily')}>
        <div className="grid-auto">
          {HYBRIDS.map((h) => (
            <a key={h.id} href={`#/hybrid/${h.id}`} className="card" style={{ textDecoration: 'none', color: 'inherit' }}>
              <h3 style={{ color: 'var(--accent)', marginTop: 0 }}>{h.name}</h3>
              <div className="coverage">{sk ? COVERAGE_LABEL_SK[h.coverage] : COVERAGE_LABEL[h.coverage]}</div>
              <p className="muted" style={{ fontSize: '0.88rem' }}>{h.layout}</p>
              <p className="faint" style={{ fontSize: '0.82rem' }}>{sk ? HYBRID_SK[h.id]?.advertised ?? h.advertised : h.advertised}</p>
            </a>
          ))}
        </div>
        {sk
          ? <p className="muted" style={{ marginTop: 10 }}>Čo zostáva rovnaké ako v Qwen3: iba dekodér, reziduálne bloky s pred-normalizáciou, smerovaní experti SwiGLU, kauzálne predpovedanie ďalšieho tokenu. Čo sa mení: tri zo štyroch vrstiev miešajúcich tokeny sú <strong>Gated DeltaNet</strong> (lineárna pozornosť s maticovým stavom, ktorý slabne) namiesto softmax pozornosti; ostatné vrstvy používajú pozornosť so sigmoidovo hradlovaným výstupom; MoE pridáva zdieľaného experta; a Qwen3.8-Flash-Next ďalej pridáva riedky výber blokov (QSA), štyri hradlované reziduálne vetvy a hašované n-gramové tabuľky.</p>
          : <p className="muted" style={{ marginTop: 10 }}>What stays the same as Qwen3: decoder-only, pre-normalized residual blocks, routed SwiGLU experts, causal next-token prediction. What changes: three of every four token-mixing layers are <strong>Gated DeltaNet</strong> (linear attention with a decaying matrix state) instead of softmax attention; the remaining layers use a sigmoid-gated attention output; MoE adds a shared expert; and Qwen3.8-Flash-Next further adds sparse block selection (QSA), four gated residual branches and hashed n-gram tables.</p>}
      </Section>
    </div>
  );
}
