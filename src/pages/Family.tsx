import { FAMILY, HYBRIDS, COVERAGE_LABEL } from '../models/hybrid';
import { Cite, Evidence, Section } from '../components/ui';
import { EnglishOnlyNotice } from '../i18n/i18n';

export default function Family() {
  return (
    <div className="page">
      <h1>The Qwen family and newer hybrid architectures</h1>
      <EnglishOnlyNotice />
      <p className="muted">Dates are the creation dates of the named Hugging Face repositories (Hub API), used as a consistent, checkable proxy for release order. Different model versions are kept separate: properties of the hybrid lines are never imported into the Qwen3 reference presets.<Cite ids={['hf-api']} /></p>
      <Section title="Timeline" badges={<Evidence kind="verified" />}>
        <svg viewBox="0 0 900 150" style={{ width: '100%', height: 'auto' }} role="img" aria-label="Timeline from Qwen 2023 to Qwen3.8-Flash-Next 2026, branching into ordinary decoders and hybrid models">
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
          <text x={460} y={140} fontSize={11} fill="var(--text-3)" textAnchor="middle">filled circles at right = hybrid Gated DeltaNet lines · highlighted = Qwen3 reference presets of this atlas</text>
        </svg>
        <div className="table-wrap"><table className="t">
          <thead><tr><th>Generation</th><th>Reference repo</th><th>Architecture (as documented)</th><th>Coverage in this atlas</th></tr></thead>
          <tbody>{FAMILY.map((f) => <tr key={f.name}><td><strong>{f.name}</strong><br /><span className="faint">{f.date}</span></td><td className="mono" style={{ fontSize: '0.8rem' }}>{f.repo}</td><td>{f.architecture}<br /><span className="faint">{f.relation}</span></td><td>{f.inAtlas}</td></tr>)}</tbody>
        </table></div>
        <p className="faint" style={{ fontSize: '0.84rem' }}>Earlier generations are listed for orientation only; their internals are not analyzed here. A “Qwen 4” has been described by the Qwen3.8-Flash-Next model card only as the future generation this architecture previews; no Qwen4 specification is claimed.</p>
      </Section>
      <Section title="Hybrid profiles">
        <div className="grid-auto">
          {HYBRIDS.map((h) => (
            <a key={h.id} href={`#/hybrid/${h.id}`} className="card" style={{ textDecoration: 'none', color: 'inherit' }}>
              <h3 style={{ color: 'var(--accent)', marginTop: 0 }}>{h.name}</h3>
              <div className="coverage">{COVERAGE_LABEL[h.coverage]}</div>
              <p className="muted" style={{ fontSize: '0.88rem' }}>{h.layout}</p>
              <p className="faint" style={{ fontSize: '0.82rem' }}>{h.advertised}</p>
            </a>
          ))}
        </div>
        <p className="muted" style={{ marginTop: 10 }}>What stays the same as Qwen3: decoder-only, pre-normalized residual blocks, routed SwiGLU experts, causal next-token prediction. What changes: three of every four token-mixing layers are <strong>Gated DeltaNet</strong> (linear attention with a decaying matrix state) instead of softmax attention; the remaining layers use a sigmoid-gated attention output; MoE adds a shared expert; and Qwen3.8-Flash-Next further adds sparse block selection (QSA), four gated residual branches and hashed n-gram tables.</p>
      </Section>
    </div>
  );
}
