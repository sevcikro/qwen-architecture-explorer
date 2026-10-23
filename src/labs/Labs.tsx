import { WeightLab, UnitLab, MatrixLab } from './WeightUnitMatrix';
import { AttentionLab, RopeLab } from './AttentionRope';
import { CompareLab, RouterLab } from './MoeLabs';
import { LayersLab, CacheLab } from './LayersCache';
import { SamplingLab, TrainingLab } from './SamplingTraining';
import { MemoryLab } from './MemoryLab';
import { EvidenceLegend } from '../components/ui';
import { useLang } from '../i18n/i18n';
import { LAB_META } from './meta';

const COMPONENTS = { weight: WeightLab, unit: UnitLab, matrix: MatrixLab, attention: AttentionLab, rope: RopeLab, compare: CompareLab, router: RouterLab, layers: LayersLab, cache: CacheLab, sampling: SamplingLab, training: TrainingLab, memory: MemoryLab } as const;
export const LABS = LAB_META.map((m) => ({ ...m, C: COMPONENTS[m.id] }));

export default function Labs({ lab }: { lab?: string }) {
  const { t, tr } = useLang();
  const L = LABS.find((x) => x.id === lab);
  if (lab && !L) return <div className="page"><h1>{t('labs_unknown')}</h1><p><code>{lab}</code></p><a href="#/labs">{t('labs_all')}</a></div>;
  if (L) {
    const i = LABS.indexOf(L);
    return (
      <div className="page wide">
        <nav className="row" aria-label="Lab navigation" style={{ justifyContent: 'space-between', marginBottom: 8 }}>
          <a href="#/labs">{t('labs_all')}</a>
          <span className="row">
            {i > 0 && <a href={`#/labs/${LABS[i - 1].id}`}>‹ {tr(LABS[i - 1].title)}</a>}
            {i < LABS.length - 1 && <a href={`#/labs/${LABS[i + 1].id}`}>{tr(LABS[i + 1].title)} ›</a>}
          </span>
        </nav>
        <div className="callout" role="note" style={{ fontSize: '0.9rem' }}>
          <strong>{tr(L.title)}:</strong> {tr(L.blurb)} <a href={`#/learn/${L.learn}`}>{t('lab_theory')}</a>
        </div>
        <L.C />
      </div>
    );
  }
  return (
    <div className="page">
      <h1>{t('labs_title')}</h1>
      <p className="muted">{t('labs_intro')}</p>
      <EvidenceLegend />
      <div className="grid-auto" style={{ marginTop: 14 }}>
        {LABS.map((x) => (
          <a key={x.id} href={`#/labs/${x.id}`} className="card" style={{ textDecoration: 'none', color: 'inherit' }}>
            <div className="faint mono">{t('labs_lab')} {x.n}</div>
            <h3 style={{ margin: '2px 0 4px', color: 'var(--accent)' }}>{tr(x.title)}</h3>
            <div className="muted" style={{ fontSize: '0.9rem' }}>{tr(x.blurb)}</div>
          </a>
        ))}
      </div>
    </div>
  );
}
