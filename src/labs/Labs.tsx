import { WeightLab, UnitLab, MatrixLab } from './WeightUnitMatrix';
import { AttentionLab, RopeLab } from './AttentionRope';
import { CompareLab, RouterLab } from './MoeLabs';
import { LayersLab, CacheLab } from './LayersCache';
import { SamplingLab, TrainingLab } from './SamplingTraining';
import { MemoryLab } from './MemoryLab';
import { EvidenceLegend } from '../components/ui';
import { EnglishOnlyNotice, useLang, type L10n } from '../i18n/i18n';

const b = (en: string, sk: string): L10n => ({ en, sk });

export const LABS = [
  { id: 'weight', n: 1, title: b('Weight microscope', 'Mikroskop váhy'), blurb: b('Edit one stored number and watch exactly what changes downstream.', 'Upravte jedno uložené číslo a sledujte, čo presne sa zmení ďalej.'), learn: 'neuron', C: WeightLab },
  { id: 'unit', n: 2, title: b('Unit microscope', 'Mikroskop jednotky'), blurb: b('Every product and sum in one SwiGLU gated channel.', 'Každý súčin a súčet v jednom hradlovanom kanáli SwiGLU.'), learn: 'ffn', C: UnitLab },
  { id: 'matrix', n: 3, title: b('Matrix explorer', 'Prieskumník matíc'), blurb: b('Shapes, dtypes, counts, checkpoint mapping and real sample tiles.', 'Tvary, typy čísel, počty, umiestnenie v checkpointe a skutočné vzorky.'), learn: 'matrix', C: MatrixLab },
  { id: 'attention', n: 4, title: b('Attention lab', 'Laboratórium pozornosti'), blurb: b('Q·K, scaling, causal mask, stable softmax, weighted V, GQA.', 'Q·K, škálovanie, kauzálna maska, stabilný softmax, vážené V, GQA.'), learn: 'attention', C: AttentionLab },
  { id: 'rope', n: 5, title: b('RoPE lab', 'Laboratórium RoPE'), blurb: b('Position-dependent rotations and why only offsets matter.', 'Otáčania závislé od pozície a prečo záleží len na vzdialenosti.'), learn: 'attention', C: RopeLab },
  { id: 'compare', n: 6, title: b('Dense vs MoE', 'Hustý model vs MoE'), blurb: b('The real profiles side by side plus a dimension-matched toy experiment.', 'Skutočné profily vedľa seba a hračkový pokus s rovnakými rozmermi.'), learn: 'moe', C: CompareLab },
  { id: 'router', n: 7, title: b('Router lab', 'Laboratórium smerovača'), blurb: b('Logits → probabilities → top-k → renormalization → mixture.', 'Logity → pravdepodobnosti → top-k → prepočet → zmes.'), learn: 'moe', C: RouterLab },
  { id: 'layers', n: 8, title: b('Layer traversal', 'Prechod vrstvami'), blurb: b('Walk a token through every block and residual add.', 'Preveďte token každým blokom a reziduálnym sčítaním.'), learn: 'block', C: LayersLab },
  { id: 'cache', n: 9, title: b('Prefill / decode / cache', 'Prefill / decode / cache'), blurb: b('Cache growth, absolute positions, and a recomputation check.', 'Rast cache, absolútne pozície a kontrola úplným prepočtom.'), learn: 'generate', C: CacheLab },
  { id: 'sampling', n: 10, title: b('Sampling lab', 'Laboratórium vzorkovania'), blurb: b('Temperature, top-k, top-p, seeded draws and the token loop.', 'Teplota, top-k, top-p, náhodný výber s osivom a slučka tokenov.'), learn: 'generate', C: SamplingLab },
  { id: 'training', n: 11, title: b('Training lab', 'Laboratórium trénovania'), blurb: b('Targets, loss, gradients, an SGD step, and the actual new loss.', 'Ciele, chyba, gradienty, krok SGD a skutočná nová chyba.'), learn: 'learning', C: TrainingLab },
  { id: 'memory', n: 12, title: b('Memory lab', 'Laboratórium pamäte'), blurb: b('Weights, quantization, K/V cache and training memory, with formulas.', 'Váhy, kvantizácia, K/V cache a pamäť na trénovanie, so vzorcami.'), learn: 'hardware', C: MemoryLab },
] as const;

export default function Labs({ lab }: { lab?: string }) {
  const { t, tr, lang } = useLang();
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
        {lang === 'sk' && (
          <div className="callout" lang="sk" role="note" style={{ fontSize: '0.9rem' }}>
            <strong>{tr(L.title)}:</strong> {tr(L.blurb)} {t('english_only')} <a href={`#/learn/${L.learn}`}>{t('english_only_link')}</a>
          </div>
        )}
        <L.C />
      </div>
    );
  }
  return (
    <div className="page">
      <h1>{t('labs_title')}</h1>
      <p className="muted">{t('labs_intro')}</p>
      <EnglishOnlyNotice />
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
