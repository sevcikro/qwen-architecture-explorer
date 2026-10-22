import { WeightLab, UnitLab, MatrixLab } from './WeightUnitMatrix';
import { AttentionLab, RopeLab } from './AttentionRope';
import { CompareLab, RouterLab } from './MoeLabs';
import { LayersLab, CacheLab } from './LayersCache';
import { SamplingLab, TrainingLab } from './SamplingTraining';
import { MemoryLab } from './MemoryLab';
import { EvidenceLegend } from '../components/ui';

export const LABS = [
  { id: 'weight', n: 1, title: 'Weight microscope', blurb: 'Edit one stored number and watch exactly what changes downstream.', C: WeightLab },
  { id: 'unit', n: 2, title: 'Unit microscope', blurb: 'Every product and sum in one SwiGLU gated channel.', C: UnitLab },
  { id: 'matrix', n: 3, title: 'Matrix explorer', blurb: 'Shapes, dtypes, counts, checkpoint mapping and real sample tiles.', C: MatrixLab },
  { id: 'attention', n: 4, title: 'Attention lab', blurb: 'Q·K, scaling, causal mask, stable softmax, weighted V, GQA.', C: AttentionLab },
  { id: 'rope', n: 5, title: 'RoPE lab', blurb: 'Position-dependent rotations and why only offsets matter.', C: RopeLab },
  { id: 'compare', n: 6, title: 'Dense vs MoE', blurb: 'The real profiles side by side plus a dimension-matched toy experiment.', C: CompareLab },
  { id: 'router', n: 7, title: 'Router lab', blurb: 'Logits → probabilities → top-k → renormalization → mixture.', C: RouterLab },
  { id: 'layers', n: 8, title: 'Layer traversal', blurb: 'Walk a token through every block and residual add.', C: LayersLab },
  { id: 'cache', n: 9, title: 'Prefill / decode / cache', blurb: 'Cache growth, absolute positions, and a recomputation check.', C: CacheLab },
  { id: 'sampling', n: 10, title: 'Sampling lab', blurb: 'Temperature, top-k, top-p, seeded draws and the token loop.', C: SamplingLab },
  { id: 'training', n: 11, title: 'Training lab', blurb: 'Targets, loss, gradients, an SGD step, and the actual new loss.', C: TrainingLab },
  { id: 'memory', n: 12, title: 'Memory lab', blurb: 'Weights, quantization, K/V cache and training memory, with formulas.', C: MemoryLab },
] as const;

export default function Labs({ lab }: { lab?: string }) {
  const L = LABS.find((x) => x.id === lab);
  if (lab && !L) return <div className="page"><h1>Unknown lab</h1><p>There is no lab called “{lab}”.</p><a href="#/labs">All labs</a></div>;
  if (L) {
    const i = LABS.indexOf(L);
    return (
      <div className="page wide">
        <nav className="row" aria-label="Lab navigation" style={{ justifyContent: 'space-between', marginBottom: 8 }}>
          <a href="#/labs">← All labs</a>
          <span className="row">
            {i > 0 && <a href={`#/labs/${LABS[i - 1].id}`}>‹ {LABS[i - 1].title}</a>}
            {i < LABS.length - 1 && <a href={`#/labs/${LABS[i + 1].id}`}>{LABS[i + 1].title} ›</a>}
          </span>
        </nav>
        <L.C />
      </div>
    );
  }
  return (
    <div className="page">
      <h1>Laboratories</h1>
      <p className="muted">Twelve connected labs. They share one seeded toy simulator, so a weight you edit in one lab is the same weight every other lab computes with (and the change is always shown). Real-model views show verified structure and only actual sampled values.</p>
      <EvidenceLegend />
      <div className="grid-auto" style={{ marginTop: 14 }}>
        {LABS.map((x) => (
          <a key={x.id} href={`#/labs/${x.id}`} className="card" style={{ textDecoration: 'none', color: 'inherit' }}>
            <div className="faint mono">Lab {x.n}</div>
            <h3 style={{ margin: '2px 0 4px', color: 'var(--accent)' }}>{x.title}</h3>
            <div className="muted" style={{ fontSize: '0.9rem' }}>{x.blurb}</div>
          </a>
        ))}
      </div>
    </div>
  );
}
