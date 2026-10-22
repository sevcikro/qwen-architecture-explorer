import { QWEN3_30B_A3B, QWEN3_8B } from '../models/profiles';
import { accounting } from '../models/shapes';
import { Evidence, EvidenceLegend, fmtB, fmtInt } from '../components/ui';
import { explorerHref } from '../state/route';
import { TinyExample } from '../labs/TinyExample';

const START_WEIGHT = explorerHref({ m: 'qwen3-8b', lv: 0, b: 0, t: 'q_proj', r: 0, c: 0 });

function StackDiagram() {
  const a = accounting(QWEN3_8B);
  const rows: { y: number; h: number; label: string; sub: string; href: string; kind: 'op' | 'param' | 'norm' | 'block' }[] = [
    { y: 8, h: 34, label: 'Text → chat template → token ids', sub: 'tokenizer (no learned weights)', href: '#/pipeline', kind: 'op' },
    { y: 52, h: 38, label: 'Embedding table [151,936 × 4,096]', sub: `${fmtB(a.embedding)} parameters · one row looked up per token`, href: explorerHref({ m: 'qwen3-8b', lv: 2, t: 'embed_tokens' }), kind: 'param' },
  ];
  const blocks = 36;
  return (
    <svg viewBox="0 0 460 470" role="img" aria-labelledby="hd-t hd-d" style={{ width: '100%', maxWidth: 520, height: 'auto' }}>
      <title id="hd-t">Qwen3-8B at a glance</title>
      <desc id="hd-d">Token ids enter an embedding table, pass through 36 decoder blocks each containing attention and a SwiGLU feed-forward network with residual additions, then a final RMSNorm and an untied vocabulary projection produce logits for sampling. Each part links into the explorer.</desc>
      {rows.map((r) => (
        <a key={r.label} href={r.href}>
          <rect x={30} y={r.y} width={400} height={r.h} rx={8} fill={r.kind === 'param' ? 'color-mix(in srgb, var(--weight) 22%, var(--panel))' : 'var(--panel)'} stroke={r.kind === 'param' ? 'var(--weight)' : 'var(--text-3)'} strokeDasharray={r.kind === 'op' ? '5 4' : undefined} strokeWidth={2} />
          <text x={230} y={r.y + 15} textAnchor="middle" fill="var(--text)" fontSize={12.5} fontWeight={600}>{r.label}</text>
          <text x={230} y={r.y + 29} textAnchor="middle" fill="var(--text-2)" fontSize={10}>{r.sub}</text>
        </a>
      ))}
      <rect x={20} y={100} width={420} height={250} rx={12} fill="none" stroke="var(--line)" />
      <text x={34} y={118} fill="var(--text-2)" fontSize={11}>36 decoder blocks · distinct weights per block · {fmtB(a.perBlock)} parameters each</text>
      {Array.from({ length: blocks }, (_, i) => {
        const c = i % 9, r = Math.floor(i / 9);
        return (
          <a key={i} href={explorerHref({ m: 'qwen3-8b', lv: 4, b: i })} aria-label={`Open block ${i}`}>
            <rect x={34 + c * 44} y={128 + r * 54} width={38} height={46} rx={5} fill="var(--panel)" stroke="var(--line)" />
            <rect x={38} y={0} width={30} height={14} rx={3} transform={`translate(${c * 44} ${132 + r * 54})`} fill="color-mix(in srgb, var(--attn) 30%, transparent)" stroke="var(--attn)" />
            <rect x={38} y={0} width={30} height={14} rx={3} transform={`translate(${c * 44} ${150 + r * 54})`} fill="color-mix(in srgb, var(--ffn) 30%, transparent)" stroke="var(--ffn)" />
            <text x={53 + c * 44} y={170 + r * 54} textAnchor="middle" fontSize={8.5} fill="var(--text-3)">{i}</text>
          </a>
        );
      })}
      <g fontSize={10} fill="var(--text-2)">
        <rect x={300} y={352 - 12} width={10} height={8} fill="color-mix(in srgb, var(--attn) 30%, transparent)" stroke="var(--attn)" /><text x={314} y={348}>attention</text>
        <rect x={370} y={352 - 12} width={10} height={8} fill="color-mix(in srgb, var(--ffn) 30%, transparent)" stroke="var(--ffn)" /><text x={384} y={348}>FFN</text>
      </g>
      <a href={explorerHref({ m: 'qwen3-8b', lv: 3, mod: 'final_norm' })}>
        <rect x={30} y={362} width={400} height={30} rx={8} fill="color-mix(in srgb, var(--norm) 16%, var(--panel))" stroke="var(--norm)" strokeWidth={2} />
        <text x={230} y={381} textAnchor="middle" fill="var(--text)" fontSize={12.5} fontWeight={600}>Final RMSNorm (4,096 learned gains)</text>
      </a>
      <a href={explorerHref({ m: 'qwen3-8b', lv: 2, t: 'lm_head' })}>
        <rect x={30} y={400} width={400} height={30} rx={8} fill="color-mix(in srgb, var(--weight) 22%, var(--panel))" stroke="var(--weight)" strokeWidth={2} />
        <text x={230} y={419} textAnchor="middle" fill="var(--text)" fontSize={12.5} fontWeight={600}>Untied output head → {fmtInt(151936)} logits</text>
      </a>
      <a href="#/labs/sampling">
        <rect x={30} y={438} width={400} height={28} rx={8} fill="var(--panel)" stroke="var(--text-3)" strokeDasharray="5 4" strokeWidth={2} />
        <text x={230} y={456} textAnchor="middle" fill="var(--text)" fontSize={12} fontWeight={600}>Sampling → next token (no learned weights)</text>
      </a>
    </svg>
  );
}

export function Home() {
  const a8 = accounting(QWEN3_8B), a30 = accounting(QWEN3_30B_A3B);
  return (
    <div className="page">
      <section className="hero">
        <div>
          <h1>Where are the weights, neurons and experts in a Qwen model?</h1>
          <p className="muted" style={{ fontSize: '1.05rem' }}>
            Qwen Atlas lets you zoom from one stored number all the way out to the whole model, for <strong>Qwen3-8B</strong> (dense) and <strong>Qwen3-30B-A3B</strong> (mixture of experts). Every number is labeled: verified from the official files, derived by a shown formula, simulated in a tiny toy model, or actually read from a checkpoint.
          </p>
          <div className="row" style={{ margin: '14px 0' }}>
            <a className="btn primary" href={START_WEIGHT} data-testid="start-weight">● Start with one weight</a>
            <a className="btn" href={explorerHref({ m: 'qwen3-8b', lv: 5 })}>Start from the whole model</a>
            <a className="btn" href="#/labs">Open the labs</a>
          </div>
          <div className="callout answer">
            <strong>Short answer:</strong> the learned parameters are the stored numbers in these matrices and vectors{' '}
            (<a href={explorerHref({ m: 'qwen3-8b', lv: 0, b: 0, t: 'q_proj', r: 0, c: 0 })}>see one real one</a>). The changing highlighted values flowing between modules are{' '}
            <a href="#/labs/unit">activations</a>. Qwen3-8B has <strong>{fmtInt(a8.total)}</strong> parameters; that is not 8 billion neurons.
          </div>
          <EvidenceLegend />
        </div>
        <div className="card" style={{ padding: 10 }}>
          <StackDiagram />
          <div className="faint" style={{ fontSize: '0.8rem', padding: '0 6px' }}>Every box is a link into the explorer. <Evidence kind="verified" /> structure from the pinned Qwen3-8B config and checkpoint headers.</div>
        </div>
      </section>

      <h2>Three ways in</h2>
      <div className="grid3">
        <div className="card"><h3>Bottom-up</h3><p className="muted">One BF16 number → the weighted sum it joins → its matrix → attention / FFN → the block → the stack.</p><a href={START_WEIGHT}>Begin at Level 0 →</a></div>
        <div className="card"><h3>Overview first</h3><p className="muted">Start with the full model and open blocks, modules and tensors as you need them.</p><a href={explorerHref({ m: 'qwen3-8b', lv: 5 })}>Begin at Level 5 →</a></div>
        <div className="card"><h3>Guided tours</h3><p className="muted">Short, accessible walk-throughs.</p>
          <ul style={{ margin: 0, paddingLeft: 18 }}>
            <li><a href="#/tours/weights">Where are the weights?</a></li>
            <li><a href="#/tours/neuron">What is a neuron here?</a></li>
            <li><a href="#/tours/token">How does one token move?</a></li>
            <li><a href="#/tours/moe">What changes in MoE?</a></li>
          </ul>
        </div>
      </div>

      <h2 style={{ marginTop: 22 }}>Parameters, activations, neurons: the distinctions that matter</h2>
      <div className="grid2">
        <div className="card">
          <ul style={{ paddingLeft: 18, margin: 0 }}>
            <li><strong>Parameter</strong>: one learned scalar. Most are entries of weight matrices or embedding tables; normalization gains and routers also hold parameters.</li>
            <li><strong>Weight matrix</strong>: many parameters. Qwen3-8B's W_Q alone has {fmtInt(a8.qkvoPerBlock.q)} of them per block.</li>
            <li><strong>Activation</strong>: a value computed for a particular input. It changes when the input changes even though the weights stay fixed.</li>
            <li><strong>Neuron / unit / channel</strong>: a computational coordinate. A transformer has several different widths (4,096 residual features, 128 per head, 12,288 FFN channels…), so there is no single convention-free “neuron count”.</li>
            <li><strong>Token</strong>: a tokenizer unit, not necessarily a word.</li>
          </ul>
        </div>
        <div className="card">
          <ul style={{ paddingLeft: 18, margin: 0 }}>
            <li><strong>Attention weights</strong> are computed mixing coefficients; the learned W_Q, W_K, W_V, W_O are different objects.</li>
            <li><strong>Routing weights</strong> can mean the learned router matrix or the per-token mixture coefficients; this atlas always names which.</li>
            <li>An <strong>expert</strong> is an FFN inside a decoder block, not a separate model or chatbot.</li>
            <li>Residual additions and RoPE have <strong>no learned weights</strong>; normalization gains are learned.</li>
            <li>Units are not processor cores: matrices are stored in memory and evaluated by software kernels.</li>
            <li>Qwen3-30B-A3B stores <strong>{fmtInt(a30.total)}</strong> parameters but uses only about 3.3B “activated” per token (a convention; see <a href="#/parameters?m=qwen3-30b-a3b">the accounting</a>).</li>
          </ul>
        </div>
      </div>
      <div className="card" style={{ marginTop: 16 }}>
        <h2 style={{ marginTop: 0 }}>A tiny editable example</h2>
        <TinyExample />
      </div>
    </div>
  );
}
