import { QWEN3_30B_A3B, QWEN3_8B } from '../models/profiles';
import { accounting } from '../models/shapes';
import { Evidence, EvidenceLegend, fmtB } from '../components/ui';
import { explorerHref } from '../state/route';
import { TinyExample } from '../labs/TinyExample';
import { Rich, useLang } from '../i18n/i18n';

const START_WEIGHT = explorerHref({ m: 'qwen3-8b', lv: 0, b: 0, t: 'q_proj', r: 0, c: 0 });

function StackDiagram() {
  const { lang, num } = useLang();
  const T = (en: string, sk: string) => (lang === 'sk' ? sk : en);
  const a = accounting(QWEN3_8B);
  const B = (n: number) => fmtB(n).replace('.', lang === 'sk' ? ',' : '.');
  const rows: { y: number; h: number; label: string; sub: string; href: string; kind: 'op' | 'param' }[] = [
    { y: 8, h: 34, label: T('Text → chat template → token ids', 'Text → šablóna rozhovoru → ID tokenov'), sub: T('tokenizer (no learned weights)', 'tokenizér (bez naučených váh)'), href: '#/pipeline', kind: 'op' },
    { y: 52, h: 38, label: T(`Embedding table [${num(151936)} × ${num(4096)}]`, `Tabuľka embeddingov [${num(151936)} × ${num(4096)}]`), sub: T(`${B(a.embedding)} parameters · one row looked up per token`, `${B(a.embedding)} parametrov · pre každý token sa vyhľadá jeden riadok`), href: explorerHref({ m: 'qwen3-8b', lv: 2, t: 'embed_tokens' }), kind: 'param' },
  ];
  return (
    <svg viewBox="0 0 460 470" role="group" aria-labelledby="hd-t hd-d" style={{ width: '100%', maxWidth: 520, height: 'auto' }}>
      <title id="hd-t">{T('Qwen3-8B at a glance', 'Qwen3-8B na prvý pohľad')}</title>
      <desc id="hd-d">{T('Token ids enter an embedding table, pass through 36 decoder blocks each containing attention and a feed-forward network with residual additions, then a final RMSNorm and an untied vocabulary projection produce scores for sampling. Each part links into the explorer.', 'ID tokenov vstúpia do tabuľky embeddingov, prejdú 36 blokmi dekodéra, z ktorých každý obsahuje pozornosť a doprednú sieť s reziduálnymi sčítaniami; potom záverečná RMSNorm a samostatná výstupná projekcia vytvoria skóre pre vzorkovanie. Každá časť je odkazom do prieskumníka.')}</desc>
      {rows.map((r) => (
        <a key={r.label} href={r.href}>
          <rect x={30} y={r.y} width={400} height={r.h} rx={8} fill={r.kind === 'param' ? 'color-mix(in srgb, var(--weight) 22%, var(--panel))' : 'var(--panel)'} stroke={r.kind === 'param' ? 'var(--weight)' : 'var(--text-3)'} strokeDasharray={r.kind === 'op' ? '5 4' : undefined} strokeWidth={2} />
          <text x={230} y={r.y + 15} textAnchor="middle" fill="var(--text)" fontSize={12.5} fontWeight={600}>{r.label}</text>
          <text x={230} y={r.y + 29} textAnchor="middle" fill="var(--text-2)" fontSize={10}>{r.sub}</text>
        </a>
      ))}
      <rect x={20} y={100} width={420} height={250} rx={12} fill="none" stroke="var(--line)" />
      <text x={34} y={118} fill="var(--text-2)" fontSize={11}>{T(`36 decoder blocks · distinct weights per block · ${B(a.perBlock)} parameters each`, `36 blokov dekodéra · každý blok má vlastné váhy · ${B(a.perBlock)} parametrov v každom`)}</text>
      {Array.from({ length: 36 }, (_, i) => {
        const c = i % 9, r = Math.floor(i / 9);
        return (
          <a key={i} href={explorerHref({ m: 'qwen3-8b', lv: 4, b: i })} aria-label={T(`Open block ${i}`, `Otvoriť blok ${i}`)}>
            <rect x={34 + c * 44} y={128 + r * 54} width={38} height={46} rx={5} fill="var(--panel)" stroke="var(--line)" />
            <rect x={38 + c * 44} y={132 + r * 54} width={30} height={14} rx={3} fill="color-mix(in srgb, var(--attn) 30%, transparent)" stroke="var(--attn)" />
            <rect x={38 + c * 44} y={150 + r * 54} width={30} height={14} rx={3} fill="color-mix(in srgb, var(--ffn) 30%, transparent)" stroke="var(--ffn)" />
            <text x={53 + c * 44} y={170 + r * 54} textAnchor="middle" fontSize={8.5} fill="var(--text-3)">{i}</text>
          </a>
        );
      })}
      <g fontSize={10} fill="var(--text-2)">
        <rect x={290} y={340} width={10} height={8} fill="color-mix(in srgb, var(--attn) 30%, transparent)" stroke="var(--attn)" /><text x={304} y={348}>{T('attention', 'pozornosť')}</text>
        <rect x={370} y={340} width={10} height={8} fill="color-mix(in srgb, var(--ffn) 30%, transparent)" stroke="var(--ffn)" /><text x={384} y={348}>FFN</text>
      </g>
      <a href={explorerHref({ m: 'qwen3-8b', lv: 3, mod: 'final_norm' })}>
        <rect x={30} y={362} width={400} height={30} rx={8} fill="color-mix(in srgb, var(--norm) 16%, var(--panel))" stroke="var(--norm)" strokeWidth={2} />
        <text x={230} y={381} textAnchor="middle" fill="var(--text)" fontSize={12.5} fontWeight={600}>{T(`Final RMSNorm (${num(4096)} learned gains)`, `Záverečná RMSNorm (${num(4096)} naučených ziskov)`)}</text>
      </a>
      <a href={explorerHref({ m: 'qwen3-8b', lv: 2, t: 'lm_head' })}>
        <rect x={30} y={400} width={400} height={30} rx={8} fill="color-mix(in srgb, var(--weight) 22%, var(--panel))" stroke="var(--weight)" strokeWidth={2} />
        <text x={230} y={419} textAnchor="middle" fill="var(--text)" fontSize={12.5} fontWeight={600}>{T(`Untied output head → ${num(151936)} logits`, `Samostatná výstupná hlava → ${num(151936)} logitov`)}</text>
      </a>
      <a href="#/labs/sampling">
        <rect x={30} y={438} width={400} height={28} rx={8} fill="var(--panel)" stroke="var(--text-3)" strokeDasharray="5 4" strokeWidth={2} />
        <text x={230} y={456} textAnchor="middle" fill="var(--text)" fontSize={12} fontWeight={600}>{T('Sampling → next token (no learned weights)', 'Vzorkovanie → ďalší token (bez naučených váh)')}</text>
      </a>
    </svg>
  );
}

export function Home() {
  const { t, lang, num } = useLang();
  const T = (en: string, sk: string) => (lang === 'sk' ? sk : en);
  const a8 = accounting(QWEN3_8B), a30 = accounting(QWEN3_30B_A3B);
  const answer = t('home_answer').replace('{w}', START_WEIGHT).replace('{n}', num(a8.total)).replace('{n2}', '8');
  return (
    <div className="page">
      <section className="hero">
        <div>
          <h1>{t('home_h1')}</h1>
          <p className="muted" style={{ fontSize: '1.05rem' }}><Rich text={t('home_lead')} /></p>
          <div className="row" style={{ margin: '14px 0' }}>
            <a className="btn primary" href="#/learn" data-testid="start-learn">{t('home_learn')}</a>
            <a className="btn" href={START_WEIGHT} data-testid="start-weight">{t('home_weight')}</a>
            <a className="btn" href={explorerHref({ m: 'qwen3-8b', lv: 5 })}>{t('home_model')}</a>
            <a className="btn" href="#/labs">{t('home_labs')}</a>
          </div>
          <div className="callout answer">
            <strong>{t('home_short')}</strong> <Rich text={answer} />
          </div>
          <EvidenceLegend />
        </div>
        <div className="card" style={{ padding: 10 }}>
          <StackDiagram />
          <div className="faint" style={{ fontSize: '0.8rem', padding: '0 6px' }}>{t('home_diagram_note')} <Evidence kind="verified" /> {t('home_diagram_src')}</div>
        </div>
      </section>

      <h2>{t('home_ways')}</h2>
      <div className="grid-auto">
        <div className="card"><h3>{t('home_w_learn_t')}</h3><p className="muted">{t('home_w_learn')}</p><a href="#/learn/what">{t('home_w_learn_a')}</a></div>
        <div className="card"><h3>{t('home_w_up_t')}</h3><p className="muted">{t('home_w_up')}</p><a href={START_WEIGHT}>{t('home_w_up_a')}</a></div>
        <div className="card"><h3>{t('home_w_down_t')}</h3><p className="muted">{t('home_w_down')}</p><a href={explorerHref({ m: 'qwen3-8b', lv: 5 })}>{t('home_w_down_a')}</a></div>
        <div className="card"><h3>{t('home_w_tour_t')}</h3><p className="muted">{t('home_w_tour')}</p>
          <ul style={{ margin: 0, paddingLeft: 18 }}>
            <li><a href="#/tours/weights">{T('Where are the weights?', 'Kde sú váhy?')}</a></li>
            <li><a href="#/tours/neuron">{T('What is a neuron here?', 'Čo je tu neurón?')}</a></li>
            <li><a href="#/tours/token">{T('How does one token move?', 'Ako sa pohybuje jeden token?')}</a></li>
            <li><a href="#/tours/moe">{T('What changes in MoE?', 'Čo sa mení v MoE?')}</a></li>
          </ul>
        </div>
      </div>

      <h2 style={{ marginTop: 22 }}>{t('home_dist')}</h2>
      <div className="grid2">
        <div className="card">
          <ul style={{ paddingLeft: 18, margin: 0 }}>
            <li><Rich text={T('**Parameter**: one learned scalar. Most are entries of weight matrices or embedding tables; normalization gains and routers also hold parameters.', '**Parameter**: jedno naučené číslo. Väčšina sú prvky matíc váh alebo tabuliek embeddingov; parametre majú aj zisky normalizácie a smerovače.')} /></li>
            <li><Rich text={T(`**Weight matrix**: many parameters. Qwen3-8B’s W_Q alone has ${num(a8.qkvoPerBlock.q)} of them per block.`, `**Matica váh**: mnoho parametrov. Len W_Q v Qwen3-8B ich má ${num(a8.qkvoPerBlock.q)} v každom bloku.`)} /></li>
            <li><Rich text={T('**Activation**: a value computed for a particular input. It changes when the input changes even though the weights stay fixed.', '**Aktivácia**: hodnota vypočítaná pre konkrétny vstup. Mení sa so vstupom, hoci váhy zostávajú rovnaké.')} /></li>
            <li><Rich text={T('**Neuron / unit / channel**: a computational coordinate. A transformer has several different widths (4,096 residual features, 128 per head, 12,288 FFN channels…), so there is no single convention-free “neuron count”.', '**Neurón / jednotka / kanál**: výpočtová súradnica. Transformer má viac rôznych šírok (4 096 vlastností reziduálneho prúdu, 128 na hlavu, 12 288 kanálov FFN…), preto neexistuje jediný „počet neurónov“ nezávislý od dohody.')} /></li>
            <li><Rich text={T('**Token**: a tokenizer unit, not necessarily a word.', '**Token**: jednotka tokenizéra, nie nevyhnutne slovo.')} /></li>
          </ul>
        </div>
        <div className="card">
          <ul style={{ paddingLeft: 18, margin: 0 }}>
            <li><Rich text={T('**Attention weights** are computed mixing coefficients; the learned W_Q, W_K, W_V, W_O are different objects.', '**Váhy pozornosti** sú vypočítané koeficienty miešania; naučené matice W_Q, W_K, W_V, W_O sú iné objekty.')} /></li>
            <li><Rich text={T('**Routing weights** can mean the learned router matrix or the per-token mixture coefficients; this atlas always names which.', '**Váhy smerovania** môžu znamenať naučenú maticu smerovača alebo koeficienty zmesi pre konkrétny token; tento atlas vždy uvádza, ktoré z toho.')} /></li>
            <li><Rich text={T('An **expert** is an FFN inside a decoder block, not a separate model or chatbot.', '**Expert** je dopredná sieť vo vnútri bloku dekodéra, nie samostatný model ani chatbot.')} /></li>
            <li><Rich text={T('Residual additions and RoPE have **no learned weights**; normalization gains are learned.', 'Reziduálne sčítania a RoPE **nemajú naučené váhy**; zisky normalizácie sú naučené.')} /></li>
            <li><Rich text={T('Units are not processor cores: matrices are stored in memory and evaluated by software kernels.', 'Jednotky nie sú jadrá procesora: matice sú uložené v pamäti a počítajú ich softvérové jadrá (kernely).')} /></li>
            <li><Rich text={T(`Qwen3-30B-A3B stores **${num(a30.total)}** parameters but uses only about 3.3B “activated” per token (a convention; see [the accounting](#/parameters?m=qwen3-30b-a3b)).`, `Qwen3-30B-A3B ukladá **${num(a30.total)}** parametrov, ale na jeden token používa len približne 3,3 mld. „aktivovaných“ (spôsob počítania; pozri [účtovanie](#/parameters?m=qwen3-30b-a3b)).`)} /></li>
          </ul>
        </div>
      </div>
      <div className="card" style={{ marginTop: 16 }}>
        <h2 style={{ marginTop: 0 }}>{t('home_tiny')}</h2>
        <TinyExample />
      </div>
    </div>
  );
}
