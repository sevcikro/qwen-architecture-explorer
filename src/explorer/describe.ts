import type { EvidenceClass } from '../evidence/types';
import type { CoreProfile } from '../models/profiles';
import { accounting, prod, tensorFamilies, tensorName, type TensorFamily } from '../models/shapes';
import { explorerHref, type ExplorerState } from '../state/route';
import { formatNum, type Lang } from '../i18n/i18n';

export type Role = 'param' | 'norm' | 'act' | 'state' | 'op' | 'group';

export interface InspectorInfo {
  id: string;
  title: string;
  role: Role;
  what: string;
  where: string;
  inputs: string;
  outputs: string;
  params: { count: number; formula: string } | null;
  persistence: string;
  equation?: string;
  checkpoint?: string[];
  implementation?: string;
  evidence: { kind: EvidenceClass; note: string }[];
  sources: string[];
  below: { label: string; href: string }[];
  above?: { label: string; href: string };
  notes?: string[];
}

const V = (p: CoreProfile) => (p.status === 'official' ? 'verified' : 'simulation') as EvidenceClass;
const srcs = (p: CoreProfile, extra: string[] = []) => [...(p.status === 'official' ? p.sources : ['tf-qwen3']), ...extra];

export function famByKey(p: CoreProfile, key: string): TensorFamily | undefined {
  return tensorFamilies(p).find((f) => f.key === key);
}

/** Small language kit used by the pure description builders. */
function kit(lang: Lang) {
  const T = (en: string, sk: string) => (lang === 'sk' ? sk : en);
  const F = (n: number) => formatNum(lang, n);
  const shape = (s: number[]) => `[${s.map(F).join(', ')}]`;
  return { T, F, shape };
}

export function describeTensor(p: CoreProfile, f: TensorFamily, s: Partial<ExplorerState>, lang: Lang = 'en'): InspectorInfo {
  const { T, F, shape } = kit(lang);
  const b = s.b ?? 0;
  const name = tensorName(f, b, s.e ?? 0);
  const n = prod(f.shape);
  const isMat = f.shape.length === 2;
  const blockLabel = (bb?: number) => (bb === undefined ? T(`every decoder block (×${p.layers})`, `každý blok dekodéra (×${p.layers})`) : T(`decoder block ${bb} of ${p.layers} (code index ${bb}, human count ${bb + 1})`, `blok dekodéra ${bb} z ${p.layers} (index v kóde ${bb}, ľudské poradie ${bb + 1})`));
  const eq: Record<string, string> = {
    embed_tokens: 'x_t = E[\\text{id}_t,\\,:]',
    q_proj: 'Q = X\\,W_Q^{\\top}',
    k_proj: 'K = X\\,W_K^{\\top}',
    v_proj: 'V = X\\,W_V^{\\top}',
    o_proj: 'Y = Z\\,W_O^{\\top}',
    q_norm: '\\hat q_h = \\gamma_q \\odot q_h / \\sqrt{\\operatorname{mean}(q_h^2)+\\epsilon}',
    k_norm: '\\hat k_h = \\gamma_k \\odot k_h / \\sqrt{\\operatorname{mean}(k_h^2)+\\epsilon}',
    input_layernorm: '\\operatorname{RMSNorm}(x)=\\gamma \\odot x/\\sqrt{\\operatorname{mean}(x^2)+\\epsilon}',
    post_attention_layernorm: '\\operatorname{RMSNorm}(h)=\\gamma \\odot h/\\sqrt{\\operatorname{mean}(h^2)+\\epsilon}',
    gate_proj: 'g = \\operatorname{SiLU}(X\\,W_{gate}^{\\top})',
    up_proj: 'u = X\\,W_{up}^{\\top}',
    down_proj: 'Y = (g \\odot u)\\,W_{down}^{\\top}',
    router: 'r = x\\,W_{router}^{\\top},\\; p = \\operatorname{softmax}(r)',
    expert_gate_proj: 'g_e = \\operatorname{SiLU}(x\\,W_{gate,e}^{\\top})',
    expert_up_proj: 'u_e = x\\,W_{up,e}^{\\top}',
    expert_down_proj: '\\operatorname{Expert}_e(x) = (g_e \\odot u_e)\\,W_{down,e}^{\\top}',
    norm: '\\operatorname{RMSNorm}(h_L)=\\gamma \\odot h_L/\\sqrt{\\operatorname{mean}(h_L^2)+\\epsilon}',
    lm_head: '\\text{logits} = \\hat h\\,W_{vocab}^{\\top}',
  };
  const nx = T('normalized X', 'normalizované X'), nh = T('normalized h', 'normalizované h');
  const inOut: Record<string, [string, string]> = {
    embed_tokens: [T('token ids [B, T] (integers)', 'ID tokenov [B, T] (celé čísla)'), `X [B, T, ${p.d}]`],
    q_proj: [`${nx} [B, T, ${p.d}]`, `Q [B, T, ${p.hq * p.headDim}] → ${T('reshaped', 'preskupené')} [B, ${p.hq}, T, ${p.headDim}]`],
    k_proj: [`${nx} [B, T, ${p.d}]`, `K [B, T, ${p.hkv * p.headDim}] → ${T('reshaped', 'preskupené')} [B, ${p.hkv}, T, ${p.headDim}]`],
    v_proj: [`${nx} [B, T, ${p.d}]`, `V [B, T, ${p.hkv * p.headDim}] → ${T('reshaped', 'preskupené')} [B, ${p.hkv}, T, ${p.headDim}]`],
    o_proj: [`${T('concatenated head outputs', 'spojené výstupy hláv')} Z [B, T, ${p.hq * p.headDim}]`, `${T('attention branch output', 'výstup vetvy pozornosti')} [B, T, ${p.d}]`],
    q_norm: [`${T('each query head vector', 'vektor každej hlavy dopytov')} [${p.headDim}]`, `${T('normalized query head', 'normalizovaná hlava dopytov')} [${p.headDim}]`],
    k_norm: [`${T('each key head vector', 'vektor každej hlavy kľúčov')} [${p.headDim}]`, `${T('normalized key head', 'normalizovaná hlava kľúčov')} [${p.headDim}]`],
    input_layernorm: [`${T('block input', 'vstup bloku')} x [B, T, ${p.d}]`, `${T('normalized x', 'normalizované x')} [B, T, ${p.d}]`],
    post_attention_layernorm: [`h = x + ${T('attention', 'pozornosť')} [B, T, ${p.d}]`, `${nh} [B, T, ${p.d}]`],
    norm: [`${T('last block output', 'výstup posledného bloku')} [B, T, ${p.d}]`, `${T('normalized', 'normalizované')} [B, T, ${p.d}]`],
    lm_head: [`${T('normalized final hidden state', 'normalizovaný záverečný skrytý stav')} [B, T, ${p.d}]`, `${T('logits', 'logity')} [B, T, ${F(p.vocabRows)}]`],
    router: [`${nh}, ${T('one token', 'jeden token')} [${p.d}]`, `${p.moe?.numExperts} ${T('router logits', 'logitov smerovača')}`],
  };
  const io = inOut[f.key] ?? (f.key.startsWith('expert_')
    ? (f.key === 'expert_down_proj' ? [`${T('gated channels', 'hradlované kanály')} [${p.moe!.expertWidth}]`, `${T('expert output', 'výstup experta')} [${p.d}]`] : [`${nh}, ${T('one routed token', 'jeden nasmerovaný token')} [${p.d}]`, `[${p.moe!.expertWidth}] ${T('channels', 'kanálov')}`])
    : f.key === 'down_proj' ? [`${T('gated channels', 'hradlované kanály')} z [B, T, ${p.ffnWidth}]`, `${T('FFN output', 'výstup FFN')} [B, T, ${p.d}]`] : [`${nh} [B, T, ${p.d}]`, `[B, T, ${p.ffnWidth}]`]);
  const notes: string[] = [];
  if (f.key === 'embed_tokens' && p.tokenizerVocab) notes.push(T(`${F(p.vocabRows)} rows is the configured table size. The tokenizer defines ${F(p.tokenizerVocab)} token ids (151,643 regular + 26 added); the remaining ${F(p.vocabRows - p.tokenizerVocab)} rows are padding that no token id maps to.`, `${F(p.vocabRows)} riadkov je nastavená veľkosť tabuľky. Tokenizér definuje ${F(p.tokenizerVocab)} ID tokenov (151 643 bežných + 26 pridaných); zvyšných ${F(p.vocabRows - p.tokenizerVocab)} riadkov je výplň, na ktorú neukazuje žiadne ID tokenu.`));
  if (f.key === 'router') notes.push(T('Naming trap: the checkpoint calls this tensor "mlp.gate". It is the router, not the SwiGLU gate_proj inside each expert.', 'Pozor na názov: checkpoint volá tento tenzor „mlp.gate“. Je to smerovač, nie gate_proj zo SwiGLU v každom expertovi.'));
  if (f.key.startsWith('expert_')) notes.push(T('Checkpoint stores each expert as its own tensor. Current Transformers packs all experts of a layer into gate_up_proj [E, 2I, H] and down_proj [E, H, I] at load time; the parameter count is the same scalars counted once.', 'Checkpoint ukladá každého experta ako samostatný tenzor. Súčasná knižnica Transformers pri načítaní zbalí všetkých expertov vrstvy do gate_up_proj [E, 2I, H] a down_proj [E, H, I]; počet parametrov tvoria tie isté čísla, počítané raz.'));
  if (f.key === 'q_norm' || f.key === 'k_norm') notes.push(T(`One ${p.headDim}-value gain vector shared by every ${f.key === 'q_norm' ? 'query' : 'key'} head in the block: ${p.headDim} parameters, not ${p.headDim} × heads.`, `Jeden vektor ziskov so ${p.headDim} hodnotami, ktorý zdieľajú všetky hlavy ${f.key === 'q_norm' ? 'dopytov' : 'kľúčov'} v bloku: ${p.headDim} parametrov, nie ${p.headDim} × počet hláv.`));
  if (f.key === 'lm_head') notes.push(T('Untied: this is a separate matrix from the input embedding table, so the two together hold 2 × vocab × d parameters.', 'Neviazaná: je to samostatná matica, odlišná od vstupnej tabuľky embeddingov, takže spolu majú 2 × slovník × d parametrov.'));
  if (p.kind === 'moe' && f.key === 'q_proj') notes.push(T(`Here d = ${p.d} but Hq·dh = ${p.hq * p.headDim}: the query projection expands width, and W_O brings it back to ${p.d}. head_dim is a config field (${p.headDim}), not d / Hq (= ${p.d / p.hq}).`, `Tu je d = ${p.d}, ale Hq·dh = ${p.hq * p.headDim}: projekcia dopytov šírku zväčší a W_O ju vráti späť na ${p.d}. head_dim je pole konfigurácie (${p.headDim}), nie d / Hq (= ${p.d / p.hq}).`));
  const level2: ExplorerState = { m: p.id, lv: 2, b: f.scope === 'model' ? undefined : b, t: f.key, e: f.scope === 'expert' ? s.e ?? 0 : undefined };
  return {
    id: `tensor:${f.key}`,
    title: lang === 'sk' ? f.labelSk : f.label,
    role: f.kind === 'vector' ? 'norm' : 'param',
    what: lang === 'sk' ? f.roleSk : f.role,
    where: `${f.scope === 'model' ? T('Model-level component (one copy)', 'Súčasť na úrovni modelu (jedna kópia)') : f.scope === 'expert' ? T(`Expert ${s.e ?? 0} of ${p.moe!.numExperts} in ${blockLabel(s.b)}`, `Expert ${s.e ?? 0} z ${p.moe!.numExperts} v časti ${blockLabel(s.b)}`) : T(`Inside ${blockLabel(s.b)}`, `V časti ${blockLabel(s.b)}`)}. ${T('Logical shape', 'Logický tvar')} ${shape(f.shape)} (${f.axes.join(' × ')}).`,
    inputs: io[0],
    outputs: io[1],
    params: { count: n, formula: `${f.shape.map(F).join(' × ')} = ${F(n)}${f.multiplicity > 1 ? T(`  (× ${F(f.multiplicity)} such tensors = ${F(n * f.multiplicity)} in the whole model)`, `  (× ${F(f.multiplicity)} takých tenzorov = ${F(n * f.multiplicity)} v celom modeli)`) : ''}` },
    persistence: T(`Persistent learned weights (${p.status === 'official' ? `stored as ${p.dtype} in the checkpoint` : 'toy float64 values'}). The ${isMat ? 'products computed with it' : 'normalized values it scales'} are runtime activations that change with every input.`, `Trvalé naučené váhy (${p.status === 'official' ? `uložené v checkpointe ako ${p.dtype}` : 'hračkové hodnoty float64'}). ${isMat ? 'Súčiny vypočítané s nimi' : 'Normalizované hodnoty, ktoré škálujú,'} sú aktivácie počas behu, ktoré sa menia s každým vstupom.`),
    equation: eq[f.key],
    checkpoint: [name],
    implementation: f.symbol,
    evidence: [evStruct(p, lang), { kind: 'derived', note: T(`Parameter count = product of the logical shape. ${isMat ? 'PyTorch nn.Linear stores [out, in] and computes X W^T.' : ''}`, `Počet parametrov = súčin logického tvaru. ${isMat ? 'PyTorch nn.Linear ukladá [výstup, vstup] a počíta X W^T.' : ''}`) }],
    sources: srcs(p, isMat ? ['torch-linear'] : ['rmsnorm']),
    below: [
      { label: T('Open this tensor (Level 2: matrix tile)', 'Otvoriť tento tenzor (úroveň 2: výrez matice)'), href: explorerHref(level2) },
      { label: isMat ? T('Open output row 0 (Level 1: unit)', 'Otvoriť výstupný riadok 0 (úroveň 1: jednotka)') : T('Open element 0 (Level 0: scalar)', 'Otvoriť prvok 0 (úroveň 0: číslo)'), href: explorerHref({ ...level2, lv: isMat ? 1 : 0, r: 0 }) },
    ],
    notes,
  };
}

function evStruct(p: CoreProfile, lang: Lang): { kind: EvidenceClass; note: string } {
  const { T } = kit(lang);
  return p.status === 'official'
    ? { kind: 'verified', note: T(`Structure and shapes from ${p.repo} config.json + checkpoint headers @ ${p.revision!.slice(0, 12)} and the pinned Transformers implementation.`, `Štruktúra a tvary z config.json a hlavičiek checkpointu ${p.repo} @ ${p.revision!.slice(0, 12)} a z implementácie Transformers v pevnej verzii.`) }
    : { kind: 'simulation', note: T('Toy simulator: Qwen3 structure at reduced size with seeded synthetic weights. Not Qwen dimensions or values.', 'Hračkový simulátor: štruktúra Qwen3 v zmenšenej veľkosti so syntetickými váhami z osiva. Nie sú to rozmery ani hodnoty modelu Qwen.') };
}

export function describeElement(p: CoreProfile, id: string, s: Partial<ExplorerState>, lang: Lang = 'en'): InspectorInfo {
  const { T, F } = kit(lang);
  const a = accounting(p);
  const b = s.b;
  const blockLabel = (bb?: number) => (bb === undefined ? T(`every decoder block (×${p.layers})`, `každý blok dekodéra (×${p.layers})`) : T(`decoder block ${bb} of ${p.layers} (code index ${bb}, human count ${bb + 1})`, `blok dekodéra ${bb} z ${p.layers} (index v kóde ${bb}, ľudské poradie ${bb + 1})`));
  const base = { evidence: [evStruct(p, lang)], sources: srcs(p), below: [] as { label: string; href: string }[] };
  const blockHref = (bb: number, extra: Partial<ExplorerState> = {}) => explorerHref({ m: p.id, lv: 4, b: bb, ...extra });
  const modHref = (mod: string, extra: Partial<ExplorerState> = {}) => explorerHref({ m: p.id, lv: 3, b: b ?? 0, mod, ...extra });
  const noParams = T('no parameters', 'žiadne parametre');
  if (id.startsWith('tensor:')) {
    const f = famByKey(p, id.slice(7));
    if (f) return describeTensor(p, f, s, lang);
  }
  if (id.startsWith('block:')) {
    const bb = Number(id.slice(6));
    return {
      ...base,
      id,
      title: T(`Decoder block ${bb}`, `Blok dekodéra ${bb}`),
      role: 'group',
      what: T(`One of ${p.layers} stacked decoder blocks. Each block owns its own separate matrices and norm gains; block ${bb}'s weights are different numbers from block ${bb + 1}'s, even though the shapes are identical. Within the block the same weights are reused for every token position.`, `Jeden z ${p.layers} blokov dekodéra uložených na sebe. Každý blok má vlastné matice a zisky normalizácie; váhy bloku ${bb} sú iné čísla než váhy bloku ${bb + 1}, hoci tvary sú rovnaké. V rámci bloku sa tie isté váhy používajú pre každú pozíciu tokenu.`),
      where: `${blockLabel(bb)}. ${T('Checkpoint prefix', 'Predpona v checkpointe')} model.layers.${bb}.*`,
      inputs: `x [B, T, ${p.d}] ${T('from', 'z')} ${bb === 0 ? T('the embedding lookup', 'vyhľadania embeddingu') : T(`block ${bb - 1}`, `bloku ${bb - 1}`)}`,
      outputs: `y [B, T, ${p.d}] ${T('to', 'do')} ${bb === p.layers - 1 ? T('the final RMSNorm', 'záverečnej RMSNorm') : T(`block ${bb + 1}`, `bloku ${bb + 1}`)}`,
      params: { count: a.perBlock, formula: `${T('attention', 'pozornosť')} ${F(a.attentionPerBlock)} + ${T('norms', 'normy')} ${F(a.normsPerBlock)} + ${p.kind === 'dense' ? `FFN ${F(a.ffnPerBlock)}` : `${T('experts', 'experti')} ${F(a.ffnPerBlock)} + ${T('router', 'smerovač')} ${F(a.routerPerBlock)}`} = ${F(a.perBlock)}` },
      persistence: T('Weights are persistent; the vectors flowing through are activations recomputed for each input. During generation the block also keeps a K/V cache (runtime state, not trained parameters).', 'Váhy sú trvalé; vektory, ktoré cez blok tečú, sú aktivácie počítané nanovo pre každý vstup. Pri generovaní blok uchováva aj K/V cache (stav počas behu, nie natrénované parametre).'),
      equation: `h = x + \\operatorname{Attn}(\\operatorname{RMSNorm}_{attn}(x)),\\quad y = h + \\operatorname{${p.kind === 'dense' ? 'FFN' : 'MoE'}}(\\operatorname{RMSNorm}_{ffn}(h))`,
      checkpoint: [`model.layers.${bb}.*`],
      implementation: p.kind === 'dense' ? 'Qwen3DecoderLayer' : 'Qwen3MoeDecoderLayer',
      below: [{ label: T(`Open block ${bb} (Level 4)`, `Otvoriť blok ${bb} (úroveň 4)`), href: blockHref(bb) }],
      above: { label: T('Back to the full model', 'Späť na celý model'), href: explorerHref({ m: p.id, lv: 5, b: bb }) },
    };
  }
  type Entry = Omit<InspectorInfo, 'evidence' | 'sources' | 'below'> & { below?: InspectorInfo['below']; sources?: string[]; evidence?: InspectorInfo['evidence'] };
  const common: Record<string, Entry> = {
    model: {
      id, title: p.name, role: 'group',
      what: p.status === 'official'
        ? T(`${p.kind === 'dense' ? 'Dense' : 'Mixture-of-experts'} decoder-only language model. All of its learned parameters are numbers stored in the tensors listed in the catalog; nothing else in the diagram holds learned values.`, `${p.kind === 'dense' ? 'Hustý' : 'Zmes expertov –'} jazykový model typu len-dekodér. Všetky jeho naučené parametre sú čísla uložené v tenzoroch uvedených v katalógu; nič iné v diagrame naučené hodnoty neobsahuje.`)
        : T('The toy simulator: a tiny model with the same operations and tensor names as Qwen3, filled with seeded synthetic weights.', 'Hračkový simulátor: malý model s rovnakými operáciami a názvami tenzorov ako Qwen3, naplnený syntetickými váhami z osiva.'),
      where: p.repo ? `${p.repo} @ ${p.revision!.slice(0, 12)}` : T('Runs entirely in your browser.', 'Beží celý vo vašom prehliadači.'),
      inputs: T('text → chat template → token ids', 'text → šablóna rozhovoru → ID tokenov'),
      outputs: T(`logits over ${F(p.vocabRows)} vocabulary rows → sampled next token`, `logity pre ${F(p.vocabRows)} riadkov slovníka → vybraný ďalší token`),
      params: { count: a.total, formula: `${T('embedding', 'embedding')} ${F(a.embedding)} + ${p.layers} × ${F(a.perBlock)} + ${T('final norm', 'záverečná norma')} ${F(a.finalNorm)} + ${T('head', 'hlava')} ${F(a.head)} = ${F(a.total)}` },
      persistence: T('Learned parameters are fixed during inference. Activations, the K/V cache and sampled tokens are runtime data.', 'Naučené parametre sú počas inferencie pevné. Aktivácie, K/V cache a vybrané tokeny sú údaje vznikajúce počas behu.'),
    },
    tokenizer: {
      id, title: T('Chat template + tokenizer', 'Šablóna rozhovoru + tokenizér'), role: 'op',
      what: T('Turns messages into text with role markers (<|im_start|>user …) and then into integer token ids using byte-level BPE. It is a preprocessing program, not part of the neural network, and has no trainable weights in the checkpoint.', 'Premení správy na text s označeniami rolí (<|im_start|>user …) a potom pomocou byte-level BPE na celočíselné ID tokenov. Je to program na predspracovanie, nie súčasť neurónovej siete, a v checkpointe nemá žiadne trénovateľné váhy.'),
      where: T('Before the model. Files tokenizer.json / tokenizer_config.json in the repository.', 'Pred modelom. Súbory tokenizer.json / tokenizer_config.json v repozitári.'),
      inputs: T('messages / text', 'správy / text'), outputs: T('token ids [B, T]', 'ID tokenov [B, T]'), params: null,
      persistence: T('Fixed vocabulary and merge rules (data files, not parameters).', 'Pevný slovník a pravidlá spájania (dátové súbory, nie parametre).'),
      sources: ['q8-tokenizer', 'qwen3-report'],
      below: [{ label: T('See real Qwen3 tokenizer outputs', 'Pozrieť skutočné výstupy tokenizéra Qwen3'), href: '#/pipeline' }],
    },
    stack: {
      id, title: T(`Decoder stack (${p.layers} blocks)`, `Zásobník dekodéra (${p.layers} blokov)`), role: 'group',
      what: T(`${p.layers} blocks applied one after another. The output of block i is the input of block i+1, so blocks are a sequential dependency; within a block all prompt positions can be computed in parallel during prefill.`, `${p.layers} blokov použitých jeden po druhom. Výstup bloku i je vstupom bloku i+1, preto bloky na sebe postupne závisia; v rámci bloku sa všetky pozície zadania môžu počas prefill počítať súčasne.`),
      where: T('Between the embedding lookup and the final norm.', 'Medzi vyhľadaním embeddingu a záverečnou normou.'), inputs: `X [B, T, ${p.d}]`, outputs: `[B, T, ${p.d}]`,
      params: { count: a.perBlock * p.layers, formula: `${p.layers} × ${F(a.perBlock)}` },
      persistence: T('Distinct weights per block; reused across token positions.', 'Každý blok má vlastné váhy; používajú sa pre všetky pozície tokenov.'),
    },
    sampling: {
      id, title: T('Next-token selection (sampling)', 'Výber ďalšieho tokenu (vzorkovanie)'), role: 'op',
      what: T('A decoding procedure, not a neural-network layer: temperature, top-k and top-p filtering, renormalization and a random draw (or greedy argmax) pick one token id, which is appended and fed back in.', 'Postup dekódovania, nie vrstva neurónovej siete: teplota, filtrovanie top-k a top-p, prepočet a náhodný výber (alebo výber najpravdepodobnejšieho) určia jedno ID tokenu, ktoré sa pripojí a znovu vloží do modelu.'),
      where: T('After the vocabulary projection.', 'Za výstupnou projekciou slovníka.'), inputs: T(`logits [${F(p.vocabRows)}] for the last position`, `logity [${F(p.vocabRows)}] pre poslednú pozíciu`), outputs: T('one token id', 'jedno ID tokenu'), params: null,
      persistence: T('No learned parameters. Settings (e.g. temperature 0.6, top-k 20, top-p 0.95 by default for these models) live in generation_config.json.', 'Žiadne naučené parametre. Nastavenia (pre tieto modely predvolene teplota 0,6, top-k 20, top-p 0,95) sú v súbore generation_config.json.'),
      sources: ['q8-gen', 'q8-card'],
      below: [{ label: T('Open the sampling lab', 'Otvoriť laboratórium vzorkovania'), href: '#/labs/sampling' }],
    },
    attention: {
      id, title: T('Self-attention (GQA, QK-norm, RoPE)', 'Pozornosť (GQA, QK-norm, RoPE)'), role: 'param',
      what: T(`Lets each position read from earlier positions. ${p.hq} query heads compare against ${p.hkv} key heads (${p.hq / p.hkv} query heads share each K/V head), then mix value vectors with the resulting causal attention weights.`, `Umožňuje každej pozícii čítať z predchádzajúcich pozícií. ${p.hq} hláv dopytov sa porovnáva s ${p.hkv} hlavami kľúčov (${p.hq / p.hkv} hlavy dopytov zdieľajú jednu hlavu K/V) a potom zmiešajú vektory hodnôt podľa výsledných kauzálnych váh pozornosti.`),
      where: T(`${blockLabel(b)}, after the first RMSNorm.`, `${blockLabel(b)}, za prvou RMSNorm.`),
      inputs: `${T('normalized x', 'normalizované x')} [B, T, ${p.d}]`, outputs: T(`[B, T, ${p.d}] added to the residual stream`, `[B, T, ${p.d}] pripočítané k reziduálnemu prúdu`),
      params: { count: a.attentionPerBlock + 2 * p.headDim, formula: `W_Q ${F(a.qkvoPerBlock.q)} + W_K ${F(a.qkvoPerBlock.k)} + W_V ${F(a.qkvoPerBlock.v)} + W_O ${F(a.qkvoPerBlock.o)} + ${T('q/k norm gains', 'zisky q/k noriem')} 2 × ${p.headDim}` },
      persistence: T('W_Q/W_K/W_V/W_O and the q/k norm gains are learned. Attention weights (softmax probabilities) are computed activations, recomputed per input. Keys/values of past tokens are cached at runtime.', 'W_Q/W_K/W_V/W_O a zisky q/k noriem sú naučené. Váhy pozornosti (pravdepodobnosti zo softmaxu) sú vypočítané aktivácie, počítané nanovo pre každý vstup. Kľúče a hodnoty minulých tokenov sa počas behu ukladajú do cache.'),
      equation: 'A = \\operatorname{softmax}\\!\\left(\\tfrac{QK^{\\top}}{\\sqrt{d_h}} + M_{causal}\\right),\\; Z = AV',
      implementation: p.kind === 'dense' ? 'Qwen3Attention' : 'Qwen3MoeAttention',
      sources: [...srcs(p), 'vaswani', 'gqa', 'rope'],
      below: [{ label: T('Open the attention module (Level 3)', 'Otvoriť modul pozornosti (úroveň 3)'), href: modHref('attention') }, { label: T('Attention lab (numbers)', 'Laboratórium pozornosti (čísla)'), href: '#/labs/attention' }],
    },
    attn_norm: {
      id, title: T('RMSNorm before attention', 'RMSNorm pred pozornosťou'), role: 'norm',
      what: T('Rescales each token vector to unit root-mean-square, then multiplies elementwise by a learned gain γ. Unlike LayerNorm it does not subtract the mean.', 'Preškáluje vektor každého tokenu na jednotkovú strednú kvadratickú hodnotu a potom ho po prvkoch vynásobí naučeným ziskom γ. Na rozdiel od LayerNorm neodčítava priemer.'),
      where: T(`${blockLabel(b)} entrance (input_layernorm).`, `${blockLabel(b)}, vstup (input_layernorm).`), inputs: `x [B, T, ${p.d}]`, outputs: `[B, T, ${p.d}]`,
      params: { count: p.d, formula: T(`γ has ${F(p.d)} values`, `γ má ${F(p.d)} hodnôt`) },
      persistence: T('γ is learned; the RMS of each token is computed at runtime. Transformers computes the normalization in float32 even when weights are stored in BF16.', 'γ je naučené; RMS každého tokenu sa počíta počas behu. Transformers počíta normalizáciu vo float32, aj keď sú váhy uložené v BF16.'),
      equation: '\\operatorname{RMSNorm}(x)=\\gamma\\odot\\frac{x}{\\sqrt{\\tfrac1d\\sum_i x_i^2+\\epsilon}}', sources: [...srcs(p), 'rmsnorm'],
    },
    ffn_norm: {
      id, title: T('RMSNorm before the FFN / MoE', 'RMSNorm pred FFN / MoE'), role: 'norm',
      what: T('Same operation as the first norm with its own learned γ (post_attention_layernorm).', 'Rovnaká operácia ako prvá norma, s vlastným naučeným γ (post_attention_layernorm).'),
      where: T(`${blockLabel(b)}, after the first residual addition.`, `${blockLabel(b)}, za prvým reziduálnym sčítaním.`), inputs: `h [B, T, ${p.d}]`, outputs: `[B, T, ${p.d}]`,
      params: { count: p.d, formula: T(`γ has ${F(p.d)} values`, `γ má ${F(p.d)} hodnôt`) }, persistence: T('γ learned; normalization computed at runtime.', 'γ je naučené; normalizácia sa počíta počas behu.'),
      equation: '\\operatorname{RMSNorm}(h)=\\gamma\\odot h/\\sqrt{\\operatorname{mean}(h^2)+\\epsilon}', sources: [...srcs(p), 'rmsnorm'],
    },
    residual1: {
      id, title: T('Residual addition after attention', 'Reziduálne sčítanie za pozornosťou'), role: 'op',
      what: T('Adds the attention branch output to the unchanged block input: h = x + attention(...). The addition itself has no learned weights. The values being added are both activations.', 'Pripočíta výstup vetvy pozornosti k nezmenenému vstupu bloku: h = x + pozornosť(...). Samotné sčítanie nemá naučené váhy. Obe sčítavané hodnoty sú aktivácie.'),
      where: blockLabel(b), inputs: T(`x [B, T, ${p.d}] and attention output [B, T, ${p.d}]`, `x [B, T, ${p.d}] a výstup pozornosti [B, T, ${p.d}]`), outputs: `h [B, T, ${p.d}]`, params: { count: 0, formula: noParams },
      persistence: T('Pure runtime arithmetic.', 'Čisté počítanie počas behu.'), equation: 'h = x + \\operatorname{Attn}(\\operatorname{RMSNorm}(x))',
      below: [{ label: T('See the two vectors being added (toy simulator)', 'Pozrieť dva sčítavané vektory (hračkový simulátor)'), href: '#/labs/layers' }],
    },
    residual2: {
      id, title: T('Residual addition after the FFN / MoE', 'Reziduálne sčítanie za FFN / MoE'), role: 'op',
      what: T('Adds the FFN (or MoE) branch output to h: y = h + FFN(...). No learned weights.', 'Pripočíta výstup vetvy FFN (alebo MoE) k h: y = h + FFN(...). Bez naučených váh.'),
      where: blockLabel(b), inputs: T(`h and FFN output, both [B, T, ${p.d}]`, `h a výstup FFN, obe [B, T, ${p.d}]`), outputs: `y [B, T, ${p.d}]`, params: { count: 0, formula: noParams },
      persistence: T('Pure runtime arithmetic.', 'Čisté počítanie počas behu.'), equation: `y = h + \\operatorname{${p.kind === 'dense' ? 'FFN' : 'MoE'}}(\\operatorname{RMSNorm}(h))`,
      below: [{ label: T('See the two vectors being added (toy simulator)', 'Pozrieť dva sčítavané vektory (hračkový simulátor)'), href: '#/labs/layers' }],
    },
    ffn: {
      id, title: T('SwiGLU feed-forward network', 'Dopredná sieť SwiGLU'), role: 'param',
      what: T(`Applied to each token separately. ${F(p.ffnWidth ?? 0)} gated channels: each channel j is SiLU(gate row j · x) × (up row j · x); the down projection mixes all channels back to ${F(p.d)} features.`, `Použije sa na každý token zvlášť. ${F(p.ffnWidth ?? 0)} hradlovaných kanálov: každý kanál j je SiLU(riadok j matice gate · x) × (riadok j matice up · x); projekcia down zmieša všetky kanály späť do ${F(p.d)} vlastností.`),
      where: T(`${blockLabel(b)}, after the second RMSNorm.`, `${blockLabel(b)}, za druhou RMSNorm.`), inputs: `${T('normalized h', 'normalizované h')} [B, T, ${p.d}]`, outputs: `[B, T, ${p.d}]`,
      params: { count: a.ffnPerBlock, formula: `3 × d × f = 3 × ${F(p.d)} × ${F(p.ffnWidth ?? 0)} = ${F(a.ffnPerBlock)}` },
      persistence: T('Three learned matrices; g, u, z are activations.', 'Tri naučené matice; g, u, z sú aktivácie.'), equation: 'Y = \\big(\\operatorname{SiLU}(XW_{gate}^{\\top})\\odot XW_{up}^{\\top}\\big)W_{down}^{\\top}',
      implementation: 'Qwen3MLP', sources: [...srcs(p), 'swiglu', 'torch-silu'],
      below: [{ label: T('Open the FFN module (Level 3)', 'Otvoriť modul FFN (úroveň 3)'), href: modHref('ffn') }, { label: T('Unit microscope', 'Mikroskop jednotky'), href: '#/labs/unit' }],
    },
    moe: {
      id, title: T(`Routed mixture of experts (${p.moe?.numExperts} experts, top-${p.moe?.topK})`, `Smerovaná zmes expertov (${p.moe?.numExperts} expertov, top-${p.moe?.topK})`), role: 'param',
      what: T(`Replaces the dense FFN. For every token in this block the router scores all ${p.moe?.numExperts} experts, keeps the ${p.moe?.topK} most probable, renormalizes their probabilities to sum to 1, and adds those experts' outputs weighted by the renormalized coefficients. ${p.moe?.sharedExperts === 0 ? 'There is no shared expert in this model.' : ''}`, `Nahrádza hustú FFN. Pre každý token v tomto bloku smerovač ohodnotí všetkých ${p.moe?.numExperts} expertov, ponechá ${p.moe?.topK} najpravdepodobnejších, prepočíta ich pravdepodobnosti tak, aby dávali spolu 1, a sčíta výstupy týchto expertov vážené prepočítanými koeficientmi. ${p.moe?.sharedExperts === 0 ? 'Tento model nemá zdieľaného experta.' : ''}`),
      where: T(`${blockLabel(b)}, after the second RMSNorm. Attention stays shared (not routed).`, `${blockLabel(b)}, za druhou RMSNorm. Pozornosť zostáva spoločná (nesmeruje sa).`), inputs: `${T('normalized h, per token', 'normalizované h, pre každý token')} [${p.d}]`, outputs: `[${p.d}] ${T('per token', 'pre každý token')}`,
      params: { count: a.ffnPerBlock + a.routerPerBlock, formula: `${p.moe?.numExperts} ${T('experts', 'expertov')} × ${F(a.oneExpert ?? 0)} + ${T('router', 'smerovač')} ${F(a.routerPerBlock)}` },
      persistence: T('All experts are stored (and must be in memory or reachable) even though only a few run per token. Router logits, probabilities, the selected ids and mixture coefficients are per-token activations.', 'Všetci experti sú uložení (a musia byť v pamäti alebo dostupní), hoci pre token beží len niekoľko z nich. Logity smerovača, pravdepodobnosti, vybrané čísla expertov a koeficienty zmesi sú aktivácie pre konkrétny token.'),
      equation: 'y=\\sum_{e\\in\\operatorname{TopK}(p,k)}\\frac{p_e}{\\sum_{j\\in S}p_j}\\operatorname{Expert}_e(x)', implementation: 'Qwen3MoeSparseMoeBlock (TopKRouter + Experts)',
      sources: [...srcs(p), 'switch'],
      below: [{ label: T('Open the MoE module (Level 3)', 'Otvoriť modul MoE (úroveň 3)'), href: modHref('moe') }, { label: T('Router lab', 'Laboratórium smerovača'), href: '#/labs/router' }],
    },
    embedding: {
      id, title: T('Input embedding lookup', 'Vyhľadanie vstupného embeddingu'), role: 'param',
      what: T('For each token id, copy that row of the embedding table. Only one row per token is read; the rest of the table is untouched for that token.', 'Pre každé ID tokenu sa skopíruje príslušný riadok tabuľky embeddingov. Pre token sa číta len jeden riadok; zvyšok tabuľky zostáva pre tento token nepoužitý.'),
      where: T('Model entrance (model.embed_tokens.weight).', 'Vstup modelu (model.embed_tokens.weight).'), inputs: T('token ids [B, T]', 'ID tokenov [B, T]'), outputs: `X [B, T, ${p.d}]`,
      params: { count: a.embedding, formula: `${F(p.vocabRows)} × ${F(p.d)}` },
      persistence: T('The table is learned; the looked-up rows become the first activations. There is no learned position vector added here: position enters later through RoPE inside attention.', 'Tabuľka je naučená; vyhľadané riadky sa stanú prvými aktiváciami. Nepridáva sa tu žiadny naučený vektor pozície: pozícia vstupuje až neskôr cez RoPE v pozornosti.'),
      equation: 'x_t = E[\\text{id}_t]',
    },
    final_norm: {
      id, title: T('Final RMSNorm', 'Záverečná RMSNorm'), role: 'norm', what: T('Normalizes the last block output before the vocabulary projection.', 'Normalizuje výstup posledného bloku pred projekciou do slovníka.'), where: T('After the last block (model.norm.weight).', 'Za posledným blokom (model.norm.weight).'),
      inputs: `[B, T, ${p.d}]`, outputs: `[B, T, ${p.d}]`, params: { count: a.finalNorm, formula: T(`γ has ${F(p.d)} values`, `γ má ${F(p.d)} hodnôt`) }, persistence: T('γ learned.', 'γ je naučené.'),
      equation: '\\operatorname{RMSNorm}(h_L)', sources: [...srcs(p), 'rmsnorm'],
    },
    lm_head: {
      id, title: T('Vocabulary projection (LM head)', 'Projekcia do slovníka (výstupná hlava)'), role: 'param',
      what: T(`Computes one score (logit) for each of the ${F(p.vocabRows)} vocabulary rows by a dot product with that row of its own weight matrix. Logits are not probabilities; softmax turns them into probabilities.`, `Vypočíta jedno skóre (logit) pre každý z ${F(p.vocabRows)} riadkov slovníka skalárnym súčinom s príslušným riadkom vlastnej matice váh. Logity nie sú pravdepodobnosti; na pravdepodobnosti ich premení softmax.`),
      where: T('Model exit (lm_head.weight).', 'Výstup modelu (lm_head.weight).'), inputs: `[B, T, ${p.d}]`, outputs: `${T('logits', 'logity')} [B, T, ${F(p.vocabRows)}]`,
      params: { count: a.head, formula: `${F(p.vocabRows)} × ${F(p.d)} (${T('untied', 'neviazaná')})` }, persistence: T('Learned and untied from the input embedding in this model.', 'Naučená a v tomto modeli neviazaná so vstupným embeddingom.'),
      equation: '\\text{logits}=\\hat h W_{vocab}^{\\top}',
    },
  };
  const rt = T('Runtime.', 'Počas behu.');
  const ops: Record<string, Omit<InspectorInfo, 'evidence' | 'sources' | 'below'> & { sources?: string[] }> = {
    reshape: { id, title: T('Reshape into heads', 'Preskupenie do hláv'), role: 'op', what: T(`Splits the ${p.hq * p.headDim}-wide query vector into ${p.hq} heads of ${p.headDim}, and the ${p.hkv * p.headDim}-wide key/value vectors into ${p.hkv} heads. A view change with no arithmetic and no weights.`, `Rozdelí vektor dopytov so šírkou ${p.hq * p.headDim} na ${p.hq} hláv po ${p.headDim} a vektory kľúčov a hodnôt so šírkou ${p.hkv * p.headDim} na ${p.hkv} hlavy. Iba zmena pohľadu na dáta, bez počítania a bez váh.`), where: blockLabel(b), inputs: `[B, T, ${p.hq * p.headDim}] / [B, T, ${p.hkv * p.headDim}]`, outputs: `[B, ${p.hq}, T, ${p.headDim}] / [B, ${p.hkv}, T, ${p.headDim}]`, params: null, persistence: T('No storage of its own.', 'Nemá vlastné úložisko.') },
    rope: { id, title: T('Rotary position embedding (RoPE)', 'Rotačné kódovanie pozície (RoPE)'), role: 'op', what: T(`Rotates pairs of coordinates (i, i + ${p.headDim / 2}) of every query and key head by an angle proportional to the token's absolute position, with frequencies θ^(−2i/d_h), θ = ${F(p.ropeTheta)}. Applied to Q and K only, not to V. It has no learned weights and is not a vector added at the embedding.`, `Otáča dvojice súradníc (i, i + ${p.headDim / 2}) každej hlavy dopytov a kľúčov o uhol úmerný absolútnej pozícii tokenu, s frekvenciami θ^(−2i/d_h), θ = ${F(p.ropeTheta)}. Používa sa len na Q a K, nie na V. Nemá naučené váhy a nie je to vektor pridaný pri embeddingu.`), where: T(`${blockLabel(b)}, after q/k norm.`, `${blockLabel(b)}, za q/k normou.`), inputs: T('normalized q, k heads + absolute positions', 'normalizované hlavy q, k + absolútne pozície'), outputs: T('rotated q, k', 'otočené q, k'), params: null, persistence: T('cos/sin tables are computed (non-persistent buffers), not trained.', 'Tabuľky cos/sin sa vypočítajú (nie sú trvalo uložené), nie sú trénované.'), equation: "q'_i = q_i\\cos(m\\omega_i) - q_{i+d/2}\\sin(m\\omega_i)", sources: ['rope', 'tf-qwen3'] },
    sdpa: { id, title: T('Causal scaled dot-product attention (GQA)', 'Kauzálna škálovaná pozornosť so skalárnym súčinom (GQA)'), role: 'op', what: T(`Each query head h uses K/V head ⌊h / ${p.hq / p.hkv}⌋. Scores q·k/√${p.headDim}, future positions masked, softmax per row, weighted sum of V. Optimized kernels (e.g. FlashAttention) compute the same result in tiles without storing the full T×T matrix.`, `Každá hlava dopytov h používa hlavu K/V ⌊h / ${p.hq / p.hkv}⌋. Skóre q·k/√${p.headDim}, budúce pozície zamaskované, softmax po riadkoch, vážený súčet V. Optimalizované jadrá (napr. FlashAttention) vypočítajú ten istý výsledok po blokoch bez uloženia celej matice T×T.`), where: blockLabel(b), inputs: `Q [B, ${p.hq}, Tq, ${p.headDim}], K/V [B, ${p.hkv}, Tk, ${p.headDim}]`, outputs: `Z [B, ${p.hq}, Tq, ${p.headDim}] (${T('logical scores', 'logické skóre')} [B, ${p.hq}, Tq, Tk])`, params: null, persistence: T('Attention probabilities are activations; K/V are cached during generation.', 'Pravdepodobnosti pozornosti sú aktivácie; K/V sa pri generovaní ukladajú do cache.'), equation: '\\operatorname{softmax}(QK^{\\top}/\\sqrt{d_h}+M)V', sources: ['vaswani', 'gqa', 'flashattn', 'tf-qwen3'] },
    concat: { id, title: T('Concatenate head outputs', 'Spojenie výstupov hláv'), role: 'op', what: T(`Puts the ${p.hq} head outputs side by side into one ${p.hq * p.headDim}-wide vector per token.`, `Uloží ${p.hq} výstupov hláv vedľa seba do jedného vektora so šírkou ${p.hq * p.headDim} pre každý token.`), where: blockLabel(b), inputs: `[B, ${p.hq}, T, ${p.headDim}]`, outputs: `[B, T, ${p.hq * p.headDim}]`, params: null, persistence: T('No storage of its own.', 'Nemá vlastné úložisko.') },
    silu: { id, title: T('SiLU nonlinearity', 'Nelinearita SiLU'), role: 'op', what: T('SiLU(a) = a·sigmoid(a), applied to each gate pre-activation separately.', 'SiLU(a) = a·sigmoid(a), použitá na každú pred-aktiváciu hradla zvlášť.'), where: T('Inside the FFN / each expert.', 'Vo FFN / v každom expertovi.'), inputs: T('gate pre-activations', 'pred-aktivácie hradla'), outputs: 'g', params: null, persistence: T('No weights.', 'Bez váh.'), equation: '\\operatorname{SiLU}(a)=a\\,\\sigma(a)', sources: ['torch-silu', 'swiglu'] },
    mul: { id, title: T('Multiplicative gate (⊙)', 'Násobiace hradlo (⊙)'), role: 'op', what: T('Elementwise product z_j = g_j × u_j. This multiplication is why a SwiGLU channel is not the same as a textbook "neuron" with a single weighted sum and activation.', 'Súčin po prvkoch z_j = g_j × u_j. Práve kvôli tomuto násobeniu nie je kanál SwiGLU to isté ako učebnicový „neurón“ s jedným váženým súčtom a aktiváciou.'), where: T('Inside the FFN / each expert.', 'Vo FFN / v každom expertovi.'), inputs: T('g and u', 'g a u'), outputs: 'z', params: null, persistence: T('No weights.', 'Bez váh.') },
    softmax: { id, title: T('Router softmax over all experts', 'Softmax smerovača cez všetkých expertov'), role: 'op', what: T(`Turns the ${p.moe?.numExperts} router logits of a token into probabilities that sum to 1 (computed in float32).`, `Premení ${p.moe?.numExperts} logitov smerovača pre token na pravdepodobnosti, ktoré dávajú spolu 1 (počítané vo float32).`), where: T('MoE router', 'smerovač MoE'), inputs: `${p.moe?.numExperts} ${T('logits', 'logitov')}`, outputs: `${p.moe?.numExperts} ${T('probabilities', 'pravdepodobností')}`, params: null, persistence: T('Per-token activation.', 'Aktivácia pre konkrétny token.'), sources: ['tf-qwen3moe'] },
    topk: { id, title: T(`Top-${p.moe?.topK} selection`, `Výber top-${p.moe?.topK}`), role: 'op', what: T(`Keeps the ${p.moe?.topK} experts with the highest probability. A discrete choice: it has no gradient with respect to which experts are chosen.`, `Ponechá ${p.moe?.topK} expertov s najvyššou pravdepodobnosťou. Je to diskrétna voľba: vzhľadom na to, ktorí experti sú vybraní, nemá gradient.`), where: T('MoE router', 'smerovač MoE'), inputs: T('probabilities', 'pravdepodobnosti'), outputs: T(`${p.moe?.topK} expert ids + their probabilities`, `${p.moe?.topK} čísel expertov + ich pravdepodobnosti`), params: null, persistence: T('Per-token activation.', 'Aktivácia pre konkrétny token.'), sources: ['tf-qwen3moe'] },
    renorm: { id, title: T('Renormalize selected probabilities', 'Prepočet vybraných pravdepodobností'), role: 'op', what: T(`norm_topk_prob = ${p.moe?.normTopkProb}: divide the ${p.moe?.topK} kept probabilities by their sum, so the mixture coefficients α sum to 1.`, `norm_topk_prob = ${p.moe?.normTopkProb}: ${p.moe?.topK} ponechaných pravdepodobností sa vydelí ich súčtom, aby koeficienty zmesi α dávali spolu 1.`), where: T('MoE router', 'smerovač MoE'), inputs: `${p.moe?.topK} ${T('probabilities', 'pravdepodobností')}`, outputs: 'α', params: null, persistence: T('Per-token activation.', 'Aktivácia pre konkrétny token.'), equation: '\\alpha_e = p_e / \\sum_{j\\in S} p_j', sources: ['q30-config', 'tf-qwen3moe'] },
    wsum: { id, title: T('Weighted sum of expert outputs', 'Vážený súčet výstupov expertov'), role: 'op', what: T('Adds the selected experts’ outputs, each scaled by its α. Unselected experts contribute nothing for this token in this block.', 'Sčíta výstupy vybraných expertov, každý vynásobený svojím α. Nevybraní experti pre tento token v tomto bloku nič neprispejú.'), where: T('MoE output', 'výstup MoE'), inputs: `${p.moe?.topK} ${T('expert outputs', 'výstupov expertov')}`, outputs: `[${p.d}]`, params: null, persistence: T('No weights.', 'Bez váh.') },
    sq: { id, title: T('Square and mean', 'Druhá mocnina a priemer'), role: 'op', what: T('Computes mean(x²) over the feature axis for each token.', 'Vypočíta mean(x²) cez os vlastností pre každý token.'), where: T('Inside RMSNorm', 'V RMSNorm'), inputs: `[${p.d}]`, outputs: T('one number per token', 'jedno číslo na token'), params: null, persistence: rt },
    rsqrt: { id, title: T('1 / √(mean + ε)', '1 / √(priemer + ε)'), role: 'op', what: T(`Adds ε = ${p.rmsEps} for numerical safety and takes the reciprocal square root.`, `Pripočíta ε = ${p.rmsEps} pre numerickú bezpečnosť a vypočíta prevrátenú hodnotu odmocniny.`), where: T('Inside RMSNorm', 'V RMSNorm'), inputs: 'mean(x²)', outputs: T('scale factor', 'škálovací faktor'), params: null, persistence: rt },
    scalex: { id, title: T('Scale x', 'Škálovanie x'), role: 'op', what: T('Multiplies every feature by the per-token scale factor.', 'Vynásobí každú vlastnosť škálovacím faktorom daného tokenu.'), where: T('Inside RMSNorm', 'V RMSNorm'), inputs: T('x and scale', 'x a faktor'), outputs: T('normalized x', 'normalizované x'), params: null, persistence: rt },
    lookup: { id, title: T('Row lookup', 'Vyhľadanie riadku'), role: 'op', what: T('Indexing, not a matrix multiply: row id_t of the table becomes x_t.', 'Indexovanie, nie násobenie matíc: riadok id_t tabuľky sa stane x_t.'), where: 'Embedding', inputs: T('token id', 'ID tokenu'), outputs: `[${p.d}]`, params: null, persistence: rt },
    logits: { id, title: T('Logits', 'Logity'), role: 'act', what: T('Unnormalized scores, one per vocabulary row. Large positive = more likely after softmax.', 'Nenormalizované skóre, jedno pre každý riadok slovníka. Veľké kladné = po softmaxe pravdepodobnejšie.'), where: T('Model output', 'Výstup modelu'), inputs: '', outputs: `[${F(p.vocabRows)}]`, params: null, persistence: T('Runtime activation.', 'Aktivácia počas behu.') },
    temp: { id, title: T('Temperature', 'Teplota'), role: 'op', what: T('Divides logits by T. T < 1 sharpens, T > 1 flattens; T = 0 is treated as explicit greedy argmax.', 'Vydelí logity hodnotou T. T < 1 rozdelenie zaostrí, T > 1 ho sploští; T = 0 znamená vždy vybrať najpravdepodobnejší token.'), where: T('Sampling', 'Vzorkovanie'), inputs: T('logits', 'logity'), outputs: T('scaled logits', 'škálované logity'), params: null, persistence: T('A setting, not a parameter.', 'Nastavenie, nie parameter.'), sources: ['q8-gen'] },
    filter: { id, title: T('Top-k / top-p filtering + renormalization', 'Filtrovanie top-k / top-p + prepočet'), role: 'op', what: T('Removes unlikely tokens, then renormalizes the remaining probabilities to sum to 1.', 'Odstráni nepravdepodobné tokeny a zvyšné pravdepodobnosti prepočíta tak, aby dávali spolu 1.'), where: T('Sampling', 'Vzorkovanie'), inputs: T('probabilities', 'pravdepodobnosti'), outputs: T('filtered distribution', 'filtrované rozdelenie'), params: null, persistence: T('Setting.', 'Nastavenie.') },
    draw: { id, title: T('Random draw', 'Náhodný výber'), role: 'op', what: T('Samples one token id from the filtered distribution with a pseudo-random number (seeded in the simulator).', 'Vyberie jedno ID tokenu z filtrovaného rozdelenia pomocou pseudonáhodného čísla (v simulátore z osiva).'), where: T('Sampling', 'Vzorkovanie'), inputs: T('distribution', 'rozdelenie'), outputs: T('token id', 'ID tokenu'), params: null, persistence: T('Advances the RNG state.', 'Posunie stav generátora náhodných čísel.') },
    x_in: { id, title: T('Block input x', 'Vstup bloku x'), role: 'act', what: T('The residual-stream vector entering this block for each token: an activation.', 'Vektor reziduálneho prúdu, ktorý vstupuje do tohto bloku pre každý token: aktivácia.'), where: blockLabel(b), inputs: '', outputs: `[B, T, ${p.d}]`, params: null, persistence: T('Runtime activation.', 'Aktivácia počas behu.') },
    x_out: { id, title: T('Block output y', 'Výstup bloku y'), role: 'act', what: T('The residual-stream vector leaving this block.', 'Vektor reziduálneho prúdu, ktorý z tohto bloku vystupuje.'), where: blockLabel(b), inputs: '', outputs: `[B, T, ${p.d}]`, params: null, persistence: T('Runtime activation.', 'Aktivácia počas behu.') },
    experts: { id, title: T(`${p.moe?.numExperts ?? 0} expert FFNs`, `${p.moe?.numExperts ?? 0} expertných sietí FFN`), role: 'param', what: T('Each expert is an ordinary SwiGLU FFN with its own gate, up and down matrices. An expert is not a separate language model. No human-readable job description is assigned to any expert id: specialization is learned and may be distributed.', 'Každý expert je obyčajná sieť SwiGLU FFN s vlastnými maticami gate, up a down. Expert nie je samostatný jazykový model. Žiadnemu číslu experta nie je priradený ľudsky čitateľný popis úlohy: špecializácia je naučená a môže byť rozptýlená.'), where: blockLabel(b), inputs: T(`token vector [${p.d}] (only for tokens routed to it)`, `vektor tokenu [${p.d}] (len pre tokeny nasmerované k nemu)`), outputs: `[${p.d}]`, params: { count: a.ffnPerBlock, formula: `${p.moe?.numExperts} × 3 × ${p.d} × ${p.moe?.expertWidth}` }, persistence: T('All stored; a few computed per token.', 'Uložení sú všetci; pre token sa počíta len niekoľko.') },
  };
  const c = common[id] ?? ops[id];
  if (c) {
    const src = (c as { sources?: string[] }).sources ?? srcs(p);
    return {
      evidence: (c as { evidence?: InspectorInfo['evidence'] }).evidence ?? [evStruct(p, lang)],
      below: (c as { below?: InspectorInfo['below'] }).below ?? [],
      ...c,
      sources: src,
      above: b !== undefined && (BLOCKISH.has(id) || ops[id]) ? { label: T(`Back to block ${b}`, `Späť na blok ${b}`), href: blockHref(b) } : { label: T('Back to the full model', 'Späť na celý model'), href: explorerHref({ m: p.id, lv: 5 }) },
    } as InspectorInfo;
  }
  if (id.startsWith('expert:')) {
    const e = Number(id.slice(7));
    return {
      id, title: T(`Expert ${e}`, `Expert ${e}`), role: 'param',
      what: T(`Expert ${e} of ${p.moe!.numExperts} in this block: a SwiGLU FFN of width ${p.moe!.expertWidth}. Its id is just an index; no specialization label is claimed.`, `Expert ${e} z ${p.moe!.numExperts} v tomto bloku: sieť SwiGLU FFN so šírkou ${p.moe!.expertWidth}. Jeho číslo je len index; žiadna špecializácia sa netvrdí.`),
      where: `${blockLabel(b)}, mlp.experts.${e}`, inputs: T(`[${p.d}] for routed tokens`, `[${p.d}] pre nasmerované tokeny`), outputs: `[${p.d}]`,
      params: { count: a.oneExpert!, formula: `3 × ${p.d} × ${p.moe!.expertWidth} = ${F(a.oneExpert!)}` },
      persistence: T('Stored whether or not it is selected.', 'Uložený bez ohľadu na to, či je vybraný.'),
      checkpoint: ['gate', 'up', 'down'].map((w) => `model.layers.${b ?? 0}.mlp.experts.${e}.${w}_proj.weight`),
      evidence: [evStruct(p, lang)], sources: srcs(p),
      below: ['expert_gate_proj', 'expert_up_proj', 'expert_down_proj'].map((k) => ({ label: T(`Open ${k.replace('expert_', '')} matrix`, `Otvoriť maticu ${k.replace('expert_', '')}`), href: explorerHref({ m: p.id, lv: 2, b: b ?? 0, t: k, e }) })),
      above: { label: T('Back to the MoE module', 'Späť na modul MoE'), href: explorerHref({ m: p.id, lv: 3, b: b ?? 0, mod: 'moe' }) },
    };
  }
  return {
    id, title: id, role: 'op', what: T('Unknown element.', 'Neznámy prvok.'), where: '', inputs: '', outputs: '', params: null, persistence: '', evidence: [{ kind: 'unknown', note: T('No description available.', 'Popis nie je k dispozícii.') }], sources: [], below: [],
  };
}

const BLOCKISH = new Set(['attention', 'attn_norm', 'ffn_norm', 'residual1', 'residual2', 'ffn', 'moe']);

export const isEvidenceSim = (p: CoreProfile) => V(p) === 'simulation';
