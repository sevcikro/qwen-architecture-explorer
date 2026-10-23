import type { L10n } from '../i18n/i18n';

const g = (enT: string, en: string, skT: string, sk: string) => ({ term: { en: enT, sk: skT } as L10n, def: { en, sk } as L10n });

export const GLOSSARY: { term: L10n; def: L10n }[] = [
  g('Activation', 'A value computed for a particular input during a forward pass (e.g. a normalized vector, a query, a gated channel). It changes when the input changes even though the weights stay fixed.',
    'Aktivácia', 'Hodnota vypočítaná pre konkrétny vstup počas priechodu modelom (napr. normalizovaný vektor, dopyt, hradlovaný kanál). Mení sa so vstupom, hoci váhy zostávajú rovnaké.'),
  g('Attention weights (probabilities)', 'Computed mixing coefficients softmax(QKᵀ/√d)·mask for one input. Not learned parameters; the learned matrices are W_Q, W_K, W_V and W_O.',
    'Váhy pozornosti (pravdepodobnosti)', 'Vypočítané koeficienty miešania softmax(QKᵀ/√d)·maska pre jeden vstup. Nie sú to naučené parametre; naučené sú matice W_Q, W_K, W_V a W_O.'),
  g('BF16', 'bfloat16: a 16-bit floating-point format with float32’s 8-bit exponent and 7 fraction bits. The published Qwen3 checkpoints store weights in BF16.',
    'BF16', 'bfloat16: 16-bitový formát s pohyblivou desatinnou čiarkou s 8-bitovým exponentom ako float32 a 7 bitmi zlomku. Zverejnené checkpointy Qwen3 ukladajú váhy v BF16.'),
  g('Block (decoder block, “layer”)', 'One repeated unit: RMSNorm → attention → residual add → RMSNorm → FFN or MoE → residual add. “36 layers” means 36 blocks; the embedding, final norm and head are separate components.',
    'Blok (blok dekodéra, „vrstva“)', 'Jedna opakujúca sa jednotka: RMSNorm → pozornosť → reziduálne sčítanie → RMSNorm → FFN alebo MoE → reziduálne sčítanie. „36 vrstiev“ znamená 36 blokov; embedding, záverečná normalizácia a výstupná hlava sú samostatné časti.'),
  g('Causal mask', 'Prevents a position from attending to later positions; their probabilities are exactly zero.',
    'Kauzálna maska', 'Zabraňuje pozícii pozerať sa na neskoršie pozície; ich pravdepodobnosti sú presne nula.'),
  g('Checkpoint / shard', 'The saved parameter files. Qwen3-8B uses 5 safetensors shards; each tensor is stored whole in one shard, and an index file maps names to shards.',
    'Checkpoint / shard', 'Uložené súbory s parametrami. Qwen3-8B používa 5 súborov safetensors (shardov); každý tenzor je celý uložený v jednom z nich a indexový súbor priraďuje názvy k súborom.'),
  g('Embedding', 'The table that turns a token id into its starting vector (one learned row per token).',
    'Embedding', 'Tabuľka, ktorá premení ID tokenu na jeho počiatočný vektor (jeden naučený riadok na token).'),
  g('Expert', 'In a MoE block, one of the alternative FFNs (here SwiGLU of width 768). Not a separate model or chatbot; its specialization, if any, is learned and not labeled here.',
    'Expert', 'V bloku MoE jedna z alternatívnych dopredných sietí (tu SwiGLU so šírkou 768). Nie je to samostatný model ani chatbot; jeho prípadná špecializácia je naučená a tu sa neoznačuje.'),
  g('FFN / MLP', 'The position-wise feed-forward network in each block. In Qwen3 it is SwiGLU: down(SiLU(gate·x) ⊙ up·x).',
    'FFN / MLP (dopredná sieť)', 'Dopredná sieť v každom bloku, použitá na každý token zvlášť. V Qwen3 je to SwiGLU: down(SiLU(gate·x) ⊙ up·x).'),
  g('GQA (grouped-query attention)', 'Several query heads share one key/value head. Qwen3-8B: 32 query heads, 8 K/V heads (groups of 4). Qwen3-30B-A3B: 32 and 4 (groups of 8).',
    'GQA (pozornosť so skupinovými dopytmi)', 'Viac hláv dopytov zdieľa jednu hlavu kľúčov a hodnôt. Qwen3-8B: 32 hláv dopytov, 8 hláv K/V (skupiny po 4). Qwen3-30B-A3B: 32 a 4 (skupiny po 8).'),
  g('Gated DeltaNet', 'A linear-attention token mixer used by Qwen3-Next, Qwen3.5 and Qwen3.8-Flash-Next: a fixed-size matrix state that decays and is updated with a delta-rule write.',
    'Gated DeltaNet', 'Mixér tokenov s lineárnou pozornosťou v modeloch Qwen3-Next, Qwen3.5 a Qwen3.8-Flash-Next: maticový stav pevnej veľkosti, ktorý postupne slabne a aktualizuje sa zápisom podľa delta pravidla.'),
  g('Gradient', 'For each weight, the direction and strength in which a small change would increase the loss; training moves the weight the opposite way.',
    'Gradient', 'Pre každú váhu smer a sila, ktorou by malá zmena zvýšila chybu; trénovanie posúva váhu opačným smerom.'),
  g('Head (attention head)', 'An independent attention computation over a 128-dimensional slice (head_dim). Head outputs are concatenated and mixed by W_O.',
    'Hlava (hlava pozornosti)', 'Samostatný výpočet pozornosti na 128-rozmernom úseku (head_dim). Výstupy hláv sa spoja a zmieša ich matica W_O.'),
  g('K/V cache', 'Runtime storage of past keys and values per block so decoding does not recompute them. Computed state, not trained parameters.',
    'K/V cache', 'Úložisko kľúčov a hodnôt minulých tokenov v každom bloku počas behu, aby sa pri dekódovaní nemuseli počítať znova. Vypočítaný stav, nie natrénované parametre.'),
  g('Logit', 'An unnormalized score; the output head produces one per vocabulary row. Softmax turns logits into probabilities.',
    'Logit', 'Nenormalizované skóre; výstupná hlava vytvorí jedno pre každý riadok slovníka. Softmax premení logity na pravdepodobnosti.'),
  g('Loss', 'A number measuring how wrong the predictions are during training (here: cross-entropy, −log of the probability of the correct next token).',
    'Chyba (loss)', 'Číslo, ktoré počas trénovania meria, ako veľmi sú predpovede nesprávne (tu: krížová entropia, −log pravdepodobnosti správneho ďalšieho tokenu).'),
  g('MoE (mixture of experts)', 'A block whose FFN is replaced by a router plus many experts, of which a few are used per token.',
    'MoE (zmes expertov)', 'Blok, ktorého doprednú sieť nahrádza smerovač a mnoho expertov, z ktorých sa pre každý token použije len niekoľko.'),
  g('Neuron / unit / channel', 'A computational coordinate (one output of a weighted sum, or one gated FFN channel). Transformers have several different widths, so there is no single convention-free neuron count.',
    'Neurón / jednotka / kanál', 'Výpočtová súradnica (jeden výstup váženého súčtu alebo jeden hradlovaný kanál FFN). Transformery majú viac rôznych šírok, preto neexistuje jediný počet neurónov nezávislý od spôsobu počítania.'),
  g('Parameter', 'One learned scalar stored in the checkpoint. Most are entries of weight matrices or embedding tables; normalization gains and router matrices are parameters too.',
    'Parameter', 'Jedno naučené číslo uložené v checkpointe. Väčšina sú prvky matíc váh alebo tabuliek embeddingov; parametrami sú aj zisky normalizácie a matice smerovača.'),
  g('Prefill / decode', 'Prefill processes the whole prompt (positions in parallel inside each block); decode then generates one token at a time using the cache.',
    'Prefill / decode', 'Prefill spracuje celé zadanie (pozície súčasne v každom bloku); decode potom generuje po jednom tokene s využitím cache.'),
  g('Residual connection', 'Adds a sub-layer’s output to its input (x + f(x)). The addition has no learned weights.',
    'Reziduálne spojenie', 'Pripočíta výstup podvrstvy k jej vstupu (x + f(x)). Samotné sčítanie nemá naučené váhy.'),
  g('RMSNorm', 'Divides a vector by its root-mean-square (plus ε) and multiplies by a learned gain γ. Unlike LayerNorm it does not subtract the mean.',
    'RMSNorm', 'Vydelí vektor jeho strednou kvadratickou hodnotou (plus ε) a vynásobí ho naučeným ziskom γ. Na rozdiel od LayerNorm neodčítava priemer.'),
  g('RoPE', 'Rotary position embedding: rotates pairs of query/key coordinates by position-dependent angles inside attention. No learned weights; V is not rotated.',
    'RoPE', 'Rotačné kódovanie pozície: v rámci pozornosti otáča dvojice súradníc dopytov a kľúčov o uhly závislé od pozície. Bez naučených váh; hodnoty V sa neotáčajú.'),
  g('Router / routing weights', '“Router weights” can mean the learned matrix W_router [experts × d] or the per-token mixture coefficients α. This site always says which.',
    'Smerovač / váhy smerovania', '„Váhy smerovača“ môžu znamenať naučenú maticu W_router [experti × d] alebo koeficienty zmesi α pre konkrétny token. Táto stránka vždy uvádza, ktoré z toho.'),
  g('Sampling', 'The decoding procedure that picks the next token from logits (temperature, top-k, top-p, random draw). Not a neural-network layer.',
    'Vzorkovanie (sampling)', 'Postup, ktorý z logitov vyberie ďalší token (teplota, top-k, top-p, náhodný výber). Nie je to vrstva neurónovej siete.'),
  g('Token', 'A tokenizer unit (byte-level BPE piece), not necessarily a word or character.',
    'Token', 'Jednotka tokenizéra (kúsok byte-level BPE), nie nevyhnutne slovo alebo znak.'),
  g('Toy simulator', 'This site’s tiny, seeded model with Qwen3’s structure (d = 8, 2 blocks). Its values are illustrative simulations, never observed Qwen behavior.',
    'Hračkový simulátor', 'Malý model tejto stránky so štruktúrou Qwen3 (d = 8, 2 bloky) a váhami z osiva. Jeho hodnoty sú ilustračné simulácie, nikdy nie pozorované správanie modelu Qwen.'),
  g('Untied embeddings', 'The input embedding table and the output head are separate matrices (tie_word_embeddings = false for both reference models).',
    'Neviazané embeddingy', 'Vstupná tabuľka embeddingov a výstupná hlava sú samostatné matice (tie_word_embeddings = false pre oba referenčné modely).'),
  g('Context window', 'The number of tokens a model can take into account at once: prompt and generated answer together. Qwen3: 32,768 natively, 131,072 with YaRN.',
    'Kontextové okno', 'Počet tokenov, ktoré model dokáže naraz zohľadniť: zadanie aj vygenerovaná odpoveď spolu. Qwen3: 32 768 natívne, 131 072 s YaRN.'),
  g('Transformer', 'A neural-network architecture built from attention and per-token feed-forward layers with residual connections. Qwen uses its decoder-only form.',
    'Transformer', 'Architektúra neurónovej siete z vrstiev pozornosti a doprednej siete pre každý token, s reziduálnymi spojeniami. Qwen používa jej podobu iba s dekodérom.'),
  g('Weight', 'A learned number that multiplies an input inside the model; “weight” and “parameter” are used almost interchangeably.',
    'Váha', 'Naučené číslo, ktoré v modeli násobí vstup; „váha“ a „parameter“ sa používajú takmer zameniteľne.'),
  g('YaRN', 'A RoPE scaling method the model cards describe for extending context from 32,768 to 131,072 tokens (factor 4). Opt-in via rope_scaling; not enabled in the published config.',
    'YaRN', 'Metóda škálovania RoPE, ktorú karty modelov opisujú na rozšírenie kontextu z 32 768 na 131 072 tokenov (faktor 4). Zapína sa cez rope_scaling; v zverejnenej konfigurácii nie je zapnutá.'),
];

// Word patterns used to link the first mention of a term in a chapter to its definition.
// Unicode-aware; Slovak patterns cover the usual inflected forms.
const W = (s: string) => s;
export const GLOSS_MATCH: Record<string, { en: string; sk: string }> = {
  'Activation': { en: W('activations?'), sk: W('aktivác\\p{L}*') },
  'Attention weights (probabilities)': { en: W('attention weights'), sk: W('váh\\p{L}* pozornosti') },
  'BF16': { en: 'BF16', sk: 'BF16' },
  'Block (decoder block, “layer”)': { en: 'decoder blocks?', sk: 'blok\\p{L}* dekodéra' },
  'Causal mask': { en: 'causal mask', sk: 'kauzáln\\p{L}* mask\\p{L}*' },
  'Checkpoint / shard': { en: 'checkpoints?', sk: 'checkpoint\\p{L}*' },
  'Embedding': { en: 'embeddings?', sk: 'embedding\\p{L}*' },
  'Expert': { en: 'experts?', sk: 'expert(?:i|a|ov|om|mi|och|ovi)?' },
  'FFN / MLP': { en: 'feed-forward network|FFN', sk: 'dopredn\\p{L}* sie[tť]\\p{L}*|FFN' },
  'GQA (grouped-query attention)': { en: 'GQA|grouped-query attention', sk: 'GQA' },
  'Gated DeltaNet': { en: 'Gated DeltaNet', sk: 'Gated DeltaNet' },
  'Gradient': { en: 'gradients?', sk: 'gradient(?:y|ov|om|mi|u|e|och)?' },
  'Head (attention head)': { en: 'attention heads?|query heads?', sk: 'hlav\\p{L}* pozornosti|hlav\\p{L}* dopytov' },
  'K/V cache': { en: 'K/V cache', sk: 'cache K/V' },
  'Logit': { en: 'logits?', sk: 'logit(?:y|ov|om|mi|u|e|och)?' },
  'Loss': { en: 'loss', sk: 'chyb\\p{L}* pri trénovaní' },
  'MoE (mixture of experts)': { en: 'MoE|mixture of experts', sk: 'MoE|zmes\\p{L}* expertov' },
  'Neuron / unit / channel': { en: 'neurons?', sk: 'neur[oó]n(?:y|ov|om|mi|u|e|och|a)?' },
  'Parameter': { en: 'parameters?', sk: 'parametr(?:y|ov|om|mi|u|e|och|a)|parameter' },
  'Prefill / decode': { en: 'prefill', sk: 'predvyplnen\\p{L}*' },
  'Residual connection': { en: 'residual (?:connection|stream|add(?:ition)?s?)', sk: 'reziduáln\\p{L}* (?:spojen\\p{L}*|prúd\\p{L}*|sčítan\\p{L}*)' },
  'RMSNorm': { en: 'RMSNorm', sk: 'RMSNorm' },
  'RoPE': { en: 'RoPE', sk: 'RoPE' },
  'Router / routing weights': { en: 'routers?', sk: 'smerovač(?:a|e|om|i|ov|mi)?' },
  'Sampling': { en: 'sampling', sk: 'vzorkovan\\p{L}*' },
  'Token': { en: 'tokens?', sk: 'token(?:y|ov|om|mi|u|e|och|a)?' },
  'Toy simulator': { en: 'toy simulator', sk: 'hračkov\\p{L}* simulátor\\p{L}*' },
  'Untied embeddings': { en: 'untied', sk: 'neprepojen\\p{L}*|neviazan\\p{L}*' },
  'Weight': { en: 'weights?', sk: 'váh(?:a|y|u|ou|ami|ách)?' },
  'Context window': { en: 'context window', sk: 'kontextov\\p{L}* okn\\p{L}*' },
  'Transformer': { en: 'transformers?', sk: 'transformer(?:y|a|ov|om|mi|i|och|e)?' },
  'YaRN': { en: 'YaRN', sk: 'YaRN' },
};

export interface GlossHit { term: string; start: number; end: number }

const compiled = new Map<string, RegExp>();
const rx = (term: string, lang: 'en' | 'sk') => {
  const k = `${term}|${lang}`;
  let r = compiled.get(k);
  if (!r) { r = new RegExp(String.raw`(?<![\p{L}\p{N}_/-])(?:` + GLOSS_MATCH[term][lang] + String.raw`)(?![\p{L}\p{N}_/-])`, 'iu'); compiled.set(k, r); }
  return r;
};

/** First match of each wanted term in a plain text, as non-overlapping ranges (earliest first). */
export function findTerms(text: string, terms: string[], lang: 'en' | 'sk'): GlossHit[] {
  const hits: GlossHit[] = [];
  for (const t of terms) {
    const m = rx(t, lang).exec(text);
    if (m) hits.push({ term: t, start: m.index, end: m.index + m[0].length });
  }
  hits.sort((a, b) => a.start - b.start);
  const out: GlossHit[] = [];
  for (const h of hits) if (!out.length || h.start >= out[out.length - 1].end) out.push(h);
  return out;
}

/** Does this text mention the term? (used to assign each term to its first block in a chapter) */
export const mentions = (text: string, term: string, lang: 'en' | 'sk') => rx(term, lang).test(text);
