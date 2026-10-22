import type { L10n } from '../i18n/i18n';
import { explorerHref } from '../state/route';
import { QWEN3_8B } from '../models/profiles';
import { accounting } from '../models/shapes';

const A8 = accounting(QWEN3_8B);
const pct = (x: number) => ((x / A8.total) * 100).toFixed(0);
const FFN = pct(A8.ffnPerBlock * QWEN3_8B.layers), ATT = pct(A8.attentionPerBlock * QWEN3_8B.layers), VOC = pct(A8.embedding + A8.head);

export interface TourStep { title: L10n; text: L10n; href: string; cta: L10n }
export interface Tour { id: string; title: L10n; summary: L10n; steps: TourStep[] }
const b = (en: string, sk: string): L10n => ({ en, sk });
const st = (title: L10n, text: L10n, href: string, cta: L10n): TourStep => ({ title, text, href, cta });

export const TOURS: Tour[] = [
  {
    id: 'weights',
    title: b('Where are the weights?', 'Kde sú váhy?'),
    summary: b('From one real stored number in Qwen3-8B out to the 8.19 billion of them.', 'Od jedného skutočného uloženého čísla v Qwen3-8B až po všetkých 8,19 miliardy.'),
    steps: [
      st(b('One stored number', 'Jedno uložené číslo'), b('This is element [0, 0] of the query projection of block 0: a single BF16 value actually read from the checkpoint file, with its exact byte position. A parameter is exactly this: one learned scalar.', 'Toto je prvok [0, 0] projekcie dopytov v bloku 0: jedna hodnota BF16 skutočne prečítaná zo súboru checkpointu, s presnou pozíciou v bajtoch. Parameter je presne toto: jedno naučené číslo.'), explorerHref({ m: 'qwen3-8b', lv: 0, b: 0, t: 'q_proj', r: 0, c: 0 }), b('See the number', 'Pozrieť číslo')),
      st(b('Its row', 'Jeho riadok'), b('Row 0 of W_Q holds the 4,096 weights that produce query coordinate 0. The output is a weighted sum of the 4,096 normalized input features; the toy panel shows each product.', 'Riadok 0 matice W_Q obsahuje 4 096 váh, ktoré vytvárajú súradnicu dopytu 0. Výstup je vážený súčet 4 096 normalizovaných vstupov; hračkový panel ukazuje každý súčin.'), explorerHref({ m: 'qwen3-8b', lv: 1, b: 0, t: 'q_proj', r: 0 }), b('Open the unit', 'Otvoriť jednotku')),
      st(b('Its matrix', 'Jeho matica'), b('W_Q is a [4,096 × 4,096] matrix: 16.8 million parameters, stored once, applied to every token position.', 'W_Q je matica [4 096 × 4 096]: 16,8 milióna parametrov, uložených raz a použitých na každú pozíciu tokenu.'), explorerHref({ m: 'qwen3-8b', lv: 2, b: 0, t: 'q_proj' }), b('Open the matrix', 'Otvoriť maticu')),
      st(b('Its module', 'Jeho modul'), b('Attention owns four matrices (Q, K, V, O) plus two 128-value norm gains. RoPE, the softmax and the head reshapes own no weights.', 'Pozornosť vlastní štyri matice (Q, K, V, O) a dva zisky normalizácie po 128 hodnotách. RoPE, softmax a preskupenie do hláv nemajú žiadne váhy.'), explorerHref({ m: 'qwen3-8b', lv: 3, b: 0, mod: 'attention' }), b('Open attention', 'Otvoriť pozornosť')),
      st(b('Its block', 'Jeho blok'), b('A block adds two RMSNorms and the SwiGLU FFN (three 50-million-parameter matrices). The residual additions have no weights.', 'Blok pridáva dve RMSNorm a doprednú sieť SwiGLU (tri matice po 50 miliónov parametrov). Reziduálne sčítania nemajú žiadne váhy.'), explorerHref({ m: 'qwen3-8b', lv: 4, b: 0 }), b('Open block 0', 'Otvoriť blok 0')),
      st(b('All of them', 'Všetky'), b(`Where the ${A8.total.toLocaleString('en-US')} parameters sit: about ${FFN}% in FFN matrices, ${ATT}% in attention projections, ${VOC}% in the two vocabulary-sized tables, and under 0.01% in norm gains.`, `Kde je ${A8.total.toLocaleString('sk-SK')} parametrov: približne ${FFN} % v maticiach doprednej siete, ${ATT} % v projekciách pozornosti, ${VOC} % v dvoch tabuľkách veľkosti slovníka a menej ako 0,01 % v ziskoch normalizácie.`), '#/parameters?m=qwen3-8b', b('See the treemap', 'Pozrieť mapu')),
    ],
  },
  {
    id: 'neuron',
    title: b('What is a neuron here?', 'Čo je tu neurón?'),
    summary: b('Why “8B parameters” is not 8 billion neurons, and what a SwiGLU channel computes.', 'Prečo „8B parametrov“ nie je 8 miliárd neurónov a čo počíta kanál SwiGLU.'),
    steps: [
      st(b('Weights vs activations', 'Váhy vs aktivácie'), b('Change the input and the outputs change while the six weights stay the same; change a weight and you have changed the model.', 'Zmeňte vstup a výstupy sa zmenia, kým šesť váh zostane rovnakých; zmeňte váhu a zmenili ste model.'), '#/labs/weight', b('Try the tiny example', 'Vyskúšať malý príklad')),
      st(b('A gated channel', 'Hradlovaný kanál'), b('In Qwen3 the FFN “neuron” j uses two rows (gate and up), a SiLU, a multiplication, and a column of W_down. Qwen3-8B has 12,288 such channels per block.', 'V Qwen3 „neurón“ j doprednej siete používa dva riadky (gate a up), SiLU, násobenie a stĺpec matice W_down. Qwen3-8B má 12 288 takých kanálov v každom bloku.'), '#/labs/unit', b('Open the unit microscope', 'Otvoriť mikroskop jednotky')),
      st(b('Many widths', 'Mnoho šírok'), b('There are 4,096 residual features, 32 query heads × 128 coordinates, 8 K/V heads × 128, and 12,288 FFN channels per block. Which of these you call “neurons” is a convention.', 'Existuje 4 096 vlastností reziduálneho prúdu, 32 hláv dopytov × 128 súradníc, 8 hláv K/V × 128 a 12 288 kanálov doprednej siete v bloku. Čo z toho nazvete „neurónmi“, je vec dohody.'), explorerHref({ m: 'qwen3-8b', lv: 4, b: 0 }), b('See the widths in a block', 'Pozrieť šírky v bloku')),
      st(b('No fixed meanings', 'Žiadne pevné významy'), b('Features are often distributed across many coordinates, and this site does not label any unit as “the English neuron”. Attention maps alone are not causal explanations.', 'Vlastnosti sú často rozložené do mnohých súradníc a táto stránka neoznačuje žiadnu jednotku ako „neurón pre angličtinu“. Samotné mapy pozornosti nie sú príčinným vysvetlením.'), '#/learn/ffn', b('Read chapter 8', 'Prečítať 8. kapitolu')),
    ],
  },
  {
    id: 'token',
    title: b('How does one token move?', 'Ako sa pohybuje jeden token?'),
    summary: b('From text to token id to embedding, through every block, to the next token.', 'Od textu cez ID tokenu a embedding, cez každý blok, až k ďalšiemu tokenu.'),
    steps: [
      st(b('Text to ids', 'Text na ID'), b('The chat template adds role markers, then byte-level BPE produces ids. These are real Qwen3 tokenizer outputs.', 'Šablóna rozhovoru pridá označenia rolí, potom byte-level BPE vytvorí ID. Sú to skutočné výstupy tokenizéra Qwen3.'), '#/pipeline', b('See real tokenization', 'Pozrieť skutočnú tokenizáciu')),
      st(b('Embedding row', 'Riadok embeddingu'), b('Each id selects one 4,096-number row of the embedding table. No position vector is added here.', 'Každé ID vyberie jeden riadok tabuľky embeddingov so 4 096 číslami. Nepridáva sa tu žiadny vektor pozície.'), explorerHref({ m: 'qwen3-8b', lv: 3, mod: 'embedding' }), b('Open the embedding', 'Otvoriť embedding')),
      st(b('Through the blocks', 'Cez bloky'), b('Walk a toy token through each block and see the exact vectors added at the two residual connections.', 'Preveďte hračkový token každým blokom a pozrite si presné vektory pripočítané v dvoch reziduálnych spojeniach.'), '#/labs/layers', b('Layer traversal', 'Prechod vrstvami')),
      st(b('Attention reads the past', 'Pozornosť číta minulosť'), b('The query compares with cached keys of earlier positions; future positions have probability exactly zero.', 'Dopyt sa porovná s uloženými kľúčmi skorších pozícií; budúce pozície majú pravdepodobnosť presne nula.'), '#/labs/attention', b('Attention lab', 'Laboratórium pozornosti')),
      st(b('Prefill and decode', 'Prefill a decode'), b('Prefill processes the prompt block by block; decoding then adds one token at a time using the K/V cache, and the result equals recomputation.', 'Prefill spracuje zadanie blok po bloku; dekódovanie potom pridáva po jednom tokene s využitím K/V cache a výsledok sa rovná úplnému prepočtu.'), '#/labs/cache', b('Cache lab', 'Laboratórium cache')),
      st(b('Choosing the next token', 'Výber ďalšieho tokenu'), b('Logits → temperature → top-k → top-p → renormalize → seeded draw → append → repeat.', 'Logity → teplota → top-k → top-p → prepočet → náhodný výber → pripojenie → opakovanie.'), '#/labs/sampling', b('Sampling lab', 'Laboratórium vzorkovania')),
    ],
  },
  {
    id: 'moe',
    title: b('What changes in MoE?', 'Čo sa mení v MoE?'),
    summary: b('Qwen3-30B-A3B: same attention, 128 expert FFNs per block, 8 used per token.', 'Qwen3-30B-A3B: rovnaká pozornosť, 128 expertov v každom bloku, 8 použitých na token.'),
    steps: [
      st(b('Same block, different FFN', 'Rovnaký blok, iná FFN'), b('Attention, norms and residuals are as in the dense model; the FFN sub-layer is replaced by a router and 128 experts.', 'Pozornosť, normalizácie a reziduálne spojenia sú ako v hustom modeli; podvrstvu FFN nahrádza smerovač a 128 expertov.'), explorerHref({ m: 'qwen3-30b-a3b', lv: 4, b: 0 }), b('Open a MoE block', 'Otvoriť blok MoE')),
      st(b('The router', 'Smerovač'), b('All 128 router logits are computed for every token in every block; softmax, keep the top 8, renormalize to α.', 'Všetkých 128 logitov smerovača sa vypočíta pre každý token v každom bloku; softmax, ponechať 8 najlepších, prepočítať na α.'), '#/labs/router', b('Router lab', 'Laboratórium smerovača')),
      st(b('An expert is an FFN', 'Expert je FFN'), b('Each expert is an ordinary SwiGLU of width 768 with its own three matrices. No job description is attached to any expert id.', 'Každý expert je obyčajná sieť SwiGLU so šírkou 768 a vlastnými tromi maticami. Žiadnemu číslu experta nie je priradené pracovné zameranie.'), explorerHref({ m: 'qwen3-30b-a3b', lv: 3, b: 0, mod: 'moe' }), b('See all 128', 'Pozrieť všetkých 128')),
      st(b('Stored vs used', 'Uložené vs použité'), b('30.5B parameters stored, about 3.3B “activated” per token: a convention, not the number of scalars read or FLOPs.', '30,5 mld. uložených parametrov, približne 3,3 mld. „aktivovaných“ na token: spôsob počítania, nie počet prečítaných čísel ani operácií.'), '#/parameters?m=qwen3-30b-a3b', b('The accounting', 'Účtovanie')),
      st(b('A fair comparison', 'Férové porovnanie'), b('The two reference models differ in width and depth too; the dimension-matched toy comparison isolates the FFN change.', 'Dva referenčné modely sa líšia aj šírkou a hĺbkou; hračkové porovnanie s rovnakými rozmermi izoluje len zmenu FFN.'), '#/labs/compare', b('Dense vs MoE', 'Hustý vs MoE')),
    ],
  },
];
