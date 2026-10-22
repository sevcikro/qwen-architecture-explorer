import type { L10n } from '../i18n/i18n';
import { QWEN3_30B_A3B, QWEN3_8B } from '../models/profiles';
import { accounting } from '../models/shapes';
import { kvBytesPerToken } from '../math/memory';
import { explorerHref } from '../state/route';

// ---------------------------------------------------------------------------
// The "Learn" course: from one weight and one neuron to the whole model.
// Every Qwen number is computed from the shared, verified accounting (never typed by hand).
// ---------------------------------------------------------------------------

export type WidgetId =
  | 'next-token' | 'real-weight' | 'neuron' | 'silu' | 'matvec' | 'tokens' | 'rms' | 'attention'
  | 'gated' | 'router' | 'stack' | 'sampling' | 'train1d' | 'memory';

export type Block =
  | { k: 'p'; t: L10n }
  | { k: 'h'; t: L10n }
  | { k: 'key'; t: L10n }
  | { k: 'analogy'; t: L10n }
  | { k: 'qwen'; t: L10n }
  | { k: 'list'; items: L10n[] }
  | { k: 'math'; tex: string; t?: L10n }
  | { k: 'widget'; w: WidgetId }
  | { k: 'quiz'; q: L10n; options: L10n[]; correct: number; explain: L10n }
  | { k: 'links'; items: { href: string; label: L10n }[] };

export interface Chapter { id: string; n: number; title: L10n; summary: L10n; minutes: number; blocks: Block[] }

const b = (en: string, sk: string): L10n => ({ en, sk });
const A8 = accounting(QWEN3_8B);
const A30 = accounting(QWEN3_30B_A3B);
const E = (n: number) => n.toLocaleString('en-US');
const S = (n: number) => n.toLocaleString('sk-SK');
const EB = (n: number, d = 1) => `${(n / 1e9).toFixed(d)}`;
const SB = (n: number, d = 1) => (n / 1e9).toFixed(d).replace('.', ',');
const L8 = QWEN3_8B.layers;
const FFN_SHARE = Math.round(((A8.ffnPerBlock * L8) / A8.total) * 100);
const ATT_SHARE = Math.round(((A8.attentionPerBlock * L8) / A8.total) * 100);
const KV8 = kvBytesPerToken(QWEN3_8B) / 1024;

export const CHAPTERS: Chapter[] = [
  {
    id: 'what', n: 1, minutes: 4,
    title: b('What a language model actually does', 'Čo jazykový model v skutočnosti robí'),
    summary: b('One step repeated: guess how likely each next piece of text is, pick one, repeat.', 'Jeden krok stále dookola: odhadnúť, aký pravdepodobný je každý ďalší kúsok textu, jeden vybrať a opakovať.'),
    blocks: [
      { k: 'p', t: b('A language model such as Qwen does one thing over and over. Given the text so far, it computes **how likely every possible next piece of text is**, picks one, appends it, and repeats. Answering questions, translating or writing code all emerge from repeating this single step many times.', 'Jazykový model, napríklad Qwen, robí stále dookola jednu vec. Z doterajšieho textu vypočíta, **aký pravdepodobný je každý možný ďalší kúsok textu**, jeden vyberie, pripojí ho a pokračuje. Odpovedanie na otázky, prekladanie či písanie kódu vzniká opakovaním tohto jediného kroku.') },
      { k: 'p', t: b('A computer cannot calculate with words directly. So the text is first cut into small pieces called **tokens**, and every token is turned into a list of numbers. The model itself is a long, fixed chain of multiplications and additions on such lists. At the very end it produces one score for every entry of its vocabulary, and those scores are turned into probabilities.', 'Počítač nevie priamo počítať so slovami. Text sa preto najprv rozdelí na malé kúsky nazývané **tokeny** a každý token sa premení na zoznam čísel. Samotný model je dlhý, pevne daný reťazec násobení a sčítaní nad takýmito zoznamami. Na konci vytvorí jedno skóre pre každú položku svojho slovníka a z týchto skóre sa vypočítajú pravdepodobnosti.') },
      { k: 'widget', w: 'next-token' },
      { k: 'key', t: b('A model is a fixed recipe of arithmetic. Its ingredients are billions of stored numbers called **weights** or **parameters**. They were set during training. While you chat with the model, they do not change.', 'Model je pevný recept na počítanie. Jeho ingredienciami sú miliardy uložených čísel nazývaných **váhy** alebo **parametre**. Nastavili sa počas trénovania. Keď sa s modelom rozprávate, nemenia sa.') },
      { k: 'analogy', t: b('It resembles the word suggestions on a phone keyboard, but vastly larger: it takes the whole conversation into account and its “suggestion rules” are billions of numbers learned from text.', 'Pripomína to návrhy slov na klávesnici v mobile, len v obrovskom meradle: berie do úvahy celý rozhovor a jeho „pravidlá návrhov“ tvoria miliardy čísel naučených z textov.') },
      { k: 'qwen', t: b(`Qwen3-8B performs this step with **${E(A8.total)}** stored numbers, organized in ${L8} repeated blocks. Its vocabulary has ${E(QWEN3_8B.vocabRows)} entries, so every step ends with ${E(QWEN3_8B.vocabRows)} scores.`, `Qwen3-8B vykonáva tento krok s **${S(A8.total)}** uloženými číslami usporiadanými do ${L8} opakujúcich sa blokov. Jeho slovník má ${S(QWEN3_8B.vocabRows)} položiek, preto každý krok končí ${S(QWEN3_8B.vocabRows)} skóre.`) },
      { k: 'quiz', q: b('When you send a message to the model, do its weights change?', 'Zmenia sa váhy modelu, keď mu pošlete správu?'), options: [b('Yes, it learns from every message', 'Áno, učí sa z každej správy'), b('No, the weights stay fixed; only temporary computed values change', 'Nie, váhy zostávajú rovnaké; menia sa len dočasne vypočítané hodnoty'), b('Only the first weight changes', 'Zmení sa len prvá váha')], correct: 1, explain: b('Using a model (inference) only reads the weights. They change only during training, which is a separate, expensive process.', 'Používanie modelu (inferencia) váhy iba číta. Menia sa len počas trénovania, čo je samostatný a nákladný proces.') },
      { k: 'links', items: [{ href: explorerHref({ m: 'qwen3-8b', lv: 5 }), label: b('See the whole Qwen3-8B model', 'Pozrite si celý model Qwen3-8B') }, { href: '#/pipeline', label: b('Real tokens produced by the Qwen tokenizer', 'Skutočné tokeny z tokenizéra Qwen') }] },
    ],
  },
  {
    id: 'weight', n: 2, minutes: 4,
    title: b('One stored number: a weight', 'Jedno uložené číslo: váha'),
    summary: b('The smallest piece of the model is a single number in a file.', 'Najmenšou súčasťou modelu je jedno číslo v súbore.'),
    blocks: [
      { k: 'p', t: b('The smallest piece of a model is a single number, for example **0.000568389892578125**. It is called a **parameter** or **weight**. On its own it means nothing. It matters only together with millions of other numbers it is combined with.', 'Najmenšou súčasťou modelu je jedno číslo, napríklad **0,000568389892578125**. Nazýva sa **parameter** alebo **váha**. Samo osebe nič neznamená. Význam má len spolu s miliónmi ďalších čísel, s ktorými sa kombinuje.') },
      { k: 'widget', w: 'real-weight' },
      { k: 'p', t: b('Qwen stores each weight in a 16-bit format called **BF16**: 1 bit for the sign (plus or minus), 8 bits for the exponent (how large), and 7 bits for the digits. That is only about 2–3 significant decimal digits, but it is enough for neural networks and needs half the memory of ordinary 32-bit numbers.', 'Qwen ukladá každú váhu v 16-bitovom formáte **BF16**: 1 bit pre znamienko (plus alebo mínus), 8 bitov pre exponent (aké veľké číslo je) a 7 bitov pre číslice. To sú len asi 2 až 3 platné desatinné číslice, no pre neurónové siete to stačí a zaberá to polovicu pamäte bežných 32-bitových čísel.') },
      { k: 'p', t: b('Where does such a number live? In a file on disk called a **checkpoint**, at an exact byte position. When the model runs, the file is loaded into memory (RAM or the memory of a graphics card).', 'Kde také číslo „býva“? V súbore na disku, ktorý sa nazýva **checkpoint**, na presnej pozícii v bajtoch. Keď model beží, súbor sa načíta do pamäte (RAM alebo pamäte grafickej karty).') },
      { k: 'key', t: b('**Parameter** = a learned number, stored permanently. **Activation** = a temporary number computed while the model processes one particular input. Parameters are the same for every question; activations differ for every question.', '**Parameter** = naučené číslo, uložené natrvalo. **Aktivácia** = dočasné číslo vypočítané počas spracovania konkrétneho vstupu. Parametre sú rovnaké pre každú otázku; aktivácie sú pre každú otázku iné.') },
      { k: 'qwen', t: b(`Qwen3-8B has ${E(A8.total)} such numbers. At 2 bytes each that is about ${EB(A8.total * 2)} GB of weights.`, `Qwen3-8B má ${S(A8.total)} takýchto čísel. Pri 2 bajtoch na číslo je to približne ${SB(A8.total * 2)} GB váh.`) },
      { k: 'quiz', q: b('What does the “8B” in Qwen3-8B count?', 'Čo počíta označenie „8B“ v názve Qwen3-8B?'), options: [b('About 8 billion neurons', 'Približne 8 miliárd neurónov'), b('About 8 billion stored numbers (parameters)', 'Približne 8 miliárd uložených čísel (parametrov)'), b('8 billion words it knows', '8 miliárd slov, ktoré pozná')], correct: 1, explain: b(`“B” means billion parameters: ${E(A8.total)} learned numbers, rounded. Neurons are a different, smaller quantity (chapter 8).`, `„B“ znamená miliardy parametrov: ${S(A8.total)} naučených čísel po zaokrúhlení. Neuróny sú iná, menšia veličina (kapitola 8).`) },
      { k: 'links', items: [{ href: explorerHref({ m: 'qwen3-8b', lv: 0, b: 0, t: 'q_proj', r: 0, c: 0 }), label: b('Open this exact weight in the explorer', 'Otvoriť túto konkrétnu váhu v prieskumníku') }] },
    ],
  },
  {
    id: 'neuron', n: 3, minutes: 6,
    title: b('A single neuron: a weighted sum', 'Jeden neurón: vážený súčet'),
    summary: b('Multiply each input by its weight, add everything up, pass it through a simple function.', 'Každý vstup vynásobiť jeho váhou, všetko sčítať a výsledok poslať cez jednoduchú funkciu.'),
    blocks: [
      { k: 'p', t: b('A **neuron** in an artificial network is a tiny calculation. It takes several input numbers, multiplies each one by its own weight, and adds up the products. The result is then passed through a simple function called an **activation function**.', '**Neurón** v umelej sieti je malý výpočet. Vezme niekoľko vstupných čísel, každé vynásobí jeho vlastnou váhou a súčiny sčíta. Výsledok potom prejde jednoduchou funkciou, ktorá sa nazýva **aktivačná funkcia**.') },
      { k: 'math', tex: 'y = f\\big(w_1 x_1 + w_2 x_2 + w_3 x_3\\big)', t: b('x = inputs (activations), w = weights (parameters), f = activation function', 'x = vstupy (aktivácie), w = váhy (parametre), f = aktivačná funkcia') },
      { k: 'widget', w: 'neuron' },
      { k: 'p', t: b('The weights decide how much each input matters. A large positive weight lets an input push the output up. A negative weight pushes it down. A weight near zero means the input is almost ignored.', 'Váhy rozhodujú, nakoľko na každom vstupe záleží. Veľká kladná váha umožní vstupu zvýšiť výstup. Záporná váha ho znižuje. Váha blízka nule znamená, že vstup sa takmer ignoruje.') },
      { k: 'h', t: b('Why an activation function?', 'Prečo aktivačná funkcia?') },
      { k: 'p', t: b('If neurons only multiplied and added, stacking many of them would still be equivalent to one single multiplication: nothing more expressive would appear. A bent (nonlinear) function in between makes depth useful. Qwen uses **SiLU**: SiLU(a) = a · σ(a). It is close to 0 for negative inputs, close to the input itself for large positive ones, and smooth in between.', 'Keby neuróny len násobili a sčítavali, aj ich veľké množstvo by bolo rovnocenné jedinému násobeniu: nevzniklo by nič schopnejšie. Ohnutá (nelineárna) funkcia medzi nimi robí hĺbku užitočnou. Qwen používa **SiLU**: SiLU(a) = a · σ(a). Pre záporné vstupy je blízko 0, pre veľké kladné takmer rovná vstupu a medzi tým sa mení plynulo.') },
      { k: 'widget', w: 'silu' },
      { k: 'analogy', t: b('Think of a judge weighing evidence: each piece of evidence (input) gets an importance (weight), the judge adds up the weighted evidence, and only a convincing total leads to a strong verdict (activation).', 'Predstavte si sudcu, ktorý zvažuje dôkazy: každý dôkaz (vstup) má svoju dôležitosť (váhu), sudca vážené dôkazy sčíta a len presvedčivý súčet vedie k silnému verdiktu (aktivácii).') },
      { k: 'qwen', t: b('Textbook neurons often add an extra constant called a **bias**. Qwen3’s projections have **no bias**: every output is purely a weighted sum.', 'Učebnicové neuróny často pridávajú ešte konštantu nazývanú **bias** (posun). Projekcie v Qwen3 **bias nemajú**: každý výstup je čistý vážený súčet.') },
      { k: 'quiz', q: b('A weight is exactly 0. What happens to its input?', 'Váha je presne 0. Čo sa stane s jej vstupom?'), options: [b('It is doubled', 'Zdvojnásobí sa'), b('It has no effect on this neuron’s sum', 'Na súčet tohto neurónu nemá žiadny vplyv'), b('The neuron stops working', 'Neurón prestane fungovať')], correct: 1, explain: b('0 × anything = 0, so that input contributes nothing to this particular sum (it can still matter for other neurons).', '0 × čokoľvek = 0, takže tento vstup do tohto súčtu nič neprispeje (pre iné neuróny však môže byť dôležitý).') },
      { k: 'links', items: [{ href: '#/labs/weight', label: b('Edit a weight and watch the output change', 'Upravte váhu a sledujte zmenu výstupu') }, { href: '#/labs/unit', label: b('Every product in a real gated channel', 'Každý súčin v skutočnom hradlovanom kanáli') }] },
    ],
  },
  {
    id: 'matrix', n: 4, minutes: 5,
    title: b('Many neurons at once: matrices', 'Veľa neurónov naraz: matice'),
    summary: b('A layer of neurons is a table of weights; computing it is matrix multiplication.', 'Vrstva neurónov je tabuľka váh; jej výpočet je násobenie matíc.'),
    blocks: [
      { k: 'p', t: b('A layer has many neurons and all of them read the same input numbers. Write each neuron’s weights as one row and you get a table of numbers: a **matrix**. Computing all neurons of the layer at once is then a **matrix-vector multiplication**: every row is multiplied with the input and summed.', 'Vrstva má veľa neurónov a všetky čítajú tie isté vstupné čísla. Ak váhy každého neurónu zapíšete ako jeden riadok, dostanete tabuľku čísel: **maticu**. Výpočet všetkých neurónov vrstvy naraz je potom **násobenie matice vektorom**: každý riadok sa vynásobí so vstupom a sčíta.') },
      { k: 'widget', w: 'matvec' },
      { k: 'p', t: b('A matrix is described by its shape **[outputs, inputs]**: the number of rows (neurons) times the number of columns (inputs). The number of weights it holds is simply rows × columns.', 'Maticu opisuje jej tvar **[výstupy, vstupy]**: počet riadkov (neurónov) krát počet stĺpcov (vstupov). Počet váh, ktoré obsahuje, je jednoducho riadky × stĺpce.') },
      { k: 'p', t: b('The same matrix is used for **every token** in the text. The weights are stored once; for a sentence of 20 tokens the multiplication simply happens 20 times (in parallel on a graphics card).', 'Tá istá matica sa používa pre **každý token** v texte. Váhy sú uložené raz; pre vetu s 20 tokenmi sa násobenie jednoducho vykoná 20-krát (na grafickej karte súčasne).') },
      { k: 'key', t: b('Almost all parameters of a language model live in matrices like these.', 'Takmer všetky parametre jazykového modelu sa nachádzajú v takýchto maticiach.') },
      { k: 'qwen', t: b(`In Qwen3-8B the query matrix W_Q has shape [4,096 × 4,096] = ${E(A8.qkvoPerBlock.q)} weights. Each of the ${L8} blocks has 4 attention matrices and 3 feed-forward matrices, each block with its own values.`, `V Qwen3-8B má matica dopytov W_Q tvar [4 096 × 4 096] = ${S(A8.qkvoPerBlock.q)} váh. Každý z ${L8} blokov má 4 matice pozornosti a 3 matice doprednej siete, každý blok s vlastnými hodnotami.`) },
      { k: 'quiz', q: b('How many weights are in a matrix with 4,096 rows and 4,096 columns?', 'Koľko váh má matica so 4 096 riadkami a 4 096 stĺpcami?'), options: [b('8,192', '8 192'), b('16,777,216', '16 777 216'), b('4,096', '4 096')], correct: 1, explain: b('Rows × columns = 4,096 × 4,096 = 16,777,216.', 'Riadky × stĺpce = 4 096 × 4 096 = 16 777 216.') },
      { k: 'links', items: [{ href: explorerHref({ m: 'qwen3-8b', lv: 2, b: 0, t: 'q_proj' }), label: b('Look inside the real W_Q matrix', 'Pozrite sa do skutočnej matice W_Q') }, { href: '#/labs/matrix', label: b('All matrices of both models', 'Všetky matice oboch modelov') }] },
    ],
  },
  {
    id: 'tokens', n: 5, minutes: 5,
    title: b('From text to numbers: tokens and embeddings', 'Od textu k číslam: tokeny a embeddingy'),
    summary: b('Text becomes token ids; each id selects a row of numbers.', 'Text sa zmení na čísla tokenov; každé číslo vyberie riadok čísel.'),
    blocks: [
      { k: 'p', t: b('A **tokenizer** cuts text into tokens. Frequent words are usually one token; rare words, names or words in less common languages are split into several pieces, sometimes even single letters or bytes. Each token has an integer **id**.', '**Tokenizér** rozdelí text na tokeny. Časté slová sú zvyčajne jeden token; zriedkavé slová, mená alebo slová v menej rozšírených jazykoch sa rozdelia na viac kúskov, niekedy až na jednotlivé písmená či bajty. Každý token má celé číslo, **ID**.') },
      { k: 'widget', w: 'tokens' },
      { k: 'p', t: b('Next comes the **embedding table**: a big matrix with one row per token id. The model simply **looks up** the row for each token; no multiplication is needed. That row of numbers is the token’s starting representation inside the model.', 'Nasleduje **tabuľka embeddingov**: veľká matica s jedným riadkom pre každé ID tokenu. Model pre každý token riadok jednoducho **vyhľadá**; nie je potrebné nič násobiť. Tento riadok čísel je počiatočná reprezentácia tokenu v modeli.') },
      { k: 'p', t: b('Chat messages are first wrapped in a template with special tokens such as `<|im_start|>user`, so the model can tell who is speaking.', 'Správy v rozhovore sa najprv zabalia do šablóny so špeciálnymi tokenmi, napríklad `<|im_start|>user`, aby model rozlíšil, kto hovorí.') },
      { k: 'qwen', t: b(`Qwen’s tokenizer defines 151,669 token ids. The embedding table has ${E(QWEN3_8B.vocabRows)} rows × 4,096 numbers = ${E(A8.embedding)} parameters in Qwen3-8B; 267 rows are padding that no token uses.`, `Tokenizér modelu Qwen definuje 151 669 ID tokenov. Tabuľka embeddingov má ${S(QWEN3_8B.vocabRows)} riadkov × 4 096 čísel = ${S(A8.embedding)} parametrov v Qwen3-8B; 267 riadkov je výplň, ktorú nepoužíva žiadny token.`) },
      { k: 'quiz', q: b('Is a token always one word?', 'Je token vždy jedno slovo?'), options: [b('Yes, always', 'Áno, vždy'), b('No: it can be a word, part of a word, a symbol or a byte', 'Nie: môže to byť slovo, časť slova, znak alebo bajt')], correct: 1, explain: b('Tokens are pieces chosen by the tokenizer’s statistics. A Slovak word like “uložené” is split into several tokens (see the experiment above).', 'Tokeny sú kúsky zvolené podľa štatistiky tokenizéra. Slovenské slovo ako „uložené“ sa rozdelí na viac tokenov (pozrite pokus vyššie).') },
      { k: 'links', items: [{ href: '#/pipeline', label: b('Real Qwen3 tokenization, including Slovak sentences', 'Skutočná tokenizácia Qwen3 vrátane slovenských viet') }, { href: explorerHref({ m: 'qwen3-8b', lv: 3, mod: 'embedding' }), label: b('The embedding lookup in the explorer', 'Vyhľadanie embeddingu v prieskumníku') }] },
    ],
  },
  {
    id: 'residual', n: 6, minutes: 5,
    title: b('Keeping numbers healthy: normalization and the residual stream', 'Udržiavanie čísel v poriadku: normalizácia a reziduálny prúd'),
    summary: b('Each block adds a correction to a running vector, and normalizes before it reads it.', 'Každý blok pridáva k priebežnému vektoru opravu a pred čítaním ho normalizuje.'),
    blocks: [
      { k: 'p', t: b('The token’s list of numbers now travels through dozens of blocks. A block does not replace the list; it **adds a correction** to it: x ← x + block(x). This running list is called the **residual stream**. The addition itself has no weights at all.', 'Zoznam čísel tokenu teraz putuje cez desiatky blokov. Blok zoznam nenahrádza; **pridáva k nemu opravu**: x ← x + blok(x). Tento priebežný zoznam sa nazýva **reziduálny prúd**. Samotné sčítanie nemá žiadne váhy.') },
      { k: 'p', t: b('Before each step reads the stream, the numbers are **normalized** with RMSNorm: divide by their root-mean-square so their typical size is about 1, then multiply each position by a learned gain γ. This keeps the computation stable even after many blocks.', 'Predtým, ako ďalší krok prúd prečíta, sa čísla **normalizujú** pomocou RMSNorm: vydelia sa svojou strednou kvadratickou hodnotou, aby ich typická veľkosť bola okolo 1, a potom sa každá pozícia vynásobí naučeným ziskom γ. Výpočet tak zostáva stabilný aj po mnohých blokoch.') },
      { k: 'widget', w: 'rms' },
      { k: 'analogy', t: b('The residual stream is like a shared notepad passed along a line of experts: each one reads a cleaned-up copy (normalization) and writes a few notes into it (addition), without erasing what others wrote.', 'Reziduálny prúd je ako spoločný zápisník, ktorý si podáva rad odborníkov: každý si prečíta upravenú kópiu (normalizácia) a pripíše pár poznámok (sčítanie), bez toho, aby zmazal, čo napísali ostatní.') },
      { k: 'key', t: b('Residual additions: no weights. Normalization: very few learned weights (one gain per number in the vector).', 'Reziduálne sčítania: žiadne váhy. Normalizácia: veľmi málo naučených váh (jeden zisk pre každé číslo vo vektore).') },
      { k: 'qwen', t: b(`Each Qwen3-8B block has two RMSNorms (4,096 gains each) and two small head norms (128 each): ${E(A8.normsPerBlock)} parameters per block, less than 0.01% of the model.`, `Každý blok Qwen3-8B má dve RMSNorm (každá so 4 096 ziskami) a dve malé normy hláv (po 128): ${S(A8.normsPerBlock)} parametrov na blok, menej ako 0,01 % modelu.`) },
      { k: 'quiz', q: b('Does the residual addition contain learned weights?', 'Obsahuje reziduálne sčítanie naučené váhy?'), options: [b('Yes, one per number', 'Áno, jednu pre každé číslo'), b('No, it is a plain addition', 'Nie, je to obyčajné sčítanie')], correct: 1, explain: b('It just adds two lists of numbers elementwise. The learned parts are inside attention, the FFN and the norm gains.', 'Iba po prvkoch sčíta dva zoznamy čísel. Naučené časti sú v pozornosti, v doprednej sieti a v ziskoch normalizácie.') },
      { k: 'links', items: [{ href: '#/labs/layers', label: b('See the exact vectors being added', 'Pozrite si presné sčítavané vektory') }, { href: explorerHref({ m: 'qwen3-8b', lv: 3, b: 0, mod: 'attn_norm' }), label: b('RMSNorm step by step', 'RMSNorm krok za krokom') }] },
    ],
  },
  {
    id: 'attention', n: 7, minutes: 8,
    title: b('Attention: how tokens look at each other', 'Pozornosť: ako sa tokeny navzájom vnímajú'),
    summary: b('Queries, keys and values let each token collect information from earlier tokens.', 'Dopyty, kľúče a hodnoty umožňujú každému tokenu zbierať informácie z predchádzajúcich tokenov.'),
    blocks: [
      { k: 'p', t: b('So far each token was processed on its own. **Attention** is the step where a token gathers information from the tokens before it. Every token computes three new lists of numbers using three weight matrices: a **query** (“what am I looking for?”), a **key** (“what do I offer?”) and a **value** (“what will I pass on?”).', 'Doteraz sa každý token spracúval samostatne. **Pozornosť** (attention) je krok, v ktorom token zbiera informácie z tokenov pred ním. Každý token vypočíta pomocou troch matíc váh tri nové zoznamy čísel: **dopyt** („čo hľadám?“), **kľúč** („čo ponúkam?“) a **hodnotu** („čo odovzdám?“).') },
      { k: 'p', t: b('The query of the current token is compared with the key of every earlier token by a dot product (multiply position by position and add). A larger number means “more relevant”. These scores are turned into percentages that sum to 100% (softmax). The output is the percentage-weighted mix of the values.', 'Dopyt aktuálneho tokenu sa porovná s kľúčom každého predchádzajúceho tokenu skalárnym súčinom (vynásobiť pozíciu po pozícii a sčítať). Väčšie číslo znamená „relevantnejšie“. Tieto skóre sa premenia na percentá, ktoré dávajú spolu 100 % (softmax). Výstupom je zmes hodnôt vážená týmito percentami.') },
      { k: 'math', tex: '\\text{attention}(q, K, V) = \\operatorname{softmax}\\!\\Big(\\frac{q K^{\\top}}{\\sqrt{d}}\\Big) V' },
      { k: 'widget', w: 'attention' },
      { k: 'p', t: b('A token may only look **backwards**: future tokens do not exist yet while text is being generated. This is called the **causal mask**: their percentage is exactly zero.', 'Token sa smie pozerať len **dozadu**: budúce tokeny počas generovania textu ešte neexistujú. Nazýva sa to **kauzálna maska**: ich percento je presne nula.') },
      { k: 'p', t: b('Attention runs in several **heads** at once, each on its own slice of 128 numbers, so different heads can track different relationships. To save memory, Qwen lets groups of query heads share one key/value head (**grouped-query attention**).', 'Pozornosť beží súčasne vo viacerých **hlavách**, každá na vlastnom úseku 128 čísel, takže rôzne hlavy môžu sledovať rôzne vzťahy. Na úsporu pamäte Qwen necháva skupiny hláv dopytov zdieľať jednu hlavu kľúčov a hodnôt (**grouped-query attention**).') },
      { k: 'p', t: b('How does attention know word order? Qwen rotates query and key vectors by an angle that depends on the token’s position (**RoPE**). When two rotated vectors are compared, the result depends on how far apart the tokens are. RoPE has no learned weights.', 'Ako pozornosť pozná poradie slov? Qwen otáča vektory dopytov a kľúčov o uhol, ktorý závisí od pozície tokenu (**RoPE**). Keď sa porovnajú dva otočené vektory, výsledok závisí od toho, ako ďaleko sú tokeny od seba. RoPE nemá žiadne naučené váhy.') },
      { k: 'key', t: b('The attention percentages are **computed anew for every input**. The learned weights are the four matrices W_Q, W_K, W_V and W_O that produce queries, keys, values and the final mix.', 'Percentá pozornosti sa **počítajú nanovo pre každý vstup**. Naučenými váhami sú štyri matice W_Q, W_K, W_V a W_O, ktoré vytvárajú dopyty, kľúče, hodnoty a výslednú zmes.') },
      { k: 'qwen', t: b(`Qwen3-8B: 32 query heads share 8 key/value heads (groups of 4), each head 128 numbers wide; ${E(A8.attentionPerBlock)} attention weights per block, about ${ATT_SHARE}% of the whole model.`, `Qwen3-8B: 32 hláv dopytov zdieľa 8 hláv kľúčov a hodnôt (skupiny po 4), každá hlava má šírku 128 čísel; ${S(A8.attentionPerBlock)} váh pozornosti na blok, približne ${ATT_SHARE} % celého modelu.`) },
      { k: 'quiz', q: b('Are the attention percentages for your sentence stored in the model file?', 'Sú percentá pozornosti pre vašu vetu uložené v súbore modelu?'), options: [b('Yes, for every possible sentence', 'Áno, pre každú možnú vetu'), b('No, they are computed from your text each time', 'Nie, vždy sa vypočítajú z vášho textu')], correct: 1, explain: b('Only the matrices are stored. The percentages are activations: they exist only while your text is being processed.', 'Uložené sú len matice. Percentá sú aktivácie: existujú len počas spracovania vášho textu.') },
      { k: 'links', items: [{ href: '#/labs/attention', label: b('Attention lab: every number of the computation', 'Laboratórium pozornosti: každé číslo výpočtu') }, { href: '#/labs/rope', label: b('RoPE: rotating vectors by position', 'RoPE: otáčanie vektorov podľa pozície') }] },
    ],
  },
  {
    id: 'ffn', n: 8, minutes: 6,
    title: b('The feed-forward network: where most parameters live', 'Dopredná sieť: kde je väčšina parametrov'),
    summary: b('Each token is widened, gated and narrowed again; a “neuron” here is a gated channel.', 'Každý token sa rozšíri, prejde hradlom a znova zúži; „neurón“ je tu hradlovaný kanál.'),
    blocks: [
      { k: 'p', t: b('After attention, each token goes through a **feed-forward network** (FFN) on its own. It widens the list of numbers to many intermediate **channels**, applies a nonlinearity, and narrows it back.', 'Po pozornosti prejde každý token samostatne **doprednou sieťou** (FFN). Tá rozšíri zoznam čísel na mnoho medziľahlých **kanálov**, použije nelinearitu a opäť ho zúži.') },
      { k: 'p', t: b('Qwen uses **SwiGLU**. Each channel computes **two** weighted sums of the input: a **gate** and an **up** value. The gate passes through SiLU and then multiplies the up value: z = SiLU(gate) × up. The gate decides how much of the up value gets through. A third matrix (down) mixes all channels back.', 'Qwen používa **SwiGLU**. Každý kanál vypočíta zo vstupu **dva** vážené súčty: **hradlo** (gate) a hodnotu **up**. Hradlo prejde funkciou SiLU a potom vynásobí hodnotu up: z = SiLU(gate) × up. Hradlo rozhoduje, koľko z hodnoty up prejde ďalej. Tretia matica (down) všetky kanály opäť zmieša.') },
      { k: 'widget', w: 'gated' },
      { k: 'p', t: b(`This is what “neuron” most often means inside a transformer: one such channel. Qwen3-8B has ${E(QWEN3_8B.ffnWidth!)} per block, ${E(QWEN3_8B.ffnWidth! * L8)} in total. But the model also has other widths (4,096 residual numbers, 128 per head…), so a “number of neurons” always depends on what you decide to count.`, `Toto najčastejšie znamená „neurón“ vo vnútri transformera: jeden takýto kanál. Qwen3-8B ich má ${S(QWEN3_8B.ffnWidth!)} na blok, spolu ${S(QWEN3_8B.ffnWidth! * L8)}. Model má však aj iné šírky (4 096 čísel reziduálneho prúdu, 128 na hlavu…), takže „počet neurónov“ vždy závisí od toho, čo sa rozhodnete počítať.`) },
      { k: 'p', t: b('Researchers cannot reliably say what one channel “means”. Concepts are usually spread across many channels at once; this atlas never labels a channel as “the English neuron” or similar.', 'Výskumníci nevedia spoľahlivo povedať, čo jeden kanál „znamená“. Pojmy sú zvyčajne rozložené naraz do mnohých kanálov; tento atlas nikdy neoznačuje kanál ako „neurón pre angličtinu“ a podobne.') },
      { k: 'key', t: b(`The FFN holds about ${FFN_SHARE}% of Qwen3-8B’s parameters: three matrices per block.`, `Dopredná sieť obsahuje približne ${FFN_SHARE} % parametrov Qwen3-8B: tri matice v každom bloku.`) },
      { k: 'qwen', t: b(`3 × 4,096 × ${E(QWEN3_8B.ffnWidth!)} = ${E(A8.ffnPerBlock)} weights per block in Qwen3-8B.`, `3 × 4 096 × ${S(QWEN3_8B.ffnWidth!)} = ${S(A8.ffnPerBlock)} váh na blok v Qwen3-8B.`) },
      { k: 'quiz', q: b('How many weighted sums of the input does one SwiGLU channel compute?', 'Koľko vážených súčtov vstupu vypočíta jeden kanál SwiGLU?'), options: [b('One', 'Jeden'), b('Two (gate and up)', 'Dva (gate a up)'), b('Three', 'Tri')], correct: 1, explain: b('Two: gate and up. Their product is then carried back to the residual stream by a column of the down matrix.', 'Dva: gate a up. Ich súčin potom do reziduálneho prúdu prenesie stĺpec matice down.') },
      { k: 'links', items: [{ href: '#/labs/unit', label: b('Unit microscope: one channel, every product', 'Mikroskop jednotky: jeden kanál, každý súčin') }, { href: explorerHref({ m: 'qwen3-8b', lv: 3, b: 0, mod: 'ffn' }), label: b('The FFN in the explorer', 'Dopredná sieť v prieskumníku') }] },
    ],
  },
  {
    id: 'moe', n: 9, minutes: 6,
    title: b('Mixture of experts: many FFNs, few used', 'Zmes expertov: veľa sietí, použitých len pár'),
    summary: b('A router picks a few small FFNs per token; all of them are stored.', 'Smerovač vyberie pre každý token niekoľko malých sietí; uložené sú všetky.'),
    blocks: [
      { k: 'p', t: b('Qwen3-30B-A3B replaces the single FFN in each block with **128 smaller FFNs** called **experts**. For every token, in every block, a small **router** scores all 128 experts, keeps the 8 best, and mixes their outputs with weights that add up to 1.', 'Qwen3-30B-A3B nahrádza jednu doprednú sieť v každom bloku **128 menšími sieťami**, ktoré sa nazývajú **experti**. Pre každý token v každom bloku malý **smerovač** (router) ohodnotí všetkých 128 expertov, ponechá 8 najlepších a zmieša ich výstupy s váhami, ktoré dávajú spolu 1.') },
      { k: 'widget', w: 'router' },
      { k: 'p', t: b('All 128 experts must be stored and available, but only 8 actually compute for a given token. That is why the model **stores 30.5 billion** parameters but only about **3.3 billion are “activated”** per token.', 'Všetkých 128 expertov musí byť uložených a dostupných, no pre daný token skutočne počíta len 8. Preto model **ukladá 30,5 miliardy** parametrov, ale „**aktivovaných**“ je na jeden token len približne **3,3 miliardy**.') },
      { k: 'p', t: b('An expert is not a separate chatbot or a specialist with a job title. Which expert handles what is learned and often spread out; labels like “the math expert” are not justified by the evidence.', 'Expert nie je samostatný chatbot ani špecialista s pracovným zaradením. Čo ktorý expert spracúva, je naučené a často rozptýlené; označenia ako „expert na matematiku“ nemajú oporu v dôkazoch.') },
      { k: 'key', t: b('Mixture of experts gives a model more stored knowledge for a similar amount of computation per token, but it still needs memory for all experts.', 'Zmes expertov dáva modelu viac uložených znalostí pri podobnom množstve výpočtov na token, no stále potrebuje pamäť pre všetkých expertov.') },
      { k: 'qwen', t: b(`One expert: 3 × 2,048 × 768 = ${E(A30.oneExpert!)} weights. 128 experts per block: ${E(A30.ffnPerBlock)}. Router per block: ${E(A30.routerPerBlock)}. Qwen3-30B-A3B has no shared expert.`, `Jeden expert: 3 × 2 048 × 768 = ${S(A30.oneExpert!)} váh. 128 expertov v bloku: ${S(A30.ffnPerBlock)}. Smerovač v bloku: ${S(A30.routerPerBlock)}. Qwen3-30B-A3B nemá zdieľaného experta.`) },
      { k: 'quiz', q: b('Do all tokens of a sentence use the same 8 experts?', 'Používajú všetky tokeny vety tých istých 8 expertov?'), options: [b('Yes, the choice is made once per sentence', 'Áno, výber sa urobí raz pre celú vetu'), b('No, each token and each block chooses separately', 'Nie, každý token a každý blok vyberá samostatne')], correct: 1, explain: b('Routing happens per token and per block, so a whole sentence typically uses many more than 8 experts in total.', 'Smerovanie prebieha pre každý token a každý blok zvlášť, takže celá veta zvyčajne využije oveľa viac než 8 expertov.') },
      { k: 'links', items: [{ href: '#/labs/router', label: b('Router lab with the real 128 / top-8 rule', 'Laboratórium smerovača so skutočným pravidlom 128 / top-8') }, { href: '#/labs/compare', label: b('Dense vs mixture of experts', 'Hustý model vs zmes expertov') }] },
    ],
  },
  {
    id: 'block', n: 10, minutes: 5,
    title: b('Putting it together: the block and the whole model', 'Spojenie: blok a celý model'),
    summary: b('Norm, attention, add; norm, FFN, add — repeated with different weights in every block.', 'Normalizácia, pozornosť, sčítanie; normalizácia, FFN, sčítanie — opakované s inými váhami v každom bloku.'),
    blocks: [
      { k: 'p', t: b('One **decoder block** combines everything so far: normalize → attention → add to the stream; normalize → feed-forward network (or experts) → add to the stream. The model stacks many such blocks. They have identical shapes but **different weights**.', 'Jeden **blok dekodéra** spája všetko doterajšie: normalizácia → pozornosť → pripočítanie k prúdu; normalizácia → dopredná sieť (alebo experti) → pripočítanie k prúdu. Model takýchto blokov uloží na seba veľa. Majú rovnaký tvar, ale **iné váhy**.') },
      { k: 'math', tex: 'h = x + \\operatorname{Attn}(\\operatorname{Norm}(x)),\\qquad y = h + \\operatorname{FFN}(\\operatorname{Norm}(h))' },
      { k: 'widget', w: 'stack' },
      { k: 'p', t: b('After the last block comes one more normalization and the **output head**: a matrix with one row per vocabulary entry. It turns the final vector into one score (**logit**) per possible next token.', 'Za posledným blokom nasleduje ešte jedna normalizácia a **výstupná hlava**: matica s jedným riadkom pre každú položku slovníka. Premení výsledný vektor na jedno skóre (**logit**) pre každý možný ďalší token.') },
      { k: 'qwen', t: b(`Qwen3-8B: embedding ${E(A8.embedding)} + ${L8} blocks × ${E(A8.perBlock)} + final norm ${E(A8.finalNorm)} + output head ${E(A8.head)} = **${E(A8.total)}** parameters.`, `Qwen3-8B: embedding ${S(A8.embedding)} + ${L8} blokov × ${S(A8.perBlock)} + záverečná norma ${S(A8.finalNorm)} + výstupná hlava ${S(A8.head)} = **${S(A8.total)}** parametrov.`) },
      { k: 'quiz', q: b('Do all 36 blocks of Qwen3-8B use the same weights?', 'Používa všetkých 36 blokov Qwen3-8B tie isté váhy?'), options: [b('Yes, one block is reused 36 times', 'Áno, jeden blok sa použije 36-krát'), b('No, each block has its own weights of the same shape', 'Nie, každý blok má vlastné váhy rovnakého tvaru')], correct: 1, explain: b('Shapes repeat, values do not. What is reused is each block’s weights across all tokens of the text.', 'Tvary sa opakujú, hodnoty nie. Opakovane sa používajú váhy každého bloku pre všetky tokeny textu.') },
      { k: 'links', items: [{ href: explorerHref({ m: 'qwen3-8b', lv: 5 }), label: b('Every block of Qwen3-8B in the explorer', 'Každý blok Qwen3-8B v prieskumníku') }, { href: '#/parameters', label: b('Where exactly all parameters are', 'Kde presne sú všetky parametre') }] },
    ],
  },
  {
    id: 'generate', n: 11, minutes: 6,
    title: b('Generating text: from scores to words', 'Generovanie textu: od skóre k slovám'),
    summary: b('Scores become probabilities, one token is drawn, and the loop repeats with a cache.', 'Zo skóre sa stanú pravdepodobnosti, vyberie sa jeden token a slučka sa opakuje s pomocou cache.'),
    blocks: [
      { k: 'p', t: b('The output scores (logits) become probabilities with softmax. Then a **sampling** procedure picks one token: **temperature** makes the choice more cautious (low) or more adventurous (high); **top-k** and **top-p** throw away unlikely tokens first. Temperature 0 means always taking the most likely token.', 'Výstupné skóre (logity) sa pomocou softmaxu zmenia na pravdepodobnosti. Potom postup **vzorkovania** (sampling) vyberie jeden token: **teplota** robí výber opatrnejším (nízka) alebo odvážnejším (vysoká); **top-k** a **top-p** najprv vyradia nepravdepodobné tokeny. Teplota 0 znamená vždy vybrať najpravdepodobnejší token.') },
      { k: 'widget', w: 'sampling' },
      { k: 'p', t: b('The chosen token is appended and the whole model runs again for the next one. To avoid recomputing everything, the model keeps the keys and values of all earlier tokens in a **K/V cache**. Processing the prompt is called **prefill**; adding one token at a time is **decode**.', 'Vybraný token sa pripojí a celý model sa spustí znova pre ďalší. Aby nemusel všetko prepočítavať, uchováva kľúče a hodnoty všetkých predchádzajúcich tokenov v **K/V cache**. Spracovanie zadania sa nazýva **prefill**; pridávanie po jednom tokene je **decode**.') },
      { k: 'p', t: b('Sampling is not part of the trained network; it is a simple procedure applied to the scores. The “thinking” mode of Qwen3 is also not a different network: the same model receives a different template and writes its reasoning first.', 'Vzorkovanie nie je súčasťou natrénovanej siete; je to jednoduchý postup použitý na skóre. Ani „premýšľací“ režim Qwen3 nie je iná sieť: ten istý model dostane inú šablónu a najprv napíše svoje úvahy.') },
      { k: 'qwen', t: b(`Qwen3-8B’s K/V cache grows by ${KV8} KiB per token (at BF16); a 32,768-token conversation needs 4.5 GiB just for it. The default sampling settings are temperature 0.6, top-k 20, top-p 0.95.`, `K/V cache modelu Qwen3-8B rastie o ${KV8} KiB na token (v BF16); rozhovor s 32 768 tokenmi potrebuje len na ňu 4,5 GiB. Predvolené nastavenia vzorkovania sú teplota 0,6, top-k 20, top-p 0,95.`) },
      { k: 'quiz', q: b('Is sampling a learned layer of the network?', 'Je vzorkovanie naučená vrstva siete?'), options: [b('Yes, the last one', 'Áno, posledná'), b('No, it is a procedure applied to the output scores', 'Nie, je to postup použitý na výstupné skóre')], correct: 1, explain: b('It has settings (temperature, top-k, top-p) but no trained weights.', 'Má nastavenia (teplotu, top-k, top-p), ale žiadne natrénované váhy.') },
      { k: 'links', items: [{ href: '#/labs/sampling', label: b('Sampling lab', 'Laboratórium vzorkovania') }, { href: '#/labs/cache', label: b('Prefill, decode and the cache, with a correctness check', 'Prefill, decode a cache s kontrolou správnosti') }] },
    ],
  },
  {
    id: 'learning', n: 12, minutes: 7,
    title: b('Learning: where the weights come from', 'Učenie: odkiaľ sa berú váhy'),
    summary: b('Predict, measure the error, nudge every weight a little, repeat — trillions of times.', 'Predpovedať, zmerať chybu, každú váhu trochu posunúť, opakovať — biliónkrát.'),
    blocks: [
      { k: 'p', t: b('At the start of training the weights are random and the predictions are nonsense. Training shows the model huge amounts of text. At every position it predicts the next token, and the **loss** measures how wrong it was: the lower the probability it gave to the correct token, the larger the loss.', 'Na začiatku trénovania sú váhy náhodné a predpovede nezmyselné. Trénovanie ukazuje modelu obrovské množstvo textu. Na každej pozícii predpovedá ďalší token a **chyba** (loss) meria, ako veľmi sa mýlil: čím menšiu pravdepodobnosť dal správnemu tokenu, tým väčšia je chyba.') },
      { k: 'p', t: b('For every single weight, the **gradient** tells in which direction a tiny change would reduce the loss, and how strongly. It is computed for all weights at once by **backpropagation** (the chain rule of calculus). Each weight is then nudged: w ← w − learning rate × gradient.', 'Pre každú jednu váhu **gradient** určuje, ktorým smerom by malá zmena chybu znížila a ako výrazne. Pre všetky váhy naraz sa počíta **spätným šírením** (backpropagation, reťazové pravidlo derivácií). Každá váha sa potom trochu posunie: w ← w − rýchlosť učenia × gradient.') },
      { k: 'widget', w: 'train1d' },
      { k: 'p', t: b('A too large learning rate overshoots and the loss can get worse; a too small one learns very slowly. Real training uses smarter optimizers (such as Adam) that adapt the step for each weight.', 'Príliš veľká rýchlosť učenia „prestrelí“ a chyba sa môže zhoršiť; príliš malá sa učí veľmi pomaly. Skutočné trénovanie používa šikovnejšie optimalizátory (napríklad Adam), ktoré prispôsobujú krok každej váhe.') },
      { k: 'qwen', t: b('Qwen3 was pretrained on more than 36 trillion tokens in 119 languages and dialects, then further trained to follow instructions and reason. Qwen3-8B and Qwen3-30B-A3B learned partly by imitating larger Qwen3 models (distillation). The exact data mixtures and settings are not published.', 'Qwen3 bol predtrénovaný na viac ako 36 biliónoch tokenov v 119 jazykoch a nárečiach a potom ďalej trénovaný, aby plnil pokyny a uvažoval. Qwen3-8B a Qwen3-30B-A3B sa čiastočne učili napodobňovaním väčších modelov Qwen3 (destilácia). Presné zloženie dát a nastavenia nie sú zverejnené.') },
      { k: 'key', t: b('Training changes the weights. Using the model never does.', 'Trénovanie mení váhy. Používanie modelu ich nemení nikdy.') },
      { k: 'quiz', q: b('A weight’s gradient is negative. What does that mean?', 'Gradient váhy je záporný. Čo to znamená?'), options: [b('Increasing this weight would reduce the loss', 'Zväčšenie tejto váhy by chybu znížilo'), b('The weight must be deleted', 'Váhu treba vymazať'), b('The weight is already perfect', 'Váha je už dokonalá')], correct: 0, explain: b('The update w − lr × (negative) increases w, which is the direction that lowers the loss.', 'Úprava w − lr × (záporné číslo) váhu zväčší, a to je smer, ktorý chybu znižuje.') },
      { k: 'links', items: [{ href: '#/labs/training', label: b('Training lab: gradients you can check', 'Laboratórium trénovania: gradienty, ktoré si môžete overiť') }] },
    ],
  },
  {
    id: 'hardware', n: 13, minutes: 5,
    title: b('Where the numbers physically live: memory and hardware', 'Kde čísla fyzicky sú: pamäť a hardvér'),
    summary: b('Files, memory, graphics cards — and why a neuron is not a processor core.', 'Súbory, pamäť, grafické karty — a prečo neurón nie je jadro procesora.'),
    blocks: [
      { k: 'p', t: b('The weights sit in checkpoint files on disk. To run the model they are loaded into memory, usually the memory of a graphics card (GPU). The GPU then performs the matrix multiplications with thousands of small arithmetic units working in parallel.', 'Váhy sú uložené v súboroch checkpointu na disku. Na spustenie modelu sa načítajú do pamäte, zvyčajne do pamäte grafickej karty (GPU). GPU potom vykonáva násobenia matíc pomocou tisícok malých výpočtových jednotiek pracujúcich súčasne.') },
      { k: 'p', t: b('A “neuron” is not a physical part of the chip. It is just a position in a list of numbers; the same hardware computes millions of them one tile of a matrix after another.', '„Neurón“ nie je fyzická súčasť čipu. Je to len pozícia v zozname čísel; ten istý hardvér ich vypočíta milióny, jeden blok matice za druhým.') },
      { k: 'widget', w: 'memory' },
      { k: 'p', t: b('Memory need = number of parameters × bytes per number, plus the K/V cache, plus working space. Storing numbers with fewer bits (quantization, e.g. 4-bit) shrinks the memory but not the number of parameters.', 'Potrebná pamäť = počet parametrov × bajty na číslo, plus K/V cache, plus pracovný priestor. Ukladanie čísel s menším počtom bitov (kvantizácia, napr. 4-bitová) zmenší pamäť, ale nie počet parametrov.') },
      { k: 'quiz', q: b('Why does Qwen3-30B-A3B need memory for 30.5B parameters if only ~3.3B are active per token?', 'Prečo Qwen3-30B-A3B potrebuje pamäť pre 30,5 mld. parametrov, keď je na token aktívnych len ~3,3 mld.?'), options: [b('It doesn’t; 3.3B is enough', 'Nepotrebuje; stačí 3,3 mld.'), b('Different tokens and blocks pick different experts, so all must be available', 'Rôzne tokeny a bloky vyberajú rôznych expertov, preto musia byť dostupní všetci')], correct: 1, explain: b('The next token may need any of the 128 experts in any block.', 'Ďalší token môže potrebovať ktoréhokoľvek zo 128 expertov v ktoromkoľvek bloku.') },
      { k: 'links', items: [{ href: '#/labs/memory', label: b('Memory lab with all formulas', 'Laboratórium pamäte so všetkými vzorcami') }, { href: '#/catalog', label: b('Every tensor and where it sits in the files', 'Každý tenzor a jeho umiestnenie v súboroch') }] },
    ],
  },
  {
    id: 'beyond', n: 14, minutes: 4,
    title: b('Newer designs and open questions', 'Novšie návrhy a otvorené otázky'),
    summary: b('What newer Qwen models change, and what nobody knows yet.', 'Čo menia novšie modely Qwen a čo zatiaľ nevie nikto.'),
    blocks: [
      { k: 'p', t: b('Newer Qwen models (Qwen3-Next, Qwen3.5, Qwen3.8-Flash-Next) keep the same basic idea of stacked blocks but change parts of it. Three of every four attention layers are replaced by **Gated DeltaNet**, a layer with a fixed-size memory that is updated token by token. Experts get a shared companion that every token uses. Qwen3.8-Flash-Next additionally uses **sparse attention** that only reads selected blocks of earlier text, **four parallel residual streams**, and large learned tables for short word combinations (**n-grams**).', 'Novšie modely Qwen (Qwen3-Next, Qwen3.5, Qwen3.8-Flash-Next) zachovávajú základnú myšlienku blokov na sebe, ale menia niektoré ich časti. Tri zo štyroch vrstiev pozornosti nahrádza **Gated DeltaNet**, vrstva s pamäťou pevnej veľkosti, ktorá sa aktualizuje token po tokene. Experti dostávajú zdieľaného partnera, ktorého používa každý token. Qwen3.8-Flash-Next navyše používa **riedku pozornosť**, ktorá číta len vybrané úseky predchádzajúceho textu, **štyri paralelné reziduálne prúdy** a veľké naučené tabuľky pre krátke spojenia slov (**n-gramy**).') },
      { k: 'p', t: b('What is still unknown: what individual weights or channels mean, and exactly why a model gives a particular answer. This is an active research field (interpretability). This atlas shows the **mechanics** precisely, not the model’s “thoughts”.', 'Čo zostáva neznáme: čo znamenajú jednotlivé váhy či kanály a prečo presne model dáva konkrétnu odpoveď. Je to aktívna oblasť výskumu (interpretovateľnosť). Tento atlas presne ukazuje **mechaniku**, nie „myšlienky“ modelu.') },
      { k: 'list', items: [b('Explore the real structure level by level in the explorer.', 'Preskúmajte skutočnú štruktúru úroveň po úrovni v prieskumníku.'), b('Experiment with every computation in the labs.', 'Experimentujte s každým výpočtom v laboratóriách.'), b('Check every fact on the Sources page.', 'Každý fakt si overte na stránke Zdroje.')] },
      { k: 'quiz', q: b('Does this atlas show what the model “thinks”?', 'Ukazuje tento atlas, čo si model „myslí“?'), options: [b('Yes, the attention maps show its thoughts', 'Áno, mapy pozornosti ukazujú jeho myšlienky'), b('No, it shows the exact mechanics; meanings of internal values are largely unknown', 'Nie, ukazuje presnú mechaniku; význam vnútorných hodnôt je zväčša neznámy')], correct: 1, explain: b('Attention maps and activations are measurable, but turning them into reliable explanations is an open research problem.', 'Mapy pozornosti a aktivácie sa dajú zmerať, ale premeniť ich na spoľahlivé vysvetlenia je otvorený výskumný problém.') },
      { k: 'links', items: [{ href: '#/hybrid/qwen3.8-flash-next', label: b('Qwen3.8-Flash-Next in detail', 'Qwen3.8-Flash-Next podrobne') }, { href: '#/family', label: b('The Qwen family map', 'Mapa rodiny Qwen') }, { href: explorerHref({ m: 'qwen3-8b', lv: 5 }), label: b('Open the explorer', 'Otvoriť prieskumník') }, { href: '#/labs', label: b('All labs', 'Všetky laboratóriá') }] },
    ],
  },
];
