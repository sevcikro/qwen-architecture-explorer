// Slovak versions of user-facing data texts (claims, sources, family, hybrid profiles).
// English stays in the data modules; these maps are looked up by id and fall back to English.
import type { Lang } from './i18n';
import type { Coverage } from '../models/hybrid';

export const COVERAGE_LABEL_SK: Record<Coverage, string> = {
  'full-toy': 'Úplná číselná hračková obdoba + overený štrukturálny pohľad',
  structural: 'Overený štrukturálny pohľad (konfigurácia, hlavičky checkpointu, implementácia)',
  overview: 'Vysvetľujúci prehľad',
};

export const HYBRID_SK: Record<string, { attnLabel: string; residual: string; context: string; advertised: string; extras: string[]; extraToys: string[] }> = {
  'qwen3-next-80b-a3b': {
    attnLabel: 'Hradlovaná pozornosť (bežná kauzálna softmax pozornosť so sigmoidovým výstupným hradlom)',
    residual: 'Bežný jediný reziduálny prúd so šírkou 2048',
    context: '262 144 natívne (karta modelu); rozšíriteľné na 1 010 000',
    advertised: '80 mld. spolu, 3 mld. aktivovaných (karta modelu)',
    extras: ['Predpovedanie viacerých tokenov (MTP)', 'Layernorm centrovaná na nulu s váhovým útlmom (dokumentácia Transformers)'],
    extraToys: ['Rekurencia Gated DeltaNet (spoločná s hračkou Qwen3.8)'],
  },
  'qwen3.5-35b-a3b': {
    attnLabel: 'Hradlovaná pozornosť',
    residual: 'Bežný jediný reziduálny prúd so šírkou 2048',
    context: '262 144 natívne (karta modelu); rozšíriteľné na 1 010 000',
    advertised: '35 mld. spolu, 3 mld. aktivovaných (karta modelu); vrátane kódovača obrazu',
    extras: ['Kódovač obrazu (model pre obraz a jazyk so skorým spojením)', 'MTP trénované vo viacerých krokoch'],
    extraToys: [],
  },
  'qwen3.8-flash-next': {
    attnLabel: 'Qwen Sparse Attention (QSA): hradlovaná softmax pozornosť obmedzená na 4-tokenové mikrobloky vybrané indexerom plus koncový blok. Konfigurácia tieto vrstvy volá „full_attention“, ale implementácia pridáva masku výberu.',
    residual: '4 reziduálne vetvy so šírkou 2560, nesené ako jeden tenzor [B, T, 10240] (hc_count = 4)',
    context: '262 144 natívne, rozšíriteľné na 1 000 000 (karta modelu); max_position_embeddings 262 144 (konfigurácia)',
    advertised: '125 mld. so 6 mld. aktivovanými, plus 51 mld. n-gramových embeddingov a 4 mld. MTP (karta modelu)',
    extras: ['Hašované bigramové/trigramové tabuľky vo „vrstve 2“ (konfigurácia ple_layer_ids=[2]; index bloku v kóde 1)', 'Modul MTP (1 vrstva)', 'Kódovač obrazu (27 blokov)'],
    extraToys: ['Rekurencia Gated DeltaNet', 'Výber blokov QSA', 'Čítanie/zápis hradlovaného rezídua'],
  },
};

export const COMPONENT_SK: Record<string, string> = {
  routed_experts: 'Smerovaní experti (48 × 512)',
  gated_deltanet: 'Miešače Gated DeltaNet (36 blokov)',
  gated_residual: 'Čítanie/zápis hradlovaného rezídua (2 na blok + záverečný miešač)',
  embed: 'Vstupný embedding [248320, 2560]',
  lm_head: 'Výstupná hlava [248320, 2560]',
  qsa_attention: 'Projekcie a normalizácie pozornosti QSA (12 blokov)',
  shared_expert: 'Zdieľaní experti + hradlá zdieľaných expertov (48)',
  router: 'Smerovače (48 × [512, 2560])',
  qsa_indexer: 'Indexery QSA (12)',
  ngram_tables: 'N-gramové tabuľky (128 častí, 320 001 536 riadkov × 160)',
  ple_other: 'Projekcie vkladania n-gramov, normalizácie, konvolúcia (+35 položiek int64 bufferov)',
  mtp: 'Tenzory modulu MTP (mtp.*)',
  vision: 'Kódovač obrazu (model.visual.*)',
};

export const FAMILY_SK: Record<string, { architecture: string; relation: string; inAtlas: string }> = {
  'Qwen (first generation)': { architecture: 'Hustý dekodér', relation: 'Pôvodná séria', inAtlas: 'Len na časovej osi' },
  'Qwen1.5': { architecture: 'Hustý dekodér (a varianty Qwen1.5-MoE)', relation: 'Predchodca', inAtlas: 'Len na časovej osi' },
  'Qwen2': { architecture: 'Hustý + MoE; GQA; bias QKV (podľa správy o Qwen3, ktorá ho odstraňuje)', relation: 'Predchodca', inAtlas: 'Len na časovej osi' },
  'Qwen2.5': { architecture: 'Hustý; Qwen2.5-MoE používal zdieľaných expertov (podľa správy o Qwen3)', relation: 'Priamy predchodca Qwen3', inAtlas: 'Len na časovej osi' },
  'Qwen3 (dense + MoE)': { architecture: 'Dekodér s pred-normalizáciou, GQA, QK-Norm, RoPE, SwiGLU; MoE so 128 expertmi / top-8, bez zdieľaného experta', relation: 'Referenčné predvoľby tohto atlasu', inAtlas: 'Najhlbšie pokrytie: úplná hračková obdoba + overená štruktúra' },
  'Qwen3-Next': { architecture: 'Hybrid 3:1 Gated DeltaNet / hradlovaná pozornosť, 512 expertov (10 smerovaných + 1 zdieľaný), MTP', relation: 'Prvá hybridná línia Qwen', inAtlas: 'Overený štrukturálny pohľad' },
  'Qwen3.5': { architecture: 'Hybrid 3:1 Gated DeltaNet / hradlovaná pozornosť, MoE so zdieľaným expertom, kódovač obrazu', relation: 'Hybrid, obraz a jazyk', inAtlas: 'Overený štrukturálny pohľad' },
  'Qwen3.6': { architecture: 'Rovnaká trieda architektúry v Transformers ako Qwen3.5 (Qwen3_5MoeForConditionalGeneration; 40 vrstiev, 256 expertov, top-8)', relation: 'Hybrid, obraz a jazyk', inAtlas: 'Len uvedený (trieda konfigurácie overená; samostatne neanalyzovaný)' },
  'Qwen3.8-Flash-Next': { architecture: 'model_type qwen4_exp: Gated DeltaNet + Qwen Sparse Attention, 4-vetvové hradlované rezíduum, hašované n-gramové tabuľky, 512 expertov (10 + 1 zdieľaný), MTP, kódovač obrazu', relation: '„Experimentálna ukážka architektúry, na ktorej bude stáť Qwen4“ (karta modelu)', inAtlas: 'Overený štrukturálny pohľad + hračkové obdoby troch mechanizmov' },
};

export const SCOPE_SK: Record<string, string> = { both: 'oba', 'Toy simulator': 'Hračkový simulátor' };

export const CLAIM_SK: Record<string, { claim: string; note?: string }> = {
  'c-8b-dims': { claim: '36 blokov dekodéra; d = 4096; 32 hláv dopytov / 8 hláv K/V; head_dim 128; šírka FFN 12 288; 151 936 riadkov slovníka; neprepojené; bez biasov v lineárnych vrstvách; RMSNorm eps 1e-6; rope_theta 1e6.' },
  'c-30b-dims': { claim: '48 blokov dekodéra; d = 2048; 32 / 4 hláv; head_dim 128 (teda Hq·dh = 4096 ≠ d); 128 expertov, top-8, šírka experta 768, norm_topk_prob = true; bez zdieľaného experta.' },
  'c-30b-intermediate': { claim: 'intermediate_size = 6144 je v konfigurácii, ale žiadny blok nepoužíva hustú FFN: mlp_only_layers = [] a decoder_sparse_step = 1, takže všetkých 48 blokov je MoE so šírkou experta 768.' },
  'c-totals': { claim: 'Presné počty parametrov: 8 190 735 360 (Qwen3-8B) a 30 532 122 624 (Qwen3-30B-A3B).', note: 'Vypočítané z tvarov v konfigurácii; nezávisle sa rovnajú súčtu prvkov v pripnutých hlavičkách safetensors aj total_size v indexe / 2 bajty.' },
  'c-advertised': { claim: 'Karty modelov ich zaokrúhľujú na 8,2 mld. (6,95 mld. bez embeddingov) a 30,5 mld. (29,9 mld. bez embeddingov, 3,3 mld. aktivovaných).', note: 'Čísla bez embeddingov zodpovedajú výpočtu spolu − vstupný embedding − výstupná hlava: 6 946 075 648 ≈ 6,95 mld. a 29 909 792 768 ≈ 29,9 mld. Odčítanie len vstupnej tabuľky (7,57 mld. / 30,22 mld.) nesedí.' },
  'c-active': { claim: 'Vybraní experti cez všetkých 48 blokov: 1 811 939 328 parametrov; so všetkými parametrami mimo expertov (vrátane celej tabuľky embeddingov): 3 353 032 704.', note: 'Druhý spôsob reprodukuje ≈3,3 mld. z karty, ale počíta celú tabuľku embeddingov, ktorú vyhľadanie nečíta. Nie je to počet operácií, prenosov ani oneskorenie.' },
  'c-context': { claim: 'Natívny kontext 32 768 tokenov; 131 072 s YaRN (faktor 4, voliteľné rope_scaling); v konfigurácii max_position_embeddings 40 960.', note: 'Tabuľka v správe o Qwen3 uvádza kontext 128K; karta modelu vysvetľuje 40 960 = 32 768 výstup + 8 192 rezerva na zadanie. Sú to rôzne veličiny a zobrazujú sa oddelene.' },
  'c-tokenizer': { claim: 'Tokenizér: BPE na úrovni bajtov so 151 643 bežnými + 26 pridanými tokenmi = 151 669 ID; embedding/hlava majú 151 936 riadkov (267 nepoužitých riadkov výplne).' },
  'c-qknorm': { claim: 'Q a K sa RMS-normalizujú v rámci každej hlavy s jedným ziskom veľkosti head_dim na blok (spoločným pre hlavy), pred RoPE.' },
  'c-rope': { claim: 'RoPE páruje súradnicu i so súradnicou i + head_dim/2 (rotate_half), frekvencie θ^(−2i/d), len na Q a K, bez naučených parametrov.' },
  'c-rmsnorm': { claim: 'RMSNorm počíta vo float32: weight * (x * rsqrt(mean(x²) + eps)), bez odčítania priemeru.' },
  'c-experts-layout': { claim: 'Checkpoint ukladá 128 × 3 samostatné tenzory expertov na blok; súčasné Transformers ich pri načítaní zbalia do gate_up_proj [128, 1536, 2048] (najprv gate) a down_proj [128, 2048, 768].' },
  'c-router': { claim: 'Smerovač: logity = x W_routerᵀ; softmax cez všetkých 128 expertov vo float32; top-8; prepočítanie 8 pravdepodobností.' },
  'c-lb': { claim: 'Trénovanie používalo vyvažovanie záťaže cez celú dávku; v konfigurácii router_aux_loss_coef = 0,001.' },
  'c-kv': { claim: 'Cache K/V pri BF16, dávka 1: 147 456 B/token (8B), 98 304 B/token (30B-A3B); 4,5 GiB a 3 GiB pri 32 768 tokenoch (len K/V).' },
  'c-train': { claim: 'Predtrénovanie na viac než 36 bil. tokenov (119 jazykov) v troch fázach; 8B a 30B-A3B dotrénované destiláciou zo silného na slabý.' },
  'c-train-unknown': { claim: 'Presné zloženie dát, nastavenia optimalizátora a rýchlosti učenia pre jednotlivé modely a význam jednotlivých neurónov či expertov.', note: 'Na tejto stránke sa nereprodukujú; nič tieto medzery nevypĺňa.' },
  'c-thinking': { claim: 'Režim s premýšľaním / bez neho sa volí šablónou chatu (enable_thinking) alebo /think, /no_think; váhy aj vrstvy sú rovnaké.' },
  'c-38-layout': { claim: '48 blokov = 12 × (3 Gated DeltaNet + 1 Qwen Sparse Attention), za každým MoE s 512 expertmi (10 smerovaných + 1 zdieľaný, šírka 640).' },
  'c-38-qsa': { claim: 'Vrstvy označené v konfigurácii ako full_attention sú QSA: kľúče sa združia do 4-tokenových blokov, indexer ponechá najlepších 2048/4 = 512 blokov plus neúplný koniec, použité ako ďalšia maska na hradlovanú softmax pozornosť.' },
  'c-38-residual': { claim: 'Reziduálny stav je 4 vetvy × 2560 = [B, T, 10240]; v každej podvrstve: sigmoidové čítacie hradlo po prvkoch (úzke hrdlo hodnosti 320), priemer cez vetvy, zapisovacie hradlo 2·sigmoid pre každú vetvu.' },
  'c-38-ngram': { claim: 'Hašované bigramové/trigramové tabuľky v „vrstve 2“ konfigurácie = index bloku v kóde 1 (layers.1.ple.*): 128 častí, 320 001 536 riadkov × 160 = 51 200 245 760 parametrov.', note: 'ple_layer_ids = [2] sa počíta od 1: implementácia testuje layer_idx + 1 ∈ ple_layer_ids.' },
  'c-38-accounting': { claim: 'Odvodené z hlavičiek: textová kostra 125 710 814 080; n-gramy 51 233 085 475 (vrátane vkladania); MTP 2 607 150 848; obraz 448 931 056; všetkých uložených prvkov 179 999 981 459.' },
  'c-38-mtp': { claim: 'Ako karta modelu dospela k „4B MTP“.', note: 'Tenzory mtp.* majú spolu 2,61 mld.; pripočítanie spoločného embeddingu a hlavy dá 3,88 mld. ≈ 4 mld., čo môže byť použitý spôsob, ale žiadny overený zdroj to neuvádza.' },
  'c-38-gdn': { claim: 'Rekurencia Gated DeltaNet: S ← S·exp(g); δ = β(v − Sᵀk); S ← S + kδᵀ; o = Sᵀq, kde g = −exp(A_log)·softplus(a + dt_bias), β = σ(b), q/k normalizované L2.' },
  'c-toy': { claim: 'Všetky hračkové hodnoty (váhy, aktivácie, smerovania, mapy pozornosti, generované hračkové tokeny) sú názorné simulácie so syntetickými váhami z osiva.' },
  'c-samples': { claim: 'Malé dlaždice váh boli prečítané z pripnutých častí checkpointu s overenými odpoveďami HTTP 206 na rozsahy; každá má SHA-256 svojich surových bajtov.' },
  'c-06-params': { claim: 'Jedinečné naučené parametre: 596 049 920 (prepojené embeddingy). Súbor checkpointu ukladá 751 632 384 prvkov, lebo uloží aj lm_head.weight, bajtovo identickú kópiu tabuľky embeddingov.', note: 'Karta: 0,6 mld. spolu, 0,44 mld. bez embeddingov (= 596 049 920 − 155 582 464 = 440 467 456).' },
  'c-06-journey': { claim: 'Zaznamenaný ťah v chate: pre „What is a parameter?“ (bez premýšľania) vytvorí hladové dekódovanie 139 tokenov a zastaví sa na <|im_end|>; výstupy príkladov v Pythone v kurze pochádzajú zo skutočných behov.', note: 'Malý model a jedno pevné zadanie; časy sa netvrdia.' },
  'c-06-trace': { claim: 'Namerané aktivácie troch zadaní (napr. „The capital of France is“ → „ Paris“ s p = 0,657).', note: 'Iný, menší model než referenčné predvoľby; nie sú to aktivácie Qwen3-8B ani Qwen3-30B-A3B. Samotné pravdepodobnosti pozornosti nie sú príčinným vysvetlením.' },
};

export const SOURCE_SK: Record<string, string> = {
  'q8-config': 'hidden_size 4096, 36 vrstiev, 32 hláv Q / 8 hláv KV, head_dim 128, intermediate_size 12288, vocab_size 151936, neprepojené embeddingy, bez biasu pozornosti, rms_norm_eps 1e-6, rope_theta 1e6, max_position_embeddings 40960, bfloat16.',
  'q8-card': '8,2 mld. parametrov (6,95 mld. bez embeddingov), natívny kontext 32 768, 131 072 s YaRN (faktor 4), prečo je max_position_embeddings 40 960, odporúčania vzorkovania, prepínanie premýšľania.',
  'q8-index': '399 tenzorov v 5 častiach; total_size 16 381 470 720 bajtov = 8 190 735 360 prvkov BF16; presné názvy, tvary, dátové typy a bajtové posuny tenzorov (hlavičky načítané cez HTTP Range).',
  'q8-gen': 'Predvolené vzorkovanie: temperature 0,6, top_k 20, top_p 0,95; ID konca 151645 a 151643.',
  'q8-tokenizer': 'BPE na úrovni bajtov so 151 643 bežnými tokenmi + 26 pridanými tokenmi (ID 151643–151668); šablóna chatu s <|im_start|>/<|im_end|> a spracovaním <think>.',
  'q30-config': 'hidden_size 2048, 48 vrstiev, 32 hláv Q / 4 hlavy KV, head_dim 128, num_experts 128, num_experts_per_tok 8, moe_intermediate_size 768, intermediate_size 6144 (bloky MoE ho nepoužívajú: mlp_only_layers=[] a decoder_sparse_step=1), norm_topk_prob true, router_aux_loss_coef 0,001.',
  'q30-card': '30,5 mld. spolu, 3,3 mld. aktivovaných, 29,9 mld. bez embeddingov, 128 expertov, 8 aktivovaných, kontext 32 768 natívne / 131 072 s YaRN.',
  'q30-index': '18 867 tenzorov v 16 častiach; experti uložení ako samostatné tenzory model.layers.{i}.mlp.experts.{e}.{gate,up,down}_proj.weight; smerovač model.layers.{i}.mlp.gate.weight; spolu 30 532 122 624 prvkov BF16.',
  'tf-qwen3': 'Reziduálny blok s pred-normalizáciou, q_norm/k_norm cez head_dim, párovanie RoPE po polovičkách (rotate_half), RMSNorm vo float32 a potom násobenie váhou, SwiGLU down(act(gate(x))*up(x)), softmax pozornosti v režime eager vo float32.',
  'tf-qwen3moe': 'Smerovač: logity = x W^T, softmax vo float32 cez všetkých expertov, top-k, prepočet pri norm_topk_prob. Experti sú počas behu držaní zbalení ako gate_up_proj [E, 2*I, H] (najprv gate, potom up cez chunk(2)) a down_proj [E, H, I].',
  'tf-convmap': 'Ako sa tenzory jednotlivých expertov z checkpointu pri načítaní zbalia do parametrov gate_up_proj a down_proj.',
  'qwen3-report': 'QK-Norm, bez biasu QKV, GQA, SwiGLU, RoPE, pred-RMSNorm; MoE bez zdieľaných expertov a s vyvažovaním záťaže cez celú dávku; slovník tokenizéra 151 669; 36 bil. tokenov, 119 jazykov; fázy S1/S2/dlhý kontext; základ RoPE 10k → 1M cez ABF; YaRN + DCA; štvorfázové dotrénovanie a destilácia zo silného na slabý pre 8B a 30B-A3B; režimy s premýšľaním a bez neho.',
  'qn-card': '80 mld. spolu / 3 mld. aktivovaných; 48 vrstiev ako 12 × (3 × (Gated DeltaNet → MoE) → 1 × (hradlovaná pozornosť → MoE)); 512 expertov, 10 smerovaných + 1 zdieľaný; natívny kontext 262 144.',
  'q35-card': '35 mld. spolu / 3 mld. aktivovaných; 40 vrstiev ako 10 × (3 × (Gated DeltaNet → MoE) → 1 × (hradlovaná pozornosť → MoE)); 256 expertov, 8 smerovaných + 1 zdieľaný; kódovač obrazu; slovník s výplňou 248 320; natívny kontext 262 144.',
  'q38-card': '125 mld. so 6 mld. aktivovanými plus 51 mld. n-gramových embeddingov a 4 mld. MTP; 48 vrstiev ako 12 × (3 × (Gated DeltaNet → MoE) → 1 × (Qwen Sparse Attention → MoE)); 512 expertov, 10 smerovaných + 1 zdieľaný, šírka experta 640; hradlované rezíduum so 4 vetvami, úzke hrdlo hodnosti 320; rozpočet QSA „512 blokov alebo 2048 tokenov“; natívny kontext 262 144; licencia qwen-community-1.0.',
  'q38-config': 'model_type qwen4_exp; text_config: hidden_size 2560, hc_count 4, hc_lowrank 320, layer_types (36 linear_attention, 12 full_attention), indexer_budget 2048, indexer_compress_ratio 4, ple_layer_ids [2], ngram_size 3, ngram_vocab_size_base 20 000 000, slovník 248320, partial_rotary_factor 0,25; vision_config depth 27.',
  'q38-index': '1 658 tenzorov v 131 častiach, 179 999 981 459 uložených prvkov (BF16 plus tri malé buffery int64); n-gramové tabuľky v bloku checkpointu s indexom 1 (model.language_model.layers.1.ple.*); zbalení experti gate_up_proj [512,1280,2560].',
  'tf-qwen4exp': 'Presná rekurencia Gated DeltaNet, združovanie/hodnotenie/výber blokov QSA, sigmoidovo hradlovaný výstup pozornosti, čítanie/zápis 4-vetvového hradlovaného rezídua, hašovaný n-gramový embedding vo vrstve dekodéra, ktorej index od 0 + 1 je v ple_layer_ids, sigmoidové hradlo zdieľaného experta. configuration_qwen4_exp.py: norm_topk_prob je predvolene True.',
  'q38-report': 'Mikrobloky QSA s r=4 združenými kľúčmi, indexer MQA (4 hlavy dopytov, 1 hlava kľúčov), až 512 blokov pre rozpočet 2048 tokenov, koncové tokeny vždy ponechané; čítacie hradlo hradlovaného rezídua a zapisovacie hradlo 2*sigmoid pre každú vetvu; n-gramové tabuľky navrhnuté na odloženie do pamäte hostiteľa; Muon pre 2-D lineárne váhy a AdamW pre embeddingy/hlavu/smerovač/nízkohodnostné časti GR; bez zahrievania veľkosti dávky.',
  'q06-config': '28 vrstiev, hidden_size 1024, 16 hláv Q / 8 hláv KV, head_dim 128, intermediate_size 3072, tie_word_embeddings true. Jediný súbor safetensors (SHA-256 f47f7117…, rovnaký ako hash LFS na Hube) obsahuje 311 tenzorov / 751 632 384 prvkov vrátane lm_head.weight, ktorá je bajtovo zhodná s model.embed_tokens.weight.',
  'q06-card': '0,6 mld. parametrov, 0,44 mld. bez embeddingov, 28 vrstiev, 16 hláv Q / 8 hláv KV, kontext 32 768; Apache-2.0.',
  'q06-trace': 'Skutočné dopredné prechody na CPU (torch 2.14.1+cpu, transformers 5.20.2, float32, pozornosť eager) troch pevných zadaní s háčikmi: RMS rezídua v každom bloku, črty rezídua, pravdepodobnosti pozornosti, aktivácie hradiel, pravdepodobnosti ďalšieho tokenu.',
  'q06-journey': 'Hladové dekódovanie s cache K/V (float32, CPU) správy v chate „What is a parameter?“ s vypnutým premýšľaním: 17 tokenov zadania, 139 generovaných tokenov zakončených <|im_end|>, päť najpravdepodobnejších kandidátov v každom kroku. Aj zaznamenané výstupy príkladov v Pythone z kurzu (scripts/run_examples.py).',
  yarn: 'Rozšírenie kontextového okna modelu s RoPE preškálovaním frekvencií otáčania, s malým dodatočným trénovaním.',
  kaplan: 'Chyba jazykového modelu klesá plynulo ako mocninová funkcia počtu parametrov, dát a výpočtov.',
  chinchilla: 'Pri pevnom rozpočte výpočtov majú parametre a trénovacie tokeny rásť spolu (asi 20 tokenov na parameter pre skúmané modely).',
  'lost-middle': 'V testovaných modeloch sa dôležitá informácia využila najlepšie na začiatku alebo na konci dlhého vstupu a najhoršie v strede.',
  bert: 'Transformer iba s kódovačom trénovaný predpovedaním zamaskovaných tokenov; pozornosť vidí celý vstup oboma smermi.',
  vaswani: 'Škálovaná pozornosť cez skalárny súčin softmax(QK^T/sqrt(d_k))V a dekodér Transformer.',
  rmsnorm: 'RMSNorm škáluje strednou kvadratickou hodnotou bez odčítania priemeru, s naučeným ziskom.',
  swiglu: 'Dopredná sieť SwiGLU: (Swish(xW) * xV) W2, bez biasov.',
  rope: 'Rotačné pozičné kódovanie: otáčanie párov súradníc dopytov/kľúčov o uhly závislé od pozície spôsobí, že q·k závisí od relatívnej pozície.',
  gqa: 'Pozornosť so zoskupenými dopytmi: viac hláv dopytov zdieľa jednu hlavu kľúčov/hodnôt.',
  'hf-cache': 'Stavy kľúčov a hodnôt sa pri generovaní ukladajú pre každú vrstvu, aby sa skoršie pozície neprepočítavali.',
  safetensors: 'Rozloženie súboru: 8-bajtová dĺžka hlavičky (little-endian), hlavička JSON s dtype/shape/data_offsets každého tenzora, potom surové súvislé bajty tenzorov.',
  'torch-linear': 'Linear ukladá váhu v tvare (out_features, in_features) a počíta y = x W^T (+ b).',
  'torch-silu': 'SiLU(x) = x * sigmoid(x).',
  flashattn: 'Presná pozornosť počítaná po dlaždiciach s priebežným softmaxom, bez uloženia celej matice skóre T × T do pamäte GPU.',
  switch: 'Pomocná chyba na vyvažovanie záťaže smerovaných expertov; paralelizmus expertov s komunikáciou rozoslania/zozbierania a limitmi kapacity.',
  gdn: 'Hradlované delta pravidlo: maticový stav s útlmom, aktualizovaný zápisom podľa delta pravidla.',
  adam: 'Adam si drží odhady prvého a druhého momentu pre každý parameter (stav optimalizátora).',
  gguf: 'GGUF je formát kontajnera, ktorého tenzory môžu používať mnoho rôznych typov kvantizácie.',
  'hf-api': 'Dátumy vytvorenia repozitárov a nemenné ID revízií použité v mape rodiny.',
};

/** Pick the Slovak text when available. */
export const pick = (lang: Lang, en: string, sk: string | undefined) => (lang === 'sk' && sk ? sk : en);
