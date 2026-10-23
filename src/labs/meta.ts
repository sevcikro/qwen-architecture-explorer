import type { L10n } from '../i18n/i18n';

const b = (en: string, sk: string): L10n => ({ en, sk });

/** Lab titles and summaries (kept apart from the lab components so search can import them cheaply). */
export const LAB_META = [
  { id: 'weight', n: 1, title: b('Weight microscope', 'Mikroskop váhy'), blurb: b('Edit one stored number and watch exactly what changes downstream.', 'Upravte jedno uložené číslo a sledujte, čo presne sa zmení ďalej.'), learn: 'neuron' },
  { id: 'unit', n: 2, title: b('Unit microscope', 'Mikroskop jednotky'), blurb: b('Every product and sum in one SwiGLU gated channel.', 'Každý súčin a súčet v jednom hradlovanom kanáli SwiGLU.'), learn: 'ffn' },
  { id: 'matrix', n: 3, title: b('Matrix explorer', 'Prieskumník matíc'), blurb: b('Shapes, dtypes, counts, checkpoint mapping and real sample tiles.', 'Tvary, typy čísel, počty, umiestnenie v checkpointe a skutočné vzorky.'), learn: 'matrix' },
  { id: 'attention', n: 4, title: b('Attention lab', 'Laboratórium pozornosti'), blurb: b('Q·K, scaling, causal mask, stable softmax, weighted V, GQA.', 'Q·K, škálovanie, kauzálna maska, stabilný softmax, vážené V, GQA.'), learn: 'attention' },
  { id: 'rope', n: 5, title: b('RoPE lab', 'Laboratórium RoPE'), blurb: b('Position-dependent rotations and why only offsets matter.', 'Otáčania závislé od pozície a prečo záleží len na vzdialenosti.'), learn: 'attention' },
  { id: 'compare', n: 6, title: b('Dense vs MoE', 'Hustý model vs MoE'), blurb: b('The real profiles side by side plus a dimension-matched toy experiment.', 'Skutočné profily vedľa seba a hračkový pokus s rovnakými rozmermi.'), learn: 'moe' },
  { id: 'router', n: 7, title: b('Router lab', 'Laboratórium smerovača'), blurb: b('Logits → probabilities → top-k → renormalization → mixture.', 'Logity → pravdepodobnosti → top-k → prepočet → zmes.'), learn: 'moe' },
  { id: 'layers', n: 8, title: b('Layer traversal', 'Prechod vrstvami'), blurb: b('Walk a token through every block and residual add.', 'Preveďte token každým blokom a reziduálnym sčítaním.'), learn: 'block' },
  { id: 'cache', n: 9, title: b('Prefill / decode / cache', 'Prefill / decode / cache'), blurb: b('Cache growth, absolute positions, and a recomputation check.', 'Rast cache, absolútne pozície a kontrola úplným prepočtom.'), learn: 'generate' },
  { id: 'sampling', n: 10, title: b('Sampling lab', 'Laboratórium vzorkovania'), blurb: b('Temperature, top-k, top-p, seeded draws and the token loop.', 'Teplota, top-k, top-p, náhodný výber s osivom a slučka tokenov.'), learn: 'generate' },
  { id: 'training', n: 11, title: b('Training lab', 'Laboratórium trénovania'), blurb: b('Targets, loss, gradients, an SGD step, and the actual new loss.', 'Ciele, chyba, gradienty, krok SGD a skutočná nová chyba.'), learn: 'learning' },
  { id: 'memory', n: 12, title: b('Memory lab', 'Laboratórium pamäte'), blurb: b('Weights, quantization, K/V cache and training memory, with formulas.', 'Váhy, kvantizácia, K/V cache a pamäť na trénovanie, so vzorcami.'), learn: 'hardware' },
] as const;
