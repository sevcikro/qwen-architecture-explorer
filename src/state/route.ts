import { useCallback, useEffect, useState } from 'react';
import { z } from 'zod';
import { ALL_CORE, coreById, type CoreProfile } from '../models/profiles';
import { prod, tensorFamilies, type ModuleId, type TensorFamily } from '../models/shapes';

// ---------------------------------------------------------------------------
// Hash routing: #/path/segments?key=value — works on any static host and survives reloads.
// ---------------------------------------------------------------------------
export interface Route { path: string[]; query: URLSearchParams }

export function parseHash(hash: string): Route {
  const h = hash.replace(/^#/, '');
  const [p, q = ''] = h.split('?');
  return { path: p.split('/').filter(Boolean), query: new URLSearchParams(q) };
}

export function hrefFor(path: string, query?: Record<string, string | number | undefined>): string {
  const qs = new URLSearchParams();
  if (query) for (const [k, v] of Object.entries(query)) if (v !== undefined && v !== '') qs.set(k, String(v));
  const s = qs.toString();
  return `#/${path.replace(/^\//, '')}${s ? `?${s}` : ''}`;
}

export function useHashRoute(): [Route, (href: string, replace?: boolean) => void] {
  const [route, setRoute] = useState(() => parseHash(window.location.hash));
  useEffect(() => {
    const on = () => setRoute(parseHash(window.location.hash));
    window.addEventListener('hashchange', on);
    return () => window.removeEventListener('hashchange', on);
  }, []);
  const nav = useCallback((href: string, replace = false) => {
    if (replace) {
      history.replaceState(null, '', href);
      setRoute(parseHash(href));
    } else window.location.hash = href.replace(/^#/, '');
  }, []);
  return [route, nav];
}

// ---------------------------------------------------------------------------
// Explorer view state (semantic level + selection), validated on every load.
// ---------------------------------------------------------------------------
export const LEVELS = [
  { lv: 0, name: 'Scalar', long: 'Level 0 · One stored number' },
  { lv: 1, name: 'Unit', long: 'Level 1 · Weighted sum / gated channel' },
  { lv: 2, name: 'Matrix', long: 'Level 2 · Vector / matrix / tensor' },
  { lv: 3, name: 'Module', long: 'Level 3 · Functional module' },
  { lv: 4, name: 'Block', long: 'Level 4 · Decoder block' },
  { lv: 5, name: 'Model', long: 'Level 5 · Full model' },
  { lv: 6, name: 'Execution', long: 'Level 6 · Execution & learning' },
] as const;

export type ExModule = 'embedding' | 'attn_norm' | 'attention' | 'residual1' | 'ffn_norm' | 'ffn' | 'moe' | 'residual2' | 'final_norm' | 'lm_head' | 'sampling';
export const BLOCK_MODULES: ExModule[] = ['attn_norm', 'attention', 'residual1', 'ffn_norm', 'ffn', 'moe', 'residual2'];
export const MODEL_MODULES: ExModule[] = ['embedding', 'final_norm', 'lm_head', 'sampling'];

export const ExplorerSchema = z.object({
  m: z.string(),
  lv: z.number().int().min(0).max(6),
  b: z.number().int().min(0).optional(),
  mod: z.string().optional(),
  t: z.string().optional(),
  e: z.number().int().min(0).optional(),
  r: z.number().int().min(0).optional(),
  c: z.number().int().min(0).optional(),
  h: z.number().int().min(0).optional(),
});
export type ExplorerState = z.infer<typeof ExplorerSchema>;

export const DEFAULT_EXPLORER: ExplorerState = { m: 'qwen3-8b', lv: 5 };

export function moduleOfFamily(f: TensorFamily): ExModule {
  const map: Record<ModuleId, ExModule> = { embedding: 'embedding', attn_norm: 'attn_norm', attention: 'attention', ffn_norm: 'ffn_norm', ffn: 'ffn', router: 'moe', experts: 'moe', final_norm: 'final_norm', lm_head: 'lm_head' };
  return map[f.module];
}

export function familyRows(f: TensorFamily): number { return f.shape[0]; }
export function familyCols(f: TensorFamily): number | null { return f.shape.length > 1 ? f.shape[1] : null; }

/** Normalize a possibly-invalid state for a profile: clamp indices, drop incompatible fields, lower the level if needed. */
export function normalizeExplorer(raw: Partial<ExplorerState>, lang: 'en' | 'sk' = 'en'): { state: ExplorerState; profile: CoreProfile; warnings: string[] } {
  const warnings: string[] = [];
  const W = (en: string, sk: string) => (lang === 'sk' ? sk : en);
  let profile = coreById(raw.m ?? '');
  if (!profile) {
    if (raw.m) warnings.push(W(`Unknown model "${raw.m}"; showing Qwen3-8B.`, `Neznámy model „${raw.m}“; zobrazuje sa Qwen3-8B.`));
    profile = ALL_CORE[0];
  }
  const s: ExplorerState = { m: profile.id, lv: raw.lv ?? 5 };
  if (raw.b !== undefined) {
    if (raw.b < profile.layers) s.b = raw.b;
    else warnings.push(W(`Block ${raw.b} does not exist in ${profile.name} (${profile.layers} blocks); selection cleared.`, `Blok ${raw.b} v ${profile.name} neexistuje (${profile.layers} blokov); výber zrušený.`));
  }
  const fams = tensorFamilies(profile);
  let fam: TensorFamily | undefined;
  if (raw.t) {
    fam = fams.find((f) => f.key === raw.t);
    if (!fam) warnings.push(W(`Tensor "${raw.t}" does not exist in ${profile.name}; selection cleared.`, `Tenzor „${raw.t}“ v ${profile.name} neexistuje; výber zrušený.`));
  }
  if (raw.mod) {
    const valid = profile.kind === 'dense' ? raw.mod !== 'moe' : raw.mod !== 'ffn';
    if ([...BLOCK_MODULES, ...MODEL_MODULES].includes(raw.mod as ExModule) && valid) s.mod = raw.mod;
    else warnings.push(W(`Module "${raw.mod}" is not part of ${profile.name}; selection cleared.`, `Modul „${raw.mod}“ nie je súčasťou ${profile.name}; výber zrušený.`));
  }
  if (fam) {
    s.t = fam.key;
    s.mod = moduleOfFamily(fam);
    if (fam.scope !== 'model' && s.b === undefined) s.b = 0;
    if (fam.scope === 'model') delete s.b;
    if (fam.scope === 'expert') {
      const E = profile.moe!.numExperts;
      s.e = raw.e !== undefined && raw.e < E ? raw.e : 0;
      if (raw.e !== undefined && raw.e >= E) warnings.push(W(`Expert ${raw.e} does not exist (0–${E - 1}).`, `Expert ${raw.e} neexistuje (0–${E - 1}).`));
    }
    if (raw.r !== undefined) {
      if (raw.r < familyRows(fam)) s.r = raw.r;
      else warnings.push(W(`Row ${raw.r} is outside ${fam.label} (${familyRows(fam)} rows).`, `Riadok ${raw.r} je mimo ${fam.labelSk} (${familyRows(fam)} riadkov).`));
    }
    const cols = familyCols(fam);
    if (raw.c !== undefined && s.r !== undefined) {
      if (cols === null) s.c = undefined;
      else if (raw.c < cols) s.c = raw.c;
      else warnings.push(W(`Column ${raw.c} is outside ${fam.label} (${cols} columns).`, `Stĺpec ${raw.c} je mimo ${fam.labelSk} (${cols} stĺpcov).`));
    }
  } else if (raw.e !== undefined && s.mod === 'moe' && profile.moe && raw.e < profile.moe.numExperts) s.e = raw.e;
  if (raw.h !== undefined && raw.h < profile.hq) s.h = raw.h;
  if (BLOCK_MODULES.includes(s.mod as ExModule) && s.b === undefined) s.b = 0;
  // level must be supported by the selection
  let lv = raw.lv ?? 5;
  const maxDepth = () => {
    if (lv === 6 || lv === 5) return lv;
    if (lv <= 0 && s.t && s.r !== undefined && (s.c !== undefined || familyCols(fam!) === null)) return 0;
    if (lv <= 1 && s.t && s.r !== undefined) return 1;
    if (lv <= 2 && s.t) return 2;
    if (lv <= 3 && s.mod) return 3;
    if (lv <= 4 && s.b !== undefined) return 4;
    return 5;
  };
  const resolved = maxDepth();
  if (resolved !== lv) {
    warnings.push(W(`Level ${lv} needs a deeper selection; showing level ${resolved}.`, `Úroveň ${lv} potrebuje hlbší výber; zobrazuje sa úroveň ${resolved}.`));
    lv = resolved;
  }
  s.lv = lv;
  return { state: s, profile, warnings };
}

export function explorerFromQuery(q: URLSearchParams): Partial<ExplorerState> {
  const num = (k: string) => {
    const v = q.get(k);
    if (v === null || v === '') return undefined;
    const n = Number(v);
    return Number.isInteger(n) && n >= 0 ? n : undefined;
  };
  const raw: Partial<ExplorerState> = { m: q.get('m') ?? undefined, lv: num('lv'), b: num('b'), mod: q.get('mod') ?? undefined, t: q.get('t') ?? undefined, e: num('e'), r: num('r'), c: num('c'), h: num('h') };
  const parsed = ExplorerSchema.partial().safeParse(raw);
  return parsed.success ? parsed.data : {};
}

export const explorerHref = (s: Partial<ExplorerState>) => hrefFor('explore', s as Record<string, string | number | undefined>);

export const tensorElements = (f: TensorFamily) => prod(f.shape);
