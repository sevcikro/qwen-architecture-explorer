import type { CoreProfile } from '../models/profiles';
import { tensorFamilies } from '../models/shapes';

/** Inverse of tensorName: which family / block / expert a checkpoint tensor name belongs to. */
export function locateTensor(p: CoreProfile, name: string): { key: string; b?: number; e?: number } | null {
  for (const f of tensorFamilies(p)) {
    const re = new RegExp(`^${f.pathTemplate.replace(/\./g, '\\.').replace('{layer}', '(\\d+)').replace('{expert}', '(\\d+)')}$`);
    const m = name.match(re);
    if (m) return { key: f.key, b: f.scope === 'model' ? undefined : Number(m[1]), e: f.scope === 'expert' ? Number(m[2]) : undefined };
  }
  return null;
}
