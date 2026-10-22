// The five evidence classes required throughout the app.
export type EvidenceClass = 'verified' | 'derived' | 'simulation' | 'sample' | 'unknown';

export const EVIDENCE_META: Record<EvidenceClass, { label: string; short: string; glyph: string; description: string }> = {
  verified: {
    label: 'Verified architecture',
    short: 'Verified',
    glyph: '✓',
    description: 'Directly supported by an identified configuration file, checkpoint header or implementation at a pinned revision.',
  },
  derived: {
    label: 'Derived calculation',
    short: 'Derived',
    glyph: '=',
    description: 'Computed from verified facts. The formula and counting convention are shown next to the result.',
  },
  simulation: {
    label: 'Illustrative simulation',
    short: 'Simulation',
    glyph: '~',
    description: 'A reproducible calculation with reduced dimensions and synthetic (seeded) weights. True for this simulator, not an observed Qwen run.',
  },
  sample: {
    label: 'Actual checkpoint sample',
    short: 'Real sample',
    glyph: '◆',
    description: 'Values actually read from the named checkpoint file at a pinned revision, with byte-level provenance.',
  },
  unknown: {
    label: 'Unknown / not verified',
    short: 'Unknown',
    glyph: '?',
    description: 'A real information gap. The app says so instead of filling it with plausible-looking data.',
  },
};
