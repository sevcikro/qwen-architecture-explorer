import { useEffect, useId, useRef, useState } from 'react';
import { GLOSSARY } from '../content/glossary';
import { useLang } from '../i18n/i18n';

// A glossary term inside chapter text: dotted underline; hover (mouse) or tap/Enter opens a small definition card.

export function GlossTerm({ term, text }: { term: string; text: string }) {
  const { lang, tr } = useLang();
  const g = GLOSSARY.find((x) => x.term.en === term);
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState<{ left: number; top: number; above: boolean }>({ left: 0, top: 0, above: false });
  const btn = useRef<HTMLButtonElement>(null);
  const id = useId();
  const place = () => {
    const r = btn.current?.getBoundingClientRect();
    if (!r) return;
    const w = Math.min(320, window.innerWidth - 24);
    const left = Math.max(12, Math.min(r.left, window.innerWidth - w - 12));
    const above = r.bottom + 180 > window.innerHeight && r.top > 200;
    setPos({ left, top: above ? r.top - 8 : r.bottom + 6, above });
  };
  useEffect(() => {
    if (!open) return;
    const close = (e: Event) => { if (e.type === 'keydown' && (e as KeyboardEvent).key !== 'Escape') return; setOpen(false); };
    const outside = (e: MouseEvent) => { if (!btn.current?.parentElement?.contains(e.target as Node)) setOpen(false); };
    window.addEventListener('scroll', close, true);
    window.addEventListener('keydown', close);
    window.addEventListener('mousedown', outside);
    return () => { window.removeEventListener('scroll', close, true); window.removeEventListener('keydown', close); window.removeEventListener('mousedown', outside); };
  }, [open]);
  if (!g) return <>{text}</>;
  return (
    <span className="gloss" onMouseEnter={(e) => { if ((e.nativeEvent as PointerEvent).pointerType !== 'touch') { place(); setOpen(true); } }} onMouseLeave={() => setOpen(false)}>
      <button ref={btn} type="button" className="gloss-term" aria-expanded={open} aria-describedby={open ? id : undefined} onClick={() => { place(); setOpen(!open); }}>{text}</button>
      {open && (
        <span id={id} role="tooltip" className="gloss-pop" style={{ left: pos.left, top: pos.top, transform: pos.above ? 'translateY(-100%)' : undefined }}>
          <strong>{tr(g.term)}</strong>
          <span className="gloss-def">{tr(g.def)}</span>
          <a href={`#/glossary?q=${encodeURIComponent(g.term.en.split(' ')[0])}`}>{lang === 'sk' ? 'Celý slovník →' : 'Full glossary →'}</a>
        </span>
      )}
    </span>
  );
}
