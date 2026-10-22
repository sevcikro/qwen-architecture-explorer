import { useCallback, useEffect, useLayoutEffect, useRef, useState, type KeyboardEvent } from 'react';
import type { GraphSpec, NodeSpec } from './graph';

interface Xf { k: number; x: number; y: number }

/**
 * Viewport zoom/pan (wheel, drag, pinch, buttons, +/-/0 keys). Viewport zoom only changes how large things
 * are drawn and how much detail is revealed; opening a deeper SEMANTIC level is a separate, explicit action.
 */
export function GraphCanvas({ spec, selected, onSelect, onOpen, fitKey }: { spec: GraphSpec; selected?: string; onSelect: (id: string) => void; onOpen: (n: NodeSpec) => void; fitKey: string }) {
  const wrap = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ w: 800, h: 600 });
  const [xf, setXf] = useState<Xf>({ k: 1, x: 0, y: 0 });
  const pointers = useRef(new Map<number, { x: number; y: number }>());
  const drag = useRef<{ x: number; y: number; moved: boolean; pinch?: number } | null>(null);
  const nodeRefs = useRef(new Map<string, SVGGElement>());

  const fit = useCallback(() => {
    const k = Math.min(size.w / spec.width, size.h / spec.height) * 0.96;
    setXf({ k, x: (size.w - spec.width * k) / 2, y: Math.max(8, (size.h - spec.height * k) / 2) });
  }, [size.w, size.h, spec.width, spec.height]);

  useLayoutEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setSize({ w: el.clientWidth || 800, h: el.clientHeight || 600 }));
    ro.observe(el);
    setSize({ w: el.clientWidth || 800, h: el.clientHeight || 600 });
    return () => ro.disconnect();
  }, []);
  useEffect(() => { fit(); }, [fitKey, size.w, size.h]); // eslint-disable-line react-hooks/exhaustive-deps

  const zoomAt = useCallback((factor: number, cx: number, cy: number) => {
    setXf((t) => {
      const k = Math.max(0.15, Math.min(8, t.k * factor));
      const f = k / t.k;
      return { k, x: cx - (cx - t.x) * f, y: cy - (cy - t.y) * f };
    });
  }, []);

  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const r = el.getBoundingClientRect();
      zoomAt(Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0015)), e.clientX - r.left, e.clientY - r.top);
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [zoomAt]);

  const onPointerDown = (e: React.PointerEvent) => {
    pointers.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pointers.current.size === 2) {
      const [a, b] = [...pointers.current.values()];
      drag.current = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, moved: true, pinch: Math.hypot(a.x - b.x, a.y - b.y) };
    } else drag.current = { x: e.clientX, y: e.clientY, moved: false };
  };
  const onPointerMove = (e: React.PointerEvent) => {
    if (!pointers.current.has(e.pointerId) || !drag.current) return;
    pointers.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
    const r = wrap.current!.getBoundingClientRect();
    if (pointers.current.size === 2 && drag.current.pinch) {
      const [a, b] = [...pointers.current.values()];
      const d = Math.hypot(a.x - b.x, a.y - b.y);
      zoomAt(d / drag.current.pinch, (a.x + b.x) / 2 - r.left, (a.y + b.y) / 2 - r.top);
      drag.current.pinch = d;
      return;
    }
    const dx = e.clientX - drag.current.x, dy = e.clientY - drag.current.y;
    if (!drag.current.moved && Math.hypot(dx, dy) < 5) return;
    if (!drag.current.moved) (e.currentTarget as Element).setPointerCapture?.(e.pointerId);
    drag.current = { x: e.clientX, y: e.clientY, moved: true };
    setXf((t) => ({ ...t, x: t.x + dx, y: t.y + dy }));
  };
  const onPointerUp = (e: React.PointerEvent) => {
    pointers.current.delete(e.pointerId);
    if (pointers.current.size === 0) setTimeout(() => (drag.current = null), 0);
  };
  const wasDrag = () => drag.current?.moved === true;

  const focusDir = (from: NodeSpec, dx: number, dy: number) => {
    const c = (n: NodeSpec) => [n.x + n.w / 2, n.y + n.h / 2];
    const [fx, fy] = c(from);
    let best: NodeSpec | null = null, bestScore = Infinity;
    for (const n of spec.nodes) {
      if (n.id === from.id) continue;
      const [x, y] = c(n);
      const vx = x - fx, vy = y - fy;
      const along = vx * dx + vy * dy;
      if (along <= 1) continue;
      const across = Math.abs(vx * dy - vy * dx);
      const s = along + 2.5 * across;
      if (s < bestScore) { bestScore = s; best = n; }
    }
    if (best) nodeRefs.current.get(best.id)?.focus();
  };

  const onKey = (e: KeyboardEvent, n: NodeSpec) => {
    const dirs: Record<string, [number, number]> = { ArrowDown: [0, 1], ArrowUp: [0, -1], ArrowLeft: [-1, 0], ArrowRight: [1, 0] };
    if (dirs[e.key]) { e.preventDefault(); focusDir(n, ...dirs[e.key]); }
    else if (e.key === 'Enter') { e.preventDefault(); if (n.open) onOpen(n); else onSelect(n.id); }
    else if (e.key === ' ') { e.preventDefault(); onSelect(n.id); }
  };

  const onCanvasKey = (e: KeyboardEvent) => {
    if (e.key === '+' || e.key === '=') zoomAt(1.25, size.w / 2, size.h / 2);
    else if (e.key === '-' || e.key === '_') zoomAt(0.8, size.w / 2, size.h / 2);
    else if (e.key === '0') fit();
  };

  const k = xf.k;
  const lodLines = k >= 1.6 ? 3 : k >= 1.12 ? 2 : k >= 0.72 ? 1 : 0;
  return (
    <div className="canvas-wrap" ref={wrap} onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={onPointerUp} onPointerCancel={onPointerUp} onKeyDown={onCanvasKey}>
      <svg className="canvas" role="group" aria-label={`${spec.title}. ${spec.description} Use Tab or arrow keys to move between parts, Enter to open a deeper level, Space to inspect.`}>
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M0 0 L10 5 L0 10 z" fill="var(--text-3)" />
          </marker>
        </defs>
        <g transform={`translate(${xf.x} ${xf.y}) scale(${k})`}>
          {spec.edges.map((e) => (
            <g key={e.id}>
              <polyline className={`edge${e.resid ? ' resid' : ''}`} points={e.pts.map((p) => p.join(',')).join(' ')} markerEnd="url(#arrow)" />
              {e.label && lodLines >= 1 && (
                <text className="edge-label" x={(e.pts[0][0] + e.pts[1][0]) / 2 + 6} y={(e.pts[0][1] + e.pts[1][1]) / 2 + 3}>{e.label}</text>
              )}
            </g>
          ))}
          {spec.nodes.map((n) => (
            <Node key={n.id} n={n} lod={lodLines} selected={selected === n.id} refCb={(el) => (el ? nodeRefs.current.set(n.id, el) : nodeRefs.current.delete(n.id))}
              onClick={() => !wasDrag() && onSelect(n.id)} onDouble={() => n.open && onOpen(n)} onKey={(e) => onKey(e, n)} />
          ))}
        </g>
      </svg>
      <div className="zoom-ctl" role="toolbar" aria-label="Viewport zoom (does not change semantic level)">
        <button aria-label="Zoom in" title="Zoom in (+)" onClick={() => zoomAt(1.3, size.w / 2, size.h / 2)}>+</button>
        <button aria-label="Zoom out" title="Zoom out (−)" onClick={() => zoomAt(1 / 1.3, size.w / 2, size.h / 2)}>−</button>
        <button aria-label="Fit view" title="Fit (0)" onClick={fit}>⤢</button>
      </div>
      <div className="lod-hint" aria-live="polite">
        Viewport {Math.round(k * 100)}% · {lodLines === 0 ? 'zoom in to reveal shapes and parameter counts' : lodLines === 1 ? 'showing shapes; zoom further for parameter counts' : lodLines === 2 ? 'showing counts; zoom further for checkpoint names' : 'full detail'} · double-click / Enter opens a deeper level
      </div>
    </div>
  );
}

const ROLE_GLYPH: Record<string, string> = { param: 'W', norm: 'γ', act: 'a', state: '⧉', op: 'ƒ', group: '' };

function Node({ n, lod, selected, onClick, onDouble, onKey, refCb }: { n: NodeSpec; lod: number; selected: boolean; onClick: () => void; onDouble: () => void; onKey: (e: KeyboardEvent) => void; refCb: (el: SVGGElement | null) => void }) {
  const style: Record<string, { fill: string; stroke: string; dash?: string; rx: number }> = {
    param: { fill: 'color-mix(in srgb, var(--weight) 20%, var(--panel))', stroke: 'var(--weight)', rx: 8 },
    norm: { fill: 'color-mix(in srgb, var(--norm) 16%, var(--panel))', stroke: 'var(--norm)', rx: 8 },
    act: { fill: 'var(--bg)', stroke: 'var(--act)', rx: 16 },
    state: { fill: 'var(--bg)', stroke: 'var(--state)', rx: 6 },
    op: { fill: 'color-mix(in srgb, var(--panel) 60%, transparent)', stroke: 'var(--text-3)', dash: '5 4', rx: 8 },
    group: { fill: 'color-mix(in srgb, var(--panel) 55%, transparent)', stroke: 'var(--line)', rx: 12 },
  };
  const s = style[n.role];
  const isGroupBox = n.role === 'group' && (n.id === 'stack' || n.id === 'experts');
  const small = n.w < 60;
  const subs = (n.sub ?? []).slice(0, lod);
  const label = `${n.label}${n.sub?.length ? `. ${n.sub.join('. ')}` : ''}${n.open ? '. Press Enter to open.' : ''}`;
  const ty = isGroupBox ? n.y + 20 : n.y + n.h / 2 - (subs.length * 12) / 2 + 4;
  return (
    <g ref={refCb} className={`node${selected ? ' selected' : ''}`} tabIndex={0} role="button" aria-label={label} aria-pressed={selected}
      onClick={(e) => { e.stopPropagation(); onClick(); }} onDoubleClick={(e) => { e.stopPropagation(); onDouble(); }} onKeyDown={onKey} data-node={n.id}>
      {n.shape === 'circle' ? (
        <circle className="node-box" cx={n.x + n.w / 2} cy={n.y + n.h / 2} r={n.h / 2} fill={s.fill} stroke={s.stroke} strokeWidth={2} strokeDasharray={s.dash} />
      ) : (
        <rect className="node-box" x={n.x} y={n.y} width={n.w} height={n.h} rx={s.rx} fill={s.fill} stroke={s.stroke} strokeWidth={isGroupBox ? 1.2 : 2} strokeDasharray={s.dash} />
      )}
      {n.role === 'state' && <rect x={n.x + 3} y={n.y + 3} width={n.w - 6} height={n.h - 6} rx={4} fill="none" stroke={s.stroke} strokeWidth={1} />}
      {!small && ROLE_GLYPH[n.role] && n.shape !== 'circle' && (
        <text x={n.x + 7} y={n.y + 13} fontSize={10} fill={s.stroke} fontFamily="var(--mono)" aria-hidden="true">{ROLE_GLYPH[n.role]}</text>
      )}
      <text className="node-label" x={isGroupBox ? n.x + 16 : n.x + n.w / 2} y={ty} textAnchor={isGroupBox ? 'start' : 'middle'} fontSize={small ? 10 : n.shape === 'circle' ? 20 : 13}>{n.label}</text>
      {!isGroupBox && subs.map((t, i) => (
        <text key={i} className="node-sub" x={n.x + n.w / 2} y={ty + 13 + i * 12} textAnchor="middle">{t}</text>
      ))}
      {isGroupBox && subs[0] && <text className="node-sub" x={n.x + n.w - 14} y={n.y + 20} textAnchor="end">{subs[0]}</text>}
      {n.open && !small && n.shape !== 'circle' && <text x={n.x + n.w - 8} y={n.y + 13} fontSize={10} textAnchor="end" fill="var(--text-3)" aria-hidden="true">⤵</text>}
    </g>
  );
}
