import { Suspense, lazy, useEffect, useState } from 'react';
import { useHashRoute } from './state/route';
import { ToyProvider } from './state/toyStore';
import { Explorer } from './explorer/Explorer';
import { Home } from './pages/Home';
import buildInfo from './generated/build-info.json';

const Labs = lazy(() => import('./labs/Labs'));
const Parameters = lazy(() => import('./pages/Parameters'));
const Catalog = lazy(() => import('./pages/Catalog'));
const Pipeline = lazy(() => import('./pages/Pipeline'));
const Family = lazy(() => import('./pages/Family'));
const Hybrid = lazy(() => import('./pages/Hybrid'));
const Tours = lazy(() => import('./pages/Tours'));
const Sources = lazy(() => import('./pages/Sources'));
const Glossary = lazy(() => import('./pages/Glossary'));
const About = lazy(() => import('./pages/About'));

const NAV = [
  ['', 'Home'],
  ['explore', 'Explorer'],
  ['labs', 'Labs'],
  ['parameters', 'Parameters'],
  ['catalog', 'Catalog'],
  ['pipeline', 'Tokens'],
  ['family', 'Hybrids'],
  ['tours', 'Tours'],
  ['sources', 'Sources'],
  ['glossary', 'Glossary'],
  ['about', 'About'],
] as const;

export function App() {
  const [route, nav] = useHashRoute();
  const [menu, setMenu] = useState(false);
  const [theme, setTheme] = useState<'system' | 'dark' | 'light'>(() => {
    try { return (localStorage.getItem('qa-theme') as 'dark' | 'light' | null) ?? 'system'; } catch { return 'system'; }
  });
  useEffect(() => {
    const el = document.documentElement;
    if (theme === 'system') el.removeAttribute('data-theme'); else el.setAttribute('data-theme', theme);
    try { localStorage.setItem('qa-theme', theme); } catch { /* storage may be unavailable */ }
  }, [theme]);
  useEffect(() => setMenu(false), [route.path.join('/')]);
  const top = route.path[0] ?? '';
  useEffect(() => {
    const titles: Record<string, string> = { '': 'Qwen Atlas', explore: 'Explorer · Qwen Atlas', labs: 'Labs · Qwen Atlas' };
    document.title = titles[top] ?? `${NAV.find((n) => n[0] === top)?.[1] ?? 'Qwen Atlas'} · Qwen Atlas`;
    if (top !== 'explore') window.scrollTo(0, 0);
  }, [top, route.path[1]]); // eslint-disable-line react-hooks/exhaustive-deps

  let page;
  switch (top) {
    case '': page = <Home />; break;
    case 'explore': page = <Explorer route={route} nav={nav} />; break;
    case 'labs': page = <Labs lab={route.path[1]} />; break;
    case 'parameters': page = <Parameters route={route} />; break;
    case 'catalog': page = <Catalog route={route} />; break;
    case 'pipeline': page = <Pipeline />; break;
    case 'family': page = <Family />; break;
    case 'hybrid': page = <Hybrid id={route.path[1]} />; break;
    case 'tours': page = <Tours id={route.path[1]} />; break;
    case 'sources': page = <Sources />; break;
    case 'glossary': page = <Glossary />; break;
    case 'about': page = <About />; break;
    default: page = (
      <div className="page"><h1>Page not found</h1><p>The address <code>{window.location.hash}</code> does not match a page.</p><a className="btn primary" href="#/">Go to the start page</a></div>
    );
  }
  const activeTop = top === 'hybrid' ? 'family' : top;
  return (
    <ToyProvider>
      <div className="app">
        <a className="skip" href="#main" onClick={(e) => { e.preventDefault(); document.getElementById('main')?.focus(); }}>Skip to content</a>
        <header className="topbar">
          <a className="brand" href="#/" aria-label="Qwen Atlas home">
            <svg width="26" height="26" viewBox="0 0 32 32" aria-hidden="true"><rect width="32" height="32" rx="7" fill="#0b1630" /><circle cx="16" cy="16" r="9" fill="none" stroke="#46d4e6" strokeWidth="2.4" /><circle cx="16" cy="16" r="3" fill="#b18cff" /></svg>
            Qwen Atlas
          </a>
          <button className="menu-btn small" aria-expanded={menu} aria-controls="mainnav" onClick={() => setMenu(!menu)}>☰ Menu</button>
          <nav id="mainnav" className={`nav${menu ? ' open' : ''}`} aria-label="Main">
            {NAV.map(([k, label]) => (
              <a key={k} href={`#/${k}`} aria-current={activeTop === k ? 'page' : undefined}>{label}</a>
            ))}
          </nav>
          <div className="spacer" />
          <label className="sr-only" htmlFor="theme">Color theme</label>
          <select id="theme" value={theme} onChange={(e) => setTheme(e.target.value as typeof theme)} style={{ minHeight: 32 }}>
            <option value="system">Auto theme</option>
            <option value="dark">Dark</option>
            <option value="light">Light</option>
          </select>
        </header>
        <main id="main" tabIndex={-1}>
          <Suspense fallback={<div className="page muted">Loading page…</div>}>{page}</Suspense>
        </main>
        {top !== 'explore' && (
          <footer className="foot">
            <span>Qwen Atlas is an independent educational project, not affiliated with or endorsed by Alibaba / the Qwen team. Model facts are cited to pinned primary sources.</span>
            <span className="mono" data-testid="build-id">build {buildInfo.shortCommit}{buildInfo.dirty ? ' (uncommitted changes)' : ''}</span>
          </footer>
        )}
      </div>
    </ToyProvider>
  );
}
