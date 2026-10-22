import { Suspense, lazy, useEffect, useState } from 'react';
import { useHashRoute } from './state/route';
import { ToyProvider } from './state/toyStore';
import { LangSwitch, useLang } from './i18n/i18n';
import type { UiKey } from './i18n/ui';
import { Explorer } from './explorer/Explorer';
import { Home } from './pages/Home';
import buildInfo from './generated/build-info.json';
import { version as APP_VERSION } from '../package.json';

const Labs = lazy(() => import('./labs/Labs'));
const Learn = lazy(() => import('./pages/Learn'));
const Parameters = lazy(() => import('./pages/Parameters'));
const Catalog = lazy(() => import('./pages/Catalog'));
const Pipeline = lazy(() => import('./pages/Pipeline'));
const Family = lazy(() => import('./pages/Family'));
const Hybrid = lazy(() => import('./pages/Hybrid'));
const Tours = lazy(() => import('./pages/Tours'));
const Sources = lazy(() => import('./pages/Sources'));
const Glossary = lazy(() => import('./pages/Glossary'));
const About = lazy(() => import('./pages/About'));

const NAV: [string, UiKey][] = [
  ['', 'nav_home'],
  ['learn', 'nav_learn'],
  ['explore', 'nav_explore'],
  ['labs', 'nav_labs'],
  ['parameters', 'nav_parameters'],
  ['catalog', 'nav_catalog'],
  ['pipeline', 'nav_pipeline'],
  ['family', 'nav_family'],
  ['tours', 'nav_tours'],
  ['sources', 'nav_sources'],
  ['glossary', 'nav_glossary'],
  ['about', 'nav_about'],
];

export function App() {
  const [route, nav] = useHashRoute();
  const { t, lang } = useLang();
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
  useEffect(() => {
    const el = document.querySelector('.topbar');
    if (!el) return;
    const set = () => document.documentElement.style.setProperty('--topbar-h', `${Math.ceil(el.getBoundingClientRect().height)}px`);
    const ro = new ResizeObserver(set);
    ro.observe(el);
    set();
    return () => ro.disconnect();
  }, []);
  const top = route.path[0] ?? '';
  useEffect(() => {
    const key = NAV.find((n) => n[0] === top)?.[1];
    document.title = top === '' || !key ? 'Qwen Atlas' : `${t(key)} · Qwen Atlas`;
    if (top !== 'explore') window.scrollTo(0, 0);
  }, [top, route.path[1], lang]); // eslint-disable-line react-hooks/exhaustive-deps

  let page;
  switch (top) {
    case '': page = <Home />; break;
    case 'explore': page = <Explorer route={route} nav={nav} />; break;
    case 'labs': page = <Labs lab={route.path[1]} />; break;
    case 'learn': page = <Learn id={route.path[1]} />; break;
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
      <div className="page"><h1>{t('not_found')}</h1><p>{t('not_found_text')} <code>{window.location.hash}</code></p><a className="btn primary" href="#/">{t('go_home')}</a></div>
    );
  }
  const activeTop = top === 'hybrid' ? 'family' : top;
  return (
    <ToyProvider>
      <div className="app">
        <a className="skip" href="#main" onClick={(e) => { e.preventDefault(); document.getElementById('main')?.focus(); }}>{t('skip')}</a>
        <header className="topbar">
          <a className="brand" href="#/" aria-label="Qwen Atlas home">
            <svg width="26" height="26" viewBox="0 0 32 32" aria-hidden="true"><rect width="32" height="32" rx="7" fill="#0b1630" /><circle cx="16" cy="16" r="9" fill="none" stroke="#46d4e6" strokeWidth="2.4" /><circle cx="16" cy="16" r="3" fill="#b18cff" /></svg>
            Qwen Atlas
          </a>
          <button className="menu-btn small" aria-expanded={menu} aria-controls="mainnav" onClick={() => setMenu(!menu)}>{t('menu')}</button>
          <nav id="mainnav" className={`nav${menu ? ' open' : ''}`} aria-label="Main">
            {NAV.map(([k, label]) => (
              <a key={k} href={`#/${k}`} aria-current={activeTop === k ? 'page' : undefined}>{t(label)}</a>
            ))}
          </nav>
          <LangSwitch />
          <label className="sr-only" htmlFor="theme">{t('theme')}</label>
          <select id="theme" value={theme} onChange={(e) => setTheme(e.target.value as typeof theme)} style={{ minHeight: 32 }}>
            <option value="system">{t('theme_auto')}</option>
            <option value="dark">{t('theme_dark')}</option>
            <option value="light">{t('theme_light')}</option>
          </select>
        </header>
        <main id="main" tabIndex={-1}>
          <Suspense fallback={<div className="page muted">{t('loading_page')}</div>}>{page}</Suspense>
        </main>
        {top !== 'explore' && (
          <footer className="foot">
            <span>{t('footer_independent')}</span>
            <span className="mono" data-testid="build-id">{t('footer_build')} {buildInfo.shortCommit}{buildInfo.dirty ? t('footer_dirty') : ''} · v{APP_VERSION}</span>
          </footer>
        )}
      </div>
    </ToyProvider>
  );
}
