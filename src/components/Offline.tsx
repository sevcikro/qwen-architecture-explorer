import { useEffect, useState } from 'react';
import { useLang } from '../i18n/i18n';

// Offline use and installing the site as an app (About page).
type InstallEvent = Event & { prompt: () => Promise<void>; userChoice: Promise<{ outcome: string }> };
const EXTRA_FILES = ['/data/offsets-qwen3-30b-a3b.json', '/og.png', '/icon-maskable-512.png'];

export function OfflinePanel() {
  const { lang } = useLang();
  const T = (en: string, sk: string) => (lang === 'sk' ? sk : en);
  const [sw, setSw] = useState<'checking' | 'active' | 'none' | 'unsupported'>('checking');
  const [saved, setSaved] = useState<'idle' | 'saving' | 'done'>('idle');
  const [install, setInstall] = useState<InstallEvent | null>(() => ((window as unknown as { qaInstall?: InstallEvent }).qaInstall ?? null));
  const standalone = typeof window !== 'undefined' && window.matchMedia?.('(display-mode: standalone)').matches;
  useEffect(() => {
    if (!('serviceWorker' in navigator)) { setSw('unsupported'); return; }
    navigator.serviceWorker.getRegistration().then((r) => setSw(r?.active ? 'active' : 'none'), () => setSw('none'));
    navigator.serviceWorker.ready.then(() => setSw('active'), () => { /* never ready: stays 'none' */ });
    const onReady = () => setInstall((window as unknown as { qaInstall?: InstallEvent }).qaInstall ?? null);
    const onMsg = (e: MessageEvent) => { if (e.data?.type === 'cache-all-done') setSaved('done'); };
    window.addEventListener('qa-install-ready', onReady);
    navigator.serviceWorker.addEventListener('message', onMsg);
    return () => { window.removeEventListener('qa-install-ready', onReady); navigator.serviceWorker.removeEventListener('message', onMsg); };
  }, []);
  const saveAll = async () => {
    const reg = await navigator.serviceWorker.getRegistration();
    if (!reg?.active) return;
    setSaved('saving');
    reg.active.postMessage({ type: 'cache-all', urls: EXTRA_FILES });
  };
  return (
    <div className="card" data-testid="offline-panel">
      <p style={{ marginTop: 0 }}>{T('After your first visit, the whole course (chapters, labs, explorer, diagrams and data) is stored in this browser and keeps working without internet. Only the optional live model needs a connection the first time.', 'Po prvej návšteve sa celý kurz (kapitoly, laboratóriá, prieskumník, diagramy a údaje) uloží v tomto prehliadači a funguje aj bez internetu. Pripojenie potrebuje len voliteľný živý model, a to len prvýkrát.')}</p>
      <p className="mono" style={{ fontSize: '0.86rem' }} data-testid="offline-status">
        {sw === 'active' ? T('✓ Offline copy active', '✓ Offline kópia je aktívna') : sw === 'checking' ? T('Checking…', 'Overujem…') : sw === 'unsupported' ? T('This browser does not support offline mode.', 'Tento prehliadač nepodporuje režim offline.') : T('Offline copy not active yet (it starts after the page has loaded once over https).', 'Offline kópia ešte nie je aktívna (spustí sa, keď sa stránka raz načíta cez https).')}
      </p>
      <div className="row">
        <button disabled={sw !== 'active' || saved !== 'idle'} onClick={saveAll}>{saved === 'done' ? T('✓ Everything saved', '✓ Všetko uložené') : saved === 'saving' ? T('Saving…', 'Ukladám…') : T('Also save the large catalog file (1.5 MB)', 'Uložiť aj veľký súbor katalógu (1,5 MB)')}</button>
        {install && !standalone && <button className="primary" onClick={async () => { await install.prompt(); setInstall(null); }}>{T('Install as an app', 'Nainštalovať ako aplikáciu')}</button>}
      </div>
      {!install && !standalone && <p className="faint" style={{ fontSize: '0.84rem' }}>{T('To install it as an app: in Chrome or Edge use the install icon in the address bar or “Install app” in the menu; on iPhone and iPad tap Share → “Add to Home Screen”; on Android, Chrome’s menu → “Add to Home screen”.', 'Inštalácia ako aplikácia: v Chrome alebo Edge použite ikonu inštalácie v paneli s adresou alebo „Inštalovať aplikáciu“ v ponuke; na iPhone a iPad ťuknite na Zdieľať → „Pridať na plochu“; na Androide ponuka Chrome → „Pridať na plochu“.')}</p>}
      {standalone && <p className="faint">{T('You are using the installed app.', 'Používate nainštalovanú aplikáciu.')}</p>}
    </div>
  );
}
