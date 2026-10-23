import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './styles.css';
import { App } from './App';
import { ErrorBoundary } from './components/ErrorBoundary';
import { LangProvider } from './i18n/i18n';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ErrorBoundary>
      <LangProvider>
        <App />
      </LangProvider>
    </ErrorBoundary>
  </StrictMode>,
);

// Remember the browser's install offer so the About page can show an "Install app" button.
window.addEventListener('beforeinstallprompt', (e) => { e.preventDefault(); (window as unknown as { qaInstall?: Event }).qaInstall = e; window.dispatchEvent(new Event('qa-install-ready')); });

// Offline support: only in the production build, only over https or on localhost.
if (import.meta.env.PROD && 'serviceWorker' in navigator && (location.protocol === 'https:' || location.hostname === 'localhost')) {
  window.addEventListener('load', () => { navigator.serviceWorker.register('/sw.js').catch(() => { /* offline mode is optional */ }); });
}

