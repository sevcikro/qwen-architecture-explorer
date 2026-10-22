import { Component, type ReactNode } from 'react';

export class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null };
  static getDerivedStateFromError(error: Error) {
    return { error };
  }
  render() {
    if (this.state.error) {
      // Class component: read the language directly (same sources the LangProvider uses).
      let sk = false;
      try { sk = new URLSearchParams(window.location.hash.split('?')[1] ?? '').get('lang') === 'sk' || (document.documentElement.lang === 'sk'); } catch { /* ignore */ }
      const T = (en: string, s: string) => (sk ? s : en);
      return (
        <div style={{ maxWidth: 720, margin: '40px auto', padding: 16, fontFamily: 'system-ui' }}>
          <h1>{T('Something went wrong', 'Niečo sa pokazilo')}</h1>
          <p>{T('The view could not be displayed:', 'Pohľad sa nepodarilo zobraziť:')} <code>{this.state.error.message}</code></p>
          <p>{T('Your link may reference a state this version cannot show. Reset to a known-good view:', 'Váš odkaz možno odkazuje na stav, ktorý táto verzia nevie zobraziť. Prejdite na funkčný pohľad:')}</p>
          <button onClick={() => { window.location.hash = '#/'; this.setState({ error: null }); }}>{T('Go to the start page', 'Prejsť na úvodnú stránku')}</button>{' '}
          <button onClick={() => { window.location.hash = '#/explore'; this.setState({ error: null }); }}>{T('Open the explorer', 'Otvoriť prieskumník')}</button>
        </div>
      );
    }
    return this.props.children;
  }
}
