import { Component, type ReactNode } from 'react';

export class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null };
  static getDerivedStateFromError(error: Error) {
    return { error };
  }
  render() {
    if (this.state.error) {
      return (
        <div style={{ maxWidth: 720, margin: '40px auto', padding: 16, fontFamily: 'system-ui' }}>
          <h1>Something went wrong</h1>
          <p>The view could not be displayed: <code>{this.state.error.message}</code></p>
          <p>Your link may reference a state this version cannot show. Reset to a known-good view:</p>
          <button onClick={() => { window.location.hash = '#/'; this.setState({ error: null }); }}>Go to the start page</button>{' '}
          <button onClick={() => { window.location.hash = '#/explore'; this.setState({ error: null }); }}>Open the explorer</button>
        </div>
      );
    }
    return this.props.children;
  }
}
