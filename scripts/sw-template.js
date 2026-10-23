/* Qwen Atlas service worker (generated at build time by vite.config.ts from scripts/sw-template.js).
   Precaches the app so the course works offline; never touches requests to other sites. */
const VERSION = '__VERSION__';
const CACHE = `qwen-atlas-${VERSION}`;
const PRECACHE = __PRECACHE__;

self.addEventListener('install', (event) => {
  event.waitUntil(caches.open(CACHE).then((c) => c.addAll(PRECACHE)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => Promise.all(keys.filter((k) => k.startsWith('qwen-atlas-') && k !== CACHE).map((k) => caches.delete(k)))).then(() => self.clients.claim()),
  );
});

self.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'cache-all' && Array.isArray(event.data.urls)) {
    event.waitUntil(caches.open(CACHE).then((c) => c.addAll(event.data.urls)).then(() => event.source && event.source.postMessage({ type: 'cache-all-done' })));
  }
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return; // e.g. the optional in-browser model download: not cached here
  if (url.pathname.startsWith('/ort/')) return; // the 27 MB model runtime: left to the normal browser cache
  if (req.mode === 'navigate') {
    // network first, so a new release shows up at once; offline: the cached app shell
    event.respondWith(fetch(req).catch(() => caches.match('/index.html', { ignoreSearch: true, ignoreVary: true })));
    return;
  }
  if (url.pathname.startsWith('/assets/')) {
    event.respondWith(caches.match(req, { ignoreVary: true }).then((hit) => hit || fetch(req).then((res) => { if (res.ok) caches.open(CACHE).then((c) => c.put(req, res.clone())); return res; })));
    return;
  }
  // data and other files: serve the cached copy at once, refresh it in the background
  event.respondWith(caches.open(CACHE).then((c) => c.match(req, { ignoreVary: true }).then((hit) => {
    const net = fetch(req).then((res) => { if (res.ok) c.put(req, res.clone()); return res; }).catch(() => hit);
    return hit || net;
  })));
});
