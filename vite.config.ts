import { defineConfig, type Plugin } from 'vite';
import { createReadStream, readFileSync, readdirSync, rmSync, statSync } from 'node:fs';
import { join } from 'node:path';
import react from '@vitejs/plugin-react';

// here.now serves the site from "/". A GitHub Pages mirror under /REPO/ would need
// VITE_BASE=/qwen-architecture-explorer/ at build time (see docs/deployment.md).
// Generates dist/sw.js: precache list = the built JS/CSS/fonts + app files + data files under 250 kB.
function serviceWorker(): Plugin {
  return {
    name: 'qa-service-worker',
    apply: 'build',
    generateBundle(_opts, bundle) {
      const info = JSON.parse(readFileSync('src/generated/build-info.json', 'utf8')) as { shortCommit: string; dirty: boolean; builtAt: string };
      const assets = Object.keys(bundle).filter((f) => /\.(js|css|woff2)$/.test(f)).map((f) => `/${f}`);
      const data = readdirSync('public/data').filter((f) => statSync(join('public/data', f)).size < 250_000).map((f) => `/data/${f}`);
      const precache = ['/', '/index.html', '/manifest.webmanifest', '/icon-192.png', '/icon-512.png', ...assets, ...data];
      const version = `${info.shortCommit}${info.dirty ? `-dirty-${info.builtAt}` : ''}`;
      const source = readFileSync('scripts/sw-template.js', 'utf8').replace('__VERSION__', version).replace('__PRECACHE__', JSON.stringify(precache));
      this.emitFile({ type: 'asset', fileName: 'sw.js', source });
    },
  };
}

// The ONNX runtime used by the optional in-browser model is served by this site, not a CDN.
const ORT_DIR = 'node_modules/onnxruntime-web/dist';
const ORT_FILES = ['ort-wasm-simd-threaded.asyncify.mjs', 'ort-wasm-simd-threaded.asyncify.wasm'];
function onnxRuntimeFiles(): Plugin {
  return {
    name: 'qa-onnx-runtime',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const f = req.url?.startsWith('/ort/') ? req.url.slice(5).split('?')[0] : null;
        if (!f || !ORT_FILES.includes(f)) return next();
        res.setHeader('Content-Type', f.endsWith('.wasm') ? 'application/wasm' : 'text/javascript');
        createReadStream(join(ORT_DIR, f)).pipe(res);
      });
    },
    generateBundle() {
      for (const f of ORT_FILES) this.emitFile({ type: 'asset', fileName: `ort/${f}`, source: readFileSync(join(ORT_DIR, f)) });
    },
    closeBundle() {
      // The worker bundle also emits a hashed copy of the runtime; the worker loads /ort/ instead, so drop it.
      const dir = join('dist', 'assets');
      for (const f of readdirSync(dir)) if (/^ort-wasm-simd-threaded.*\.wasm$/.test(f)) rmSync(join(dir, f));
    },
  };
}

export default defineConfig({
  base: process.env.VITE_BASE ?? '/',
  plugins: [react(), serviceWorker(), onnxRuntimeFiles()],
  worker: { format: 'es' },
  build: { outDir: 'dist', sourcemap: false, chunkSizeWarningLimit: 900 },
});
