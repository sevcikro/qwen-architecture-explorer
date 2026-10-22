import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// here.now serves the site from "/". A GitHub Pages mirror under /REPO/ would need
// VITE_BASE=/qwen-architecture-explorer/ at build time (see docs/deployment.md).
export default defineConfig({
  base: process.env.VITE_BASE ?? '/',
  plugins: [react()],
  build: { outDir: 'dist', sourcemap: false, chunkSizeWarningLimit: 900 },
});
