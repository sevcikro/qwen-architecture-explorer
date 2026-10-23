// Renders the social preview image and the app icons into public/ (run: node scripts/make_images.mjs).
import { chromium } from 'playwright';
import { readFileSync, writeFileSync } from 'node:fs';
const b = await chromium.launch();
const og = await b.newPage({ viewport: { width: 1200, height: 630 } });
await og.setContent(readFileSync('scripts/assets/og.html', 'utf8'));
await og.screenshot({ path: 'public/og.png' });
const tpl = readFileSync('scripts/assets/icon.html', 'utf8');
for (const [name, size, maskable] of [['icon-192.png', 192, false], ['icon-512.png', 512, false], ['icon-maskable-512.png', 512, true]]) {
  const p = await b.newPage({ viewport: { width: size, height: size } });
  // maskable icons keep the logo inside the central 80 % safe zone on a full-bleed background
  const html = tpl.replace('RX', maskable ? '0' : '7').replace('R1', maskable ? '6.5' : '9').replace('R2', maskable ? '2.2' : '3').replace('SW', maskable ? '1.8' : '2.4').replace(/width="512" height="512"/, `width="${size}" height="${size}"`);
  await p.setContent(html);
  await p.locator('#i').screenshot({ path: `public/${name}`, omitBackground: true });
}
await b.close();
writeFileSync('scripts/assets/.done', 'ok');
console.log('images written');
