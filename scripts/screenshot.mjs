/**
 * Visual smoke test: drives the built app in headless Chrome, screenshots a
 * handful of states and reports any console / page errors.
 *
 * Usage:  npm run build && npm run preview   (in another shell)
 *         node scripts/screenshot.mjs
 */

import puppeteer from 'puppeteer-core';
import { mkdirSync } from 'node:fs';

const CHROME =
  process.env.CHROME_PATH ||
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const URL = process.env.APP_URL || 'http://localhost:4173/';
const OUT = 'shots';

mkdirSync(OUT, { recursive: true });

const clickByText = async (page, selector, text) => {
  const found = await page.evaluate(
    (sel, txt) => {
      const el = [...document.querySelectorAll(sel)].find((n) =>
        (n.textContent || '').replace(/\s+/g, ' ').trim().includes(txt),
      );
      if (!el) return false;
      el.click();
      return true;
    },
    selector,
    text,
  );
  if (!found) throw new Error(`could not find ${selector} containing "${text}"`);
  await new Promise((r) => setTimeout(r, 700));
};

const toggleLayer = async (page, name) => {
  await page.evaluate((txt) => {
    const label = [...document.querySelectorAll('.toggle')].find((n) =>
      n.textContent.includes(txt),
    );
    label.querySelector('input').click();
  }, name);
  await new Promise((r) => setTimeout(r, 500));
};

const shot = async (page, name) => {
  await page.screenshot({ path: `${OUT}/${name}.png` });
  console.log('  captured', `${OUT}/${name}.png`);
};

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: 'new',
  args: [
    '--no-sandbox',
    '--disable-gpu-sandbox',
    '--ignore-gpu-blocklist',
    '--enable-unsafe-swiftshader',
    '--use-gl=angle',
    '--use-angle=swiftshader',
    '--enable-webgl',
    '--hide-scrollbars',
  ],
});

const page = await browser.newPage();
await page.setViewport({ width: 1500, height: 940, deviceScaleFactor: 1 });

const problems = [];
page.on('console', (m) => {
  if (m.type() === 'error') problems.push('console.error: ' + m.text());
});
page.on('pageerror', (e) => problems.push('pageerror: ' + e.message));

try {
  await page.goto(URL, { waitUntil: 'networkidle0', timeout: 30000 });
  await page.waitForSelector('.viewport canvas', { timeout: 15000 });
  await new Promise((r) => setTimeout(r, 1200));

  await shot(page, '01-2d-default');

  await toggleLayer(page, 'Eigenvectors');
  await toggleLayer(page, 'Null space');
  await shot(page, '02-2d-eigen-nullspace');

  await toggleLayer(page, 'Transpose grid');
  await toggleLayer(page, 'Column space');
  await shot(page, '03-2d-all-layers');

  await clickByText(page, 'button', 'Show A⁻¹ instead of A');
  await shot(page, '04-2d-inverse');

  await clickByText(page, 'button', 'Back to drawing A');

  await clickByText(page, 'button', 'Start demo');
  await new Promise((r) => setTimeout(r, 600));
  await clickByText(page, '.step', 'A·B');
  await shot(page, '05-2d-composition');

  await clickByText(page, 'button', 'Exit');

  // rank-1 collapse: shows the flattening
  const choose = async (name) => {
    await page.evaluate((txt) => {
      const sel = document.querySelector('.select');
      const opt = [...sel.options].find((o) => o.textContent.includes(txt));
      sel.value = opt.value;
      sel.dispatchEvent(new Event('change', { bubbles: true }));
    }, name);
    await new Promise((r) => setTimeout(r, 900));
  };

  await choose('Rank-1 collapse');
  await shot(page, '06-2d-singular');

  // identity: the blue image lattice must land exactly on the square lattice
  await clickByText(page, 'button', 'Reset everything');
  await choose('Identity');
  await shot(page, '07-2d-identity');

  // 3D
  await clickByText(page, '.seg', '3 × 3');
  await Promise.race([
    page.waitForSelector('.viewport-3d-host canvas', { timeout: 15000 }),
    page.waitForSelector('.viewport-fallback', { timeout: 15000 }),
  ]);
  await new Promise((r) => setTimeout(r, 2000));
  await shot(page, '08-3d-default');

  const has3d = await page.$('.viewport-3d-host canvas');
  if (has3d) {
    await toggleLayer(page, 'Eigenvectors');
    await new Promise((r) => setTimeout(r, 600));
    await shot(page, '09-3d-eigen');

    await choose('Rotation about z');
    await shot(page, '10-3d-rotation');
  }
} catch (err) {
  problems.push('script: ' + err.message);
}

await browser.close();

if (problems.length) {
  console.log('\nPROBLEMS:');
  for (const p of [...new Set(problems)]) console.log('  - ' + p);
  process.exit(1);
}
console.log('\nno console or page errors');
