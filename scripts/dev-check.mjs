/** Dev-server smoke check: loads the page twice (StrictMode double-mount) and reports errors. */
import puppeteer from 'puppeteer-core';

const CHROME =
  process.env.CHROME_PATH ||
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const URL = process.env.APP_URL || 'http://localhost:5173/';

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
await page.setViewport({ width: 1500, height: 940 });

const problems = [];
page.on('console', (m) => {
  if (m.type() === 'error') problems.push('console.error: ' + m.text());
});
page.on('pageerror', (e) => problems.push('pageerror: ' + e.message));
page.on('requestfailed', (r) =>
  problems.push('requestfailed: ' + r.url() + ' ' + r.failure()?.errorText),
);

try {
  await page.goto(URL, { waitUntil: 'networkidle0', timeout: 30000 });
  await page.waitForSelector('.viewport canvas', { timeout: 15000 });
  await new Promise((r) => setTimeout(r, 1500));

  // exercise a few interactions in dev mode
  await page.evaluate(() => {
    const label = [...document.querySelectorAll('.toggle')].find((n) =>
      n.textContent.includes('Eigenvectors'),
    );
    label.querySelector('input').click();
  });
  await new Promise((r) => setTimeout(r, 400));

  const cells = await page.$$('.cell');
  console.log('  editable cells:', cells.length);
  if (cells.length !== 6) problems.push('expected 6 cells in 2D, got ' + cells.length);

  await page.type('.cell', '5');
  await new Promise((r) => setTimeout(r, 600));

  // switch to 3D (exercises StrictMode remount + lazy chunk)
  await page.evaluate(() => {
    [...document.querySelectorAll('.seg')].find((n) => n.textContent.includes('3 × 3')).click();
  });
  await page.waitForSelector('.viewport-3d-host canvas, .viewport-fallback', {
    timeout: 15000,
  });
  await new Promise((r) => setTimeout(r, 1500));

  const cells3 = await page.$$('.cell');
  console.log('  editable cells (3D):', cells3.length);
  if (cells3.length !== 12) problems.push('expected 12 cells in 3D, got ' + cells3.length);

  // and back to 2D — remounts the canvas
  await page.evaluate(() => {
    [...document.querySelectorAll('.seg')].find((n) => n.textContent.includes('2 × 2')).click();
  });
  await new Promise((r) => setTimeout(r, 1200));
} catch (err) {
  problems.push('script: ' + err.message);
}

await browser.close();

if (problems.length) {
  console.log('\nPROBLEMS:');
  for (const p of [...new Set(problems)]) console.log('  - ' + p);
  process.exit(1);
}
console.log('\ndev server: no errors across 2D <-> 3D round trip');
