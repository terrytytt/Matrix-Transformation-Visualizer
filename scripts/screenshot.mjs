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

/** Scroll-zoom the 2D viewport about its centre (factor = e^(-deltaY·0.0015)). */
const zoom2d = async (page, deltaY) => {
  await page.$eval(
    '.viewport canvas',
    (el, d) => {
      const r = el.getBoundingClientRect();
      el.dispatchEvent(
        new WheelEvent('wheel', {
          deltaY: d,
          clientX: r.left + r.width / 2,
          clientY: r.top + r.height / 2,
          bubbles: true,
          cancelable: true,
        }),
      );
    },
    deltaY,
  );
  await new Promise((r) => setTimeout(r, 500));
};

/**
 * Set the transition slider (a React controlled range input: the native
 * value setter first, then input/change so onChange picks it up).
 */
const setScrub = async (page, pct) => {
  await page.$eval('.tb-slider', (el, p) => {
    const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
    setter.call(el, String(p));
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }, pct);
  await new Promise((r) => setTimeout(r, 600));
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

  await clickByText(page, '.chip', 'Inverse');
  await clickByText(page, 'button', 'Show A⁻¹ instead of A');
  await shot(page, '04-2d-inverse');

  await clickByText(page, 'button', 'Back to drawing A');

  await clickByText(page, '.chip', 'Composition');
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

  // idempotent matrix: A² = A, so the A² step must not move the grid
  await choose('Projection onto line y = x');
  await clickByText(page, '.chip', 'Idempotence');
  await clickByText(page, 'button', 'Animate I → A → A²');
  await new Promise((r) => setTimeout(r, 700));
  await clickByText(page, '.step', 'A²');
  await shot(page, '11-2d-idempotent');
  await clickByText(page, 'button', 'Exit');

  // eigenvalues ↔ ellipse: shear has λ = 1 but σ = golden ratio, axes at 45°
  await clickByText(page, 'button', 'Reset everything');
  await choose('Shear (horizontal)');
  await toggleLayer(page, 'Circle → ellipse');
  await toggleLayer(page, 'Level sets xᵀAx = c');
  await shot(page, '12-2d-ellipse');

  // indefinite form: warm/cool sign field split by the null cone, plus the
  // hyperbolic level sets. (ellipse off so the field + contours read clearly)
  await choose('Flip + stretch');
  await toggleLayer(page, 'Circle → ellipse');
  await toggleLayer(page, 'Quadratic form xᵀAx');
  await shot(page, '14-2d-indefinite');

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

    // ellipse ↔ eigenvalues in 3D: setDim resets the layers, so re-toggle.
    // Eigenvectors were left on at step 09 — turn them off so the σ arrows
    // and the ellipsoid are easy to read.
    await choose('Symmetric stretch 3D');
    await toggleLayer(page, 'Eigenvectors');
    await toggleLayer(page, 'Circle → ellipse');
    await toggleLayer(page, 'Level sets xᵀAx = c');
    await new Promise((r) => setTimeout(r, 600));
    await shot(page, '13-3d-ellipse');

    // indefinite in 3D: rings recolored by the sign of c (warm c>0, cool c<0).
    // Reflection (xy-plane) has S = diag(1,1,-1) → both sign families draw.
    await choose('Reflection (xy-plane)');
    await toggleLayer(page, 'Circle → ellipse');
    await toggleLayer(page, 'Quadratic form xᵀAx');
    await new Promise((r) => setTimeout(r, 600));
    await shot(page, '15-3d-indefinite');
  }

  // spectral decomposition, 2D: back to Symmetric stretch (setDim resets the
  // layers), run the demo and jump to the ΛQᵀ step — eigen layer comes on
  // automatically with the demo.
  await clickByText(page, '.seg', '2 × 2');
  await new Promise((r) => setTimeout(r, 800));
  await clickByText(page, '.chip', 'Spectral');
  await clickByText(page, 'button', 'Run the spectral demo');
  await clickByText(page, '.step', 'ΛQᵀ');
  await shot(page, '16-2d-spectral');

  // SVD, 2D: switch chips (exits the spectral demo), drop the eigen arrows so
  // the white σ markers read against the ellipse. ΣVᵀ step puts the markers
  // on the ellipse semi-axes.
  await clickByText(page, '.chip', 'SVD');
  await toggleLayer(page, 'Eigenvectors');
  await clickByText(page, 'button', 'Run the SVD demo');
  await clickByText(page, '.step', 'ΣVᵀ');
  await shot(page, '17-2d-svd');

  // spectral + SVD in 3D (setDim clears the demo flags and resets layers).
  await clickByText(page, '.seg', '3 × 3');
  await Promise.race([
    page.waitForSelector('.viewport-3d-host canvas', { timeout: 15000 }),
    page.waitForSelector('.viewport-fallback', { timeout: 15000 }),
  ]);
  await new Promise((r) => setTimeout(r, 2000));
  const has3d2 = await page.$('.viewport-3d-host canvas');
  if (has3d2) {
    await clickByText(page, '.chip', 'Spectral');
    await clickByText(page, 'button', 'Run the spectral demo');
    await clickByText(page, '.step', 'ΛQᵀ');
    await shot(page, '18-3d-spectral');

    await clickByText(page, '.chip', 'SVD');
    await toggleLayer(page, 'Eigenvectors');
    await clickByText(page, 'button', 'Run the SVD demo');
    await clickByText(page, '.step', 'ΣVᵀ');
    await shot(page, '19-3d-svd');
  }

  // ---------------- PCA: datasets, projection, rotation ----------------
  // 2D cloud with its 1σ/2σ contour and the PC arrows (selecting a dataset
  // auto-enables the points + data-ellipse layers; setDim gave a clean slate).
  await clickByText(page, '.seg', '2 × 2');
  await new Promise((r) => setTimeout(r, 800));
  await choose('Correlated cloud');
  await toggleLayer(page, 'Vector x → Ax');
  await toggleLayer(page, 'Determinant region');
  await toggleLayer(page, 'Transformed grid'); // dense S-lattice buries the cloud
  await toggleLayer(page, 'Basis vectors');
  await toggleLayer(page, 'Eigenvectors'); // PC arrows over the cloud
  await shot(page, '20-2d-pca-cloud');

  // projection payoff at P₁: grid + points collapse onto PC1, dashed residuals
  // (the lattice comes back so its collapse onto the PC1 line is visible)
  await toggleLayer(page, 'Transformed grid');
  await clickByText(page, '.chip', 'PCA projection');
  await clickByText(page, 'button', 'Run the PCA projection demo');
  await clickByText(page, '.step', 'P₁');
  await new Promise((r) => setTimeout(r, 900)); // let the eased cloud settle
  await shot(page, '21-2d-pca-projection');

  // 3D pair: the cloud itself, then the ΛQᵀ rotation payoff
  await clickByText(page, '.seg', '3 × 3');
  await Promise.race([
    page.waitForSelector('.viewport-3d-host canvas', { timeout: 15000 }),
    page.waitForSelector('.viewport-fallback', { timeout: 15000 }),
  ]);
  await new Promise((r) => setTimeout(r, 2000));
  const has3d3 = await page.$('.viewport-3d-host canvas');
  if (has3d3) {
    await choose('Correlated cloud');
    await toggleLayer(page, 'Vector x → Ax');
    await toggleLayer(page, 'Determinant region');
    await toggleLayer(page, 'Transformed grid'); // dense S-lattice buries the cloud
    await toggleLayer(page, 'Basis vectors');
    await shot(page, '22-3d-pca-cloud');

    await clickByText(page, '.chip', 'PCA rotation');
    await clickByText(page, 'button', 'Run the PCA rotation demo');
    await clickByText(page, '.step', 'ΛQᵀ');
    await new Promise((r) => setTimeout(r, 900));
    await shot(page, '23-3d-pca-rotation');
  }

  // the classic failure mode: two clusters → one ellipse stretched across
  // the empty gap, PC1 pointing at nothing.
  await clickByText(page, '.seg', '2 × 2');
  await new Promise((r) => setTimeout(r, 800));
  await choose('Two clusters');
  await toggleLayer(page, 'Vector x → Ax');
  await toggleLayer(page, 'Determinant region');
  await toggleLayer(page, 'Transformed grid'); // let the spanning ellipse read
  await toggleLayer(page, 'Basis vectors');
  await toggleLayer(page, 'Eigenvectors');
  await zoom2d(page, -400); // the two blobs + spanning ellipse deserve the frame
  await shot(page, '24-2d-pca-fails');

  // ---------------- 4×4 two-plane block view ----------------
  // setDim clears demos, layers and the dataset — we land on "Coupled planes".
  await clickByText(page, '.seg', '4 × 4');
  await new Promise((r) => setTimeout(r, 1600));
  await shot(page, '25-4d-coupled');

  await choose('Two independent planes');
  await shot(page, '26-4d-block-diagonal');

  // Schur elimination: back to the coupled default, run the demo, jump to U —
  // L has cleared C and U carries the Schur complement S in its corner.
  await clickByText(page, 'button', 'Reset everything');
  await new Promise((r) => setTimeout(r, 600));
  await clickByText(page, '.chip', 'Schur elimination');
  await clickByText(page, 'button', 'Run the Schur elimination demo');
  await clickByText(page, '.step', 'U');
  await new Promise((r) => setTimeout(r, 800));
  await shot(page, '27-4d-schur');
  await clickByText(page, 'button', 'Exit');

  // block multiply: N → M → M·N; the Partitioned card lists the four sums.
  await clickByText(page, '.chip', 'Block multiply');
  await clickByText(page, 'button', 'Run the block multiply demo');
  await clickByText(page, '.step', 'M·N');
  await new Promise((r) => setTimeout(r, 800));
  await shot(page, '28-4d-block-mul');
  await clickByText(page, 'button', 'Exit');

  // conditional covariance: symmetric preset → Sylvester row + var(y | x) story
  await choose('Covariance pairs');
  await shot(page, '29-4d-covariance-pairs');

  // ---------------- transition: scrub, ghosts, split view ----------------
  // Back to 2×2 (setDim restores the default layers) and scrub to ~45%, so
  // the before-square ghost, destination outlines and travel streaks show.
  await clickByText(page, '.seg', '2 × 2');
  await new Promise((r) => setTimeout(r, 900));
  await setScrub(page, 45);
  await shot(page, '30-2d-transition-scrub');

  // split view: plain space on the left, the full transformation on the right
  await setScrub(page, 100);
  await clickByText(page, '.tb-toggle', 'Split');
  await new Promise((r) => setTimeout(r, 600));
  await shot(page, '31-2d-split');

  // the same split in 3D (scissored double pass through one orbit camera)
  await clickByText(page, '.seg', '3 × 3');
  await Promise.race([
    page.waitForSelector('.viewport-3d-host canvas', { timeout: 15000 }),
    page.waitForSelector('.viewport-fallback', { timeout: 15000 }),
  ]);
  await new Promise((r) => setTimeout(r, 2000));
  const has3d4 = await page.$('.viewport-3d-host canvas');
  if (has3d4) await shot(page, '32-3d-split');

  // 4×4: outputs mid-scrub — identity + destination lattice ghosts and
  // streaks from each contribution's plain-space start.
  await clickByText(page, '.seg', '4 × 4');
  await new Promise((r) => setTimeout(r, 1600));
  await setScrub(page, 50);
  await shot(page, '33-4d-transition-scrub');
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
