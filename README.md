# Matrix Transformations — Interactive Visualizer

An interactive website for learning what a matrix *actually does* to space.
Type in a **2×2** or **3×3** matrix and watch the lattice, basis vectors and a
vector `x` deform in real time — then switch on the determinant, the inverse,
eigenvectors, the null space, the transpose, or walk through a composition
`A·B` one step at a time.

Built with **Vite + React + TypeScript**, a hand-written **Canvas 2D** renderer
for the plane and **three.js** for three dimensions.

---

## Run it locally

```bash
npm install
npm run dev        # http://localhost:5173
```

Other scripts:

```bash
npm test           # linear-algebra smoke tests (determinant, inverse, eigen…)
npm run build      # type-check + production build into dist/
npm run preview    # serve the production build
npm run shots      # drive the built app in headless Chrome, write screenshots to shots/
```

---

## What you can explore

| Layer / control | What it shows geometrically |
|---|---|
| **Transformed grid** | The integer lattice of space, drawn before (faint square) and after (blue) the transformation. Straight lines stay straight — that is what "linear" means. |
| **Basis vectors** | `e₁, e₂(, e₃)` dashed, their images `Ae₁, Ae₂(, Ae₃)` solid. The columns of A *are* those images. |
| **Vector x → Ax** | Drag the amber handle in the 2D view. `Ax` is a weighted blend of the columns of A. |
| **Determinant region** | The image of the unit square / cube. Green = positive det, red = negative (orientation flipped), grey = zero (collapsed). |
| **Show A⁻¹** | Animates the transform back to the identity, with `A⁻¹A ≈ I` shown numerically. Singular matrices explain why no inverse exists. |
| **Eigenvectors** | Directions that only get *stretched*: `Av = λv`. Complex eigenvalues are reported as "no real eigenvector exists". Repeated eigenvalues show their multiplicity and eigenspace. |
| **Composition A·B** | Step `I → B → A·B` and see that the order matters (`A·B ≠ B·A`). |
| **Transpose grid Aᵀ** | A second lattice from Aᵀ so you can compare the two side by side. |
| **Column space / Null space** | The reachable outputs `Col(A)` and the directions that are erased, `Null(A)`. Rank–nullity is shown live. |
| **3×3 view** | Orbit the transformed cube and lattice in three dimensions (drag to orbit, scroll to zoom, right-drag to pan). |

### Presets worth trying

* **Shear** — area unchanged, but a square becomes a parallelogram
* **Projection / Rank-1 collapse** — det = 0, the plane or space flattens
* **Flip + stretch** — det < 0, so orientation flips as well as scaling
* **Quarter turn / Permutation (120° turn)** — complex eigenvalues in action
* **Symmetric stretch** — the cleanest eigenvalue demo

---

## Deploy to GitHub Pages

The build already uses `base: './'` in `vite.config.ts`, so the same `dist/`
works under any repository path.

1. Push this folder to a GitHub repository:

   ```bash
   git init
   git add .
   git commit -m "Matrix transformation visualizer"
   git branch -M main
   git remote add origin https://github.com/<you>/<repo>.git
   git push -u origin main
   ```

2. In the repository on GitHub: **Settings → Pages → Build and deployment →
   Source: GitHub Actions**.

3. Push again (or open **Actions → Deploy to GitHub Pages → Run workflow**).
   `.github/workflows/deploy.yml` installs, type-checks, builds and publishes
   `dist/`.

Your site will be at `https://<you>.github.io/<repo>/`.

> Want to serve it from the root of a custom domain or a user site? Remove
> `base: './'` from `vite.config.ts` and set `base: '/'`.

---

## Project structure

```
src/
├── math/
│   ├── matrix.ts        # det, inverse, transpose, rref, rank, Col/Null bases
│   ├── eigen.ts         # 2×2 closed form + 3×3 numerical decomposition
│   └── lerp.ts          # MatrixTween — eased transitions between matrices
├── state/
│   ├── store.ts         # zustand store: matrix, vector, layers, modes
│   └── hooks.ts         # useActiveMatrix() — which matrix is on screen
├── presets.ts           # the example transformations
├── components/
│   ├── Viewport2D.tsx   # canvas renderer + pan / zoom / drag interactions
│   ├── Viewport3D.tsx   # three.js renderer (lazy-loaded chunk)
│   ├── MatrixInput.tsx  # editable matrix / vector cells
│   ├── ControlPanel.tsx # left sidebar
│   └── InfoPanel.tsx    # right sidebar: det, eigen, subspaces…
tests/math.test.mjs      # npm test
scripts/screenshot.mjs   # npm run shots
```

three.js is loaded only when you switch to the 3×3 view, so the initial page
stays small (~76 kB gzipped).
