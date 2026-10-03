# Matrix Transformations — Interactive Visualizer

An interactive website for learning what a matrix *actually does* to space.
Type in a **2×2**, **3×3** or **4×4** matrix and watch the lattice, basis
vectors and a vector `x` deform in real time — then switch on the determinant,
the inverse,
eigenvectors, the null space, the transpose, the **image ellipse** and the
**level sets of `xᵀAx`** (with a live **definiteness** verdict and a sign
field for the quadratic form), walk through a composition
`A·B` one step at a time, check whether **A² = A** (an idempotent matrix —
a projection, which does nothing the second time you apply it), or animate the
two big factorizations — the **spectral decomposition** `A = P D P⁻¹`
(`A = QΛQᵀ` when A is symmetric) and the **SVD** `A = UΣVᵀ` — step by step.
Load a **PCA dataset** (a seeded point cloud) and the matrix slot becomes its
covariance `S = Cov(x)`: step through **PCA** — centering, the data ellipse
(`√λ` semi-axes), the principal components as eigenvectors of `S`, projection
onto PC1 with its reconstruction error, and the `Qᵀ → ΛQᵀ` rotation that ties
PCA back to the spectral theorem and the SVD (`σᵢ(Xc) = √((n−1)λᵢ)`).
The **4×4 two-plane view** partitions the matrix into blocks
`M = [[A, B], [C, D]]` acting on `x = (x₁, x₂)` as `y₁ = A x₁ + B x₂`,
`y₂ = C x₁ + D x₂` across four linked panels — and turns the block stories
into demos: **Schur elimination** (`M → L → U`, giving `S = D − CA⁻¹B` and
`det M = det A · det S`) and **block multiply** (`M·N` one slot at a time,
each block the sum of two 2×2 products).

And when you want to *watch* rather than read: the **transition bar** scrubs
the whole picture from plain space (0%) to the full transformation (100%) —
press **▶** to replay the morph, or flip **Split** for a linked
**before | after** view (two canvas halves in 2D, scissored halves in 3D),
with ghosts marking where the picture started and where it is heading.

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
npm test           # linear-algebra smoke tests (determinant, inverse, eigen, blocks…)
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
| **Idempotence A² = A** | Step `I → A → A²`: if the grid refuses to move a second time, the matrix is a **projection**. A live badge and an `A²` vs `A² − A` readout tell you the verdict for any matrix you type. |
| **Schur elimination** (4×4) | Step `M → L → U → M = L·U`: the block shear `L = [I 0; CA⁻¹I]` has det 1 and clears the lower-left block `C`, leaving `U = L⁻¹M` block upper-triangular with the **Schur complement** `S = D − CA⁻¹B` in its corner — so `det M = det A · det S`. Sylvester gets a bonus: for symmetric `M`, `M ≻ 0 ⟺ A ≻ 0 and S ≻ 0`. Disabled with an explanation when block `A` is singular. |
| **Block multiply** (4×4) | Step `N → M → M·N`: each block of the product appears as two 2×2 products added together — `AE + BG`, `AF + BH`, `CE + DG`, `CF + DH` — and the finished `M·N` joins up at the end, with every slot checked against `matMul` in the Partitioned matrix card. *Run* loads a sample second factor while the N slot still holds the identity (a hand-edited N is never overwritten). |
| **Spectral decomposition A = P D P⁻¹** | Step `I → P⁻¹ → D·P⁻¹ → A` (`I → Qᵀ → ΛQᵀ → A` for symmetric A): undo to the eigenbasis, scale each coordinate by λ, map back. **Defective** matrices (a repeated eigenvalue with too few directions) and **complex** eigenvalues get an explanation and a disabled demo; the card shows `D`/`P` and the residual `‖PDP⁻¹ − A‖`. |
| **SVD A = UΣVᵀ** | Step `I → Vᵀ → ΣVᵀ → A`: turn the input, stretch by the singular values, turn into place — for *every* matrix, including singular ones where `σₙ = 0` (flagged on the card). White dashed markers follow each input direction `vᵢ` through the four steps onto `σᵢuᵢ`, landing on the ellipse semi-axes. |
| **Circle → ellipse** | The unit circle (ball in 3D) mapped by A — an ellipse whose semi-axes are the **singular values** `σ₁ ≥ σ₂(≥ σ₃)`. Area = `π·σ₁σ₂ = π|det|`. For symmetric A the axes *are* the eigenvectors and `σᵢ = |λᵢ|`. |
| **Level sets xᵀAx = c** | Contours of the quadratic form for `c = ±1, ±3`: ellipses when c agrees with the signs of the symmetric part's eigenvalues, hyperbolas when it doesn't. Their axes are the eigenvectors of `S = (A+Aᵀ)/2` — the directions where the cross term vanishes. |
| **Definiteness of xᵀAx** | A live verdict — **positive / negative (semi-)definite, indefinite or zero** — from the eigenvalues of `S = (A+Aᵀ)/2`, with Sylvester's criterion (the leading minors Δ₁, Δ₂, Δ₃) as a cross-check. Non-symmetric matrices included: only the symmetric part decides the sign. |
| **Quadratic form sign field** | Shade the plane by the *sign* of `xᵀAx`: warm where positive, cool where negative, fading out at the zero set — an indefinite matrix shows its null cone as a dark seam splitting the two. In 3D the level-set rings are recolored warm/cool by the sign of `c`. |
| **Transpose grid Aᵀ** | A second lattice from Aᵀ so you can compare the two side by side. |
| **Column space / Null space** | The reachable outputs `Col(A)` and the directions that are erased, `Null(A)`. Rank–nullity is shown live. |
| **PCA datasets** (Space & example) | Seeded point clouds — correlated / axis-aligned / isotropic blobs, **two clusters**, **different units (y ×5)**. Choosing one loads the samples into the viewport and puts their covariance `S = Cov(x)` in the Matrix A slot; the *Point cloud* card's sliders (correlation ρ, rotation θ, spreads, noise) regenerate the very same sample live, with A following as S. Editing A by hand detaches the dataset. |
| **Data points (+ mean)** | The cloud's samples as rose dots with the white mean marker `x̄` — which lands on the origin the moment the data is centered. |
| **Data ellipse 1σ · 2σ** | Gaussian contours: solid 1σ, dashed 2σ, semi-axes `√λᵢ` along the principal components (the *Circle → ellipse* layer instead uses `λᵢ`, because that one maps the unit circle through A). |
| **PCA projection** | Step `data → x − x̄ → PC1, PC2 → P₁`: center, take the eigenbasis of `S`, then collapse onto PC1 with the rank-1 projector `P₁ = v₁v₁ᵀ` (det = 0) — dashed residuals show exactly what PC1 threw away. |
| **PCA rotation** | Step `data → x − x̄ → Qᵀ → ΛQᵀ`: rotate the cloud into its own principal coordinates, then scale each axis by its variance — the spectral decomposition of `S` with the data along for the ride. |
| **Principal components** (inspector) | For a loaded cloud: `x̄`, per-PC `λ · √λ · share` rows, a stacked explained-variance bar (colored to match the PC arrows), the `trace = Σλ` cross-check and the rank-1 reconstruction error `λ₂(+λ₃)` — with preset-aware notes: two clusters are invisible to a linear method, different units need standardizing, an isotropic cloud has no preferred direction. |
| **3×3 view** | Orbit the transformed cube and lattice in three dimensions (drag to orbit, scroll to zoom, right-drag to pan). The PCA layers draw the point cloud, mean marker, residual tethers and two translucent √S ellipsoids. |
| **4×4 two-plane view** | `M = [[A, B], [C, D]]` acting on `x = (x₁, x₂)`, drawn as four linked panels: inputs `x₁/x₂` on the left, outputs `y₁/y₂` on the right, with the four blocks as coloured connectors (rose A, amber B, blue C, green D — a zero block fades its line out). Drag either amber tip to move `x₁` or `x₂`; every arrow inside an output panel is colored by the block that produced it, so `y₁ = A x₁ + B x₂` reads as two head-to-tail contributions. Per-block image lattices and unit squares carry each `det`; block-diagonal / upper / lower structure decouples the planes (`λ(M) = λ(A) ∪ λ(D)`). |
| **Partitioned matrix** (inspector, 4×4) | The four blocks colour-coded in a grid, the Schur complement `S = D − CA⁻¹B`, the live check `det M = det A · det S`, the trace rule `tr M = tr A + tr D`, the eigen-union line (or a note that coupling breaks it), **Sylvester via Schur**, the bridge to PCA's `var(y | x)` — and during *Block multiply*, all four slot sums with their labeled terms. |
| **Transition bar** (all views) | Scrub **0 → 100%** with the slider (or press **▶** to replay the morph from plain space to the active matrix over 2 s — dragging the slider pauses playback). In between, the viewport shows the transition itself: a ghost of the **unit square / identity lattice where the picture started**, dashed outlines of **where it is heading**, and **streaks** tracing each tracked tip's straight-line travel (`e₁, e₂(, e₃), (1,1), x`). **Show before** pins the start-state ghost even at 100%; **Split** gives a linked **before | after** view — two canvas halves sharing pan/zoom in 2D (drag the amber tip on the right half), two scissored passes through one orbit camera in 3D, hidden in 4×4 where the input/output panels already show both ends. Any matrix, preset, dataset or dimension change snaps the scrubber back to 100% — **▶** is the deliberate way to watch it again. |

### How the page is organized

* **Left column** — *Space & example* (dimension + presets, including the **PCA datasets** group), a *Point cloud* card with the dataset's live sliders (ρ, θ, spreads, noise — disabled until a cloud is loaded), the *Matrix A · Vector x* editors, *Layers* grouped under **Space / Structure / Quadratic form / Data**, and one **Animate** card whose chips (**Inverse · Composition · Idempotence · Schur elimination · Block multiply · Spectral · SVD · PCA projection · PCA rotation** — nine in all) open each demo's step controls (`I → B → A·B`, `I → A → A²`, `M → L → U`, `N → M → M·N`, `I → P⁻¹ → D·P⁻¹`, `I → Vᵀ → ΣVᵀ`, `data → x − x̄ → P₁`, …). Only one demo runs at a time; *Reset everything* is at the bottom. Starting a factorization demo switches on the layer that explains it (eigenvectors for Spectral, image ellipse for SVD, points + data ellipse + residuals for the PCA demos). In the 4×4 view the chips are *Inverse · Composition · Idempotence · Schur elimination · Block multiply · SVD*, the PCA datasets and Point cloud card step aside, and the editors split into four coloured blocks (`M` and `N` here instead of `A`/`B`).
* **Right column** — sticky section headers **Overview · Checks · Decompositions · Subspaces** with a jump-pill bar on top. Numbers, badges and a lead sentence stay visible; longer explanations fold behind `▸` expanders.
* **Viewport** — the colour legend is a translucent, non-interactive overlay in the bottom-left corner, just above the interaction hint; the **transition bar** (scrub / ▶ / Show before / Split) floats centred one row above the legend.

### Presets worth trying

* **Shear** — area unchanged, but a square becomes a parallelogram
* **Projection / Rank-1 collapse** — det = 0, the plane or space flattens
* **Projection onto line y = x / Oblique projection / Projection onto plane z = x + y** — idempotent matrices (`A² = A`): run the `I → A → A²` demo and watch the picture stop moving
* **Flip + stretch** — det < 0, so orientation flips as well as scaling — and *indefinite*: a saddle where `xᵀAx` takes both signs (turn on the sign field to see the seam)
* **Negative definite** — the upside-down bowl: `xᵀAx < 0` everywhere; only negative-c level sets draw
* **Quarter turn / Permutation (120° turn)** — complex eigenvalues in action (yet the form is still definite — the rotation's `S = cos θ · I`)
* **Symmetric stretch** — the cleanest eigenvalue demo
* **Correlated cloud / Axis-aligned cloud** (PCA datasets) — the canonical PC picture: PC1 along the long diagonal vs. along the axes; drag ρ/θ/spreads and watch the PCs follow
* **Isotropic cloud** — λ₁ ≈ λ₂: explained variance splits ~50/50 and the PC axes are arbitrary
* **Two clusters** — PCA's classic failure: one covariance ellipse stretched across the empty gap, PC1 pointing at nothing
* **Different units (y ×5)** — PC1 is just the y-axis until you standardize (correlation matrix)
* **Coupled planes** (4×4, default) — all four blocks share the work: each output plane is a mix of both inputs, so nothing decouples
* **Two independent planes** (4×4) — block-diagonal: the planes never talk — `λ(M) = λ(A) ∪ λ(D)` and the determinants just multiply
* **Block upper triangular** (4×4) — plane 1 drives both outputs; elimination is a no-op and `det M = det A · det D`
* **Shear between planes** (4×4) — `C` shears plane 2 as plane 1 slides past, with unit determinant
* **Covariance pairs** (4×4) — a symmetric block covariance: the Schur complement *is* a conditional covariance `var(blue | rose)`
* **Project onto plane 1** (4×4) — an idempotent block projection (`M² = M`): the second pass leaves the picture frozen

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
│   ├── matrix.ts        # det / inverse (closed form n ≤ 3, LU paths for n ≥ 4), trace, transpose, rref, rank, Col/Null bases, isIdempotent, symmetricPart, quadraticContours
│   ├── eigen.ts         # eigen() for any n (closed forms + ml-matrix EVD), singular values, ellipse axes, definiteness, SVD & spectral factors
│   ├── blocks.ts        # 4×4 partitioning: split/join, block structures, Schur complement, block L·U, products, eigen-union, trace/det rules, conditional variance
│   ├── pca.ts           # seeded point-cloud generators, covariance, PCA readouts, √S for the data ellipse, PC1 projector
│   └── lerp.ts          # MatrixTween — eased transitions between matrices; blendFromIdentity — the scrubbed I → M exposure
├── state/
│   ├── store.ts         # zustand store: matrix, vector, layers, modes, active dataset, dimension 2/3/4
│   └── hooks.ts         # useActiveMatrix() — which matrix is on screen, useDataFacts(), useBlockFacts()
├── presets.ts           # the example transformations (including the six 4×4 block presets)
├── components/
│   ├── Viewport2D.tsx   # canvas renderer + pan / zoom / drag interactions
│   ├── ViewportBlocks.tsx # four-panel 4×4 two-plane renderer (lazy-loaded chunk)
│   ├── Viewport3D.tsx   # three.js renderer (lazy-loaded chunk)
│   ├── MatrixInput.tsx  # editable matrix / vector cells
│   ├── ControlPanel.tsx # left sidebar: space/example, matrix·x, layers, animate
│   ├── TransitionBar.tsx # before→after scrubber overlay (slider, ▶, Show before, Split)
│   └── InfoPanel.tsx    # right sidebar: grouped cards + jump pills
tests/math.test.mjs      # npm test
scripts/screenshot.mjs   # npm run shots
```

three.js is loaded only when you switch to the 3×3 view, so the initial page
stays small (~76 kB gzipped).
