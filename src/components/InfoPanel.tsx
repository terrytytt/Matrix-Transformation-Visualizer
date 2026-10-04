import { useMemo } from 'react';
import { Dim, useApp } from '../state/store';
import { useActiveMatrix, useBlockFacts, useDataFacts, useDecompFacts } from '../state/hooks';
import {
  Matrix,
  Vector,
  clean,
  columnSpaceBasis,
  det,
  inverse,
  isIdempotent,
  isSymmetric,
  matMul,
  nullSpaceBasis,
  rank,
  symmetricPart,
} from '../math/matrix';
import { Definiteness, definiteness, eigen, singularValues } from '../math/eigen';
import { useEquationBlocks } from './EquationOverlay';
import {
  BLOCK_COLORS,
  blockProducts,
  blockTranspose,
  conditionalVariance2,
} from '../math/blocks';
import { presetsFor } from '../presets';

const n = (x: number, digits = 4) => {
  if (!Number.isFinite(x)) return '—';
  const r = Math.round(x * 10 ** digits) / 10 ** digits;
  return Object.is(r, -0) ? '0' : String(r);
};

const vecText = (v: Vector) => `[${clean(v).map((x) => n(x)).join(', ')}]`;

const matrixText = (M: Matrix) =>
  M.map((row) => row.map((x) => n(x).padStart(7)).join('  ')).join('\n');

const matrixCompact = (M: Matrix) =>
  `[${M.map((row) => row.map((x) => n(x)).join(', ')).join(' | ')}]`;

const sameMatrix = (A: Matrix, B: Matrix) =>
  A.length === B.length &&
  A.every((row, i) => row.every((x, j) => Math.abs(x - B[i][j]) < 1e-9));

/** Right-panel sections, in reading order — the pill nav jumps between them. */
const GROUPS = [
  { id: 'overview', label: 'Overview' },
  { id: 'checks', label: 'Checks' },
  { id: 'decompositions', label: 'Decompositions' },
  { id: 'subspaces', label: 'Subspaces' },
] as const;

/**
 * Explanatory prose: the first sentence stays visible, the rest folds behind
 * a native `<details>` (summary = the lead sentence). Single-sentence copy
 * stays fully open.
 */
function Prose({ text, className = '' }: { text: string; className?: string }) {
  const m = text.match(/^(.+?[.!?…])\s+(.+)$/s);
  if (!m) return <p className={`muted small ${className}`}>{text}</p>;
  return (
    <details className={`expander ${className}`}>
      <summary>{m[1]}</summary>
      <p className="muted small">{m[2]}</p>
    </details>
  );
}

/** Verdict badge for the Definiteness card, keyed by classification. */
const defBadge: Record<Definiteness, { text: string; cls: string }> = {
  positiveDefinite: { text: 'xᵀAx > 0', cls: 'badge badge-pos' },
  negativeDefinite: { text: 'xᵀAx < 0', cls: 'badge badge-warn' },
  positiveSemidefinite: { text: 'xᵀAx ≥ 0', cls: 'badge badge-pos' },
  negativeSemidefinite: { text: 'xᵀAx ≤ 0', cls: 'badge badge-warn' },
  indefinite: { text: 'sign varies', cls: 'badge badge-amber' },
  zero: { text: 'xᵀAx ≡ 0', cls: 'badge' },
};

/** One-line interpretation of each verdict. */
const defVerb: Record<Definiteness, string> = {
  positiveDefinite:
    'Every x ≠ 0 stores positive energy: xᵀAx > 0 — the form is a bowl opening upward, and A is invertible.',
  negativeDefinite:
    'Every x ≠ 0 gives negative energy: xᵀAx < 0 — an upside-down bowl (equivalently, −A is positive definite).',
  positiveSemidefinite:
    'Never negative, but zero along a direction — the bowl slides along the floor instead of lifting off it.',
  negativeSemidefinite:
    'Never positive, but zero along a direction — an inverted bowl that just touches zero.',
  indefinite:
    'Some directions give energy, others take it away — a saddle point, where the form changes sign.',
  zero: 'The form vanishes identically: xᵀAx = 0 for every vector (try the Quarter turn).',
};

const sub = (n: number) => ['', '₁', '₂', '₃', '₄', '₅'][n] ?? String(n);

/** Short form of a definiteness verdict (for the Sylvester row). */
const shortDef = (d: Definiteness): string =>
  ({
    positiveDefinite: '≻ 0',
    negativeDefinite: '≺ 0',
    positiveSemidefinite: '⪰ 0',
    negativeSemidefinite: '⪯ 0',
    indefinite: 'indefinite',
    zero: 'zero',
  })[d];

/** PC colours mirror the eigenvector layer, so bar segments and arrows agree. */
const PC_COLORS = ['#c084fc', '#f472b6', '#818cf8'];

function detCaption(d: number, size: number) {
  const unit = size === 2 ? 'area' : size === 4 ? '4-volume' : 'volume';
  if (Math.abs(d) < 1e-9) {
    return `Singular: the ${unit} collapses to 0 — the shape is squashed onto a lower dimension.`;
  }
  const shape = size === 2 ? 'square' : size === 4 ? '4-cube' : 'cube';
  const flip = d < 0 ? ' Orientation is flipped, so a mirror happened too.' : '';
  return `The unit ${shape} becomes a shape with ${unit} × ${n(Math.abs(d))}.${flip}`;
}

/** Human description of A, taken from the preset catalogue when it matches. */
function describe(A: Matrix, dim: Dim): string {
  const hit = presetsFor(dim).find(
    (p) =>
      p.matrix.length === A.length &&
      p.matrix.every((row, i) => row.every((x, j) => Math.abs(x - A[i][j]) < 1e-9)),
  );
  if (hit) return hit.description;
  return 'A custom matrix — edit the numbers and watch the geometry respond.';
}

export function InfoPanel() {
  const dim = useApp((s) => s.dim);
  const A = useApp((s) => s.matrix);
  const B = useApp((s) => s.matrixB);
  const vector = useApp((s) => s.vector);
  const showInverse = useApp((s) => s.showInverse);
  const compActive = useApp((s) => s.compActive);
  const eqMode = useApp((s) => s.eqMode);
  const toggleEqMode = useApp((s) => s.toggleEqMode);
  const eqBlocks = useEquationBlocks();
  const idemActive = useApp((s) => s.idemActive);
  const decomp = useApp((s) => s.decomp);
  const presetName = useApp((s) => s.presetName);
  const dataset = useApp((s) => s.dataset);
  const active = useActiveMatrix();
  const { svd, spectral } = useDecompFacts();
  const dataFacts = useDataFacts(dataset);
  const pca = dataFacts?.pca ?? null;
  /** Everything the Partitioned matrix card shows (4×4 only, else null). */
  const blockFacts = useBlockFacts(A);
  /** What this panel calls the main matrix: M in the 4×4 block view. */
  const nm = dim === 4 ? 'M' : 'A';
  /** Input component names the equation card prints: x₁ … xₙ. */
  const eqVars = Array.from({ length: dim }, (_, i) => `x${['₁', '₂', '₃', '₄'][i]}`).join(', ');

  const bf = blockFacts;
  const defA = useMemo(() => (bf ? definiteness(bf.blocks.A) : null), [bf]);
  const defS = useMemo(() => (bf?.S ? definiteness(bf.S) : null), [bf]);
  const products = useMemo(
    () => (decomp === 'blockMul' ? blockProducts(A, B) : null),
    [decomp, A, B],
  );
  const structLabel = bf
    ? (
        {
          diagonal: 'block-diagonal',
          upper: 'block upper triangular',
          lower: 'block lower triangular',
          coupled: 'coupled',
        } as const
      )[bf.structure]
    : '';

  const facts = useMemo(() => {
    const d = det(A);
    const r = rank(A);
    const inv = inverse(A);
    const entries = eigen(A);
    const cols = columnSpaceBasis(A);
    const nulls = nullSpaceBasis(A);
    // Ellipse / level-set facts.
    const sv = singularValues(A);
    const sym = isSymmetric(A);
    const symEntries = eigen(symmetricPart(A));
    const symZero = symEntries.every((e) => Math.abs(e.real) < 1e-9);
    const def = definiteness(A);
    return { d, r, inv, entries, cols, nulls, sv, sym, symEntries, symZero, def };
  }, [A]);

  const Ax = useMemo(
    () => A.map((row) => row.reduce((s, a, k) => s + a * (vector[k] ?? 0), 0)),
    [A, vector],
  );
  const AB = useMemo(() => matMul(A, B), [A, B]);
  const BA = useMemo(() => matMul(B, A), [B, A]);
  // The top card describes whatever is on screen (A, A⁻¹, B, A·B…),
  // the cards below always describe A as edited on the left.
  const shown = useMemo(() => ({ d: det(active.matrix), r: rank(active.matrix) }), [active.matrix]);
  const shownIdem = useMemo(() => isIdempotent(active.matrix), [active.matrix]);

  const A2 = useMemo(() => matMul(A, A), [A]);
  const idem = useMemo(() => isIdempotent(A), [A]);
  const residual = useMemo(
    () => A.map((row, i) => row.map((x, j) => A2[i][j] - x)),
    [A, A2],
  );

  const complex = facts.entries.filter((e) => Math.abs(e.imag) > 1e-9);
  const def = facts.def;
  /** 1+1 Schur complement of the covariance — var(y | x), dim 2 only. */
  const condVar = dim === 2 && pca ? conditionalVariance2(pca.S) : null;
  const maxAbsLambda = complex.length
    ? null
    : facts.entries.reduce<number | null>(
        (m, e) => (m === null ? Math.abs(e.real) : Math.max(m, Math.abs(e.real))),
        null,
      );

  const jump = (id: string) => {
    document
      .getElementById(`grp-${id}`)
      ?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  return (
    <>
      <nav className="pill-nav" aria-label="Jump to section">
        {GROUPS.map((g) => (
          <button key={g.id} type="button" className="pill" onClick={() => jump(g.id)}>
            {g.label}
          </button>
        ))}
      </nav>

      <div className="panel-stack">
        {/* ------------------------------- overview --------------------------- */}
        <h3 className="group-header" id="grp-overview">
          Overview
        </h3>

        <section className="card">
          <h2 className="card-title">
            Currently drawn <span className="badge badge-strong">{active.label}</span>
            {active.singular && <span className="badge badge-warn">no inverse</span>}
            {shownIdem && <span className="badge">{nm}² = {nm}</span>}
          </h2>
          <p className="card-sub">{presetName ?? 'Custom matrix'}</p>
          <pre className="matrix-readout">{matrixText(active.matrix)}</pre>
          <div className="stat-grid">
            <Stat label="det" value={n(shown.d)} tone={shown.d < 0 ? 'neg' : 'pos'} />
            <Stat label="rank" value={`${shown.r} / ${dim}`} />
          </div>
          <Prose text={detCaption(shown.d, dim)} />
          <Prose
            text={
              decomp === 'spectral'
                ? `Step through ${spectral.orthogonal ? 'I → Qᵀ → ΛQᵀ' : 'I → P⁻¹ → D·P⁻¹'}: undo to the eigenbasis, scale by the eigenvalues, then rotate back — the product is A again.`
                : decomp === 'svd'
                  ? 'Step through I → Vᵀ → ΣVᵀ: turn the input, stretch along the axes, then turn into place — the product is A again.'
                  : decomp === 'pcaProjection'
                    ? 'Center → PC axes → P₁: step through how PCA keeps only PC1 — the grid flattens onto the line it spans (det = 0) and the dashed residuals measure what was lost.'
                    : decomp === 'pcaRotation'
                      ? 'Center → Qᵀ → ΛQᵀ: rotate the cloud into its own principal coordinates, then scale each axis by its variance — the spectral decomposition of S, with the data along for the ride.'
                      : compActive
                        ? dim === 4
                          ? 'N is applied first, then M. Step through I → N → M·N to watch the second transformation build on the result of the first.'
                          : 'B is applied first, then A. Step through I → B → A·B to watch the second transformation build on the result of the first.'
                        : idemActive
                          ? `Step through I → ${nm} → ${nm}²: if the picture stops moving between steps 2 and 3, then ${nm}² = ${nm} and the matrix is idempotent.`
                          : decomp === 'schur'
                            ? 'Step through M → L → U: the block shear L clears the lower-left block C, U = L⁻¹M comes out block upper-triangular with the Schur complement S in its corner — and det M = det A · det S because det L = 1.'
                            : decomp === 'blockMul'
                              ? 'Each block of M·N is a sum of two 2×2 products — step through N → M → M·N and watch the four block products add up in the Partitioned matrix card.'
                              : dataset
                            ? 'A is the covariance matrix S of the loaded point cloud: variance along PCᵢ lives in λᵢ, the eigenvectors are the principal components, and the Definiteness card proves S is positive semidefinite for free.'
                            : describe(A, dim)
            }
          />
        </section>

        <section className="card">
          <h2 className="card-title">{nm} applied to x</h2>
          <div className="kv">
            <span className="k">x</span>
            <span className="v mono">{vecText(vector)}</span>
          </div>
          <div className="kv">
            <span className="k accent-a">{nm} x</span>
            <span className="v mono">{vecText(Ax)}</span>
          </div>
          <p className="muted small">
            Each column of {nm} records where a basis vector lands, so {nm} x is a weighted blend of
            the columns:{' '}
            <span className="mono">
              {dim === 4
                ? 'x·col₁ + y·col₂ + u·col₃ + v·col₄'
                : `x₁·col₁ + x₂·col₂${dim === 3 ? ' + x₃·col₃' : ''}`}
            </span>
            .
          </p>
          <button
            type="button"
            className={eqMode ? 'btn btn-toggle on' : 'btn btn-toggle'}
            onClick={toggleEqMode}
            aria-pressed={eqMode}
          >
            {eqMode ? 'Hide equations' : 'Show as equations'}
          </button>
        </section>

        {eqBlocks && (
          <section className="card">
            <h2 className="card-title">The map as equations</h2>
            {eqBlocks.map((b) => (
              <div className="eq-block" key={b.title}>
                <span className="eq-title mono">{b.title}</span>
                <pre className="matrix-readout">{b.rows.join('\n')}</pre>
              </div>
            ))}
            <p className="muted small">
              {compActive
                ? `The second block is the first one with p = ${dim === 4 ? 'N' : 'B'} x substituted in — the entries of the product emerge by plugging one linear map into the other.`
                : dim === 4
                  ? 'These hold for the generic point x = (x, y, u, v): the four rows produce the output components y₁ … y₄, and every vector obeys them — not just the amber inputs.'
                  : `These hold for the generic point x = (${eqVars}): every vector obeys them, not just the amber one on the grid.`}
            </p>
          </section>
        )}

        {/* -------------------------------- checks ---------------------------- */}
        <h3 className="group-header" id="grp-checks">
          Checks
        </h3>

        <section className="card">
          <h2 className="card-title">Inverse of {nm}</h2>
          {facts.inv ? (
            <>
              <pre className="matrix-readout">{matrixText(facts.inv)}</pre>
              <p className="muted small">
                Applying {nm} then {nm}⁻¹ returns every vector home — that is exactly why their
                product is the identity.
              </p>
              <button
                type="button"
                className="btn full"
                onClick={() => useApp.getState().setShowInverse(!showInverse)}
              >
                {showInverse ? `Back to drawing ${nm}` : `Animate ${nm} → ${nm}⁻¹`}
              </button>
            </>
          ) : (
            <Prose
              text={`det = 0, so ${nm} is singular: information was destroyed and nothing can undo it. The null space below lists exactly what got erased.`}
              className="warn-text"
            />
          )}
        </section>

        <section className="card">
          <h2 className="card-title">
            Idempotence{' '}
            <span className={idem ? 'badge' : 'badge badge-warn'}>
              {idem ? `${nm}² = ${nm} ✓` : `${nm}² ≠ ${nm}`}
            </span>
          </h2>
          <div className="kv">
            <span className="k mono">{nm}²</span>
            <span className="v mono">{matrixCompact(A2)}</span>
          </div>
          {!idem && (
            <div className="kv">
              <span className="k mono">
                {nm}² − {nm}
              </span>
              <span className="v mono">{matrixCompact(residual)}</span>
            </div>
          )}
          <details className="expander">
            <summary>
              {idem ? (
                <>
                  Applying <span className="mono">{nm}</span> again lands on the exact same matrix,
                  so the second pass leaves the picture untouched.
                </>
              ) : (
                <>
                  The entries above show how far <span className="mono">{nm}²</span> drifts from{' '}
                  <span className="mono">{nm}</span> — a second application keeps changing space.
                </>
              )}
            </summary>
            <p className="muted small">
              {idem ? (
                <>
                  Such a matrix is a <b>projection</b>: its eigenvalues are only 0 and 1,{' '}
                  <span className="mono">trace = rank</span>, and <span className="mono">det = 0</span>{' '}
                  (unless {nm} = I).
                </>
              ) : (
                <>
                  Only projections satisfy <span className="mono">{nm}² = {nm}</span> exactly — run
                  the I → {nm} → {nm}² demo on the left to watch a projection settle.
                </>
              )}
            </p>
          </details>
        </section>

        {compActive && (
          <section className="card">
            <h2 className="card-title">Order matters</h2>
            <div className="kv">
              <span className="k">{dim === 4 ? 'M·N' : 'A·B'}</span>
              <span className="v mono">{matrixCompact(AB)}</span>
            </div>
            <div className="kv">
              <span className="k">{dim === 4 ? 'N·M' : 'B·A'}</span>
              <span className="v mono">{matrixCompact(BA)}</span>
            </div>
            <p className="muted small">
              {sameMatrix(AB, BA)
                ? 'These two happen to be equal — try a different pair.'
                : 'These are different: “rotate then shear” is not the same as “shear then rotate”.'}
            </p>
          </section>
        )}

        {/* ---------------------------- decompositions ------------------------ */}
        <h3 className="group-header" id="grp-decompositions">
          Decompositions
        </h3>

        <section className="card">
          <h2 className="card-title">Eigenvalues of {nm}</h2>
          {facts.entries.length === 0 ? (
            <p className="muted small">Could not decompose this matrix.</p>
          ) : (
            <ul className="eigen-list">
              {facts.entries.map((e, i) => (
                <li key={i} className="eigen">
                  <span className="eigen-value mono">
                    λ{sub(i + 1)} = {n(e.real)}
                    {Math.abs(e.imag) > 1e-9 && (
                      <>
                        {e.imag > 0 ? ' + ' : ' − '}
                        {n(Math.abs(e.imag))}i
                      </>
                    )}
                    {e.multiplicity > 1 && <span className="mult"> ×{e.multiplicity}</span>}
                  </span>
                  {e.vector && (
                    <span className="eigen-vector mono">
                      {e.vectors
                        .map((v, j) => `${j === 0 ? 'v' : `v${sub(j + 1)}`} = ${vecText(v)}`)
                        .join('   ')}
                    </span>
                  )}
                  {Math.abs(e.imag) > 1e-9 && (
                    <span className="eigen-note">no real eigenvector exists</span>
                  )}
                  {e.fullSpace && (
                    <span className="eigen-note">every direction is an eigenvector</span>
                  )}
                  {!e.fullSpace &&
                    Math.abs(e.imag) <= 1e-9 &&
                    e.multiplicity > e.vectors.length && (
                      <span className="eigen-note">
                        repeated {e.multiplicity}× but only {e.vectors.length} independent
                        direction{e.vectors.length === 1 ? '' : 's'} — a shear-like (defective)
                        matrix
                      </span>
                    )}
                </li>
              ))}
            </ul>
          )}
          {complex.length > 0 ? (
            <p className="muted small">
              Complex eigenvalues mean there is <b>no real direction</b> that survives unchanged —
              the transformation has a rotational twist.
            </p>
          ) : (
            facts.entries.length > 0 && (
              <p className="muted small">
                Each eigenvector keeps its direction: {nm} v is just a rescaled copy of v, stretched
                by λ.
              </p>
            )
          )}
        </section>

        {/* --------------------- partitioned matrix (4×4) -------------------- */}
        {bf && (
          <section className="card">
            <h2 className="card-title">
              Partitioned matrix <span className="badge badge-strong mono">M = [A B; C D]</span>
              <span className="badge badge-amber">{structLabel}</span>
            </h2>

            <div className="block-grid" aria-label="The four blocks of M">
              {(
                [
                  ['A', 'B'],
                  ['C', 'D'],
                ] as const
              ).map((row) => (
                <div className="block-row" key={row[0]}>
                  {row.map((letter) => (
                    <div
                      className="block-box"
                      key={letter}
                      style={{ borderColor: BLOCK_COLORS[letter] }}
                    >
                      <span className="block-tag" style={{ color: BLOCK_COLORS[letter] }}>
                        {letter}
                      </span>
                      <span className="block-vals mono">
                        {bf.blocks[letter]
                          .map((r) => r.map((x) => n(x)).join('  '))
                          .join(' / ')}
                      </span>
                    </div>
                  ))}
                </div>
              ))}
            </div>

            {bf.S ? (
              <>
                <div className="kv">
                  <span className="k mono">S = D − CA⁻¹B</span>
                  <span className="v mono">{matrixCompact(bf.S)}</span>
                </div>
                <div className="kv">
                  <span className="k mono">det M = det A · det S</span>
                  <span className="v mono">
                    {n(bf.detM)} = {n(bf.detA)} · {bf.detS !== null ? n(bf.detS) : '—'}{' '}
                    {bf.detS !== null &&
                    Math.abs(bf.detM - bf.detA * bf.detS) <
                      1e-6 * Math.max(1, Math.abs(bf.detM))
                      ? '✓'
                      : '✗'}
                  </span>
                </div>
              </>
            ) : (
              <p className="warn-text small">
                Block A is singular, so nothing can be eliminated through it — the Schur complement
                doesn't exist for this matrix.
              </p>
            )}

            <div className="kv">
              <span className="k mono">tr M = tr A + tr D</span>
              <span className="v mono">
                {n(bf.traceInfo.total)} = {n(bf.traceInfo.parts)} ✓
              </span>
            </div>

            {bf.union ? (
              <>
                <div className="kv">
                  <span className="k mono">λ(M) = λ(A) ∪ λ(D)</span>
                  <span className="v mono">
                    {[...bf.union.fromA, ...bf.union.fromD]
                      .map((e) =>
                        Math.abs(e.imag) > 1e-9
                          ? `${n(e.real)}${e.imag > 0 ? '+' : '−'}${n(Math.abs(e.imag))}i`
                          : n(e.real),
                      )
                      .join('  ')}
                  </span>
                </div>
                <p className="muted small">
                  Block-triangular structure factors the characteristic polynomial: det(λI − M) =
                  det(λI − A) · det(λI − D), so every eigenvalue of either diagonal block is an
                  eigenvalue of M.
                </p>
              </>
            ) : (
              <p className="muted small">
                Coupled: B and C mix the planes, so the characteristic polynomial no longer factors
                — λ(M) has no block shortcut (the trace rule above still holds).
              </p>
            )}

            {bf.sym && defA && defS && (
              <>
                <div className="kv">
                  <span className="k">Sylvester via Schur</span>
                  <span className="v mono">
                    A: {shortDef(defA.kind)}
                    {defA.kind === 'positiveDefinite' ? ' ✓' : ''} · S: {shortDef(defS.kind)}
                    {defS.kind === 'positiveDefinite' ? ' ✓' : ''} → M:{' '}
                    {defA.kind === 'positiveDefinite' && defS.kind === 'positiveDefinite'
                      ? 'positive definite ✓'
                      : 'not positive definite'}
                  </span>
                </div>
                <details className="expander">
                  <summary>
                    S is also a conditional covariance: the blue block's variance left after
                    regressing the rose block out.
                  </summary>
                  <p className="muted small">
                    For a symmetric M read as a covariance, S = D − BᵀA⁻¹B is exactly{' '}
                    <span className="mono">var(blue | rose)</span> — the spread the blue variables
                    keep once the rose ones are known. Same block formula, one level up from the
                    1+1 version in the Principal components card.
                  </p>
                </details>
              </>
            )}

            {products && (
              <>
                <div className="subhead">Block products of M·N</div>
                {products.map((p) => (
                  <div className="kv" key={p.slot}>
                    <span className="k mono">{p.slot}</span>
                    <span className="v mono">{matrixCompact(p.sum)}</span>
                  </div>
                ))}
                <details className="expander">
                  <summary>
                    Each slot is the sum of two 2×2 products: AE + BG, AF + BH, CE + DG, CF + DH.
                  </summary>
                  {products.map((p) => (
                    <p className="muted small mono" key={p.slot}>
                      {p.terms.map((t) => `${t.label} = ${matrixCompact(t.M)}`).join('  +  ')} ={' '}
                      {matrixCompact(p.sum)}
                    </p>
                  ))}
                </details>
              </>
            )}

            <details className="expander">
              <summary>
                Why det M = det A · det S: the shear L = [I 0; CA⁻¹ I] has det 1 and clears C
                without changing the determinant.
              </summary>
              <p className="muted small">
                Block row-reduce M by L⁻¹: the lower-left block vanishes and the lower right
                becomes S, giving the block LU. det L = 1 · det I = 1 leaves the determinant
                untouched, and the triangular determinant rule splits det U = det A · det S.
              </p>
            </details>
            <details className="expander">
              <summary>Transpose is blockwise: Mᵀ = [Aᵀ Cᵀ; Bᵀ Dᵀ].</summary>
              <p className="muted small mono">
                {blockTranspose(A)
                  .named.map((t) => `${t.label} = ${matrixCompact(t.M)}`)
                  .join('   ')}
              </p>
            </details>
          </section>
        )}

        <section className="card">
          <h2 className="card-title">
            Spectral decomposition{' '}
            <span
              className={
                spectral.kind === 'orthogonal' || spectral.kind === 'general'
                  ? 'badge badge-strong'
                  : 'badge badge-amber'
              }
            >
              {spectral.kind === 'orthogonal'
                ? 'A = QΛQᵀ'
                : spectral.kind === 'general'
                  ? 'A = P D P⁻¹'
                  : spectral.kind === 'defective'
                    ? 'not diagonalizable'
                    : 'no real form'}
            </span>
          </h2>
          {(spectral.kind === 'orthogonal' || spectral.kind === 'general') &&
          spectral.P &&
          spectral.D ? (
            <>
              <div className="kv">
                <span className="k mono">{spectral.orthogonal ? 'Λ' : 'D'}</span>
                <span className="v mono">{matrixCompact(spectral.D)}</span>
              </div>
              <div className="kv">
                <span className="k mono">{spectral.orthogonal ? 'Q' : 'P'}</span>
                <span className="v mono">{matrixCompact(spectral.P)}</span>
              </div>
              <div className="kv">
                <span className="k">residual</span>
                <span className="v mono">{n(spectral.residual)}</span>
              </div>
              <Prose
                text={
                  spectral.orthogonal
                    ? dim === 4
                      ? 'M is symmetric, so its eigenvectors are orthogonal and Q⁻¹ = Qᵀ: M = QΛQᵀ changes to the eigenbasis, scales along each axis, and changes back.'
                      : 'A is symmetric, so its eigenvectors are orthogonal and Q⁻¹ = Qᵀ: A = QΛQᵀ changes to the eigenbasis, scales along each axis, and changes back. The demo on the left undoes Q, applies Λ, then rotates into place.'
                    : dim === 4
                      ? 'M = P D P⁻¹ keeps the eigenvalues on the diagonal of D and the eigenvectors in the columns of P. Because those columns span the whole space, every vector splits into eigen-directions and M just rescales each piece.'
                      : 'A = P D P⁻¹ keeps the eigenvalues on the diagonal of D and the eigenvectors in the columns of P. Because those columns span the whole space, every vector splits into eigen-directions and A just rescales each piece.'
                }
              />
            </>
          ) : (
            <>
              <div className="kv">
                <span className="k">λ</span>
                <span className="v mono">
                  {facts.entries.length
                    ? facts.entries
                        .map(
                          (e) =>
                            `${n(e.real)}${
                              Math.abs(e.imag) > 1e-9
                                ? `${e.imag > 0 ? ' + ' : ' − '}${n(Math.abs(e.imag))}i`
                                : ''
                            }`,
                        )
                        .join('  ')
                    : '—'}
                </span>
              </div>
              <Prose
                className="warn-text"
                text={
                  spectral.kind === 'defective'
                    ? 'A is defective: the eigenvectors do not span Rⁿ, because a repeated eigenvalue is missing an independent direction. No change of basis can diagonalize it — the shear preset is the classic 2×2 example.'
                    : 'A has complex eigenvalues, so no real line maps onto itself — the action is a rotation mixed with a stretch, describable only with the complex pair a ± b i. The demo stays off because there is no real eigenbasis to draw.'
                }
              />
            </>
          )}
        </section>

        {dim !== 4 && (
          <section className="card">
            <h2 className="card-title">
              Ellipse of A
              <span className="badge">
                {dim === 2 ? 'circle → ellipse' : 'ball → ellipsoid'}
              </span>
            </h2>
          <div className="kv">
            <span className="k">σ</span>
            <span className="v mono">
              {facts.sv.length
                ? facts.sv.map((s, i) => `σ${sub(i + 1)} = ${n(s)}`).join('   ')
                : '—'}
            </span>
          </div>
          <div className="kv">
            <span className="k">{dim === 2 ? 'area' : 'volume'}</span>
            <span className="v mono">
              {dim === 2
                ? `π σ₁σ₂ = ${n(Math.PI * (facts.sv[0] ?? 0) * (facts.sv[1] ?? 0))}`
                : `(4/3)π σ₁σ₂σ₃ = ${n((4 / 3) * Math.PI * (facts.sv[0] ?? 0) * (facts.sv[1] ?? 0) * (facts.sv[2] ?? 0))}`}
              {' = '}
              {dim === 2 ? 'π' : '(4/3)π'} |det A|
            </span>
          </div>
          <p className="muted small">
            {facts.sym ? (
              <>
                A is <b>symmetric</b>, so the ellipse axes are exactly the eigenvector directions and{' '}
                <span className="mono">σᵢ = |λᵢ|</span> — eigenvalues measure the stretch along each
                axis directly.
              </>
            ) : complex.length > 0 && dim === 2 ? (
              <>
                The complex pair <span className="mono">λ = a ± b i</span> names no real direction, but
                the ellipse is perfectly real: its area is{' '}
                <span className="mono">π|det A| = π(a² + b²)</span>, the squared distance of λ from
                the origin.
              </>
            ) : (
              <>
                A is not symmetric, so its eigenvectors are <b>not orthogonal</b> and miss the
                ellipse's true axes: <span className="mono">σ₁ ≥ max|λ|</span>
                {maxAbsLambda !== null && (
                  <>
                    {' '}
                    here <span className="mono">σ₁ = {n(facts.sv[0] ?? 0)} ≥ {n(maxAbsLambda)}</span>
                  </>
                )}
                . A shear is the extreme case: λ = 1, yet σ = (1±√5)/2 ≈ 1.618 / 0.618 — the golden
                ratio, with the ellipse rotated 45° away from the eigenvector.
              </>
            )}
          </p>
          {facts.symZero ? (
            <p className="muted small">
              The symmetric part <span className="mono">S = (A+Aᵀ)/2</span> is zero, so{' '}
              <span className="mono">xᵀAx = 0</span> for every vector — there are no level sets to
              draw (try the Quarter turn).
            </p>
          ) : (
            <details className="expander">
              <summary>
                Level sets <span className="mono">xᵀAx = c</span> depend only on{' '}
                <span className="mono">S = (A+Aᵀ)/2</span>.
              </summary>
              <p className="muted small">
                Ellipses when c agrees with the signs of S's eigenvalues, hyperbolas when it
                doesn't, with semi-axes <span className="mono">√(c/λᵢ)</span> along S's
                eigenvectors.
              </p>
            </details>
            )}
          </section>
        )}

        <section className="card">
          <h2 className="card-title">
            Singular value decomposition{' '}
            <span className="badge badge-strong">{nm} = UΣVᵀ</span>
            {svd && svd.s[svd.s.length - 1] < 1e-9 && (
              <span className="badge badge-warn">singular: σₙ = 0</span>
            )}
          </h2>
          {svd ? (
            <>
              <div className="kv">
                <span className="k">σ</span>
                <span className="v mono">
                  {svd.s.map((s, i) => `σ${sub(i + 1)} = ${n(s)}`).join('   ')}
                </span>
              </div>
              <div className="kv">
                <span className="k mono">U</span>
                <span className="v mono">{matrixCompact(svd.U)}</span>
              </div>
              <div className="kv">
                <span className="k mono">Vᵀ</span>
                <span className="v mono">{matrixCompact(svd.Vt)}</span>
              </div>
              <div className="kv">
                <span className="k">residual</span>
                <span className="v mono">{n(svd.residual)}</span>
              </div>
              <Prose
                text={`Every matrix equals a turn (Vᵀ), a stretch along the axes by σ, and a second turn (U) — ${
                  dim === 4
                    ? 'the σᵢ measure exactly how many directions M stretches each input direction'
                    : 'the σᵢ are exactly the semi-axes the ellipse layer draws'
                }. When ${nm} is singular the last σ is 0: ${
                  dim === 4 ? 'a direction is flattened to nothing' : 'the circle is flattened to nothing along vₙ'
                }, which is why no inverse exists, yet UΣVᵀ still rebuilds ${nm} exactly.`}
              />
            </>
          ) : (
            <p className="warn-text small">Could not decompose this matrix.</p>
          )}
        </section>

        <section className="card">
          <h2 className="card-title">
            Principal components{' '}
            <span className={dataset ? 'badge badge-strong' : 'badge'}>
              {dataset ? 'A = S(x)' : 'PCA'}
            </span>
            {pca && <span className="badge">n = {pca.n}</span>}
          </h2>
          {pca && dataset ? (
            <>
              <div className="kv">
                <span className="k">x̄</span>
                <span className="v mono">{vecText(pca.mean)}</span>
              </div>
              <ul className="eigen-list">
                {pca.lambdas.map((lam, i) => (
                  <li key={i} className="eigen">
                    <span className="eigen-value mono">
                      PC{sub(i + 1)} · λ = {n(lam)}
                    </span>
                    <span className="eigen-vector mono">{vecText(pca.vectors[i])}</span>
                    <span className="eigen-note">
                      √λ = {n(pca.std[i])} · {n(pca.explained[i] * 100, 1)}% of the variance
                    </span>
                  </li>
                ))}
              </ul>
              <div
                className="var-bar"
                role="img"
                aria-label={`explained variance: ${pca.explained
                  .map((e, i) => `PC${sub(i + 1)} ${n(e * 100, 1)}%`)
                  .join(', ')}`}
              >
                {pca.explained.map((e, i) => (
                  <span
                    key={i}
                    className="var-seg"
                    style={{ width: `${e * 100}%`, background: PC_COLORS[i] }}
                  />
                ))}
              </div>
              <p className="muted small">
                {pca.explained
                  .map((e, i) => `PC${sub(i + 1)} ${n(e * 100, 1)}%`)
                  .join(' · ')}
              </p>
              <div className="kv">
                <span className="k">trace</span>
                <span className="v mono">Σλ = {n(pca.trace)} = total variance</span>
              </div>
              <div className="kv">
                <span className="k">error (k = 1)</span>
                <span className="v mono">
                  {n(pca.recon1)} = {dim === 2 ? 'λ₂' : 'λ₂ + λ₃'}
                </span>
              </div>
              {condVar !== null && (
                <>
                  <div className="kv">
                    <span className="k">var(y | x)</span>
                    <span className="v mono">
                      {n(condVar)} = s_yy − s_xy² / s_xx
                    </span>
                  </div>
                  <details className="expander">
                    <summary>
                      var(y | x) is a Schur complement — the 1+1 case of the Partitioned matrix
                      card.
                    </summary>
                    <p className="muted small">
                      Split S into the x column and the rest, eliminate x, and what remains in the
                      corner is the variance y keeps once x is known — one regression, one 2×2
                      Schur step. The same elimination on whole 2×2 blocks is what the Partitioned
                      matrix card computes in the 4×4 view.
                    </p>
                  </details>
                </>
              )}
              {presetName === 'Two clusters' && (
                <p className="warn-text small">
                  PCA sees one covariance ellipse, not two clusters: it only reads second
                  moments, so the empty gap between the blobs is invisible to it.
                </p>
              )}
              {presetName === 'Different units (y ×5)' && (
                <p className="warn-text small">
                  y is measured in units ~5× coarser than x, so PC1 is simply the y-axis.
                  Divide each axis by its standard deviation (the correlation matrix)
                  before comparing variances.
                </p>
              )}
              {presetName === 'Isotropic cloud' && (
                <p className="muted small">
                  λ₁ ≈ λ₂: no direction stands out — the split is ≈ 50/50 and the PC axes
                  are essentially arbitrary.
                </p>
              )}
              <details className="expander">
                <summary>
                  Center first, then diagonalize the covariance S = Cov(x).
                </summary>
                <p className="muted small">
                  PCA is only defined for centered data — otherwise the "direction of
                  maximal variance" just points at the mean. S = (1/(n−1))·XcᵀXc is
                  symmetric, so the spectral theorem hands it an orthonormal eigenbasis:
                  the principal components. λᵢ is the variance along PCᵢ, √λᵢ its
                  standard deviation (the 1σ radius the rose ellipse draws), and the
                  Definiteness card proves S is positive semidefinite for free.
                </p>
              </details>
              <details className="expander">
                <summary>
                  SVD link: σᵢ(Xc) = √((n−1)·λᵢ) — the PCs are the V of the centered
                  data's SVD.
                </summary>
                <p className="muted small">
                  Run the SVD chip while a cloud is loaded: its right singular vectors
                  are exactly these principal directions, with singular values rescaled
                  by n−1. The sign of a PC is arbitrary — v and −v are both valid, so
                  the arrows may flip between runs.
                </p>
              </details>
              <details className="expander">
                <summary>Choosing k: keep components until the cumulative share is enough.</summary>
                <p className="muted small">
                  Reconstructing every point from its first k coordinates keeps
                  Σᵢ₌₁..k λᵢ of the total variance and discards the rest — the mean
                  squared reconstruction error is exactly λ₍ₖ₊₁₎ + … + λ_d, which is
                  the error (k = 1) number above.
                </p>
              </details>
            </>
          ) : (
            <Prose
              text={
                dim === 4
                  ? 'PCA is a story about data — switch to the 2×2 or 3×3 view and load a preset under PCA datasets on the left; the covariance S then becomes Matrix A.'
                  : 'PCA finds the directions where a point cloud spreads most — load a preset under PCA datasets on the left, and Matrix A becomes the covariance S, so every card above describes the data.'
              }
            />
          )}
        </section>

        <section className="card">
          <h2 className="card-title">
            Definiteness{' '}
            <span className={defBadge[def.kind].cls}>{defBadge[def.kind].text}</span>
          </h2>
          <div className="kv">
            <span className="k">{dim === 4 ? 'λ((M+Mᵀ)/2)' : 'λ(S)'}</span>
            <span className="v mono">
              {def.lambdas.length ? def.lambdas.map((l) => n(l)).join('  ') : '—'}
            </span>
          </div>
          <div className="kv">
            <span className="k">Sylvester</span>
            <span className="v mono">
              {def.minors.map((m, i) => {
                const ok =
                  def.kind === 'positiveDefinite'
                    ? m > 0
                    : def.kind === 'negativeDefinite'
                      ? (i % 2 === 0 ? m < 0 : m > 0)
                      : null;
                return `${i > 0 ? '  ' : ''}Δ${sub(i + 1)} = ${n(m)}${ok === null ? '' : ok ? ' ✓' : ' ✗'}`;
              }).join('')}
            </span>
          </div>
          <p className="muted small">{defVerb[def.kind]}</p>
          {(complex.length > 0 || def.viaSymmetricPart) && (
            <p className="muted small">
              {complex.length > 0 ? (
                <>
                  A's eigenvalues are complex <span className="mono">(λ = a ± b i)</span> — no real
                  direction survives — but the verdict still comes from the real symmetric part{' '}
                  <span className="mono">S = (A+Aᵀ)/2</span> alone.
                </>
              ) : (
                <>
                  A isn't symmetric, but only <span className="mono">S = (A+Aᵀ)/2</span> decides the
                  sign of <span className="mono">xᵀAx</span> — A's skew part cancels out of the
                  quadratic form.
                </>
              )}
            </p>
          )}
          <details className="expander">
            <summary>See it in the layers.</summary>
            <p className="muted small">
              {def.kind === 'positiveDefinite' || def.kind === 'negativeDefinite' ? (
                <>
                  The level sets of one sign are nested ellipses while the opposite sign draws
                  nothing; the sign field is a single color everywhere.
                </>
              ) : def.kind === 'zero' ? (
                <>
                  With <span className="mono">S = 0</span> neither layer has anything to draw —
                  switch to another preset to see them.
                </>
              ) : def.kind === 'indefinite' ? (
                <>
                  Both signs of c draw curves — hyperbolas closing around the saddle — and the sign
                  field splits into warm and cool halves along the null cone.
                </>
              ) : (
                <>
                  One sign draws nested ellipses, the opposite sign draws nothing, and in the zero
                  direction the level sets flatten to parallel lines.
                </>
              )}
            </p>
          </details>
        </section>

        {/* ------------------------------ subspaces --------------------------- */}
        <h3 className="group-header" id="grp-subspaces">
          Subspaces
        </h3>

        <section className="card">
          <h2 className="card-title">Subspaces of {nm}</h2>
          <div className="kv">
            <span className="k">dim Col(A)</span>
            <span className="v mono">{facts.cols.length}</span>
          </div>
          <div className="kv">
            <span className="k">basis</span>
            <span className="v mono">{facts.cols.map(vecText).join(' ') || '—'}</span>
          </div>
          <div className="kv">
            <span className="k">dim Null(A)</span>
            <span className="v mono">{facts.nulls.length}</span>
          </div>
          <div className="kv">
            <span className="k">basis</span>
            <span className="v mono">{facts.nulls.length ? facts.nulls.map(vecText).join(' ') : '{0}'}</span>
          </div>
          <p className="muted small">
            Rank–nullity: {facts.r} + {facts.nulls.length} = {dim}. Reachable outputs and erased
            directions always add up to the whole space.
          </p>
        </section>
      </div>
    </>
  );
}

function Stat({ label, value, tone }: { label: string; value: string; tone?: 'pos' | 'neg' }) {
  return (
    <div className={`stat ${tone ? `stat-${tone}` : ''}`}>
      <span className="stat-label">{label}</span>
      <span className="stat-value mono">{value}</span>
    </div>
  );
}
