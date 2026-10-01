import { useMemo } from 'react';
import { useApp } from '../state/store';
import { useActiveMatrix } from '../state/hooks';
import {
  Matrix,
  Vector,
  clean,
  columnSpaceBasis,
  det,
  inverse,
  matMul,
  nullSpaceBasis,
  rank,
} from '../math/matrix';
import { eigen } from '../math/eigen';
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

const sub = (n: number) => ['', '₁', '₂', '₃', '₄', '₅'][n] ?? String(n);

function detCaption(d: number, size: number) {
  const unit = size === 2 ? 'area' : 'volume';
  if (Math.abs(d) < 1e-9) {
    return `Singular: the ${unit} collapses to 0 — the shape is squashed onto a lower dimension.`;
  }
  const flip = d < 0 ? ' Orientation is flipped, so a mirror happened too.' : '';
  return `The unit ${size === 2 ? 'square' : 'cube'} becomes a shape with ${unit} × ${n(
    Math.abs(d),
  )}.${flip}`;
}

/** Human description of A, taken from the preset catalogue when it matches. */
function describe(A: Matrix, dim: 2 | 3): string {
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
  const presetName = useApp((s) => s.presetName);
  const active = useActiveMatrix();

  const facts = useMemo(() => {
    const d = det(A);
    const r = rank(A);
    const inv = inverse(A);
    const entries = eigen(A);
    const cols = columnSpaceBasis(A);
    const nulls = nullSpaceBasis(A);
    return { d, r, inv, entries, cols, nulls };
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

  const complex = facts.entries.filter((e) => Math.abs(e.imag) > 1e-9);

  return (
    <div className="panel-stack">
      <section className="card">
        <h2 className="card-title">
          Currently drawn <span className="badge badge-strong">{active.label}</span>
          {active.singular && <span className="badge badge-warn">no inverse</span>}
        </h2>
        <pre className="matrix-readout">{matrixText(active.matrix)}</pre>
        <div className="stat-grid">
          <Stat label="det" value={n(shown.d)} tone={shown.d < 0 ? 'neg' : 'pos'} />
          <Stat label="rank" value={`${shown.r} / ${dim}`} />
        </div>
        <p className="muted small">{detCaption(shown.d, dim)}</p>
      </section>

      <section className="card">
        <h2 className="card-title">{presetName ?? 'Custom matrix'}</h2>
        <p className="muted small">
          {compActive
            ? 'B is applied first, then A. Step through I → B → A·B to watch the second transformation build on the result of the first.'
            : describe(A, dim)}
        </p>
      </section>

      <section className="card">
        <h2 className="card-title">A applied to x</h2>
        <div className="kv">
          <span className="k">x</span>
          <span className="v mono">{vecText(vector)}</span>
        </div>
        <div className="kv">
          <span className="k accent-a">A x</span>
          <span className="v mono">{vecText(Ax)}</span>
        </div>
        <p className="muted small">
          Each column of A records where a basis vector lands, so A x is a weighted blend of the
          columns: <span className="mono">x₁·col₁ + x₂·col₂{dim === 3 ? ' + x₃·col₃' : ''}</span>.
        </p>
      </section>

      <section className="card">
        <h2 className="card-title">Inverse of A</h2>
        {facts.inv ? (
          <>
            <pre className="matrix-readout">{matrixText(facts.inv)}</pre>
            <p className="muted small">
              Applying A then A⁻¹ returns every vector home — that is exactly why their product is
              the identity.
            </p>
            <button
              type="button"
              className="btn full"
              onClick={() => useApp.getState().setShowInverse(!showInverse)}
            >
              {showInverse ? 'Back to drawing A' : 'Animate A → A⁻¹'}
            </button>
          </>
        ) : (
          <p className="muted small warn-text">
            det = 0, so A is <b>singular</b>: information was destroyed and nothing can undo it. The
            null space below lists exactly what got erased.
          </p>
        )}
      </section>

      <section className="card">
        <h2 className="card-title">Eigenvalues of A</h2>
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
              Each eigenvector keeps its direction: A v is just a rescaled copy of v, stretched by λ.
            </p>
          )
        )}
      </section>

      <section className="card">
        <h2 className="card-title">Subspaces of A</h2>
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

      {compActive && (
        <section className="card">
          <h2 className="card-title">Order matters</h2>
          <div className="kv">
            <span className="k">A·B</span>
            <span className="v mono">{matrixCompact(AB)}</span>
          </div>
          <div className="kv">
            <span className="k">B·A</span>
            <span className="v mono">{matrixCompact(BA)}</span>
          </div>
          <p className="muted small">
            {sameMatrix(AB, BA)
              ? 'These two happen to be equal — try a different pair.'
              : 'These are different: “rotate then shear” is not the same as “shear then rotate”.'}
          </p>
        </section>
      )}
    </div>
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
