/**
 * Small linear-algebra core for square matrices.
 *
 * Closed-form det/inverse for 2×2 / 3×3 (they show up verbatim on the
 * cards), LU-based paths for larger n (the 4×4 block view needs them).
 *
 * Everything is stored row-major as `number[][]` so it can be dropped
 * straight into React state and rendered without conversions.
 */

export type Matrix = number[][];
export type Vector = number[];

/** Numerical tolerance used to decide whether something is "zero". */
export const EPS = 1e-9;

export const dim = (A: Matrix): number => A.length;

export function zeros(n: number): Matrix {
  return Array.from({ length: n }, () => Array.from({ length: n }, () => 0));
}

export function identity(n: number): Matrix {
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)),
  );
}

export function clone(A: Matrix): Matrix {
  return A.map((row) => row.slice());
}

export function isApproxEqual(A: Matrix, B: Matrix, tol = 1e-7): boolean {
  if (A.length !== B.length) return false;
  for (let i = 0; i < A.length; i++) {
    for (let j = 0; j < A.length; j++) {
      if (Math.abs(A[i][j] - B[i][j]) > tol) return false;
    }
  }
  return true;
}

/* ------------------------------------------------------------------ */
/* Basic operations                                                    */
/* ------------------------------------------------------------------ */

export function matMul(A: Matrix, B: Matrix): Matrix {
  const n = A.length;
  const out = zeros(n);
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < n; k++) {
      const aik = A[i][k];
      if (aik === 0) continue;
      for (let j = 0; j < n; j++) out[i][j] += aik * B[k][j];
    }
  }
  return out;
}

export function matVec(A: Matrix, v: Vector): Vector {
  return A.map((row) => row.reduce((s, a, k) => s + a * (v[k] ?? 0), 0));
}

/**
 * Idempotent means A·A = A: applying the transformation a second time
 * changes nothing. Geometrically A is a projection onto Col(A) along
 * Null(A), so its only possible eigenvalues are 0 and 1.
 */
export function isIdempotent(A: Matrix, tol = 1e-7): boolean {
  return isApproxEqual(matMul(A, A), A, tol);
}

/** True when A = Aᵀ (within `tol`). Symmetric matrices have real eigenvalues. */
export function isSymmetric(A: Matrix, tol = 1e-9): boolean {
  const n = A.length;
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++) if (Math.abs(A[i][j] - A[j][i]) > tol) return false;
  return true;
}

/**
 * Symmetric part (A + Aᵀ)/2.
 *
 * xᵀAx = xᵀ((A+Aᵀ)/2)x for every x — the level sets of the quadratic form
 * are governed entirely by this matrix, which is why its eigenvectors are
 * the natural axes of those contours.
 */
export function symmetricPart(A: Matrix): Matrix {
  const n = A.length;
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (A[i][j] + A[j][i]) / 2),
  );
}

export function transpose(A: Matrix): Matrix {
  const n = A.length;
  return Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => A[j][i]));
}

/* ------------------------------------------------------------------ */
/* Quadratic form: xᵀAx and its level sets                             */
/* ------------------------------------------------------------------ */

/** Evaluate the quadratic form xᵀAx (symmetric part only — see symmetricPart). */
export function quadraticForm(A: Matrix, x: Vector): number {
  const Ax = matVec(A, x);
  return x.reduce((s, xi, i) => s + xi * Ax[i], 0);
}

/**
 * Level sets of the quadratic form in *eigen-coordinates*:
 *
 *     { y : λ₁y₁² + λ₂y₂² = c }
 *
 * The caller rotates into the eigenbasis of the symmetric part first (see
 * `symmetricBasis`), so every contour here is axis-aligned. Returns a list
 * of polylines (each a `Vector[]` of [y₁, y₂] points), one per branch:
 *
 *  - **ellipse**        λ₁, λ₂ have the same sign as c → one closed loop;
 *  - **hyperbola**      signs differ → two branches;
 *  - **parallel lines** one λ ≈ 0 and c/λ > 0 → two straight lines;
 *  - **empty**          no real points (e.g. definite form, c of the wrong
 *                       sign) → `[]`.
 *
 * `samples` controls the resolution of curved branches; `extent` bounds the
 * open branches (hyperbolae / lines) in world units.
 */
export function quadraticContours(
  l1: number,
  l2: number,
  c: number,
  samples = 96,
  extent = 14,
): Vector[][] {
  const z1 = Math.abs(l1) < 1e-12;
  const z2 = Math.abs(l2) < 1e-12;
  if (Math.abs(c) < 1e-12) return []; // c = 0 is the degenerate cone point
  if (z1 && z2) return []; // S = 0: xᵀAx ≡ 0, no level set for c ≠ 0

  // One zero eigenvalue: the form reduces to λⱼyⱼ² = c → parallel lines
  // (or nothing, if the sign doesn't work out).
  if (z1 || z2) {
    const lj = z1 ? l2 : l1;
    const r2 = c / lj;
    if (r2 <= 0) return [];
    const r = Math.sqrt(r2);
    // Line y₂ = ±r runs along y₁ (when λ₂ = 0), or y₁ = ±r along y₂.
    return z1
      ? [
          [[-extent, r], [extent, r]],
          [[-extent, -r], [extent, -r]],
        ]
      : [
          [[r, -extent], [r, extent]],
          [[-r, -extent], [-r, extent]],
        ];
  }

  // Normalized: (y₁/a)² + (y₂/b)² = 1 with a² = c/λ₁, b² = c/λ₂.
  const a2 = c / l1;
  const b2 = c / l2;

  if (a2 > 0 && b2 > 0) {
    const a = Math.sqrt(a2);
    const b = Math.sqrt(b2);
    const pts: Vector[] = [];
    for (let k = 0; k <= samples; k++) {
      const t = (2 * Math.PI * k) / samples;
      pts.push([a * Math.cos(t), b * Math.sin(t)]);
    }
    pts[samples] = pts[0].slice(); // exact closure
    return [pts];
  }

  if (a2 > 0 && b2 < 0) {
    // Hyperbola opening along y₁: y = (±a·cosh t, √(-b²)·sinh t).
    const a = Math.sqrt(a2);
    const b = Math.sqrt(-b2);
    const tMax = Math.acosh(Math.max(2, extent / a));
    const half = (sgn: number): Vector[] => {
      const pts: Vector[] = [];
      for (let k = 0; k <= samples; k++) {
        const t = -tMax + (2 * tMax * k) / samples;
        pts.push([sgn * a * Math.cosh(t), b * Math.sinh(t)]);
      }
      return pts;
    };
    return [half(1), half(-1)];
  }

  if (a2 < 0 && b2 > 0) {
    // Hyperbola opening along y₂: swap the roles of the coordinates.
    const a = Math.sqrt(-a2);
    const b = Math.sqrt(b2);
    const tMax = Math.acosh(Math.max(2, extent / b));
    const half = (sgn: number): Vector[] => {
      const pts: Vector[] = [];
      for (let k = 0; k <= samples; k++) {
        const t = -tMax + (2 * tMax * k) / samples;
        pts.push([a * Math.sinh(t), sgn * b * Math.cosh(t)]);
      }
      return pts;
    };
    return [half(1), half(-1)];
  }

  return []; // both squared semi-axes negative → no real points
}

/**
 * Determinant. Closed form for n ≤ 3 (exact and easy to explain on the
 * card), LU with partial pivoting for anything larger (the 4×4 block view).
 */
export function det(A: Matrix): number {
  if (A.length === 1) return A[0][0];
  if (A.length === 2) {
    return A[0][0] * A[1][1] - A[0][1] * A[1][0];
  }
  if (A.length === 3) {
    return (
      A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
      A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
      A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0])
    );
  }
  return detLU(A);
}

/** det from an in-place LU elimination (Gaussian, partial pivoting). */
function detLU(A: Matrix): number {
  const n = A.length;
  const M = A.map((row) => row.slice());
  let d = 1;
  for (let k = 0; k < n; k++) {
    let p = k;
    for (let i = k + 1; i < n; i++) {
      if (Math.abs(M[i][k]) > Math.abs(M[p][k])) p = i;
    }
    if (Math.abs(M[p][k]) < 1e-300) return 0; // singular column
    if (p !== k) {
      const tmp = M[p];
      M[p] = M[k];
      M[k] = tmp;
      d = -d;
    }
    d *= M[k][k];
    for (let i = k + 1; i < n; i++) {
      const f = M[i][k] / M[k][k];
      for (let j = k; j < n; j++) M[i][j] -= f * M[k][j];
    }
  }
  return d;
}

/**
 * Inverse via the adjugate formula (n = 2, 3) or LU solve (n ≥ 4).
 * Returns `null` for singular matrices (|det| below tolerance) so callers
 * can render an educational message.
 */
export function inverse(A: Matrix): Matrix | null {
  const n = A.length;
  if (n === 2 || n === 3) {
    const d = det(A);
    if (!Number.isFinite(d) || Math.abs(d) < 1e-12) return null;

    if (n === 2) {
      return [
        [A[1][1] / d, -A[0][1] / d],
        [-A[1][0] / d, A[0][0] / d],
      ];
    }

    // 3x3: cofactor matrix, then transpose to obtain the adjugate.
    const cof = zeros(3);
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        // Minor of A at (i, j): the 2x2 block left after deleting row i, col j.
        const r = [0, 1, 2].filter((k) => k !== i);
        const c = [0, 1, 2].filter((k) => k !== j);
        const minor =
          A[r[0]][c[0]] * A[r[1]][c[1]] - A[r[0]][c[1]] * A[r[1]][c[0]];
        cof[i][j] = (i + j) % 2 === 0 ? minor : -minor;
      }
    }
    // adj(A) = cofactor(A)^T, then scale by 1/det.
    return transpose(cof).map((row) => row.map((v) => v / d));
  }

  const d = det(A);
  if (!Number.isFinite(d) || Math.abs(d) < 1e-12) return null;
  return inverseLU(A);
}

/** A⁻¹ by solving A X = I column by column through one LU factorization. */
function inverseLU(A: Matrix): Matrix | null {
  const n = A.length;
  const M = A.map((row) => row.slice());
  const perm = Array.from({ length: n }, (_, i) => i);

  for (let k = 0; k < n; k++) {
    let p = k;
    for (let i = k + 1; i < n; i++) {
      if (Math.abs(M[i][k]) > Math.abs(M[p][k])) p = i;
    }
    if (Math.abs(M[p][k]) < 1e-300) return null;
    if (p !== k) {
      const tmp = M[p];
      M[p] = M[k];
      M[k] = tmp;
      const t = perm[p];
      perm[p] = perm[k];
      perm[k] = t;
    }
    for (let i = k + 1; i < n; i++) {
      M[i][k] /= M[k][k];
      for (let j = k + 1; j < n; j++) M[i][j] -= M[i][k] * M[k][j];
    }
  }

  const solve = (b: Vector): Vector => {
    const y = perm.map((row) => b[row]); // apply the row permutation
    for (let i = 1; i < n; i++) {
      for (let j = 0; j < i; j++) y[i] -= M[i][j] * y[j];
    }
    const x = Array.from({ length: n }, () => 0);
    for (let i = n - 1; i >= 0; i--) {
      let s = y[i];
      for (let j = i + 1; j < n; j++) s -= M[i][j] * x[j];
      x[i] = s / M[i][i];
    }
    return x;
  };

  const X = zeros(n);
  for (let j = 0; j < n; j++) {
    const e = Array.from({ length: n }, (_, i) => (i === j ? 1 : 0));
    const col = solve(e);
    for (let i = 0; i < n; i++) X[i][j] = col[i];
  }
  return X;
}

/** Sum of the diagonal entries — tr M = tr A + tr D for any block split. */
export function trace(A: Matrix): number {
  let t = 0;
  for (let i = 0; i < A.length; i++) t += A[i][i];
  return t;
}

/* ------------------------------------------------------------------ */
/* Row echelon form, rank, column space, null space                    */
/* ------------------------------------------------------------------ */

export interface RrefResult {
  r: Matrix;
  pivots: number[];
}

/** Reduced row echelon form with partial pivoting. */
export function rref(input: Matrix, tol = 1e-9): RrefResult {
  const m = input.map((row) => row.slice());
  const rows = m.length;
  const cols = m[0]?.length ?? 0;
  const pivots: number[] = [];
  let r = 0;

  for (let c = 0; c < cols && r < rows; c++) {
    let best = r;
    for (let i = r; i < rows; i++) {
      if (Math.abs(m[i][c]) > Math.abs(m[best][c])) best = i;
    }
    if (Math.abs(m[best][c]) < tol) continue;

    const tmp = m[r];
    m[r] = m[best];
    m[best] = tmp;

    const pv = m[r][c];
    for (let j = 0; j < cols; j++) m[r][j] /= pv;

    for (let i = 0; i < rows; i++) {
      if (i === r) continue;
      const f = m[i][c];
      if (Math.abs(f) < tol) continue;
      for (let j = 0; j < cols; j++) m[i][j] -= f * m[r][j];
    }

    pivots.push(c);
    r++;
  }

  // Clean up floating point noise so readouts look nice.
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      if (Math.abs(m[i][j]) < 1e-12) m[i][j] = 0;
    }
  }
  return { r: m, pivots };
}

export function rank(A: Matrix): number {
  return rref(A).pivots.length;
}

/** Basis of Col(A): the pivot columns of the original matrix. */
export function columnSpaceBasis(A: Matrix): Vector[] {
  const { pivots } = rref(A);
  return pivots.map((c) => A.map((row) => row[c]));
}

/** Basis of Null(A): one vector per free variable. */
export function nullSpaceBasis(A: Matrix): Vector[] {
  const { r, pivots } = rref(A);
  const n = A.length;
  const free = Array.from({ length: n }, (_, c) => c).filter((c) => !pivots.includes(c));
  const basis: Vector[] = [];

  for (const f of free) {
    const v = Array.from({ length: n }, () => 0);
    v[f] = 1;
    pivots.forEach((p, idx) => {
      v[p] = -r[idx][f];
    });
    basis.push(v);
  }
  return basis;
}

export function norm(v: Vector): number {
  return Math.sqrt(v.reduce((s, x) => s + x * x, 0));
}

export function normalize(v: Vector): Vector {
  const n = norm(v);
  if (n < 1e-12) return v.slice();
  return v.map((x) => x / n);
}

/** Remove near-zero entries so tiny numerical dust doesn't show in the UI. */
export function clean(v: Vector, tol = 1e-10): Vector {
  return v.map((x) => (Math.abs(x) < tol ? 0 : x));
}

/** Snap values that are within `tol` of a "nice" number to that number. */
export function tidy(x: number, tol = 1e-9): number {
  return Math.abs(x) < tol ? 0 : x;
}
