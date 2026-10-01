/**
 * Small linear-algebra core for square 2x2 / 3x3 matrices.
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

export function transpose(A: Matrix): Matrix {
  const n = A.length;
  return Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => A[j][i]));
}

/** Determinant (closed form for n = 2 and n = 3). */
export function det(A: Matrix): number {
  if (A.length === 2) {
    return A[0][0] * A[1][1] - A[0][1] * A[1][0];
  }
  return (
    A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
    A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
    A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0])
  );
}

/**
 * Inverse via the adjugate formula. Returns `null` for singular matrices
 * (|det| below tolerance) so callers can render an educational message.
 */
export function inverse(A: Matrix): Matrix | null {
  const n = A.length;
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
