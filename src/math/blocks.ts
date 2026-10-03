/**
 * Partitioned (block) matrices — a 4×4 map split into 2×2 blocks acting on
 * two planes R²⊕R²:
 *
 *     M = [ A  B ]      y₁ = A x₁ + B x₂
 *         [ C  D ]      y₂ = C x₁ + D x₂
 *
 * The block view draws exactly that: two input planes on the left, two
 * output planes on the right, and every block's contribution colored
 * separately. The algebra here backs the Partitioned matrix card —
 * Schur complements, the det factorization, block structure detection,
 * the eigenvalue union of block-triangular matrices and the conditional
 * covariance that ties the Schur complement to Feature 6.
 */

import { EigenEntry, eigen } from './eigen';
import { Matrix, det, inverse, matMul, trace, transpose } from './matrix';

/** The four 2×2 blocks of M = [[A, B], [C, D]]. */
export interface Blocks {
  A: Matrix;
  B: Matrix;
  C: Matrix;
  D: Matrix;
}

/** Block colors shared by the editor, the card and the viewport. */
export const BLOCK_COLORS: Record<keyof Blocks, string> = {
  A: '#fb7185',
  B: '#fbbf24',
  C: '#60a5fa',
  D: '#4ade80',
};

export function isBlocksMatrix(M: Matrix): boolean {
  return M.length === 4 && M.every((row) => row.length === 4);
}

export function splitBlocks(M: Matrix): Blocks {
  const part = (r0: number, c0: number): Matrix =>
    M.slice(r0, r0 + 2).map((row) => row.slice(c0, c0 + 2));
  return { A: part(0, 0), B: part(0, 2), C: part(2, 0), D: part(2, 2) };
}

export function joinBlocks(b: Blocks): Matrix {
  return [
    [b.A[0][0], b.A[0][1], b.B[0][0], b.B[0][1]],
    [b.A[1][0], b.A[1][1], b.B[1][0], b.B[1][1]],
    [b.C[0][0], b.C[0][1], b.D[0][0], b.D[0][1]],
    [b.C[1][0], b.C[1][1], b.D[1][0], b.D[1][1]],
  ];
}

/** Block-diagonal matrix diag(A, D) — the decoupled two-plane picture. */
export function blockDiagonal(A: Matrix, D: Matrix): Matrix {
  return [
    [A[0][0], A[0][1], 0, 0],
    [A[1][0], A[1][1], 0, 0],
    [0, 0, D[0][0], D[0][1]],
    [0, 0, D[1][0], D[1][1]],
  ];
}

const maxAbs = (K: Matrix): number =>
  K.reduce((m, row) => Math.max(m, ...row.map(Math.abs)), 0);

const isZeroBlock = (K: Matrix, tol = 1e-9): boolean => maxAbs(K) <= tol;

/** Largest entry — used by the viewport to fade out zeroed blocks. */
export const blockMagnitude = (K: Matrix): number => maxAbs(K);

export type BlockStructure = 'diagonal' | 'upper' | 'lower' | 'coupled';

/**
 * The zero pattern of the off-diagonal blocks. 'diagonal' takes precedence
 * over the triangular cases; 'coupled' means both planes feed each other.
 */
export function blockStructure(M: Matrix): BlockStructure {
  const b = splitBlocks(M);
  const noB = isZeroBlock(b.B);
  const noC = isZeroBlock(b.C);
  if (noB && noC) return 'diagonal';
  if (noC) return 'upper';
  if (noB) return 'lower';
  return 'coupled';
}

/**
 * Schur complement of the top-left block: S = D − C A⁻¹ B.
 * `null` when A isn't invertible (nothing to eliminate through).
 */
export function schur(M: Matrix): Matrix | null {
  const b = splitBlocks(M);
  const Ainv = inverse(b.A);
  if (!Ainv) return null;
  const corr = matMul(matMul(b.C, Ainv), b.B);
  return b.D.map((row, i) => row.map((x, j) => x - corr[i][j]));
}

/**
 * The block unit-lower-triangular factor L = [[I, 0], [C A⁻¹, I]] whose
 * multiplication clears the lower-left block: L⁻¹ M = [[A, B], [0, S]].
 * `null` when A isn't invertible.
 */
export function blockLowerUnit(M: Matrix): Matrix | null {
  const b = splitBlocks(M);
  const Ainv = inverse(b.A);
  if (!Ainv) return null;
  const CAinv = matMul(b.C, Ainv);
  return [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [CAinv[0][0], CAinv[0][1], 1, 0],
    [CAinv[1][0], CAinv[1][1], 0, 1],
  ];
}

/** The block upper-triangular factor U = L⁻¹M = [[A, B], [0, S]]. */
export function blockUpper(M: Matrix): Matrix | null {
  const S = schur(M);
  if (!S) return null;
  const b = splitBlocks(M);
  return [
    [b.A[0][0], b.A[0][1], b.B[0][0], b.B[0][1]],
    [b.A[1][0], b.A[1][1], b.B[1][0], b.B[1][1]],
    [0, 0, S[0][0], S[0][1]],
    [0, 0, S[1][0], S[1][1]],
  ];
}

/**
 * The eigenvalue union of a block-triangular / block-diagonal M:
 * det(λI − M) = det(λI − A) · det(λI − D), so every eigenvalue of either
 * diagonal block is an eigenvalue of M. Returns `null` when M is coupled
 * (the characteristic polynomial no longer factors). The card displays
 * λ(A) and λ(D) side by side as the two halves of the union.
 */
export function eigenUnion(M: Matrix): { fromA: EigenEntry[]; fromD: EigenEntry[] } | null {
  if (blockStructure(M) === 'coupled') return null;
  const b = splitBlocks(M);
  return { fromA: eigen(b.A), fromD: eigen(b.D) };
}

/** Four block products of X·Y, grouped by slot of the result. */
export interface BlockProductSlot {
  /** Result slot: '(1,1)' … '(2,2)'. */
  slot: string;
  /** The two 2×2 terms, with display labels like 'A·E'. */
  terms: { label: string; M: Matrix }[];
  /** The sum — the actual block of X·Y. */
  sum: Matrix;
}

const SECOND_FACTOR_LETTERS: Record<keyof Blocks, string> = {
  A: 'E',
  B: 'F',
  C: 'G',
  D: 'H',
};

/**
 * The blockwise product X·Y: each result block is a sum of two 2×2
 * products — (AE + BG, AF + BH; CE + DG, CF + DH) for factors whose
 * blocks are named A/B/C/D and E/F/G/H.
 */
export function blockProducts(X: Matrix, Y: Matrix): BlockProductSlot[] {
  const x = splitBlocks(X);
  const y = splitBlocks(Y);
  const grid: (keyof Blocks)[][] = [
    ['A', 'B'],
    ['C', 'D'],
  ];

  const out: BlockProductSlot[] = [];
  for (let r = 0; r < 2; r++) {
    for (let c = 0; c < 2; c++) {
      const t1 = {
        label: `${grid[r][0]}·${SECOND_FACTOR_LETTERS[grid[0][c]]}`,
        M: matMul(x[grid[r][0]], y[grid[0][c]]),
      };
      const t2 = {
        label: `${grid[r][1]}·${SECOND_FACTOR_LETTERS[grid[1][c]]}`,
        M: matMul(x[grid[r][1]], y[grid[1][c]]),
      };
      const sum = t1.M.map((row, i) => row.map((v, j) => v + t2.M[i][j]));
      out.push({ slot: `(${r + 1},${c + 1})`, terms: [t1, t2], sum });
    }
  }
  return out;
}

/** Trace rule: tr M = tr A + tr D, whatever the off-diagonal blocks are. */
export function traceBlocks(M: Matrix): { total: number; parts: number } {
  const b = splitBlocks(M);
  return { total: trace(M), parts: trace(b.A) + trace(b.D) };
}

/** The classic identity det M = det A · det S (needs A invertible). */
export function detFactorization(M: Matrix): {
  detM: number;
  detA: number;
  detS: number | null;
} {
  const b = splitBlocks(M);
  const S = schur(M);
  return { detM: det(M), detA: det(b.A), detS: S ? det(S) : null };
}

/**
 * Conditional variance of y given x for a symmetric 2×2 covariance S —
 * the 1+1 Schur complement s_yy − s_xy²/s_xx. The Feature 6 bridge.
 */
export function conditionalVariance2(S: Matrix): number | null {
  const sxx = S[0][0];
  if (Math.abs(sxx) < 1e-12) return null;
  return S[1][1] - (S[0][1] * S[1][0]) / sxx;
}

/** Block transpose rule as data: Mᵀ = [[Aᵀ, Cᵀ], [Bᵀ, Dᵀ]]. */
export function blockTranspose(M: Matrix): { named: { label: string; M: Matrix }[]; M: Matrix } {
  const b = splitBlocks(M);
  return {
    named: [
      { label: 'Aᵀ', M: transpose(b.A) },
      { label: 'Cᵀ', M: transpose(b.C) },
      { label: 'Bᵀ', M: transpose(b.B) },
      { label: 'Dᵀ', M: transpose(b.D) },
    ],
    M: transpose(M),
  };
}
