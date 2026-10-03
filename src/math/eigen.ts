/**
 * Eigenvalue / eigenvector helpers.
 *
 *  - 2x2 uses the closed-form quadratic (exact and easy to explain).
 *  - n ≥ 3 delegates to ml-matrix's eigenvalue decomposition (Hessenberg + QR).
 *
 * Results are grouped by *value*, so a repeated eigenvalue shows up once with
 * its algebraic multiplicity and the list of independent eigenvectors that
 * span its eigenspace. Complex values are reported (never drawn) so the UI
 * can explain that no real invariant direction exists.
 */

import { EigenvalueDecomposition, SingularValueDecomposition } from 'ml-matrix';
import {
  EPS,
  Matrix,
  Vector,
  det,
  identity,
  inverse,
  isSymmetric,
  matMul,
  matVec,
  norm,
  normalize,
  symmetricPart,
  tidy,
  transpose,
} from './matrix';

export interface EigenEntry {
  /** Real part of the eigenvalue. */
  real: number;
  /** Imaginary part (0 for real eigenvalues). */
  imag: number;
  /** The first real eigenvector (unit length), or `null` when complex. */
  vector: Vector | null;
  /** Every independent real eigenvector found for this value. */
  vectors: Vector[];
  /** Algebraic multiplicity — how often this value is repeated. */
  multiplicity: number;
  /** True when A − λI = 0, i.e. *every* direction is an eigenvector. */
  fullSpace: boolean;
}

const isComplex = (e: EigenEntry) => Math.abs(e.imag) > EPS;

/** Round to a stable key so floats coming out of the solver can be grouped. */
const key = (x: number) => Math.round(x * 1e7) / 1e7;

/* ------------------------------------------------------------------ */
/* 2x2 — closed form                                                   */
/* ------------------------------------------------------------------ */

function eigen2(A: Matrix): EigenEntry[] {
  const [[a, b], [c, d]] = A;
  const tr = a + d;
  const dt = a * d - b * c;
  const disc = tr * tr - 4 * dt;

  /** Null space of (A − λI) for a 2×2: pick the longest row. */
  const nullVector = (lam: number): Vector => {
    const rows: Vector[] = [
      [a - lam, b],
      [c, d - lam],
    ];
    const best = norm(rows[1]) > norm(rows[0]) ? 1 : 0;
    if (norm(rows[best]) < 1e-7) return [1, 0]; // λI — any direction works
    const [p, q] = rows[best];
    return normalize([q, -p]);
  };

  const fullSpace = (lam: number) =>
    Math.abs(a - lam) < 1e-9 && Math.abs(b) < 1e-9 && Math.abs(c) < 1e-9 && Math.abs(d - lam) < 1e-9;

  const make = (lam: number, mult: number): EigenEntry => {
    const v = nullVector(lam);
    const full = fullSpace(lam);
    return {
      real: tidy(lam),
      imag: 0,
      vector: v,
      vectors: full ? identity(2) : [v],
      multiplicity: mult,
      fullSpace: full,
    };
  };

  if (disc >= -1e-12) {
    const s = Math.sqrt(Math.max(disc, 0));
    const l1 = (tr + s) / 2;
    const l2 = (tr - s) / 2;
    if (Math.abs(l1 - l2) <= 1e-9) return [make(l1, 2)];
    return [make(l1, 1), make(l2, 1)];
  }

  const re = tr / 2;
  const im = Math.sqrt(-disc) / 2;
  return [
    { real: tidy(re), imag: tidy(im), vector: null, vectors: [], multiplicity: 1, fullSpace: false },
    { real: tidy(re), imag: tidy(-im), vector: null, vectors: [], multiplicity: 1, fullSpace: false },
  ];
}

/* ------------------------------------------------------------------ */
/* n ≥ 3 — numerical decomposition                                     */
/* ------------------------------------------------------------------ */

function eigen3(A: Matrix): EigenEntry[] {
  let evd: EigenvalueDecomposition;
  try {
    evd = new EigenvalueDecomposition(A);
  } catch {
    return [];
  }

  const re = evd.realEigenvalues;
  const im = evd.imaginaryEigenvalues;
  const V = evd.eigenvectorMatrix.to2DArray();
  const n = A.length;

  interface Bucket {
    real: number;
    imag: number;
    vectors: Vector[];
    multiplicity: number;
  }
  const buckets = new Map<string, Bucket>();
  const bucketOf = (r: number, i: number): Bucket => {
    const k = `${key(r)}|${key(i)}`;
    let b = buckets.get(k);
    if (!b) {
      b = { real: key(r), imag: key(i), vectors: [], multiplicity: 0 };
      buckets.set(k, b);
    }
    return b;
  };

  for (let i = 0; i < re.length; i++) {
    const lam = re[i];
    const mu = im[i] ?? 0;
    const bucket = bucketOf(lam, mu);
    bucket.multiplicity++;
    if (Math.abs(mu) > 1e-9) continue; // complex: no real eigenvector to draw

    const raw: Vector = V.map((row) => row[i] ?? 0);
    if (norm(raw) < 1e-9) continue;

    // Guard against defective matrices where the solver returns a vector that
    // is only approximately an eigenvector.
    const Av = matVec(A, raw);
    const resid = norm(Av.map((x, k) => x - raw[k] * lam));
    if (resid > 1e-6 * Math.max(1, norm(raw))) continue;

    const v = normalize(raw);
    if (!bucket.vectors.some((s) => isParallel(s, v))) bucket.vectors.push(v);
  }

  const out: EigenEntry[] = [];
  for (const b of buckets.values()) {
    const vectors = b.vectors.slice(0, n);
    // A ≈ λI means every direction is an eigenvector.
    const full =
      Math.abs(b.imag) <= EPS &&
      A.every((row, i) => row.every((x, j) => Math.abs(x - (i === j ? b.real : 0)) < 1e-9));
    out.push({
      real: b.real,
      imag: b.imag,
      vector: vectors[0] ?? null,
      vectors,
      multiplicity: b.multiplicity,
      fullSpace: full,
    });
  }

  // Stable, descending-by-|value| order reads better than solver order.
  out.sort((x, y) => y.real - x.real || y.imag - x.imag);
  return out;
}

function isParallel(a: Vector, b: Vector, tol = 1e-6): boolean {
  const scale = Math.max(norm(a), norm(b));
  if (scale < 1e-12) return true;
  // ‖a‖²‖b‖² − (a·b)² = ‖a × b‖² — works in every dimension (and equals
  // the 2D cross product for n = 2), unlike a hand-rolled 3D cross product.
  const dot = a.reduce((s, x, i) => s + x * (b[i] ?? 0), 0);
  const na = norm(a);
  const nb = norm(b);
  const sinNorm = Math.sqrt(Math.max(0, na * na * nb * nb - dot * dot));
  return sinNorm < tol * na * nb;
}

/* ------------------------------------------------------------------ */
/* Public API                                                          */
/* ------------------------------------------------------------------ */

export function eigen(A: Matrix): EigenEntry[] {
  if (A.length === 2) return eigen2(A);
  if (A.length >= 3) return eigen3(A); // ml-matrix's EVD handles any n
  return [];
}

/** True when at least one eigenvalue is complex. */
export function hasComplexEigenvalue(entries: EigenEntry[]): boolean {
  return entries.some(isComplex);
}

/** Entries that have at least one drawable (real) eigenvector. */
export function realEigenpairs(entries: EigenEntry[]): EigenEntry[] {
  return entries.filter((e) => e.vectors.length > 0);
}

/* ------------------------------------------------------------------ */
/* Singular values & the image ellipse                                 */
/* ------------------------------------------------------------------ */

export interface EllipseAxis {
  /** Semi-axis length σᵢ (a singular value of A). */
  sigma: number;
  /** Unit direction of the semi-axis in the *image* space. */
  dir: Vector;
  /** Endpoint of the semi-axis: σᵢ · dir, i.e. A applied to a unit vector. */
  tip: Vector;
}

/**
 * Singular values σ₁ ≥ σ₂ (≥ σ₃) of A — the factors by which A stretches
 * the principal directions of the unit ball. They are exactly the
 * semi-axis lengths of the ellipse/sphere that A maps the unit
 * circle/ball onto, and they always satisfy σ₁·σ₂ = |det A| (2×2).
 *
 * Uses ml-matrix's SVD; `svd.s` is a plain `number[]`, descending.
 */
export function singularValues(A: Matrix): number[] {
  try {
    const svd = new SingularValueDecomposition(A);
    return [...svd.s].sort((a, b) => b - a);
  } catch {
    return [];
  }
}

/**
 * Semi-axes of the image of the unit circle/ball under A: pairs
 * (σᵢ, direction uᵢ) where uᵢ is the i-th left singular vector, sorted by
 * descending σᵢ. The tip σᵢ·uᵢ is where the corresponding principal
 * direction of the unit circle lands.
 */
export function ellipseAxes(A: Matrix): EllipseAxis[] {
  try {
    const svd = new SingularValueDecomposition(A);
    const U = svd.leftSingularVectors.to2DArray();
    const axes: EllipseAxis[] = svd.s.map((sigma, i) => {
      const dir = normalize(U.map((row) => row[i]).slice(0, A.length));
      return { sigma, dir, tip: dir.map((x) => x * sigma) };
    });
    return axes.sort((a, b) => b.sigma - a.sigma);
  } catch {
    return [];
  }
}

/* ------------------------------------------------------------------ */
/* Symmetric part — the axes of the quadratic form                     */
/* ------------------------------------------------------------------ */

export interface SymPair {
  /** Eigenvalue λᵢ. */
  lam: number;
  /** Matching unit direction (an eigenvector of the symmetric matrix). */
  dir: Vector;
}

/**
 * Orthonormal eigenpairs of the symmetric matrix S = (A+Aᵀ)/2, sorted by
 * descending eigenvalue. These directions are the natural axes of the
 * level sets of xᵀAx, because xᵀAx = xᵀSx for every x.
 *
 * Eigenvectors of a symmetric matrix are orthogonal in exact arithmetic;
 * we re-orthonormalize with Gram–Schmidt to clean up solver noise and pad
 * any missing directions so the result always spans Rⁿ.
 */
export function symmetricBasis(S: Matrix): SymPair[] {
  const n = S.length;
  const raw: SymPair[] = [];
  for (const e of eigen(S)) {
    if (Math.abs(e.imag) > EPS) continue;
    for (const v of e.vectors) raw.push({ lam: e.real, dir: v });
  }

  const orthonormal: SymPair[] = [];
  const add = (p: SymPair) => {
    let v = p.dir.slice();
    for (const q of orthonormal) {
      const d = v.reduce((s, x, i) => s + x * q.dir[i], 0);
      v = v.map((x, i) => x - d * q.dir[i]);
    }
    const nv = norm(v);
    if (nv > 1e-7) orthonormal.push({ lam: p.lam, dir: v.map((x) => x / nv) });
  };
  raw.forEach(add);

  // Fill in whatever the solver failed to provide (should be rare).
  if (orthonormal.length < n) {
    for (let k = 0; k < n && orthonormal.length < n; k++) {
      const e: Vector = Array.from({ length: n }, (_, i) => (i === k ? 1 : 0));
      add({ lam: 0, dir: e });
    }
  }

  return orthonormal.sort((a, b) => b.lam - a.lam);
}

/* ------------------------------------------------------------------ */
/* Definiteness of the quadratic form xᵀAx                             */
/* ------------------------------------------------------------------ */

export type Definiteness =
  | 'positiveDefinite'
  | 'negativeDefinite'
  | 'positiveSemidefinite'
  | 'negativeSemidefinite'
  | 'indefinite'
  | 'zero';

export interface DefinitenessInfo {
  kind: Definiteness;
  /** Eigenvalues of S = (A+Aᵀ)/2, descending. */
  lambdas: number[];
  /** Leading principal minors of S: Δ₁, Δ₂ (, Δ₃) — Sylvester's criterion. */
  minors: number[];
  /** True when A itself isn't symmetric, i.e. S decided the verdict. */
  viaSymmetricPart: boolean;
}

/**
 * Classify the sign of the quadratic form xᵀAx.
 *
 * xᵀAx = xᵀSx with S = (A+Aᵀ)/2, so *only the symmetric part decides* —
 * a non-symmetric matrix can still be positive definite (a shear is!).
 * Eigenvalues of S give the verdict:
 *
 *   all λ > 0        positive definite   (xᵀAx > 0 for every x ≠ 0)
 *   all λ < 0        negative definite   (always negative — an upside-down bowl)
 *   all λ ≥ 0 (some 0)  positive semi-definite
 *   all λ ≤ 0 (some 0)  negative semi-definite
 *   mixed signs      indefinite          (a saddle — both signs occur)
 *   S = 0            zero                (form vanishes identically)
 *
 * The leading principal minors of S are returned too, so the UI can show
 * Sylvester's criterion as a cross-check (PD ⇔ every Δᵢ > 0).
 */
export function definiteness(A: Matrix): DefinitenessInfo {
  const S = symmetricPart(A);
  const lambdas: number[] = [];
  for (const e of eigen(S)) {
    if (Math.abs(e.imag) > EPS) continue;
    for (let m = 0; m < e.multiplicity; m++) lambdas.push(e.real);
  }
  lambdas.sort((a, b) => b - a);

  const scale = Math.max(0, ...lambdas.map(Math.abs));
  let kind: Definiteness;
  if (scale < 1e-9) {
    kind = 'zero';
  } else {
    const tol = 1e-9 * scale;
    const pos = lambdas.filter((l) => l > tol).length;
    const neg = lambdas.filter((l) => l < -tol).length;
    const z = lambdas.length - pos - neg;
    if (pos === lambdas.length) kind = 'positiveDefinite';
    else if (neg === lambdas.length) kind = 'negativeDefinite';
    else if (neg === 0 && z > 0) kind = 'positiveSemidefinite';
    else if (pos === 0 && z > 0) kind = 'negativeSemidefinite';
    else kind = 'indefinite';
  }

  // Leading principal minors of S (Sylvester): Δ₁ = s₁₁, then det of the
  // top-left k×k block.
  const minors: number[] = [S[0][0]];
  for (let k = 2; k <= A.length; k++) {
    const sub = S.slice(0, k).map((row) => row.slice(0, k));
    minors.push(det(sub));
  }

  return { kind, lambdas, minors, viaSymmetricPart: !isSymmetric(A) };
}

/* ------------------------------------------------------------------ */
/* Factorizations: SVD and the spectral decomposition                  */
/* ------------------------------------------------------------------ */

/** Diagonal matrix from a list of diagonal entries. */
const diagOf = (l: number[]): Matrix =>
  l.map((_, i) => l.map((_, j) => (i === j ? l[i] : 0)));

/** Build a matrix from column vectors. */
const columnsOf = (cols: Vector[]): Matrix =>
  cols.length === 0 ? [] : cols[0].map((_, i) => cols.map((c) => c[i]));

/** max |A − B| entrywise — the residual reported by both factorizations. */
const maxAbsDiff = (A: Matrix, B: Matrix): number => {
  let m = 0;
  for (let i = 0; i < A.length; i++) {
    for (let j = 0; j < A.length; j++) {
      m = Math.max(m, Math.abs((A[i][j] ?? 0) - (B[i]?.[j] ?? 0)));
    }
  }
  return m;
};

export interface SvdFactors {
  /** Left singular vectors as columns — the output directions uᵢ. */
  U: Matrix;
  /** Diagonal matrix of singular values (descending). */
  sigma: Matrix;
  /** Vᵀ — the input directions vᵢ are its rows (columns of V). */
  Vt: Matrix;
  /** Singular values σ₁ ≥ σ₂ (≥ σ₃), descending. */
  s: number[];
  /** max |U Σ Vᵀ − A| — how closely the factors reproduce A. */
  residual: number;
}

/**
 * A = U Σ Vᵀ — turn the input by Vᵀ, stretch along the coordinate axes by
 * σ, then point the result with U. Works for *every* matrix: when A is
 * singular the smallest σ is 0 (the collapse direction), and the product
 * still reconstructs A exactly.
 *
 * The three factors are sorted together by descending σ so the readout
 * stays in the conventional order without breaking U Σ Vᵀ = A.
 */
export function svdFactors(A: Matrix): SvdFactors | null {
  const n = A.length;
  let svd: SingularValueDecomposition;
  try {
    svd = new SingularValueDecomposition(A);
  } catch {
    return null;
  }
  const s: number[] = [...svd.s];
  if (s.length !== n) return null; // non-square input: out of scope for this app
  const Uraw = svd.leftSingularVectors.to2DArray();
  const Vraw = svd.rightSingularVectors.to2DArray(); // A = U·diag(s)·Vᵀ
  if (Uraw.length !== n || Vraw.length !== n) return null;

  // Permute the U and V columns *together* so Σ stays aligned with them.
  const order = s
    .map((sigma, i) => ({ sigma, i }))
    .sort((a, b) => b.sigma - a.sigma)
    .map((x) => x.i);
  const U = columnsOf(order.map((k) => Uraw.map((row) => row[k])));
  const V = columnsOf(order.map((k) => Vraw.map((row) => row[k])));
  const sorted = order.map((k) => s[k]);
  const sigma = diagOf(sorted);
  const Vt = transpose(V);

  return { U, sigma, Vt, s: sorted, residual: maxAbsDiff(matMul(matMul(U, sigma), Vt), A) };
}

export type SpectralKind = 'orthogonal' | 'general' | 'defective' | 'complex';

export interface SpectralFactors {
  kind: SpectralKind;
  /** Eigenvector basis as columns — Q (symmetric) or P; null when there is no real diagonalization. */
  P: Matrix | null;
  /** Diagonal eigenvalue matrix Λ (symmetric) / D. */
  D: Matrix | null;
  /** P⁻¹ — equal to Qᵀ when `orthogonal`. */
  Pinv: Matrix | null;
  /** Eigenvalues along the diagonal of D, in column order of P. */
  lambdas: number[];
  /** max |P D P⁻¹ − A|. */
  residual: number;
  /** True when A is symmetric, so P is orthonormal and P⁻¹ = Pᵀ. */
  orthogonal: boolean;
}

/**
 * A = P D P⁻¹ (or A = QΛQᵀ when A is symmetric) — the spectral
 * decomposition. Failure modes are reported instead of guessed:
 *
 *   complex    some λ is not real → no real eigenbasis exists (rotations);
 *   defective  a repeated λ lacks a second independent direction (shears),
 *              so the columns of P can't span Rⁿ.
 */
export function spectralFactors(A: Matrix): SpectralFactors {
  const n = A.length;
  const fail = (kind: SpectralKind): SpectralFactors => ({
    kind,
    P: null,
    D: null,
    Pinv: null,
    lambdas: [],
    residual: 0,
    orthogonal: false,
  });

  // Symmetric: the spectral theorem gives an orthonormal eigenbasis.
  if (isSymmetric(A)) {
    const pairs = symmetricBasis(A);
    const Q = columnsOf(pairs.map((p) => p.dir));
    const D = diagOf(pairs.map((p) => p.lam));
    const Qt = transpose(Q);
    return {
      kind: 'orthogonal',
      P: Q,
      D,
      Pinv: Qt,
      lambdas: pairs.map((p) => p.lam),
      residual: maxAbsDiff(matMul(matMul(Q, D), Qt), A),
      orthogonal: true,
    };
  }

  const entries = eigen(A);
  if (entries.some((e) => Math.abs(e.imag) > EPS)) return fail('complex');

  // Greedily take n linearly independent real eigenvectors.
  const chosen: { lam: number; vec: Vector }[] = [];
  for (const e of entries) {
    for (const vec of e.vectors) {
      if (chosen.length >= n) break;
      let w = vec.slice(); // Gram–Schmidt against what we already accepted
      for (const c of chosen) {
        const d = w.reduce((s, x, i) => s + x * c.vec[i], 0);
        w = w.map((x, i) => x - d * c.vec[i]);
      }
      if (norm(w) > 1e-6) chosen.push({ lam: e.real, vec });
    }
    if (chosen.length >= n) break;
  }
  if (chosen.length < n) return fail('defective');

  const P = columnsOf(chosen.map((c) => c.vec));
  const D = diagOf(chosen.map((c) => c.lam));
  const Pinv = inverse(P);
  if (!Pinv) return fail('defective');
  return {
    kind: 'general',
    P,
    D,
    Pinv,
    lambdas: chosen.map((c) => c.lam),
    residual: maxAbsDiff(matMul(matMul(P, D), Pinv), A),
    orthogonal: false,
  };
}
