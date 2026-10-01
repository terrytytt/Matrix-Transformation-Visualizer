/**
 * Eigenvalue / eigenvector helpers.
 *
 *  - 2x2 uses the closed-form quadratic (exact and easy to explain).
 *  - 3x3 delegates to ml-matrix's eigenvalue decomposition (Hessenberg + QR).
 *
 * Results are grouped by *value*, so a repeated eigenvalue shows up once with
 * its algebraic multiplicity and the list of independent eigenvectors that
 * span its eigenspace. Complex values are reported (never drawn) so the UI
 * can explain that no real invariant direction exists.
 */

import { EigenvalueDecomposition } from 'ml-matrix';
import { EPS, Matrix, Vector, identity, matVec, norm, normalize, tidy } from './matrix';

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
/* 3x3 — numerical decomposition                                       */
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
  if (a.length === 2) {
    return Math.abs(a[0] * b[1] - a[1] * b[0]) < tol * scale;
  }
  const c: Vector = [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
  return norm(c) < tol * scale;
}

/* ------------------------------------------------------------------ */
/* Public API                                                          */
/* ------------------------------------------------------------------ */

export function eigen(A: Matrix): EigenEntry[] {
  if (A.length === 2) return eigen2(A);
  if (A.length === 3) return eigen3(A);
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
