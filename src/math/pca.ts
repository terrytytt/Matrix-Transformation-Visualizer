/**
 * PCA helpers: point-cloud datasets, their covariance matrix, and the
 * principal-component readouts the inspector shows.
 *
 * The whole story hangs on one fact: the covariance matrix S of a point
 * cloud is symmetric, so everything the app already knows about symmetric
 * matrices (eigenbasis, ellipse, spectral theorem, definiteness) applies to
 * the *data*. PCA = "diagonalize S and read the variances off Λ".
 */

import { Matrix, Vector, matVec } from './matrix';
import { symmetricBasis } from './eigen';

export type DatasetKind = 'blob' | 'clusters';

/** Live controls on the Point cloud card. */
export interface DatasetParams {
  /** Correlation of the underlying gaussian, −0.95 … 0.95. */
  rho: number;
  /** Rotation of the cloud (and of the cluster axis), degrees. */
  theta: number;
  /** Spread along the local x axis. */
  sx: number;
  /** Spread along the local y axis. */
  sy: number;
  /** Spread along z (3D only; ignored in the plane). */
  sz: number;
  /** Extra isotropic jitter, 0 … 1. */
  noise: number;
}

/** A generated sample plus everything the store needs to keep in sync. */
export interface Dataset {
  kind: DatasetKind;
  params: DatasetParams;
  /** Raw samples as generated (mean usually ≠ 0 — the point of step 1). */
  points: Vector[];
  /** Points minus their mean — what PCA actually works on. */
  centered: Vector[];
  mean: Vector;
}

/** Sample sizes stay modest so every frame remains cheap. */
export const datasetN = (dim: 2 | 3): number => (dim === 2 ? 90 : 150);

/* ------------------------------------------------------------------ */
/* Seeded generators                                                   */
/* ------------------------------------------------------------------ */

/** mulberry32 — tiny deterministic PRNG so screenshots never drift. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard normal draws (Box–Muller). */
function gaussians(rand: () => number, count: number): number[] {
  const out: number[] = [];
  while (out.length < count) {
    const u = Math.max(rand(), 1e-12);
    const v = rand();
    const r = Math.sqrt(-2 * Math.log(u));
    out.push(r * Math.cos(2 * Math.PI * v));
    if (out.length < count) out.push(r * Math.sin(2 * Math.PI * v));
  }
  return out;
}

const rotZ = (deg: number): Matrix => {
  const t = (deg * Math.PI) / 180;
  const c = Math.cos(t);
  const s = Math.sin(t);
  return [
    [c, -s, 0],
    [s, c, 0],
    [0, 0, 1],
  ];
};

/**
 * Generate `n` samples for a dataset preset.
 *
 * `blob` draws z ~ N(0, I), pushes it through a Cholesky factor of the
 * target covariance
 *
 *     B = [[sx², ρ·sx·sy, 0],
 *          [ρ·sx·sy, sy²,  0],
 *          [0,      0,   sz²]]
 *
 * (so ρ controls the correlation and the spreads control the marginal
 * variances), rotates the result by θ and adds isotropic jitter. The sample
 * covariance of the output converges to R·B·Rᵀ + noise²·I exactly.
 *
 * `clusters` reuses the same blob shape around two centers sitting symmetrically
 * on the rotated x axis — the classic "PCA sees one ellipse, not two clusters"
 * counter-example.
 *
 * The seed depends only on (kind, dim), so dragging a slider morphs the *same*
 * underlying gaussian sample instead of reshuffling it.
 */
export function generateCloud(
  kind: DatasetKind,
  p: DatasetParams,
  dim: 2 | 3,
): Vector[] {
  const n = datasetN(dim);
  const seed = (kind === 'clusters' ? 0x517cc1b7 : 0x9e3779b9) ^ (dim * 7919);
  const rand = mulberry32(seed);
  const rho = Math.max(-0.95, Math.min(0.95, p.rho));
  const z = gaussians(rand, n * dim);
  const w = gaussians(rand, n * dim);
  const R = rotZ(p.theta);

  // Lower Cholesky factor of B (2D block; z is independent).
  const root = Math.sqrt(Math.max(1 - rho * rho, 1e-9));
  const L: Matrix =
    dim === 2
      ? [
          [p.sx, 0],
          [rho * p.sy, p.sy * root],
        ]
      : [
          [p.sx, 0, 0],
          [rho * p.sy, p.sy * root, 0],
          [0, 0, p.sz],
        ];

  const sep = 2.6;
  const centers: Vector[] =
    kind === 'clusters'
      ? [
          matVec(R, [sep / 2, 0, 0]).slice(0, dim),
          matVec(R, [-sep / 2, 0, 0]).slice(0, dim),
        ]
      : [];

  const out: Vector[] = [];
  for (let i = 0; i < n; i++) {
    const base: Vector = Array.from({ length: dim }, (_, k) => z[i * dim + k]);
    const shaped = matVec(L, base).slice(0, dim);
    const rotated = matVec(R, [...shaped, 0]).slice(0, dim);
    const jitter: Vector = Array.from(
      { length: dim },
      (_, k) => p.noise * w[i * dim + k],
    );
    const c = kind === 'clusters' ? centers[i % 2] : [0, 0, 0].slice(0, dim);
    out.push(rotated.map((x, k) => x + jitter[k] + c[k]));
  }
  return out;
}

/* ------------------------------------------------------------------ */
/* Statistics                                                          */
/* ------------------------------------------------------------------ */

export function meanOf(points: Vector[]): Vector {
  const n = points.length;
  const d = points[0]?.length ?? 0;
  const m: Vector = Array.from({ length: d }, () => 0);
  for (const p of points) for (let k = 0; k < d; k++) m[k] += p[k] / n;
  return m;
}

/** Sample covariance (centered internally, n − 1 denominator). */
export function covarianceOf(points: Vector[]): Matrix {
  const n = points.length;
  const d = points[0]?.length ?? 0;
  if (n < 2) return Array.from({ length: d }, () => Array.from({ length: d }, () => 0));
  const m = meanOf(points);
  const S: Matrix = Array.from({ length: d }, () => Array.from({ length: d }, () => 0));
  for (const p of points) {
    for (let i = 0; i < d; i++) {
      const a = p[i] - m[i];
      for (let j = 0; j < d; j++) S[i][j] += (a * (p[j] - m[j])) / (n - 1);
    }
  }
  return S;
}

export interface PcaResult {
  n: number;
  mean: Vector;
  /** Covariance matrix S — the symmetric matrix everything else reads. */
  S: Matrix;
  /** Variances λ₁ ≥ λ₂ (≥ λ₃) ≥ 0, clamped against solver noise. */
  lambdas: number[];
  /** Unit principal directions, matching `lambdas`. */
  vectors: Vector[];
  /** √λ — the 1σ semi-axes of the data ellipse. */
  std: number[];
  /** λᵢ / Σλ — how much variance each PC explains. */
  explained: number[];
  /** Running total of `explained`. */
  cumulative: number[];
  /** Σλ = trace(S) = total variance. */
  trace: number;
  /** Mean squared reconstruction error keeping only PC1: λ₂ (+ λ₃). */
  recon1: number;
}

/** Mean, covariance and the eigendecomposition of S in one pass. */
export function pcaOf(points: Vector[]): PcaResult {
  const mean = meanOf(points);
  const S = covarianceOf(points);
  const pairs = symmetricBasis(S);
  const lambdas = pairs.map((p) => Math.max(p.lam, 0));
  const vectors = pairs.map((p) => p.dir);
  const trace = lambdas.reduce((s, l) => s + l, 0);
  const explained = lambdas.map((l) => (trace > 1e-12 ? l / trace : 1 / lambdas.length));
  let acc = 0;
  const cumulative = explained.map((e) => (acc += e));
  return {
    n: points.length,
    mean,
    S,
    lambdas,
    vectors,
    std: lambdas.map((l) => Math.sqrt(l)),
    explained,
    cumulative,
    trace,
    recon1: trace - (lambdas[0] ?? 0),
  };
}

/* ------------------------------------------------------------------ */
/* Geometry used by the viewports and the demos                        */
/* ------------------------------------------------------------------ */

/**
 * Symmetric square root √S = Q·√Λ·Qᵀ. The data ellipse (1σ contour) is the
 * image of the unit circle under √S — note that this is *not* the existing
 * "ellipse of A" layer when A = S, whose semi-axes are λ, not √λ.
 */
export function matrixSqrt(S: Matrix): Matrix {
  const pairs = symmetricBasis(S);
  const d = S.length;
  const out: Matrix = Array.from({ length: d }, () => Array.from({ length: d }, () => 0));
  for (const { lam, dir } of pairs) {
    const r = Math.sqrt(Math.max(lam, 0));
    for (let i = 0; i < d; i++) {
      for (let j = 0; j < d; j++) out[i][j] += r * dir[i] * dir[j];
    }
  }
  return out;
}

/** Rank-1 projector v vᵀ — the matrix behind "project onto PC1". */
export function outerProject(v: Vector): Matrix {
  return v.map((a) => v.map((b) => a * b));
}

/** Orthogonal projection of p onto the span of the first `k` directions. */
export function projectTo(p: Vector, dirs: Vector[], k: number): Vector {
  const out: Vector = p.map(() => 0);
  for (let i = 0; i < k && i < dirs.length; i++) {
    const d = dirs[i];
    const c = p.reduce((s, x, j) => s + x * d[j], 0);
    for (let j = 0; j < out.length; j++) out[j] += c * d[j];
  }
  return out;
}

/** Covariance of the preset's default cloud — lets a dataset preset fill the Matrix A slot at declaration time. */
export function datasetCovariance(
  kind: DatasetKind,
  params: DatasetParams,
  dim: 2 | 3,
): Matrix {
  return covarianceOf(generateCloud(kind, params, dim));
}


