import { DEFAULT_MATRIX_4, Dim } from './state/store';
import { Matrix, Vector } from './math/matrix';
import { DatasetKind, DatasetParams, datasetCovariance } from './math/pca';

export interface Preset {
  name: string;
  dim: Dim;
  matrix: Matrix;
  vector?: Vector;
  /** One-line geometric interpretation shown in the info panel. */
  description: string;
  /**
   * Present on the PCA dataset presets: choosing one loads a point cloud
   * instead of a plain matrix, and `matrix` holds that cloud's covariance
   * S (so the Matrix A slot already reads correctly for declarations and
   * the preset lookup in describe()).
   */
  dataset?: { kind: DatasetKind; params: DatasetParams };
}

const c = (deg: number) => Math.round(Math.cos((deg * Math.PI) / 180) * 1e4) / 1e4;
const s = (deg: number) => Math.round(Math.sin((deg * Math.PI) / 180) * 1e4) / 1e4;

const rot2 = (deg: number): Matrix => [
  [c(deg), -s(deg)],
  [s(deg), c(deg)],
];

const rotZ = (deg: number): Matrix => [
  [c(deg), -s(deg), 0],
  [s(deg), c(deg), 0],
  [0, 0, 1],
];

const rotY = (deg: number): Matrix => [
  [c(deg), 0, s(deg)],
  [0, 1, 0],
  [-s(deg), 0, c(deg)],
];

const rotX = (deg: number): Matrix => [
  [1, 0, 0],
  [0, c(deg), -s(deg)],
  [0, s(deg), c(deg)],
];

/**
 * PCA dataset presets: the declared `matrix` is the covariance S of the
 * default cloud, so picking one from the select still shows a meaningful
 * matrix — the ControlPanel's applyDataset then regenerates the actual
 * samples (deterministically, so it always lands on this S).
 */
const datasetPreset = (
  name: string,
  dim: 2 | 3,
  kind: DatasetKind,
  params: DatasetParams,
  description: string,
): Preset => ({
  name,
  dim,
  matrix: datasetCovariance(kind, params, dim),
  description,
  dataset: { kind, params },
});

/** Default slider values per dataset preset (sz only matters in 3D). */
const P = (
  rho: number,
  theta: number,
  sx: number,
  sy: number,
  sz: number,
  noise: number,
): DatasetParams => ({ rho, theta, sx, sy, sz, noise });

const DATASETS_2D: Preset[] = [
  datasetPreset(
    'Correlated cloud',
    2,
    'blob',
    P(0.9, 0, 1.6, 0.7, 0.8, 0.15),
    'A tilted gaussian blob — the PC1 direction is the long diagonal, and the sliders morph this very sample.',
  ),
  datasetPreset(
    'Axis-aligned cloud',
    2,
    'blob',
    P(0.1, 0, 2.2, 0.55, 0.8, 0.15),
    'Variance mostly along x: S is nearly diagonal, so the principal components sit on the axes.',
  ),
  datasetPreset(
    'Isotropic cloud',
    2,
    'blob',
    P(0, 0, 1.3, 1.3, 1.3, 0.15),
    'No direction stands out: λ₁ ≈ λ₂, explained variance splits ~50/50 and the PC axes are arbitrary.',
  ),
  datasetPreset(
    'Two clusters',
    2,
    'clusters',
    P(0.25, 0, 0.55, 0.4, 0.5, 0.12),
    'Two blobs with an empty gap: PCA only reads second moments, so it sees one ellipse stretched across the gap.',
  ),
  datasetPreset(
    'Different units (y ×5)',
    2,
    'blob',
    P(0.35, 0, 1, 4, 1.5, 0.3),
    'y is measured in units ~5× coarser than x, so PC1 is the y-axis until you standardize.',
  ),
];

export const PRESETS_2D: Preset[] = [
  {
    name: 'Identity',
    dim: 2,
    matrix: [
      [1, 0],
      [0, 1],
    ],
    description: 'The do-nothing matrix. Basis vectors stay put, area is preserved.',
  },
  {
    name: 'Rotation 45°',
    dim: 2,
    matrix: rot2(45),
    description: 'A rigid turn. det = 1, so area and length are unchanged.',
  },
  {
    name: 'Scale ×2',
    dim: 2,
    matrix: [
      [2, 0],
      [0, 2],
    ],
    description: 'Uniform blow-up: every length ×2, so the area ×4 (det = 4).',
  },
  {
    name: 'Stretch x, squash y',
    dim: 2,
    matrix: [
      [2, 0],
      [0, 0.5],
    ],
    description: 'Axis-aligned stretch. det = 1 — the two effects cancel exactly.',
  },
  {
    name: 'Shear (horizontal)',
    dim: 2,
    matrix: [
      [1, 1],
      [0, 1],
    ],
    description: 'Slides rows of the grid sideways; area unchanged (det = 1). One repeated eigenvalue λ = 1.',
  },
  {
    name: 'Reflection (y-axis)',
    dim: 2,
    matrix: [
      [-1, 0],
      [0, 1],
    ],
    description: 'Mirrors the plane. det = -1: area is preserved but orientation flips.',
  },
  {
    name: 'Projection onto x-axis',
    dim: 2,
    matrix: [
      [1, 0],
      [0, 0],
    ],
    description:
      'Everything collapses onto a line. Singular (det = 0), rank 1, null space = y-axis. Idempotent: A² = A.',
  },
  {
    name: 'Projection onto line y = x',
    dim: 2,
    matrix: [
      [0.5, 0.5],
      [0.5, 0.5],
    ],
    description:
      'Flattens the plane onto the diagonal, along the direction [1,-1]. Idempotent: A² = A — a second pass changes nothing.',
  },
  {
    name: 'Oblique projection',
    dim: 2,
    matrix: [
      [1, 0],
      [2, 0],
    ],
    description:
      'Slides every point onto the slanted line y = 2x along the y-axis. Still idempotent: A² = A.',
  },
  {
    name: 'Symmetric stretch',
    dim: 2,
    matrix: [
      [2, 1],
      [1, 2],
    ],
    description: 'det = 3 with positive orientation. Eigenvectors [1,1] (λ = 3) and [1,-1] (λ = 1).',
  },
  {
    name: 'Flip + stretch',
    dim: 2,
    matrix: [
      [1, 2],
      [2, 1],
    ],
    description:
      'det = -3: the unit square is tripled AND turned inside out. Indefinite — a saddle, where xᵀAx takes both signs.',
  },
  {
    name: 'Negative definite',
    dim: 2,
    matrix: [
      [-2, -1],
      [-1, -2],
    ],
    description:
      'The upside-down bowl: xᵀAx < 0 for every x ≠ 0 (λ of S = -3, -1). Level sets of negative c are ellipses; positive c has none.',
  },
  {
    name: 'Quarter turn',
    dim: 2,
    matrix: [
      [0, -1],
      [1, 0],
    ],
    description: 'A 90° rotation. Eigenvalues are ±i — no real direction survives unchanged.',
  },
  {
    name: 'Rank-1 collapse',
    dim: 2,
    matrix: [
      [1, 2],
      [2, 4],
    ],
    description: 'Columns are parallel, so the plane flattens onto the line spanned by [1,2].',
  },
  ...DATASETS_2D,
];

export const PRESETS_3D: Preset[] = [
  {
    name: 'Identity',
    dim: 3,
    matrix: identity3(),
    description: 'The do-nothing matrix in space.',
  },
  {
    name: 'Rotation about z 45°',
    dim: 3,
    matrix: rotZ(45),
    description: 'Spins the xy-plane; the z-axis is an eigenvector with λ = 1.',
  },
  {
    name: 'Rotation about y 45°',
    dim: 3,
    matrix: rotY(45),
    description: 'Spins around the vertical axis; x and z trade places.',
  },
  {
    name: 'Rotation about x 45°',
    dim: 3,
    matrix: rotX(45),
    description: 'Tilts the yz-plane; the x-axis is an eigenvector with λ = 1.',
  },
  {
    name: 'Scale ×2',
    dim: 3,
    matrix: [
      [2, 0, 0],
      [0, 2, 0],
      [0, 0, 2],
    ],
    description: 'Uniform scaling: lengths ×2, volume ×8 (det = 8).',
  },
  {
    name: 'Stretch x, squash z',
    dim: 3,
    matrix: [
      [2, 0, 0],
      [0, 1, 0],
      [0, 0, 0.5],
    ],
    description: 'det = 1 — the cube is reshaped but keeps exactly the same volume.',
  },
  {
    name: 'Shear along x',
    dim: 3,
    matrix: [
      [1, 1, 0],
      [0, 1, 0],
      [0, 0, 1],
    ],
    description: 'Slides layers of space sideways; volume unchanged (det = 1).',
  },
  {
    name: 'Reflection (xy-plane)',
    dim: 3,
    matrix: [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, -1],
    ],
    description:
      'Flips space upside down. det = -1: volume kept, handedness reversed. Indefinite — the xy-plane gives energy, z gives the opposite.',
  },
  {
    name: 'Symmetric stretch 3D',
    dim: 3,
    matrix: [
      [2, 1, 0],
      [1, 2, 0],
      [0, 0, 1],
    ],
    description: 'det = 3. Eigenvectors [1,1,0] (λ = 3), [1,-1,0] (λ = 1), [0,0,1] (λ = 1).',
  },
  {
    name: 'Permutation (120° turn)',
    dim: 3,
    matrix: [
      [0, 0, 1],
      [1, 0, 0],
      [0, 1, 0],
    ],
    description: 'A 120° rotation about the diagonal [1,1,1]. Eigenvalues are the cube roots of 1.',
  },
  {
    name: 'Projection onto xy-plane',
    dim: 3,
    matrix: [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 0],
    ],
    description: 'Squashes space flat. Rank 2, null space = z-axis, det = 0. Idempotent: A² = A.',
  },
  {
    name: 'Projection onto plane z = x + y',
    dim: 3,
    matrix: [
      [1, 0, 0],
      [0, 1, 0],
      [1, 1, 0],
    ],
    description:
      'Slides the cube straight down the z-axis onto a tilted plane. Idempotent: A² = A — points on the plane are already fixed.',
  },
  {
    name: 'Projection onto diagonal line',
    dim: 3,
    matrix: [
      [1 / 3, 1 / 3, 1 / 3],
      [1 / 3, 1 / 3, 1 / 3],
      [1 / 3, 1 / 3, 1 / 3],
    ],
    description:
      'Everything collapses onto the line through [1,1,1]. Rank 1, det = 0, and idempotent: A² = A.',
  },
  {
    name: 'Rank-1 collapse',
    dim: 3,
    matrix: [
      [1, 2, 3],
      [2, 4, 6],
      [3, 6, 9],
    ],
    description: 'All columns are parallel: the whole solid collapses onto a single line.',
  },
  datasetPreset(
    'Correlated cloud',
    3,
    'blob',
    P(0.9, 0, 1.6, 0.7, 0.8, 0.15),
    'A tilted gaussian blob in space — PC1 is the long diagonal, PC2 and PC3 complete the frame.',
  ),
  datasetPreset(
    'Axis-aligned cloud',
    3,
    'blob',
    P(0.1, 0, 2.2, 0.55, 1, 0.15),
    'Variance mostly along x: S is nearly diagonal, so the principal components sit on the axes.',
  ),
  datasetPreset(
    'Isotropic cloud',
    3,
    'blob',
    P(0, 0, 1.3, 1.3, 1.3, 0.15),
    'No direction stands out: λ₁ ≈ λ₂ ≈ λ₃ and the PC axes are arbitrary.',
  ),
  datasetPreset(
    'Two clusters',
    3,
    'clusters',
    P(0.25, 0, 0.55, 0.4, 0.5, 0.12),
    'Two blobs with an empty gap: PCA only reads second moments, so it sees one ellipsoid across the gap.',
  ),
  datasetPreset(
    'Different units (y ×5)',
    3,
    'blob',
    P(0.35, 0, 1, 4, 1.5, 0.3),
    'y is measured in units ~5× coarser than x, so PC1 is the y-axis until you standardize.',
  ),
];

function identity3(): Matrix {
  return [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
}

/* ------------------------------------------------------------------ */
/* 4×4 — partitioned matrices acting on two coupled planes             */
/* ------------------------------------------------------------------ */

const block2 = (
  A: Matrix,
  B: Matrix,
  C: Matrix,
  D: Matrix,
): Matrix => [
  [A[0][0], A[0][1], B[0][0], B[0][1]],
  [A[1][0], A[1][1], B[1][0], B[1][1]],
  [C[0][0], C[0][1], D[0][0], D[0][1]],
  [C[1][0], C[1][1], D[1][0], D[1][1]],
];

const I2: Matrix = [
  [1, 0],
  [0, 1],
];

export const PRESETS_4D: Preset[] = [
  {
    name: 'Coupled planes',
    dim: 4,
    matrix: DEFAULT_MATRIX_4.map((r) => r.slice()),
    description:
      'All four blocks are live: each plane feeds the other — y₁ = A x₁ + B x₂ and y₂ = C x₁ + D x₂.',
  },
  {
    name: 'Two independent planes',
    dim: 4,
    matrix: block2(
      [
        [1.6, 0.4],
        [0.2, 0.9],
      ],
      [
        [0, 0],
        [0, 0],
      ],
      [
        [0, 0],
        [0, 0],
      ],
      [
        [0.6, -0.8],
        [0.8, 0.6],
      ],
    ),
    description:
      'Block-diagonal: the planes never interact — det = det A · det D, and the eigenvalues of M are the union of both blocks’.',
  },
  {
    name: 'Block upper triangular',
    dim: 4,
    matrix: block2(
      [
        [1.5, 0.5],
        [0, 1],
      ],
      [
        [0.8, 0],
        [0.4, -0.6],
      ],
      [
        [0, 0],
        [0, 0],
      ],
      [
        [1.2, 0.3],
        [0, 0.8],
      ],
    ),
    description:
      'Plane 2 evolves on its own while plane 1 also feels plane 2 through B — block triangular, so det = det A · det D and λ(M) = λ(A) ∪ λ(D).',
  },
  {
    name: 'Shear between planes',
    dim: 4,
    matrix: block2(
      I2,
      [
        [0, 0],
        [0, 0],
      ],
      [
        [1, 0.5],
        [0, 1],
      ],
      I2,
    ),
    description:
      'A pure block shear: plane 1’s position slides plane 2 sideways. det = 1 — the 4-volume is untouched — and every eigenvalue is 1.',
  },
  {
    name: 'Covariance pairs',
    dim: 4,
    matrix: block2(
      [
        [2, 0.5],
        [0.5, 1.2],
      ],
      [
        [0.4, 0.2],
        [-0.1, 0.3],
      ],
      [
        [0.4, -0.1],
        [0.2, 0.3],
      ],
      [
        [1.4, 0.3],
        [0.3, 0.9],
      ],
    ),
    description:
      'A symmetric, positive definite covariance of two (x, y) pairs — the Schur complement of the rose block is the blue block’s covariance after regressing the rose one out.',
  },
  {
    name: 'Project onto plane 1',
    dim: 4,
    matrix: block2(
      I2,
      [
        [0, 0],
        [0, 0],
      ],
      [
        [0, 0],
        [0, 0],
      ],
      [
        [0, 0],
        [0, 0],
      ],
    ),
    description:
      'Block-diagonal projection: plane 1 is kept, plane 2 collapses to the origin. Idempotent (M² = M), rank 2, det = 0.',
  },
];

export function presetsFor(dim: Dim): Preset[] {
  return dim === 2 ? PRESETS_2D : dim === 3 ? PRESETS_3D : PRESETS_4D;
}

/** Matrices used by the composition demo — deliberately non-commuting. */
export const COMPOSITION_PAIR_2: { A: Matrix; B: Matrix; vector: Vector; name: string } = {
  name: 'Composition: rotate then shear',
  A: rot2(45),
  B: [
    [1, 1],
    [0, 1],
  ],
  vector: [2, 0],
};

export const COMPOSITION_PAIR_3: { A: Matrix; B: Matrix; vector: Vector; name: string } = {
  name: 'Composition: rotate then scale',
  A: rotZ(45),
  B: [
    [2, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ],
  vector: [2, 0, 0],
};

export const COMPOSITION_PAIR_4: { A: Matrix; B: Matrix; vector: Vector; name: string } = {
  name: 'Composition: two-plane rotate then shear',
  // Both factors are block-diagonal, so each plane composes on its own —
  // but rot·shear ≠ shear·rot inside plane 1, so the order still matters.
  A: block2(rot2(45), rot2(-30), zero2(), zero2()),
  B: block2(
    [
      [1, 1],
      [0, 1],
    ],
    zero2(),
    zero2(),
    rot2(30),
  ),
  vector: [2, 0, 1, 0],
};

/**
 * The second factor the Block multiply demo loads when the N slot still
 * holds the identity (a user-edited N is never overwritten). All four
 * blocks E…H are nonzero, so every slot of M·N genuinely shows two 2×2
 * products being added — AE + BG, AF + BH, CE + DG, CF + DH — and
 * det N ≈ 1.206 ≠ 0, so the product doesn't collapse.
 */
export const BLOCK_FACTOR_4: Matrix = block2(
  [
    [1, 0.5],
    [0, 1],
  ],
  [
    [0, 0.4],
    [0.4, 0],
  ],
  [
    [0.3, 0],
    [0, 0.3],
  ],
  [
    [1, -0.4],
    [0.4, 1],
  ],
);

function zero2(): Matrix {
  return [
    [0, 0],
    [0, 0],
  ];
}
