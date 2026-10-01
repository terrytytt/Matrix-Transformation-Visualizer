import { Dim } from './state/store';
import { Matrix, Vector } from './math/matrix';

export interface Preset {
  name: string;
  dim: Dim;
  matrix: Matrix;
  vector?: Vector;
  /** One-line geometric interpretation shown in the info panel. */
  description: string;
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
    description: 'Everything collapses onto a line. Singular (det = 0), rank 1, null space = y-axis.',
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
    description: 'det = -3: the unit square is tripled AND turned inside out.',
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
    description: 'Flips space upside down. det = -1: volume kept, handedness reversed.',
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
    description: 'Squashes space flat. Rank 2, null space = z-axis, det = 0.',
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
];

function identity3(): Matrix {
  return [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
}

export function presetsFor(dim: Dim): Preset[] {
  return dim === 2 ? PRESETS_2D : PRESETS_3D;
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
