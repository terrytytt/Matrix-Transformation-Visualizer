import { create } from 'zustand';
import {
  Matrix,
  Vector,
  identity,
  inverse,
  matMul,
} from '../math/matrix';

export type Dim = 2 | 3;
export type CompStep = 0 | 1 | 2;

export interface Layers {
  grid: boolean;
  basis: boolean;
  vector: boolean;
  determinant: boolean;
  eigen: boolean;
  columnSpace: boolean;
  nullSpace: boolean;
  transpose: boolean;
}

export interface AppState {
  dim: Dim;
  /** The matrix A the user is studying. */
  matrix: Matrix;
  /** The second matrix B used by the composition demo. */
  matrixB: Matrix;
  /** The vector x being transformed. */
  vector: Vector;

  showInverse: boolean;
  compActive: boolean;
  compStep: CompStep;
  layers: Layers;
  presetName: string | null;

  setDim: (d: Dim) => void;
  setMatrix: (M: Matrix) => void;
  setMatrixEntry: (i: number, j: number, v: number) => void;
  setMatrixB: (M: Matrix) => void;
  setMatrixBEntry: (i: number, j: number, v: number) => void;
  setVector: (v: Vector) => void;
  setVectorEntry: (i: number, v: number) => void;

  setShowInverse: (v: boolean) => void;
  setCompActive: (v: boolean) => void;
  setCompStep: (s: CompStep) => void;
  toggleLayer: (k: keyof Layers) => void;
  setLayers: (patch: Partial<Layers>) => void;

  applyMatrices: (A: Matrix, B: Matrix, vector: Vector, name: string) => void;
  transposeA: () => void;
  invertAPermanently: () => void;
  multiplyAB: (order: 'AB' | 'BA') => void;
  reset: () => void;
}

export const DEFAULT_MATRIX_2: Matrix = [
  [2, 1],
  [1, 2],
];

export const DEFAULT_MATRIX_3: Matrix = [
  [2, 1, 0],
  [1, 2, 0],
  [0, 0, 1],
];

export const DEFAULT_VECTOR_2: Vector = [2, -1];
export const DEFAULT_VECTOR_3: Vector = [1, 1, 1];

const ALL_LAYERS: Layers = {
  grid: true,
  basis: true,
  vector: true,
  determinant: true,
  eigen: false,
  columnSpace: false,
  nullSpace: false,
  transpose: false,
};

function defaultMatrices(d: Dim): { A: Matrix; B: Matrix; x: Vector } {
  if (d === 2) {
    return { A: DEFAULT_MATRIX_2.map((r) => r.slice()), B: identity(2), x: [...DEFAULT_VECTOR_2] };
  }
  return { A: DEFAULT_MATRIX_3.map((r) => r.slice()), B: identity(3), x: [...DEFAULT_VECTOR_3] };
}

export const useApp = create<AppState>((set, get) => ({
  dim: 2,
  matrix: DEFAULT_MATRIX_2.map((r) => r.slice()),
  matrixB: identity(2),
  vector: [...DEFAULT_VECTOR_2],

  showInverse: false,
  compActive: false,
  compStep: 0,
  layers: { ...ALL_LAYERS },
  presetName: 'Symmetric stretch',

  setDim: (d) => {
    if (get().dim === d) return;
    const { A, B, x } = defaultMatrices(d);
    set({
      dim: d,
      matrix: A,
      matrixB: B,
      vector: x,
      showInverse: false,
      compActive: false,
      compStep: 0,
      layers: { ...ALL_LAYERS },
      presetName: d === 2 ? 'Symmetric stretch' : 'Symmetric stretch 3D',
    });
  },

  setMatrix: (M) => set({ matrix: M, presetName: null }),
  setMatrixEntry: (i, j, v) =>
    set((s) => {
      const M = s.matrix.map((r) => r.slice());
      M[i][j] = v;
      return { matrix: M, presetName: null };
    }),

  setMatrixB: (M) => set({ matrixB: M }),
  setMatrixBEntry: (i, j, v) =>
    set((s) => {
      const M = s.matrixB.map((r) => r.slice());
      M[i][j] = v;
      return { matrixB: M };
    }),

  setVector: (v) => set({ vector: v }),
  setVectorEntry: (i, v) =>
    set((s) => {
      const x = s.vector.slice();
      x[i] = v;
      return { vector: x };
    }),

  setShowInverse: (v) => set({ showInverse: v, compActive: v ? false : get().compActive }),
  setCompActive: (v) => set({ compActive: v, showInverse: v ? false : get().showInverse }),
  setCompStep: (s) => set({ compStep: s }),

  toggleLayer: (k) => set((s) => ({ layers: { ...s.layers, [k]: !s.layers[k] } })),
  setLayers: (patch) => set((s) => ({ layers: { ...s.layers, ...patch } })),

  applyMatrices: (A, B, x, name) =>
    set({
      matrix: A,
      matrixB: B,
      vector: x,
      presetName: name,
      showInverse: false,
      compActive: false,
      compStep: 0,
    }),

  transposeA: () =>
    set((s) => {
      const n = s.dim;
      const M = Array.from({ length: n }, (_, i) =>
        Array.from({ length: n }, (_, j) => s.matrix[j][i]),
      );
      return { matrix: M, presetName: null };
    }),

  invertAPermanently: () =>
    set((s) => {
      const inv = inverse(s.matrix);
      return inv ? { matrix: inv, presetName: null } : s;
    }),

  multiplyAB: (order) =>
    set((s) => {
      const M = order === 'AB' ? matMul(s.matrix, s.matrixB) : matMul(s.matrixB, s.matrix);
      return { matrix: M, presetName: null };
    }),

  reset: () => {
    const s = get();
    const { A, B, x } = defaultMatrices(s.dim);
    set({
      matrix: A,
      matrixB: B,
      vector: x,
      showInverse: false,
      compActive: false,
      compStep: 0,
      layers: { ...ALL_LAYERS },
      presetName: s.dim === 2 ? 'Symmetric stretch' : 'Symmetric stretch 3D',
    });
  },
}));
