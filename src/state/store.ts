import { create } from 'zustand';
import {
  Matrix,
  Vector,
  identity,
  inverse,
  matMul,
} from '../math/matrix';
import {
  Dataset,
  DatasetKind,
  DatasetParams,
  covarianceOf,
  generateCloud,
  meanOf,
} from '../math/pca';

// Re-exported so components can type against the store without reaching
// into the math layer themselves.
export type { Dataset, DatasetKind, DatasetParams } from '../math/pca';

export type Dim = 2 | 3 | 4;
export type CompStep = 0 | 1 | 2;
/**
 * Which factorization / block / PCA demo is running: UΣVᵀ, P D P⁻¹,
 * the two PCA stories, block Schur elimination and block multiplication.
 */
export type Decomp = 'svd' | 'spectral' | 'pcaProjection' | 'pcaRotation' | 'schur' | 'blockMul';
export type DecompStep = 0 | 1 | 2 | 3;

export interface Layers {
  grid: boolean;
  basis: boolean;
  vector: boolean;
  determinant: boolean;
  eigen: boolean;
  columnSpace: boolean;
  nullSpace: boolean;
  transpose: boolean;
  /** Unit circle → image ellipse (with σ semi-axes). */
  ellipse: boolean;
  /** Level sets of the quadratic form xᵀAx = c. */
  levelSets: boolean;
  /** Sign field of xᵀAx — warm positive / cool negative. */
  form: boolean;
  /** The dataset's samples (plus the white mean marker). */
  points: boolean;
  /** 1σ / 2σ contours of the data — semi-axes √λ along the PCs. */
  dataEllipse: boolean;
  /** Dashed segments from each centered point to its PC1 reconstruction. */
  residuals: boolean;
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
  /** I → A → A² demo: does a second application change anything? */
  idemActive: boolean;
  idemStep: CompStep;
  /** Factorization demo: 4 steps I → factor₁ → factor₁·factor₂ → A. */
  decomp: Decomp | null;
  decompStep: DecompStep;
  layers: Layers;
  presetName: string | null;
  /** Loaded point cloud (PCA datasets); Matrix A then holds its covariance S. */
  dataset: Dataset | null;

  /**
   * "Watch the transformation" scrubber: how much of the active matrix is
   * exposed — 0 draws plain space (before), 1 the fully transformed picture
   * (after). Between the two the viewport shows the morph together with
   * before/after ghosts and travel streaks.
   */
  transT: number;
  /** True while the ▶ Play morph (identity → active matrix) is running. */
  transPlaying: boolean;
  /** Pin the before-state ghosts visible even at t = 1. */
  ghostAlways: boolean;
  /** Side-by-side before | after viewport (2D and 3D). */
  splitView: boolean;

  setTransT: (t: number) => void;
  playTransition: () => void;
  pauseTransition: () => void;
  toggleGhost: () => void;
  toggleSplit: () => void;

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
  setIdemActive: (v: boolean) => void;
  setIdemStep: (s: CompStep) => void;
  setDecomp: (v: Decomp | null) => void;
  setDecompStep: (s: DecompStep) => void;
  toggleLayer: (k: keyof Layers) => void;
  setLayers: (patch: Partial<Layers>) => void;

  applyMatrices: (A: Matrix, B: Matrix, vector: Vector, name: string) => void;
  /** Load a PCA dataset: generate the cloud, set A = Cov(x), reveal the data layers. */
  applyDataset: (name: string, kind: DatasetKind, params: DatasetParams) => void;
  /** Re-generate the active cloud with tweaked slider params (A follows as S). */
  updateDataset: (patch: Partial<DatasetParams>) => void;
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

export const DEFAULT_MATRIX_4: Matrix = [
  [1.5, 0.3, 0, 0.7],
  [0, 1, 0.5, 0],
  [-0.4, 0, 1, -0.5],
  [0.6, -0.3, 0.2, 1.2],
];

/** x₁ = (2, −1) in plane 1, x₂ = (1, 0.5) in plane 2. */
export const DEFAULT_VECTOR_4: Vector = [2, -1, 1, 0.5];

const ALL_LAYERS: Layers = {
  grid: true,
  basis: true,
  vector: true,
  determinant: true,
  eigen: false,
  columnSpace: false,
  nullSpace: false,
  transpose: false,
  ellipse: false,
  levelSets: false,
  form: false,
  points: false,
  dataEllipse: false,
  residuals: false,
};

function defaultMatrices(d: Dim): { A: Matrix; B: Matrix; x: Vector } {
  if (d === 2) {
    return { A: DEFAULT_MATRIX_2.map((r) => r.slice()), B: identity(2), x: [...DEFAULT_VECTOR_2] };
  }
  if (d === 4) {
    return { A: DEFAULT_MATRIX_4.map((r) => r.slice()), B: identity(4), x: [...DEFAULT_VECTOR_4] };
  }
  return { A: DEFAULT_MATRIX_3.map((r) => r.slice()), B: identity(3), x: [...DEFAULT_VECTOR_3] };
}

/** The preset each dimension boots into (matched by the select + describe()). */
export const defaultPresetName = (d: Dim): string =>
  d === 2 ? 'Symmetric stretch' : d === 3 ? 'Symmetric stretch 3D' : 'Coupled planes';

/**
 * State patch applied whenever A is edited by hand: the loaded cloud no
 * longer describes A (A ≠ S anymore), so it detaches — and a running PCA
 * demo, which tells the cloud's story, exits with it.
 */
function detach(s: AppState): Partial<AppState> {
  return {
    dataset: null,
    ...(s.decomp === 'pcaProjection' || s.decomp === 'pcaRotation'
      ? { decomp: null, decompStep: 0 }
      : {}),
  };
}

/** Rebuild a dataset's samples after a parameter change (seeded → stable). */
function rebuild(
  kind: DatasetKind,
  params: DatasetParams,
  dim: 2 | 3,
): Pick<Dataset, 'points' | 'centered' | 'mean'> {
  const points = generateCloud(kind, params, dim);
  const mean = meanOf(points);
  const centered = points.map((p) => p.map((x, i) => x - mean[i]));
  return { points, centered, mean };
}

/** Length of the ▶ Play morph from identity to the active matrix. */
export const TRANSITION_MS = 2000;

/**
 * Bumped whenever the scrubber is settled or moved by hand, so a stale rAF
 * from an in-flight Play loop can never overwrite a reset.
 */
let transitionToken = 0;

/**
 * Patch applied by every matrix / dataset / dim change: the picture snaps
 * back to the full after-state (t = 1, no ghosts) — press ▶ to demonstrate
 * the morph again.
 */
function settleTransition(): Pick<AppState, 'transT' | 'transPlaying'> {
  transitionToken++;
  return { transT: 1, transPlaying: false };
}

export const useApp = create<AppState>((set, get) => ({
  dim: 2,
  matrix: DEFAULT_MATRIX_2.map((r) => r.slice()),
  matrixB: identity(2),
  vector: [...DEFAULT_VECTOR_2],

  showInverse: false,
  compActive: false,
  compStep: 0,
  idemActive: false,
  idemStep: 0,
  decomp: null,
  decompStep: 0,
  layers: { ...ALL_LAYERS },
  presetName: 'Symmetric stretch',
  dataset: null,

  transT: 1,
  transPlaying: false,
  ghostAlways: false,
  splitView: false,

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
      idemActive: false,
      idemStep: 0,
      decomp: null,
      decompStep: 0,
      layers: { ...ALL_LAYERS },
      presetName: defaultPresetName(d),
      dataset: null,
      ...settleTransition(),
    });
  },

  // Editing A by hand breaks the link to the dataset (A is no longer S) —
  // a running PCA demo exits with it; the other demos keep their own rules.
  setMatrix: (M) => set((s) => ({ matrix: M, presetName: null, ...detach(s), ...settleTransition() })),
  setMatrixEntry: (i, j, v) =>
    set((s) => {
      const M = s.matrix.map((r) => r.slice());
      M[i][j] = v;
      return { matrix: M, presetName: null, ...detach(s), ...settleTransition() };
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

  setShowInverse: (v) =>
    set({
      showInverse: v,
      compActive: v ? false : get().compActive,
      idemActive: v ? false : get().idemActive,
      decomp: v ? null : get().decomp,
    }),
  setCompActive: (v) =>
    set({
      compActive: v,
      showInverse: v ? false : get().showInverse,
      idemActive: v ? false : get().idemActive,
      decomp: v ? null : get().decomp,
    }),
  setCompStep: (s) => set({ compStep: s }),
  setIdemActive: (v) =>
    set({
      idemActive: v,
      showInverse: v ? false : get().showInverse,
      compActive: v ? false : get().compActive,
      idemStep: v ? 0 : get().idemStep,
      decomp: v ? null : get().decomp,
    }),
  setIdemStep: (s) => set({ idemStep: s }),
  setDecomp: (v) =>
    set({
      decomp: v,
      decompStep: 0,
      showInverse: v ? false : get().showInverse,
      compActive: v ? false : get().compActive,
      idemActive: v ? false : get().idemActive,
    }),
  setDecompStep: (s) => set({ decompStep: s }),

  toggleLayer: (k) => set((s) => ({ layers: { ...s.layers, [k]: !s.layers[k] } })),
  setLayers: (patch) => set((s) => ({ layers: { ...s.layers, ...patch } })),

  // Scrubbing pauses playback: the slider is the manual override.
  setTransT: (t) => {
    transitionToken++;
    set({ transT: Math.min(1, Math.max(0, t)), transPlaying: false });
  },
  pauseTransition: () => {
    transitionToken++;
    set({ transPlaying: false });
  },
  playTransition: () => {
    const token = ++transitionToken;
    const t0 = performance.now();
    set({ transT: 0, transPlaying: true });
    const step = () => {
      if (transitionToken !== token || !get().transPlaying) return;
      const t = Math.min(1, (performance.now() - t0) / TRANSITION_MS);
      set({ transT: t, transPlaying: t < 1 });
      if (t < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  },
  toggleGhost: () => set((s) => ({ ghostAlways: !s.ghostAlways })),
  toggleSplit: () => set((s) => ({ splitView: !s.splitView })),

  applyMatrices: (A, B, x, name) =>
    set({
      matrix: A,
      matrixB: B,
      vector: x,
      presetName: name,
      showInverse: false,
      compActive: false,
      compStep: 0,
      idemActive: false,
      idemStep: 0,
      decomp: null,
      decompStep: 0,
      dataset: null,
      ...settleTransition(),
    }),

  applyDataset: (name, kind, params) =>
    set((s) => {
      if (s.dim === 4) return {}; // datasets live in 2D/3D only
      const fresh = rebuild(kind, params, s.dim);
      return {
        matrix: covarianceOf(fresh.points),
        dataset: { kind, params, ...fresh },
        presetName: name,
        showInverse: false,
        compActive: false,
        compStep: 0,
        idemActive: false,
        idemStep: 0,
        decomp: null,
        decompStep: 0,
        layers: { ...s.layers, points: true, dataEllipse: true },
        ...settleTransition(),
      };
    }),

  updateDataset: (patch) =>
    set((s) => {
      if (!s.dataset || s.dim === 4) return {};
      const params = { ...s.dataset.params, ...patch };
      const fresh = rebuild(s.dataset.kind, params, s.dim);
      return {
        matrix: covarianceOf(fresh.points),
        dataset: { ...s.dataset, params, ...fresh },
        showInverse: false,
        compActive: false,
        compStep: 0,
        idemActive: false,
        idemStep: 0,
        decomp: null,
        decompStep: 0,
        ...settleTransition(),
      };
    }),

  transposeA: () =>
    set((s) => {
      const n = s.dim;
      const M = Array.from({ length: n }, (_, i) =>
        Array.from({ length: n }, (_, j) => s.matrix[j][i]),
      );
      return { matrix: M, presetName: null, ...detach(s), ...settleTransition() };
    }),

  invertAPermanently: () =>
    set((s) => {
      const inv = inverse(s.matrix);
      return inv ? { matrix: inv, presetName: null, ...detach(s), ...settleTransition() } : s;
    }),

  multiplyAB: (order) =>
    set((s) => {
      const M = order === 'AB' ? matMul(s.matrix, s.matrixB) : matMul(s.matrixB, s.matrix);
      return { matrix: M, presetName: null, ...detach(s), ...settleTransition() };
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
      idemActive: false,
      idemStep: 0,
      decomp: null,
      decompStep: 0,
      layers: { ...ALL_LAYERS },
      presetName: defaultPresetName(s.dim),
      dataset: null,
      ...settleTransition(),
    });
  },
}));
