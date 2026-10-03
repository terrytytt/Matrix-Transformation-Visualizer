import { useMemo } from 'react';
import { Matrix, Vector, det, identity, inverse, isSymmetric, matMul } from '../math/matrix';
import { EigenEntry, SvdFactors, SpectralFactors, spectralFactors, svdFactors } from '../math/eigen';
import {
  BlockStructure,
  Blocks,
  blockLowerUnit,
  blockStructure,
  blockUpper,
  eigenUnion,
  schur,
  splitBlocks,
  traceBlocks,
} from '../math/blocks';
import {
  PcaResult,
  matrixSqrt,
  outerProject,
  pcaOf,
  projectTo,
} from '../math/pca';
import { Dataset, useApp } from './store';

export interface ActiveMatrix {
  matrix: Matrix;
  /** Symbolic label for the readout: "A", "A⁻¹", "I", "B", "A·B", "A²", "ΣVᵀ"… */
  label: string;
  /** Set when the matrix is singular and no inverse exists. */
  singular: boolean;
}

/**
 * The matrix the viewport should draw right now.
 *
 * Derived values are computed with `useMemo` rather than inside a zustand
 * selector: selectors must return a stable reference or React re-renders
 * forever, and `matMul` / `inverse` always allocate.
 */
export function useActiveMatrix(): ActiveMatrix {
  const A = useApp((s) => s.matrix);
  const B = useApp((s) => s.matrixB);
  const dim = useApp((s) => s.dim);
  const showInverse = useApp((s) => s.showInverse);
  const compActive = useApp((s) => s.compActive);
  const compStep = useApp((s) => s.compStep);
  const idemActive = useApp((s) => s.idemActive);
  const idemStep = useApp((s) => s.idemStep);
  const decomp = useApp((s) => s.decomp);
  const decompStep = useApp((s) => s.decompStep);

  return useMemo(() => {
    // In the 4×4 block view the matrix is called M (its blocks are A…D and
    // the second factor N, so nothing collides); everywhere else it's A.
    const nm = dim === 4 ? 'M' : 'A';
    const nm2 = dim === 4 ? 'N' : 'B';
    // Factorization demos: I → factor₁ → factor₁·factor₂ → A.
    if (decomp === 'svd') {
      const f = svdFactors(A);
      if (f) {
        const mats = [identity(dim), f.Vt, matMul(f.sigma, f.Vt), A];
        const labels = ['I', 'Vᵀ', 'ΣVᵀ', nm];
        const k = Math.min(decompStep, 3);
        return { matrix: mats[k], label: labels[k], singular: false };
      }
      return { matrix: A, label: nm, singular: false }; // defensive: SVD failed
    }
    if (decomp === 'spectral') {
      const f = spectralFactors(A);
      if (f.P && f.D && f.Pinv) {
        const mats = [identity(dim), f.Pinv, matMul(f.D, f.Pinv), A];
        const labels = f.orthogonal
          ? ['I', 'Qᵀ', 'ΛQᵀ', nm]
          : ['I', 'P⁻¹', 'D·P⁻¹', nm];
        const k = Math.min(decompStep, 3);
        return { matrix: mats[k], label: labels[k], singular: false };
      }
      return { matrix: A, label: nm, singular: false }; // defective / complex: show A
    }
    // Block demos (4×4 two-plane view). Schur elimination peels the block
    // lower-triangular shear L off M, leaving U = L⁻¹M block upper
    // triangular with the Schur complement S in its lower-right block;
    // M = L·U at the end, so det M = det A · det S (det L = 1).
    if (decomp === 'schur' && A.length === 4) {
      const L = blockLowerUnit(A);
      const U = blockUpper(A);
      if (L && U) {
        const mats = [A, L, U, A];
        const labels = ['M', 'L', 'U', 'M = L·U'];
        const k = Math.min(decompStep, 3);
        return {
          matrix: mats[k],
          label: labels[k],
          singular: Math.abs(det(mats[k])) < 1e-9,
        };
      }
      return { matrix: A, label: 'M', singular: Math.abs(det(A)) < 1e-9 };
    }
    // Block multiply: show each factor, then the blockwise product M·N.
    if (decomp === 'blockMul' && A.length === 4) {
      const k = Math.min(decompStep, 2);
      const mats = [B, A, matMul(A, B)];
      const labels = ['N', 'M', 'M·N'];
      return {
        matrix: mats[k],
        label: labels[k],
        singular: k === 2 && Math.abs(det(mats[k])) < 1e-9,
      };
    }
    // PCA demos: the grid stays neutral while the data story runs (centering
    // isn't linear), then applies the payoff matrix — the rank-1 projector
    // P₁ = v₁v₁ᵀ for the projection demo, Qᵀ → ΛQᵀ for the rotation.
    if (decomp === 'pcaProjection' || decomp === 'pcaRotation') {
      const f = spectralFactors(A);
      if (f.orthogonal && f.P && f.D && f.Pinv) {
        const k = Math.min(decompStep, 3);
        if (decomp === 'pcaProjection') {
          const v1 = f.P.map((row) => row[0] ?? 0);
          const mats = [identity(dim), identity(dim), identity(dim), outerProject(v1)];
          const labels = ['data', 'x − x̄', 'PC1, PC2', 'P₁'];
          return { matrix: mats[k], label: labels[k], singular: k === 3 };
        }
        const mats = [identity(dim), identity(dim), f.Pinv, matMul(f.D, f.Pinv)];
        const labels = ['data', 'x − x̄', 'Qᵀ', 'ΛQᵀ'];
        return {
          matrix: mats[k],
          label: labels[k],
          singular: k === 3 && f.lambdas.some((l) => Math.abs(l) < 1e-9),
        };
      }
      return { matrix: A, label: nm, singular: false }; // defensive: A isn't a data covariance
    }
    if (idemActive) {
      if (idemStep === 0) return { matrix: identity(dim), label: 'I', singular: false };
      if (idemStep === 1) return { matrix: A, label: nm, singular: false };
      return { matrix: matMul(A, A), label: `${nm}²`, singular: false };
    }
    if (compActive) {
      if (compStep === 0) return { matrix: identity(dim), label: 'I', singular: false };
      if (compStep === 1) return { matrix: B, label: nm2, singular: false };
      return { matrix: matMul(A, B), label: `${nm}·${nm2}`, singular: false };
    }
    if (showInverse) {
      const inv = inverse(A);
      return inv
        ? { matrix: inv, label: `${nm}⁻¹`, singular: false }
        : { matrix: A, label: nm, singular: true };
    }
    return { matrix: A, label: nm, singular: false };
  }, [
    A,
    B,
    dim,
    showInverse,
    compActive,
    compStep,
    idemActive,
    idemStep,
    decomp,
    decompStep,
  ]);
}

export interface DecompFacts {
  /** U Σ Vᵀ factors, or null when the SVD could not be computed. */
  svd: SvdFactors | null;
  /** P D P⁻¹ factors — check `kind` for defective / complex failure modes. */
  spectral: SpectralFactors;
}

/**
 * Both factorizations of A, memoized once per matrix — shared by the
 * factor cards (InfoPanel), the Animate controls and the viewport
 * markers so nobody recomputes an SVD on every frame.
 */
export function useDecompFacts(): DecompFacts {
  const A = useApp((s) => s.matrix);
  return useMemo(() => ({ svd: svdFactors(A), spectral: spectralFactors(A) }), [A]);
}

export interface DataFacts {
  /** Mean / covariance / PCs / λ / √λ / explained variance of the cloud. */
  pca: PcaResult;
  /** √S — the matrix whose image of the unit circle is the 1σ data ellipse. */
  sqrtS: Matrix;
  /** Each centered point projected onto PC1 (the reconstruction kept at k = 1). */
  proj: Vector[];
}

/**
 * Everything the viewports draw for a dataset that costs more than a frame
 * to derive: the eigendecomposition of S, √S for the data ellipse and the
 * per-point PC1 projections behind the residual segments.
 */
export function useDataFacts(dataset: Dataset | null): DataFacts | null {
  return useMemo(() => {
    if (!dataset) return null;
    const pca = pcaOf(dataset.points);
    const sqrtS = matrixSqrt(pca.S);
    const proj = dataset.centered.map((p) => projectTo(p, pca.vectors, 1));
    return { pca, sqrtS, proj };
  }, [dataset]);
}

export interface BlockFacts {
  /** The four 2×2 blocks of M. */
  blocks: Blocks;
  /** Zero pattern of the off-diagonal blocks. */
  structure: BlockStructure;
  /** Schur complement S = D − C A⁻¹ B (null when A isn't invertible). */
  S: Matrix | null;
  /** Block LU factors of the elimination: M = L·U. */
  L: Matrix | null;
  U: Matrix | null;
  detM: number;
  detA: number;
  detD: number;
  detS: number | null;
  /** tr M versus tr A + tr D (equal by the trace rule). */
  traceInfo: { total: number; parts: number };
  /** λ(A) ∪ λ(D) when the block pattern lets the characteristic polynomial factor. */
  union: { fromA: EigenEntry[]; fromD: EigenEntry[] } | null;
  /** True when M is symmetric (blocks satisfy B = Cᵀ). */
  sym: boolean;
}

/**
 * Everything the Partitioned matrix card shows for a 4×4 — memoized once
 * per matrix like `useDecompFacts`. Returns null in 2D/3D.
 */
export function useBlockFacts(M: Matrix): BlockFacts | null {
  return useMemo(() => {
    if (M.length !== 4) return null;
    const blocks = splitBlocks(M);
    const S = schur(M);
    return {
      blocks,
      structure: blockStructure(M),
      S,
      L: blockLowerUnit(M),
      U: blockUpper(M),
      detM: det(M),
      detA: det(blocks.A),
      detD: det(blocks.D),
      detS: S ? det(S) : null,
      traceInfo: traceBlocks(M),
      union: eigenUnion(M),
      sym: isSymmetric(M),
    };
  }, [M]);
}
