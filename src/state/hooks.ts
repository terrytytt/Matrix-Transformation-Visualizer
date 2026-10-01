import { useMemo } from 'react';
import { Matrix, identity, inverse, matMul } from '../math/matrix';
import { useApp } from './store';

export interface ActiveMatrix {
  matrix: Matrix;
  /** Symbolic label for the readout: "A", "A⁻¹", "I", "B", "A·B". */
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

  return useMemo(() => {
    if (compActive) {
      if (compStep === 0) return { matrix: identity(dim), label: 'I', singular: false };
      if (compStep === 1) return { matrix: B, label: 'B', singular: false };
      return { matrix: matMul(A, B), label: 'A·B', singular: false };
    }
    if (showInverse) {
      const inv = inverse(A);
      return inv
        ? { matrix: inv, label: 'A⁻¹', singular: false }
        : { matrix: A, label: 'A', singular: true };
    }
    return { matrix: A, label: 'A', singular: false };
  }, [A, B, dim, showInverse, compActive, compStep]);
}
