import { useApp } from '../state/store';
import { useActiveMatrix } from '../state/hooks';
import { identity } from '../math/matrix';
import { BLOCK_VARS, chainEquations, systemEquations } from '../math/equations';

/** One titled group of equations (`p = B x` plus its rows). */
export interface EqBlock {
  title: string;
  rows: string[];
}

/**
 * The equations behind whatever is currently drawn, as blocks of
 * `yᵢ = aᵢ₁x₁ + aᵢ₂x₂` rows:
 *
 *  - plain mode: the active matrix itself — `y = A x`, `y = A⁻¹ x`, the
 *    current decomposition step, … (equations follow the drawn map);
 *  - composition demo: the substitution chain, one block per hop, so the
 *    entries of A·B are seen to emerge by plugging p = B x into y = A p.
 *
 * Returns `null` when equation mode is off. The 4×4 block view reads the
 * same rows against its component names x, y, u, v (BLOCK_VARS) and its
 * factor names N (first) then M (second) — the InfoPanel card takes them,
 * but the block viewport gets no floating overlay.
 */
export function useEquationBlocks(): EqBlock[] | null {
  const eqMode = useApp((s) => s.eqMode);
  const dim = useApp((s) => s.dim);
  const compActive = useApp((s) => s.compActive);
  const compStep = useApp((s) => s.compStep);
  const A = useApp((s) => s.matrix);
  const B = useApp((s) => s.matrixB);
  const active = useActiveMatrix();

  if (!eqMode) return null;
  const vars = dim === 4 ? BLOCK_VARS : undefined;
  const nm = dim === 4 ? 'M' : 'A';
  const nm2 = dim === 4 ? 'N' : 'B';
  if (compActive) {
    if (compStep === 0)
      return [{ title: 'y = x', rows: systemEquations(identity(dim), { vars }) }];
    if (compStep === 1) return [{ title: `y = ${nm2} x`, rows: systemEquations(B, { vars }) }];
    const chain = chainEquations(A, B, vars);
    return [
      { title: `p = ${nm2} x`, rows: chain.first },
      { title: `y = ${nm} p`, rows: chain.substituted },
    ];
  }
  return [{ title: `y = ${active.label} x`, rows: systemEquations(active.matrix, { vars }) }];
}

/**
 * Viewport overlay for equation mode: the same blocks the InfoPanel card
 * shows, floating over the canvas — top-left in overlay mode, over the
 * "after · live" half in split view. Purely informational: it never eats
 * pointer events, so the grid stays draggable underneath.
 */
export function EquationOverlay() {
  const blocks = useEquationBlocks();
  const splitView = useApp((s) => s.splitView);
  const dim = useApp((s) => s.dim);
  // The 4×4 block view keeps its equations in the panel card only.
  if (!blocks || dim === 4) return null;
  return (
    <div className={`eq-overlay${splitView ? ' eq-overlay-split' : ''}`}>
      {blocks.map((b) => (
        <div className="eq-block" key={b.title}>
          <div className="eq-title mono">{b.title}</div>
          <div className="eq-rows mono">{b.rows.join('\n')}</div>
        </div>
      ))}
    </div>
  );
}
