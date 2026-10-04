/**
 * The map as equations of a *generic* x — the symbolic twin of the numeric
 * readouts.
 *
 *   y = A x        y₁ = 2x₁ + 3x₂
 *   x = (x₁, x₂)   y₂ =  x₁ − 0.5x₂
 *
 * During the composition demo the same machinery prints the substitution
 * chain: p = Bx first, then y = Ap, so the entries of the product A·B are
 * seen to *emerge* from plugging one linear equation into the other.
 *
 * Everything here is pure formatting — no state, no canvas — so tests can
 * pin the exact strings.
 */

import { Matrix } from './matrix';

/** Subscript digits for x₁ … x₄ / y₁ … y₄. */
const SUB = ['₁', '₂', '₃', '₄'];

/**
 * The 4×4 block view draws two 2-D planes, so it reads the input's four
 * components as coordinates — plane 1 = (x, y), plane 2 = (u, v) — the
 * same names its panel titles use.
 */
export const BLOCK_VARS = ['x', 'y', 'u', 'v'];

/** Coefficient formatter: 4 decimals, trailing zeros trimmed, −0 → 0. */
const cfmt = (x: number): string => {
  if (!Number.isFinite(x)) return 'NaN';
  const r = Math.round(x * 1e4) / 1e4;
  return Object.is(r, -0) ? '0' : String(r);
};

/**
 * One row of a linear system as an equation in the generic variables
 * `yᵢ = aᵢ₁x₁ + aᵢ₂x₂ + …`:
 *
 *  - terms whose coefficient rounds to 0 are dropped;
 *  - leading terms carry an explicit sign (`− 3x₂`), the first does not;
 *  - a coefficient of ±1 is elided (`x₁`, not `1x₁`);
 *  - an all-zero row collapses to `y₁ = 0`.
 *
 * `labels` swaps the variable names (e.g. the composition chain reads in
 * `p₁`, `p₂`).
 */
export function rowEquation(
  row: number[],
  rowIndex: number,
  labels?: { out?: string; vars?: string[] },
): string {
  const out = `${labels?.out ?? 'y'}${SUB[rowIndex]}`;
  const vars = labels?.vars ?? row.map((_, j) => `x${SUB[j]}`);
  let body = '';
  row.forEach((a, j) => {
    const v = Math.round(a * 1e4) / 1e4;
    if (v === 0) return;
    const sign = v < 0 ? '−' : '+';
    const mag = Math.abs(v);
    const coef = mag === 1 ? '' : cfmt(mag);
    const term = `${coef}${vars[j] ?? `x${SUB[j]}`}`;
    if (body === '') body = v < 0 ? `− ${term}` : term;
    else body += ` ${sign} ${term}`;
  });
  return `${out} = ${body === '' ? '0' : body}`;
}

/** Every row of `M` as an equation — the whole map y = M x. */
export function systemEquations(M: Matrix, labels?: { out?: string; vars?: string[] }): string[] {
  return M.map((row, i) => rowEquation(row, i, labels));
}

export interface ChainEquations {
  /** `p = B x` — the first hop, read straight off B's rows. */
  first: string[];
  /** `y = A p` — the second hop, in terms of the intermediate p. */
  second: string[];
  /** The same second hop after substituting p = Bx: coefficients of (A·B). */
  composed: string[];
  /** Both readings on one line: `y₁ = 2p₁ + p₂ = 2x₁`. */
  substituted: string[];
}

/**
 * The substitution chain behind matrix multiplication:
 *
 *   p = B x  :  p₁ = b₁₁x₁ + b₁₁x₂
 *   y = A p  :  y₁ = a₁₁p₁ + a₁₂p₂
 *             = (A·B)₁₁x₁ + (A·B)₁₂x₂
 *
 * `first`/`second` are the literal readings of each factor's rows;
 * `composed` is the expanded form, whose coefficients are exactly the
 * entries of matMul(A, B) — the identity the tests pin.
 *
 * `vars` swaps the input variable names (the 4×4 block view reads them
 * x, y, u, v); the default is x₁ … xₙ.
 */
export function chainEquations(A: Matrix, B: Matrix, vars?: string[]): ChainEquations {
  const n = A.length;
  const first = B.map((row, i) => rowEquation(row, i, { out: 'p', vars }));
  // yᵢ = Σⱼ aᵢⱼ pⱼ — written with explicit p terms even when aᵢⱼ is 0 is
  // noise, so reuse rowEquation against the p variables.
  const second = A.map((row, i) =>
    rowEquation(row, i, { out: 'y', vars: row.map((_, j) => `p${SUB[j]}`) }),
  );
  // Expanded coefficients: row i of A·B, printed against x again.
  const composed: string[] = [];
  for (let i = 0; i < n; i++) {
    const terms: number[] = [];
    for (let j = 0; j < n; j++) {
      let acc = 0;
      for (let k = 0; k < n; k++) acc += A[i][k] * B[k][j];
      terms.push(acc);
    }
    composed.push(rowEquation(terms, i, { vars }));
  }
  // One line per row carrying both readings: y₁ = a₁₁p₁ + a₁₂p₂ = (A·B)₁₁x₁ + …
  const substituted = second.map((s, i) => {
    const rhs = composed[i].slice(composed[i].indexOf('=') + 1).trim();
    return `${s} = ${rhs}`;
  });
  return { first, second, composed, substituted };
}
