import { Matrix, Vector, identity } from './matrix';

export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

export function lerpMatrix(A: Matrix, B: Matrix, t: number): Matrix {
  return A.map((row, i) => row.map((x, j) => lerp(x, B[i][j], t)));
}

export function lerpVector(a: Vector, b: Vector, t: number): Vector {
  return a.map((x, i) => lerp(x, b[i] ?? x, t));
}

/** Smooth ease so transitions never look mechanical. */
export function easeInOutCubic(t: number): number {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

/**
 * Animates a matrix from its previous value to a new target.
 * Renderers call `current()` every frame; `set()` whenever the target changes.
 */
export class MatrixTween {
  private from: Matrix;
  private to: Matrix;
  private start = 0;
  private readonly duration: number;
  private signature: string;

  constructor(initial: Matrix, duration = 550) {
    this.from = identity(initial.length);
    this.to = initial.map((r) => r.slice());
    this.duration = duration;
    this.signature = sig(this.to);
    this.start = performance.now() - duration; // start settled
  }

  /** Feed the desired matrix in; animation restarts only if it changed. */
  set(target: Matrix): void {
    const next = sig(target);
    if (next === this.signature) return;
    this.from = this.current();
    this.to = target.map((r) => r.slice());
    this.signature = next;
    this.start = performance.now();
  }

  /** True while an interpolation is still in flight (drives rAF). */
  get animating(): boolean {
    return performance.now() - this.start < this.duration;
  }

  current(now = performance.now()): Matrix {
    const raw = (now - this.start) / this.duration;
    if (raw >= 1) return this.to.map((r) => r.slice());
    if (raw <= 0) return this.from.map((r) => r.slice());
    return lerpMatrix(this.from, this.to, easeInOutCubic(raw));
  }
}

function sig(M: Matrix): string {
  return M.map((r) => r.map((x) => x.toFixed(9)).join(',')).join(';');
}
