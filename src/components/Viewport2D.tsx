import { useEffect, useMemo, useRef } from 'react';
import { useApp } from '../state/store';
import { useActiveMatrix } from '../state/hooks';
import {
  Matrix,
  Vector,
  columnSpaceBasis,
  det,
  inverse,
  matVec,
  normalize,
  nullSpaceBasis,
  rank,
} from '../math/matrix';
import { MatrixTween } from '../math/lerp';
import { EigenEntry, eigen, realEigenpairs } from '../math/eigen';

export interface View {
  /** pixels per world unit */
  scale: number;
  /** screen x of world x = 0 */
  ox: number;
  /** screen y of world y = 0 */
  oy: number;
}

const COLORS = {
  grid: 'rgba(148, 163, 184, 0.24)',
  axis: 'rgba(148, 163, 184, 0.5)',
  gridImage: 'rgba(96, 165, 250, 0.40)',
  e1: '#f87171',
  e2: '#4ade80',
  vec: '#fbbf24',
  detPos: 'rgba(34, 197, 94, 0.30)',
  detNeg: 'rgba(239, 68, 68, 0.32)',
  detFlat: 'rgba(148, 163, 184, 0.22)',
  eigen: ['#c084fc', '#f472b6', '#818cf8'],
  nullSpace: '#fb923c',
  colSpace: '#22d3ee',
  transpose: 'rgba(250, 204, 21, 0.34)',
  text: '#e2e8f0',
};

const MAX_LINES = 80;

export function Viewport2D() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);

  const targetMatrix = useActiveMatrix();
  const A = useApp((s) => s.matrix);
  const vector = useApp((s) => s.vector);
  const layers = useApp((s) => s.layers);
  const setVector = useApp((s) => s.setVector);

  const view = useRef<View>({ scale: 44, ox: 0, oy: 0 });
  const tween = useRef<MatrixTween | null>(null);
  const drag = useRef<{ mode: 'none' | 'pan' | 'vector'; x: number; y: number }>({
    mode: 'none',
    x: 0,
    y: 0,
  });
  const latest = useRef({ M: targetMatrix, vector, layers, A });
  latest.current = { M: targetMatrix, vector, layers, A };

  // Lazily initialised on the first frame so the tween starts settled.
  const ensureTween = () => (tween.current ??= new MatrixTween(targetMatrix.matrix));

  const derived = useMemo(
    () => ({
      invA: inverse(A),
      eigA: eigen(A),
      rankA: rank(A),
      colA: columnSpaceBasis(A),
      nullA: nullSpaceBasis(A),
      detA: det(A),
    }),
    [A],
  );
  const derivedRef = useRef(derived);
  derivedRef.current = derived;

  /* ---------------------------------------------------------------- */
  /* Interaction                                                       */
  /* ---------------------------------------------------------------- */

  useEffect(() => {
    const canvas = canvasRef.current!;
    const wrap = wrapRef.current!;

    const resize = () => {
      const rect = wrap.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.max(1, Math.round(rect.width * dpr));
      canvas.height = Math.max(1, Math.round(rect.height * dpr));
      canvas.style.width = `${rect.width}px`;
      canvas.style.height = `${rect.height}px`;
      if (view.current.ox === 0 && view.current.oy === 0) {
        view.current.ox = rect.width / 2;
        view.current.oy = rect.height / 2;
      }
    };
    resize();

    const ro = new ResizeObserver(resize);
    ro.observe(wrap);

    const toWorld = (cx: number, cy: number): Vector => {
      const rect = canvas.getBoundingClientRect();
      const sx = cx - rect.left;
      const sy = cy - rect.top;
      const { scale, ox, oy } = view.current;
      return [(sx - ox) / scale, (oy - sy) / scale];
    };

    const onPointerDown = (e: PointerEvent) => {
      const w = toWorld(e.clientX, e.clientY);
      const tip = latest.current.vector;
      const { scale, ox, oy } = view.current;
      const px = ox + tip[0] * scale;
      const py = oy - tip[1] * scale;
      const rect = canvas.getBoundingClientRect();
      const dist = Math.hypot(e.clientX - rect.left - px, e.clientY - rect.top - py);

      drag.current =
        dist < 14 && latest.current.layers.vector
          ? { mode: 'vector', x: e.clientX, y: e.clientY }
          : { mode: 'pan', x: e.clientX, y: e.clientY };
      canvas.setPointerCapture(e.pointerId);
      canvas.style.cursor = drag.current.mode === 'vector' ? 'grabbing' : 'move';
      e.preventDefault();
    };

    const onPointerMove = (e: PointerEvent) => {
      const d = drag.current;
      if (d.mode === 'none') {
        // hover feedback
        const tip = latest.current.vector;
        const { scale, ox, oy } = view.current;
        const rect = canvas.getBoundingClientRect();
        const px = ox + tip[0] * scale;
        const py = oy - tip[1] * scale;
        const near = Math.hypot(e.clientX - rect.left - px, e.clientY - rect.top - py) < 14;
        canvas.style.cursor = near ? 'grab' : 'default';
        return;
      }

      if (d.mode === 'pan') {
        view.current.ox += e.clientX - d.x;
        view.current.oy += e.clientY - d.y;
        d.x = e.clientX;
        d.y = e.clientY;
        return;
      }

      const w = toWorld(e.clientX, e.clientY);
      setVector([round2(w[0]), round2(w[1])]);
    };

    const onPointerUp = (e: PointerEvent) => {
      if (drag.current.mode !== 'none') {
        canvas.releasePointerCapture?.(e.pointerId);
      }
      drag.current.mode = 'none';
      canvas.style.cursor = 'default';
    };

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const sx = e.clientX - rect.left;
      const sy = e.clientY - rect.top;
      const { scale, ox, oy } = view.current;
      const wx = (sx - ox) / scale;
      const wy = (oy - sy) / scale;
      const factor = Math.exp(-e.deltaY * 0.0015);
      const next = clamp(scale * factor, 8, 400);
      view.current.scale = next;
      view.current.ox = sx - wx * next;
      view.current.oy = sy + wy * next;
    };

    canvas.addEventListener('pointerdown', onPointerDown);
    canvas.addEventListener('pointermove', onPointerMove);
    canvas.addEventListener('pointerup', onPointerUp);
    canvas.addEventListener('pointercancel', onPointerUp);
    canvas.addEventListener('wheel', onWheel, { passive: false });

    return () => {
      ro.disconnect();
      canvas.removeEventListener('pointerdown', onPointerDown);
      canvas.removeEventListener('pointermove', onPointerMove);
      canvas.removeEventListener('pointerup', onPointerUp);
      canvas.removeEventListener('pointercancel', onPointerUp);
      canvas.removeEventListener('wheel', onWheel);
    };
  }, [setVector]);

  /* ---------------------------------------------------------------- */
  /* Render loop                                                       */
  /* ---------------------------------------------------------------- */

  useEffect(() => {
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext('2d')!;
    let raf = 0;

    const frame = () => {
      raf = requestAnimationFrame(frame);
      const dpr = canvas.width / Math.max(1, canvas.clientWidth);
      const W = canvas.clientWidth;
      const H = canvas.clientHeight;

      const now = latest.current;
      const ly = now.layers;
      const tw = ensureTween();
      tw.set(now.M.matrix);
      const M = tw.current();

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);

      const v = view.current;
      const toScreen = (p: Vector): [number, number] => [v.ox + p[0] * v.scale, v.oy - p[1] * v.scale];

      paintBackground(ctx, W, H);
      drawOriginalGrid(ctx, W, H, v, toScreen);
      if (ly.determinant) drawDeterminantRegion(ctx, M, v, toScreen);
      drawImageGrid(ctx, W, H, v, M, toScreen);

      if (ly.transpose) {
        drawImageGrid(ctx, W, H, v, transposeOf(M), toScreen, COLORS.transpose, [5, 4]);
      }
      if (ly.columnSpace) drawColumnSpace(ctx, derivedRef.current, v, toScreen);
      if (ly.nullSpace) drawNullSpace(ctx, derivedRef.current, v, toScreen);
      if (ly.basis) drawBasis(ctx, M, v, toScreen);
      if (ly.vector) drawVectorPair(ctx, M, now.vector, v, toScreen);
      if (ly.eigen) drawEigenvectors(ctx, derivedRef.current.eigA, v, toScreen);
    };

    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  }, []);

  return (
    <div className="viewport viewport-2d" ref={wrapRef}>
      <canvas ref={canvasRef} aria-label="2D transformation visualisation" />
      <Hint />
    </div>
  );
}

/* ================================================================== */
/* Drawing helpers                                                     */
/* ================================================================== */

function transposeOf(M: Matrix): Matrix {
  const n = M.length;
  return Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => M[j][i]));
}

function paintBackground(ctx: CanvasRenderingContext2D, W: number, H: number) {
  const g = ctx.createLinearGradient(0, 0, 0, H);
  g.addColorStop(0, '#0b1020');
  g.addColorStop(1, '#080b14');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, W, H);
}

/** Integer lattice covering the visible world rectangle. */
function worldBounds(W: number, H: number, v: View) {
  const x0 = (0 - v.ox) / v.scale;
  const x1 = (W - v.ox) / v.scale;
  const y0 = (v.oy - H) / v.scale;
  const y1 = v.oy / v.scale;
  return { x0, x1, y0, y1 };
}

function linesFor(b: { x0: number; x1: number; y0: number; y1: number }) {
  const lo = (a: number) => Math.max(Math.floor(a), -MAX_LINES);
  const hi = (a: number) => Math.min(Math.ceil(a), MAX_LINES);
  return { i0: lo(b.x0), i1: hi(b.x1), j0: lo(b.y0), j1: hi(b.y1) };
}

function drawOriginalGrid(
  ctx: CanvasRenderingContext2D,
  W: number,
  H: number,
  v: View,
  toScreen: (p: Vector) => [number, number],
) {
  const b = worldBounds(W, H, v);
  const { i0, i1, j0, j1 } = linesFor(b);

  ctx.lineWidth = 1;
  ctx.strokeStyle = COLORS.grid;
  ctx.beginPath();
  for (let i = i0; i <= i1; i++) {
    const [ax, ay] = toScreen([i, b.y0]);
    const [bx, by] = toScreen([i, b.y1]);
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
  }
  for (let j = j0; j <= j1; j++) {
    const [ax, ay] = toScreen([b.x0, j]);
    const [bx, by] = toScreen([b.x1, j]);
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
  }
  ctx.stroke();

  // axes
  ctx.strokeStyle = COLORS.axis;
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  const [ox, oy] = toScreen([0, 0]);
  ctx.moveTo(0, oy);
  ctx.lineTo(W, oy);
  ctx.moveTo(ox, 0);
  ctx.lineTo(ox, H);
  ctx.stroke();
}

/**
 * The image of the lattice under `M`. Because `M` is linear a straight grid
 * line stays straight, so two endpoints per line are enough.
 *
 * The source rectangle is chosen as `A⁻¹(view)` so the deformed lattice
 * still fills the screen even when A shrinks space.
 */
function drawImageGrid(
  ctx: CanvasRenderingContext2D,
  W: number,
  H: number,
  v: View,
  M: Matrix,
  toScreen: (p: Vector) => [number, number],
  color = COLORS.gridImage,
  dash: number[] = [],
) {
  const view = worldBounds(W, H, v);
  const src = sourceBounds(view, M);

  const { i0, i1, j0, j1 } = linesFor(src);
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.2;
  if (dash.length) ctx.setLineDash(dash);

  ctx.beginPath();
  for (let i = i0; i <= i1; i++) {
    const a = toScreen(matVec(M, [i, src.y0]));
    const b = toScreen(matVec(M, [i, src.y1]));
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
  }
  for (let j = j0; j <= j1; j++) {
    const a = toScreen(matVec(M, [src.x0, j]));
    const b = toScreen(matVec(M, [src.x1, j]));
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
  }
  ctx.stroke();
  ctx.restore();
}

function sourceBounds(view: { x0: number; x1: number; y0: number; y1: number }, M: Matrix) {
  const base = { ...view };
  const inv = inverse(M);
  if (!inv) return pad(base, 2);

  const corners: Vector[] = [
    [view.x0, view.y0],
    [view.x1, view.y0],
    [view.x0, view.y1],
    [view.x1, view.y1],
  ].map((c) => matVec(inv, c));

  const xs = corners.map((c) => c[0]);
  const ys = corners.map((c) => c[1]);
  const ext = {
    x0: Math.min(...xs, view.x0),
    x1: Math.max(...xs, view.x1),
    y0: Math.min(...ys, view.y0),
    y1: Math.max(...ys, view.y1),
  };
  return pad(ext, 1);
}

function pad(b: { x0: number; x1: number; y0: number; y1: number }, m: number) {
  return { x0: b.x0 - m, x1: b.x1 + m, y0: b.y0 - m, y1: b.y1 + m };
}

function drawDeterminantRegion(
  ctx: CanvasRenderingContext2D,
  M: Matrix,
  v: View,
  toScreen: (p: Vector) => [number, number],
) {
  const d = det(M);
  const corners: Vector[] = [
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1],
  ].map((c) => matVec(M, c));

  ctx.beginPath();
  corners.forEach((c, i) => {
    const [x, y] = toScreen(c);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.closePath();

  ctx.fillStyle = Math.abs(d) < 1e-9 ? COLORS.detFlat : d > 0 ? COLORS.detPos : COLORS.detNeg;
  ctx.fill();
  ctx.strokeStyle =
    Math.abs(d) < 1e-9 ? 'rgba(148,163,184,0.5)' : d > 0 ? 'rgba(74,222,128,0.75)' : 'rgba(248,113,113,0.75)';
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // Anchor on the far corner A·(1,1): the centroid sits between Ae₁ and Ae₂,
  // which is exactly where the basis labels already live.
  const anchor = matVec(M, [1, 1]);
  const [lx, ly] = toScreen(anchor);
  drawLabel(ctx, lx, ly, `det = ${fmtNum(d)}`, d >= 0 ? '#86efac' : '#fca5a5', 'center', 0, 14);
}

function drawBasis(
  ctx: CanvasRenderingContext2D,
  M: Matrix,
  v: View,
  toScreen: (p: Vector) => [number, number],
) {
  const e: Vector[] = [
    [1, 0],
    [0, 1],
  ];
  const colors = [COLORS.e1, COLORS.e2];

  e.forEach((unit, k) => {
    const image = matVec(M, unit);
    const mid: Vector = [unit[0] * 0.62, unit[1] * 0.62];
    drawArrow(ctx, toScreen([0, 0]), toScreen(unit), colors[k], 1.6, [4, 4], 0.75);
    drawLabel(ctx, ...toScreen(mid), `e${sub(k + 1)}`, colors[k], 'center', 0, 14, 0.95);
    drawArrow(ctx, toScreen([0, 0]), toScreen(image), colors[k], 2.6, [], 1);
    drawLabel(ctx, ...toScreen(image), `Ae${sub(k + 1)}`, colors[k], 'left', 8, -11);
  });
}

function drawVectorPair(
  ctx: CanvasRenderingContext2D,
  M: Matrix,
  x: Vector,
  v: View,
  toScreen: (p: Vector) => [number, number],
) {
  const image = matVec(M, x);
  drawArrow(ctx, toScreen([0, 0]), toScreen(x), COLORS.vec, 1.8, [5, 4], 0.85);
  drawLabel(ctx, ...toScreen(x), 'x', COLORS.vec, 'left', 11, 12);
  drawArrow(ctx, toScreen([0, 0]), toScreen(image), COLORS.vec, 3, [], 1);
  drawLabel(ctx, ...toScreen(image), 'Ax', COLORS.vec, 'left', 8, -11);

  // draggable handle
  const [hx, hy] = toScreen(x);
  ctx.beginPath();
  ctx.arc(hx, hy, 5, 0, Math.PI * 2);
  ctx.fillStyle = COLORS.vec;
  ctx.fill();
  ctx.strokeStyle = '#0b1020';
  ctx.lineWidth = 2;
  ctx.stroke();
}

function drawEigenvectors(
  ctx: CanvasRenderingContext2D,
  entries: EigenEntry[],
  v: View,
  toScreen: (p: Vector) => [number, number],
) {
  const pairs = realEigenpairs(entries);
  pairs.forEach((pair, k) => {
    const color = COLORS.eigen[k % COLORS.eigen.length];
    const lam = pair.real;
    const scale = 1.5;

    pair.vectors.slice(0, 2).forEach((vec, vi) => {
      const dir = normalize(vec);

      // the invariant line that the eigenvector lives on
      ctx.save();
      ctx.setLineDash([2, 6]);
      ctx.strokeStyle = withAlpha(color, 0.4);
      ctx.lineWidth = 1;
      ctx.beginPath();
      const p1 = toScreen([-dir[0] * 40, -dir[1] * 40]);
      const p2 = toScreen([dir[0] * 40, dir[1] * 40]);
      ctx.moveTo(p1[0], p1[1]);
      ctx.lineTo(p2[0], p2[1]);
      ctx.stroke();
      ctx.restore();

      const base: Vector = [dir[0] * scale, dir[1] * scale];
      const mid: Vector = [dir[0] * scale * 0.55, dir[1] * scale * 0.55];
      drawArrow(ctx, toScreen([0, 0]), toScreen(base), color, 1.6, [3, 3], 0.8);
      drawLabel(ctx, ...toScreen(mid), vi === 0 ? 'v' : `v${sub(vi + 1)}`, color, 'center', 0, -13, 0.95);

      if (Math.abs(lam) > 1e-9) {
        const image: Vector = [base[0] * lam, base[1] * lam];
        drawArrow(ctx, toScreen([0, 0]), toScreen(image), color, 3, [], 1);
        drawLabel(
          ctx,
          ...toScreen(image),
          vi === 0 ? `λv ·  λ = ${fmtNum(lam)}` : `λ = ${fmtNum(lam)}`,
          color,
          'left',
          8,
          -12,
        );
      } else {
        drawLabel(ctx, ...toScreen(base), 'λ = 0 → origin', color, 'left', 7, 16);
      }
    });
  });
}

function drawColumnSpace(
  ctx: CanvasRenderingContext2D,
  d: { colA: Vector[]; rankA: number },
  v: View,
  toScreen: (p: Vector) => [number, number],
) {
  ctx.save();
  ctx.strokeStyle = withAlpha(COLORS.colSpace, 0.85);
  ctx.fillStyle = COLORS.colSpace;

  d.colA.forEach((col, k) => {
    drawArrow(ctx, toScreen([0, 0]), toScreen(col), COLORS.colSpace, 2, [], 0.8);
    drawLabel(ctx, ...toScreen(col), `col${k + 1}`, COLORS.colSpace, 'left', 7, 13);
  });

  if (d.rankA < 2 && d.colA.length > 0) {
    const dir = normalize(d.colA[0]);
    ctx.setLineDash([]);
    ctx.lineWidth = 5;
    ctx.strokeStyle = withAlpha(COLORS.colSpace, 0.35);
    ctx.beginPath();
    const a = toScreen([-dir[0] * 60, -dir[1] * 60]);
    const b = toScreen([dir[0] * 60, dir[1] * 60]);
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
    ctx.stroke();
  }
  ctx.restore();
}

function drawNullSpace(
  ctx: CanvasRenderingContext2D,
  d: { nullA: Vector[]; rankA: number },
  v: View,
  toScreen: (p: Vector) => [number, number],
) {
  ctx.save();
  if (d.nullA.length === 0) {
    const [ox, oy] = toScreen([0, 0]);
    ctx.beginPath();
    ctx.arc(ox, oy, 5, 0, Math.PI * 2);
    ctx.fillStyle = COLORS.nullSpace;
    ctx.fill();
    drawLabel(ctx, ox, oy, 'null(A) = {0}', COLORS.nullSpace, 'right', -12, 32);
  } else {
    const dir = normalize(d.nullA[0]);
    ctx.lineWidth = 5;
    ctx.strokeStyle = withAlpha(COLORS.nullSpace, 0.55);
    ctx.setLineDash([]);
    ctx.beginPath();
    const a = toScreen([-dir[0] * 80, -dir[1] * 80]);
    const b = toScreen([dir[0] * 80, dir[1] * 80]);
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
    ctx.stroke();
    drawLabel(ctx, ...toScreen([dir[0] * 3, dir[1] * 3]), 'null(A)', COLORS.nullSpace, 'left', 9, -11);
  }
  ctx.restore();
}

/* ================================================================== */
/* Primitives                                                          */
/* ================================================================== */

function drawArrow(
  ctx: CanvasRenderingContext2D,
  from: [number, number],
  to: [number, number],
  color: string,
  width: number,
  dash: number[],
  alpha = 1,
) {
  const dx = to[0] - from[0];
  const dy = to[1] - from[1];
  const len = Math.hypot(dx, dy);
  if (len < 0.5) return;

  const angle = Math.atan2(dy, dx);
  const head = Math.min(11, 4 + width * 2.2);
  const ux = Math.cos(angle);
  const uy = Math.sin(angle);
  const tipX = to[0] - ux * head * 0.75;
  const tipY = to[1] - uy * head * 0.75;

  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = width;
  ctx.lineCap = 'round';
  if (dash.length) ctx.setLineDash(dash);

  ctx.beginPath();
  ctx.moveTo(from[0], from[1]);
  ctx.lineTo(tipX, tipY);
  ctx.stroke();

  ctx.setLineDash([]);
  ctx.beginPath();
  ctx.moveTo(to[0], to[1]);
  ctx.lineTo(to[0] - ux * head - uy * head * 0.45, to[1] - uy * head + ux * head * 0.45);
  ctx.lineTo(to[0] - ux * head + uy * head * 0.45, to[1] - uy * head - ux * head * 0.45);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function drawLabel(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  text: string,
  color: string,
  align: CanvasTextAlign = 'left',
  dx = 6,
  dy = -6,
  alpha = 1,
  bg = true,
) {
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.font = '600 12px ui-monospace, SFMono-Regular, Menlo, monospace';
  ctx.textAlign = align;
  ctx.textBaseline = 'middle';
  const tx = x + dx;
  const ty = y + dy;

  if (bg) {
    const w = ctx.measureText(text).width;
    const padX = 5;
    const padY = 4;
    const half = 7; // half the line height used for the middle baseline
    let x0 = tx;
    if (align === 'center') x0 = tx - w / 2;
    else if (align === 'right') x0 = tx - w;
    x0 -= padX;
    const y0 = ty - half - padY;
    const bw = w + padX * 2;
    const bh = half * 2 + padY * 2;

    ctx.beginPath();
    if (typeof (ctx as CanvasRenderingContext2D & { roundRect?: unknown }).roundRect === 'function') {
      (ctx as CanvasRenderingContext2D & { roundRect: (a: number, b: number, c: number, d: number, e: number) => void }).roundRect(
        x0,
        y0,
        bw,
        bh,
        5,
      );
    } else {
      ctx.rect(x0, y0, bw, bh);
    }
    ctx.fillStyle = 'rgba(8, 11, 20, 0.82)';
    ctx.fill();
    ctx.strokeStyle = color;
    ctx.globalAlpha = alpha * 0.5;
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.globalAlpha = alpha;
  }

  ctx.lineWidth = 3;
  ctx.strokeStyle = 'rgba(8, 11, 20, 0.9)';
  ctx.strokeText(text, tx, ty);
  ctx.fillStyle = color;
  ctx.fillText(text, tx, ty);
  ctx.restore();
}

function Hint() {
  return (
    <div className="viewport-hint">
      drag the amber tip to move <b>x</b> · scroll to zoom · drag background to pan
    </div>
  );
}

/* ================================================================== */
/* Utilities                                                           */
/* ================================================================== */

const clamp = (x: number, a: number, b: number) => Math.min(b, Math.max(a, x));
const round2 = (x: number) => Math.round(x * 100) / 100;

function sub(n: number) {
  return ['', '₁', '₂', '₃'][n] ?? String(n);
}

function fmtNum(x: number): string {
  if (!Number.isFinite(x)) return '—';
  const r = Math.round(x * 1e4) / 1e4;
  return String(r);
}

function withAlpha(hex: string, a: number): string {
  const h = hex.replace('#', '');
  const r = parseInt(h.slice(0, 2), 16);
  const g = parseInt(h.slice(2, 4), 16);
  const b = parseInt(h.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${a})`;
}
