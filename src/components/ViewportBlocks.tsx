import { useEffect, useRef } from 'react';
import { useApp } from '../state/store';
import { useActiveMatrix } from '../state/hooks';
import { Matrix, Vector, det, identity, inverse, matVec } from '../math/matrix';
import { MatrixTween, blendFromIdentity } from '../math/lerp';
import { BLOCK_COLORS, Blocks, blockMagnitude, splitBlocks } from '../math/blocks';
import { drawArrow, drawLabel } from './Viewport2D';

/**
 * The 4×4 two-plane viewport: one canvas, four panels.
 *
 *     x₁ (plane 1) ──A──▶ y₁        x₁ ──C──▶ y₂
 *     x₂ (plane 2) ──B──▶ y₁        x₂ ──D──▶ y₂
 *
 * Left column = inputs, right column = outputs; the coloured lines in the
 * gutter are exactly the four blocks of M = [A B; C D]. Every arrow inside
 * an output panel is colored by the block that produced it, so the formula
 * y₁ = A x₁ + B x₂ is visible as two head-to-tail contributions.
 */

interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

interface BlocksView {
  scale: number;
  ox: number[];
  oy: number[];
}

const MARGIN = 14;
const GUT_X = 104;
const GUT_Y = 48;
const HEAD = 32;
const MAX_LINES = 60;
const PANEL_TITLES = [
  'x₁ = (x, y)  ·  input plane 1',
  'x₂ = (u, v)  ·  input plane 2',
  'y₁ = A x₁ + B x₂',
  'y₂ = C x₁ + D x₂',
];

const C = {
  grid: 'rgba(148, 163, 184, 0.24)',
  axis: 'rgba(148, 163, 184, 0.5)',
  panel: 'rgba(148, 163, 184, 0.18)',
  e1: '#f87171',
  e2: '#4ade80',
  vec: '#fbbf24',
  sum: '#f8fafc',
  head: '#64748b',
  text: '#e2e8f0',
};

/** '#rrggbb' + alpha → 'rgba(...)'. */
const alpha = (hex: string, a: number): string => {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${a})`;
};

const clamp = (x: number, a: number, b: number) => Math.min(b, Math.max(a, x));
const round2 = (x: number) => Math.round(x * 100) / 100;
const fmt = (x: number) => {
  if (!Number.isFinite(x)) return '—';
  const r = Math.round(x * 1e4) / 1e4;
  return Object.is(r, -0) ? '0' : String(r);
};

function computeLayout(W: number, H: number): Rect[] {
  const colW = Math.max(80, (W - MARGIN * 2 - GUT_X) / 2);
  const rowH = Math.max(60, (H - MARGIN * 2 - HEAD - GUT_Y) / 2);
  const y0 = MARGIN + HEAD;
  const x1 = MARGIN + colW + GUT_X;
  return [
    { x: MARGIN, y: y0, w: colW, h: rowH },
    { x: MARGIN, y: y0 + rowH + GUT_Y, w: colW, h: rowH },
    { x: x1, y: y0, w: colW, h: rowH },
    { x: x1, y: y0 + rowH + GUT_Y, w: colW, h: rowH },
  ];
}

export default function ViewportBlocks() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);

  const targetMatrix = useActiveMatrix();
  const vector = useApp((s) => s.vector);
  const layers = useApp((s) => s.layers);
  const transT = useApp((s) => s.transT);
  const ghostAlways = useApp((s) => s.ghostAlways);
  const setVector = useApp((s) => s.setVector);

  const view = useRef<BlocksView>({ scale: 40, ox: [], oy: [] });
  const tween = useRef<MatrixTween | null>(null);
  const rectsRef = useRef<Rect[]>([]);
  const drag = useRef<{ mode: 'none' | 'pan' | 'vector'; which: 0 | 1; x: number; y: number }>({
    mode: 'none',
    which: 0,
    x: 0,
    y: 0,
  });
  const latest = useRef({ M: targetMatrix, vector, layers, transT, ghostAlways });
  latest.current = { M: targetMatrix, vector, layers, transT, ghostAlways };

  const ensureTween = () => (tween.current ??= new MatrixTween(targetMatrix.matrix));

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
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(wrap);

    const local = (cx: number, cy: number): [number, number] => {
      const rect = canvas.getBoundingClientRect();
      return [cx - rect.left, cy - rect.top];
    };

    const panelAt = (sx: number, sy: number): number => {
      const rs = rectsRef.current;
      for (let i = 0; i < rs.length; i++) {
        const r = rs[i];
        if (sx >= r.x && sx <= r.x + r.w && sy >= r.y && sy <= r.y + r.h) return i;
      }
      return -1;
    };

    const tipScreen = (which: 0 | 1): [number, number] => {
      const { scale, ox, oy } = view.current;
      const v = latest.current.vector;
      const i = which;
      return [ox[i] + v[which * 2] * scale, oy[i] - v[which * 2 + 1] * scale];
    };

    const nearTip = (sx: number, sy: number): 0 | 1 | null => {
      if (!latest.current.layers.vector) return null;
      for (const which of [0, 1] as const) {
        const [px, py] = tipScreen(which);
        if (Math.hypot(sx - px, sy - py) < 14) return which;
      }
      return null;
    };

    const onPointerDown = (e: PointerEvent) => {
      const [sx, sy] = local(e.clientX, e.clientY);
      const hit = nearTip(sx, sy);
      drag.current =
        hit === null
          ? { mode: 'pan', which: 0, x: e.clientX, y: e.clientY }
          : { mode: 'vector', which: hit, x: e.clientX, y: e.clientY };
      canvas.setPointerCapture(e.pointerId);
      canvas.style.cursor = drag.current.mode === 'vector' ? 'grabbing' : 'move';
      e.preventDefault();
    };

    const onPointerMove = (e: PointerEvent) => {
      const d = drag.current;
      if (d.mode === 'none') {
        const [sx, sy] = local(e.clientX, e.clientY);
        canvas.style.cursor = nearTip(sx, sy) !== null ? 'grab' : 'default';
        return;
      }

      if (d.mode === 'pan') {
        const v = view.current;
        const dx = e.clientX - d.x;
        const dy = e.clientY - d.y;
        for (let i = 0; i < v.ox.length; i++) {
          v.ox[i] += dx;
          v.oy[i] += dy;
        }
        d.x = e.clientX;
        d.y = e.clientY;
        return;
      }

      const [sx, sy] = local(e.clientX, e.clientY);
      const { scale, ox, oy } = view.current;
      const i = d.which;
      const wx = (sx - ox[i]) / scale;
      const wy = (oy[i] - sy) / scale;
      const cur = latest.current.vector;
      const next: Vector =
        i === 0
          ? [round2(wx), round2(wy), cur[2], cur[3]]
          : [cur[0], cur[1], round2(wx), round2(wy)];
      setVector(next);
    };

    const onPointerUp = (e: PointerEvent) => {
      if (drag.current.mode !== 'none') canvas.releasePointerCapture?.(e.pointerId);
      drag.current.mode = 'none';
      canvas.style.cursor = 'default';
    };

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const [sx, sy] = local(e.clientX, e.clientY);
      const v = view.current;
      const factor = Math.exp(-e.deltaY * 0.0015);
      const next = clamp(v.scale * factor, 8, 400);
      const f = next / v.scale;
      v.scale = next;
      for (let i = 0; i < v.ox.length; i++) {
        v.ox[i] = sx + (v.ox[i] - sx) * f;
        v.oy[i] = sy + (v.oy[i] - sy) * f;
      }
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
      const dpr = canvas.width / Math.max(1, canvas.clientWidth);
      const W = canvas.clientWidth;
      const H = canvas.clientHeight;

      const rects = computeLayout(W, H);
      const v = view.current;
      if (v.ox.length !== 4) {
        v.ox = rects.map((r) => r.x + r.w / 2);
        v.oy = rects.map((r) => r.y + r.h / 2);
      } else {
        const prev = rectsRef.current;
        const changed =
          prev.length === 4 && (prev[0].w !== rects[0].w || prev[0].h !== rects[0].h);
        if (changed) {
          v.ox = rects.map((r, i) => r.x + r.w / 2 + (v.ox[i] - (prev[i].x + prev[i].w / 2)));
          v.oy = rects.map((r, i) => r.y + r.h / 2 + (v.oy[i] - (prev[i].y + prev[i].h / 2)));
        }
      }
      rectsRef.current = rects;

      const state = latest.current;
      const ly = state.layers;
      const tweenRef = ensureTween();
      tweenRef.set(state.M.matrix);
      const base = tweenRef.current();
      const t = state.transT;
      // Exposure blend: identity at t = 0 (plain space), full M at t = 1.
      const M = blendFromIdentity(base, t);
      const blocks = splitBlocks(M);
      // Ghost landmarks while the scrub runs (or "Show before" is pinned):
      // where each output sits under the identity, and where it's heading.
      const destBlocks = t < 1 ? splitBlocks(base) : null;
      const showGhost = state.ghostAlways || t < 1;
      const vec = state.vector;
      const x1: Vector = [vec[0] ?? 0, vec[1] ?? 0];
      const x2: Vector = [vec[2] ?? 0, vec[3] ?? 0];
      const y1: Vector = [
        blocks.A[0][0] * x1[0] + blocks.A[0][1] * x1[1] + blocks.B[0][0] * x2[0] + blocks.B[0][1] * x2[1],
        blocks.A[1][0] * x1[0] + blocks.A[1][1] * x1[1] + blocks.B[1][0] * x2[0] + blocks.B[1][1] * x2[1],
      ];
      const y2: Vector = [
        blocks.C[0][0] * x1[0] + blocks.C[0][1] * x1[1] + blocks.D[0][0] * x2[0] + blocks.D[0][1] * x2[1],
        blocks.C[1][0] * x1[0] + blocks.C[1][1] * x1[1] + blocks.D[1][0] * x2[0] + blocks.D[1][1] * x2[1],
      ];

      const dprScale = dpr;
      ctx.save();
      ctx.scale(dprScale, dprScale);

      // background
      const g = ctx.createLinearGradient(0, 0, 0, H);
      g.addColorStop(0, '#0b1020');
      g.addColorStop(1, '#080b14');
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, W, H);

      drawHeaders(ctx, rects);

      // ---------------- input panels ----------------
      for (const i of [0, 1] as const) {
        const r = rects[i];
        const toS = mkToScreen(v, i);
        ctx.save();
        ctx.beginPath();
        ctx.rect(r.x, r.y, r.w, r.h);
        ctx.clip();

        if (ly.grid) drawLattice(ctx, r, v, i, toS);
        drawAxesPanel(ctx, r, toS);

        if (ly.basis) {
          drawArrow(ctx, toS([0, 0]), toS([1, 0]), C.e1, 2, [], 0.9);
          drawArrow(ctx, toS([0, 0]), toS([0, 1]), C.e2, 2, [], 0.9);
          drawLabel(ctx, toS([1, 0])[0], toS([1, 0])[1], 'e₁', C.e1, 'left', 6, 12, 0.95);
          drawLabel(ctx, toS([0, 1])[0], toS([0, 1])[1], 'e₂', C.e2, 'left', 6, -8, 0.95);
        }

        if (ly.vector) {
          const p = i === 0 ? x1 : x2;
          const tip = toS(p);
          drawArrow(ctx, toS([0, 0]), tip, C.vec, 2.6, []);
          // grab handle
          ctx.beginPath();
          ctx.arc(tip[0], tip[1], 6, 0, Math.PI * 2);
          ctx.fillStyle = 'rgba(8, 11, 20, 0.9)';
          ctx.fill();
          ctx.lineWidth = 2.2;
          ctx.strokeStyle = C.vec;
          ctx.stroke();
          drawLabel(ctx, tip[0], tip[1], i === 0 ? 'x₁' : 'x₂', C.vec, 'left', 10, -12);
        }

        ctx.restore();
        drawPanelFrame(ctx, r);
        drawPanelTitle(ctx, r, PANEL_TITLES[i]);
      }

      // ---------------- output panels ----------------
      const out: [number, Matrix, Matrix, 'A' | 'B' | 'C' | 'D', 'A' | 'B' | 'C' | 'D', Vector, Vector][] = [
        [2, blocks.A, blocks.B, 'A', 'B', x1, x2],
        [3, blocks.C, blocks.D, 'C', 'D', x1, x2],
      ];

      for (const [i, K1, K2, l1, l2, p1, p2] of out) {
        const r = rects[i];
        const toS = mkToScreen(v, i);
        const target = i === 2 ? y1 : y2;
        const img1: Vector = matVec(K1, p1); // block1's contribution
        const img2: Vector = matVec(K2, p2); // block2's contribution

        ctx.save();
        ctx.beginPath();
        ctx.rect(r.x, r.y, r.w, r.h);
        ctx.clip();

        // Morph landmarks: the output's plain-space seat (identity — the
        // input panels already show it) and its destination (full M), both
        // under the live picture the scrub is drawing.
        if (showGhost && ly.grid) {
          drawImageLattice(ctx, r, v, i, identity(2), toS, 'rgba(148, 163, 184, 0.13)');
        }
        if (destBlocks && ly.grid) {
          const [d1, d2] = i === 2 ? [destBlocks.A, destBlocks.B] : [destBlocks.C, destBlocks.D];
          drawImageLattice(ctx, r, v, i, d1, toS, alpha(BLOCK_COLORS[l1], 0.17));
          drawImageLattice(ctx, r, v, i, d2, toS, alpha(BLOCK_COLORS[l2], 0.17));
        }

        if (ly.determinant) {
          drawSquareImage(ctx, K1, toS, BLOCK_COLORS[l1], l1);
          drawSquareImage(ctx, K2, toS, BLOCK_COLORS[l2], l2);
        }

        if (ly.grid) {
          drawImageLattice(ctx, r, v, i, K1, toS, alpha(BLOCK_COLORS[l1], 0.34));
          drawImageLattice(ctx, r, v, i, K2, toS, alpha(BLOCK_COLORS[l2], 0.34));
        }

        drawAxesPanel(ctx, r, toS);

        if (ly.basis) {
          // images of e₁, e₂ under each block
          const imgs: [Vector, Vector, string][] = [
            [matVec(K1, [1, 0]), matVec(K1, [0, 1]), l1],
            [matVec(K2, [1, 0]), matVec(K2, [0, 1]), l2],
          ];
          imgs.forEach(([a1, a2, letter], idx) => {
            const col = BLOCK_COLORS[letter as 'A'];
            const dash = idx === 1 ? [5, 4] : [];
            drawArrow(ctx, toS([0, 0]), toS(a1), col, 2, dash, 0.95);
            drawArrow(ctx, toS([0, 0]), toS(a2), col, 2, dash, 0.95);
            drawLabel(ctx, toS(a1)[0], toS(a1)[1], `${letter}e₁`, col, 'left', 6, 10, 0.95);
            drawLabel(ctx, toS(a2)[0], toS(a2)[1], `${letter}e₂`, col, 'left', 6, -8, 0.95);
          });
        }

        if (ly.vector) {
          // head-to-tail: origin → block1 contribution → + block2 contribution = y
          drawArrow(ctx, toS([0, 0]), toS(img1), BLOCK_COLORS[l1], 2.6, []);
          drawArrow(ctx, toS(img1), toS([img1[0] + img2[0], img1[1] + img2[1]]), BLOCK_COLORS[l2], 2.6, []);
          drawArrow(ctx, toS([0, 0]), toS(target), C.sum, 3.2, [], 0.95);
          const tip = toS(target);
          ctx.beginPath();
          ctx.arc(tip[0], tip[1], 5, 0, Math.PI * 2);
          ctx.fillStyle = C.sum;
          ctx.fill();
          const mid1 = toS([img1[0] / 2, img1[1] / 2]);
          drawLabel(ctx, mid1[0], mid1[1], `${l1} ${labelOf(l1)}`, BLOCK_COLORS[l1], 'center', 0, -12, 0.95);
          const mid2 = toS([img1[0] + img2[0] / 2, img1[1] + img2[1] / 2]);
          drawLabel(ctx, mid2[0], mid2[1], `${l2} ${labelOf(l2)}`, BLOCK_COLORS[l2], 'center', 0, 14, 0.95);
          drawLabel(ctx, tip[0], tip[1], i === 2 ? 'y₁' : 'y₂', C.sum, 'left', 12, -14);
        }

        // Straight-line travel so far: each contribution starts at its input
        // (what the output would be if this block did nothing) and streaks
        // toward its live tip while the scrub runs.
        if (t > 0 && t < 1) {
          drawStreak(ctx, p1, img1, BLOCK_COLORS[l1], toS);
          drawStreak(ctx, p2, img2, BLOCK_COLORS[l2], toS);
          drawStreak(ctx, [p1[0] + p2[0], p1[1] + p2[1]], target, C.sum, toS);
        }

        ctx.restore();
        drawPanelFrame(ctx, r);
        drawPanelTitle(ctx, r, PANEL_TITLES[i]);
      }

      drawConnectors(ctx, rects, blocks);

      ctx.restore();

      // hint (CSS layer, outside the canvas transform)
      raf = requestAnimationFrame(frame);
    };

    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="viewport viewport-blocks" ref={wrapRef}>
      <canvas ref={canvasRef} aria-label="Two-plane 4×4 block visualisation" />
      <div className="viewport-hint">
        drag an amber tip to move x₁ / x₂ · scroll to zoom · drag a panel to pan
      </div>
    </div>
  );
}

/* ================================================================== */
/* Drawing helpers                                                     */
/* ================================================================== */

const mkToScreen =
  (v: BlocksView, i: number) =>
  (p: Vector): [number, number] => [v.ox[i] + p[0] * v.scale, v.oy[i] - p[1] * v.scale];

const labelOf = (letter: string): string => {
  const i = ({ A: 0, B: 1, C: 2, D: 3 } as const)[letter as 'A'] ?? 0;
  return i < 2 ? 'x₁' : 'x₂';
};

/** The straight path a tip has travelled so far (scrub position t < 1). */
function drawStreak(
  ctx: CanvasRenderingContext2D,
  from: Vector,
  to: Vector,
  color: string,
  toS: (p: Vector) => [number, number],
) {
  const a = toS(from);
  const b = toS(to);
  if (Math.hypot(b[0] - a[0], b[1] - a[1]) < 0.5) return;
  ctx.save();
  ctx.strokeStyle = alpha(color, 0.45);
  ctx.lineWidth = 1.6;
  ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.moveTo(a[0], a[1]);
  ctx.lineTo(b[0], b[1]);
  ctx.stroke();
  ctx.restore();
}

function drawPanelFrame(ctx: CanvasRenderingContext2D, r: Rect) {
  ctx.strokeStyle = C.panel;
  ctx.lineWidth = 1;
  ctx.strokeRect(r.x + 0.5, r.y + 0.5, r.w - 1, r.h - 1);
}

function drawPanelTitle(ctx: CanvasRenderingContext2D, r: Rect, text: string) {
  drawLabel(ctx, r.x + 8, r.y + 15, text, C.head, 'left', 0, 0, 0.95, false);
}

function drawHeaders(ctx: CanvasRenderingContext2D, rects: Rect[]) {
  const [tl] = rects;
  const tr = rects[2];
  ctx.save();
  ctx.font = '600 11px ui-monospace, SFMono-Regular, Menlo, monospace';
  ctx.textBaseline = 'middle';
  ctx.textAlign = 'center';
  const put = (x: number, text: string) => {
    const y = MARGIN + 8;
    ctx.lineWidth = 3;
    ctx.strokeStyle = 'rgba(8, 11, 20, 0.9)';
    ctx.strokeText(text, x, y);
    ctx.fillStyle = C.head;
    ctx.fillText(text, x, y);
  };
  put(tl.x + tl.w / 2, 'INPUT   x = (x₁, x₂)');
  put(tr.x + tr.w / 2, 'OUTPUT   y = M x');
  ctx.restore();
}

function panelBounds(r: Rect, v: BlocksView, i: number) {
  return {
    x0: (r.x - v.ox[i]) / v.scale,
    x1: (r.x + r.w - v.ox[i]) / v.scale,
    y0: (v.oy[i] - (r.y + r.h)) / v.scale,
    y1: (v.oy[i] - r.y) / v.scale,
  };
}

function linesFor(b: { x0: number; x1: number; y0: number; y1: number }) {
  const lo = (a: number) => Math.max(Math.floor(a), -MAX_LINES);
  const hi = (a: number) => Math.min(Math.ceil(a), MAX_LINES);
  return { i0: lo(b.x0), i1: hi(b.x1), j0: lo(b.y0), j1: hi(b.y1) };
}

/** The untouched unit lattice inside one panel. */
function drawLattice(
  ctx: CanvasRenderingContext2D,
  r: Rect,
  v: BlocksView,
  i: number,
  toS: (p: Vector) => [number, number],
) {
  const b = panelBounds(r, v, i);
  const { i0, i1, j0, j1 } = linesFor(b);
  ctx.strokeStyle = C.grid;
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let k = i0; k <= i1; k++) {
    const [ax, ay] = toS([k, b.y0]);
    const [bx, by] = toS([k, b.y1]);
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
  }
  for (let k = j0; k <= j1; k++) {
    const [ax, ay] = toS([b.x0, k]);
    const [bx, by] = toS([b.x1, k]);
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
  }
  ctx.stroke();
}

/** The lattice image under a 2×2 block, clipped to one panel. */
function drawImageLattice(
  ctx: CanvasRenderingContext2D,
  r: Rect,
  v: BlocksView,
  i: number,
  K: Matrix,
  toS: (p: Vector) => [number, number],
  color: string,
) {
  const view = panelBounds(r, v, i);
  const inv = inverse(K);
  let src = view;
  if (inv) {
    const corners: Vector[] = [
      [view.x0, view.y0],
      [view.x1, view.y0],
      [view.x0, view.y1],
      [view.x1, view.y1],
    ].map((c) => matVec(inv, c));
    const xs = corners.map((c) => c[0]);
    const ys = corners.map((c) => c[1]);
    src = {
      x0: Math.min(...xs, view.x0) - 1,
      x1: Math.max(...xs, view.x1) + 1,
      y0: Math.min(...ys, view.y0) - 1,
      y1: Math.max(...ys, view.y1) + 1,
    };
  } else {
    src = { x0: view.x0 - 2, x1: view.x1 + 2, y0: view.y0 - 2, y1: view.y1 + 2 };
  }

  const { i0, i1, j0, j1 } = linesFor(src);
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  for (let k = i0; k <= i1; k++) {
    const a = toS(matVec(K, [k, src.y0]));
    const b = toS(matVec(K, [k, src.y1]));
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
  }
  for (let k = j0; k <= j1; k++) {
    const a = toS(matVec(K, [src.x0, k]));
    const b = toS(matVec(K, [src.x1, k]));
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
  }
  ctx.stroke();
}

function drawAxesPanel(
  ctx: CanvasRenderingContext2D,
  r: Rect,
  toS: (p: Vector) => [number, number],
) {
  ctx.strokeStyle = C.axis;
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  const [ox, oy] = toS([0, 0]);
  ctx.moveTo(r.x, oy);
  ctx.lineTo(r.x + r.w, oy);
  ctx.moveTo(ox, r.y);
  ctx.lineTo(ox, r.y + r.h);
  ctx.stroke();
}

/** The block's image of the unit square + its determinant label. */
function drawSquareImage(
  ctx: CanvasRenderingContext2D,
  K: Matrix,
  toS: (p: Vector) => [number, number],
  color: string,
  letter: string,
) {
  const corners: Vector[] = [
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1],
  ].map((c) => matVec(K, c));

  ctx.beginPath();
  corners.forEach((c, idx) => {
    const [x, y] = toS(c);
    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.closePath();
  const d = det(K);
  ctx.fillStyle = Math.abs(d) < 1e-9 ? 'rgba(148, 163, 184, 0.10)' : alpha(color, 0.14);
  ctx.fill();
  ctx.strokeStyle = Math.abs(d) < 1e-9 ? 'rgba(148, 163, 184, 0.45)' : alpha(color, 0.6);
  ctx.lineWidth = 1.5;
  ctx.stroke();

  const anchor = toS(matVec(K, [1, 1]));
  drawLabel(ctx, anchor[0], anchor[1], `det ${letter} = ${fmt(d)}`, color, 'left', 8, 8, 0.95);
}

/**
 * The four blocks as colored connectors between the panels:
 * A: plane 1 → 1, B: plane 2 → 1, C: plane 1 → 2, D: plane 2 → 2.
 */
function drawConnectors(ctx: CanvasRenderingContext2D, rects: Rect[], b: Blocks) {
  const [r0, r1, r2, r3] = rects;
  const left = r0.x + r0.w;
  const right = r2.x;
  const conns: [keyof Blocks, [number, number], [number, number], number][] = [
    ['A', [left, r0.y + r0.h / 2], [right, r2.y + r2.h / 2], 0.5],
    ['B', [left, r1.y + r1.h / 2], [right, r2.y + r2.h / 2], 0.3],
    ['C', [left, r0.y + r0.h / 2], [right, r3.y + r3.h / 2], 0.7],
    ['D', [left, r1.y + r1.h / 2], [right, r3.y + r3.h / 2], 0.5],
  ];

  for (const [letter, from, to, t] of conns) {
    const color = BLOCK_COLORS[letter];
    const mag = blockMagnitude(b[letter]);
    const a = 0.2 + 0.8 * Math.min(1, mag);
    ctx.save();
    ctx.globalAlpha = a;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.setLineDash([7, 5]);
    ctx.beginPath();
    ctx.moveTo(from[0], from[1]);
    ctx.lineTo(to[0], to[1]);
    ctx.stroke();
    ctx.setLineDash([]);

    // letter chip on the line
    const cx = from[0] + (to[0] - from[0]) * t;
    const cy = from[1] + (to[1] - from[1]) * t;
    const w = 22;
    const h = 20;
    ctx.beginPath();
    if (typeof ctx.roundRect === 'function') ctx.roundRect(cx - w / 2, cy - h / 2, w, h, 5);
    else ctx.rect(cx - w / 2, cy - h / 2, w, h);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.strokeStyle = 'rgba(8, 11, 20, 0.7)';
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.font = '700 13px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = '#0b1020';
    ctx.fillText(letter, cx, cy + 0.5);
    ctx.restore();
  }
}
