import { useEffect, useMemo, useRef } from 'react';
import { Decomp, useApp } from '../state/store';
import { DecompFacts, useActiveMatrix, useDataFacts, useDecompFacts } from '../state/hooks';
import {
  Matrix,
  Vector,
  columnSpaceBasis,
  det,
  identity,
  inverse,
  matVec,
  norm,
  normalize,
  nullSpaceBasis,
  quadraticContours,
  rank,
  symmetricPart,
} from '../math/matrix';
import { MatrixTween, blendFromIdentity } from '../math/lerp';
import { EigenEntry, eigen, ellipseAxes, realEigenpairs, symmetricBasis } from '../math/eigen';
import { EquationOverlay } from './EquationOverlay';

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
  ellipse: '#e879f9',
  circle: 'rgba(203, 213, 225, 0.55)',
  /** Composition chain: the hop B produced, in matrix B's accent. */
  chainB: '#a78bfa',
  levelSets: '#2dd4bf',
  // Sign-field ramp: warm (positive) → cool (negative), as [r,g,b].
  formWarm: [251, 191, 36],
  formCool: [129, 140, 248],
  /** Factor directions of the running SVD / spectral demo. */
  decomp: '#f8fafc',
  /** PCA datasets: rose samples, lighter rose for the 1σ/2σ contours. */
  data: '#fb7185',
  dataEllipse: '#fda4af',
  /** Dashed distance from each centered point to its PC1 reconstruction. */
  residual: '#94a3b8',
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
  const decomp = useApp((s) => s.decomp);
  const decompStep = useApp((s) => s.decompStep);
  const decompFacts = useDecompFacts();
  const dataset = useApp((s) => s.dataset);
  const dataFacts = useDataFacts(dataset);
  const transT = useApp((s) => s.transT);
  const ghostAlways = useApp((s) => s.ghostAlways);
  const splitView = useApp((s) => s.splitView);
  const compActive = useApp((s) => s.compActive);
  const compStep = useApp((s) => s.compStep);
  const matrixB = useApp((s) => s.matrixB);
  const eqMode = useApp((s) => s.eqMode);

  const view = useRef<View>({ scale: 44, ox: 0, oy: 0 });
  const tween = useRef<MatrixTween | null>(null);
  /** Eased positions of the data cloud (raw ↔ centered ↔ projected). */
  const cloudShown = useRef<Vector[]>([]);
  const lastT = useRef(performance.now());
  /** Tracks the split toggle so the shared pan offset can follow it. */
  const splitRef = useRef(false);
  const drag = useRef<{ mode: 'none' | 'pan' | 'vector'; x: number; y: number }>({
    mode: 'none',
    x: 0,
    y: 0,
  });
  const latest = useRef({
    M: targetMatrix,
    vector,
    layers,
    A,
    B: matrixB,
    decomp,
    decompStep,
    decompFacts,
    dataset,
    dataFacts,
    transT,
    ghostAlways,
    splitView,
    compActive,
    compStep,
    eqMode,
  });
  latest.current = {
    M: targetMatrix,
    vector,
    layers,
    A,
    B: matrixB,
    decomp,
    decompStep,
    decompFacts,
    dataset,
    dataFacts,
    transT,
    ghostAlways,
    splitView,
    compActive,
    compStep,
    eqMode,
  };

  // Lazily initialised on the first frame so the tween starts settled.
  const ensureTween = () => (tween.current ??= new MatrixTween(targetMatrix.matrix));

  const derived = useMemo(
    () => {
      const sym = symmetricPart(A);
      return {
        invA: inverse(A),
        eigA: eigen(A),
        rankA: rank(A),
        colA: columnSpaceBasis(A),
        nullA: nullSpaceBasis(A),
        detA: det(A),
        // Axes of the quadratic form xᵀAx: eigenbasis of (A+Aᵀ)/2.
        symPairs: symmetricBasis(sym),
        // The symmetric part itself — sign-field and level sets read from it.
        sym,
      };
    },
    [A],
  );
  const derivedRef = useRef(derived);
  derivedRef.current = derived;

  // Half-res sign-field rendering, re-baked only when A or the view changes.
  // Split view keeps one bake per half (their origins differ), hence a small map.
  const formCache = useRef<Map<string, HTMLCanvasElement> | null>(null);

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
      // In split view the left half is a read-only "before" mirror: only the
      // right (after) half reaches for the vector tip — its handles are the
      // ones drawn with this origin.
      const reachable =
        !latest.current.splitView || e.clientX - rect.left >= canvas.clientWidth / 2;

      drag.current =
        reachable && dist < 14 && latest.current.layers.vector
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
        const sx = e.clientX - rect.left;
        const px = ox + tip[0] * scale;
        const py = oy - tip[1] * scale;
        const reachable = !latest.current.splitView || sx >= canvas.clientWidth / 2;
        const near = reachable && Math.hypot(sx - px, e.clientY - rect.top - py) < 14;
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
      // Split view: zoom about the cursor inside whichever half it's over —
      // each half keeps its own origin, exactly W/2 apart.
      const Wc = canvas.clientWidth;
      const left = latest.current.splitView && sx < Wc / 2;
      const effOx = left ? ox - Wc / 2 : ox;
      const wx = (sx - effOx) / scale;
      const wy = (oy - sy) / scale;
      const factor = Math.exp(-e.deltaY * 0.0015);
      const next = clamp(scale * factor, 8, 400);
      view.current.scale = next;
      view.current.ox = sx - wx * next + (left ? Wc / 2 : 0);
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
    /** Seconds since the previous frame — read by the pass body (cloud easing). */
    let frameDt = 0;

    /**
     * One render pass: everything clipped to `rect`, drawn through its own
     * world origin `ox`. Split view runs two of these — a read-only
     * "before" half at identity and the live "after" half at the blended
     * matrix; the overlay mode runs a single full-canvas pass.
     *
     * `dest` is the full matrix the morph is heading for (non-null only
     * while t < 1 on the live side), used for the destination ghosts.
     */
    const drawPass = (
      rect: { x: number; y: number; w: number; h: number; tag: 'before' | 'after' | null },
      ox: number,
      M: Matrix,
      dest: Matrix | null,
    ) => {
      const now = latest.current;
      const ly = now.layers;
      const before = rect.tag === 'before';
      const t = now.transT;
      const v = view.current;
      const vl: View = { scale: v.scale, ox, oy: v.oy };
      const toScreen = (p: Vector): [number, number] => [vl.ox + p[0] * vl.scale, vl.oy - p[1] * vl.scale];

      ctx.save();
      ctx.translate(rect.x, rect.y);
      ctx.beginPath();
      ctx.rect(0, 0, rect.w, rect.h);
      ctx.clip();

      if (ly.form) drawFormField(ctx, derivedRef.current.sym, vl, rect.w, rect.h, formCache);
      if (ly.grid) drawOriginalGrid(ctx, rect.w, rect.h, vl, toScreen);
      drawAxes(ctx, rect.w, rect.h, toScreen);
      if (ly.levelSets) drawLevelSets(ctx, derivedRef.current.symPairs, vl, toScreen);
      if (ly.determinant) drawDeterminantRegion(ctx, M, vl, toScreen);
      if (ly.grid) drawImageGrid(ctx, rect.w, rect.h, vl, M, toScreen);

      if (ly.transpose) {
        drawImageGrid(ctx, rect.w, rect.h, vl, transposeOf(M), toScreen, COLORS.transpose, [5, 4]);
      }
      if (ly.ellipse) drawImageEllipse(ctx, M, vl, toScreen);
      if (ly.columnSpace) drawColumnSpace(ctx, derivedRef.current, vl, toScreen);
      if (ly.nullSpace) drawNullSpace(ctx, derivedRef.current, vl, toScreen);
      if (ly.basis) drawBasis(ctx, M, vl, toScreen);
      if (ly.vector) {
        if (now.compActive) {
          // Composition: draw the whole hop chain from the exact matVec
          // results — the arrows are the destination, the grid is the
          // journey it tweens along. The "before" half shows only step 0.
          drawVectorChain(
            ctx,
            now.A,
            now.B,
            now.vector,
            before ? 0 : now.compStep,
            now.eqMode,
            toScreen,
          );
        } else {
          // The before half draws at identity, so its equation label names I;
          // everywhere else the label follows the drawn map.
          drawVectorPair(ctx, M, now.vector, vl, toScreen, now.eqMode, before ? 'I' : now.M.label);
        }
      }

      // PCA datasets: ease the cloud between its per-step targets (raw →
      // centered → transformed by the step matrix), then draw the 1σ/2σ
      // contours and the PC1 residuals underneath the samples. The "before"
      // pass snaps to the raw snapshot instead — it must never fight the
      // eased array the live pass is animating.
      const ds = now.dataset;
      const df = now.dataFacts;
      if (ds && df && (ly.points || ly.dataEllipse || ly.residuals)) {
        const isPca = now.decomp === 'pcaProjection' || now.decomp === 'pcaRotation';
        const ref = isPca && now.decompStep > 0 ? ds.centered : ds.points;
        let shown: Vector[];
        if (before) {
          shown = ref;
        } else {
          if (cloudShown.current.length !== ref.length) {
            cloudShown.current = ref.map((p) => p.slice());
          }
          const a = 1 - Math.exp(-frameDt * 14);
          const cloud = cloudShown.current;
          for (let i = 0; i < ref.length; i++) {
            const target = isPca ? matVec(M, ref[i]) : ref[i];
            const row = cloud[i];
            for (let c = 0; c < row.length; c++) row[c] += (target[c] - row[c]) * a;
          }
          shown = cloud;
        }
        const mean: Vector = [
          shown.reduce((s, p) => s + p[0], 0) / shown.length,
          shown.reduce((s, p) => s + p[1], 0) / shown.length,
        ];
        if (ly.residuals) drawResiduals(ctx, ds.centered, df.proj, toScreen);
        if (ly.dataEllipse && !before) drawDataEllipse(ctx, M, df.sqrtS, mean, toScreen);
        if (ly.points) {
          drawDataPoints(ctx, shown, toScreen);
          drawMeanMarker(ctx, mean, toScreen);
        }
      }

      if (ly.eigen) drawEigenvectors(ctx, derivedRef.current.eigA, vl, toScreen);
      if (now.decomp === 'svd' || now.decomp === 'spectral')
        drawDecompMarkers(ctx, M, now.decomp, now.decompStep, now.decompFacts, toScreen);

      // Morph landmarks on the live side: the square where the picture
      // started, a dashed outline of where it's heading, and streaks tracing
      // each tracked tip's straight-line travel so far.
      if (!before) {
        const showGhost = now.ghostAlways || t < 1;
        if (showGhost && ly.determinant) {
          drawSquareOutline(
            ctx,
            ([[0, 0], [1, 0], [1, 1], [0, 1]] as Vector[]),
            toScreen,
            'rgba(148, 163, 184, 0.75)',
            [4, 3],
          );
        }
        // Composition step 2: pin the B-image ghost — the violet lattice and
        // unit square where the picture sat after the first hop — so the
        // second hop reads as motion *away* from it. Part of the demo, so it
        // ignores ghostAlways / the transition scrubber.
        if (now.compActive && now.compStep === 2) {
          if (ly.grid) {
            drawImageGrid(ctx, rect.w, rect.h, vl, now.B, toScreen, 'rgba(167, 139, 250, 0.34)', [3, 4]);
          }
          if (ly.determinant) {
            drawSquareOutline(
              ctx,
              ([[0, 0], [1, 0], [1, 1], [0, 1]] as Vector[]).map((c) => matVec(now.B, c)),
              toScreen,
              'rgba(167, 139, 250, 0.75)',
              [6, 4],
            );
          }
        }
        if (dest) {
          if (ly.grid) {
            drawImageGrid(ctx, rect.w, rect.h, vl, dest, toScreen, 'rgba(96, 165, 250, 0.34)', [3, 4]);
          }
          if (ly.determinant) {
            drawSquareOutline(
              ctx,
              ([[0, 0], [1, 0], [1, 1], [0, 1]] as Vector[]).map((c) => matVec(dest, c)),
              toScreen,
              'rgba(163, 230, 53, 0.7)',
              [6, 4],
            );
          }
          if (ly.basis) {
            drawArrow(ctx, toScreen([0, 0]), toScreen(matVec(dest, [1, 0])), COLORS.e1, 1.6, [4, 4], 0.5);
            drawArrow(ctx, toScreen([0, 0]), toScreen(matVec(dest, [0, 1])), COLORS.e2, 1.6, [4, 4], 0.5);
          }
          if (ly.vector && !now.compActive) {
            drawArrow(ctx, toScreen([0, 0]), toScreen(matVec(dest, now.vector)), COLORS.vec, 1.6, [5, 4], 0.5);
          }
        }
        if (t > 0 && t < 1) drawStreaks(ctx, M, now.vector, toScreen);
      }

      if (rect.tag) {
        drawLabel(
          ctx,
          10,
          15,
          before ? 'before · plain space' : 'after · live',
          before ? '#94a3b8' : '#93c5fd',
          'left',
          0,
          0,
          0.95,
        );
      }

      ctx.restore();
    };

    const frame = () => {
      raf = requestAnimationFrame(frame);
      const dpr = canvas.width / Math.max(1, canvas.clientWidth);
      const W = canvas.clientWidth;
      const H = canvas.clientHeight;

      const now = latest.current;
      const tw = ensureTween();
      tw.set(now.M.matrix);
      const base = tw.current();
      const t = now.transT;
      const M = blendFromIdentity(base, t);

      const tNow = performance.now();
      frameDt = Math.min(0.1, (tNow - lastT.current) / 1000);
      lastT.current = tNow;

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);

      const v = view.current;

      // Toggling split keeps the world point at the canvas centre pinned to
      // each half's centre: shift the stored origin by a quarter width.
      if (splitRef.current !== now.splitView) {
        v.ox += now.splitView ? W / 4 : -W / 4;
        splitRef.current = now.splitView;
      }

      paintBackground(ctx, W, H);

      if (now.splitView) {
        const lw = Math.floor(W / 2);
        drawPass({ x: 0, y: 0, w: lw, h: H, tag: 'before' }, v.ox - W / 2, identity(2), null);
        // The after pass is translated by lw, so its world origin must be
        // given in that local frame — passing v.ox here double-shifted every
        // vector/basis/det draw off the right edge of the canvas.
        drawPass({ x: lw, y: 0, w: W - lw, h: H, tag: 'after' }, v.ox - lw, M, t < 1 ? base : null);

        ctx.save();
        ctx.strokeStyle = 'rgba(148, 163, 184, 0.35)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(lw + 0.5, 0);
        ctx.lineTo(lw + 0.5, H);
        ctx.stroke();
        ctx.restore();
      } else {
        drawPass({ x: 0, y: 0, w: W, h: H, tag: null }, v.ox, M, t < 1 ? base : null);
      }
    };

    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  }, []);

  return (
    <div className="viewport viewport-2d" ref={wrapRef}>
      <canvas ref={canvasRef} aria-label="2D transformation visualisation" />
      <EquationOverlay />
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

/**
 * Sign field of the quadratic form xᵀAx (drawn from the symmetric part S,
 * since xᵀAx = xᵀSx): warm amber where positive, cool violet where negative,
 * fading to transparent at the zero set — so an indefinite form shows its
 * null cone as a dark seam splitting warm from cool.
 *
 * Rendered at half resolution into a cached offscreen canvas; the cache is
 * keyed on (S, view, size) so the field is only re-baked when something it
 * depends on actually changes.
 */
function drawFormField(
  ctx: CanvasRenderingContext2D,
  S: Matrix,
  v: View,
  W: number,
  H: number,
  cache: { current: Map<string, HTMLCanvasElement> | null },
) {
  const key = `${S.flat().join(',')}|${v.scale}|${v.ox}|${v.oy}|${W}|${H}`;
  const store = cache.current ?? (cache.current = new Map());
  if (!store.has(key)) {
    const fw = Math.max(1, Math.ceil(W / 2));
    const fh = Math.max(1, Math.ceil(H / 2));
    const cv = document.createElement('canvas');
    cv.width = fw;
    cv.height = fh;
    const cctx = cv.getContext('2d')!;
    const img = cctx.createImageData(fw, fh);
    const data = img.data;

    const [[s11, s12], [, s22]] = S;
    // Normalize each sign separately (sampled over a coarse grid of the
    // visible world): |q| peaks very differently on the two sides when
    // |λ|max ≫ |λ|min, and both families should reach full intensity.
    const b = worldBounds(W, H, v);
    let refPos = 1e-12;
    let refNeg = 1e-12;
    const N = 16;
    for (let j = 0; j <= N; j++) {
      const wy = b.y0 + ((b.y1 - b.y0) * j) / N;
      for (let i = 0; i <= N; i++) {
        const wx = b.x0 + ((b.x1 - b.x0) * i) / N;
        const q = s11 * wx * wx + 2 * s12 * wx * wy + s22 * wy * wy;
        if (q > refPos) refPos = q;
        if (-q > refNeg) refNeg = -q;
      }
    }

    const [wr, wg, wb] = COLORS.formWarm;
    const [cr, cg, cb] = COLORS.formCool;
    const MAX_A = 0.4;

    for (let j = 0; j < fh; j++) {
      const wy = (v.oy - j * 2) / v.scale;
      for (let i = 0; i < fw; i++) {
        const wx = (i * 2 - v.ox) / v.scale;
        const q = s11 * wx * wx + 2 * s12 * wx * wy + s22 * wy * wy;
        // sqrt ramp: q grows quadratically, so a linear |q|/ref ramp would
        // leave the whole mid-field nearly invisible while corners saturate.
        const t = Math.sqrt(Math.min(1, Math.abs(q) / (q >= 0 ? refPos : refNeg)));
        const o = (j * fw + i) * 4;
        if (q >= 0) {
          data[o] = wr;
          data[o + 1] = wg;
          data[o + 2] = wb;
        } else {
          data[o] = cr;
          data[o + 1] = cg;
          data[o + 2] = cb;
        }
        data[o + 3] = Math.round(255 * MAX_A * t);
      }
    }
    cctx.putImageData(img, 0, 0);
    if (store.size >= 4) store.clear();
    store.set(key, cv);
  }

  ctx.save();
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(store.get(key)!, 0, 0, W, H);
  ctx.restore();
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
}

/** The x/y axes through the origin — always drawn, even when the lattice is off. */
function drawAxes(
  ctx: CanvasRenderingContext2D,
  W: number,
  H: number,
  toScreen: (p: Vector) => [number, number],
) {
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

/**
 * Layer ① of the eigenvalue/ellipse story: the unit circle (grey, dashed)
 * and its image under the *currently drawn* matrix `M` — an ellipse whose
 * semi-axes are the singular values σ₁ ≥ σ₂, pointed out with arrows.
 */
function drawImageEllipse(
  ctx: CanvasRenderingContext2D,
  M: Matrix,
  v: View,
  toScreen: (p: Vector) => [number, number],
) {
  // Reference unit circle.
  ctx.save();
  ctx.setLineDash([5, 5]);
  ctx.strokeStyle = COLORS.circle;
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  const [cx, cy] = toScreen([0, 0]);
  ctx.arc(cx, cy, v.scale, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();

  // Image of the circle: sample the unit circle and push every point through M.
  const N = 128;
  const pts: [number, number][] = [];
  for (let k = 0; k < N; k++) {
    const t = (2 * Math.PI * k) / N;
    pts.push(toScreen(matVec(M, [Math.cos(t), Math.sin(t)])));
  }

  ctx.save();
  ctx.beginPath();
  pts.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)));
  ctx.closePath();
  ctx.fillStyle = withAlpha(COLORS.ellipse, 0.10);
  ctx.fill();
  ctx.strokeStyle = COLORS.ellipse;
  ctx.lineWidth = 2.2;
  ctx.stroke();
  ctx.restore();

  // Semi-axes σᵢ along the image's principal directions.
  ellipseAxes(M).forEach((ax, k) => {
    if (ax.sigma < 1e-6) return; // collapsed axis (rank-deficient A)
    drawArrow(ctx, toScreen([0, 0]), toScreen(ax.tip), COLORS.ellipse, 2.4, [], 0.95);
    drawLabel(
      ctx,
      ...toScreen(ax.tip),
      `σ${sub(k + 1)} = ${fmtNum(ax.sigma)}`,
      COLORS.ellipse,
      'left',
      8,
      -11,
    );
  });
}

/**
 * Layer ②: level sets xᵀAx = c for c ∈ {±1, ±3}, drawn from the static
 * matrix A via the eigenbasis of its symmetric part (the directions in
 * which the quadratic form has no cross term).
 */
function drawLevelSets(
  ctx: CanvasRenderingContext2D,
  pairs: { lam: number; dir: Vector }[],
  v: View,
  toScreen: (p: Vector) => [number, number],
) {
  if (pairs.length < 2) return;
  const [p1, p2] = pairs;
  // Open branches are drawn out to ±14 world units; the canvas clips the rest.
  const extent = 14;

  ctx.save();
  ctx.lineWidth = 1.6;
  ctx.strokeStyle = withAlpha(COLORS.levelSets, 0.9);

  for (const c of [1, -1, 3, -3]) {
    for (const branch of quadraticContours(p1.lam, p2.lam, c, 96, extent)) {
      ctx.beginPath();
      branch.forEach((y, i) => {
        const world: Vector = [
          p1.dir[0] * y[0] + p2.dir[0] * y[1],
          p1.dir[1] * y[0] + p2.dir[1] * y[1],
        ];
        const [sx, sy] = toScreen(world);
        if (i === 0) ctx.moveTo(sx, sy);
        else ctx.lineTo(sx, sy);
      });
      ctx.stroke();
    }
  }
  ctx.restore();
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
  eq = false,
  label = 'A',
) {
  const image = matVec(M, x);
  drawArrow(ctx, toScreen([0, 0]), toScreen(x), COLORS.vec, 1.8, [5, 4], 0.85);
  // Equation mode swaps the numeric tips for the generic reading: this
  // arrow is the point x = (x₁, x₂), the image is the map applied to it.
  drawLabel(ctx, ...toScreen(x), eq ? 'x = (x₁, x₂)' : 'x', COLORS.vec, 'left', 11, 12);
  drawArrow(ctx, toScreen([0, 0]), toScreen(image), COLORS.vec, 3, [], 1);
  drawLabel(ctx, ...toScreen(image), eq ? `y = ${label} x` : 'Ax', COLORS.vec, 'left', 8, -11);

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

/** Dashed connector between two arrow tips — the path a point travelled. */
function drawHop(
  ctx: CanvasRenderingContext2D,
  from: [number, number],
  to: [number, number],
  color: string,
) {
  ctx.save();
  ctx.setLineDash([4, 4]);
  ctx.strokeStyle = color;
  ctx.globalAlpha = 0.7;
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  ctx.moveTo(from[0], from[1]);
  ctx.lineTo(to[0], to[1]);
  ctx.stroke();
  ctx.restore();
}

/**
 * Composition demo chain: x → Bx → A(Bx), drawn from the exact matVec
 * results while the grid tweens between steps — each arrow marks where its
 * hop *ends*, so "transforming twice" reads from a single picture.
 *
 * Colour follows the hop: amber x (faint, dimming as the chain grows),
 * matrix B's violet for Bx, amber again for the final image — the same
 * colour the single-map view gives "the image of x". Dashed connectors run
 * tip to tip, each in its destination's colour.
 */
function drawVectorChain(
  ctx: CanvasRenderingContext2D,
  A: Matrix,
  B: Matrix,
  x: Vector,
  step: 0 | 1 | 2,
  eq: boolean,
  toScreen: (p: Vector) => [number, number],
) {
  const p = matVec(B, x);
  const y = matVec(A, p);
  const nums = (w: Vector) => `(${w.map(fmtNum).join(', ')})`;
  const xAlpha = step === 0 ? 0.9 : step === 1 ? 0.55 : 0.4;

  // Hop lines first so the arrows sit on top of them.
  if (step >= 1) drawHop(ctx, toScreen(x), toScreen(p), COLORS.chainB);
  if (step >= 2) drawHop(ctx, toScreen(p), toScreen(y), COLORS.vec);

  // x — always the anchor, staying draggable at its tip.
  drawArrow(ctx, toScreen([0, 0]), toScreen(x), COLORS.vec, 1.8, [5, 4], xAlpha);
  drawLabel(
    ctx,
    ...toScreen(x),
    eq ? 'x = (x₁, x₂)' : `x = ${nums(x)}`,
    COLORS.vec,
    'left',
    11,
    12,
    xAlpha,
  );

  if (step >= 1) {
    // At step 1 the result is still "the answer" (y = B x); by step 2 it
    // has become the input of the next hop (p = B x).
    drawArrow(ctx, toScreen([0, 0]), toScreen(p), COLORS.chainB, 2.6, [], step === 1 ? 1 : 0.85);
    drawLabel(
      ctx,
      ...toScreen(p),
      eq ? (step === 1 ? 'y = B x' : 'p = B x') : `Bx = ${nums(p)}`,
      COLORS.chainB,
      'left',
      8,
      -11,
      step === 1 ? 1 : 0.9,
    );
  }

  if (step === 2) {
    drawArrow(ctx, toScreen([0, 0]), toScreen(y), COLORS.vec, 3, [], 1);
    drawLabel(
      ctx,
      ...toScreen(y),
      eq ? 'y = A p' : `A(Bx) = ${nums(y)}`,
      COLORS.vec,
      'left',
      8,
      -11,
    );
  }

  // draggable handle on x's tip — same hit-test as the single-map view
  const [hx, hy] = toScreen(x);
  ctx.beginPath();
  ctx.arc(hx, hy, 5, 0, Math.PI * 2);
  ctx.fillStyle = COLORS.vec;
  ctx.fill();
  ctx.strokeStyle = '#0b1020';
  ctx.lineWidth = 2;
  ctx.stroke();
}

/** Dashed outline of a quadrilateral (the before / destination squares). */
function drawSquareOutline(
  ctx: CanvasRenderingContext2D,
  corners: Vector[],
  toScreen: (p: Vector) => [number, number],
  color: string,
  dash: number[],
) {
  ctx.save();
  ctx.setLineDash(dash);
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  corners.forEach((c, i) => {
    const [x, y] = toScreen(c);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.closePath();
  ctx.stroke();
  ctx.restore();
}

/**
 * Streaks for the tracked tips while the scrub runs (0 < t < 1): the
 * straight path each point has travelled from its plain-space start to its
 * current position — exact, because lerp(I, M, t) moves every point
 * linearly.
 */
function drawStreaks(
  ctx: CanvasRenderingContext2D,
  M: Matrix,
  x: Vector,
  toScreen: (p: Vector) => [number, number],
) {
  const tracks: [Vector, string][] = [
    [[1, 0], COLORS.e1],
    [[0, 1], COLORS.e2],
    [[1, 1], '#a3e635'],
    [x, COLORS.vec],
  ];
  ctx.save();
  ctx.lineWidth = 1.6;
  ctx.lineCap = 'round';
  for (const [p, color] of tracks) {
    const cur = matVec(M, p);
    const a = toScreen(p);
    const b = toScreen(cur);
    ctx.strokeStyle = withAlpha(color, 0.5);
    ctx.beginPath();
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
    ctx.stroke();
  }
  ctx.restore();
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

/**
 * Factor directions for the running SVD / spectral demo: white dashed
 * arrows drawn through the *tweened* step matrix, so they move with the
 * animation from vᵢ → eᵢ → σᵢeᵢ → σᵢuᵢ (SVD) or vᵢ → eᵢ → λᵢeᵢ → λᵢvᵢ
 * (spectral). Steps that collapse a direction (σ = 0, λ = 0) draw nothing.
 */
function drawDecompMarkers(
  ctx: CanvasRenderingContext2D,
  M: Matrix,
  decomp: Decomp,
  step: number,
  facts: DecompFacts,
  toScreen: (p: Vector) => [number, number],
) {
  const n = M.length;
  const dirs: Vector[] = [];
  if (decomp === 'svd') {
    // Rows of Vᵀ are the right-singular vectors vᵢ.
    for (let i = 0; i < n && facts.svd?.Vt[i]; i++) dirs.push(facts.svd.Vt[i]);
  } else {
    // Columns of P are the eigenvectors.
    const P = facts.spectral.P;
    if (P) for (let i = 0; i < n; i++) dirs.push(P.map((row) => row[i] ?? 0));
  }
  if (dirs.length === 0) return;

  const k = Math.min(Math.max(step, 0), 3);
  const labelFor = (i: number): string => {
    const s = sub(i + 1);
    if (decomp === 'svd') return [`v${s}`, `e${s}`, `σ${s}e${s}`, `σ${s}u${s}`][k];
    return [`v${s}`, `e${s}`, `λ${s}e${s}`, `λ${s}v${s}`][k];
  };

  const [ox, oy] = toScreen([0, 0]);
  for (let i = 0; i < dirs.length; i++) {
    const tip = matVec(M, dirs[i]);
    if (norm(tip) < 1e-6) continue;
    const [tx, ty] = toScreen(tip);
    drawArrow(ctx, [ox, oy], [tx, ty], COLORS.decomp, 1.8, [6, 5], 0.95);
    // Down-left: the ellipse's σ labels and the layer's Ae labels all sit
    // up-right, so markers landing on an axis tip stay readable.
    drawLabel(ctx, tx, ty, labelFor(i), COLORS.decomp, 'right', -10, 13);
  }
}

/* ------------------------------------------------------------------ */
/* PCA datasets — samples, mean, 1σ/2σ contours, PC1 residuals        */
/* ------------------------------------------------------------------ */

/** The cloud: rose dots with a dark rim so they read over grid lines. */
function drawDataPoints(
  ctx: CanvasRenderingContext2D,
  pts: Vector[],
  toScreen: (p: Vector) => [number, number],
) {
  ctx.save();
  for (const p of pts) {
    const [x, y] = toScreen(p);
    ctx.beginPath();
    ctx.arc(x, y, 3.2, 0, Math.PI * 2);
    ctx.fillStyle = COLORS.data;
    ctx.fill();
    ctx.lineWidth = 1;
    ctx.strokeStyle = 'rgba(8, 11, 20, 0.75)';
    ctx.stroke();
  }
  ctx.restore();
}

/** White cross + x̄ label at the cloud's mean (rides the demo, so it lands on the origin once centered). */
function drawMeanMarker(
  ctx: CanvasRenderingContext2D,
  mean: Vector,
  toScreen: (p: Vector) => [number, number],
) {
  const [x, y] = toScreen(mean);
  ctx.save();
  ctx.strokeStyle = COLORS.decomp;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(x - 7, y);
  ctx.lineTo(x + 7, y);
  ctx.moveTo(x, y - 7);
  ctx.lineTo(x, y + 7);
  ctx.stroke();
  ctx.restore();
  drawLabel(ctx, x, y, 'x̄', COLORS.decomp, 'left', 9, -10);
}

/**
 * 1σ (solid) and 2σ (dashed) contours: the circle pushed through √S and
 * then through the step matrix, centred on the cloud's mean — semi-axes
 * √λ along the PCs, not the λ of the "ellipse of A" layer.
 */
function drawDataEllipse(
  ctx: CanvasRenderingContext2D,
  M: Matrix,
  sqrtS: Matrix,
  center: Vector,
  toScreen: (p: Vector) => [number, number],
) {
  const N = 96;
  const ring = (scale: number): Array<[number, number]> =>
    Array.from({ length: N }, (_, k) => {
      const t = (2 * Math.PI * k) / N;
      const u = matVec(sqrtS, [Math.cos(t), Math.sin(t)]).map((x) => x * scale);
      const w = matVec(M, u);
      return toScreen([center[0] + w[0], center[1] + w[1]]);
    });

  ctx.save();
  // 2σ first, behind.
  ctx.setLineDash([6, 5]);
  ctx.lineWidth = 1.4;
  ctx.strokeStyle = withAlpha(COLORS.dataEllipse, 0.7);
  ctx.beginPath();
  ring(2).forEach(([x, y], i) => (i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)));
  ctx.closePath();
  ctx.stroke();
  // 1σ solid with a faint fill.
  ctx.setLineDash([]);
  const p1 = ring(1);
  ctx.beginPath();
  p1.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)));
  ctx.closePath();
  ctx.fillStyle = withAlpha(COLORS.data, 0.08);
  ctx.fill();
  ctx.lineWidth = 1.9;
  ctx.strokeStyle = COLORS.dataEllipse;
  ctx.stroke();
  ctx.restore();
}

/** Dashed segments from each centered sample to its PC1 reconstruction. */
function drawResiduals(
  ctx: CanvasRenderingContext2D,
  centered: Vector[],
  proj: Vector[],
  toScreen: (p: Vector) => [number, number],
) {
  ctx.save();
  ctx.setLineDash([3, 3]);
  ctx.lineWidth = 1.2;
  ctx.strokeStyle = withAlpha(COLORS.residual, 0.95);
  ctx.beginPath();
  for (let i = 0; i < centered.length && i < proj.length; i++) {
    const [ax, ay] = toScreen(centered[i]);
    const [bx, by] = toScreen(proj[i]);
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
  }
  ctx.stroke();
  ctx.restore();
}

/* ================================================================== */
/* Primitives                                                          */
/* ================================================================== */

export function drawArrow(
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

export function drawLabel(
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
      drag the amber tip to move <b>x</b> · scroll to zoom · drag background to pan ·
      scrub the bar below to watch the morph
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
