import { useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { useApp, Decomp } from '../state/store';
import { DecompFacts, useActiveMatrix, useDataFacts, useDecompFacts } from '../state/hooks';
import { EquationOverlay } from './EquationOverlay';
import {
  Matrix,
  Vector,
  columnSpaceBasis,
  det,
  identity,
  inverse,
  matMul,
  matVec,
  norm,
  normalize,
  nullSpaceBasis,
  quadraticContours,
  rank,
  symmetricPart,
} from '../math/matrix';
import { MatrixTween, blendFromIdentity } from '../math/lerp';
import { eigen, ellipseAxes, realEigenpairs, symmetricBasis } from '../math/eigen';

const EXTENT = 5;
/** Lattice lines are kept shorter than the axes so the scene stays readable. */
const LATTICE = 4;
const C = {
  grid: 0x3b4a63,
  axis: 0x7c8ea8,
  gridImage: 0x60a5fa,
  cube: 0x94a3b8,
  cubeImage: 0x38bdf8,
  detPos: 0x22c55e,
  detNeg: 0xef4444,
  detFlat: 0x64748b,
  basis: [0xf87171, 0x4ade80, 0x60a5fa],
  vec: 0xfbbf24,
  /** Composition chain: the hop B produced, in matrix B's accent. */
  chainB: 0xa78bfa,
  eigen: [0xc084fc, 0xf472b6, 0x818cf8],
  nullSpace: 0xfb923c,
  colSpace: 0x22d3ee,
  transpose: 0xfacc15,
  circle: 0xcbd5e1,
  ellipse: 0xe879f9,
  levelSets: 0x2dd4bf,
  formWarm: 0xfbbf24,
  formCool: 0x818cf8,
  /** Factor directions of the running SVD / spectral demo. */
  decomp: 0xf8fafc,
  /** PCA datasets: rose samples, lighter rose for the 1σ/2σ contours. */
  data: 0xfb7185,
  dataEllipse: 0xfda4af,
  /** Dashed-distance residual segments (solid lines in 3D). */
  residual: 0x94a3b8,
};

/** c-values shared by the 2D and 3D level-set layers. */
const LEVEL_C = [1, -1, 3, -3];

/* ------------------------------------------------------------------ */
/* Small helpers                                                       */
/* ------------------------------------------------------------------ */

function pushLine(out: number[], a: number[], b: number[]) {
  out.push(a[0], a[1], a[2], b[0], b[1], b[2]);
}

/** The backing `Float32Array` of a geometry's position attribute. */
function posArr(g: THREE.BufferGeometry): Float32Array {
  return (g.getAttribute('position') as THREE.BufferAttribute).array as Float32Array;
}

function markDirty(g: THREE.BufferGeometry) {
  (g.getAttribute('position') as THREE.BufferAttribute).needsUpdate = true;
}

/** Lattice lines lying in the three coordinate planes. */
function gridPositions(extent = LATTICE): Float32Array {
  const p: number[] = [];
  for (let k = -extent; k <= extent; k++) {
    pushLine(p, [k, -extent, 0], [k, extent, 0]);
    pushLine(p, [-extent, k, 0], [extent, k, 0]);
    pushLine(p, [k, 0, -extent], [k, 0, extent]);
    pushLine(p, [-extent, 0, k], [extent, 0, k]);
    pushLine(p, [0, k, -extent], [0, k, extent]);
    pushLine(p, [0, -extent, k], [0, extent, k]);
  }
  return new Float32Array(p);
}

function unitCubeEdges(): THREE.BufferGeometry {
  const g = new THREE.BoxGeometry(1, 1, 1);
  g.translate(0.5, 0.5, 0.5);
  return new THREE.EdgesGeometry(g);
}

/**
 * Wireframe of the unit sphere as line segments (latitude + longitude
 * circles). Used as the source positions for the image ellipsoid.
 */
function sphereWireframe(latitudes = 7, longitudes = 12, segments = 48): Float32Array {
  const p: number[] = [];
  // latitude circles (constant y)
  for (let i = 1; i < latitudes; i++) {
    const phi = -Math.PI / 2 + (Math.PI * i) / latitudes;
    const r = Math.cos(phi);
    const y = Math.sin(phi);
    for (let s = 0; s < segments; s++) {
      const t0 = (2 * Math.PI * s) / segments;
      const t1 = (2 * Math.PI * (s + 1)) / segments;
      pushLine(p, [r * Math.cos(t0), y, r * Math.sin(t0)], [r * Math.cos(t1), y, r * Math.sin(t1)]);
    }
  }
  // longitude circles (great circles through the poles)
  for (let m = 0; m < longitudes; m++) {
    const th = (Math.PI * m) / longitudes;
    for (let s = 0; s < segments; s++) {
      const t0 = (2 * Math.PI * s) / segments;
      const t1 = (2 * Math.PI * (s + 1)) / segments;
      pushLine(
        p,
        [Math.sin(t0) * Math.cos(th), Math.cos(t0), Math.sin(t0) * Math.sin(th)],
        [Math.sin(t1) * Math.cos(th), Math.cos(t1), Math.sin(t1) * Math.sin(th)],
      );
    }
  }
  return new Float32Array(p);
}

/**
 * Line segments for the level sets xᵀAx = c in 3D, built as cross-sections
 * of the quadric in the eigenbasis of the symmetric part:
 *
 *   λ₁y₁² + λ₂y₂² + λ₃y₃² = c
 *
 * For each coordinate plane (3 of them) and each height h, the slice is a
 * 2D level set of `quadraticContours` with c′ = c − λₖh², lifted back into
 * world space along the matching eigenvector. Handles ellipse / hyperbola /
 * degenerate (parallel-line) slices; empty c′ contributes nothing.
 *
 * Returns the segments bucketed by the sign of `c` — the sign field layer
 * colors the two families (warm for c > 0, cool for c < 0), while the plain
 * level-set layer draws both in teal. Each entry is a `Float32Array` of
 * line-segment endpoints (6 floats per segment).
 */
function levelSetSegments(
  pairs: { lam: number; dir: Vector }[],
  samples = 64,
  extent = 10,
): { pos: Float32Array; neg: Float32Array } {
  if (pairs.length < 3) return { pos: new Float32Array(0), neg: new Float32Array(0) };
  const [p1, p2, p3] = pairs;
  const outPos: number[] = [];
  const outNeg: number[] = [];

  // Heights per axis: ±0.6·L and 0, where L is a natural size of the slice.
  const heightScale = (lam: number, c: number) =>
    Math.abs(lam) < 1e-12 ? 1 : Math.sqrt(Math.abs(c / lam)) || 1;

  // Planes (i, j) with their perpendicular axis k, cyclically permuted.
  const axes: Array<[{ lam: number; dir: Vector }, { lam: number; dir: Vector }, { lam: number; dir: Vector }]> = [
    [p1, p2, p3],
    [p2, p3, p1],
    [p3, p1, p2],
  ];

  for (const c of LEVEL_C) {
    const out = c > 0 ? outPos : outNeg;
    for (const [pi, pj, pk] of axes) {
      const L = heightScale(pk.lam, c);
      const lift = (q: number[], h: number): Vector => [
        pi.dir[0] * q[0] + pj.dir[0] * q[1] + pk.dir[0] * h,
        pi.dir[1] * q[0] + pj.dir[1] * q[1] + pk.dir[1] * h,
        pi.dir[2] * q[0] + pj.dir[2] * q[1] + pk.dir[2] * h,
      ];
      for (const h of [-0.6 * L, 0, 0.6 * L]) {
        const cPrime = c - pk.lam * h * h;
        if (Math.abs(cPrime) < 1e-12) continue;
        for (const branch of quadraticContours(pi.lam, pj.lam, cPrime, samples, extent)) {
          for (let s = 0; s + 1 < branch.length; s++) {
            pushLine(out, lift(branch[s], h), lift(branch[s + 1], h));
          }
        }
      }
    }
  }
  return { pos: new Float32Array(outPos), neg: new Float32Array(outNeg) };
}

function unitCubeSolid(): THREE.BufferGeometry {
  const g = new THREE.BoxGeometry(1, 1, 1);
  g.translate(0.5, 0.5, 0.5);
  return g;
}

function applyMatrixToPositions(M: Matrix, src: Float32Array, dst: Float32Array) {
  for (let i = 0; i < src.length; i += 3) {
    const x = src[i];
    const y = src[i + 1];
    const z = src[i + 2];
    if (M.length === 2) {
      dst[i] = M[0][0] * x + M[0][1] * y;
      dst[i + 1] = M[1][0] * x + M[1][1] * y;
      dst[i + 2] = z;
    } else {
      dst[i] = M[0][0] * x + M[0][1] * y + M[0][2] * z;
      dst[i + 1] = M[1][0] * x + M[1][1] * y + M[1][2] * z;
      dst[i + 2] = M[2][0] * x + M[2][1] * y + M[2][2] * z;
    }
  }
}

/** Canvas-backed text sprite so vectors can be named in 3D. */
class Label3D {
  readonly sprite: THREE.Sprite;
  private canvas = document.createElement('canvas');
  private ctx: CanvasRenderingContext2D;
  private tex: THREE.CanvasTexture;
  private text = '';
  private color = '';

  constructor(color: number, height = 0.4) {
    this.canvas.width = 512;
    this.canvas.height = 128;
    this.ctx = this.canvas.getContext('2d')!;
    this.tex = new THREE.CanvasTexture(this.canvas);
    this.tex.minFilter = THREE.LinearFilter;
    const mat = new THREE.SpriteMaterial({ map: this.tex, transparent: true, depthTest: false });
    this.sprite = new THREE.Sprite(mat);
    this.sprite.renderOrder = 10;
    this.sprite.scale.set(height * 4, height, 1);
    this.color = `#${color.toString(16).padStart(6, '0')}`;
  }

  set(text: string, position: THREE.Vector3) {
    this.sprite.position.copy(position);
    if (text === this.text) return;
    this.text = text;

    const { ctx, canvas } = this;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.font = '600 64px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.lineWidth = 12;
    ctx.strokeStyle = 'rgba(8, 11, 20, 0.95)';
    ctx.strokeText(text, canvas.width / 2, canvas.height / 2);
    ctx.fillStyle = this.color;
    ctx.fillText(text, canvas.width / 2, canvas.height / 2);
    this.tex.needsUpdate = true;
  }

  dispose() {
    this.tex.dispose();
    (this.sprite.material as THREE.SpriteMaterial).dispose();
  }
}

/* ------------------------------------------------------------------ */
/* Component                                                           */
/* ------------------------------------------------------------------ */

export function Viewport3D() {
  const hostRef = useRef<HTMLDivElement>(null);
  const resetRef = useRef<() => void>(() => {});
  const [glError, setGlError] = useState<string | null>(null);

  const active = useActiveMatrix();
  const A = useApp((s) => s.matrix);
  const vector = useApp((s) => s.vector);
  const layers = useApp((s) => s.layers);
  const decomp = useApp((s) => s.decomp);
  const decompStep = useApp((s) => s.decompStep);
  const decompFacts = useDecompFacts();
  const dataset = useApp((s) => s.dataset);
  const dataFacts = useDataFacts(dataset);
  const transT = useApp((s) => s.transT);
  const splitView = useApp((s) => s.splitView);
  const compActive = useApp((s) => s.compActive);
  const compStep = useApp((s) => s.compStep);
  const B = useApp((s) => s.matrixB);
  const eqMode = useApp((s) => s.eqMode);

  const derived = useMemo(
    () => {
      const symPairs = symmetricBasis(symmetricPart(A));
      return {
        invA: inverse(A),
        eigA: eigen(A),
        rankA: rank(A),
        colA: columnSpaceBasis(A),
        nullA: nullSpaceBasis(A),
        detA: det(A),
        // Precomputed level-set geometry (recomputed only when A changes).
        levelSegs: levelSetSegments(symPairs),
      };
    },
    [A],
  );

  const live = useRef({
    active,
    vector,
    layers,
    A,
    B,
    derived,
    decomp,
    decompStep,
    decompFacts,
    dataset,
    dataFacts,
    transT,
    splitView,
    compActive,
    compStep,
    eqMode,
  });
  live.current = {
    active,
    vector,
    layers,
    A,
    B,
    derived,
    decomp,
    decompStep,
    decompFacts,
    dataset,
    dataFacts,
    transT,
    splitView,
    compActive,
    compStep,
    eqMode,
  };

  useEffect(() => {
    const host = hostRef.current!;
    let width = host.clientWidth || 1;
    let height = host.clientHeight || 1;

    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    } catch {
      setGlError(
        '3D view needs WebGL, which this browser/session has disabled. The 2×2 plane view works everywhere.',
      );
      return;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(width, height);
    renderer.setClearColor(0x0b1020, 1);
    host.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 500);
    camera.position.set(6, 5, 9);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 2.5;
    controls.maxDistance = 60;
    controls.maxPolarAngle = Math.PI * 0.98;
    resetRef.current = () => {
      camera.position.set(6, 5, 9);
      controls.target.set(0, 0, 0);
      controls.update();
    };

    /* ---------------- static lattice ---------------- */
    const baseGrid = gridPositions();
    const gridGeom = new THREE.BufferGeometry();
    gridGeom.setAttribute('position', new THREE.BufferAttribute(baseGrid.slice(), 3));
    const gridMesh = new THREE.LineSegments(
      gridGeom,
      new THREE.LineBasicMaterial({ color: C.grid, transparent: true, opacity: 0.55 }),
    );
    scene.add(gridMesh);

    const imageGridGeom = new THREE.BufferGeometry();
    imageGridGeom.setAttribute(
      'position',
      new THREE.BufferAttribute(new Float32Array(baseGrid.length), 3),
    );
    const imageGridMesh = new THREE.LineSegments(
      imageGridGeom,
      new THREE.LineBasicMaterial({ color: C.gridImage, transparent: true, opacity: 0.3 }),
    );
    scene.add(imageGridMesh);

    const transposeGridGeom = new THREE.BufferGeometry();
    transposeGridGeom.setAttribute(
      'position',
      new THREE.BufferAttribute(new Float32Array(baseGrid.length), 3),
    );
    const transposeGridMesh = new THREE.LineSegments(
      transposeGridGeom,
      new THREE.LineBasicMaterial({ color: C.transpose, transparent: true, opacity: 0.35 }),
    );
    transposeGridMesh.visible = false;
    scene.add(transposeGridMesh);

    /* ---------------- axes ---------------- */
    const axisPts: number[] = [];
    pushLine(axisPts, [-EXTENT, 0, 0], [EXTENT, 0, 0]);
    pushLine(axisPts, [0, -EXTENT, 0], [0, EXTENT, 0]);
    pushLine(axisPts, [0, 0, -EXTENT], [0, 0, EXTENT]);
    const axesGeom = new THREE.BufferGeometry();
    axesGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(axisPts), 3));
    scene.add(
      new THREE.LineSegments(axesGeom, new THREE.LineBasicMaterial({ color: C.axis })),
    );

    /* ---------------- unit cube ---------------- */
    const cubeEdges = unitCubeEdges();
    const cubeBase = posArr(cubeEdges);
    const cubeOrig = new THREE.LineSegments(
      cubeEdges,
      new THREE.LineBasicMaterial({ color: C.cube, transparent: true, opacity: 0.9 }),
    );
    scene.add(cubeOrig);

    const cubeImgGeom = new THREE.BufferGeometry();
    cubeImgGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(cubeBase.length), 3));
    const cubeImg = new THREE.LineSegments(
      cubeImgGeom,
      new THREE.LineBasicMaterial({ color: C.cubeImage }),
    );
    scene.add(cubeImg);

    /* -------- destination ghosts (visible only while the scrub runs) -------- */
    const destGridGeom = new THREE.BufferGeometry();
    destGridGeom.setAttribute(
      'position',
      new THREE.BufferAttribute(new Float32Array(baseGrid.length), 3),
    );
    const destGridMesh = new THREE.LineSegments(
      destGridGeom,
      new THREE.LineBasicMaterial({ color: C.gridImage, transparent: true, opacity: 0.16 }),
    );
    destGridMesh.visible = false;
    scene.add(destGridMesh);

    const destCubeGeom = new THREE.BufferGeometry();
    destCubeGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(cubeBase.length), 3));
    const destCubeMesh = new THREE.LineSegments(
      destCubeGeom,
      new THREE.LineBasicMaterial({ color: C.cubeImage, transparent: true, opacity: 0.4 }),
    );
    destCubeMesh.visible = false;
    scene.add(destCubeMesh);

    /* ---- composition ghost: the B-image lattice where hop 1 landed ---- */
    const bGridGeom = new THREE.BufferGeometry();
    bGridGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(baseGrid.length), 3));
    const bGridMesh = new THREE.LineSegments(
      bGridGeom,
      new THREE.LineBasicMaterial({ color: C.chainB, transparent: true, opacity: 0.34 }),
    );
    bGridMesh.visible = false;
    scene.add(bGridMesh);

    const bCubeGeom = new THREE.BufferGeometry();
    bCubeGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(cubeBase.length), 3));
    const bCubeMesh = new THREE.LineSegments(
      bCubeGeom,
      new THREE.LineBasicMaterial({ color: C.chainB, transparent: true, opacity: 0.75 }),
    );
    bCubeMesh.visible = false;
    scene.add(bCubeMesh);

    /* ---------------- determinant volume ---------------- */
    const solidGeom = unitCubeSolid();
    // Must be an independent copy: the geometry's own buffer is rewritten
    // every frame, so aliasing it would compound the transform.
    const solidBase = posArr(solidGeom).slice();
    const solidMat = new THREE.MeshBasicMaterial({
      color: C.detPos,
      transparent: true,
      opacity: 0.2,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    const solidMesh = new THREE.Mesh(solidGeom, solidMat);
    scene.add(solidMesh);

    /* ---------------- unit sphere → ellipsoid ---------------- */
    const sphereBase = sphereWireframe();
    const sphereOrigGeom = new THREE.BufferGeometry();
    sphereOrigGeom.setAttribute('position', new THREE.BufferAttribute(sphereBase.slice(), 3));
    const sphereOrig = new THREE.LineSegments(
      sphereOrigGeom,
      new THREE.LineBasicMaterial({ color: C.circle, transparent: true, opacity: 0.35 }),
    );
    sphereOrig.visible = false;
    scene.add(sphereOrig);

    const sphereImgGeom = new THREE.BufferGeometry();
    sphereImgGeom.setAttribute(
      'position',
      new THREE.BufferAttribute(new Float32Array(sphereBase.length), 3),
    );
    const sphereImg = new THREE.LineSegments(
      sphereImgGeom,
      new THREE.LineBasicMaterial({ color: C.ellipse, transparent: true, opacity: 0.85 }),
    );
    sphereImg.visible = false;
    scene.add(sphereImg);

    /* ---------------- level-set rings ---------------- */
    // Generous preallocation per sign: 2 c-values × 3 planes × 3 heights ×
    // worst-case hyperbola (2 branches × samples segments × 6 floats).
    const LEVEL_MAX_FLOATS = 2 * 3 * 3 * 2 * 64 * 6;
    const mkLevelMesh = (color: number) => {
      const geom = new THREE.BufferGeometry();
      geom.setAttribute(
        'position',
        new THREE.BufferAttribute(new Float32Array(LEVEL_MAX_FLOATS), 3),
      );
      const mesh = new THREE.LineSegments(
        geom,
        new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.8 }),
      );
      mesh.visible = false;
      scene.add(mesh);
      return { geom, mesh, src: null as Float32Array | null };
    };
    // Two families: c > 0 and c < 0 — teal normally, warm/cool with `form`.
    const levelPos = mkLevelMesh(C.levelSets);
    const levelNeg = mkLevelMesh(C.levelSets);

    /* ---------------- arrows ---------------- */
    const mkArrow = (color: number, opacity = 1) =>
      new THREE.ArrowHelper(
        new THREE.Vector3(1, 0, 0),
        new THREE.Vector3(0, 0, 0),
        1,
        color,
        0.35,
        0.2,
      );

    const basisOrig = [0, 1, 2].map((i) => {
      const a = mkArrow(C.basis[i]);
      a.line.material = new THREE.LineBasicMaterial({
        color: C.basis[i],
        transparent: true,
        opacity: 0.45,
      });
      a.cone.material = new THREE.MeshBasicMaterial({
        color: C.basis[i],
        transparent: true,
        opacity: 0.5,
      });
      scene.add(a);
      return a;
    });

    const basisImg = [0, 1, 2].map((i) => {
      const a = mkArrow(C.basis[i]);
      scene.add(a);
      return a;
    });

    // Faint copies of the basis images parked at their final positions —
    // the destination the morph is heading for while the scrub runs.
    const destBasis = [0, 1, 2].map((i) => {
      const a = mkArrow(C.basis[i]);
      (a.line.material as THREE.Material).transparent = true;
      (a.line.material as THREE.Material).opacity = 0.3;
      (a.cone.material as THREE.Material).transparent = true;
      (a.cone.material as THREE.Material).opacity = 0.35;
      a.visible = false;
      scene.add(a);
      return a;
    });

    /* ---- travel streaks (0 < t < 1): start → current for e₁, e₂, e₃, x ---- */
    const STREAKS = 4;
    const streakGeom = new THREE.BufferGeometry();
    streakGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(STREAKS * 6), 3));
    {
      const col = new Float32Array(STREAKS * 6);
      const put = (k: number, hex: number) => {
        // Color() converts sRGB → the renderer's linear working space.
        const c = new THREE.Color(hex);
        for (const vert of [k * 2, k * 2 + 1]) {
          col[vert * 3] = c.r;
          col[vert * 3 + 1] = c.g;
          col[vert * 3 + 2] = c.b;
        }
      };
      put(0, C.basis[0]);
      put(1, C.basis[1]);
      put(2, C.basis[2]);
      put(3, C.vec);
      streakGeom.setAttribute('color', new THREE.BufferAttribute(col, 3));
    }
    const streakMesh = new THREE.LineSegments(
      streakGeom,
      new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.55 }),
    );
    streakMesh.visible = false;
    scene.add(streakMesh);

    const vecOrig = mkArrow(C.vec);
    (vecOrig.line.material as THREE.Material).transparent = true;
    (vecOrig.line.material as THREE.Material).opacity = 0.45;
    (vecOrig.cone.material as THREE.Material).transparent = true;
    (vecOrig.cone.material as THREE.Material).opacity = 0.5;
    scene.add(vecOrig);

    const vecImg = mkArrow(C.vec);
    scene.add(vecImg);

    /* ---- composition chain: the violet Bx hop and its connectors ---- */
    const vecBx = mkArrow(C.chainB);
    (vecBx.line.material as THREE.Material).transparent = true;
    (vecBx.line.material as THREE.Material).opacity = 0.85;
    (vecBx.cone.material as THREE.Material).transparent = true;
    (vecBx.cone.material as THREE.Material).opacity = 0.9;
    vecBx.visible = false;
    scene.add(vecBx);

    /** One dashed-in-2D tip-to-tip hop, drawn solid here (the residual convention). */
    const mkHop = (color: number) => {
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(6), 3));
      const m = new THREE.LineSegments(
        g,
        new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.7 }),
      );
      m.visible = false;
      scene.add(m);
      return m;
    };
    const hop1 = mkHop(C.chainB);
    const hop2 = mkHop(C.vec);

    const eigenArrows = [0, 1, 2].map((i) => {
      const orig = mkArrow(C.eigen[i]);
      const img = mkArrow(C.eigen[i]);
      for (const arrow of [orig, img]) {
        (arrow.line.material as THREE.Material).transparent = true;
        (arrow.cone.material as THREE.Material).transparent = true;
      }
      (orig.line.material as THREE.Material).opacity = 0.4;
      (orig.cone.material as THREE.Material).opacity = 0.45;
      (img.line.material as THREE.Material).opacity = 1;
      scene.add(orig, img);
      return { orig, img };
    });

    /* ---------------- ellipse semi-axes (σ) ---------------- */
    const sigmaArrows = [0, 1, 2].map(() => mkArrow(C.ellipse));

    /* -------- factor directions (SVD / spectral demo) -------- */
    const decompArrows = [0, 1, 2].map(() => mkArrow(C.decomp));

    /* ---------------- PCA datasets ---------------- */
    const MAX_DATA = 400;
    const ptsGeom = new THREE.BufferGeometry();
    ptsGeom.setAttribute(
      'position',
      new THREE.BufferAttribute(new Float32Array(MAX_DATA * 3), 3),
    );
    ptsGeom.setDrawRange(0, 0);
    const ptsMesh = new THREE.Points(
      ptsGeom,
      new THREE.PointsMaterial({ color: C.data, size: 0.13, sizeAttenuation: true }),
    );
    ptsMesh.visible = false;
    scene.add(ptsMesh);

    const meanMesh = new THREE.Mesh(
      new THREE.SphereGeometry(0.07, 12, 12),
      new THREE.MeshBasicMaterial({ color: C.decomp }),
    );
    meanMesh.visible = false;
    scene.add(meanMesh);
    const meanLabel = new Label3D(C.decomp, 0.3);

    // 1σ / 2σ data contours: the unit-sphere wireframe pushed through √S
    // (and 2√S), centred on the cloud's mean.
    const mkDataEllipse = (opacity: number) => {
      const geom = new THREE.BufferGeometry();
      geom.setAttribute(
        'position',
        new THREE.BufferAttribute(new Float32Array(sphereBase.length), 3),
      );
      const mesh = new THREE.LineSegments(
        geom,
        new THREE.LineBasicMaterial({ color: C.dataEllipse, transparent: true, opacity }),
      );
      mesh.visible = false;
      scene.add(mesh);
      return { geom, mesh };
    };
    const dataEll1 = mkDataEllipse(0.85);
    const dataEll2 = mkDataEllipse(0.4);

    // PC1 residuals: each centered sample tethered to its reconstruction
    // (static geometry — rebuilt only when the cloud object changes).
    const residGeom = new THREE.BufferGeometry();
    residGeom.setAttribute(
      'position',
      new THREE.BufferAttribute(new Float32Array(MAX_DATA * 6), 3),
    );
    residGeom.setDrawRange(0, 0);
    const residMesh = new THREE.LineSegments(
      residGeom,
      new THREE.LineBasicMaterial({ color: C.residual, transparent: true, opacity: 0.9 }),
    );
    residMesh.visible = false;
    scene.add(residMesh);

    /* ---------------- subspaces ---------------- */
    const nullLineGeom = new THREE.BufferGeometry();
    nullLineGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(6), 3));
    const nullLine = new THREE.Line(
      nullLineGeom,
      new THREE.LineBasicMaterial({ color: C.nullSpace, linewidth: 2 }),
    );
    nullLine.visible = false;
    scene.add(nullLine);

    const planeGeom = new THREE.PlaneGeometry(60, 60);
    const nullPlane = new THREE.Mesh(
      planeGeom,
      new THREE.MeshBasicMaterial({
        color: C.nullSpace,
        transparent: true,
        opacity: 0.12,
        side: THREE.DoubleSide,
        depthWrite: false,
      }),
    );
    nullPlane.visible = false;
    scene.add(nullPlane);

    const colPlane = new THREE.Mesh(
      new THREE.PlaneGeometry(60, 60),
      new THREE.MeshBasicMaterial({
        color: C.colSpace,
        transparent: true,
        opacity: 0.1,
        side: THREE.DoubleSide,
        depthWrite: false,
      }),
    );
    colPlane.visible = false;
    scene.add(colPlane);

    const colArrows = [0, 1, 2].map((i) => {
      const a = mkArrow(C.colSpace);
      (a.line.material as THREE.Material).transparent = true;
      (a.line.material as THREE.Material).opacity = 0.85;
      scene.add(a);
      return a;
    });

    /* ---------------- labels ---------------- */
    const basisLabels = [0, 1, 2].map((i) => new Label3D(C.basis[i], 0.27));
    const basisImgLabels = [0, 1, 2].map((i) => new Label3D(C.basis[i], 0.33));
    const vecLabel = new Label3D(C.vec, 0.3);
    const vecImgLabel = new Label3D(C.vec, 0.38);
    const vecBxLabel = new Label3D(C.chainB, 0.34);
    const eigenLabels = [0, 1, 2].map((i) => new Label3D(C.eigen[i], 0.33));
    const sigmaLabels = [0, 1, 2].map(() => new Label3D(C.ellipse, 0.33));
    const decompLabels = [0, 1, 2].map(() => new Label3D(C.decomp, 0.33));
    const allLabels = [
      ...basisLabels,
      ...basisImgLabels,
      vecLabel,
      vecImgLabel,
      vecBxLabel,
      ...eigenLabels,
      ...sigmaLabels,
      ...decompLabels,
      meanLabel,
    ];
    allLabels.forEach((l) => scene.add(l.sprite));

    /* ---------------- tween + loop ---------------- */
    const tween = new MatrixTween(live.current.active.matrix);
    let raf = 0;
    /** Seconds since the previous frame — the cloud easing reads it. */
    let dt = 0;
    const I3 = identity(3);
    const tmp = new THREE.Vector3();
    // Eased data-cloud positions (raw ↔ centered ↔ step-matrix targets) and
    // the identity of the cloud they were last snapped to.
    const cloudArr = new Float32Array(MAX_DATA * 3);
    let cloudCount = -1;
    let residKey: unknown = null;
    let lastT = performance.now();

    const setArrow = (
      arrow: THREE.ArrowHelper,
      dir: Vector,
      length: number,
      visible: boolean,
    ) => {
      const n = norm(dir);
      if (!visible || n < 1e-9 || length < 1e-6) {
        arrow.visible = false;
        return;
      }
      arrow.visible = true;
      tmp.set(dir[0] / n, dir[1] / n, dir[2] / n);
      arrow.setDirection(tmp);
      arrow.setLength(length, Math.min(0.4, length * 0.3), Math.min(0.26, length * 0.2));
    };

    /**
     * Per-frame scene update for one render pass: `M` is the matrix to draw
     * with, `dest` the full matrix the scrub is heading for (non-null only
     * while 0 < t < 1 on the live side), and `side = 'before'` hides every
     * image-side object so split view's left half stays plain space.
     */
    const updateScene = (M: Matrix, side: 'before' | 'after', dest: Matrix | null) => {
      const before = side === 'before';
      const {
        vector: vec,
        layers: ly,
        A: Amat,
        B: bMat,
        compActive: ca,
        compStep: cs,
        eqMode: eq,
        active: act,
        derived: dr,
        decomp: dc,
        decompStep: ds,
        decompFacts: df,
        dataset: dset,
        dataFacts: cf,
      } = live.current;

      // lattice
      gridMesh.visible = ly.grid;
      imageGridMesh.visible = ly.grid && !before;
      applyMatrixToPositions(M, baseGrid, posArr(imageGridGeom));
      markDirty(imageGridGeom);
      destGridMesh.visible = !before && !!dest && ly.grid;
      if (dest && destGridMesh.visible) {
        applyMatrixToPositions(dest, baseGrid, posArr(destGridGeom));
        markDirty(destGridGeom);
      }
      transposeGridMesh.visible = ly.transpose && !before;
      if (ly.transpose && !before) {
        const T = Array.from({ length: M.length }, (_, i) =>
          Array.from({ length: M.length }, (_, j) => M[j][i]),
        );
        applyMatrixToPositions(T, baseGrid, posArr(transposeGridGeom));
        markDirty(transposeGridGeom);
      }

      // unit cube + determinant volume
      cubeImg.visible = !before;
      applyMatrixToPositions(M, cubeBase, posArr(cubeImgGeom));
      markDirty(cubeImgGeom);
      destCubeMesh.visible = !before && !!dest;
      if (dest && destCubeMesh.visible) {
        applyMatrixToPositions(dest, cubeBase, posArr(destCubeGeom));
        markDirty(destCubeGeom);
      }

      // Composition step 2: pin the B-image ghost — the violet lattice and
      // cube where the picture sat after hop 1 — so hop 2 reads as motion
      // *away* from it. Part of the demo, so it ignores ghostAlways and the
      // transition scrubber (the 2D view does the same).
      const showBGhost = ca && cs === 2 && !before;
      bGridMesh.visible = showBGhost && ly.grid;
      if (showBGhost && ly.grid) {
        applyMatrixToPositions(bMat, baseGrid, posArr(bGridGeom));
        markDirty(bGridGeom);
      }
      bCubeMesh.visible = showBGhost && ly.determinant;
      if (showBGhost && ly.determinant) {
        applyMatrixToPositions(bMat, cubeBase, posArr(bCubeGeom));
        markDirty(bCubeGeom);
      }

      const d = det(M);
      solidMesh.visible = ly.determinant && !before;
      if (ly.determinant && !before) {
        applyMatrixToPositions(M, solidBase, posArr(solidGeom));
        markDirty(solidGeom);
        solidMat.color.setHex(Math.abs(d) < 1e-9 ? C.detFlat : d > 0 ? C.detPos : C.detNeg);
        // Double-sided, so the effective coverage is roughly twice `opacity`.
        solidMat.opacity = Math.abs(d) < 1e-9 ? 0.4 : 0.2;
      }

      // basis
      for (let i = 0; i < 3; i++) {
        const unit = [0, 0, 0];
        unit[i] = 1;
        const image = [
          M[0]?.[i] ?? unit[0],
          M[1]?.[i] ?? unit[1],
          M[2]?.[i] ?? unit[2],
        ];
        setArrow(basisOrig[i], unit, 1, ly.basis);
        setArrow(basisImg[i], image, norm(image), ly.basis && !before);
        // destination ghost: where this basis image lands at full exposure
        const dcol: Vector = dest
          ? [dest[0]?.[i] ?? 0, dest[1]?.[i] ?? 0, dest[2]?.[i] ?? 0]
          : [0, 0, 0];
        setArrow(destBasis[i], dcol, norm(dcol), !!dest && ly.basis);

        const showLabels = ly.basis && norm(image) > 1e-6;
        basisLabels[i].sprite.visible = ly.basis;
        basisImgLabels[i].sprite.visible = showLabels && !before;
        if (ly.basis) {
          basisLabels[i].set(`e${sub(i + 1)}`, new THREE.Vector3(...unit).multiplyScalar(1.18));
          basisImgLabels[i].set(
            `Ae${sub(i + 1)}`,
            new THREE.Vector3(image[0], image[1], image[2]).add(new THREE.Vector3(0.2, 0.25, 0)),
          );
        }
      }

      // vector — the composition demo draws the whole hop chain from the
      // exact matVec results while the grid tweens between steps, mirroring
      // the 2D chain: amber x (dimming as the chain grows), violet Bx, and
      // the final A(Bx) in the usual image colour. The "before" half shows
      // only step 0 — the input.
      const step = ca ? (before ? 0 : cs) : -1;
      const xAlpha = ca ? (step === 0 ? 0.9 : step === 1 ? 0.55 : 0.4) : 0.45;
      (vecOrig.line.material as THREE.Material).opacity = xAlpha;
      (vecOrig.cone.material as THREE.Material).opacity = ca ? xAlpha : 0.5;
      setArrow(vecOrig, [vec[0], vec[1], vec[2] ?? 0], norm(vec), ly.vector);
      vecLabel.sprite.visible = ly.vector;
      if (ly.vector) {
        vecLabel.set(
          'x',
          new THREE.Vector3(vec[0], vec[1], vec[2] ?? 0).add(new THREE.Vector3(0.15, 0.25, 0)),
        );
      }

      const pHop = ca ? matVec(bMat, vec) : null; // hop 1 — exact, untweened
      const yHop = ca ? matVec(Amat, pHop!) : null; // hop 2 — exact
      const bxVec: Vector = pHop ?? [0, 0, 0];

      // violet Bx — still the answer at step 1, already the input of hop 2
      // by step 2 (its label follows that reading, like the 2D tips).
      const showBx = ly.vector && step >= 1;
      setArrow(vecBx, bxVec, norm(bxVec), showBx);
      vecBxLabel.sprite.visible = showBx && norm(bxVec) > 1e-6;
      if (showBx && norm(bxVec) > 1e-6) {
        (vecBx.line.material as THREE.Material).opacity = step === 1 ? 1 : 0.85;
        (vecBx.cone.material as THREE.Material).opacity = step === 1 ? 1 : 0.9;
        vecBxLabel.set(
          eq ? (step === 1 ? 'y = B x' : 'p = B x') : 'Bx',
          new THREE.Vector3(bxVec[0], bxVec[1], bxVec[2]).add(new THREE.Vector3(0.15, 0.25, 0)),
        );
      }

      // the image arrow: the single-map image, or the chain's final hop
      const imageVec: Vector = ca ? yHop! : matVec(M, vec);
      const showImg = ly.vector && !before && (!ca || step === 2);
      setArrow(vecImg, [imageVec[0], imageVec[1], imageVec[2] ?? 0], norm(imageVec), showImg);
      vecImgLabel.sprite.visible = showImg && norm(imageVec) > 1e-6;
      if (showImg && norm(imageVec) > 1e-6) {
        vecImgLabel.set(
          ca ? (eq ? 'y = A p' : 'A(Bx)') : eq ? `y = ${act.label} x` : 'Ax',
          new THREE.Vector3(imageVec[0], imageVec[1], imageVec[2] ?? 0).add(
            new THREE.Vector3(0.15, 0.25, 0),
          ),
        );
      }

      // tip-to-tip hops: the path each point travelled (solid in 3D, where
      // the 2D view dashes them).
      const hop1Show = ly.vector && step >= 1;
      hop1.visible = hop1Show;
      if (hop1Show && pHop) {
        const a = posArr(hop1.geometry);
        a[0] = vec[0]; a[1] = vec[1]; a[2] = vec[2] ?? 0;
        a[3] = pHop[0]; a[4] = pHop[1]; a[5] = pHop[2] ?? 0;
        markDirty(hop1.geometry);
      }
      const hop2Show = ly.vector && step === 2;
      hop2.visible = hop2Show;
      if (hop2Show && pHop && yHop) {
        const a = posArr(hop2.geometry);
        a[0] = pHop[0]; a[1] = pHop[1]; a[2] = pHop[2] ?? 0;
        a[3] = yHop[0]; a[4] = yHop[1]; a[5] = yHop[2] ?? 0;
        markDirty(hop2.geometry);
      }

      // travel streaks: the straight path each tracked tip has covered so
      // far (exact, because lerp(I, M, t) moves every point linearly).
      const showStreak = !before && dest !== null && (ly.basis || ly.vector);
      streakMesh.visible = showStreak;
      if (showStreak) {
        const arr = posArr(streakGeom);
        const tips: Vector[] = [
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1],
          [vec[0], vec[1], vec[2] ?? 0],
        ];
        tips.forEach((p, k) => {
          const off = (k < 3 && !ly.basis) || (k === 3 && !ly.vector);
          const cur = off ? p : matVec(M, p);
          arr[k * 6] = p[0];
          arr[k * 6 + 1] = p[1];
          arr[k * 6 + 2] = p[2];
          arr[k * 6 + 3] = cur[0];
          arr[k * 6 + 4] = cur[1];
          arr[k * 6 + 5] = cur[2];
        });
        markDirty(streakGeom);
      }

      // eigenvectors (flattened so a repeated eigenvalue can show two axes)
      const pairs = realEigenpairs(dr.eigA);
      const flat = pairs
        .flatMap((p) => p.vectors.map((vv) => ({ vv, lam: p.real })))
        .slice(0, 3);
      for (let i = 0; i < 3; i++) {
        const item = flat[i];
        const { orig, img } = eigenArrows[i];
        const show = ly.eigen && !!item;
        orig.visible = show;
        img.visible = show;
        eigenLabels[i].sprite.visible = show;
        if (!item) continue;

        const v3 = [item.vv[0], item.vv[1], item.vv[2] ?? 0];
        setArrow(orig, v3, 1.6, show);
        setArrow(
          img,
          v3.map((x) => x * item.lam),
          Math.abs(item.lam) * 1.6,
          show && !before && Math.abs(item.lam) > 1e-9,
        );
        const tip = new THREE.Vector3(v3[0] * 1.6, v3[1] * 1.6, v3[2] * 1.6);
        tip.add(new THREE.Vector3(0.15, 0.25, 0));
        eigenLabels[i].set(`v, λ=${round(item.lam)}`, tip);
      }

      // unit sphere → ellipsoid, with the σ semi-axes of the image
      const showEllipse = ly.ellipse;
      sphereOrig.visible = showEllipse;
      sphereImg.visible = showEllipse && !before;
      if (showEllipse && !before) {
        applyMatrixToPositions(M, sphereBase, posArr(sphereImgGeom));
        markDirty(sphereImgGeom);
      }
      const axes3 = showEllipse && !before ? ellipseAxes(M) : [];
      for (let i = 0; i < 3; i++) {
        const ax = axes3[i];
        const show = showEllipse && !before && !!ax && ax.sigma > 1e-6;
        setArrow(sigmaArrows[i], ax ? ax.tip : [0, 0, 0], ax ? ax.sigma : 0, show);
        sigmaLabels[i].sprite.visible = show;
        if (show) {
          sigmaLabels[i].set(
            `σ${sub(i + 1)} = ${round(ax.sigma)}`,
            new THREE.Vector3(ax.tip[0], ax.tip[1], ax.tip[2]).add(
              new THREE.Vector3(0.2, 0.2, 0),
            ),
          );
        }
      }

      // factor directions: where vᵢ (SVD) or each eigenvector sits right
      // now, under the tweened step matrix — labels track the step.
      let decDirs: Vector[] = [];
      if (dc === 'svd' && df.svd) {
        for (let i = 0; i < 3 && df.svd.Vt[i]; i++) decDirs.push(df.svd.Vt[i]);
      } else if (dc === 'spectral' && df.spectral.P) {
        const Pc = df.spectral.P;
        for (let i = 0; i < 3; i++) decDirs.push(Pc.map((row) => row[i] ?? 0));
      }
      const decompLabel = (i: number): string => {
        const s = sub(i + 1);
        const k = Math.min(Math.max(ds, 0), 3);
        return dc === 'svd'
          ? [`v${s}`, `e${s}`, `σ${s}e${s}`, `σ${s}u${s}`][k]
          : [`v${s}`, `e${s}`, `λ${s}e${s}`, `λ${s}v${s}`][k];
      };
      for (let i = 0; i < 3; i++) {
        const dir = decDirs[i];
        const tip = dir ? matVec(M, dir) : [0, 0, 0];
        const len = norm(tip);
        const show = !!dc && !!dir && !before;
        setArrow(decompArrows[i], tip, len, show);
        decompLabels[i].sprite.visible = show && len > 1e-6;
        if (show && len > 1e-6) {
          decompLabels[i].set(
            decompLabel(i),
            // Down-left, opposing the σ/eigen labels (up-right) so the two
            // families don't stack when a marker lands on an ellipse axis.
            new THREE.Vector3(tip[0], tip[1], tip[2]).add(new THREE.Vector3(-0.7, -0.6, 0)),
          );
        }
      }

      // PCA datasets: eased cloud (raw → centered → through the step
      // matrix), √S contours riding the same matrix, and PC1 residuals.
      const showPts = !!dset && ly.points;
      ptsMesh.visible = showPts;
      meanMesh.visible = showPts;
      meanLabel.sprite.visible = showPts;
      if (dset) {
        const isPca = dc === 'pcaProjection' || dc === 'pcaRotation';
        const ref = isPca && ds > 0 ? dset.centered : dset.points;
        const n = Math.min(ref.length, MAX_DATA);
        const dst = posArr(ptsGeom);
        if (before) {
          // Plain-space snapshot: the raw/centered points, written straight
          // to the geometry so they never fight the eased array the live
          // pass is animating.
          for (let i = 0; i < n; i++) {
            dst[i * 3] = ref[i][0];
            dst[i * 3 + 1] = ref[i][1];
            dst[i * 3 + 2] = ref[i][2] ?? 0;
          }
        } else {
          if (cloudCount !== ref.length) {
            for (let i = 0; i < n; i++) {
              const t = isPca ? matVec(M, ref[i]) : ref[i];
              cloudArr[i * 3] = t[0];
              cloudArr[i * 3 + 1] = t[1];
              cloudArr[i * 3 + 2] = t[2] ?? 0;
            }
            cloudCount = ref.length;
          } else {
            const a = 1 - Math.exp(-dt * 14);
            for (let i = 0; i < n; i++) {
              const t = isPca ? matVec(M, ref[i]) : ref[i];
              for (let c = 0; c < 3; c++) {
                const target = t[c] ?? 0;
                cloudArr[i * 3 + c] += (target - cloudArr[i * 3 + c]) * a;
              }
            }
          }
          dst.set(cloudArr.subarray(0, n * 3));
        }
        ptsGeom.setDrawRange(0, showPts ? n : 0);
        markDirty(ptsGeom);

        // mean of the *shown* cloud — lands on the origin once centered.
        let mx = 0;
        let my = 0;
        let mz = 0;
        const src = before ? dst : cloudArr;
        for (let i = 0; i < n; i++) {
          mx += src[i * 3];
          my += src[i * 3 + 1];
          mz += src[i * 3 + 2];
        }
        if (n > 0) {
          mx /= n;
          my /= n;
          mz /= n;
        }
        meanMesh.position.set(mx, my, mz);
        if (showPts) meanLabel.set('x̄', tmp.set(mx, my, mz));

        // 1σ / 2σ contours: √S (and 2√S) pushed through the step matrix.
        const showEll = ly.dataEllipse && !!cf && !before;
        dataEll1.mesh.visible = showEll;
        dataEll2.mesh.visible = showEll;
        if (showEll && cf) {
          const Em = matMul(M, cf.sqrtS);
          applyMatrixToPositions(Em, sphereBase, posArr(dataEll1.geom));
          markDirty(dataEll1.geom);
          const Em2 = Em.map((r) => r.map((x) => 2 * x));
          applyMatrixToPositions(Em2, sphereBase, posArr(dataEll2.geom));
          markDirty(dataEll2.geom);
          dataEll1.mesh.position.set(mx, my, mz);
          dataEll2.mesh.position.set(mx, my, mz);
        }

        // PC1 residuals — static segments, rebuilt when the cloud changes.
        const showRes = ly.residuals && !!cf;
        residMesh.visible = showRes;
        if (showRes && cf && residKey !== dset) {
          const arr = posArr(residGeom);
          const m = Math.min(dset.centered.length, cf.proj.length, MAX_DATA);
          for (let i = 0; i < m; i++) {
            const c = dset.centered[i];
            const p = cf.proj[i];
            arr[i * 6] = c[0];
            arr[i * 6 + 1] = c[1];
            arr[i * 6 + 2] = c[2] ?? 0;
            arr[i * 6 + 3] = p[0];
            arr[i * 6 + 4] = p[1];
            arr[i * 6 + 5] = p[2] ?? 0;
          }
          residGeom.setDrawRange(0, m * 2);
          markDirty(residGeom);
          residKey = dset;
        }
      } else {
        ptsGeom.setDrawRange(0, 0);
        dataEll1.mesh.visible = false;
        dataEll2.mesh.visible = false;
        residMesh.visible = false;
      }

      // level sets xᵀAx = c (geometry precomputed from A). `levelSets` draws
      // them in teal; the `form` layer reveals the same rings colored by the
      // sign of c (warm c > 0, cool c < 0).
      const showLevels = ly.levelSets || ly.form;
      const fillLevel = (
        slot: { geom: THREE.BufferGeometry; mesh: THREE.LineSegments; src: Float32Array | null },
        segs: Float32Array,
      ) => {
        slot.mesh.visible = showLevels && segs.length > 0;
        if (slot.src === segs || segs.length === 0) return;
        const dst = posArr(slot.geom);
        const count = Math.min(segs.length, dst.length);
        dst.set(segs.subarray(0, count));
        slot.geom.setDrawRange(0, Math.floor(count / 3));
        markDirty(slot.geom);
        slot.src = segs;
      };
      fillLevel(levelPos, dr.levelSegs.pos);
      fillLevel(levelNeg, dr.levelSegs.neg);
      const posColor = ly.form ? C.formWarm : C.levelSets;
      const negColor = ly.form ? C.formCool : C.levelSets;
      (levelPos.mesh.material as THREE.LineBasicMaterial).color.set(posColor);
      (levelNeg.mesh.material as THREE.LineBasicMaterial).color.set(negColor);

      // column space
      const showCol = ly.columnSpace;
      colPlane.visible = showCol && dr.rankA === 2 && dr.nullA.length === 1;
      colArrows.forEach((a, i) => {
        const col = dr.colA[i];
        if (!showCol || !col) {
          a.visible = false;
          return;
        }
        const c3 = [col[0], col[1], col[2] ?? 0];
        setArrow(a, c3, norm(c3), showCol);
      });
      if (colPlane.visible && dr.nullA[0]) {
        const n = normalize([...dr.nullA[0], 0].slice(0, 3));
        const nn = norm(n) < 1e-9 ? [0, 0, 1] : n;
        colPlane.quaternion.setFromUnitVectors(
          new THREE.Vector3(0, 0, 1),
          new THREE.Vector3(nn[0], nn[1], nn[2]),
        );
      }

      // null space
      nullPlane.visible = false;
      nullLine.visible = false;
      if (ly.nullSpace) {
        if (dr.nullA.length === 0) {
          nullLine.visible = true;
          const a = posArr(nullLineGeom);
          a[0] = 0; a[1] = 0; a[2] = 0;
          a[3] = 0.001; a[4] = 0.001; a[5] = 0.001;
          markDirty(nullLineGeom);
        } else if (dr.nullA.length === 1) {
          const n = normalize([...dr.nullA[0], 0].slice(0, 3));
          nullLine.visible = true;
          const a = posArr(nullLineGeom);
          a[0] = -n[0] * 30; a[1] = -n[1] * 30; a[2] = -n[2] * 30;
          a[3] = n[0] * 30; a[4] = n[1] * 30; a[5] = n[2] * 30;
          markDirty(nullLineGeom);
        } else if (dr.nullA.length >= 2 && dr.colA.length === 1) {
          nullPlane.visible = true;
          const n = normalize([...dr.colA[0], 0].slice(0, 3));
          const nn = norm(n) < 1e-9 ? [0, 0, 1] : n;
          nullPlane.quaternion.setFromUnitVectors(
            new THREE.Vector3(0, 0, 1),
            new THREE.Vector3(nn[0], nn[1], nn[2]),
          );
        }
      }

    };

    const frame = () => {
      raf = requestAnimationFrame(frame);
      controls.update();

      const tNow = performance.now();
      dt = Math.min(0.1, (tNow - lastT) / 1000);
      lastT = tNow;

      const act = live.current.active;
      tween.set(act.matrix);
      const base = tween.current();
      const t = live.current.transT;
      const M = blendFromIdentity(base, t);
      const dest = t < 1 ? base : null;

      if (live.current.splitView) {
        // Before | after: one scissored pass each through the same camera —
        // left at identity (plain space), right at the blended matrix.
        const lw = Math.floor(width / 2);
        renderer.setScissorTest(true);

        updateScene(I3, 'before', null);
        camera.aspect = lw / height;
        camera.updateProjectionMatrix();
        renderer.setViewport(0, 0, lw, height);
        renderer.setScissor(0, 0, lw, height);
        renderer.render(scene, camera);

        updateScene(M, 'after', dest);
        const rw = width - lw;
        camera.aspect = rw / height;
        camera.updateProjectionMatrix();
        renderer.setViewport(lw, 0, rw, height);
        renderer.setScissor(lw, 0, rw, height);
        renderer.render(scene, camera);

        renderer.setScissorTest(false);
        renderer.setViewport(0, 0, width, height);
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
      } else {
        renderer.setScissorTest(false);
        renderer.setViewport(0, 0, width, height);
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        updateScene(M, 'after', dest);
        renderer.render(scene, camera);
      }
    };

    raf = requestAnimationFrame(frame);

    const ro = new ResizeObserver(() => {
      width = host.clientWidth || 1;
      height = host.clientHeight || 1;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    });
    ro.observe(host);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      controls.dispose();
      scene.traverse((obj) => {
        if ((obj as THREE.Sprite).isSprite) return; // disposed via Label3D
        const any = obj as THREE.Mesh;
        if (any.geometry) any.geometry.dispose();
        const m = any.material as THREE.Material | THREE.Material[] | undefined;
        if (Array.isArray(m)) m.forEach((x) => x.dispose());
        else m?.dispose();
      });
      allLabels.forEach((l) => l.dispose());
      renderer.dispose();
      if (renderer.domElement.parentNode === host) host.removeChild(renderer.domElement);
    };
  }, []);

  return (
    <div className="viewport viewport-3d">
      <div ref={hostRef} className="viewport-3d-host" aria-label="3D transformation visualisation" />
      <EquationOverlay />
      {glError && (
        <div className="viewport-fallback" role="alert">
          <strong>3D view unavailable</strong>
          <span>{glError}</span>
        </div>
      )}
      <div className="viewport-controls">
        <button type="button" className="btn btn-ghost" onClick={() => resetRef.current()}>
          Reset view
        </button>
      </div>
      {splitView && (
        <div className="split-tags" aria-hidden="true">
          <span className="split-before">before · plain space</span>
          <span className="split-after">after · live</span>
        </div>
      )}
      <div className="viewport-hint">drag to orbit · scroll to zoom · right-drag to pan</div>
    </div>
  );
}

function sub(n: number) {
  return ['', '₁', '₂', '₃'][n] ?? String(n);
}

function round(x: number) {
  return Math.round(x * 1e4) / 1e4;
}

export default Viewport3D;
