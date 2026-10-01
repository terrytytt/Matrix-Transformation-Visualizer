import { useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { useApp } from '../state/store';
import { useActiveMatrix } from '../state/hooks';
import {
  Matrix,
  Vector,
  columnSpaceBasis,
  det,
  inverse,
  matVec,
  norm,
  normalize,
  nullSpaceBasis,
  rank,
} from '../math/matrix';
import { MatrixTween } from '../math/lerp';
import { eigen, realEigenpairs } from '../math/eigen';

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
  eigen: [0xc084fc, 0xf472b6, 0x818cf8],
  nullSpace: 0xfb923c,
  colSpace: 0x22d3ee,
  transpose: 0xfacc15,
};

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

  const live = useRef({ active, vector, layers, A, derived });
  live.current = { active, vector, layers, A, derived };

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

    const vecOrig = mkArrow(C.vec);
    (vecOrig.line.material as THREE.Material).transparent = true;
    (vecOrig.line.material as THREE.Material).opacity = 0.45;
    (vecOrig.cone.material as THREE.Material).transparent = true;
    (vecOrig.cone.material as THREE.Material).opacity = 0.5;
    scene.add(vecOrig);

    const vecImg = mkArrow(C.vec);
    scene.add(vecImg);

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
    const eigenLabels = [0, 1, 2].map((i) => new Label3D(C.eigen[i], 0.33));
    const allLabels = [
      ...basisLabels,
      ...basisImgLabels,
      vecLabel,
      vecImgLabel,
      ...eigenLabels,
    ];
    allLabels.forEach((l) => scene.add(l.sprite));

    /* ---------------- tween + loop ---------------- */
    const tween = new MatrixTween(live.current.active.matrix);
    let raf = 0;
    const tmp = new THREE.Vector3();

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

    const frame = () => {
      raf = requestAnimationFrame(frame);
      controls.update();

      const { active: act, vector: vec, layers: ly, A: Amat, derived: dr } = live.current;
      tween.set(act.matrix);
      const M = tween.current();

      // lattice
      applyMatrixToPositions(M, baseGrid, posArr(imageGridGeom));
      markDirty(imageGridGeom);
      transposeGridMesh.visible = ly.transpose;
      if (ly.transpose) {
        const T = Array.from({ length: M.length }, (_, i) =>
          Array.from({ length: M.length }, (_, j) => M[j][i]),
        );
        applyMatrixToPositions(T, baseGrid, posArr(transposeGridGeom));
        markDirty(transposeGridGeom);
      }

      // unit cube + determinant volume
      applyMatrixToPositions(M, cubeBase, posArr(cubeImgGeom));
      markDirty(cubeImgGeom);

      const d = det(M);
      solidMesh.visible = ly.determinant;
      if (ly.determinant) {
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
        setArrow(basisImg[i], image, norm(image), ly.basis);

        const showLabels = ly.basis && norm(image) > 1e-6;
        basisLabels[i].sprite.visible = ly.basis;
        basisImgLabels[i].sprite.visible = showLabels;
        if (ly.basis) {
          basisLabels[i].set(`e${sub(i + 1)}`, new THREE.Vector3(...unit).multiplyScalar(1.18));
          basisImgLabels[i].set(
            `Ae${sub(i + 1)}`,
            new THREE.Vector3(image[0], image[1], image[2]).add(new THREE.Vector3(0.2, 0.25, 0)),
          );
        }
      }

      // vector
      const imageVec = matVec(M, vec);
      setArrow(vecOrig, [vec[0], vec[1], vec[2] ?? 0], norm(vec), ly.vector);
      setArrow(vecImg, [imageVec[0], imageVec[1], imageVec[2] ?? 0], norm(imageVec), ly.vector);
      vecLabel.sprite.visible = ly.vector;
      vecImgLabel.sprite.visible = ly.vector && norm(imageVec) > 1e-6;
      if (ly.vector) {
        vecLabel.set(
          'x',
          new THREE.Vector3(vec[0], vec[1], vec[2] ?? 0).add(new THREE.Vector3(0.15, 0.25, 0)),
        );
        vecImgLabel.set(
          'Ax',
          new THREE.Vector3(imageVec[0], imageVec[1], imageVec[2] ?? 0).add(
            new THREE.Vector3(0.15, 0.25, 0),
          ),
        );
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
          show && Math.abs(item.lam) > 1e-9,
        );
        const tip = new THREE.Vector3(v3[0] * 1.6, v3[1] * 1.6, v3[2] * 1.6);
        tip.add(new THREE.Vector3(0.15, 0.25, 0));
        eigenLabels[i].set(`v, λ=${round(item.lam)}`, tip);
      }

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

      renderer.render(scene, camera);
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
