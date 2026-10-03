import { useEffect, useState } from 'react';
import {
  Decomp,
  DecompStep,
  DatasetParams,
  Layers,
  DEFAULT_VECTOR_2,
  DEFAULT_VECTOR_3,
  DEFAULT_VECTOR_4,
  useApp,
} from '../state/store';
import { useDecompFacts } from '../state/hooks';
import { MatrixInput, VectorInput } from './MatrixInput';
import {
  BLOCK_FACTOR_4,
  COMPOSITION_PAIR_2,
  COMPOSITION_PAIR_3,
  COMPOSITION_PAIR_4,
  presetsFor,
} from '../presets';
import { BLOCK_COLORS, schur, splitBlocks } from '../math/blocks';
import { identity, inverse, isIdempotent } from '../math/matrix';

const LAYER_LABELS: Record<keyof Layers, { label: string; hint: string }> = {
  grid: { label: 'Transformed grid', hint: 'How the lattice of space is deformed' },
  basis: { label: 'Basis vectors', hint: 'e₁, e₂ (, e₃) and their images A eᵢ' },
  vector: { label: 'Vector x → Ax', hint: 'Drag the tip in the 2D view' },
  determinant: { label: 'Determinant region', hint: 'Image of the unit square / cube' },
  eigen: { label: 'Eigenvectors', hint: 'Directions that only scale' },
  columnSpace: { label: 'Column space', hint: 'All possible outputs A x' },
  nullSpace: { label: 'Null space', hint: 'Vectors squashed to the origin' },
  transpose: { label: 'Transpose grid Aᵀ', hint: 'Compare A with its reflection Aᵀ' },
  ellipse: { label: 'Circle → ellipse', hint: 'Unit circle mapped by A, with semi-axes σ' },
  levelSets: { label: 'Level sets xᵀAx = c', hint: 'Contours of the quadratic form' },
  form: { label: 'Quadratic form xᵀAx', hint: 'Warm where positive, cool where negative' },
  points: { label: 'Data points (+ mean)', hint: 'The dataset samples and their white mean marker' },
  dataEllipse: { label: 'Data ellipse 1σ · 2σ', hint: 'Gaussian contours — semi-axes √λ along the PCs' },
  residuals: { label: 'Projection residuals', hint: 'Dashed distance from each point to its PC1 reconstruction' },
};

/** The Layers list, split under small sub-headings. */
const LAYER_GROUPS: Array<{ name: string; keys: Array<keyof Layers> }> = [
  { name: 'Space', keys: ['grid', 'basis', 'vector', 'determinant'] },
  { name: 'Structure', keys: ['eigen', 'columnSpace', 'nullSpace', 'transpose'] },
  { name: 'Quadratic form', keys: ['ellipse', 'levelSets', 'form'] },
  { name: 'Data', keys: ['points', 'dataEllipse', 'residuals'] },
];

/** Layers the two-plane 4×4 viewport actually draws, with its own names. */
const DIM4_LAYERS: Partial<Record<keyof Layers, { label: string; hint: string }>> = {
  grid: { label: 'Image lattices', hint: 'Input grids (left) and their block-colored images (right)' },
  basis: { label: 'Basis vectors', hint: 'e₁, e₂ per plane and where each block sends them' },
  vector: { label: 'Vector x → Mx', hint: 'Drag a tip in the input panels to move x' },
  determinant: { label: 'Unit squares', hint: 'Each block’s image of the unit square, with its det' },
};
const DIM4_LAYER_KEYS = new Set<keyof Layers>(['grid', 'basis', 'vector', 'determinant']);

/** Demo chips in the Animate card — each opens one demo's controls. */
const DEMO_TABS = [
  { id: 'inverse', label: 'Inverse' },
  { id: 'composition', label: 'Composition' },
  { id: 'idempotence', label: 'Idempotence' },
  { id: 'schur', label: 'Schur elimination' },
  { id: 'blockMul', label: 'Block multiply' },
  { id: 'spectral', label: 'Spectral' },
  { id: 'svd', label: 'SVD' },
  { id: 'pcaProjection', label: 'PCA projection' },
  { id: 'pcaRotation', label: 'PCA rotation' },
] as const;

type DemoTab = (typeof DEMO_TABS)[number]['id'];

/** Which chips make sense in which dimension (spectral needs n ≤ 3, the */
/** block demos need the 4×4 view, PCA needs a loadable cloud). */
const DIM4_DEMOS = new Set<DemoTab>(['inverse', 'composition', 'idempotence', 'svd', 'schur', 'blockMul']);
const tabsForDim = (dim: 2 | 3 | 4): readonly (typeof DEMO_TABS)[number][] =>
  dim === 4
    ? DEMO_TABS.filter((t) => DIM4_DEMOS.has(t.id as DemoTab))
    : DEMO_TABS.filter((t) => t.id !== 'schur' && t.id !== 'blockMul');

export function ControlPanel() {
  const dim = useApp((s) => s.dim);
  const matrix = useApp((s) => s.matrix);
  const matrixB = useApp((s) => s.matrixB);
  const vector = useApp((s) => s.vector);
  const layers = useApp((s) => s.layers);
  const presetName = useApp((s) => s.presetName);
  const showInverse = useApp((s) => s.showInverse);
  const compActive = useApp((s) => s.compActive);
  const compStep = useApp((s) => s.compStep);
  const idemActive = useApp((s) => s.idemActive);
  const idemStep = useApp((s) => s.idemStep);
  const decomp = useApp((s) => s.decomp);
  const decompStep = useApp((s) => s.decompStep);
  const dataset = useApp((s) => s.dataset);

  const setDim = useApp((s) => s.setDim);
  const setMatrixEntry = useApp((s) => s.setMatrixEntry);
  const setMatrixBEntry = useApp((s) => s.setMatrixBEntry);
  const setMatrixB = useApp((s) => s.setMatrixB);
  const setVectorEntry = useApp((s) => s.setVectorEntry);
  const toggleLayer = useApp((s) => s.toggleLayer);
  const setShowInverse = useApp((s) => s.setShowInverse);
  const setCompActive = useApp((s) => s.setCompActive);
  const setCompStep = useApp((s) => s.setCompStep);
  const setIdemActive = useApp((s) => s.setIdemActive);
  const setIdemStep = useApp((s) => s.setIdemStep);
  const setDecomp = useApp((s) => s.setDecomp);
  const setDecompStep = useApp((s) => s.setDecompStep);
  const setLayers = useApp((s) => s.setLayers);
  const applyMatrices = useApp((s) => s.applyMatrices);
  const transposeA = useApp((s) => s.transposeA);
  const invertAPermanently = useApp((s) => s.invertAPermanently);
  const multiplyAB = useApp((s) => s.multiplyAB);
  const reset = useApp((s) => s.reset);
  const applyDataset = useApp((s) => s.applyDataset);
  const updateDataset = useApp((s) => s.updateDataset);

  // Which demo's controls the Animate card is showing (chips act as tabs).
  const [demoTab, setDemoTab] = useState<DemoTab | null>(null);
  const [playing, setPlaying] = useState(false);
  const [idemPlaying, setIdemPlaying] = useState(false);
  const [decompPlaying, setDecompPlaying] = useState(false);

  // Factorization facts drive the step labels and the disabled-state copy.
  const { spectral, svd } = useDecompFacts();

  const presets = presetsFor(dim);
  const presetMissing = !!presetName && !presets.some((p) => p.name === presetName);

  useEffect(() => {
    if (!playing || !compActive) return;
    const id = window.setInterval(() => {
      setCompStep(((useApp.getState().compStep + 1) % 3) as 0 | 1 | 2);
    }, 1100);
    return () => window.clearInterval(id);
  }, [playing, compActive, setCompStep]);

  useEffect(() => {
    if (!idemPlaying || !idemActive) return;
    const id = window.setInterval(() => {
      setIdemStep(((useApp.getState().idemStep + 1) % 3) as 0 | 1 | 2);
    }, 1100);
    return () => window.clearInterval(id);
  }, [idemPlaying, idemActive, setIdemStep]);

  useEffect(() => {
    if (!decompPlaying || !decomp) return;
    const id = window.setInterval(() => {
      // Block multiply has three steps; the factorization demos have four.
      const modulus = decomp === 'blockMul' ? 3 : 4;
      setDecompStep(((useApp.getState().decompStep + 1) % modulus) as DecompStep);
    }, 1100);
    return () => window.clearInterval(id);
  }, [decompPlaying, decomp, setDecompStep]);

  // Leaving a dimension that doesn't host the selected demo closes its pane
  // (the chips list is dimension-filtered; demoTab is local state).
  useEffect(() => {
    setDemoTab((t) => (t && !tabsForDim(dim).some((x) => x.id === t) ? null : t));
  }, [dim]);

  const canInvert = inverse(matrix) !== null;
  /** Schur elimination needs the top-left block to be invertible. */
  const canSchur = matrix.length === 4 && schur(matrix) !== null;
  const idempotent = isIdempotent(matrix);
  /** What the app calls the main matrix here: M in the 4×4 block view. */
  const AorM = dim === 4 ? 'M' : 'A';
  /** The four 2×2 pieces shown by the block editor (4×4 only). */
  const blocks4 = dim === 4 ? splitBlocks(matrix) : null;

  // Selecting a chip shows that demo's pane; a demo from another pane exits
  // so at most one demo runs (the store setters enforce this anyway).
  // Selecting a factorization chip never *starts* it — its pane's Run button does.
  const selectDemo = (t: DemoTab) => {
    setDemoTab(t);
    const st = useApp.getState();
    const tabDecomp: Decomp | null =
      t === 'spectral'
        ? 'spectral'
        : t === 'svd'
          ? 'svd'
          : t === 'pcaProjection'
            ? 'pcaProjection'
            : t === 'pcaRotation'
              ? 'pcaRotation'
              : t === 'schur'
                ? 'schur'
                : t === 'blockMul'
                  ? 'blockMul'
                  : null;
    if (st.decomp && st.decomp !== tabDecomp) {
      st.setDecomp(null);
      setDecompPlaying(false);
    }
    if (t !== 'inverse' && st.showInverse) st.setShowInverse(false);
    if (t !== 'composition' && st.compActive) {
      st.setCompActive(false);
      setPlaying(false);
    }
    if (t !== 'idempotence' && st.idemActive) {
      st.setIdemActive(false);
      setIdemPlaying(false);
    }
  };

  const loadCompositionExample = () => {
    const pair = dim === 2 ? COMPOSITION_PAIR_2 : dim === 3 ? COMPOSITION_PAIR_3 : COMPOSITION_PAIR_4;
    applyMatrices(pair.A, pair.B, pair.vector, pair.name);
    setCompActive(true);
    setCompStep(0);
    setPlaying(true);
  };

  const startIdempotence = () => {
    setIdemActive(true);
    setIdemStep(0);
    setIdemPlaying(true);
  };

  // Running a factorization demo also switches on the layer that explains it.
  const startSpectral = () => {
    setDecomp('spectral');
    setDecompStep(0);
    setDecompPlaying(true);
    setLayers({ eigen: true });
  };

  const startSvd = () => {
    setDecomp('svd');
    setDecompStep(0);
    setDecompPlaying(true);
    setLayers({ ellipse: true });
  };

  // The PCA demos run on the loaded cloud: they switch on the layers that
  // carry the story — the points + 1σ/2σ ellipse, and either the PC arrows
  // (projection: the frame the points collapse towards) or the rotation
  // payoff without them (the static arrows would trail the rotating cloud).
  const startPcaProjection = () => {
    setDecomp('pcaProjection');
    setDecompStep(0);
    setDecompPlaying(true);
    setLayers({ points: true, dataEllipse: true, residuals: true, eigen: true });
  };

  const startPcaRotation = () => {
    setDecomp('pcaRotation');
    setDecompStep(0);
    setDecompPlaying(true);
    setLayers({ points: true, dataEllipse: true, eigen: false, residuals: false });
  };

  // Block demos (4×4 view): both stories run in the same M → L → U … step
  // machine, so they reuse the generic decomp controls below.
  const startSchur = () => {
    setDecomp('schur');
    setDecompStep(0);
    setDecompPlaying(true);
    setLayers({ grid: true, vector: true });
  };

  const startBlockMul = () => {
    // The payoff only exists when N ≠ I: while the slot still holds the
    // identity, load a factor whose four blocks are all nonzero, so each
    // slot of M·N really shows two products added. A hand-edited N is kept.
    const slotIsIdentity = matrixB.every((row, i) => row.every((v, j) => v === (i === j ? 1 : 0)));
    if (matrixB.length === 4 && slotIsIdentity) setMatrixB(BLOCK_FACTOR_4);
    setDecomp('blockMul');
    setDecompStep(0);
    setDecompPlaying(true);
    setLayers({ grid: true, basis: true, vector: true });
  };

  const doReset = () => {
    reset();
    setDemoTab(null);
    setPlaying(false);
    setIdemPlaying(false);
    setDecompPlaying(false);
  };

  return (
    <div className="panel-stack">
      {/* --------------------------- space & example -------------------------- */}
      <section className="card">
        <h2 className="card-title">Space &amp; example</h2>
        <div className="segmented" role="tablist" aria-label="Dimension">
          <button
            type="button"
            role="tab"
            aria-selected={dim === 2}
            className={dim === 2 ? 'seg active' : 'seg'}
            onClick={() => setDim(2)}
          >
            2 × 2 <span className="seg-sub">plane</span>
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={dim === 3}
            className={dim === 3 ? 'seg active' : 'seg'}
            onClick={() => setDim(3)}
          >
            3 × 3 <span className="seg-sub">space</span>
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={dim === 4}
            className={dim === 4 ? 'seg active' : 'seg'}
            onClick={() => setDim(4)}
          >
            4 × 4 <span className="seg-sub">blocks</span>
          </button>
        </div>
        <select
          className="select"
          value={presetName ?? ''}
          onChange={(e) => {
            const p = presetsFor(dim).find((x) => x.name === e.target.value);
            if (!p) return;
            if (p.dataset) {
              applyDataset(p.name, p.dataset.kind, p.dataset.params);
              return;
            }
            const fallback =
              dim === 2 ? DEFAULT_VECTOR_2.slice() : dim === 3 ? DEFAULT_VECTOR_3.slice() : DEFAULT_VECTOR_4.slice();
            const x = p.vector ?? fallback;
            applyMatrices(
              p.matrix.map((r) => r.slice()),
              identity(dim),
              x.slice(),
              p.name,
            );
          }}
        >
          <option value="" disabled>
            Custom matrix…
          </option>
          {presetMissing && <option value={presetName!}>{presetName}</option>}
          <optgroup label="Transformations">
            {presets
              .filter((p) => !p.dataset)
              .map((p) => (
                <option key={p.name} value={p.name}>
                  {p.name}
                </option>
              ))}
          </optgroup>
          {presets.some((p) => p.dataset) && (
            <optgroup label="PCA datasets">
              {presets
                .filter((p) => p.dataset)
                .map((p) => (
                  <option key={p.name} value={p.name}>
                    {p.name}
                  </option>
                ))}
            </optgroup>
          )}
        </select>
        <p className="muted small">
          {dim === 4 ? (
            <>
              Pick a block structure — the four 2×2 blocks <span className="mono">A B / C D</span>{' '}
              are edited below and drawn as two coupled planes.
            </>
          ) : (
            <>
              Pick a classic transformation, or a point cloud from{' '}
              <b>PCA datasets</b> — its covariance matrix S becomes Matrix A.
            </>
          )}
        </p>
      </section>

      {/* ---------------------------- point cloud --------------------------- */}
      {dim !== 4 && (
        <section className="card">
          <h2 className="card-title">
            Point cloud
            {dataset && <span className="badge">n = {dataset.points.length}</span>}
          </h2>
          {dataset ? (
            <>
              <p className="muted small">
                Dragging a slider regenerates the seeded cloud — Matrix A follows as its
                covariance <span className="mono">S = Cov(x)</span>.
              </p>
              <DatasetSliders params={dataset.params} dim={dim} onChange={updateDataset} />
            </>
          ) : (
            <p className="muted small">
              No cloud loaded. Choose a preset under <b>PCA datasets</b> above to study real
              samples: the cloud appears in the viewport and Matrix A holds its covariance.
            </p>
          )}
        </section>
      )}

      {/* ----------------------------- matrix & vector ------------------------ */}
      <section className="card">
        <h2 className="card-title">
          {dim === 4 ? (
            <>
              Matrix <span className="mono accent-a">M</span> · blocks{' '}
              <span className="mono" style={{ color: BLOCK_COLORS.A }}>A</span>{' '}
              <span className="mono" style={{ color: BLOCK_COLORS.B }}>B</span>{' / '}
              <span className="mono" style={{ color: BLOCK_COLORS.C }}>C</span>{' '}
              <span className="mono" style={{ color: BLOCK_COLORS.D }}>D</span>
            </>
          ) : (
            <>
              Matrix <span className="mono accent-a">A</span> · Vector{' '}
              <span className="mono accent-v">x</span>
            </>
          )}
          {showInverse && <span className="badge">showing {dim === 4 ? 'M⁻¹' : 'A⁻¹'}</span>}
        </h2>
        {dim === 4 && blocks4 ? (
          <div className="block-editor">
            {(['A', 'B', 'C', 'D'] as const).map((letter) => {
              const rowOffset = letter === 'A' || letter === 'B' ? 0 : 2;
              const colOffset = letter === 'A' || letter === 'C' ? 0 : 2;
              return (
                <div className="block-cell" key={letter}>
                  <span className="block-letter" style={{ color: BLOCK_COLORS[letter] }}>
                    {letter}
                  </span>
                  <MatrixInput
                    matrix={blocks4[letter]}
                    onEntry={setMatrixEntry}
                    accent={BLOCK_COLORS[letter]}
                    ariaLabel={`Block ${letter}`}
                    rowOffset={rowOffset}
                    colOffset={colOffset}
                  />
                </div>
              );
            })}
          </div>
        ) : (
          <MatrixInput matrix={matrix} onEntry={setMatrixEntry} ariaLabel="Matrix A" />
        )}
        <div className="button-row">
          <button type="button" className="btn" onClick={transposeA}>
            {dim === 4 ? 'M' : 'A'}
            <sup>T</sup>
          </button>
          <button
            type="button"
            className="btn"
            onClick={invertAPermanently}
            disabled={!canInvert}
            title={canInvert ? `Replace ${dim === 4 ? 'M' : 'A'} with its inverse` : `${dim === 4 ? 'M' : 'A'} is singular — no inverse exists`}
          >
            {dim === 4 ? 'M⁻¹' : 'A⁻¹'}
          </button>
        </div>
        <div className="subhead">
          Vector <span className="mono accent-v">x</span>
        </div>
        <VectorInput
          vector={vector}
          onEntry={setVectorEntry}
          ariaLabel="Vector x"
          labels={dim === 4 ? ['x', 'y', 'u', 'v'] : undefined}
        />
        <p className="muted small">
          {dim === 4 ? (
            <>
              x₁ = (x, y) lives in plane 1, x₂ = (u, v) in plane 2 — drag either tip in the
              left panels of the viewport.
            </>
          ) : (
            <>In the 2D view you can also drag the amber handle to move <span className="mono">x</span>.</>
          )}
        </p>
      </section>

      {/* -------------------------------- layers ------------------------------ */}
      <section className="card">
        <h2 className="card-title">Layers</h2>
        {(dim === 4
          ? LAYER_GROUPS.map((g) => ({
              name: g.name,
              keys: g.keys.filter((k) => DIM4_LAYER_KEYS.has(k)),
            })).filter((g) => g.keys.length > 0)
          : LAYER_GROUPS
        ).map(({ name, keys }) => (
          <div key={name}>
            <div className="subhead">{name}</div>
            <ul className="toggle-list">
              {keys.map((key) => {
                const labels = (dim === 4 && DIM4_LAYERS[key]) || LAYER_LABELS[key];
                return (
                  <li key={key}>
                    <label className="toggle" title={labels.hint}>
                      <input
                        type="checkbox"
                        checked={layers[key]}
                        onChange={() => toggleLayer(key)}
                      />
                      <span className="toggle-track" aria-hidden="true" />
                      <span className="toggle-text">{labels.label}</span>
                    </label>
                  </li>
                );
              })}
            </ul>
          </div>
        ))}
      </section>

      {/* ------------------------------- animate ------------------------------ */}
      <section className="card">
        <h2 className="card-title">Animate</h2>
        <div className="demo-chips">
          {tabsForDim(dim).map(({ id, label }) => (
            <button
              key={id}
              type="button"
              className={demoTab === id ? 'chip active' : 'chip'}
              aria-pressed={demoTab === id}
              onClick={() => selectDemo(id)}
            >
              {label}
            </button>
          ))}
        </div>

        {demoTab === null && (
          <p className="muted small">
            Pick a demo to step through its transformation right here in the viewport.
          </p>
        )}

        {/* ------------------------------ inverse ----------------------------- */}
        {demoTab === 'inverse' &&
          (!showInverse ? (
            <>
              <p className="muted small">
                Apply {AorM}, then watch {AorM}⁻¹ undo it — every vector returns to where it
                started.
              </p>
              <button
                type="button"
                className="btn btn-primary full"
                onClick={() => setShowInverse(true)}
              >
                Show {AorM}⁻¹ instead of {AorM}
              </button>
            </>
          ) : (
            <>
              <p className="muted small">
                Space is now drawn through <span className="mono">{AorM}⁻¹</span> — the exact
                undo of {AorM}.
              </p>
              <button
                type="button"
                className="btn full"
                onClick={() => setShowInverse(false)}
              >
                Back to drawing {AorM}
              </button>
            </>
          ))}

        {/* --------------------------- composition ---------------------------- */}
        {demoTab === 'composition' &&
          (!compActive ? (
            <>
              <p className="muted small">
                Apply two matrices one after the other and watch that the order matters.
              </p>
              <div className="button-row">
                <button type="button" className="btn btn-primary" onClick={loadCompositionExample}>
                  Start demo
                </button>
                <button
                  type="button"
                  className="btn"
                  onClick={() => {
                    setCompActive(true);
                    setCompStep(0);
                    setPlaying(false);
                  }}
                >
                  Use my matrices
                </button>
              </div>
            </>
          ) : (
            <>
              <p className="muted small">
                {dim === 4 ? (
                  <>
                    Matrix <span className="mono accent-b">N</span> goes first, then{' '}
                    <span className="mono accent-a">M</span> — the result is M·N.
                  </>
                ) : (
                  <>
                    Matrix <span className="mono accent-b">B</span> goes first, then{' '}
                    <span className="mono accent-a">A</span> — the result is A·B.
                  </>
                )}
              </p>
              <MatrixInput
                matrix={matrixB}
                onEntry={setMatrixBEntry}
                accent="var(--accent-b)"
                ariaLabel={dim === 4 ? 'Matrix N' : 'Matrix B'}
              />
              <div className="steps" role="group" aria-label="Composition steps">
                {(dim === 4 ? ['I', 'N', 'M·N'] : ['I', 'B', 'A·B']).map((label, i) => (
                  <button
                    key={label}
                    type="button"
                    className={compStep === i ? 'step active' : 'step'}
                    onClick={() => {
                      setPlaying(false);
                      setCompStep(i as 0 | 1 | 2);
                    }}
                  >
                    <span className="step-index">{i + 1}</span> {label}
                  </button>
                ))}
              </div>
              <div className="button-row">
                <button
                  type="button"
                  className={playing ? 'btn btn-primary on' : 'btn btn-primary'}
                  onClick={() => setPlaying(!playing)}
                >
                  {playing ? '❚❚ Pause' : '▶ Play'}
                </button>
                <button
                  type="button"
                  className="btn"
                  onClick={() => {
                    setCompActive(false);
                    setPlaying(false);
                  }}
                >
                  Exit
                </button>
              </div>
              <div className="button-row">
                <button
                  type="button"
                  className="btn"
                  onClick={() => multiplyAB('AB')}
                  title={dim === 4 ? 'Replace M with M·N' : 'Replace A with A·B'}
                >
                  {dim === 4 ? 'M·N' : 'A·B'}
                </button>
                <button
                  type="button"
                  className="btn"
                  onClick={() => multiplyAB('BA')}
                  title={dim === 4 ? 'Replace M with N·M' : 'Replace A with B·A'}
                >
                  {dim === 4 ? 'N·M' : 'B·A'}
                </button>
              </div>
            </>
          ))}

        {/* ---------------------------- idempotence --------------------------- */}
        {demoTab === 'idempotence' &&
          (!idemActive ? (
            <>
              <p className="muted small">
                {idempotent
                  ? `This matrix is idempotent: applying it twice is the same as applying it once. Step I → ${AorM} → ${AorM}² and watch the picture refuse to move a second time.`
                  : `An idempotent matrix is a projection: ${AorM}² = ${AorM}, so a second application changes nothing. Try a “Projection…” preset, then run I → ${AorM} → ${AorM}² to see it settle.`}
              </p>
              <button type="button" className="btn btn-primary full" onClick={startIdempotence}>
                Animate I → {AorM} → {AorM}²
              </button>
            </>
          ) : (
            <>
              <p className="muted small">
                Step three computes {AorM}·{AorM}. If the grid stops moving between steps 2 and
                3, then {AorM}² = {AorM}.
              </p>
              <div className="steps" role="group" aria-label="Idempotence steps">
                {['I', AorM, `${AorM}²`].map((label, i) => (
                  <button
                    key={label}
                    type="button"
                    className={idemStep === i ? 'step active' : 'step'}
                    onClick={() => {
                      setIdemPlaying(false);
                      setIdemStep(i as 0 | 1 | 2);
                    }}
                  >
                    <span className="step-index">{i + 1}</span> {label}
                  </button>
                ))}
              </div>
              <div className="button-row">
                <button
                  type="button"
                  className={idemPlaying ? 'btn btn-primary on' : 'btn btn-primary'}
                  onClick={() => setIdemPlaying(!idemPlaying)}
                >
                  {idemPlaying ? '❚❚ Pause' : '▶ Play'}
                </button>
                <button
                  type="button"
                  className="btn"
                  onClick={() => {
                    setIdemActive(false);
                    setIdemPlaying(false);
                  }}
                >
                  Exit
                </button>
              </div>
            </>
          ))}

        {/* ----------------------------- schur ------------------------------ */}
        {demoTab === 'schur' &&
          (decomp !== 'schur' ? (
            !canSchur ? (
              <>
                <p className="warn-text small">
                  Schur elimination solves through the top-left block{' '}
                  <span className="mono">A</span> — make block A invertible (nonzero det) first.
                </p>
                <button type="button" className="btn full" disabled>
                  Run the Schur elimination demo
                </button>
              </>
            ) : (
              <>
                <p className="muted small">
                  Partition <span className="mono">M = [A B; C D]</span>, clear the lower-left
                  block with the block shear <span className="mono">L = [I 0; CA⁻¹ I]</span>, and
                  what's left is upper-block-triangular with the Schur complement{' '}
                  <span className="mono">S = D − CA⁻¹B</span> in the corner: M = L·U and{' '}
                  <span className="mono">det M = det A · det S</span>.
                </p>
                <button type="button" className="btn btn-primary full" onClick={startSchur}>
                  Run the Schur elimination demo
                </button>
              </>
            )
          ) : (
            <>
              <p className="muted small">
                Step four puts the factors back together: det L = 1, so the determinant of M
                factors as det A · det S. The Partitioned matrix card shows both numbers.
              </p>
              <div className="steps steps-4" role="group" aria-label="Schur elimination steps">
                {(['M', 'L', 'U', 'M = L·U'] as const).map((label, i) => (
                  <button
                    key={label}
                    type="button"
                    className={decompStep === i ? 'step active' : 'step'}
                    onClick={() => {
                      setDecompPlaying(false);
                      setDecompStep(i as DecompStep);
                    }}
                  >
                    <span className="step-index">{i + 1}</span> {label}
                  </button>
                ))}
              </div>
              <div className="button-row">
                <button
                  type="button"
                  className={decompPlaying ? 'btn btn-primary on' : 'btn btn-primary'}
                  onClick={() => setDecompPlaying(!decompPlaying)}
                >
                  {decompPlaying ? '❚❚ Pause' : '▶ Play'}
                </button>
                <button
                  type="button"
                  className="btn"
                  onClick={() => {
                    setDecomp(null);
                    setDecompPlaying(false);
                  }}
                >
                  Exit
                </button>
              </div>
            </>
          ))}

        {/* --------------------------- block multiply ----------------------- */}
        {demoTab === 'blockMul' &&
          (decomp !== 'blockMul' ? (
            <>
              <p className="muted small">
                Partitioned matrices multiply blockwise: with{' '}
                <span className="mono">M = [A B; C D]</span> and{' '}
                <span className="mono">N = [E F; G H]</span>, each block of the product is a sum
                of two 2×2 products — <span className="mono">AE + BG, AF + BH, CE + DG, CF + DH</span>.
              </p>
              <div className="subhead">Second factor N (matrix slot B)</div>
              <MatrixInput
                matrix={matrixB}
                onEntry={setMatrixBEntry}
                accent="var(--accent-b)"
                ariaLabel="Matrix N"
              />
              <button type="button" className="btn btn-primary full" onClick={startBlockMul}>
                Run the block multiply demo
              </button>
            </>
          ) : (
            <>
              <p className="muted small">
                Step through N, M and the product M·N — the Partitioned matrix card lists the
                four block products as they add up.
              </p>
              <div className="subhead">Second factor N (matrix slot B)</div>
              <MatrixInput
                matrix={matrixB}
                onEntry={setMatrixBEntry}
                accent="var(--accent-b)"
                ariaLabel="Matrix N"
              />
              <div className="steps" role="group" aria-label="Block multiply steps">
                {(['N', 'M', 'M·N'] as const).map((label, i) => (
                  <button
                    key={label}
                    type="button"
                    className={decompStep === i ? 'step active' : 'step'}
                    onClick={() => {
                      setDecompPlaying(false);
                      setDecompStep(i as DecompStep);
                    }}
                  >
                    <span className="step-index">{i + 1}</span> {label}
                  </button>
                ))}
              </div>
              <div className="button-row">
                <button
                  type="button"
                  className={decompPlaying ? 'btn btn-primary on' : 'btn btn-primary'}
                  onClick={() => setDecompPlaying(!decompPlaying)}
                >
                  {decompPlaying ? '❚❚ Pause' : '▶ Play'}
                </button>
                <button
                  type="button"
                  className="btn"
                  onClick={() => {
                    setDecomp(null);
                    setDecompPlaying(false);
                  }}
                >
                  Exit
                </button>
              </div>
            </>
          ))}

        {/* ------------------------------ spectral --------------------------- */}
        {demoTab === 'spectral' &&
          (decomp !== 'spectral' ? (
            spectral.kind === 'defective' || spectral.kind === 'complex' ? (
              <>
                <p className="warn-text small">
                  {spectral.kind === 'defective'
                    ? 'Not diagonalizable: a repeated eigenvalue has only one independent direction (a shear), so no basis P of eigenvectors spans the space.'
                    : 'Complex eigenvalues: no real direction keeps its own line (think rotation), so there is no real eigenbasis to change into.'}
                </p>
                <button type="button" className="btn full" disabled>
                  Run the spectral demo
                </button>
              </>
            ) : (
              <>
                <p className="muted small">
                  {spectral.orthogonal
                    ? 'A is symmetric: Qᵀ lines the space up with its eigen-directions, Λ scales along them, and Q rotates back — A = QΛQᵀ.'
                    : 'A = P D P⁻¹: P⁻¹ rewrites vectors in the eigenbasis, D scales each coordinate by λ, then P maps the result back.'}
                </p>
                <button type="button" className="btn btn-primary full" onClick={startSpectral}>
                  Run the spectral demo
                </button>
              </>
            )
          ) : (
            <>
              <p className="muted small">
                Step four recomputes {spectral.orthogonal ? 'QΛQᵀ' : 'P D P⁻¹'} —
                if the grid snaps back onto A, the factors multiply out exactly.
              </p>
              {(spectral.kind === 'defective' || spectral.kind === 'complex') && (
                <p className="warn-text small">
                  A stopped being diagonalizable mid-demo — every step now shows A
                  itself. Exit, or bring back a diagonalizable matrix.
                </p>
              )}
              <div className="steps steps-4" role="group" aria-label="Spectral steps">
                {(spectral.orthogonal
                  ? ['I', 'Qᵀ', 'ΛQᵀ', 'A']
                  : ['I', 'P⁻¹', 'D·P⁻¹', 'A']
                ).map((label, i) => (
                  <button
                    key={label}
                    type="button"
                    className={decompStep === i ? 'step active' : 'step'}
                    onClick={() => {
                      setDecompPlaying(false);
                      setDecompStep(i as DecompStep);
                    }}
                  >
                    <span className="step-index">{i + 1}</span> {label}
                  </button>
                ))}
              </div>
              <div className="button-row">
                <button
                  type="button"
                  className={decompPlaying ? 'btn btn-primary on' : 'btn btn-primary'}
                  onClick={() => setDecompPlaying(!decompPlaying)}
                >
                  {decompPlaying ? '❚❚ Pause' : '▶ Play'}
                </button>
                <button
                  type="button"
                  className="btn"
                  onClick={() => {
                    setDecomp(null);
                    setDecompPlaying(false);
                  }}
                >
                  Exit
                </button>
              </div>
            </>
          ))}

        {/* -------------------------------- svd ------------------------------ */}
        {demoTab === 'svd' &&
          (decomp !== 'svd' ? (
            <>
              <p className="muted small">
                Any matrix splits into three simple moves: Vᵀ turns the input, Σ
                stretches along the axes, U points the result — A = UΣVᵀ. It works
                even when A is singular, where σₙ = 0.
              </p>
              <button
                type="button"
                className="btn btn-primary full"
                onClick={startSvd}
                disabled={!svd}
              >
                Run the SVD demo
              </button>
            </>
          ) : (
            <>
              <p className="muted small">
                Step four recomputes UΣVᵀ — if the grid snaps back onto A, the
                factors multiply out exactly.
              </p>
              <div className="steps steps-4" role="group" aria-label="SVD steps">
                {['I', 'Vᵀ', 'ΣVᵀ', 'A'].map((label, i) => (
                  <button
                    key={label}
                    type="button"
                    className={decompStep === i ? 'step active' : 'step'}
                    onClick={() => {
                      setDecompPlaying(false);
                      setDecompStep(i as DecompStep);
                    }}
                  >
                    <span className="step-index">{i + 1}</span> {label}
                  </button>
                ))}
              </div>
              <div className="button-row">
                <button
                  type="button"
                  className={decompPlaying ? 'btn btn-primary on' : 'btn btn-primary'}
                  onClick={() => setDecompPlaying(!decompPlaying)}
                >
                  {decompPlaying ? '❚❚ Pause' : '▶ Play'}
                </button>
                <button
                  type="button"
                  className="btn"
                  onClick={() => {
                    setDecomp(null);
                    setDecompPlaying(false);
                  }}
                >
                  Exit
                </button>
              </div>
            </>
          ))}

        {/* -------------------------- pca projection ------------------------ */}
        {demoTab === 'pcaProjection' &&
          (!dataset ? (
            <>
              <p className="warn-text small">
                PCA needs data: no point cloud is loaded. Pick a preset under{' '}
                <b>PCA datasets</b> in Space &amp; example first.
              </p>
              <button type="button" className="btn full" disabled>
                Run the projection demo
              </button>
            </>
          ) : decomp !== 'pcaProjection' ? (
            <>
              <p className="muted small">
                Center the data, read the PCs off the covariance matrix S, then collapse
                every point onto PC1 with the rank-1 projector{' '}
                <span className="mono">P₁ = v₁v₁ᵀ</span> — the dashed residuals are what
                PC1 throws away.
              </p>
              <button
                type="button"
                className="btn btn-primary full"
                onClick={startPcaProjection}
              >
                Run the PCA projection demo
              </button>
            </>
          ) : (
            <>
              <p className="muted small">
                Step four applies <span className="mono">P₁ = v₁v₁ᵀ</span>: det = 0, the
                grid flattens onto the PC1 line and the rose points snap to it — the
                dropped distance is the reconstruction error.
              </p>
              <div className="steps steps-4" role="group" aria-label="PCA projection steps">
                {['data', 'x − x̄', 'PC1, PC2', 'P₁'].map((label, i) => (
                  <button
                    key={label}
                    type="button"
                    className={decompStep === i ? 'step active' : 'step'}
                    onClick={() => {
                      setDecompPlaying(false);
                      setDecompStep(i as DecompStep);
                    }}
                  >
                    <span className="step-index">{i + 1}</span> {label}
                  </button>
                ))}
              </div>
              <div className="button-row">
                <button
                  type="button"
                  className={decompPlaying ? 'btn btn-primary on' : 'btn btn-primary'}
                  onClick={() => setDecompPlaying(!decompPlaying)}
                >
                  {decompPlaying ? '❚❚ Pause' : '▶ Play'}
                </button>
                <button
                  type="button"
                  className="btn"
                  onClick={() => {
                    setDecomp(null);
                    setDecompPlaying(false);
                  }}
                >
                  Exit
                </button>
              </div>
            </>
          ))}

        {/* --------------------------- pca rotation ------------------------- */}
        {demoTab === 'pcaRotation' &&
          (!dataset ? (
            <>
              <p className="warn-text small">
                PCA needs data: no point cloud is loaded. Pick a preset under{' '}
                <b>PCA datasets</b> in Space &amp; example first.
              </p>
              <button type="button" className="btn full" disabled>
                Run the rotation demo
              </button>
            </>
          ) : decomp !== 'pcaRotation' ? (
            <>
              <p className="muted small">
                Rewrite the cloud in its own principal coordinates: <span className="mono">Qᵀ</span>{' '}
                lines PC1 up with the x-axis, then <span className="mono">Λ</span> scales each
                axis by its variance — <span className="mono">S = QΛQᵀ</span> rebuilt, with the
                data along for the ride.
              </p>
              <button
                type="button"
                className="btn btn-primary full"
                onClick={startPcaRotation}
              >
                Run the PCA rotation demo
              </button>
            </>
          ) : (
            <>
              <p className="muted small">
                Step three applies <span className="mono">ΛQᵀ</span> — grid and cloud end up
                axis-aligned, spread along each axis by λ, the variance of the data there.
              </p>
              <div className="steps steps-4" role="group" aria-label="PCA rotation steps">
                {['data', 'x − x̄', 'Qᵀ', 'ΛQᵀ'].map((label, i) => (
                  <button
                    key={label}
                    type="button"
                    className={decompStep === i ? 'step active' : 'step'}
                    onClick={() => {
                      setDecompPlaying(false);
                      setDecompStep(i as DecompStep);
                    }}
                  >
                    <span className="step-index">{i + 1}</span> {label}
                  </button>
                ))}
              </div>
              <div className="button-row">
                <button
                  type="button"
                  className={decompPlaying ? 'btn btn-primary on' : 'btn btn-primary'}
                  onClick={() => setDecompPlaying(!decompPlaying)}
                >
                  {decompPlaying ? '❚❚ Pause' : '▶ Play'}
                </button>
                <button
                  type="button"
                  className="btn"
                  onClick={() => {
                    setDecomp(null);
                    setDecompPlaying(false);
                  }}
                >
                  Exit
                </button>
              </div>
            </>
          ))}
      </section>

      {/* -------------------------------- reset ------------------------------- */}
      <section className="card">
        <button type="button" className="btn btn-ghost full" onClick={doReset}>
          Reset everything
        </button>
      </section>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Point cloud card                                                    */
/* ------------------------------------------------------------------ */

interface SliderDef {
  key: keyof DatasetParams;
  label: string;
  min: number;
  max: number;
  step: number;
  fmt: (v: number) => string;
}

const f2 = (v: number) => v.toFixed(2);

const DATASET_SLIDERS: SliderDef[] = [
  { key: 'rho', label: 'Correlation ρ', min: -0.95, max: 0.95, step: 0.05, fmt: f2 },
  { key: 'theta', label: 'Rotation θ', min: 0, max: 180, step: 5, fmt: (v) => `${v}°` },
  { key: 'sx', label: 'Spread x', min: 0.3, max: 3, step: 0.05, fmt: f2 },
  { key: 'sy', label: 'Spread y', min: 0.3, max: 4.5, step: 0.05, fmt: f2 },
  { key: 'sz', label: 'Spread z', min: 0.3, max: 3, step: 0.05, fmt: f2 },
  { key: 'noise', label: 'Noise', min: 0, max: 1, step: 0.05, fmt: f2 },
];

/**
 * Live regeneration sliders — each drag reseeds the *same* gaussian sample
 * (the seed depends only on kind + dim), so the cloud morphs smoothly instead
 * of reshuffling, and Matrix A follows as its covariance S.
 */
function DatasetSliders({
  params,
  dim,
  onChange,
}: {
  params: DatasetParams;
  dim: 2 | 3;
  onChange: (patch: Partial<DatasetParams>) => void;
}) {
  return (
    <div className="sliders">
      {DATASET_SLIDERS.filter((s) => dim === 3 || s.key !== 'sz').map((s) => (
        <label className="slider" key={s.key}>
          <span className="slider-head">
            <span>{s.label}</span>
            <span className="slider-value mono">{s.fmt(params[s.key])}</span>
          </span>
          <input
            type="range"
            min={s.min}
            max={s.max}
            step={s.step}
            value={params[s.key]}
            onChange={(e) => onChange({ [s.key]: Number(e.target.value) } as Partial<DatasetParams>)}
          />
        </label>
      ))}
    </div>
  );
}
