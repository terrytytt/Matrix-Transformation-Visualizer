import { useEffect, useState } from 'react';
import { Layers, DEFAULT_VECTOR_2, DEFAULT_VECTOR_3, useApp } from '../state/store';
import { MatrixInput, VectorInput } from './MatrixInput';
import { COMPOSITION_PAIR_2, COMPOSITION_PAIR_3, presetsFor } from '../presets';
import { identity, inverse } from '../math/matrix';

const LAYER_LABELS: Array<{ key: keyof Layers; label: string; hint: string }> = [
  { key: 'grid', label: 'Transformed grid', hint: 'How the lattice of space is deformed' },
  { key: 'basis', label: 'Basis vectors', hint: 'e₁, e₂ (, e₃) and their images A eᵢ' },
  { key: 'vector', label: 'Vector x → Ax', hint: 'Drag the tip in the 2D view' },
  { key: 'determinant', label: 'Determinant region', hint: 'Image of the unit square / cube' },
  { key: 'eigen', label: 'Eigenvectors', hint: 'Directions that only scale' },
  { key: 'columnSpace', label: 'Column space', hint: 'All possible outputs A x' },
  { key: 'nullSpace', label: 'Null space', hint: 'Vectors squashed to the origin' },
  { key: 'transpose', label: 'Transpose grid Aᵀ', hint: 'Compare A with its reflection Aᵀ' },
];

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

  const setDim = useApp((s) => s.setDim);
  const setMatrixEntry = useApp((s) => s.setMatrixEntry);
  const setMatrixBEntry = useApp((s) => s.setMatrixBEntry);
  const setVectorEntry = useApp((s) => s.setVectorEntry);
  const toggleLayer = useApp((s) => s.toggleLayer);
  const setShowInverse = useApp((s) => s.setShowInverse);
  const setCompActive = useApp((s) => s.setCompActive);
  const setCompStep = useApp((s) => s.setCompStep);
  const applyMatrices = useApp((s) => s.applyMatrices);
  const transposeA = useApp((s) => s.transposeA);
  const invertAPermanently = useApp((s) => s.invertAPermanently);
  const multiplyAB = useApp((s) => s.multiplyAB);
  const reset = useApp((s) => s.reset);

  const [playing, setPlaying] = useState(false);
  const presets = presetsFor(dim);
  const presetMissing = !!presetName && !presets.some((p) => p.name === presetName);

  useEffect(() => {
    if (!playing || !compActive) return;
    const id = window.setInterval(() => {
      setCompStep(((useApp.getState().compStep + 1) % 3) as 0 | 1 | 2);
    }, 1100);
    return () => window.clearInterval(id);
  }, [playing, compActive, setCompStep]);

  const canInvert = inverse(matrix) !== null;

  const loadCompositionExample = () => {
    const pair = dim === 2 ? COMPOSITION_PAIR_2 : COMPOSITION_PAIR_3;
    applyMatrices(pair.A, pair.B, pair.vector, pair.name);
    setCompActive(true);
    setCompStep(0);
    setPlaying(true);
  };

  return (
    <div className="panel-stack">
      {/* ----------------------------- dimension ----------------------------- */}
      <section className="card">
        <h2 className="card-title">Space</h2>
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
        </div>
      </section>

      {/* ------------------------------- presets ------------------------------ */}
      <section className="card">
        <h2 className="card-title">Example</h2>
        <select
          className="select"
          value={presetName ?? ''}
          onChange={(e) => {
            const p = presetsFor(dim).find((x) => x.name === e.target.value);
            if (!p) return;
            const x = p.vector ?? (dim === 2 ? DEFAULT_VECTOR_2.slice() : DEFAULT_VECTOR_3.slice());
            applyMatrices(
              p.matrix.map((r) => r.slice()),
              identity(dim),
              x.slice(),
              p.name,
            );
            setCompActive(false);
            setShowInverse(false);
          }}
        >
          <option value="" disabled>
            Custom matrix…
          </option>
          {presetMissing && <option value={presetName!}>{presetName}</option>}
          {presets.map((p) => (
            <option key={p.name} value={p.name}>
              {p.name}
            </option>
          ))}
        </select>
        <p className="muted small">
          Pick a classic transformation, then tweak any number to watch the picture change live.
        </p>
      </section>

      {/* ------------------------------- matrix ------------------------------- */}
      <section className="card">
        <h2 className="card-title">
          Matrix <span className="mono accent-a">A</span>
          {showInverse && <span className="badge">showing A⁻¹</span>}
        </h2>
        <MatrixInput matrix={matrix} onEntry={setMatrixEntry} ariaLabel="Matrix A" />
        <div className="button-row">
          <button type="button" className="btn" onClick={transposeA}>
            A<sup>T</sup>
          </button>
          <button
            type="button"
            className="btn"
            onClick={invertAPermanently}
            disabled={!canInvert}
            title={canInvert ? 'Replace A with its inverse' : 'A is singular — no inverse exists'}
          >
            A⁻¹
          </button>
          <button
            type="button"
            className="btn"
            onClick={() => multiplyAB('AB')}
            disabled={!compActive}
            title="Replace A with A·B"
          >
            A·B
          </button>
          <button
            type="button"
            className="btn"
            onClick={() => multiplyAB('BA')}
            disabled={!compActive}
            title="Replace A with B·A"
          >
            B·A
          </button>
        </div>
        <button
          type="button"
          className={showInverse ? 'btn btn-toggle on' : 'btn btn-toggle'}
          onClick={() => setShowInverse(!showInverse)}
        >
          {showInverse ? '✓ ' : ''}Show A⁻¹ instead of A
        </button>
      </section>

      {/* -------------------------------- vector ------------------------------ */}
      <section className="card">
        <h2 className="card-title">
          Vector <span className="mono accent-v">x</span>
        </h2>
        <VectorInput vector={vector} onEntry={setVectorEntry} ariaLabel="Vector x" />
        <p className="muted small">
          In the 2D view you can also drag the amber handle to move <span className="mono">x</span>.
        </p>
      </section>

      {/* ----------------------------- composition ---------------------------- */}
      <section className="card">
        <h2 className="card-title">Composition · A·B</h2>
        {!compActive ? (
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
              Matrix <span className="mono accent-b">B</span> goes first, then{' '}
              <span className="mono accent-a">A</span> — the result is A·B.
            </p>
            <MatrixInput
              matrix={matrixB}
              onEntry={setMatrixBEntry}
              accent="var(--accent-b)"
              ariaLabel="Matrix B"
            />
            <div className="steps" role="group" aria-label="Composition steps">
              {(['I', 'B', 'A·B'] as const).map((label, i) => (
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
          </>
        )}
      </section>

      {/* -------------------------------- layers ------------------------------ */}
      <section className="card">
        <h2 className="card-title">Layers</h2>
        <ul className="toggle-list">
          {LAYER_LABELS.map(({ key, label, hint }) => (
            <li key={key}>
              <label className="toggle" title={hint}>
                <input
                  type="checkbox"
                  checked={layers[key]}
                  onChange={() => toggleLayer(key)}
                />
                <span className="toggle-track" aria-hidden="true" />
                <span className="toggle-text">{label}</span>
              </label>
            </li>
          ))}
        </ul>
      </section>

      <section className="card">
        <button type="button" className="btn btn-ghost full" onClick={reset}>
          Reset everything
        </button>
      </section>
    </div>
  );
}
