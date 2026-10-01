import { Suspense, lazy } from 'react';
import { Layers, useApp } from './state/store';
import { ControlPanel } from './components/ControlPanel';
import { InfoPanel } from './components/InfoPanel';
import { Viewport2D } from './components/Viewport2D';

// three.js is by far the largest dependency — only fetch it once the user
// actually switches to the 3×3 view.
const Viewport3D = lazy(() => import('./components/Viewport3D'));

export default function App() {
  const dim = useApp((s) => s.dim);
  const layers = useApp((s) => s.layers);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">
            |A|
          </span>
          <div>
            <h1>Matrix Transformations</h1>
            <p className="tagline">
              see what a matrix <em>does</em> to space —{' '}
              {dim === 2 ? '2×2 in the plane' : '3×3 in three dimensions'}
            </p>
          </div>
        </div>
        <Legend dim={dim} layers={layers} />
      </header>

      <main className="layout">
        <aside className="sidebar">
          <ControlPanel />
        </aside>

        <section className="stage">
          {dim === 2 ? (
            <Viewport2D />
          ) : (
            <Suspense
              fallback={<div className="viewport-loading">Loading the 3D view…</div>}
            >
              <Viewport3D />
            </Suspense>
          )}
        </section>

        <aside className="inspector">
          <InfoPanel />
        </aside>
      </main>
    </div>
  );
}

function Legend({ dim, layers }: { dim: 2 | 3; layers: Layers }) {
  const items: Array<{ color: string; label: string; dashed?: boolean }> = [];
  if (layers.basis) {
    items.push({ color: '#f87171', label: 'e₁ → Ae₁' });
    items.push({ color: '#4ade80', label: 'e₂ → Ae₂' });
    if (dim === 3) items.push({ color: '#60a5fa', label: 'e₃ → Ae₃' });
  }
  if (layers.vector) items.push({ color: '#fbbf24', label: 'x → Ax' });
  if (layers.determinant) items.push({ color: '#22c55e', label: 'det region' });
  if (layers.eigen) items.push({ color: '#c084fc', label: 'eigenvector' });
  if (layers.columnSpace) items.push({ color: '#22d3ee', label: 'Col(A)' });
  if (layers.nullSpace) items.push({ color: '#fb923c', label: 'Null(A)' });
  if (layers.transpose) items.push({ color: '#facc15', label: 'Aᵀ grid' });

  return (
    <ul className="legend" aria-label="Colour legend">
      {items.map((it) => (
        <li key={it.label}>
          <span className="swatch" style={{ background: it.color, color: it.color }} />
          <span className="legend-label">{it.label}</span>
        </li>
      ))}
    </ul>
  );
}
