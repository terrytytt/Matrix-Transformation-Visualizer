import { Suspense, lazy } from 'react';
import { Dataset, Decomp, Dim, Layers, useApp } from './state/store';
import { ControlPanel } from './components/ControlPanel';
import { InfoPanel } from './components/InfoPanel';
import { Viewport2D } from './components/Viewport2D';
import { TransitionBar } from './components/TransitionBar';
import { BLOCK_COLORS } from './math/blocks';

// three.js is by far the largest dependency — only fetch it once the user
// actually switches to the 3×3 view; the four-panel block view is fetched
// the same way for the 4×4 mode.
const Viewport3D = lazy(() => import('./components/Viewport3D'));
const ViewportBlocks = lazy(() => import('./components/ViewportBlocks'));

export default function App() {
  const dim = useApp((s) => s.dim);
  const layers = useApp((s) => s.layers);
  const decomp = useApp((s) => s.decomp);
  const dataset = useApp((s) => s.dataset);

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
              {dim === 2
                ? '2×2 in the plane'
                : dim === 3
                  ? '3×3 in three dimensions'
                  : '4×4 as two coupled planes'}
            </p>
          </div>
        </div>
      </header>

      <main className="layout">
        <aside className="sidebar">
          <ControlPanel />
        </aside>

        <section className="stage">
          {dim === 4 ? (
            <Suspense
              fallback={<div className="viewport-loading">Loading the block view…</div>}
            >
              <ViewportBlocks />
            </Suspense>
          ) : dim === 2 ? (
            <Viewport2D />
          ) : (
            <Suspense
              fallback={<div className="viewport-loading">Loading the 3D view…</div>}
            >
              <Viewport3D />
            </Suspense>
          )}
          <TransitionBar />
          <Legend dim={dim} layers={layers} decomp={decomp} dataset={dataset} />
        </section>

        <aside className="inspector">
          <InfoPanel />
        </aside>
      </main>
    </div>
  );
}

function Legend({
  dim,
  layers,
  decomp,
  dataset,
}: {
  dim: Dim;
  layers: Layers;
  decomp: Decomp | null;
  dataset: Dataset | null;
}) {
  const compActive = useApp((s) => s.compActive);
  const items: Array<{ color: string; label: string; dashed?: boolean; glow?: string }> = [];
  if (dim === 4) {
    // Two-plane view: colors follow the four blocks of M = [A B; C D].
    if (layers.grid || layers.basis) {
      items.push({ color: BLOCK_COLORS.A, label: 'A · plane 1 → 1' });
      items.push({ color: BLOCK_COLORS.B, label: 'B · plane 2 → 1' });
      items.push({ color: BLOCK_COLORS.C, label: 'C · plane 1 → 2' });
      items.push({ color: BLOCK_COLORS.D, label: 'D · plane 2 → 2' });
    }
    if (layers.vector) {
      items.push({ color: '#fbbf24', label: 'x₁, x₂ inputs' });
      items.push({ color: '#f8fafc', label: 'y₁, y₂ = Mx' });
    }
    if (layers.determinant) items.push({ color: '#a3e635', label: 'unit-square images' });
    return <LegendList items={items} />;
  }
  if (layers.basis) {
    items.push({ color: '#f87171', label: 'e₁ → Ae₁' });
    items.push({ color: '#4ade80', label: 'e₂ → Ae₂' });
    if (dim === 3) items.push({ color: '#60a5fa', label: 'e₃ → Ae₃' });
  }
  if (layers.vector) {
    if (compActive) {
      // The demo draws three arrows, so key all three colours: the dashed
      // amber path and the violet hop-1 image the second hop departs from.
      items.push({ color: '#fbbf24', dashed: true, label: 'x → Bx → A(Bx)' });
      items.push({ color: '#a78bfa', label: 'Bx · first hop' });
    } else {
      items.push({ color: '#fbbf24', label: 'x → Ax' });
    }
  }
  if (layers.determinant) items.push({ color: '#22c55e', label: 'det region' });
  if (layers.eigen) items.push({ color: '#c084fc', label: 'eigenvector' });
  if (layers.columnSpace) items.push({ color: '#22d3ee', label: 'Col(A)' });
  if (layers.nullSpace) items.push({ color: '#fb923c', label: 'Null(A)' });
  if (layers.transpose) items.push({ color: '#facc15', label: 'Aᵀ grid' });
  if (layers.ellipse)
    items.push({ color: '#e879f9', label: dim === 2 ? 'circle → ellipse' : 'ball → ellipsoid' });
  if (layers.levelSets) items.push({ color: '#2dd4bf', label: 'xᵀAx = c' });
  if (layers.form)
    items.push({
      // Warm → cool gradient; the glow color is the midpoint violet-amber.
      color: 'linear-gradient(90deg, #fbbf24, #818cf8)',
      label: 'xᵀAx sign',
      glow: '#c9a227',
    });
  if (decomp === 'svd' || decomp === 'spectral')
    items.push({
      color: '#f8fafc',
      dashed: true,
      label: decomp === 'svd' ? 'vᵢ → σᵢuᵢ' : 'vᵢ → λᵢvᵢ',
    });
  if (dataset) {
    if (layers.points) items.push({ color: '#fb7185', label: 'data points' });
    if (layers.dataEllipse) items.push({ color: '#fda4af', label: '1σ · 2σ ellipse' });
    if (layers.residuals) items.push({ color: '#94a3b8', label: 'recon. residual' });
  }

  return <LegendList items={items} />;
}

function LegendList({ items }: { items: Array<{ color: string; label: string; dashed?: boolean; glow?: string }> }) {
  return (
    <ul className="legend" aria-label="Colour legend">
      {items.map((it) => (
        <li key={it.label}>
          <span
            className="swatch"
            style={
              it.dashed
                ? {
                    background: `repeating-linear-gradient(90deg, ${it.color} 0 5px, transparent 5px 9px)`,
                    color: 'transparent',
                  }
                : { background: it.color, color: it.glow ?? it.color }
            }
          />
          <span className="legend-label">{it.label}</span>
        </li>
      ))}
    </ul>
  );
}
