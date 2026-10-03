import { TRANSITION_MS, useApp } from '../state/store';

/**
 * Transport for the before → after morph, overlaid on the stage:
 *
 *   ▶ |——●——| 45% | Show before | Split
 *
 * Drag the slider to scrub the transformation by hand, press ▶ to replay it
 * from plain space (identity → the active matrix) over TRANSITION_MS, pin
 * the before-ghosts with "Show before", or split the view into side-by-side
 * before | after panels (2D and 3D — the 4×4 view already shows inputs and
 * outputs together, so its Split is hidden).
 */
export function TransitionBar() {
  const t = useApp((s) => s.transT);
  const playing = useApp((s) => s.transPlaying);
  const ghost = useApp((s) => s.ghostAlways);
  const split = useApp((s) => s.splitView);
  const dim = useApp((s) => s.dim);
  const setTransT = useApp((s) => s.setTransT);
  const playTransition = useApp((s) => s.playTransition);
  const pauseTransition = useApp((s) => s.pauseTransition);
  const toggleGhost = useApp((s) => s.toggleGhost);
  const toggleSplit = useApp((s) => s.toggleSplit);

  const pct = Math.round(t * 100);

  return (
    <div className="transition-bar" role="group" aria-label="Before to after transition">
      <button
        type="button"
        className="tb-play"
        aria-label={playing ? 'Pause the transition' : 'Replay the transition from plain space'}
        title={`Replay identity → this matrix (${(TRANSITION_MS / 1000).toFixed(0)}s)`}
        onClick={() => (playing ? pauseTransition() : playTransition())}
      >
        {playing ? '❚❚' : '▶'}
      </button>
      <input
        className="tb-slider"
        type="range"
        min={0}
        max={100}
        step={1}
        value={pct}
        aria-label="Transition progress"
        title="0% = before (plain space) · 100% = after — drag to morph"
        onChange={(e) => setTransT(Number(e.target.value) / 100)}
      />
      <span className="tb-pct">{pct}%</span>
      <button
        type="button"
        className={`tb-toggle${ghost ? ' on' : ''}`}
        aria-pressed={ghost}
        title="Keep the before-state (unit square / identity lattice) under the picture"
        onClick={toggleGhost}
      >
        Show before
      </button>
      {dim !== 4 && (
        <button
          type="button"
          className={`tb-toggle${split ? ' on' : ''}`}
          aria-pressed={split}
          title="Side-by-side: before on the left, the live picture on the right"
          onClick={toggleSplit}
        >
          Split
        </button>
      )}
    </div>
  );
}
