import { useEffect, useRef, useState } from 'react';
import type { Matrix, Vector } from '../math/matrix';

const fmt = (v: number): string => {
  if (!Number.isFinite(v)) return '0';
  if (Number.isInteger(v)) return String(v);
  const r = Math.round(v * 1e6) / 1e6;
  return String(r);
};

const sig = (m: number[][]): string => m.flat().join(',');

function chunk(flat: number[], n: number): number[][] {
  return Array.from({ length: n }, (_, i) => flat.slice(i * n, i * n + n));
}

/**
 * Editable n×n grid of numbers.
 *
 * Text state is local so typing `-`, `1.` etc. works; it re-syncs from the
 * store only when the matrix changed from the outside (preset, transpose…).
 */
export function MatrixInput({
  matrix,
  onEntry,
  accent = 'var(--accent-a)',
  ariaLabel = 'Matrix',
}: {
  matrix: Matrix;
  onEntry: (i: number, j: number, v: number) => void;
  accent?: string;
  ariaLabel?: string;
}) {
  const n = matrix.length;
  const [text, setText] = useState<string[]>(() => matrix.flat().map(fmt));
  const lastPushed = useRef<string>(sig(matrix));

  useEffect(() => {
    const current = sig(matrix);
    if (current !== lastPushed.current) {
      lastPushed.current = current;
      setText(matrix.flat().map(fmt));
    }
  }, [matrix]);

  const update = (idx: number, raw: string) => {
    setText((prev) => {
      const next = prev.slice();
      next[idx] = raw;
      return next;
    });

    const trimmed = raw.trim();
    if (trimmed === '') return;
    const parsed = Number(trimmed);
    if (!Number.isFinite(parsed)) return;

    const flat = matrix.flat();
    flat[idx] = parsed;
    lastPushed.current = sig(chunk(flat, n));
    onEntry(Math.floor(idx / n), idx % n, parsed);
  };

  const blur = (idx: number) => {
    const raw = text[idx];
    if (raw === undefined) return;
    const trimmed = raw.trim();
    const parsed = Number(trimmed);
    if (trimmed === '' || !Number.isFinite(parsed)) {
      setText(matrix.flat().map(fmt));
    }
  };

  return (
    <div className="matrix-editor" style={{ ['--accent' as string]: accent }}>
      <div
        className="matrix-grid"
        style={{ gridTemplateColumns: `repeat(${n}, minmax(0, 1fr))` }}
        role="group"
        aria-label={ariaLabel}
      >
        {text.map((value, idx) => (
          <input
            key={idx}
            className="cell"
            type="text"
            inputMode="decimal"
            spellCheck={false}
            aria-label={`${ariaLabel} row ${Math.floor(idx / n) + 1} column ${(idx % n) + 1}`}
            value={value}
            onChange={(e) => update(idx, e.target.value)}
            onBlur={() => blur(idx)}
          />
        ))}
      </div>
      <span className="bracket bracket-left" aria-hidden="true" />
      <span className="bracket bracket-right" aria-hidden="true" />
    </div>
  );
}

export function VectorInput({
  vector,
  onEntry,
  labels = ['x', 'y', 'z'],
  ariaLabel = 'Vector',
}: {
  vector: Vector;
  onEntry: (i: number, v: number) => void;
  labels?: string[];
  ariaLabel?: string;
}) {
  const [text, setText] = useState<string[]>(() => vector.map(fmt));
  const lastPushed = useRef<string>(vector.join(','));

  useEffect(() => {
    const current = vector.join(',');
    if (current !== lastPushed.current) {
      lastPushed.current = current;
      setText(vector.map(fmt));
    }
  }, [vector]);

  const update = (idx: number, raw: string) => {
    setText((prev) => {
      const next = prev.slice();
      next[idx] = raw;
      return next;
    });
    const trimmed = raw.trim();
    if (trimmed === '') return;
    const parsed = Number(trimmed);
    if (!Number.isFinite(parsed)) return;

    const flat = vector.slice();
    flat[idx] = parsed;
    lastPushed.current = flat.join(',');
    onEntry(idx, parsed);
  };

  const blur = (idx: number) => {
    const raw = text[idx];
    if (raw === undefined) return;
    const trimmed = raw.trim();
    const parsed = Number(trimmed);
    if (trimmed === '' || !Number.isFinite(parsed)) setText(vector.map(fmt));
  };

  return (
    <div className="vector-editor">
      {text.map((value, idx) => (
        <label key={idx} className="vector-cell">
          <span className="vector-axis">{labels[idx]}</span>
          <input
            className="cell"
            type="text"
            inputMode="decimal"
            spellCheck={false}
            aria-label={`${ariaLabel} component ${labels[idx]}`}
            value={value}
            onChange={(e) => update(idx, e.target.value)}
            onBlur={() => blur(idx)}
          />
        </label>
      ))}
    </div>
  );
}
