/**
 * Smoke tests for the linear-algebra core.
 * Run with: npm test  (bundles the TS sources with esbuild first)
 */

import {
  det,
  inverse,
  matMul,
  rank,
  nullSpaceBasis,
  columnSpaceBasis,
  identity,
  norm,
} from '../.test-build/matrix.js';
import { eigen } from '../.test-build/eigen.js';

let fails = 0;
const ok = (label, cond, extra = '') => {
  if (cond) console.log('  ok   ' + label);
  else {
    fails++;
    console.log('  FAIL ' + label + '  ' + extra);
  }
};
const near = (a, b, t = 1e-9) => Math.abs(a - b) < t;
const matNear = (A, B, t = 1e-9) =>
  A.every((r, i) => r.every((x, j) => near(x, B[i][j], t)));

console.log('\ndeterminant');
ok('det 2x2', near(det([[2, 1], [1, 2]]), 3));
ok('det 3x3', near(det([[1, 2, 3], [4, 5, 6], [7, 8, 10]]), -3));
ok('det of singular', near(det([[1, 2], [2, 4]]), 0));

console.log('\ninverse');
const A2 = [[2, 1], [1, 2]];
ok('2x2: A · A⁻¹ = I', matNear(matMul(A2, inverse(A2)), identity(2)));
const A3 = [[4, 1, 0], [0, 3, 0], [1, 0, 2]];
ok('3x3: A · A⁻¹ = I', matNear(matMul(A3, inverse(A3)), identity(3)));
const B3 = [[1, 2, 3], [0, 1, 4], [5, 6, 0]];
ok('3x3 (second): A · A⁻¹ = I', matNear(matMul(B3, inverse(B3)), identity(3)));
ok('singular 2x2 → null', inverse([[1, 2], [2, 4]]) === null);
ok('singular 3x3 → null', inverse([[1, 2, 3], [2, 4, 6], [3, 6, 9]]) === null);

console.log('\nrank / subspaces');
ok('rank full', rank(A2) === 2);
ok('rank 1', rank([[1, 2], [2, 4]]) === 1);
const ns = nullSpaceBasis([[1, 2], [2, 4]]);
ok('null(A) is 1-dimensional', ns.length === 1);
ok(
  'null(A) basis really maps to 0',
  ns.every((v) => norm([1 * v[0] + 2 * v[1], 2 * v[0] + 4 * v[1]]) < 1e-9),
  JSON.stringify(ns),
);
const ns3 = nullSpaceBasis([[1, 0, 0], [0, 1, 0], [0, 0, 0]]);
ok('3D null space = z-axis', ns3.length === 1 && near(Math.abs(ns3[0][2]), 1), JSON.stringify(ns3));
ok(
  'column space of rank-2 3D matrix',
  columnSpaceBasis([[1, 0, 0], [0, 1, 0], [0, 0, 0]]).length === 2,
);

console.log('\neigenvalues (2x2)');
const e1 = eigen(A2);
ok('values are 3 and 1', [3, 1].every((v) => e1.some((x) => near(x.real, v))), JSON.stringify(e1));
ok('eigenvectors are unit length', e1.every((x) => x.vector && near(norm(x.vector), 1)));
const e2 = eigen([[0, -1], [1, 0]]);
ok('quarter turn → complex ±i', e2.length === 2 && e2.every((x) => Math.abs(x.imag) > 0 && x.vector === null), JSON.stringify(e2));
const e3 = eigen([[1, 1], [0, 1]]);
ok('defective matrix → single eigenvector', e3.length === 1 && near(e3[0].real, 1), JSON.stringify(e3));

console.log('\neigenvalues (3x3)');
const e4 = eigen(A3);
ok('values are 4, 3, 2', [4, 3, 2].every((v) => e4.some((x) => near(x.real, v) && Math.abs(x.imag) < 1e-9)), JSON.stringify(e4.map((x) => x.real)));
ok('three real eigenvectors', e4.filter((x) => x.vector).length === 3);
const resid = e4
  .filter((x) => x.vector)
  .map((x) => {
    const v = x.vector;
    const Av = A3.map((r) => r[0] * v[0] + r[1] * v[1] + r[2] * v[2]);
    return Math.hypot(Av[0] - x.real * v[0], Av[1] - x.real * v[1], Av[2] - x.real * v[2]);
  });
ok('A v = λ v for each pair', resid.every((r) => r < 1e-7), JSON.stringify(resid));
const e5 = eigen([[0, 0, 1], [1, 0, 0], [0, 1, 0]]);
ok('permutation → complex pair + 1', e5.some((x) => Math.abs(x.imag) > 1e-9), JSON.stringify(e5.map((x) => [x.real, x.imag])));

console.log('\nmultiplicity / eigenspaces');
const id2 = eigen(identity(2));
ok('identity: one value, multiplicity 2', id2.length === 1 && id2[0].multiplicity === 2, JSON.stringify(id2));
ok('identity: whole plane is an eigenspace', id2[0].fullSpace && id2[0].vectors.length === 2, JSON.stringify(id2[0]?.vectors));
const sh = eigen([[1, 1], [0, 1]]);
ok('shear: repeated value but one direction', sh[0].multiplicity === 2 && sh[0].vectors.length === 1 && !sh[0].fullSpace, JSON.stringify(sh[0]));
const sym3 = eigen([[2, 1, 0], [1, 2, 0], [0, 0, 1]]);
ok('3D: λ=1 has multiplicity 2 with two directions', sym3.some((e) => near(e.real, 1) && e.multiplicity === 2 && e.vectors.length === 2), JSON.stringify(sym3.map((e) => [e.real, e.multiplicity, e.vectors.length])));
ok('3D: λ=3 has multiplicity 1', sym3.some((e) => near(e.real, 3) && e.multiplicity === 1), JSON.stringify(sym3.map((e) => [e.real, e.multiplicity])));
const id3 = eigen(identity(3));
ok('identity3: multiplicity 3, full space', id3.length === 1 && id3[0].multiplicity === 3 && id3[0].fullSpace, JSON.stringify(id3.map((e) => [e.real, e.multiplicity, e.vectors.length])));

console.log(fails === 0 ? '\n✓ all math checks passed\n' : `\n✗ ${fails} check(s) failed\n`);
process.exit(fails ? 1 : 0);
