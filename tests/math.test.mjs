/**
 * Smoke tests for the linear-algebra core.
 * Run with: npm test  (bundles the TS sources with esbuild first)
 */

import {
  det,
  inverse,
  isApproxEqual,
  matMul,
  matVec,
  rank,
  nullSpaceBasis,
  columnSpaceBasis,
  identity,
  isIdempotent,
  isSymmetric,
  norm,
  quadraticContours,
  symmetricPart,
  transpose,
} from '../.test-build/matrix.js';
import {
  definiteness,
  eigen,
  ellipseAxes,
  singularValues,
  spectralFactors,
  svdFactors,
  symmetricBasis,
} from '../.test-build/eigen.js';

import {
  covarianceOf,
  datasetCovariance,
  generateCloud,
  matrixSqrt,
  meanOf,
  outerProject,
  pcaOf,
  projectTo,
} from '../.test-build/pca.js';

import {
  blockDiagonal,
  blockLowerUnit,
  blockProducts,
  blockStructure,
  blockTranspose,
  blockUpper,
  conditionalVariance2,
  detFactorization,
  eigenUnion,
  joinBlocks,
  schur,
  splitBlocks,
  traceBlocks,
} from '../.test-build/blocks.js';

import { blendFromIdentity, easeInOutCubic } from '../.test-build/lerp.js';
import { BLOCK_VARS, chainEquations, rowEquation, systemEquations } from '../.test-build/equations.js';

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

console.log('\nidempotence (A² = A)');
const P2 = [[1, 0], [0, 0]];
ok('projection onto x-axis is idempotent', isIdempotent(P2));
ok('identity is idempotent', isIdempotent(identity(3)));
ok('projection onto line y = x', isIdempotent([[0.5, 0.5], [0.5, 0.5]]));
ok('oblique projection [[1,0],[2,0]]', isIdempotent([[1, 0], [2, 0]]));
ok('3D projection onto plane z = x + y', isIdempotent([[1, 0, 0], [0, 1, 0], [1, 1, 0]]));
ok(
  '3D projection onto diagonal line',
  isIdempotent([[1 / 3, 1 / 3, 1 / 3], [1 / 3, 1 / 3, 1 / 3], [1 / 3, 1 / 3, 1 / 3]]),
);
ok('rotation is NOT idempotent', !isIdempotent([[0, -1], [1, 0]]));
ok('shear is NOT idempotent', !isIdempotent([[1, 1], [0, 1]]));
ok('scale is NOT idempotent', !isIdempotent([[2, 0], [0, 2]]));
ok('rank-1 collapse [[1,2],[2,4]] is NOT idempotent', !isIdempotent([[1, 2], [2, 4]]));
ok(
  'idempotent ⇒ A · A really equals A',
  matMul(P2, P2).every((r, i) => r.every((x, j) => near(x, P2[i][j]))),
);

console.log('\nsymmetric part');
ok('[[1,2],[3,4]] is not symmetric', !isSymmetric([[1, 2], [3, 4]]));
ok('[[1,2],[2,1]] is symmetric', isSymmetric([[1, 2], [2, 1]]));
ok(
  'symmetricPart averages the off-diagonals',
  JSON.stringify(symmetricPart([[1, 2], [3, 4]])) === JSON.stringify([[1, 2.5], [2.5, 4]]),
  JSON.stringify(symmetricPart([[1, 2], [3, 4]])),
);
ok('symmetricPart of a symmetric matrix is itself', isSymmetric(symmetricPart([[1, 2], [3, 4]])));
ok(
  '3D symmetricPart',
  JSON.stringify(symmetricPart([[1, 2, 3], [2, 5, 6], [3, 6, 9]])) ===
    JSON.stringify([[1, 2, 3], [2, 5, 6], [3, 6, 9]]),
);

console.log('\nsingular values / ellipse axes');
const shS = singularValues([[1, 1], [0, 1]]);
ok('shear → golden ratio 1.618 / 0.618', near(shS[0], (1 + Math.sqrt(5)) / 2, 1e-9) && near(shS[1], (Math.sqrt(5) - 1) / 2, 1e-9), JSON.stringify(shS));
ok('shear: σ₁σ₂ = |det| = 1', near(shS[0] * shS[1], Math.abs(det([[1, 1], [0, 1]])), 1e-9));
const syS = singularValues([[2, 1], [1, 2]]);
ok('symmetric ⇒ σ = |λ| = {3, 1}', near(syS[0], 3) && near(syS[1], 1), JSON.stringify(syS));
const d3S = singularValues([[2, 0, 0], [0, 1, 0], [0, 0, 0.5]]);
ok('3D diag(2,1,0.5) → [2, 1, 0.5]', d3S.length === 3 && near(d3S[0], 2) && near(d3S[1], 1) && near(d3S[2], 0.5), JSON.stringify(d3S));
const r1S = singularValues([[1, 2], [2, 4]]);
ok('rank-1 → σ₂ = 0', near(r1S[1], 0, 1e-9), JSON.stringify(r1S));
ok('rotation → σ = [1, 1]', singularValues([[0, -1], [1, 0]]).every((s) => near(s, 1)), JSON.stringify(singularValues([[0, -1], [1, 0]])));
const shAxes = ellipseAxes([[1, 1], [0, 1]]);
ok('ellipse axes: tips have length σ', shAxes.every((a) => near(Math.hypot(...a.tip), a.sigma, 1e-9)), JSON.stringify(shAxes));
ok('ellipse axes: perpendicular tips', near(shAxes[0].tip[0] * shAxes[1].tip[0] + shAxes[0].tip[1] * shAxes[1].tip[1], 0, 1e-9), JSON.stringify(shAxes));

console.log('\nsymmetricBasis (level-set axes)');
const sb = symmetricBasis(symmetricPart([[1, 1], [0, 1]]));
ok('shear S → two directions', sb.length === 2, JSON.stringify(sb));
ok('directions are unit length', sb.every((p) => near(norm(p.dir), 1)));
ok(
  'directions are orthonormal',
  near(sb[0].dir[0] * sb[1].dir[0] + sb[0].dir[1] * sb[1].dir[1], 0, 1e-9),
  JSON.stringify(sb),
);
ok('eigenvalues of S sorted descending', sb[0].lam >= sb[1].lam, JSON.stringify(sb));
ok('quarter turn → S = 0 basis still spans', symmetricBasis(symmetricPart([[0, -1], [1, 0]])).length === 2);

console.log('\nlevel sets xᵀAx = c');
const ell = quadraticContours(3, 1, 1, 8);
ok('ellipse case → one closed loop of 9 points', ell.length === 1 && ell[0].length === 9, JSON.stringify(ell.map((b) => b.length)));
ok(
  'ellipse points satisfy 3y₁² + y₂² = 1',
  ell[0].every((p) => near(3 * p[0] * p[0] + p[1] * p[1], 1, 1e-9)),
  JSON.stringify(ell[0]),
);
ok('wrong-sign c → empty', quadraticContours(3, 1, -1).length === 0);
const hyp = quadraticContours(1, -1, 1, 8, 5);
ok('indefinite → two hyperbola branches', hyp.length === 2, JSON.stringify(hyp.map((b) => b.length)));
ok(
  'hyperbola points satisfy y₁² − y₂² = 1',
  hyp.every((b) => b.every((p) => near(p[0] * p[0] - p[1] * p[1], 1, 1e-9))),
  JSON.stringify(hyp),
);
const lines = quadraticContours(1, 0, 4, 8, 5);
ok('one λ = 0 → two parallel lines', lines.length === 2 && lines.every((b) => b.length === 2), JSON.stringify(lines));
ok(
  'parallel lines sit at y₁ = ±2',
  lines.every((b) => b.every((p) => near(Math.abs(p[0]), 2))),
  JSON.stringify(lines),
);
ok('S = 0 → no contours', quadraticContours(0, 0, 1).length === 0);

console.log('\ndefiniteness (xᵀAx)');
ok('Symmetric stretch → positive definite', definiteness([[2, 1], [1, 2]]).kind === 'positiveDefinite');
ok('Scale ×2 → positive definite', definiteness([[2, 0], [0, 2]]).kind === 'positiveDefinite');
const defShear = definiteness([[1, 1], [0, 1]]);
ok('shear (not symmetric) → positive definite', defShear.kind === 'positiveDefinite');
ok('shear → viaSymmetricPart flag set', defShear.viaSymmetricPart === true);
ok(
  'shear: S = [[1,.5],[.5,1]] → λ = 1.5, 0.5',
  near(defShear.lambdas[0], 1.5) && near(defShear.lambdas[1], 0.5),
  JSON.stringify(defShear.lambdas),
);
const c45 = Math.SQRT1_2;
const defRot = definiteness([
  [c45, -c45],
  [c45, c45],
]);
ok('45° rotation (complex λ) → positive definite', defRot.kind === 'positiveDefinite');
ok(
  '45° rotation: λ(S) = cos45° ≈ 0.707',
  near(defRot.lambdas[0], c45) && near(defRot.lambdas[1], c45),
  JSON.stringify(defRot.lambdas),
);
ok('Negative definite preset → negative definite', definiteness([[-2, -1], [-1, -2]]).kind === 'negativeDefinite');
ok(
  'Negative definite: minors alternate (-, +)',
  (() => {
    const m = definiteness([[-2, -1], [-1, -2]]).minors;
    return near(m[0], -2) && near(m[1], 3);
  })(),
  JSON.stringify(definiteness([[-2, -1], [-1, -2]]).minors),
);
ok('Flip + stretch → indefinite', definiteness([[1, 2], [2, 1]]).kind === 'indefinite');
ok('diag(1,-1) → indefinite', definiteness([[1, 0], [0, -1]]).kind === 'indefinite');
ok('projection → positive semidefinite', definiteness([[1, 0], [0, 0]]).kind === 'positiveSemidefinite');
ok('-projection → negative semidefinite', definiteness([[-1, 0], [0, 0]]).kind === 'negativeSemidefinite');
ok('Quarter turn → zero form', definiteness([[0, -1], [1, 0]]).kind === 'zero');
ok('zero matrix → zero form', definiteness([[0, 0], [0, 0]]).kind === 'zero');
ok('Sylvester: PD ⇒ all minors > 0', definiteness([[2, 1], [1, 2]]).minors.every((m) => m > 0));
ok('Sylvester: Flip+stretch ⇒ Δ₂ < 0', definiteness([[1, 2], [2, 1]]).minors[1] < 0);
ok('3D diag(2,1,0.5) → positive definite', definiteness([[2, 0, 0], [0, 1, 0], [0, 0, 0.5]]).kind === 'positiveDefinite');
ok(
  '3D diag(2,1,-1) → indefinite',
  definiteness([[2, 0, 0], [0, 1, 0], [0, 0, -1]]).kind === 'indefinite',
);
ok(
  '3D diag(1,1,0) → positive semidefinite',
  definiteness([[1, 0, 0], [0, 1, 0], [0, 0, 0]]).kind === 'positiveSemidefinite',
);
ok(
  '3D reflection (xy-plane) → indefinite',
  definiteness([[1, 0, 0], [0, 1, 0], [0, 0, -1]]).kind === 'indefinite',
);
ok('3D symmetric A → viaSymmetricPart false', definiteness([[2, 1, 0], [1, 2, 0], [0, 0, 1]]).viaSymmetricPart === false);
ok(
  '3D Sylvester: PD ⇒ all three minors > 0',
  definiteness([[2, 1, 0], [1, 2, 0], [0, 0, 1]]).minors.every((m) => m > 0),
  JSON.stringify(definiteness([[2, 1, 0], [1, 2, 0], [0, 0, 1]]).minors),
);

/* ---------------- factorizations: SVD & spectral ---------------- */

const symStretch = [[2, 1], [1, 2]];
const svd1 = svdFactors(symStretch);
ok('svdFactors: U Σ Vᵀ = A (symmetric stretch)', !!svd1 && isApproxEqual(matMul(matMul(svd1.U, svd1.sigma), svd1.Vt), symStretch, 1e-9));
ok('svdFactors: σ descending', !!svd1 && svd1.s[0] >= svd1.s[1], JSON.stringify(svd1?.s));
ok('svdFactors: residual ≈ 0', !!svd1 && svd1.residual < 1e-9, String(svd1?.residual));
ok('svdFactors: σ₁σ₂ = |det A|', !!svd1 && near(svd1.s[0] * svd1.s[1], 3), JSON.stringify(svd1?.s));

const shear = [[1, 1], [0, 1]];
const svd2 = svdFactors(shear);
ok('svdFactors: shear reconstructs', !!svd2 && isApproxEqual(matMul(matMul(svd2.U, svd2.sigma), svd2.Vt), shear, 1e-9));
ok('svdFactors: shear σ descending', !!svd2 && svd2.s[0] >= svd2.s[1], JSON.stringify(svd2?.s));

const rank1 = [[0.5, 0.5], [0.5, 0.5]];
const svd3 = svdFactors(rank1);
ok('svdFactors: singular A → σ₂ = 0', !!svd3 && near(svd3.s[1], 0), JSON.stringify(svd3?.s));
ok('svdFactors: singular A still reconstructs', !!svd3 && isApproxEqual(matMul(matMul(svd3.U, svd3.sigma), svd3.Vt), rank1, 1e-9));

const svdA3 = [[2, 1, 0], [1, 2, 0], [0, 0, 1]];
const svd4 = svdFactors(svdA3);
ok('svdFactors: 3×3 reconstructs', !!svd4 && isApproxEqual(matMul(matMul(svd4.U, svd4.sigma), svd4.Vt), svdA3, 1e-9));
ok('svdFactors: 3×3 three descending σ', !!svd4 && svd4.s.length === 3 && svd4.s[0] >= svd4.s[1] && svd4.s[1] >= svd4.s[2], JSON.stringify(svd4?.s));

const spec1 = spectralFactors(symStretch);
ok('spectral: symmetric → orthogonal', spec1.kind === 'orthogonal' && spec1.orthogonal === true);
ok('spectral: symmetric Q D Qᵀ = A', !!spec1.P && !!spec1.D && isApproxEqual(matMul(matMul(spec1.P, spec1.D), spec1.Pinv), symStretch, 1e-9));
ok('spectral: Q orthonormal (QᵀQ = I)', !!spec1.P && isApproxEqual(matMul(transpose(spec1.P), spec1.P), identity(2), 1e-9));
ok('spectral: symmetric λ = 3, 1', near(spec1.lambdas[0], 3) && near(spec1.lambdas[1], 1), JSON.stringify(spec1.lambdas));
ok('spectral: residual ≈ 0', spec1.residual < 1e-9, String(spec1.residual));

const spec2 = spectralFactors([[1, 1], [0, 2]]);
ok('spectral: [[1,1],[0,2]] → general', spec2.kind === 'general' && spec2.orthogonal === false);
ok('spectral: P D P⁻¹ = A (general)', !!spec2.P && !!spec2.D && !!spec2.Pinv && isApproxEqual(matMul(matMul(spec2.P, spec2.D), spec2.Pinv), [[1, 1], [0, 2]], 1e-9));
ok('spectral: general λ = 1, 2', spec2.lambdas.length === 2 && spec2.lambdas.includes(1) && spec2.lambdas.includes(2), JSON.stringify(spec2.lambdas));

const spec3 = spectralFactors(shear);
ok('spectral: shear → defective', spec3.kind === 'defective' && spec3.P === null && spec3.Pinv === null);

const c45s = Math.SQRT1_2;
const spec4 = spectralFactors([[c45s, -c45s], [c45s, c45s]]);
ok('spectral: 45° rotation → complex', spec4.kind === 'complex' && spec4.P === null);

const spec5 = spectralFactors([[0, 0], [0, 0]]);
ok('spectral: zero matrix → orthogonal, λ = 0, 0', spec5.kind === 'orthogonal' && spec5.lambdas.every((l) => near(l, 0)) && spec5.residual < 1e-12);

const spec6 = spectralFactors([[3, 1, 0], [0, 2, 0], [0, 0, 1]]);
ok('spectral: 3×3 distinct λ → general', spec6.kind === 'general');
ok('spectral: 3×3 P D P⁻¹ = A', !!spec6.P && !!spec6.D && !!spec6.Pinv && isApproxEqual(matMul(matMul(spec6.P, spec6.D), spec6.Pinv), [[3, 1, 0], [0, 2, 0], [0, 0, 1]], 1e-9));

const spec7 = spectralFactors([[1, 0, 0], [0, 1, 0], [0, 0, -1]]);
ok('spectral: 3D reflection → orthogonal, λ = 1, 1, -1', spec7.kind === 'orthogonal' && spec7.lambdas[0] === 1 && spec7.lambdas[1] === 1 && spec7.lambdas[2] === -1, JSON.stringify(spec7.lambdas));

/* ---------------- PCA: covariance, components, generators ---------------- */

const pcaRaw = [[0, 0], [2, 0], [0, 2], [2, 2]];
ok('pca: mean of the 4-square sample = [1,1]', matNear([meanOf(pcaRaw)], [[1, 1]]));
const pcaCov = covarianceOf(pcaRaw);
ok('pca: covariance = (4/3)·I', matNear(pcaCov, [[4 / 3, 0], [0, 4 / 3]]), JSON.stringify(pcaCov));

// Exact sample: z over {±1}² pushed through L = [[1,0],[ρ,√(1−ρ²)]] ⇒
// sample covariance = (4/3)·[[1,ρ],[ρ,1]] ⇒ PC1 is exactly the diagonal.
const pcaRho = 0.6;
const pcaL2 = [[1, 0], [pcaRho, Math.sqrt(1 - pcaRho * pcaRho)]];
const pcaZs = [[1, 1], [1, -1], [-1, 1], [-1, -1]];
const pcaBuilt = pcaZs.map((z) => matVec(pcaL2, z));
const pcaRes = pcaOf(pcaBuilt);
const pcaV1 = pcaRes.vectors[0];
const pcaDiag = Math.SQRT1_2;
ok(
  'pca: exact correlated sample → PC1 = diagonal',
  Math.abs(Math.abs(pcaV1[0] * pcaDiag + pcaV1[1] * pcaDiag) - 1) < 1e-9,
  JSON.stringify(pcaV1),
);
ok('pca: λ descending', pcaRes.lambdas[0] >= pcaRes.lambdas[1], JSON.stringify(pcaRes.lambdas));
ok('pca: λ ≈ (4/3)·(1±ρ)', near(pcaRes.lambdas[0], (4 / 3) * (1 + pcaRho)) && near(pcaRes.lambdas[1], (4 / 3) * (1 - pcaRho)), JSON.stringify(pcaRes.lambdas));
ok('pca: explained shares sum to 1', near(pcaRes.explained.reduce((s, e) => s + e, 0), 1));
ok('pca: cumulative ends at 1', near(pcaRes.cumulative[pcaRes.cumulative.length - 1], 1));
ok('pca: √λ = std', pcaRes.std.every((s, i) => near(s, Math.sqrt(pcaRes.lambdas[i]))));
ok('pca: trace = Σλ', near(pcaRes.trace, pcaRes.lambdas.reduce((s, l) => s + l, 0)));
ok('pca: recon error (k=1) = trace − λ₁ in 2D', near(pcaRes.recon1, pcaRes.trace - pcaRes.lambdas[0]) && near(pcaRes.recon1, pcaRes.lambdas[1]));
ok('pca: vectors are unit length', pcaRes.vectors.every((v) => near(norm(v), 1)));

const pcaSqrtIn = [[2, 1], [1, 2]];
const pcaSqrt = matrixSqrt(pcaSqrtIn);
ok('pca: √S·√S = S', matNear(matMul(pcaSqrt, pcaSqrt), pcaSqrtIn, 1e-9), JSON.stringify(pcaSqrt));
ok('pca: √S is symmetric', matNear(pcaSqrt, transpose(pcaSqrt), 1e-12));
ok('pca: √S semi-axes = √λ', near(pcaSqrt[0][0] + pcaSqrt[1][1], Math.sqrt(3) + Math.sqrt(1)), JSON.stringify(pcaSqrt));

const pcaP1 = outerProject([Math.SQRT1_2, Math.SQRT1_2]);
ok('pca: P₁ is idempotent', isIdempotent(pcaP1));
ok('pca: P₁ is singular (rank 1)', near(det(pcaP1), 0));
ok('pca: projectTo lands on PC1 (parallel to dir)', (() => {
  const q = projectTo([3, 1], [[Math.SQRT1_2, Math.SQRT1_2]], 1);
  return near(q[0], q[1]) && near(q[0], 2, 1e-9);
})());

// Generators: deterministic, right shape, right story.
const pcaParamsA = { rho: 0.9, theta: 0, sx: 1.6, sy: 0.7, sz: 0.8, noise: 0.15 };
const pcaCloudA = generateCloud('blob', pcaParamsA, 2);
const pcaCloudB = generateCloud('blob', pcaParamsA, 2);
ok('gen: seeded → identical samples', JSON.stringify(pcaCloudA) === JSON.stringify(pcaCloudB));
ok('gen: 2D cloud is 90 points of length 2', pcaCloudA.length === 90 && pcaCloudA.every((p) => p.length === 2));
const pcaCloud3 = generateCloud('blob', pcaParamsA, 3);
ok('gen: 3D cloud is 150 points of length 3', pcaCloud3.length === 150 && pcaCloud3.every((p) => p.length === 3));
ok('gen: theta rotates the cloud (covariance not axis-aligned at θ=45°)', Math.abs(covarianceOf(generateCloud('blob', { ...pcaParamsA, rho: 0.1, theta: 45 }, 2))[0][1]) > 0.3);

const pcaIso = pcaOf(generateCloud('blob', { rho: 0, theta: 0, sx: 1.3, sy: 1.3, sz: 1.3, noise: 0.15 }, 2));
ok('gen: isotropic → ~50/50 split', pcaIso.explained[0] > 0.3 && pcaIso.explained[0] < 0.7, JSON.stringify(pcaIso.explained));
ok('gen: all λ ≥ 0', pcaIso.lambdas.every((l) => l >= 0));

const pcaClu = pcaOf(generateCloud('clusters', { rho: 0.25, theta: 0, sx: 0.55, sy: 0.4, sz: 0.5, noise: 0.12 }, 2));
ok('gen: two clusters → PC1 along the separation axis', pcaClu.lambdas[0] > pcaClu.lambdas[1] && Math.abs(pcaClu.vectors[0][0]) > 0.8, JSON.stringify(pcaClu.vectors[0]));

const pcaUnitsS = datasetCovariance('blob', { rho: 0.35, theta: 0, sx: 1, sy: 4, sz: 1.5, noise: 0.3 }, 2);
ok('gen: different-units → y variance dominates', pcaUnitsS[1][1] > 5 * pcaUnitsS[0][0], JSON.stringify(pcaUnitsS));
ok('gen: datasetCovariance is symmetric', matNear(pcaUnitsS, transpose(pcaUnitsS), 1e-12));

/* ---------------- partitioned (block) matrices, 4×4 ---------------- */

console.log('\nblocks (4×4 partitioned)');

const bA = [[2, 1], [1, 2]];
const bB = [[1, 0], [0, 1]];
const bC = [[0, 1], [1, 0]];
const bD = [[3, 1], [1, 3]];
const M4 = joinBlocks({ A: bA, B: bB, C: bC, D: bD });
const b4 = splitBlocks(M4);
ok('blocks: join ∘ split = identity', matNear(joinBlocks(b4), M4), JSON.stringify(b4));
ok('blocks: split recovers the four pieces',
  matNear(b4.A, bA) && matNear(b4.B, bB) && matNear(b4.C, bC) && matNear(b4.D, bD));

ok('blocks: structure detection (coupled)', blockStructure(M4) === 'coupled');
ok('blocks: structure detection (diagonal)', blockStructure(blockDiagonal(bA, bD)) === 'diagonal');
ok('blocks: structure detection (upper)', blockStructure(joinBlocks({ A: bA, B: bB, C: [[0, 0], [0, 0]], D: bD })) === 'upper');
ok('blocks: structure detection (lower)', blockStructure(joinBlocks({ A: bA, B: [[0, 0], [0, 0]], C: bC, D: bD })) === 'lower');

const S4 = schur(M4);
const SClosed = bD.map((row, i) => row.map((x, j) => x - matMul(matMul(bC, inverse(bA)), bB)[i][j]));
ok('blocks: S = D − CA⁻¹B', S4 !== null && matNear(S4, SClosed, 1e-9), JSON.stringify(S4));

const fac = detFactorization(M4);
ok('blocks: det M = det A · det S', fac.detS !== null && near(fac.detM, fac.detA * fac.detS, 1e-9),
  JSON.stringify(fac));

const L4 = blockLowerUnit(M4);
const U4 = blockUpper(M4);
ok('blocks: M = L·U', L4 !== null && U4 !== null && matNear(matMul(L4, U4), M4, 1e-9));
ok('blocks: det L = 1', L4 !== null && near(det(L4), 1, 1e-9), L4 ? String(det(L4)) : 'null');
ok('blocks: U is block upper-triangular', U4 !== null && blockStructure(U4) === 'upper');
ok('blocks: tr M = tr A + tr D', near(traceBlocks(M4).total, traceBlocks(M4).parts, 1e-12));

const N4 = [
  [1, 2, 0, 1],
  [3, 1, 1, 0],
  [0, 2, 1, 1],
  [1, 0, 2, 1],
];
const slots = blockProducts(M4, N4);
const product = [
  [...slots[0].sum[0], ...slots[1].sum[0]],
  [...slots[0].sum[1], ...slots[1].sum[1]],
  [...slots[2].sum[0], ...slots[3].sum[0]],
  [...slots[2].sum[1], ...slots[3].sum[1]],
];
ok('blocks: blockwise product = matMul', matNear(product, matMul(M4, N4), 1e-9), JSON.stringify(product));
ok('blocks: terms labeled AE/AF/CG…', slots[0].terms[0].label === 'A·E' && slots[3].terms[1].label === 'D·H');

const Mdiag = blockDiagonal([[1, 0], [0, 2]], [[3, 0], [0, 4]]);
const union = eigenUnion(Mdiag);
const unionVals = union
  ? [...union.fromA, ...union.fromD].map((e) => e.real).sort((a, b) => a - b)
  : [];
ok('blocks: eigen-union exists for block-diagonal', union !== null && JSON.stringify(unionVals) === JSON.stringify([1, 2, 3, 4]), JSON.stringify(unionVals));
ok('blocks: eigen-union of Mdiag matches eigen(Mdiag)', (() => {
  const full = eigen(Mdiag).flatMap((e) => Array(e.multiplicity).fill(Math.round(e.real * 1e7) / 1e7)).sort((a, b) => a - b);
  return JSON.stringify(full) === JSON.stringify([1, 2, 3, 4]);
})());
ok('blocks: eigenUnion null when coupled', eigenUnion(M4) === null);

const tr4 = blockTranspose(M4);
ok('blocks: transpose is blockwise (Mᵀ)', matNear(tr4.M, transpose(M4), 1e-12));
ok('blocks: transpose is blockwise (Aᵀ sits top-left)', matNear(tr4.named[0].M, transpose(bA)));

ok('blocks: conditional variance = s_yy − s_xy²/s_xx', near(conditionalVariance2([[2, 1], [1, 3]]), 2.5));

// Sylvester via Schur: build an SPD M from SPD A and SPD S.
const sA = [[2, 0.5], [0.5, 1.5]];
const sB = [[0.4, 0.2], [0.1, 0.3]];
const sS = [[1.5, 0.2], [0.2, 1.2]];
const sD = sS.map((row, i) => row.map((x, j) => x + matMul(matMul(transpose(sB), inverse(sA)), sB)[i][j]));
const Mspd = joinBlocks({ A: sA, B: sB, C: transpose(sB), D: sD });
ok('blocks: schur of the SPD fixture is S', matNear(schur(Mspd), sS, 1e-9), JSON.stringify(schur(Mspd)));
ok('blocks: SPD M ⟺ A ≻ 0 and S ≻ 0',
  definiteness(Mspd).kind === 'positiveDefinite' &&
    definiteness(sA).kind === 'positiveDefinite' &&
    definiteness(sS).kind === 'positiveDefinite');

ok('blocks: schur null when A is singular', schur(joinBlocks({ A: [[1, 2], [2, 4]], B: bB, C: bC, D: bD })) === null);

// n×n paths in matrix.ts (exercised by the 4×4 view).
console.log('\ndet / inverse / eigen for n = 4');
ok('det 4×4 identity', near(det(identity(4)), 1));
ok('det 4×4 diagonal', near(det([[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]]), 24));
ok('det 4×4 row swap flips sign', near(det([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]), -1));
ok('inverse 4×4: M·M⁻¹ = I', (() => {
  const inv = inverse(M4);
  return inv !== null && matNear(matMul(M4, inv), identity(4), 1e-8);
})(), JSON.stringify(inverse(M4)));
ok('inverse 4×4 singular → null', inverse([[1, 2, 0, 0], [2, 4, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) === null);
ok('eigen 4×4: Σλ = trace', (() => {
  const es = eigen([[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]]);
  const sum = es.reduce((s, e) => s + e.real * e.multiplicity, 0);
  return es.length > 0 && near(sum, 10, 1e-6);
})(), JSON.stringify(eigen([[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]]).map((e) => e.real)));
ok('definiteness 4×4: positive definite', definiteness([[2, 0, 0, 0], [0, 3, 0, 0], [0, 0, 4, 0], [0, 0, 0, 5]]).kind === 'positiveDefinite');
ok('definiteness 4×4: indefinite', definiteness([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).kind === 'indefinite');

// Transition scrubber: the viewport draws blendFromIdentity(active, t) —
// plain space at t = 0, the full matrix at t = 1, eased in between.
console.log('\ntransition scrub (blendFromIdentity)');
const tA = [[2, 1], [1, 2]];
ok('blend t=0 → identity', matNear(blendFromIdentity(tA, 0), identity(2), 1e-12));
ok('blend t=1 → M unchanged', matNear(blendFromIdentity(tA, 1), tA, 1e-12));
ok('blend t=1 → M by reference', blendFromIdentity(tA, 1) === tA);
ok('blend t=0.5 → exact halfway (eased midpoint)',
  matNear(blendFromIdentity(tA, 0.5), [[1.5, 0.5], [0.5, 1.5]], 1e-12),
  JSON.stringify(blendFromIdentity(tA, 0.5)));
ok('blend t=0.25 → I + easeInOutCubic(1/4)·(A−I)',
  matNear(blendFromIdentity(tA, 0.25), [[1.0625, 0.0625], [0.0625, 1.0625]], 1e-12),
  JSON.stringify(blendFromIdentity(tA, 0.25)));
ok('blend clamps past the ends (t=2 → M, t=−1 → I)',
  matNear(blendFromIdentity(tA, 2), tA, 1e-12) &&
    matNear(blendFromIdentity(tA, -1), identity(2), 1e-12));
ok('easeInOutCubic(0) = 0, (0.5) = 1/2, (1) = 1',
  near(easeInOutCubic(0), 0) && near(easeInOutCubic(0.5), 0.5) && near(easeInOutCubic(1), 1));

// Equation mode: the map y = M x printed as equations of a generic x.
console.log('\nequations (generic x)');
ok('row: basic 2×2',
  rowEquation([2, 3], 0) === 'y₁ = 2x₁ + 3x₂', JSON.stringify(rowEquation([2, 3], 0)));
ok('row: negative leading coefficient',
  rowEquation([-1.5, 0.25], 1) === 'y₂ = − 1.5x₁ + 0.25x₂',
  JSON.stringify(rowEquation([-1.5, 0.25], 1)));
ok('row: zero term dropped, sign kept',
  rowEquation([2, 0, -4], 2) === 'y₃ = 2x₁ − 4x₃', JSON.stringify(rowEquation([2, 0, -4], 2)));
ok('row: ±1 coefficient elided',
  rowEquation([1, -1], 0) === 'y₁ = x₁ − x₂', JSON.stringify(rowEquation([1, -1], 0)));
ok('row: all-zero row → y = 0',
  rowEquation([0, 0], 1) === 'y₂ = 0', JSON.stringify(rowEquation([0, 0], 1)));
ok('row: rounding at 4 decimals, sub-precision dropped',
  rowEquation([1 / 3, 0.00004], 0) === 'y₁ = 0.3333x₁',
  JSON.stringify(rowEquation([1 / 3, 0.00004], 0)));
ok('system: one equation per row',
  systemEquations([[1, 2], [3, 4]]).length === 2 &&
    systemEquations([[1, 2], [3, 4]])[1] === 'y₂ = 3x₁ + 4x₂',
  JSON.stringify(systemEquations([[1, 2], [3, 4]])));
ok('row: custom variable labels',
  rowEquation([1, 2], 0, { out: 'p' }) === 'p₁ = x₁ + 2x₂',
  JSON.stringify(rowEquation([1, 2], 0, { out: 'p' })));

const eqA = [[2, 1], [1, 3]];
const eqB = [[1, -1], [0, 2]];
const eqChain = chainEquations(eqA, eqB);
ok('chain: first hop reads off B',
  eqChain.first[0] === 'p₁ = x₁ − x₂' && eqChain.first[1] === 'p₂ = 2x₂',
  JSON.stringify(eqChain.first));
ok('chain: second hop reads off A in p variables',
  eqChain.second[0] === 'y₁ = 2p₁ + p₂' && eqChain.second[1] === 'y₂ = p₁ + 3p₂',
  JSON.stringify(eqChain.second));
// The invariant: expanding y = A(Bx) must give exactly A·B's entries.
const AB = matMul(eqA, eqB);
ok('chain: composed coefficients = entries of A·B',
  systemEquations(AB)[0] === eqChain.composed[0] && systemEquations(AB)[1] === eqChain.composed[1],
  JSON.stringify({ composed: eqChain.composed, AB: systemEquations(AB) }));
ok('chain: substituted line joins both readings',
  eqChain.substituted[0] === 'y₁ = 2p₁ + p₂ = 2x₁',
  JSON.stringify(eqChain.substituted));
// And numerically: substituting a concrete x through the chain lands on A(Bx).
{
  const x = [0.7, -1.3];
  const p = matVec(eqB, x);
  const viaChain = eqA.map((row) => row.reduce((s, a, j) => s + a * p[j], 0));
  const direct = matVec(eqA, matVec(eqB, x));
  ok('chain: y = A(Bx) numerically',
    viaChain.every((v, i) => near(v, direct[i], 1e-12)), JSON.stringify({ viaChain, direct }));
}
ok('chain: works for 3×3 too',
  chainEquations(A3, B3).composed.length === 3 &&
    chainEquations(A3, B3).composed[0] === systemEquations(matMul(A3, B3))[0],
  JSON.stringify(chainEquations(A3, B3).composed[0]));

// The 4×4 block view reads the input's components as x, y, u, v.
{
  const M4 = [[1, 2, 0, 1], [0, 1, 1, 0], [2, 0, 1, 1], [0, 1, 0, 1]];
  const N4 = [[1, 0, 1, 0], [0, 2, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]];
  const rows = systemEquations(M4, { vars: BLOCK_VARS });
  ok('block: system rows read in x, y, u, v',
    rows[0] === 'y₁ = x + 2y + v' && rows[1] === 'y₂ = y + u',
    JSON.stringify(rows));
  const ch4 = chainEquations(M4, N4, BLOCK_VARS);
  ok('block: first hop reads N against x, y, u, v',
    ch4.first[0] === 'p₁ = x + u' && ch4.first[1] === 'p₂ = 2y + v',
    JSON.stringify(ch4.first));
  // Same invariant as the 2×2 chain: expanded rows = entries of M·N.
  const MN = matMul(M4, N4);
  ok('block: composed = entries of M·N in x, y, u, v',
    systemEquations(MN, { vars: BLOCK_VARS })[0] === ch4.composed[0] &&
      systemEquations(MN, { vars: BLOCK_VARS })[3] === ch4.composed[3],
    JSON.stringify(ch4.composed));
}

console.log(fails === 0 ? '\n✓ all math checks passed\n' : `\n✗ ${fails} check(s) failed\n`);
process.exit(fails ? 1 : 0);
