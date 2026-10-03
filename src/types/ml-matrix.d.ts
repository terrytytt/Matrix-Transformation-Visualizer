declare module 'ml-matrix' {
  export class Matrix {
    constructor(data?: number[][] | number);
    constructor(rows: number, cols: number);
    readonly rows: number;
    readonly columns: number;
    to2DArray(): number[][];
    get(i: number, j: number): number;
    set(i: number, j: number, value: number): number;
    isSquare(): boolean;
    isSymmetric(): boolean;
    static isMatrix(x: unknown): x is Matrix;
  }

  export class EigenvalueDecomposition {
    constructor(matrix: number[][] | Matrix, options?: { assumeSymmetric?: boolean });
    readonly n: number;
    readonly realEigenvalues: number[];
    readonly imaginaryEigenvalues: number[];
    readonly eigenvectorMatrix: Matrix;
    readonly diagonalMatrix: Matrix;
    readonly eigenvectors: VectorLike[];
  }

  export interface VectorLike {
    real: number;
    imaginary: number;
    vector: Matrix;
  }

  export class SingularValueDecomposition {
    constructor(matrix: number[][] | Matrix);
    /**
     * Singular values, **descending**. NOTE: at runtime this is a plain
     * `number[]`, despite older typings claiming `Matrix`.
     */
    readonly s: number[];
    /** Columns are the left singular vectors uᵢ (A vᵢ = σᵢ uᵢ). */
    readonly leftSingularVectors: Matrix;
    /** Columns are the right singular vectors vᵢ. */
    readonly rightSingularVectors: Matrix;
  }

  export const EVD: typeof EigenvalueDecomposition;
  export const SVD: typeof SingularValueDecomposition;

  export function inverse(matrix: number[][] | Matrix): Matrix;
  export function solve(left: number[][] | Matrix, right: number[][] | Matrix): Matrix;
}
