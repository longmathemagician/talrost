//! Integer lattice algorithms: Smith and Hermite normal forms.
//!
//! The offline half of a polyhedral homotopy solver reduces each mixed cell's
//! exponent matrix: the binomial start system `x^A = b` is solved in closed
//! form through the Smith normal form `U·A·V = S` (unimodular `U`, `V`;
//! diagonal `S` with each entry dividing the next). The Hermite normal form
//! is the row-echelon analogue over ℤ (`U·A = H`).
//!
//! # Algorithm and entry growth
//!
//! Both forms use the simple gcd row/column reduction: repeatedly move the
//! smallest-magnitude nonzero entry into pivot position and take Euclidean
//! steps until it divides everything it must. This is textbook-correct and
//! allocation-free, but intermediate entries can grow much faster than the
//! input's magnitude (potentially overflowing `i64` for adversarial inputs —
//! debug builds panic on overflow). For the small exponent matrices of
//! sparse polynomial systems this is a non-issue; do not feed it large dense
//! matrices with huge entries.

use crate::matrix::Matrix;

/// `dst -= q · src` over paired rows of the working matrix and its recording
/// unimodular companion.
fn row_sub<const W1: usize, const W2: usize>(
    s: &mut [[i64; W1]],
    u: &mut [[i64; W2]],
    dst: usize,
    src: usize,
    q: i64,
) {
    if q == 0 {
        return;
    }
    // Rows are `Copy` arrays: snapshot the source row so the destination row
    // can be mutably iterated without aliasing the same slice.
    let (s_src, u_src) = (s[src], u[src]);
    for (d, v) in s[dst].iter_mut().zip(s_src) {
        *d -= q * v;
    }
    for (d, v) in u[dst].iter_mut().zip(u_src) {
        *d -= q * v;
    }
}

/// `dst += src` over paired rows.
fn row_add<const W1: usize, const W2: usize>(
    s: &mut [[i64; W1]],
    u: &mut [[i64; W2]],
    dst: usize,
    src: usize,
) {
    let (s_src, u_src) = (s[src], u[src]);
    for (d, v) in s[dst].iter_mut().zip(s_src) {
        *d += v;
    }
    for (d, v) in u[dst].iter_mut().zip(u_src) {
        *d += v;
    }
}

/// Negate row `r` in both matrices.
fn row_neg<const W1: usize, const W2: usize>(s: &mut [[i64; W1]], u: &mut [[i64; W2]], r: usize) {
    for x in s[r].iter_mut() {
        *x = -*x;
    }
    for x in u[r].iter_mut() {
        *x = -*x;
    }
}

/// Swap columns `a` and `b` in both the working matrix and its column
/// companion.
fn col_swap<const W1: usize, const W2: usize>(
    s: &mut [[i64; W1]],
    v: &mut [[i64; W2]],
    a: usize,
    b: usize,
) {
    for row in s.iter_mut() {
        row.swap(a, b);
    }
    for row in v.iter_mut() {
        row.swap(a, b);
    }
}

/// `col dst -= q · col src` in both matrices.
fn col_sub<const W1: usize, const W2: usize>(
    s: &mut [[i64; W1]],
    v: &mut [[i64; W2]],
    dst: usize,
    src: usize,
    q: i64,
) {
    if q == 0 {
        return;
    }
    for row in s.iter_mut() {
        row[dst] -= q * row[src];
    }
    for row in v.iter_mut() {
        row[dst] -= q * row[src];
    }
}

/// Smith normal form: returns `(U, S, V)` with `U·A·V = S`, where `U` (M×M)
/// and `V` (N×N) are unimodular (`|det| = 1`) and `S` is diagonal with
/// non-negative entries `d₀ | d₁ | …` (each dividing the next; trailing
/// entries may be zero when the rank is deficient).
pub fn smith_normal_form<const M: usize, const N: usize>(
    a: &Matrix<i64, M, N>,
) -> (Matrix<i64, M, M>, Matrix<i64, M, N>, Matrix<i64, N, N>) {
    let mut s = a.e;
    let mut u = Matrix::<i64, M, M>::IDENTITY.e;
    let mut v = Matrix::<i64, N, N>::IDENTITY.e;

    let rank_bound = if M < N { M } else { N };
    for t in 0..rank_bound {
        'pivot: loop {
            // Smallest-magnitude nonzero entry of the trailing submatrix.
            let mut best: Option<(usize, usize)> = None;
            for i in t..M {
                for j in t..N {
                    if s[i][j] != 0 && best.is_none_or(|(bi, bj)| s[i][j].abs() < s[bi][bj].abs()) {
                        best = Some((i, j));
                    }
                }
            }
            let Some((pi, pj)) = best else {
                break 'pivot; // trailing submatrix is zero: done
            };

            // Move the pivot to (t, t).
            if pi != t {
                s.swap(t, pi);
                u.swap(t, pi);
            }
            if pj != t {
                col_swap(&mut s, &mut v, t, pj);
            }
            let p = s[t][t];

            // Euclidean steps down the pivot column and across the pivot
            // row. A nonzero remainder leaves a smaller-magnitude entry for
            // the next round, so the loop terminates.
            let mut clean = true;
            for i in (t + 1)..M {
                if s[i][t] != 0 {
                    let q = s[i][t].div_euclid(p);
                    row_sub(&mut s, &mut u, i, t, q);
                    if s[i][t] != 0 {
                        clean = false;
                    }
                }
            }
            for j in (t + 1)..N {
                if s[t][j] != 0 {
                    let q = s[t][j].div_euclid(p);
                    col_sub(&mut s, &mut v, j, t, q);
                    if s[t][j] != 0 {
                        clean = false;
                    }
                }
            }
            if !clean {
                continue 'pivot;
            }

            // Divisibility invariant d_t | everything in the trailing
            // submatrix: fold an offending row into the pivot row and redo.
            for i in (t + 1)..M {
                for j in (t + 1)..N {
                    if s[i][j] % p != 0 {
                        row_add(&mut s, &mut u, t, i);
                        continue 'pivot;
                    }
                }
            }
            break 'pivot;
        }

        // Sign normalization: diagonal entries are non-negative.
        if s[t][t] < 0 {
            row_neg(&mut s, &mut u, t);
        }
    }

    (Matrix { e: u }, Matrix { e: s }, Matrix { e: v })
}

/// (Row-style) Hermite normal form: returns `(U, H)` with `U·A = H`, where
/// `U` (M×M) is unimodular and `H` is in row-echelon form with positive
/// pivots; in each pivot column the entries above the pivot are reduced to
/// `0 ≤ h < pivot`.
pub fn hermite_normal_form<const M: usize, const N: usize>(
    a: &Matrix<i64, M, N>,
) -> (Matrix<i64, M, M>, Matrix<i64, M, N>) {
    let mut h = a.e;
    let mut u = Matrix::<i64, M, M>::IDENTITY.e;

    let mut r = 0; // next pivot row
    for c in 0..N {
        if r == M {
            break;
        }

        // Gcd-eliminate column c among rows r..M: repeatedly bring the
        // smallest-magnitude nonzero entry to row r and reduce the others.
        loop {
            let mut best: Option<usize> = None;
            for (i, row) in h.iter().enumerate().take(M).skip(r) {
                if row[c] != 0 && best.is_none_or(|b: usize| row[c].abs() < h[b][c].abs()) {
                    best = Some(i);
                }
            }
            let Some(p) = best else {
                break; // column is zero from row r down: no pivot here
            };
            if p != r {
                h.swap(r, p);
                u.swap(r, p);
            }
            let mut done = true;
            for i in (r + 1)..M {
                if h[i][c] != 0 {
                    let q = h[i][c].div_euclid(h[r][c]);
                    row_sub(&mut h, &mut u, i, r, q);
                    if h[i][c] != 0 {
                        done = false;
                    }
                }
            }
            if done {
                break;
            }
        }
        if h[r][c] == 0 {
            continue; // no pivot in this column
        }

        // Positive pivot; entries above it reduced into [0, pivot).
        if h[r][c] < 0 {
            row_neg(&mut h, &mut u, r);
        }
        for i in 0..r {
            let q = h[i][c].div_euclid(h[r][c]);
            row_sub(&mut h, &mut u, i, r, q);
        }
        r += 1;
    }

    (Matrix { e: u }, Matrix { e: h })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// |det| of a small integer matrix, via the f64 LU determinant (test
    /// values are far below 2^53, so the conversion is exact).
    fn abs_det<const N: usize>(m: &Matrix<i64, N, N>) -> f64 {
        let mut e = [[0.0f64; N]; N];
        for (row, irow) in e.iter_mut().zip(m.e.iter()) {
            for (x, &v) in row.iter_mut().zip(irow.iter()) {
                *x = v as f64;
            }
        }
        Matrix::new(e).determinant().abs()
    }

    fn check_snf<const M: usize, const N: usize>(a: &Matrix<i64, M, N>) -> Matrix<i64, M, N> {
        let (u, s, v) = smith_normal_form(a);

        // U·A·V == S (Ring matmul over i64).
        assert_eq!(u * *a * v, s, "U·A·V != S for {:?}", a);

        // U, V unimodular.
        assert_eq!(abs_det(&u), 1.0, "det U != ±1");
        assert_eq!(abs_det(&v), 1.0, "det V != ±1");

        // S diagonal, non-negative, divisibility chain.
        for i in 0..M {
            for j in 0..N {
                if i != j {
                    assert_eq!(s.e[i][j], 0, "S not diagonal");
                }
            }
        }
        let r = if M < N { M } else { N };
        for t in 0..r {
            assert!(s.e[t][t] >= 0, "S diagonal negative");
            if t + 1 < r && s.e[t + 1][t + 1] != 0 {
                assert!(
                    s.e[t][t] != 0 && s.e[t + 1][t + 1] % s.e[t][t] == 0,
                    "divisibility chain broken: {} does not divide {}",
                    s.e[t][t],
                    s.e[t + 1][t + 1]
                );
            }
        }
        s
    }

    fn check_hnf<const M: usize, const N: usize>(a: &Matrix<i64, M, N>) -> Matrix<i64, M, N> {
        let (u, h) = hermite_normal_form(a);
        assert_eq!(u * *a, h, "U·A != H");
        assert_eq!(abs_det(&u), 1.0, "det U != ±1");
        h
    }

    #[test]
    fn snf_known_3x3() {
        // The classic example: SNF is diag(2, 6, 12).
        let a = Matrix::new([[2, 4, 4], [-6, 6, 12], [10, -4, -16]]);
        let s = check_snf(&a);
        assert_eq!(s.e, [[2, 0, 0], [0, 6, 0], [0, 0, 12]]);
    }

    #[test]
    fn snf_identity_and_zero() {
        let i3 = Matrix::<i64, 3, 3>::IDENTITY;
        assert_eq!(check_snf(&i3).e, i3.e);

        let z = Matrix::<i64, 2, 3>::ZERO;
        assert_eq!(check_snf(&z).e, z.e);
    }

    #[test]
    fn snf_diagonal_reordering_and_signs() {
        // diag(6, -4): SNF is diag(2, 12) (gcd, then product/gcd), positive.
        let a = Matrix::new([[6, 0], [0, -4]]);
        let s = check_snf(&a);
        assert_eq!(s.e, [[2, 0], [0, 12]]);
    }

    #[test]
    fn snf_rank_deficient() {
        // Rank 1: second invariant factor is 0.
        let a = Matrix::new([[2, 4], [4, 8]]);
        let s = check_snf(&a);
        assert_eq!(s.e, [[2, 0], [0, 0]]);
    }

    #[test]
    fn snf_non_square() {
        // 2×3 (wide) and 3×2 (tall) shapes.
        let a = Matrix::new([[1, 2, 3], [4, 5, 6]]);
        let s = check_snf(&a);
        assert_eq!(s.e, [[1, 0, 0], [0, 3, 0]]);

        let b = Matrix::new([[1, 4], [2, 5], [3, 6]]);
        let s = check_snf(&b);
        assert_eq!(s.e, [[1, 0], [0, 3], [0, 0]]);
    }

    #[test]
    fn snf_binomial_exponent_matrix() {
        // The kind of matrix a mixed cell produces: small, invertible over ℚ
        // but not over ℤ. |det| = number of start solutions = ∏ dᵢ.
        let a = Matrix::new([[3, 1], [1, 3]]);
        let s = check_snf(&a);
        assert_eq!(s.e, [[1, 0], [0, 8]]); // det = 8 solutions

        let b = Matrix::new([[2, 0, 0], [0, 3, 0], [1, 1, 4]]);
        let s = check_snf(&b);
        assert_eq!(s.e[0][0] * s.e[1][1] * s.e[2][2], 24);
    }

    #[test]
    fn snf_unimodular_input() {
        // A unimodular matrix has SNF = identity.
        let a = Matrix::new([[1, 2, 0], [0, 1, 3], [0, 0, 1]]);
        let s = check_snf(&a);
        assert_eq!(s.e, Matrix::<i64, 3, 3>::IDENTITY.e);
    }

    #[test]
    fn hnf_known_3x4() {
        // Wikipedia's row-style HNF example.
        let a = Matrix::new([[2, 3, 6, 2], [5, 6, 1, 6], [8, 3, 1, 1]]);
        let h = check_hnf(&a);
        assert_eq!(h.e, [[1, 0, 50, -11], [0, 3, 28, -2], [0, 0, 61, -13]]);
    }

    #[test]
    fn hnf_structure() {
        let a = Matrix::new([[4, 6], [2, 8]]);
        let h = check_hnf(&a);
        // Row echelon with positive pivots: [[2, x], [0, y]].
        assert_eq!(h.e[1][0], 0);
        assert!(h.e[0][0] > 0 && h.e[1][1] > 0);
        // Entry above the second pivot reduced into [0, pivot).
        assert!(h.e[0][1] >= 0 && h.e[0][1] < h.e[1][1]);
        assert_eq!(h.e, [[2, 8], [0, 10]]);
    }

    #[test]
    fn hnf_rank_deficient_and_zero() {
        let z = Matrix::<i64, 2, 2>::ZERO;
        assert_eq!(check_hnf(&z).e, z.e);

        // Rank 1 tall matrix: one pivot row, zero rows below.
        let a = Matrix::new([[2, 4], [4, 8], [6, 12]]);
        let h = check_hnf(&a);
        assert_eq!(h.e, [[2, 4], [0, 0], [0, 0]]);
    }

    #[test]
    fn hnf_identity() {
        let i = Matrix::<i64, 3, 3>::IDENTITY;
        assert_eq!(check_hnf(&i).e, i.e);
    }
}
