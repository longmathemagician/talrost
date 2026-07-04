//! LU factorization with partial pivoting — the crate's linear-solve
//! primitive.
//!
//! The Newton corrector of a path tracker solves `J·Δx = −H` every
//! iteration; `inverse()` is the wrong primitive for that. Factor once with
//! [`super::Matrix::lu`], then run [`Lu::solve`] per right-hand side.

use crate::algebra::Monoid;
use crate::scalar::Scalar;
use crate::vector::Vector;

use super::Matrix;

/// A packed LU factorization `P·A = L·U` of a square matrix, produced by
/// [`super::Matrix::lu`].
///
/// `lu` stores `L` (unit lower triangle, implicit ones) below the diagonal
/// and `U` on/above it; `perm` records the row permutation `P` (position `k`
/// holds the original row index) and `odd` its parity (which flips the
/// determinant's sign).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Lu<T: Scalar, const N: usize> {
    lu: [[T; N]; N],
    perm: [usize; N],
    odd: bool,
}

impl<T: Scalar, const N: usize> Lu<T, N> {
    /// Doolittle factorization with partial pivoting by `norm_sqr` (cheap,
    /// exact, and meaningful for complex scalars where `PartialOrd` is not).
    /// Returns `None` when no nonzero pivot remains — the matrix is singular
    /// to working precision.
    pub(super) fn factor(m: &Matrix<T, N, N>) -> Option<Self> {
        let mut a = m.e;
        let mut perm = [0usize; N];
        for (k, p) in perm.iter_mut().enumerate() {
            *p = k;
        }
        let mut odd = false;

        for k in 0..N {
            // Select the remaining row whose k-th entry has the largest
            // squared norm.
            let mut pivot = k;
            let mut best = a[k][k].norm_sqr();
            for r in (k + 1)..N {
                let v = a[r][k].norm_sqr();
                if v > best {
                    best = v;
                    pivot = r;
                }
            }
            if best == T::Real::ZERO {
                return None; // singular
            }
            if pivot != k {
                a.swap(k, pivot);
                perm.swap(k, pivot);
                odd = !odd;
            }
            for r in (k + 1)..N {
                let factor = a[r][k] / a[k][k];
                a[r][k] = factor; // store the L multiplier in the zeroed slot
                for c in (k + 1)..N {
                    a[r][c] = a[r][c] - factor * a[k][c];
                }
            }
        }

        Some(Self { lu: a, perm, odd })
    }

    /// Solves `A·x = b` by permuting `b`, then forward substitution through
    /// the unit-lower `L` and back substitution through `U`. Infallible: a
    /// successfully factored matrix always solves.
    pub fn solve(&self, b: &Vector<T, N>) -> Vector<T, N> {
        // y = P·b
        let mut y = [T::ZERO; N];
        for (yi, &p) in y.iter_mut().zip(self.perm.iter()) {
            *yi = b.b[p];
        }
        // L·z = y (unit diagonal, so no division)
        for i in 0..N {
            for j in 0..i {
                y[i] = y[i] - self.lu[i][j] * y[j];
            }
        }
        // U·x = z
        for i in (0..N).rev() {
            for j in (i + 1)..N {
                y[i] = y[i] - self.lu[i][j] * y[j];
            }
            y[i] /= self.lu[i][i];
        }
        Vector::new(y)
    }

    /// The determinant of the factored matrix: the product of `U`'s diagonal,
    /// negated when the row permutation is odd.
    pub fn determinant(&self) -> T {
        let mut det = T::ONE;
        for (k, row) in self.lu.iter().enumerate() {
            det *= row[k];
        }
        if self.odd {
            -det
        } else {
            det
        }
    }
}
