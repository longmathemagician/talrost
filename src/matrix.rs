//! Fixed-size row-major matrices over the algebraic tower.
//!
//! [`Matrix<T, M, N>`] is `M` rows of `N` columns on the stack. Structural
//! operations (add/sub/neg, scalar and matrix multiplication, transpose)
//! need only `T: Ring` — integer exponent matrices are first-class citizens
//! (see [`crate::lattice`]) — while the numeric operations (`determinant`,
//! `inverse`, [`Matrix::lu`]/[`Matrix::solve`]) require `T: Scalar`.

use core::ops::{Add, Index, IndexMut, Mul, Neg, Sub};

use crate::algebra::Ring;
use crate::scalar::Scalar;
use crate::vector::Vector;

mod lu;
pub use lu::Lu;

/// An M×N matrix in the conventional row-major sense: `M` rows of `N`
/// columns, stored as `e: [[T; N]; M]` (outer index = row).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Matrix<T, const M: usize, const N: usize> {
    /// The entries, row-major: `e[i][j]` is row `i`, column `j`.
    pub e: [[T; N]; M],
}

impl<T, const M: usize, const N: usize> Matrix<T, M, N> {
    /// Builds a matrix from its row-major entry array.
    pub const fn new(e: [[T; N]; M]) -> Self {
        Self { e }
    }
}

impl<T, const M: usize, const N: usize> From<[[T; N]; M]> for Matrix<T, M, N> {
    fn from(e: [[T; N]; M]) -> Self {
        Self { e }
    }
}

/// Entry access by `(row, column)` pair. Panics on out-of-range indices,
/// like a slice.
impl<T, const M: usize, const N: usize> Index<(usize, usize)> for Matrix<T, M, N> {
    type Output = T;

    fn index(&self, (i, j): (usize, usize)) -> &T {
        &self.e[i][j]
    }
}

impl<T, const M: usize, const N: usize> IndexMut<(usize, usize)> for Matrix<T, M, N> {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut T {
        &mut self.e[i][j]
    }
}

/// The default matrix is [`Matrix::ZERO`].
impl<T: Ring, const M: usize, const N: usize> Default for Matrix<T, M, N> {
    fn default() -> Self {
        Self::ZERO
    }
}

// Structural operations need only a `Ring`: exponent matrices of sparse
// polynomial systems are *integer* matrices (see `crate::lattice`), so
// construction, ZERO/IDENTITY, add/sub/neg, scalar multiplication, transpose,
// and the matrix product must not demand a `Scalar`. Norms, determinant,
// inverse, and lu/solve stay `Scalar`-bound below.
impl<T: Ring, const M: usize, const N: usize> Matrix<T, M, N> {
    /// The zero matrix (the additive identity).
    pub const ZERO: Matrix<T, M, N> = Self {
        e: [[T::ZERO; N]; M],
    };

    /// Returns the transpose, an N×M matrix.
    pub fn transpose(&self) -> Matrix<T, N, M> {
        let mut e = [[T::ZERO; M]; N];
        for (i, row) in self.e.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                e[j][i] = v;
            }
        }
        Matrix { e }
    }
}

/// Square-matrix constants. `IDENTITY` needs only a `Ring`; attempting it on
/// a non-square matrix is a *compile* error.
impl<T: Ring, const N: usize> Matrix<T, N, N> {
    /// The identity matrix (ones on the diagonal, zero elsewhere).
    pub const IDENTITY: Self = Self::identity();

    const fn identity() -> Self {
        let mut e = [[T::ZERO; N]; N];
        let mut i = 0;
        while i < N {
            e[i][i] = T::ONE;
            i += 1;
        }
        Self { e }
    }
}

/// Square-matrix numerics. Attempting these on a non-square matrix is a
/// *compile* error now, not a runtime panic:
///
/// ```compile_fail
/// use talrost::matrix::Matrix;
/// let a = Matrix::<f64, 2, 3>::new([[1., 2., 3.], [4., 5., 6.]]);
/// let _ = a.determinant(); // no method: determinant requires Matrix<T, N, N>
/// ```
impl<T: Scalar, const N: usize> Matrix<T, N, N> {
    /// LU factorization with partial pivoting (pivots chosen by `norm_sqr`,
    /// so the same code works for complex scalars). Returns `None` if the
    /// matrix is singular. This is the crate's one pivoting code path:
    /// [`Matrix::determinant`] (for `N > 3`), [`Matrix::inverse`], and
    /// [`Matrix::solve`] are all built on it; factor once, then reuse the
    /// [`Lu`] for repeated solves against different right-hand sides.
    pub fn lu(&self) -> Option<Lu<T, N>> {
        Lu::factor(self)
    }

    /// Solves `self · x = b` through [`Matrix::lu`]; `None` if singular.
    ///
    /// For repeated solves against the same matrix (e.g. a Newton corrector
    /// iterating on one Jacobian), call [`Matrix::lu`] once and reuse
    /// [`Lu::solve`].
    pub fn solve(&self, b: &Vector<T, N>) -> Option<Vector<T, N>> {
        self.lu().map(|f| f.solve(b))
    }

    /// Returns the determinant. Degrees 1–3 use closed forms (the `match` is
    /// constant-folded after monomorphization); larger matrices go through
    /// [`Matrix::lu`]. Returns `T::ZERO` for singular matrices.
    pub fn determinant(&self) -> T {
        match N {
            0 => T::ONE, // determinant of the empty matrix is the empty product
            1 => self.e[0][0],
            2 => self.e[0][0] * self.e[1][1] - self.e[0][1] * self.e[1][0],
            3 => {
                let m1 = self.e[1][1] * self.e[2][0];
                let ma1 = self.e[1][0] * self.e[2][1] - m1;
                let m2 = self.e[1][2] * self.e[2][0];
                let ma2 = self.e[1][0] * self.e[2][2] - m2;
                let m3 = self.e[1][2] * self.e[2][1];
                let ma3 = self.e[1][1] * self.e[2][2] - m3;
                let m4 = self.e[0][2] * ma1;
                let ma4 = self.e[0][1] * ma2 - m4;
                self.e[0][0] * ma3 - ma4
            }
            _ => match self.lu() {
                Some(f) => f.determinant(),
                None => T::ZERO,
            },
        }
    }

    /// Returns the inverse by solving `A·x = e_j` for each identity column
    /// through one LU factorization, or `None` if the matrix is singular.
    ///
    /// If the goal is solving `A·x = b`, use [`Matrix::solve`] (or
    /// [`Matrix::lu`] + [`Lu::solve`]) instead — it is cheaper and more
    /// accurate than forming the inverse.
    pub fn inverse(&self) -> Option<Self> {
        let f = self.lu()?;
        let mut e = [[T::ZERO; N]; N];
        for j in 0..N {
            let mut col = [T::ZERO; N];
            col[j] = T::ONE;
            let x = f.solve(&Vector::new(col));
            for (row, &xi) in e.iter_mut().zip(x.b.iter()) {
                row[j] = xi;
            }
        }
        Some(Self { e })
    }
}

/// The naive triple loop over a plain `Ring`, accumulating with `mul`+`add`.
/// This is the only multiply kernel in default builds, and the `default`
/// (non-square / non-special-size) kernel under `feature = "specialization"`.
///
/// Note (§5.6): relaxing matmul from `Scalar` to `Ring` traded the old
/// `mul_add_fast` accumulation for plain mul+add — `Ring` has no fused
/// multiply-add. Accepted: it makes integer matrix products expressible, and
/// LLVM still contracts to FMA where the target allows it.
fn mul_naive<T: Ring, const M: usize, const K: usize, const N: usize>(
    a: &Matrix<T, M, K>,
    b: &Matrix<T, K, N>,
) -> Matrix<T, M, N> {
    let mut e = [[T::ZERO; N]; M];
    for (i, row) in e.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            let mut acc = T::ZERO;
            for k in 0..K {
                acc += a.e[i][k] * b.e[k][j];
            }
            *v = acc;
        }
    }
    Matrix { e }
}

// (M×K) · (K×N) → (M×N), the conventional shape signature, over any `Ring`.
//
// Default builds use `mul_naive` unconditionally: at these sizes the naive
// loop is the numerically stable (and usually fastest) choice. With
// `feature = "specialization"` (nightly) dispatch goes through the internal
// `Gemm` trait, whose impls for concrete square sizes select the
// multiplication-saving kernels in [`kernels`].
impl<T: Ring, const M: usize, const K: usize, const N: usize> Mul<Matrix<T, K, N>>
    for Matrix<T, M, K>
{
    type Output = Matrix<T, M, N>;

    #[cfg(not(feature = "specialization"))]
    fn mul(self, x: Matrix<T, K, N>) -> Self::Output {
        mul_naive(&self, &x)
    }

    #[cfg(feature = "specialization")]
    fn mul(self, x: Matrix<T, K, N>) -> Self::Output {
        kernels::Gemm::gemm(self, x)
    }
}

// Fixed-size multiply kernels (Strassen / Laderman / AlphaTensor-style)
// dispatched by `min_specialization`; see the module docs. Declared
// out-of-line so that default (stable) builds never even parse the unstable
// `default fn` syntax.
#[cfg(feature = "specialization")]
mod kernels;

impl<T: Ring, const M: usize, const N: usize> Add<Matrix<T, M, N>> for Matrix<T, M, N> {
    type Output = Matrix<T, M, N>;

    fn add(self, x: Matrix<T, M, N>) -> Self::Output {
        let mut e = [[T::ZERO; N]; M];
        for (i, row) in e.iter_mut().enumerate() {
            for (j, v) in row.iter_mut().enumerate() {
                *v = self.e[i][j] + x.e[i][j];
            }
        }
        Self::Output { e }
    }
}

impl<T: Ring, const M: usize, const N: usize> Sub<Matrix<T, M, N>> for Matrix<T, M, N> {
    type Output = Matrix<T, M, N>;

    fn sub(self, x: Matrix<T, M, N>) -> Self::Output {
        let mut e = [[T::ZERO; N]; M];
        for (i, row) in e.iter_mut().enumerate() {
            for (j, v) in row.iter_mut().enumerate() {
                *v = self.e[i][j] - x.e[i][j];
            }
        }
        Self::Output { e }
    }
}

impl<T: Ring, const M: usize, const N: usize> Neg for Matrix<T, M, N> {
    type Output = Matrix<T, M, N>;

    fn neg(mut self) -> Self::Output {
        for row in self.e.iter_mut() {
            for v in row.iter_mut() {
                *v = -*v;
            }
        }
        self
    }
}

// Scalar multiplication (scalar on the right; `T` can never unify with
// `Matrix`, so this coexists with the matrix product above).
impl<T: Ring, const M: usize, const N: usize> Mul<T> for Matrix<T, M, N> {
    type Output = Matrix<T, M, N>;

    fn mul(mut self, rhs: T) -> Self::Output {
        for row in self.e.iter_mut() {
            for v in row.iter_mut() {
                *v *= rhs;
            }
        }
        self
    }
}

// Matrix–vector product: (M×N) · N → M, treating the vector as a column.
// The Newton corrector's residual check `J·Δx ≈ −H` is exactly this shape.
impl<T: Ring, const M: usize, const N: usize> Mul<Vector<T, N>> for Matrix<T, M, N> {
    type Output = Vector<T, M>;

    fn mul(self, x: Vector<T, N>) -> Self::Output {
        let mut b = [T::ZERO; M];
        for (out, row) in b.iter_mut().zip(self.e.iter()) {
            let mut acc = T::ZERO;
            for (&a, &v) in row.iter().zip(x.b.iter()) {
                acc += a * v;
            }
            *out = acc;
        }
        Vector { b }
    }
}

/// Scalar-on-the-left multiplication, stamped per concrete scalar type
/// (coherence forbids the blanket `impl<T: Ring> Mul<Matrix<T, M, N>> for T`
/// because `T` is a bare type parameter in the `impl` head).
macro_rules! impl_scalar_matrix_mul {
    ($($s:ty),+ $(,)?) => {
        $(
            impl<const M: usize, const N: usize> Mul<Matrix<$s, M, N>> for $s {
                type Output = Matrix<$s, M, N>;

                fn mul(self, mut rhs: Matrix<$s, M, N>) -> Self::Output {
                    for row in rhs.e.iter_mut() {
                        for v in row.iter_mut() {
                            *v *= self;
                        }
                    }
                    rhs
                }
            }
        )+
    };
}

impl_scalar_matrix_mul!(f32, f64, crate::complex::c32, crate::complex::c64);

// Writes straight to the `Formatter` (no allocation) so it works in `no_std`.
impl<T: core::fmt::Display, const M: usize, const N: usize> core::fmt::Display for Matrix<T, M, N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("\n")?;
        for row in &self.e {
            f.write_str("|")?;
            for (j, e) in row.iter().enumerate() {
                if j != 0 {
                    f.write_str(", ")?;
                }
                write!(f, "{}", e)?;
            }
            f.write_str("|\n")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::complex::c64;

    /// Elementwise |a - b| <= tol for square f64 matrices.
    fn approx_eq<const N: usize>(a: Matrix<f64, N, N>, b: Matrix<f64, N, N>, tol: f64) -> bool {
        a.e.iter()
            .zip(b.e.iter())
            .all(|(ra, rb)| ra.iter().zip(rb.iter()).all(|(x, y)| (x - y).abs() <= tol))
    }

    #[test]
    fn det_2x2() {
        let a = Matrix::new([[1., 2.], [3., 4.]]);
        let b = -2.;
        assert_eq!(a.determinant(), b);
    }

    #[test]
    fn det_3x3() {
        let a = Matrix::new([[1., 2., 3.], [4., 5., 3.], [7., 8., 9.]]);
        let b = -18.;
        assert_eq!(a.determinant(), b);
    }

    #[test]
    fn det_1x1() {
        let a = Matrix::new([[7.]]);
        assert_eq!(a.determinant(), 7.);
    }

    #[test]
    fn det_4x4() {
        // Known value: expansion along the first column gives 4 * (-60) = -240.
        let a = Matrix::<f64, 4, 4>::new([
            [4., 3., 2., 2.],
            [0., 1., -3., 3.],
            [0., -1., 3., 3.],
            [0., 3., 1., 1.],
        ]);
        assert!((a.determinant() + 240.).abs() < 1e-9);

        // Singular 4x4 (two equal rows) has determinant zero.
        let s = Matrix::new([
            [1., 2., 3., 4.],
            [5., 6., 7., 8.],
            [1., 2., 3., 4.],
            [0., 0., 0., 1.],
        ]);
        assert_eq!(s.determinant(), 0.);
    }

    #[test]
    fn det_complex() {
        // [[i, 0], [0, i]]: determinant is i^2 = -1.
        let i = c64::new(0., 1.);
        let z = c64::new(0., 0.);
        let a = Matrix::new([[i, z], [z, i]]);
        assert_eq!(a.determinant(), c64::new(-1., 0.));
    }

    #[test]
    fn transpose_shape_and_round_trip() {
        let a = Matrix::<f64, 2, 3>::new([[1., 2., 3.], [4., 5., 6.]]);
        let t: Matrix<f64, 3, 2> = a.transpose();
        assert_eq!(t, Matrix::new([[1., 4.], [2., 5.], [3., 6.]]));
        assert_eq!(t.transpose(), a);
    }

    #[test]
    fn inverse_2x2() {
        let a = Matrix::new([[1., 2.], [3., 4.]]);
        let inv = a.inverse().unwrap();
        assert!(approx_eq(inv * a, Matrix::IDENTITY, 1e-12));
        assert!(approx_eq(a * inv, Matrix::IDENTITY, 1e-12));
    }

    #[test]
    fn inverse_3x3() {
        let a = Matrix::new([[1., 2., 3.], [4., 5., 3.], [7., 8., 9.]]);
        let inv = a.inverse().unwrap();
        assert!(approx_eq(inv * a, Matrix::IDENTITY, 1e-12));
    }

    #[test]
    fn inverse_4x4() {
        let a = Matrix::new([
            [4., 3., 2., 2.],
            [0., 1., -3., 3.],
            [0., -1., 3., 3.],
            [0., 3., 1., 1.],
        ]);
        let inv = a.inverse().unwrap();
        assert!(approx_eq(inv * a, Matrix::IDENTITY, 1e-12));
    }

    #[test]
    fn inverse_singular_is_none() {
        let a = Matrix::new([[1., 2.], [2., 4.]]);
        assert!(a.inverse().is_none());

        let b = Matrix::<f64, 3, 3>::ZERO;
        assert!(b.inverse().is_none());
    }

    #[test]
    fn mult_2x2() {
        let a = Matrix::new([[1., 2.], [3., 4.]]);
        let b = Matrix::new([[5., 6.], [7., 8.]]);
        let c = Matrix::new([[19., 22.], [43., 50.]]);
        assert_eq!(a * b, c);
    }

    #[test]
    fn mult_3x3() {
        let a = Matrix::new([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
        let b = Matrix::new([[9., 8., 7.], [6., 5., 4.], [3., 2., 1.]]);
        let c = Matrix::new([[30., 24., 18.], [84., 69., 54.], [138., 114., 90.]]);
        assert_eq!(a * b, c);
    }

    #[test]
    fn mult_4x4() {
        let a = Matrix::new([
            [1., 2., 3., 4.],
            [5., 6., 7., 8.],
            [9., 10., 11., 12.],
            [13., 14., 15., 16.],
        ]);
        let b = Matrix::new([
            [17., 18., 19., 20.],
            [21., 22., 23., 24.],
            [25., 26., 27., 28.],
            [29., 30., 31., 32.],
        ]);
        let c = Matrix::new([
            [250., 260., 270., 280.],
            [618., 644., 670., 696.],
            [986., 1028., 1070., 1112.],
            [1354., 1412., 1470., 1528.],
        ]);
        assert_eq!(a * b, c);
    }

    #[test]
    fn mult_3x2_by_2x2_identity() {
        // Regression: kernel dispatch used bitwise `&` (`M == 2 & N & O`), so a
        // 3-rows-by-2-cols matrix times the 2x2 identity ran the Strassen 2x2
        // kernel and silently zeroed the third row.
        let a = Matrix::<f64, 3, 2>::new([[1., 2.], [3., 4.], [5., 6.]]);
        let b = Matrix::<f64, 2, 2>::IDENTITY;
        assert_eq!(a * b, a);
    }

    #[test]
    fn mult_non_square() {
        // 2 rows x 3 cols times 3 rows x 2 cols, hand-computed.
        let a = Matrix::<f64, 2, 3>::new([[1., 2., 3.], [4., 5., 6.]]);
        let b = Matrix::<f64, 3, 2>::new([[7., 8.], [9., 10.], [11., 12.]]);
        let c = Matrix::<f64, 2, 2>::new([[58., 64.], [139., 154.]]);
        assert_eq!(a * b, c);

        // Non-identity 2x2 on the right of a 3 rows x 2 cols matrix, hand-computed
        // (this shape hit the mis-dispatched Strassen kernel before the fix).
        let a = Matrix::<f64, 3, 2>::new([[1., 2.], [3., 4.], [5., 6.]]);
        let b = Matrix::<f64, 2, 2>::new([[7., 8.], [9., 10.]]);
        let c = Matrix::<f64, 3, 2>::new([[25., 28.], [57., 64.], [89., 100.]]);
        assert_eq!(a * b, c);
    }

    // The old `det_non_square_unimplemented` should_panic test is gone:
    // `determinant`/`inverse`/`IDENTITY` now live on `Matrix<T, N, N>` only,
    // so a non-square determinant no longer compiles (see the compile_fail
    // doctest on the square impl block).

    /// Plain triple-loop reference product, independent of whichever kernel
    /// `Mul` dispatches to.
    fn reference_mul<const M: usize, const K: usize, const N: usize>(
        a: &Matrix<f64, M, K>,
        b: &Matrix<f64, K, N>,
    ) -> Matrix<f64, M, N> {
        let mut e = [[0.0; N]; M];
        for (i, row) in e.iter_mut().enumerate() {
            for (j, v) in row.iter_mut().enumerate() {
                for k in 0..K {
                    *v += a.e[i][k] * b.e[k][j];
                }
            }
        }
        Matrix { e }
    }

    /// `Mul` must exactly match the naive loop for integer-valued float
    /// matrices, whatever kernel it dispatches to. Runs in both configs; with
    /// `--features specialization` the square sizes exercise the Strassen /
    /// Laderman / AlphaTensor `Gemm` impls (which agree exactly with the
    /// naive loop on small integers) and the non-square shapes exercise the
    /// `default` naive impl.
    #[test]
    fn mul_matches_reference_for_all_kernel_sizes() {
        // 2x2 (Strassen under the feature), with negatives.
        let a2 = Matrix::<f64, 2, 2>::new([[1., -2.], [3., 4.]]);
        let b2 = Matrix::new([[5., 6.], [-7., 8.]]);
        assert_eq!(a2 * b2, reference_mul(&a2, &b2));

        // 3x3 (Laderman under the feature).
        let a3 = Matrix::<f64, 3, 3>::new([[1., -2., 3.], [4., 5., -6.], [-7., 8., 9.]]);
        let b3 = Matrix::new([[9., 8., -7.], [-6., 5., 4.], [3., -2., 1.]]);
        assert_eq!(a3 * b3, reference_mul(&a3, &b3));

        // 4x4 (AlphaTensor-style under the feature).
        let a4 = Matrix::<f64, 4, 4>::new([
            [1., -2., 3., -4.],
            [5., 6., -7., 8.],
            [-9., 10., 11., -12.],
            [13., -14., 15., 16.],
        ]);
        let b4 = Matrix::new([
            [-16., 15., -14., 13.],
            [12., -11., 10., -9.],
            [8., 7., -6., 5.],
            [-4., 3., 2., -1.],
        ]);
        assert_eq!(a4 * b4, reference_mul(&a4, &b4));

        // Non-square shapes always take the naive path.
        let a23 = Matrix::<f64, 2, 3>::new([[1., 2., -3.], [4., -5., 6.]]);
        let b32 = Matrix::<f64, 3, 2>::new([[7., -8.], [9., 10.], [-11., 12.]]);
        assert_eq!(a23 * b32, reference_mul(&a23, &b32));

        // Square 2x2 on the right of a non-square: must not hit the 2x2 kernel.
        let a32 = Matrix::<f64, 3, 2>::new([[1., 2.], [-3., 4.], [5., -6.]]);
        assert_eq!(a32 * b2, reference_mul(&a32, &b2));

        // Square 4x4 on the right of a 2x4: must not hit the 4x4 kernel.
        let a24 = Matrix::<f64, 2, 4>::new([[1., -2., 3., 4.], [-5., 6., 7., -8.]]);
        assert_eq!(a24 * b4, reference_mul(&a24, &b4));

        // 1xK * Kx1 degenerate shapes.
        let a13 = Matrix::<f64, 1, 3>::new([[2., -3., 4.]]);
        let b31 = Matrix::<f64, 3, 1>::new([[5.], [6.], [-7.]]);
        assert_eq!(a13 * b31, reference_mul(&a13, &b31));
        assert_eq!(b31 * a13, reference_mul(&b31, &a13));

        // 5x5: larger square size, also the naive path in both configs.
        let a5 = Matrix::<f64, 5, 5>::new([
            [1., 2., 3., 4., 5.],
            [-1., -2., -3., -4., -5.],
            [2., 4., 6., 8., 10.],
            [5., 4., 3., 2., 1.],
            [0., 1., 0., -1., 0.],
        ]);
        assert_eq!(a5 * a5, reference_mul(&a5, &a5));
    }

    #[test]
    fn lu_solve_real() {
        // A·x = b with known solution x = (1, -2, 3).
        let a = Matrix::<f64, 3, 3>::new([[2., 1., 1.], [4., -6., 0.], [-2., 7., 2.]]);
        let x = Vector::new([1., -2., 3.]);
        let b = Vector::new([
            2. * 1. + 1. * -2. + 1. * 3.,
            4. * 1. + -6. * -2.,
            -2. * 1. + 7. * -2. + 2. * 3.,
        ]);

        let f = a.lu().expect("nonsingular");
        let got = f.solve(&b);
        for (g, w) in got.b.iter().zip(x.b.iter()) {
            assert!((g - w).abs() < 1e-12, "{} != {}", g, w);
        }

        // Matrix::solve is the one-shot form.
        let got = a.solve(&b).unwrap();
        for (g, w) in got.b.iter().zip(x.b.iter()) {
            assert!((g - w).abs() < 1e-12);
        }

        // Factor once, reuse for a second right-hand side.
        let b2 = Vector::new([1., 0., 0.]);
        let x2 = f.solve(&b2);
        let back = a * x2.column();
        assert!((back.e[0][0] - 1.).abs() < 1e-12);
        assert!(back.e[1][0].abs() < 1e-12);
        assert!(back.e[2][0].abs() < 1e-12);
    }

    #[test]
    fn lu_solve_complex() {
        // [[i, 1], [1, i]]·x = b, x = (1 - i, 2i):
        // b = (i(1-i) + 2i, (1-i) + i·2i) = (1 + 3i, -1 - i).
        let i = c64::new(0., 1.);
        let one = c64::new(1., 0.);
        let a = Matrix::new([[i, one], [one, i]]);
        let b = Vector::new([c64::new(1., 3.), c64::new(-1., -1.)]);
        let x = a.solve(&b).unwrap();
        let want = [c64::new(1., -1.), c64::new(0., 2.)];
        for (g, w) in x.b.iter().zip(want.iter()) {
            assert!((*g - *w).norm() < 1e-12, "{} != {}", g, w);
        }
    }

    #[test]
    fn lu_singular_is_none() {
        let s = Matrix::<f64, 2, 2>::new([[1., 2.], [2., 4.]]);
        assert!(s.lu().is_none());
        assert!(s.solve(&Vector::new([1., 1.])).is_none());
        assert!(Matrix::<f64, 3, 3>::ZERO.lu().is_none());

        // Singular only after elimination (duplicate rows cancel *exactly*;
        // the singularity test is an exact zero-pivot check, so inexact rank
        // deficiencies like [[1,2,3],[4,5,6],[7,8,9]] can survive rounding).
        let s = Matrix::<f64, 3, 3>::new([[1., 2., 3.], [4., 5., 6.], [1., 2., 3.]]);
        assert!(s.lu().is_none());
    }

    #[test]
    fn lu_determinant_consistent_with_closed_forms() {
        // 2x2 and 3x3 closed forms vs the LU product-of-pivots.
        let a2 = Matrix::<f64, 2, 2>::new([[1., 2.], [3., 4.]]);
        assert!((a2.lu().unwrap().determinant() - a2.determinant()).abs() < 1e-12);

        let a3 = Matrix::<f64, 3, 3>::new([[1., 2., 3.], [4., 5., 3.], [7., 8., 9.]]);
        assert!((a3.lu().unwrap().determinant() - a3.determinant()).abs() < 1e-12);

        // 4x4 (the N > 3 determinant arm *is* LU; check the known value and
        // that odd permutation parity is handled).
        let a4 = Matrix::<f64, 4, 4>::new([
            [0., 1., 0., 0.],
            [1., 0., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
        ]);
        assert_eq!(a4.determinant(), -1.0); // odd row swap
        assert_eq!(a4.lu().unwrap().determinant(), -1.0);

        // Complex determinant through LU: det(diag(i, i, i, 1)) = i³ = -i.
        let i = c64::new(0., 1.);
        let z = c64::new(0., 0.);
        let one = c64::new(1., 0.);
        let a = Matrix::new([[i, z, z, z], [z, i, z, z], [z, z, i, z], [z, z, z, one]]);
        let det = a.determinant();
        assert!((det - c64::new(0., -1.)).norm() < 1e-12);
    }

    #[test]
    fn lu_solve_matches_inverse() {
        let a = Matrix::<f64, 4, 4>::new([
            [4., 3., 2., 2.],
            [0., 1., -3., 3.],
            [0., -1., 3., 3.],
            [0., 3., 1., 1.],
        ]);
        let b = Vector::new([1., 2., 3., 4.]);
        let x = a.solve(&b).unwrap();
        let via_inv = a.inverse().unwrap() * b.column();
        for (g, w) in x.b.iter().zip(via_inv.e.iter()) {
            assert!((g - w[0]).abs() < 1e-11);
        }
    }

    #[test]
    fn ring_relaxed_integer_matrices() {
        // §5.6: integer matrices are first-class for structural ops.
        let a = Matrix::<i64, 2, 2>::new([[1, 2], [3, 4]]);
        let b = Matrix::<i64, 2, 2>::new([[5, 6], [7, 8]]);

        assert_eq!(a * b, Matrix::new([[19, 22], [43, 50]]));
        assert_eq!(a + b, Matrix::new([[6, 8], [10, 12]]));
        assert_eq!(b - a, Matrix::new([[4, 4], [4, 4]]));
        assert_eq!(-a, Matrix::new([[-1, -2], [-3, -4]]));
        assert_eq!(a * 3, Matrix::new([[3, 6], [9, 12]]));
        assert_eq!(a * Matrix::IDENTITY, a);
        assert_eq!(a + Matrix::ZERO, a);
        assert_eq!(a.transpose(), Matrix::new([[1, 3], [2, 4]]));

        // Non-square integer product.
        let c = Matrix::<i64, 2, 3>::new([[1, 0, -1], [2, 1, 0]]);
        let d = Matrix::<i64, 3, 2>::new([[1, 1], [0, 2], [3, -1]]);
        assert_eq!(c * d, Matrix::new([[-2, 2], [2, 4]]));

        // Integer vectors too.
        let v = Vector::new([1i64, -2, 3]);
        let w = Vector::new([4i64, 5, -6]);
        assert_eq!(v + w, Vector::new([5, 3, -3]));
        assert_eq!(v - w, Vector::new([-3, -7, 9]));
        assert_eq!(-v, Vector::new([-1, 2, -3]));
        assert_eq!(v * 2, Vector::new([2, -4, 6]));
        assert_eq!(Vector::new([1i64, 2]).cross(&Vector::new([3, 4])), -2);
    }

    #[test]
    fn matrix_vector_product() {
        // 2×3 · 3 → 2, hand-computed.
        let a = Matrix::<f64, 2, 3>::new([[1., 2., 3.], [4., 5., 6.]]);
        let x = Vector::new([7., 8., 9.]);
        assert_eq!(a * x, Vector::new([50., 122.]));

        // The identity fixes every vector; consistency with solve().
        let m = Matrix::<f64, 3, 3>::new([[2., 1., 1.], [4., -6., 0.], [-2., 7., 2.]]);
        let v = Vector::new([1., -2., 3.]);
        assert_eq!(Matrix::<f64, 3, 3>::IDENTITY * v, v);
        let b = m * v;
        let back = m.solve(&b).unwrap();
        for (g, w) in back.b.iter().zip(v.b.iter()) {
            assert!((g - w).abs() < 1e-12);
        }

        // Integer matrices act on integer vectors (Ring bound).
        let e = Matrix::<i64, 2, 2>::new([[1, 2], [3, 4]]);
        assert_eq!(e * Vector::new([1i64, 1]), Vector::new([3, 7]));
    }

    #[test]
    fn matrix_index_default_from() {
        let mut m: Matrix<f64, 2, 2> = [[1., 2.], [3., 4.]].into();
        assert_eq!(m[(0, 1)], 2.);
        assert_eq!(m[(1, 0)], 3.);
        m[(1, 1)] = 9.;
        assert_eq!(m.e[1][1], 9.);

        assert_eq!(Matrix::<f64, 2, 3>::default(), Matrix::ZERO);
        assert_eq!(
            Matrix::<i32, 2, 2>::default(),
            Matrix::new([[0, 0], [0, 0]])
        );
    }

    #[test]
    fn scalar_left_multiplication() {
        let a = Matrix::new([[1., 2.], [3., 4.]]);
        assert_eq!(2.0 * a, a * 2.0);
        assert_eq!(
            0.5_f32 * Matrix::new([[2_f32, 4.]]),
            Matrix::new([[1., 2.]])
        );

        let i = c64::new(0., 1.);
        let one = c64::new(1., 0.);
        let m = Matrix::new([[one, i]]);
        assert_eq!(i * m, Matrix::new([[i, -one]]));

        let j = crate::complex::c32::new(0., 1.);
        let n = Matrix::new([[crate::complex::c32::new(2., 0.)]]);
        assert_eq!(j * n, Matrix::new([[crate::complex::c32::new(0., 2.)]]));
    }

    #[test]
    fn matrix_sub_neg_scalar_mul_float() {
        let a = Matrix::new([[1., 2.], [3., 4.]]);
        let b = Matrix::new([[0.5, 1.], [1.5, 2.]]);
        assert_eq!(a - b, b);
        assert_eq!(-a, Matrix::new([[-1., -2.], [-3., -4.]]));
        assert_eq!(a * 0.5, b);
    }

    #[test]
    fn more_matrix_tests_assorted() {
        let x = Matrix::<f32, 3, 2>::new([[1., 2.], [3., 4.], [5., 6.]]);
        assert_eq!((x + Matrix::ZERO), x);

        let y = Matrix::new([[1., 2.], [3., 4.]]);
        assert_eq!((y * Matrix::<_, 2, 2>::IDENTITY).determinant(), -2.0);

        let a = Matrix::new([
            [1., 2., 3., 4.],
            [5., 6., 7., 8.],
            [9., 10., 11., 12.],
            [13., 14., 15., 16.],
        ]);
        let b = Matrix::new([
            [17., 18., 19., 20.],
            [21., 22., 23., 24.],
            [25., 26., 27., 28.],
            [29., 30., 31., 32.],
        ]);
        let c = Matrix::new([
            [250., 260., 270., 280.],
            [618., 644., 670., 696.],
            [986., 1028., 1070., 1112.],
            [1354., 1412., 1470., 1528.],
        ]);
        assert_eq!(a * b, c);
    }
}
