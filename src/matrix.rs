use core::ops::{Add, Mul};

use crate::algebra::Monoid;
use crate::scalar::Scalar;

/// An M×N matrix in the conventional row-major sense: `M` rows of `N`
/// columns, stored as `e: [[T; N]; M]` (outer index = row).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Matrix<T, const M: usize, const N: usize> {
    pub e: [[T; N]; M],
}

impl<T: Scalar, const M: usize, const N: usize> Matrix<T, M, N> {
    pub const ZERO: Matrix<T, M, N> = Self {
        e: [[T::ZERO; N]; M],
    };

    pub fn new(e: [[T; N]; M]) -> Self {
        Self { e }
    }

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

/// Square-matrix operations. Attempting these on a non-square matrix is a
/// *compile* error now, not a runtime panic:
///
/// ```compile_fail
/// use talrost::matrix::Matrix;
/// let a = Matrix::<f64, 2, 3>::new([[1., 2., 3.], [4., 5., 6.]]);
/// let _ = a.determinant(); // no method: determinant requires Matrix<T, N, N>
/// ```
impl<T: Scalar, const N: usize> Matrix<T, N, N> {
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

    /// Returns the determinant. Degrees 1–3 use closed forms (the `match` is
    /// constant-folded after monomorphization); larger matrices use LU
    /// decomposition with partial pivoting, choosing pivots by `norm_sqr` so
    /// the same code works for complex scalars. Returns `T::ZERO` for
    /// singular matrices.
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
            _ => self.determinant_lu(),
        }
    }

    /// LU decomposition (Doolittle, partial pivoting by `norm_sqr`); the
    /// determinant is the signed product of the pivots.
    fn determinant_lu(&self) -> T {
        let mut a = self.e;
        let mut negate = false;

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
                return T::ZERO; // singular
            }
            if pivot != k {
                a.swap(k, pivot);
                negate = !negate;
            }
            for r in (k + 1)..N {
                let factor = a[r][k] / a[k][k];
                for c in k..N {
                    a[r][c] = a[r][c] - factor * a[k][c];
                }
            }
        }

        let mut det = T::ONE;
        for (k, row) in a.iter().enumerate() {
            det *= row[k];
        }
        if negate {
            -det
        } else {
            det
        }
    }

    /// Returns the inverse via Gauss–Jordan elimination with partial
    /// pivoting by `norm_sqr`, or `None` if the matrix is singular.
    pub fn inverse(&self) -> Option<Self> {
        let mut a = self.e;
        let mut inv = Self::IDENTITY.e;

        for k in 0..N {
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
                inv.swap(k, pivot);
            }

            let d = a[k][k];
            for c in 0..N {
                a[k][c] /= d;
                inv[k][c] /= d;
            }
            for r in 0..N {
                if r == k {
                    continue;
                }
                let factor = a[r][k];
                for c in 0..N {
                    a[r][c] = a[r][c] - factor * a[k][c];
                    inv[r][c] = inv[r][c] - factor * inv[k][c];
                }
            }
        }

        Some(Self { e: inv })
    }
}

/// The naive triple loop, accumulating with [`Scalar::mul_add_fast`] (a
/// hardware FMA where the target has one, plain multiply-add elsewhere).
/// This is the only multiply kernel in default builds, and the `default`
/// (non-square / non-special-size) kernel under `feature = "specialization"`.
fn mul_naive<T: Scalar, const M: usize, const K: usize, const N: usize>(
    a: &Matrix<T, M, K>,
    b: &Matrix<T, K, N>,
) -> Matrix<T, M, N> {
    let mut e = [[T::ZERO; N]; M];
    for (i, row) in e.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            let mut acc = T::ZERO;
            for k in 0..K {
                acc = a.e[i][k].mul_add_fast(b.e[k][j], acc);
            }
            *v = acc;
        }
    }
    Matrix { e }
}

// (M×K) · (K×N) → (M×N), the conventional shape signature.
//
// Default builds use `mul_naive` unconditionally: at these sizes the naive
// loop with FMA accumulation is the numerically stable (and usually fastest)
// choice. With `feature = "specialization"` (nightly) dispatch goes through
// the internal `Gemm` trait, whose impls for concrete square sizes select the
// multiplication-saving kernels in [`kernels`].
impl<T: Scalar, const M: usize, const K: usize, const N: usize> Mul<Matrix<T, K, N>>
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

impl<T: Scalar, const M: usize, const N: usize> Add<Matrix<T, M, N>> for Matrix<T, M, N> {
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

// Writes straight to the `Formatter` (no allocation) so it works in `no_std`.
impl<T: Scalar + core::fmt::Display, const M: usize, const N: usize> core::fmt::Display
    for Matrix<T, M, N>
{
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
