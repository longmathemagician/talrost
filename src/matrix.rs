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

// (M×K) · (K×N) → (M×N), the conventional shape signature.
impl<T: Scalar, const M: usize, const K: usize, const N: usize> Mul<Matrix<T, K, N>>
    for Matrix<T, M, K>
{
    type Output = Matrix<T, M, N>;

    fn mul(self, x: Matrix<T, K, N>) -> Self::Output {
        if M == 2 && K == 2 && N == 2 {
            // Strassen
            let m1 = (self.e[0][0] + self.e[1][1]) * (x.e[0][0] + x.e[1][1]);
            let m2 = (self.e[1][0] + self.e[1][1]) * x.e[0][0];
            let m3 = self.e[0][0] * (x.e[0][1] - x.e[1][1]);
            let m4 = self.e[1][1] * (x.e[1][0] - x.e[0][0]);
            let m5 = (self.e[0][0] + self.e[0][1]) * x.e[1][1];
            let m6 = (self.e[1][0] - self.e[0][0]) * (x.e[0][0] + x.e[0][1]);
            let m7 = (self.e[0][1] - self.e[1][1]) * (x.e[1][0] + x.e[1][1]);

            let mut e = [[T::ZERO; N]; M];
            e[0][0] = m1 + m4 - m5 + m7;
            e[0][1] = m3 + m5;
            e[1][0] = m2 + m4;
            e[1][1] = m1 - m2 + m3 + m6;
            Self::Output { e }
        } else if M == 3 && K == 3 && N == 3 {
            // Laderman
            let m1 = (self.e[0][0] + self.e[0][1] + self.e[0][2]
                - self.e[1][0]
                - self.e[1][1]
                - self.e[2][1]
                - self.e[2][2])
                * x.e[1][1];
            let m2 = (self.e[0][0] - self.e[1][0]) * (x.e[1][1] - x.e[0][1]);
            let m3 = self.e[1][1]
                * (-x.e[0][0] + x.e[0][1] + x.e[1][0] - x.e[1][1] - x.e[1][2] - x.e[2][0]
                    + x.e[2][2]);
            let m4 =
                (-self.e[0][0] + self.e[1][0] + self.e[1][1]) * (x.e[0][0] - x.e[0][1] + x.e[1][1]);
            let m5 = (self.e[1][0] + self.e[1][1]) * (-x.e[0][0] + x.e[0][1]);
            let m6 = self.e[0][0] * x.e[0][0];
            let m7 =
                (-self.e[0][0] + self.e[2][0] + self.e[2][1]) * (x.e[0][0] - x.e[0][2] + x.e[1][2]);
            let m8 = (-self.e[0][0] + self.e[2][0]) * (x.e[0][2] - x.e[1][2]);
            let m9 = (self.e[2][0] + self.e[2][1]) * (-x.e[0][0] + x.e[0][2]);
            let m10 = (self.e[0][0] + self.e[0][1] + self.e[0][2]
                - self.e[1][1]
                - self.e[1][2]
                - self.e[2][0]
                - self.e[2][1])
                * x.e[1][2];
            let m11 = self.e[2][1]
                * (-x.e[0][0] + x.e[0][2] + x.e[1][0] - x.e[1][1] - x.e[1][2] - x.e[2][0]
                    + x.e[2][1]);
            let m12 =
                (-self.e[0][2] + self.e[2][1] + self.e[2][2]) * (x.e[1][1] + x.e[2][0] - x.e[2][1]);
            let m13 = (self.e[0][2] - self.e[2][2]) * (x.e[1][1] - x.e[2][1]);
            let m14 = self.e[0][2] * x.e[2][0];
            let m15 = (self.e[2][1] + self.e[2][2]) * (-x.e[2][0] + x.e[2][1]);
            let m16 =
                (-self.e[0][2] + self.e[1][1] + self.e[1][2]) * (x.e[1][2] + x.e[2][0] - x.e[2][2]);
            let m17 = (self.e[0][2] - self.e[1][2]) * (x.e[1][2] - x.e[2][2]);
            let m18 = (self.e[1][1] + self.e[1][2]) * (-x.e[2][0] + x.e[2][2]);
            let m19 = self.e[0][1] * x.e[1][0];
            let m20 = self.e[1][2] * x.e[2][1];
            let m21 = self.e[1][0] * x.e[0][2];
            let m22 = self.e[2][0] * x.e[0][1];
            let m23 = self.e[2][2] * x.e[2][2];

            let mut e = [[T::ZERO; N]; M];
            e[0][0] = m6 + m14 + m19;
            e[0][1] = m1 + m4 + m5 + m6 + m12 + m14 + m15;
            e[0][2] = m6 + m7 + m9 + m10 + m14 + m16 + m18;
            e[1][0] = m2 + m3 + m4 + m6 + m14 + m16 + m17;
            e[1][1] = m2 + m4 + m5 + m6 + m20;
            e[1][2] = m14 + m16 + m17 + m18 + m21;
            e[2][0] = m6 + m7 + m8 + m11 + m12 + m13 + m14;
            e[2][1] = m12 + m13 + m14 + m15 + m22;
            e[2][2] = m6 + m7 + m8 + m9 + m23;
            Self::Output { e }
        } else if M == 4 && K == 4 && N == 4 {
            // AlphaTensor
            let h1 = (self.e[0][0] + self.e[2][0]) * (x.e[0][0] + x.e[2][0]);
            let h2 =
                (self.e[0][0] - self.e[0][2] + self.e[2][0]) * (x.e[0][0] - x.e[0][2] + x.e[2][0]);
            let h3 = (-self.e[0][2]) * (x.e[0][0] - x.e[0][2] + x.e[2][0] - x.e[2][2]);
            let h4 = self.e[2][2] * x.e[2][2];
            let h5 = (-self.e[2][0]) * (-x.e[0][2]);
            let h6 = (self.e[0][0] - self.e[0][2] + self.e[2][0] - self.e[2][2]) * (-x.e[2][0]);
            let h7 = (-self.e[1][0] + self.e[1][1] - self.e[1][2] - self.e[1][3])
                * (-x.e[1][0] + x.e[1][1] - x.e[1][2] - x.e[1][3]);
            let h8 = (-self.e[1][0] + self.e[1][1] - self.e[1][2] - self.e[1][3] - self.e[3][0]
                + self.e[3][1])
                * (-x.e[1][0] + x.e[1][1] - x.e[1][2] - x.e[1][3] - x.e[3][0] + x.e[3][1]);
            let h9 = (self.e[0][0] - self.e[0][2]) * (x.e[0][0] - x.e[0][2]);
            let h10 = (-self.e[1][0] + self.e[1][1] - self.e[3][0] + self.e[3][1])
                * (-x.e[1][0] + x.e[1][1] - x.e[3][0] + x.e[3][1]);
            let h11 = (self.e[3][0] - self.e[3][1]) * (-x.e[1][2] - x.e[1][3]);
            let h12 = (-self.e[1][0] + self.e[1][1] - self.e[1][2] - self.e[1][3] - self.e[3][0]
                + self.e[3][1]
                - self.e[3][2]
                - self.e[3][3])
                * (x.e[3][0] - x.e[3][1]);
            let h13 = (-self.e[1][2] - self.e[1][3])
                * (-x.e[1][0] + x.e[1][1] - x.e[1][2] - x.e[1][3] - x.e[3][0] + x.e[3][1]
                    - x.e[3][2]
                    - x.e[3][3]);
            let h14 = (self.e[0][0] - self.e[0][1] + self.e[1][0] - self.e[1][1])
                * (-x.e[0][1] - x.e[0][3]);
            let h15 = (-self.e[0][1] - self.e[0][3]) * (-x.e[1][0]);
            let h16 = (self.e[0][1] + self.e[0][3] - self.e[1][0]
                + self.e[1][1]
                + self.e[1][2]
                + self.e[1][3])
                * (x.e[0][1] + x.e[0][3] - x.e[1][0] + x.e[1][1] + x.e[1][2] + x.e[1][3]);
            let h17 = (self.e[0][1] + self.e[0][3] - self.e[1][0]
                + self.e[1][1]
                + self.e[1][2]
                + self.e[1][3]
                + self.e[2][1]
                + self.e[3][0]
                - self.e[3][1])
                * (x.e[0][1] + x.e[0][3] - x.e[1][0]
                    + x.e[1][1]
                    + x.e[1][2]
                    + x.e[1][3]
                    + x.e[2][1]
                    + x.e[3][0]
                    - x.e[3][1]);
            let h18 = (self.e[0][1] - self.e[1][0] + self.e[1][1] + self.e[2][1] + self.e[3][0]
                - self.e[3][1])
                * (x.e[0][1] - x.e[1][0] + x.e[1][1] + x.e[2][1] + x.e[3][0] - x.e[3][1]);
            let h19 = (self.e[0][3] + self.e[1][2] + self.e[1][3])
                * (x.e[0][1] + x.e[0][3] - x.e[1][0]
                    + x.e[1][1]
                    + x.e[1][2]
                    + x.e[1][3]
                    + x.e[2][1]
                    + x.e[2][3]
                    + x.e[3][0]
                    - x.e[3][1]
                    - x.e[3][2]
                    - x.e[3][3]);
            let h20 = (self.e[0][1] + self.e[0][3] - self.e[1][0]
                + self.e[1][1]
                + self.e[1][2]
                + self.e[1][3]
                + self.e[2][1]
                + self.e[2][3]
                + self.e[3][0]
                - self.e[3][1]
                - self.e[3][2]
                - self.e[3][3])
                * (x.e[2][1] + x.e[3][0] - x.e[3][1]);
            let h21 =
                (self.e[2][1] + self.e[3][0] - self.e[3][1]) * (x.e[0][3] + x.e[1][2] + x.e[1][3]);
            let h22 = (self.e[0][1] + self.e[0][3] + self.e[1][1] + self.e[1][3])
                * (x.e[0][1] + x.e[0][3] + x.e[1][1] + x.e[1][3]);
            let h23 = (self.e[0][1] + self.e[0][3] + self.e[1][1] + self.e[1][3] + self.e[2][1]
                - self.e[3][1])
                * (x.e[0][1] + x.e[0][3] + x.e[1][1] + x.e[1][3] + x.e[2][1] - x.e[3][1]);
            let h24 = (self.e[0][3] + self.e[1][3])
                * (x.e[0][1] + x.e[0][3] + x.e[1][1] + x.e[1][3] + x.e[2][1] + x.e[2][3]
                    - x.e[3][1]
                    - x.e[3][3]);
            let h25 = (self.e[0][1]
                + self.e[0][3]
                + self.e[1][1]
                + self.e[1][3]
                + self.e[2][1]
                + self.e[2][3]
                - self.e[3][1]
                - self.e[3][3])
                * (x.e[2][1] - x.e[3][1]);
            let h26 = (self.e[2][1] - self.e[3][1]) * (x.e[0][3] + x.e[1][3]);
            let h27 = (self.e[2][3] - self.e[3][3]) * (x.e[2][3] - x.e[3][3]);
            let h28 =
                (self.e[2][3] - self.e[3][2] - self.e[3][3]) * (x.e[2][3] - x.e[3][2] - x.e[3][3]);
            let h29 = (self.e[0][3] + self.e[2][3]) * (-x.e[3][2]);
            let h30 = (self.e[0][2]
                + self.e[0][3]
                + self.e[1][2]
                + self.e[1][3]
                + self.e[2][2]
                + self.e[2][3]
                - self.e[3][2]
                - self.e[3][3])
                * (x.e[0][3] + x.e[2][3]);
            let h31 = (self.e[0][0] - self.e[0][1] - self.e[0][2] - self.e[0][3] + self.e[1][0]
                - self.e[1][1]
                - self.e[1][2]
                - self.e[1][3]
                + self.e[2][0]
                - self.e[2][1]
                - self.e[2][2]
                - self.e[2][3]
                - self.e[3][0]
                + self.e[3][1]
                + self.e[3][2]
                + self.e[3][3])
                * x.e[0][3];
            let h32 = -self.e[3][2]
                * (x.e[0][2] + x.e[0][3] + x.e[1][2] + x.e[1][3] + x.e[2][2] + x.e[2][3]
                    - x.e[3][2]
                    - x.e[3][3]);
            let h33 = self.e[0][3] * (-x.e[1][0] + x.e[3][0]);
            let h34 = (self.e[0][3] - self.e[2][1]) * (-x.e[1][0] + x.e[3][0] - x.e[3][2]);
            let h35 = (self.e[0][2] + self.e[0][3] + self.e[1][2] + self.e[1][3] - self.e[2][0]
                + self.e[2][1]
                + self.e[2][2]
                + self.e[2][3]
                + self.e[3][0]
                - self.e[3][1]
                - self.e[3][2]
                - self.e[3][3])
                * (x.e[0][3] - x.e[2][1]);
            let h36 = (-self.e[2][0] + self.e[2][1] + self.e[2][2] + self.e[2][3] + self.e[3][0]
                - self.e[3][1]
                - self.e[3][2]
                - self.e[3][3])
                * x.e[2][1];
            let h37 = (self.e[0][1] + self.e[2][1]) * (x.e[1][2]);
            let h38 = (self.e[2][1] + self.e[2][3]) * (x.e[3][0] - x.e[3][2]);
            let h39 = (-self.e[0][2] - self.e[0][3] - self.e[1][2] - self.e[1][3])
                * (x.e[2][1] + x.e[2][3]);
            let h40 = self.e[2][1] * (-x.e[1][0] + x.e[1][2] + x.e[3][0] - x.e[3][2]);
            let h41 = (-self.e[1][0]) * (x.e[0][0] - x.e[0][1] + x.e[1][0] - x.e[1][1]);
            let h42 = (-self.e[1][0] + self.e[3][0])
                * (x.e[0][0] - x.e[0][1] - x.e[0][2] - x.e[0][3] + x.e[1][0]
                    - x.e[1][1]
                    - x.e[1][2]
                    - x.e[1][3]
                    + x.e[2][0]
                    - x.e[2][1]
                    - x.e[2][2]
                    - x.e[2][3]
                    - x.e[3][0]
                    + x.e[3][1]
                    + x.e[3][2]
                    + x.e[3][3]);
            let h43 = (-self.e[1][0] + self.e[3][0] - self.e[3][2])
                * (x.e[0][2] + x.e[0][3] + x.e[1][2] + x.e[1][3] - x.e[2][0]
                    + x.e[2][1]
                    + x.e[2][2]
                    + x.e[2][3]
                    + x.e[3][0]
                    - x.e[3][1]
                    - x.e[3][2]
                    - x.e[3][3]);
            let h44 = (self.e[0][1] + self.e[1][1] + self.e[2][1] - self.e[3][1])
                * (x.e[0][1] + x.e[1][1] + x.e[2][1] - x.e[3][1]);
            let h45 = (-self.e[1][0] + self.e[1][2] + self.e[3][0] - self.e[3][2])
                * (-x.e[2][0] + x.e[2][1] + x.e[2][2] + x.e[2][3] + x.e[3][0]
                    - x.e[3][1]
                    - x.e[3][2]
                    - x.e[3][3]);
            let h46 = (-self.e[2][0] + self.e[2][1] + self.e[3][0] - self.e[3][1])
                * (-x.e[0][1] - x.e[2][1]);
            let h47 =
                (self.e[3][0] - self.e[3][2]) * (-x.e[0][2] - x.e[0][3] - x.e[1][2] - x.e[1][3]);
            let h48 = (-self.e[3][2] - self.e[3][3]) * (-x.e[3][2] - x.e[3][3]);

            let h49 = (-self.e[1][2]) * (-x.e[2][0] + x.e[2][1] + x.e[3][0] - x.e[3][1]);

            let mut e = [[T::ZERO; N]; M];
            e[0][0] = h1 - h2 - h5 + h9 + h15 + h33;
            e[0][1] =
                -h7 + h8 - h10 + h11 - h14 + h15 + h16 - h17 + h18 + h21 - h31 + h33 - h35 - h36;
            e[0][2] = h1 - h2 + h3 - h5 + h33 - h34 + h37 - h40;
            e[0][3] = h8 - h10 + h11 - h13 + h17 - h18 - h19 - h21 + h31 - h33 + h34 + h35 + h36
                - h37
                - h39
                + h40;
            e[1][0] = -h15 - h16 + h17 - h18 - h21 + h22 - h23 + h26 - h33 - h41 + h44 + h49;
            e[1][1] =
                h7 - h8 + h10 - h11 - h15 - h16 + h17 - h18 - h21 + h22 - h23 + h26 - h33 + h44;
            e[1][2] =
                h17 - h18 - h19 - h21 - h23 + h24 + h26 - h33 + h34 - h37 + h40 - h43 + h44 + h45
                    - h47
                    + h49;
            e[1][3] = -h8 + h10 + -h11 + h13 + -h17 + h18 + h19 + h21 + h23 - h24 - h26 + h33 - h34
                + h37
                - h40
                - h44;
            e[2][0] = h2 + h5 + h6 - h9 - h29 - h33 + h34 + h38;
            e[2][1] =
                -h7 + h8 + h11 + h12 - h16 + h17 - h20 - h21 - h29 - h33 + h34 + h36 + h38 + h46;
            e[2][3] = h11 + h21 - h28 + h29 + h30 + h33 - h34 - h35 - h36 + h39 - h40 + h48;
            e[2][2] = h4 + h5 - h29 - h33 + h34 + h40;
            e[3][0] = -h16 + h17 - h20 - h21 + h22 - h23 + h25 + h26 - h29 - h32 - h33 + h34 + h38
                - h41
                + h42
                + h43;
            e[3][1] =
                -h7 + h8 + h11 + h12 - h16 + h17 - h20 - h21 + h22 - h23 + h25 + h26 - h29 - h33
                    + h34
                    + h38;
            e[3][2] = (-h21) + h26 - h27 + h28 - h29 - h32 - h33 + h34 + h40 - h47;
            e[3][3] = h11 + h21 - h26 + h27 - h28 + h29 + h33 - h34 - h40 + h48;
            Self::Output { e }
        } else {
            // Standard iterative form
            let mut e = [[T::ZERO; N]; M];
            for i in 0..M {
                for j in 0..N {
                    for k in 0..K {
                        e[i][j] += self.e[i][k] * x.e[k][j];
                    }
                }
            }
            Self::Output { e }
        }
    }
}

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

impl<T: Scalar + core::fmt::Display, const M: usize, const N: usize> core::fmt::Display
    for Matrix<T, M, N>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        assert_ne!(M, 0);
        assert_ne!(N, 0);
        let mut output = String::from("\n");
        for row in self.e {
            output.push('|');
            for e in row {
                output.push_str(&format!("{}, ", e));
            }
            output.pop();
            output.pop();
            output.push_str("|\n");
        }
        f.write_str(&output)
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
