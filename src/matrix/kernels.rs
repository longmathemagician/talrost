//! Fixed-size multiply kernels dispatched by `min_specialization` on the
//! concrete const dimensions (nightly-only, `feature = "specialization"`).
//!
//! The dimension logic is *type-checked*: a 3×2 · 2×2 product cannot reach
//! the 2×2 kernel because `Matrix<T, 3, 2>` only matches the `default` impl.
//!
//! Honest caveats (see REVIEW.md §4.1): these kernels trade multiplications
//! for many additions. At 2×2–4×4 with scalar floats they are typically
//! *slower* than `mul_naive` with FMA on modern hardware and numerically less
//! stable, and the 47-multiplication AlphaTensor decomposition was found for
//! ℤ/2ℤ, not floats (it is still exact over ℝ, just not better-rounded).
//! They are kept as a demonstration of specialization-based kernel dispatch,
//! not as a default performance win.
//!
//! Bounds: everything here is `T: Ring` (the kernels only ever use `+`, `-`,
//! `*`, and `T::ZERO`). The bound cannot be `Scalar` while the user-facing
//! `Mul` is `Ring` (§5.6, integer matmul): `min_specialization` rejects
//! specializing impls that add non-marker trait bounds, so a `Scalar`-bound
//! fast path per size is not expressible here.

use super::{mul_naive, Matrix};
use crate::algebra::Ring;

pub(super) trait Gemm<Rhs> {
    type Output;
    fn gemm(self, rhs: Rhs) -> Self::Output;
}

impl<T: Ring, const M: usize, const K: usize, const N: usize> Gemm<Matrix<T, K, N>>
    for Matrix<T, M, K>
{
    type Output = Matrix<T, M, N>;

    default fn gemm(self, x: Matrix<T, K, N>) -> Matrix<T, M, N> {
        mul_naive(&self, &x)
    }
}

impl<T: Ring> Gemm<Matrix<T, 2, 2>> for Matrix<T, 2, 2> {
    // Strassen (7 multiplications)
    fn gemm(self, x: Matrix<T, 2, 2>) -> Matrix<T, 2, 2> {
        let m1 = (self.e[0][0] + self.e[1][1]) * (x.e[0][0] + x.e[1][1]);
        let m2 = (self.e[1][0] + self.e[1][1]) * x.e[0][0];
        let m3 = self.e[0][0] * (x.e[0][1] - x.e[1][1]);
        let m4 = self.e[1][1] * (x.e[1][0] - x.e[0][0]);
        let m5 = (self.e[0][0] + self.e[0][1]) * x.e[1][1];
        let m6 = (self.e[1][0] - self.e[0][0]) * (x.e[0][0] + x.e[0][1]);
        let m7 = (self.e[0][1] - self.e[1][1]) * (x.e[1][0] + x.e[1][1]);

        let mut e = [[T::ZERO; 2]; 2];
        e[0][0] = m1 + m4 - m5 + m7;
        e[0][1] = m3 + m5;
        e[1][0] = m2 + m4;
        e[1][1] = m1 - m2 + m3 + m6;
        Matrix { e }
    }
}

impl<T: Ring> Gemm<Matrix<T, 3, 3>> for Matrix<T, 3, 3> {
    // Laderman (23 multiplications)
    fn gemm(self, x: Matrix<T, 3, 3>) -> Matrix<T, 3, 3> {
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

        let mut e = [[T::ZERO; 3]; 3];
        e[0][0] = m6 + m14 + m19;
        e[0][1] = m1 + m4 + m5 + m6 + m12 + m14 + m15;
        e[0][2] = m6 + m7 + m9 + m10 + m14 + m16 + m18;
        e[1][0] = m2 + m3 + m4 + m6 + m14 + m16 + m17;
        e[1][1] = m2 + m4 + m5 + m6 + m20;
        e[1][2] = m14 + m16 + m17 + m18 + m21;
        e[2][0] = m6 + m7 + m8 + m11 + m12 + m13 + m14;
        e[2][1] = m12 + m13 + m14 + m15 + m22;
        e[2][2] = m6 + m7 + m8 + m9 + m23;
        Matrix { e }
    }
}

impl<T: Ring> Gemm<Matrix<T, 4, 4>> for Matrix<T, 4, 4> {
    // AlphaTensor-style (49 multiplications over ℝ)
    fn gemm(self, x: Matrix<T, 4, 4>) -> Matrix<T, 4, 4> {
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

        let mut e = [[T::ZERO; 4]; 4];
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
        Matrix { e }
    }
}
