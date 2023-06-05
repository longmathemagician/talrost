use std::ops::{Add, Mul};

use crate::number::NumberTrait;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Matrix<T: NumberTrait<NumberType = T>, const M: usize, const N: usize> {
    pub e: [[T; M]; N],
}

impl<T: NumberTrait<NumberType = T>, const M: usize, const N: usize> Matrix<T, M, N> {
    pub const ZERO: Matrix<T, M, N> = Self {
        e: [[T::ZERO; M]; N],
    };

    pub const IDENTITY: Matrix<T, M, N> = Matrix::identity();
    const fn identity() -> Self {
        if M != N {
            panic!()
        }
        let mut e = [[T::ZERO; M]; N];
        let mut i = 0;
        while i < M {
            e[i][i] = T::ONE;
            i += 1;
        }
        Self { e }
    }
    pub fn new(e: [[T; M]; N]) -> Self {
        Self { e }
    }

    pub fn transpose(&self) -> Self {
        todo!()
    }

    pub fn inverse(&self) -> Self {
        todo!()
    }

    pub fn determinant(&self) -> T {
        if M == 2 & N {
            self.e[0][0] * self.e[1][1] - self.e[0][1] * self.e[1][0]
        // } else if M == 3 & N {
        //     let m1 = self.e[1][1] * self.e[2][0];
        //     let ma1 = self.e[1][0].mul_add(self.e[2][1], -m1);
        //     let m2 = self.e[1][2] * self.e[2][0];
        //     let ma2 = self.e[1][0].mul_add(self.e[2][2], -m2);
        //     let m3 = self.e[1][2] * self.e[2][1];
        //     let ma3 = self.e[1][1].mul_add(self.e[2][2], -m3);
        //     let m4 = self.e[0][2] * ma1;
        //     let ma4 = self.e[0][1].mul_add(ma2, -m4);
        //     self.e[0][0].mul_add(ma3, -ma4)
        } else {
            todo!()
        }
    }
}

impl<T: NumberTrait<NumberType = T>, const M: usize, const N: usize, const O: usize>
    Mul<Matrix<T, O, M>> for Matrix<T, M, N>
{
    type Output = Matrix<T, O, N>;

    fn mul(self, x: Matrix<T, O, M>) -> Self::Output {
        if M == 2 & N & O {
            // Strassen
            let m1 = (self.e[0][0] + self.e[1][1]) * (x.e[0][0] + x.e[1][1]);
            let m2 = (self.e[1][0] + self.e[1][1]) * x.e[0][0];
            let m3 = self.e[0][0] * (x.e[0][1] - x.e[1][1]);
            let m4 = self.e[1][1] * (x.e[1][0] - x.e[0][0]);
            let m5 = (self.e[0][0] + self.e[0][1]) * x.e[1][1];
            let m6 = (self.e[1][0] - self.e[0][0]) * (x.e[0][0] + x.e[0][1]);
            let m7 = (self.e[0][1] - self.e[1][1]) * (x.e[1][0] + x.e[1][1]);

            let mut e = [[T::ZERO; O]; N];
            e[0][0] = m1 + m4 - m5 + m7;
            e[0][1] = m3 + m5;
            e[1][0] = m2 + m4;
            e[1][1] = m1 - m2 + m3 + m6;
            Self::Output { e }
        } else if M == 3 & N & O {
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

            let mut e = [[T::ZERO; O]; N];
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
        } else if M == 4 & N & O {
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

            let mut e = [[T::ZERO; O]; N];
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
            let mut e = [[T::ZERO; O]; N];
            for i in 0..O {
                for j in 0..N {
                    for k in 0..M {
                        e[j][i] += self.e[j][k] * x.e[k][i];
                    }
                }
            }
            Self::Output { e }
        }
    }
}

impl<T: NumberTrait<NumberType = T>, const M: usize, const N: usize> Add<Matrix<T, M, N>>
    for Matrix<T, M, N>
{
    type Output = Matrix<T, M, N>;

    fn add(self, x: Matrix<T, M, N>) -> Self::Output {
        let mut e = [[T::ZERO; M]; N];
        for i in 0..M {
            for j in 0..N {
                e[j][i] = self.e[j][i] + x.e[j][i];
            }
        }
        Self::Output { e }
    }
}

impl<T: NumberTrait<NumberType = T>, const M: usize, const N: usize> core::fmt::Display
    for Matrix<T, M, N>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        assert_ne!(M, 0);
        assert_ne!(N, 0);
        let mut output = String::from("\n");
        for i in 0..N {
            output.push_str("|");
            for e in self.e[i] {
                // output.push_str(&format!("{}, ", format_f64(e, 7)));
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
    fn more_matrix_tests_assorted() {
        let x = Matrix::<f32, 2, 3>::new([[1., 2.], [3., 4.], [5., 6.]]);
        assert_eq!((x + Matrix::ZERO), x);

        let y = Matrix::new([[1., 2.], [3., 4.]]);
        assert_eq!((y * Matrix::<_, 2, _>::IDENTITY).determinant(), -2.0);

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
