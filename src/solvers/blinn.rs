use std::marker::PhantomData;

use crate::{float::Float, number::Number, polynomial::Polynomial};

#[derive(Clone)]
pub struct Blinn<T: Number<Type = T>, const N: usize> {
    casper: PhantomData<T>,
}

impl<T: Number<Type = T>, const N: usize> Blinn<T, N>
where
    T: Float,
{
    #[inline]
    // #[target_feature(enable = "fma")]
    pub fn roots_quadratic(p: &Polynomial<T, N>) -> [T; N + 0_usize.pow(N as u32 - 1) - 1] {
        let mut output = [T::NAN; N + 0_usize.pow(N as u32 - 1) - 1]; // Ugly workaround for const generics

        // Quadratic, has either one or two real roots or two complex roots
        let [A, B, C] = [p.c[0], p.c[1] / (T::ONE + T::ONE), p.c[2]];
        let D = B.mul_add(B, -(A * C));
        if D.ge(&T::ZERO) {
            // Roots are real, use Blinn's homogeneous algorithm
            let E = D.sqrt();
            let [x1, w1]: [T; 2];
            let [x2, w2]: [T; 2];
            if B > T::ZERO {
                [x1, w1] = [-C, B + E];
                [x2, w2] = [-B - E, A];
            } else if B < T::ZERO {
                let F = -B + E;
                [x1, w1] = [F, A];
                [x2, w2] = [C, F];
            } else {
                if A.abs().ge(&C.abs()) {
                    let F = (-A * C).sqrt();
                    [x1, w1] = [F, A];
                    [x2, w2] = [-F, A];
                } else {
                    let F = (-A * C).sqrt();
                    [x1, w1] = [-C, F];
                    [x2, w2] = [C, F];
                }
            }
            output[0] = x1 / w1;
            output[1] = x2 / w2;
            return output;
            // return [x1 / w1, x2 / w2];
        } else {
            // Roots are complex
            return output;
            // return [T::NAN, T::NAN];
        }
    }

    #[inline]
    pub fn roots_quadratic_nopoly(a: T, b: T, c: T) -> [T; 2] {
        // Quadratic, has either one or two real roots or two complex roots
        let [A, B, C] = [a, b / (T::ONE + T::ONE), c];
        let D = B.mul_add(B, -(A * C));
        if D.ge(&T::ZERO) {
            // Roots are real, use Blinn's homogeneous algorithm
            let E = D.sqrt();
            let [x1, w1]: [T; 2];
            let [x2, w2]: [T; 2];
            if B > T::ZERO {
                [x1, w1] = [-C, B + E];
                [x2, w2] = [-B - E, A];
            } else if B < T::ZERO {
                let F = -B + E;
                [x1, w1] = [F, A];
                [x2, w2] = [C, F];
            } else {
                if A.abs().ge(&C.abs()) {
                    let F = (-A * C).sqrt();
                    [x1, w1] = [F, A];
                    [x2, w2] = [-F, A];
                } else {
                    let F = (-A * C).sqrt();
                    [x1, w1] = [-C, F];
                    [x2, w2] = [C, F];
                }
            }
            return [x1 / w1, x2 / w2];
        } else {
            // Roots are complex
            return [T::NAN, T::NAN];
        }
    }

    /// Slightly modified from Levien's version at https://github.com/linebender/kurbo/pull/224
    #[inline]
    pub fn roots_cubic(p: &Polynomial<T, N>) -> [T; N + 0_usize.pow(N as u32 - 1) - 1] {
        let mut output = [T::NAN; N + 0_usize.pow(N as u32 - 1) - 1];

        let a_inv = p.c[0].recip();
        let ONE_THIRD: T = T::ONE / (T::ONE + T::ONE + T::ONE); // Should be const but can't use T here
        let b: T = p.c[1] * (ONE_THIRD * a_inv);
        let c: T = p.c[2] * (ONE_THIRD * a_inv);
        let d: T = p.c[3] * a_inv;
        if !(b.is_finite() && c.is_finite() && d.is_finite()) {
            // cubic coefficient is zero or nearly so.
            let [r1, r2] = Blinn::<T, N>::roots_quadratic_nopoly(p.c[1], p.c[2], p.c[3]);
            // return [r1, r2, T::NAN];
            output[0] = r1;
            output[1] = r2;
            return output;
        }

        let h0: T = b * d - c * c;
        let h1 = (-c).mul_add(b, d);
        let h2 = (-b).mul_add(b, c);

        let todo: T = T::ONE + T::ONE + T::ONE + T::ONE;
        let h: T = todo * h0 * h2 - h1 * h1;
        // let dp = (-2.0 * b).mul_add(h2, h1);
        let todo = -(T::ONE + T::ONE);
        let dp: T = todo * b * h2 + h1;
        if h > T::ZERO {
            let t: T = h.sqrt().atan2(-dp) * ONE_THIRD;
            let (t_s, t_c) = t.sin_cos();
            let r0: T = t_c;
            let ps: T = t_s * (T::ONE + T::ONE + T::ONE).sqrt();
            let todo: T = T::ONE / (T::ONE + T::ONE);
            let r1: T = todo * (-t_c + ps);
            let r2: T = todo * (-t_c - ps);
            let todo = T::ONE + T::ONE;
            let s: T = todo * (-h2).sqrt();

            // return [s.mul_add(r0, -b), s.mul_add(r1, -b), s.mul_add(r2, -b)];
            output[0] = s.mul_add(r0, -b);
            output[1] = s.mul_add(r1, -b);
            output[2] = s.mul_add(r2, -b);
            return output;
        } else if h == T::ZERO {
            let s = (-h2).sqrt().copysign(dp);
            // return [s - b, s.mul_add(-2., -b), T::NAN];
            output[0] = s - b;
            let todo: T = -(T::ONE + T::ONE);
            output[1] = s.mul_add(todo, -b);
            return output;
        } else {
            let todo: T = -(T::ONE / (T::ONE + T::ONE + T::ONE + T::ONE));
            let rt = (todo * h).sqrt();
            let todo = -(T::ONE / (T::ONE + T::ONE));
            let r = todo * dp;
            let s = (r + rt).cbrt() + (r - rt).cbrt();
            // return [s - b, T::NAN, T::NAN];
            output[0] = s - b;
            return output;
        }
    }

    // #[inline]
    // pub fn roots_cubic_nopoly(a_: f64, b_: f64, c_: f64, d_: f64) -> [f64; 3] {
    //     let a_inv = a_.recip();
    //     const ONE_THIRD: f64 = 1. / 3.;
    //     let b = b_ * (ONE_THIRD * a_inv);
    //     let c = c_ * (ONE_THIRD * a_inv);
    //     let d = d_ * a_inv;
    //     if !(b.is_finite() && c.is_finite() && d.is_finite()) {
    //         // cubic coefficient is zero or nearly so.
    //         let [r1, r2] = Blinn::roots_quadratic_nopoly(b_, c_, d_);
    //         return [r1, r2, f64::NAN];
    //     }

    //     let h0 = b * d - c * c;
    //     let h1 = (-c).mul_add(b, d);
    //     let h2 = (-b).mul_add(b, c);

    //     let h = 4. * h0 * h2 - h1 * h1;
    //     let dp = (-2.0 * b).mul_add(h2, h1);
    //     if h > 0. {
    //         let t = h.sqrt().atan2(-dp) * ONE_THIRD;
    //         let (t_s, t_c) = t.sin_cos();
    //         let r0 = t_c;
    //         let ps = t_s * 3_f64.sqrt();
    //         let r1 = 0.5 * (-t_c + ps);
    //         let r2 = 0.5 * (-t_c - ps);
    //         let s = 2.0 * (-h2).sqrt();

    //         return [s.mul_add(r0, -b), s.mul_add(r1, -b), s.mul_add(r2, -b)];
    //     } else if h == 0. {
    //         let s = (-h2).sqrt().copysign(dp);
    //         return [s - b, s.mul_add(-2., -b), f64::NAN];
    //     } else {
    //         let rt = (-0.25 * h).sqrt();
    //         let r = -0.5 * dp;
    //         let s = (r + rt).cbrt() + (r - rt).cbrt();
    //         return [s - b, f64::NAN, f64::NAN];
    //     }
    // }
}
