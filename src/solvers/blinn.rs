// Non-snake-case names (A, B, C, D, E, F, ONE_THIRD, ...) mirror the notation of
// Blinn's homogeneous quadratic/cubic formulation.
#![allow(non_snake_case)]

use crate::polynomial::Polynomial;
use crate::real::Real;

/// Real roots of the quadratic `p.c[0]·x² + p.c[1]·x + p.c[2]`, in the
/// solver's native order. Slots without a real root are `NAN`; a degenerate
/// (linear) input yields one finite root and one infinity.
#[inline]
pub fn roots_quadratic<T: Real>(p: &Polynomial<T, 3>) -> [T; 2] {
    roots_quadratic_coeffs(p.c[0], p.c[1], p.c[2])
}

/// [`roots_quadratic`] on raw coefficients `a·x² + b·x + c`.
#[inline]
pub fn roots_quadratic_coeffs<T: Real>(a: T, b: T, c: T) -> [T; 2] {
    // Quadratic, has either one or two real roots or two complex roots
    let [A, B, C] = [a, b / T::from_u32(2), c];
    let D = B.mul_add(B, -(A * C));
    if D >= T::ZERO {
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
        } else if A.abs() >= C.abs() {
            let F = (-A * C).sqrt();
            [x1, w1] = [F, A];
            [x2, w2] = [-F, A];
        } else {
            let F = (-A * C).sqrt();
            [x1, w1] = [-C, F];
            [x2, w2] = [C, F];
        }
        [x1 / w1, x2 / w2]
    } else {
        // Roots are complex
        [T::NAN, T::NAN]
    }
}

/// Real roots of the cubic `p.c[0]·x³ + … + p.c[3]`, in the solver's native
/// order; slots without a real root are `NAN`.
///
/// Slightly modified from Levien's version at
/// https://github.com/linebender/kurbo/pull/224
#[inline]
pub fn roots_cubic<T: Real>(p: &Polynomial<T, 4>) -> [T; 3] {
    let mut output = [T::NAN; 3];

    let a_inv = p.c[0].recip();
    let ONE_THIRD: T = T::ONE / T::from_u32(3); // Should be const but can't use T here
    let b: T = p.c[1] * (ONE_THIRD * a_inv);
    let c: T = p.c[2] * (ONE_THIRD * a_inv);
    let d: T = p.c[3] * a_inv;
    if !(b.is_finite() && c.is_finite() && d.is_finite()) {
        // cubic coefficient is zero or nearly so.
        let [r1, r2] = roots_quadratic_coeffs(p.c[1], p.c[2], p.c[3]);
        output[0] = r1;
        output[1] = r2;
        return output;
    }

    let h0: T = b * d - c * c;
    let h1 = (-c).mul_add(b, d);
    let h2 = (-b).mul_add(b, c);

    let h: T = T::from_u32(4) * h0 * h2 - h1 * h1;
    // dp = (-2.0 * b).mul_add(h2, h1);
    let dp: T = -(T::from_u32(2)) * b * h2 + h1;
    if h > T::ZERO {
        let t: T = h.sqrt().atan2(-dp) * ONE_THIRD;
        let (t_s, t_c) = t.sin_cos();
        let r0: T = t_c;
        let ps: T = t_s * T::from_u32(3).sqrt();
        let half: T = T::ONE / T::from_u32(2);
        let r1: T = half * (-t_c + ps);
        let r2: T = half * (-t_c - ps);
        let s: T = T::from_u32(2) * (-h2).sqrt();

        output[0] = s.mul_add(r0, -b);
        output[1] = s.mul_add(r1, -b);
        output[2] = s.mul_add(r2, -b);
        output
    } else if h == T::ZERO {
        let s = (-h2).sqrt().copysign(dp);
        output[0] = s - b;
        output[1] = s.mul_add(-(T::from_u32(2)), -b);
        output
    } else {
        let quarter: T = T::ONE / T::from_u32(4);
        let rt = (-(quarter) * h).sqrt();
        let half: T = T::ONE / T::from_u32(2);
        let r = -(half) * dp;
        let s = (r + rt).cbrt() + (r - rt).cbrt();
        output[0] = s - b;
        output
    }
}
