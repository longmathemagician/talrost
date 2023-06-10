use crate::solvers;

use crate::{float::Float, number::Number};

#[derive(Copy, Clone, Debug)]
pub struct Polynomial<T: Number<Type = T>, const N: usize>
where
    T: Float,
    [(); N]:,
{
    pub c: [T; N],
}

impl<T: Number<Type = T>, const N: usize> Polynomial<T, N>
where
    T: Float,
    [(); N]:,
{
    pub const fn from(c: [T; N]) -> Self {
        Self { c }
    }

    pub const fn new(c: [T; N]) -> Self {
        Self { c }
    }

    pub fn eval_quadratic(&self, x: T) -> T {
        self.c[0] * x.powi(2) + self.c[1] * x + self.c[2]
    }

    pub fn eval_cubic(&self, x: T) -> T {
        self.c[0] * x.powi(3) + self.c[1] * x.powi(2) + self.c[2] * x + self.c[3]
    }

    pub fn eval_quartic(&self, x: T) -> T {
        self.c[0] * x.powi(4)
            + self.c[1] * x.powi(3)
            + self.c[2] * x.powi(2)
            + self.c[3] * x
            + self.c[4]
    }

    pub fn eval(&self, x: T) -> T {
        match N {
            1 => self.c[0],
            2 => self.c[0] * x + self.c[1],
            3 => self.eval_quadratic(x),
            4 => self.eval_cubic(x),
            5 => self.eval_quartic(x),
            // _ => self
            //     .c
            //     .iter()
            //     .enumerate()
            //     .map(|(n, k)| k * x.powi((N - n) as i32))
            //     .sum::<f64>(),
            _ => todo!(),
        }
    }

    pub fn roots(&self, tol: T) -> [T; N + 0_usize.pow(N as u32 - 1) - 1] {
        let mut output = [T::NAN; N + 0_usize.pow(N as u32 - 1) - 1];
        let roots = match N {
            1 => self.root_constant(tol),
            2 => self.root_linear(tol),
            3 => solvers::blinn::Blinn::roots_quadratic(self),
            // 4 => solvers::yuksel::roots_cubic(self),
            _ => [T::NAN; N + 0_usize.pow(N as u32 - 1) - 1],
        };
        for (i, r) in roots.iter().enumerate().take(N) {
            output[i] = *r;
        }
        output
    }

    #[inline]
    fn root_constant(&self, tol: T) -> [T; N + 0_usize.pow(N as u32 - 1) - 1] {
        // Constant polynomial, has a root at x=0 IFF p(x) = 0
        let mut output = [T::NAN; N + 0_usize.pow(N as u32 - 1) - 1];
        if self.c[0].abs() <= tol {
            output[0] = T::ZERO;
        }
        output
    }

    #[inline]
    fn root_linear(&self, _tol: T) -> [T; N + 0_usize.pow(N as u32 - 1) - 1] {
        // Linear polynomial, has exactly one root at x = -b/a
        [-self.c[1] / self.c[0]; N + 0_usize.pow(N as u32 - 1) - 1]
    }
}

impl<T: Number<Type = T>, const N: usize> core::fmt::Display for Polynomial<T, N>
where
    T: Float,
    [(); N]:,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut output = String::new();
        for i in 0..N {
            if i != (N - 1) {
                output.push_str(&format!("{}×x^{} + ", self.c[i], (N - 1) - i));
            } else {
                output.push_str(&format!("{}", self.c[i]));
            }
        }
        f.write_str(&output)
    }
}

#[cfg(test)]
mod tests {
    use crate::complex::c32;

    use super::*;

    #[test]
    fn test_c32_polynomials() {
        let a: c32 = c32::new(1.0, 0.0);
        let b: c32 = c32::new(2.0, 0.0);
        let c: c32 = c32::new(3.0, 0.0);
        let _pc_2 = Polynomial::from([a, b, c]);

        let na: c32 = c32::from(a);
        let nb: c32 = c32::from(b);
        let nc: c32 = c32::from(c);
        let _pn_2 = Polynomial::from([na, nb, nc]);
    }
    #[test]
    fn test_f64_polynomials() {
        let p_0 = Polynomial::from([1.0]);
        let p_1 = Polynomial::from([1.0, 2.0]);
        let p_2 = Polynomial::from([1.0, 2.0, 3.0]);
        let p_3 = Polynomial::from([1.0, 2.0, 3.0, 4.0]);

        assert_eq!(p_0.c, [1.0]);
        assert_eq!(p_1.c, [1.0, 2.0]);
        assert_eq!(p_2.c, [1.0, 2.0, 3.0]);
        assert_eq!(p_3.c, [1.0, 2.0, 3.0, 4.0]);

        assert_eq!(p_0.to_string(), "1");
        assert_eq!(p_1.to_string(), "1×x^1 + 2");
        assert_eq!(p_2.to_string(), "1×x^2 + 2×x^1 + 3");
        assert_eq!(p_3.to_string(), "1×x^3 + 2×x^2 + 3×x^1 + 4");

        assert_eq!(p_0.eval(-3.0), 1.0);
        assert_eq!(p_0.eval(0.0), 1.0);
        assert_eq!(p_0.eval(1.0), 1.0);
        assert_eq!(p_0.eval(2.0), 1.0);

        assert_eq!(p_1.eval(-3.0), -1.0);
        assert_eq!(p_1.eval(0.0), 2.0);
        assert_eq!(p_1.eval(1.0), 3.0);
        assert_eq!(p_1.eval(2.0), 4.0);

        assert_eq!(p_2.eval(-3.0), 6.0);
        assert_eq!(p_2.eval(0.0), 3.0);
        assert_eq!(p_2.eval(1.0), 6.0);
        assert_eq!(p_2.eval(2.0), 11.0);

        assert_eq!(p_3.eval(-3.0), -14.0);
        assert_eq!(p_3.eval(0.0), 4.0);
        assert_eq!(p_3.eval(1.0), 10.0);
        assert_eq!(p_3.eval(2.0), 26.0);
    }

    #[test]
    fn roots_0() {
        let tol = 1e-7;

        // A nonzero constant polynomial
        let x = Polynomial::from([1.]);
        let r = x.root_constant(tol);
        // assert_eq!(r[0].is_nan(), true);
        assert_eq!(r.len(), 1);

        // Additive identity cast to a polynomial
        let y = Polynomial::from([0.]);
        let s = y.root_constant(tol);
        assert_eq!(s[0], 0.);
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn roots_1() {
        let tol = 1e-7;

        // Liner polynomial p(x) = x + 1
        let x = Polynomial::from([1., 1.]);
        let r = x.roots(tol);
        assert_eq!(r[0], -1.); // check root
        assert_eq!(r.len(), 1); // check length of returned array
    }

    #[test]
    fn roots_2_default() {
        let tol = 1e-7;

        // p(x) = 0x^2 + 1x + 1 with root -1
        let x = Polynomial::new([0., 1., 1.]);
        let r = x.roots(tol);
        assert_eq!(r[0], -1.); // check first root
                               // assert_eq!(r[1].is_finite(), false); // check second root
        assert_eq!(r.len(), 2); // check array length

        // Quadratic p(x) = x^2 - x - 12 with roots 4,-3
        let x = Polynomial::new([1., -1., -12.]);
        let r = x.roots(tol);
        assert_eq!(r[0], 4.);
        assert_eq!(r[1], -3.);

        // Quadratic p(x) = x^2 - 6x + 9 with root x = 3 with multiplicity 2
        let x = Polynomial::new([1., -6., 9.]);
        let r = x.roots(tol);
        assert_eq!(r[0], 3.);
        assert_eq!(r[1], 3.);

        // Quadratic p(x) = x^2 - 3x + 5 with complex roots
        let x = Polynomial::new([1., -3., 5.]);
        let r = x.roots(tol);
        assert_eq!(r[0].is_nan(), true);
        assert_eq!(r[1].is_nan(), true);
    }

    #[test]
    fn roots_2_yuksel() {
        // p(x) = 0x^2 + 1x + 1 with root -1
        let x = Polynomial::new([0., 1., 1.]);
        let r = solvers::yuksel::roots_quadratic(&x);
        assert_eq!(r[1], -1.);
        assert_eq!(r[0].is_finite(), false);
        assert_eq!(r.len(), 2);

        // Quadratic p(x) = x^2 - x - 12 with roots 4,-3
        let x = Polynomial::new([1., -1., -12.]);
        let r = solvers::yuksel::roots_quadratic(&x);
        assert_eq!(r[1], 4.);
        assert_eq!(r[0], -3.);

        // Quadratic p(x) = x^2 - 6x + 9 with root x = 3 with multiplicity 2
        let x = Polynomial::new([1., -6., 9.]);
        let r = solvers::yuksel::roots_quadratic(&x);
        assert_eq!(r[0], 3.);
        assert_eq!(r[1].is_nan(), true);

        // Quadratic p(x) = x^2 - 3x + 5 with complex roots
        let x = Polynomial::new([1., -3., 5.]);
        let r = solvers::yuksel::roots_quadratic(&x);
        assert_eq!(r[0].is_nan(), true);
        assert_eq!(r[1].is_nan(), true);
    }

    #[test]
    fn roots_3_generic() {
        let tol = f64::EPSILON;

        // Cubic p(x) = 1x^3 + 5x^2 + -14x + 0 with roots -7, 0, 2
        let x = Polynomial::new([1., 5., -14., 0.]);
        let r = x.roots(tol);
        assert_eq!(r[0], -7.0); // check first root
        assert_eq!(r[1], 0.); // check second root
        assert_eq!(r[2], 2.0); // check third root
        assert_eq!(r.len(), 3); // check array length
    }

    #[test]
    fn roots_3_blinn() {
        let tol = f64::EPSILON;

        // Cubic p(x) = 1x^3 + 5x^2 + -14x + 0 with roots -7, 0, 2
        let x = Polynomial::new([1., 5., -14., 0.]);
        let r = solvers::blinn::Blinn::roots_cubic(&x);
        assert_eq!((r[0] - 2.0).abs() < 5.0 * tol, true); // check first root
        assert_eq!((r[1] - 0.0).abs() < 5.0 * tol, true); // check second root
        assert_eq!((r[2] + 7.0).abs() < 5.0 * tol, true); // check third root
        assert_eq!(r.len(), 3); // check array length
    }

    #[test]
    fn roots_3_yuksel() {
        let tol = f64::EPSILON;

        // Cubic p(x) = 1x^3 + 5x^2 + -14x + 0 with roots -7, 0, 2
        let x = Polynomial::new([1., 5., -14., 0.]);
        let r = solvers::yuksel::roots_cubic(&x, tol);
        assert_eq!(r[0], -7.0); // check first root
        assert_eq!(r[1], 0.); // check second root
        assert_eq!(r[2], 2.0); // check third root

        // assert_eq!(r.len(), 3); // check array length // Clean up solvers with const generics...
    }
}
