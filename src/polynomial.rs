// mod solvers;
// use solvers::*;

use crate::float::Float;

#[derive(Copy, Clone, Debug)]
pub struct Polynomial<T, const N: usize>
where
    T: Float,
    [(); N + 1]:,
{
    pub c: [T; N + 1],
}

impl<T, const N: usize> Polynomial<T, N>
where
    T: Float,
    [(); N + 1]:,
{
    pub const fn from(c: [T; N + 1]) -> Self {
        Self { c }
    }

    pub const fn new(c: [T; N + 1]) -> Self {
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
            0 => self.c[0],
            1 => self.c[0] * x + self.c[1],
            2 => self.eval_quadratic(x),
            3 => self.eval_cubic(x),
            4 => self.eval_quartic(x),
            // _ => self
            //     .c
            //     .iter()
            //     .enumerate()
            //     .map(|(n, k)| k * x.powi((N - n) as i32))
            //     .sum::<f64>(),
            _ => todo!(),
        }
    }

    pub fn roots(&self, tol: T) -> [T; N + 0_usize.pow(N as u32)] {
        match N {
            0 => self.root_constant(tol),
            1 => self.root_linear(tol),
            // 2 => solvers::blinn::roots_quadratic(self),
            _ => todo!(),
        }
    }

    #[inline]
    fn root_constant(&self, tol: T) -> [T; N + 0_usize.pow(N as u32)] {
        // Constant polynomial, has a root at x=0 IFF p(x) = 0
        let mut output = [T::NAN; N + 0_usize.pow(N as u32)];
        if self.c[0].abs() <= tol.into() {
            output[0] = T::ZERO;
        }
        output
    }

    #[inline]
    fn root_linear(&self, _tol: T) -> [T; N + 0_usize.pow(N as u32)] {
        // Linear polynomial, has exactly one root at x = -b/a
        [-self.c[1] / self.c[0]; N + 0_usize.pow(N as u32)]
    }
}

impl<T, const N: usize> core::fmt::Display for Polynomial<T, N>
where
    T: Float,
    [(); N + 1]:,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut output = String::new();
        for i in 0..N + 1 {
            if i != N {
                output.push_str(&format!("{}×x^{} + ", self.c[i], N - i));
            } else {
                output.push_str(&format!("{}", self.c[i]));
            }
        }
        f.write_str(&output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_f64_polynomials() {
        let p_0 = Polynomial::<f64, 0>::from([1.0]);
        let p_1 = Polynomial::<f64, 1>::from([1.0, 2.0]);
        let p_2 = Polynomial::<f64, 2>::from([1.0, 2.0, 3.0]);
        let p_3 = Polynomial::<f64, 3>::from([1.0, 2.0, 3.0, 4.0]);

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
        let x = Polynomial::<f64, 0>::from([1.]);
        let r = x.root_constant(tol);
        assert_eq!(r[0].is_nan(), true);
        assert_eq!(r.len(), 1);

        // Additive identity cast to a polynomial
        let y = Polynomial::<f64, 0>::from([0.]);
        let s = y.root_constant(tol);
        assert_eq!(s[0], 0.);
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn roots_1() {
        let tol = 1e-7;

        // Liner polynomial p(x) = x + 1
        let x = Polynomial::<f64, 1>::from([1., 1.]);
        let r = x.root_linear(tol);
        assert_eq!(r[0], -1.); // check root
        assert_eq!(r.len(), 1); // check length of returned array
    }

    #[test]
    fn roots_2() {
        let tol = 1e-7;

        // p(x) = 0x^2 + 1x + 1 with root -1
        let x = Polynomial::<f64, 2>::new([0., 1., 1.]);
        let r = x.roots(tol);
        assert_eq!(r[0], -1.); // check first root
        assert_eq!(r[1].is_finite(), false); // check second root
        assert_eq!(r.len(), 2); // check array length

        // Quadratic p(x) = x^2 - x - 12 with roots 4,-3
        let x = Polynomial::<f64, 2>::new([1., -1., -12.]);
        let r = x.roots(tol);
        assert_eq!(r[0], 4.);
        assert_eq!(r[1], -3.);

        // Quadratic p(x) = x^2 - 6x + 9 with root x = 3 with multiplicity 2
        let x = Polynomial::<f64, 2>::new([1., -6., 9.]);
        let r = x.roots(tol);
        assert_eq!(r[0], 3.);
        assert_eq!(r[1], 3.);

        // Quadratic p(x) = x^2 - 3x + 5 with complex roots
        let x = Polynomial::<f64, 2>::new([1., -3., 5.]);
        let r = x.roots(tol);
        assert_eq!(r[0].is_nan(), true);
        assert_eq!(r[1].is_nan(), true);
    }

    #[test]
    fn roots_3() {
        let tol = f64::EPSILON;

        // Cubic p(x) = 1x^3 + 5x^2 + -14x + 0 with roots -7, 0, 2
        let x = Polynomial::<f64, 3>::new([1., 5., -14., 0.]);
        let r = x.roots(tol);
        assert_eq!(r[0], 2.); // check first root
        assert_eq!(r[1], 0.); // check second root
        assert_eq!(r[2], -7.); // check third root
        assert_eq!(r.len(), 3); // check array length
    }
}
