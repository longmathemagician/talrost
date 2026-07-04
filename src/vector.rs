use crate::algebra::Monoid;
use crate::real::Real;
use crate::scalar::Scalar;

use super::matrix::Matrix;
use core::ops::{Add, Mul, Sub};

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Vector<T, const N: usize> {
    pub b: [T; N],
}

impl<T: Scalar, const N: usize> Vector<T, N> {
    pub const ZERO: Vector<T, N> = Self { b: [T::ZERO; N] };

    pub fn new(b: [T; N]) -> Self {
        Self { b }
    }

    /// This vector as a 1×N row matrix.
    pub fn row(&self) -> Matrix<T, 1, N> {
        Matrix { e: [self.b] }
    }

    /// This vector as an N×1 column matrix.
    pub fn column(&self) -> Matrix<T, N, 1> {
        let mut e = [[T::ZERO; 1]; N];

        for (i, e) in e.iter_mut().enumerate().take(N) {
            e[0] = self.b[i];
        }
        Matrix { e }
    }

    /// Returns the Euclidean norm of the vector: a *real* number, even for
    /// complex component types (`Vector<c64, N>::magnitude() -> f64`).
    pub fn magnitude(&self) -> T::Real {
        self.b
            .iter()
            .fold(T::Real::ZERO, |acc, &x| acc + x.norm_sqr())
            .sqrt()
    }

    /// Returns a normalized copy of the vector; each component is divided by
    /// the (real) magnitude.
    pub fn normalize(&self) -> Self {
        let mag = self.magnitude();
        let mut b = self.b;

        for e in b.iter_mut().take(N) {
            *e = *e / mag;
        }

        Self { b }
    }
}

impl<T: Scalar> Vector<T, 2> {
    /// Returns the cross product of the two vectors
    pub fn cross(&self, rhs: &Vector<T, 2>) -> T {
        self.b[0] * rhs.b[1] - self.b[1] * rhs.b[0]
    }
}

// Writes straight to the `Formatter` (no allocation) so it works in `no_std`.
impl<T: Scalar + core::fmt::Display, const N: usize> core::fmt::Display for Vector<T, N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("(")?;
        for (i, e) in self.b.iter().enumerate() {
            if i != 0 {
                f.write_str(", ")?;
            }
            write!(f, "{}", e)?;
        }
        f.write_str(")")
    }
}

impl<T: Scalar, const N: usize> Add<Vector<T, N>> for Vector<T, N> {
    type Output = Vector<T, N>;

    fn add(self, rhs: Vector<T, N>) -> Self::Output {
        let mut b = self.b;
        for (i, e) in b.iter_mut().enumerate().take(N) {
            *e += rhs.b[i];
        }

        Self::Output { b }
    }
}

impl<T: Scalar, const N: usize> Sub<Vector<T, N>> for Vector<T, N> {
    type Output = Vector<T, N>;

    fn sub(self, rhs: Vector<T, N>) -> Self::Output {
        let mut b = self.b;
        for (i, e) in b.iter_mut().enumerate().take(N) {
            *e -= rhs.b[i];
        }

        Self::Output { b }
    }
}

impl<T: Scalar, const N: usize> Mul<T> for Vector<T, N> {
    type Output = Vector<T, N>;

    fn mul(self, rhs: T) -> Self::Output {
        let mut b = self.b;
        for e in b.iter_mut().take(N) {
            *e *= rhs;
        }

        Self::Output { b }
    }
}

// Scalar-on-the-left multiplication. Coherence forbids the blanket
// `impl<T: Scalar> Mul<Vector<T, N>> for T`; Phase 5 generalizes this with a
// per-type macro.
impl<const N: usize> Mul<Vector<f64, N>> for f64 {
    type Output = Vector<f64, N>;

    fn mul(self, rhs: Vector<f64, N>) -> Self::Output {
        let mut b = rhs.b;
        for e in b.iter_mut().take(N) {
            *e *= self;
        }

        Self::Output { b }
    }
}

// Some simple tests
#[cfg(test)]
mod tests {
    use crate::complex::c64;

    use super::*;

    #[test]
    fn test_vector() {
        let v1 = Vector::new([1., 2., 3.]);
        let v2 = Vector::new([4., 5., 6.]);

        assert_eq!(v1.magnitude(), 14_f64.sqrt());
        assert_eq!(v1.normalize().magnitude(), 1.);
        assert_eq!(v1.row(), Matrix::new([[1., 2., 3.]]));
        assert_eq!(v1.column(), Matrix::new([[1.], [2.], [3.]]),);

        assert_eq!(v1 + v2, Vector::new([5., 7., 9.]));
        assert_eq!(v1 - v2, Vector::new([-3., -3., -3.]));
        assert_eq!(v1 * 2., Vector::new([2., 4., 6.]));
        assert_eq!(2. * v2, Vector::new([8., 10., 12.]));
    }

    #[test]
    fn test_cross() {
        let v1 = Vector::new([1., 2.]);
        let v2 = Vector::new([3., 4.]);
        assert_eq!(v1.cross(&v2), -2.);

        let v1 = Vector::new([3., 4.]);
        let v2 = Vector::new([1., 2.]);
        assert_eq!(v1.cross(&v2), 2.);
    }

    #[test]
    fn test_complex_vectors() {
        let vec_real = Vector::new([1.0, 2.0]);
        let vec_complex = Vector::new([c64::new(1.0, 0.0), c64::new(2.0, 0.0)]);

        assert_eq!(vec_real.magnitude(), 5_f64.sqrt());
        // The whole point of the Scalar::Real design: a complex vector's
        // magnitude is an f64, not a Complex with zero imaginary part.
        assert_eq!(vec_complex.magnitude(), 5_f64.sqrt());

        // And normalization works through Div<T::Real>.
        let n = Vector::new([c64::new(3.0, 0.0), c64::new(0.0, 4.0)]).normalize();
        assert!((n.magnitude() - 1.0).abs() < 1e-12);
    }
}
