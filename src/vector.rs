use crate::{float::Float, number::Number};

use super::matrix::Matrix;
use std::ops::{Add, Mul, Sub};

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Vector<T: Number<Type = T>, const N: usize>
where
    T: Float,
{
    pub b: [T; N],
}

impl<T: Number<Type = T> + std::iter::Sum, const N: usize> Vector<T, N>
where
    T: Float,
{
    pub const ZERO: Vector<T, N> = Self { b: [T::ZERO; N] };

    pub fn new(b: [T; N]) -> Self {
        Self { b }
    }

    pub fn row(&self) -> Matrix<T, N, 1> {
        Matrix { e: [self.b] }
    }

    pub fn column(&self) -> Matrix<T, 1, N> {
        let mut e = [[T::ZERO; 1]; N];

        for (i, e) in e.iter_mut().enumerate().take(N) {
            e[0] = self.b[i];
        }
        Matrix { e }
    }

    /// Returns the Euclidean norm of the vector
    pub fn magnitude(&self) -> T {
        self.b.iter().map(|x| x.powi(2)).sum::<T>().sqrt()
    }

    /// Returns a normalized copy of the vector
    pub fn normalize(&self) -> Self {
        let mag = self.magnitude();
        let mut b = self.b;

        for e in b.iter_mut().take(N) {
            // b[i] *= mag.recip();
            *e /= mag;
        }

        Self { b }
    }
}

impl<T: Number<Type = T>> Vector<T, 2>
where
    T: Float,
{
    /// Returns the cross product of the two vectors
    pub fn cross(&self, rhs: &Vector<T, 2>) -> T {
        self.b[0] * rhs.b[1] - self.b[1] * rhs.b[0]
    }
}

impl<T: Number<Type = T>, const N: usize> core::fmt::Display for Vector<T, N>
where
    T: Float,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        assert_ne!(N, 0);
        let mut output = String::from("(");
        for e in self.b {
            // output.push_str(&format!("{}, ", format_f64(e, 7)));
            output.push_str(&format!("{}, ", e));
        }
        output.pop();
        output.pop();
        output.push(')');
        f.write_str(&output)
    }
}

impl<T: Number<Type = T>, const N: usize> Add<Vector<T, N>> for Vector<T, N>
where
    T: Float,
{
    type Output = Vector<T, N>;

    fn add(self, rhs: Vector<T, N>) -> Self::Output {
        let mut b = self.b;
        for (i, e) in b.iter_mut().enumerate().take(N) {
            *e += rhs.b[i];
        }

        Self::Output { b }
    }
}

impl<T: Number<Type = T>, const N: usize> Sub<Vector<T, N>> for Vector<T, N>
where
    T: Float,
{
    type Output = Vector<T, N>;

    fn sub(self, rhs: Vector<T, N>) -> Self::Output {
        let mut b = self.b;
        for (i, e) in b.iter_mut().enumerate().take(N) {
            *e -= rhs.b[i];
        }

        Self::Output { b }
    }
}

impl<T: Number<Type = T>, const N: usize> Mul<T> for Vector<T, N>
where
    T: Float,
{
    type Output = Vector<T, N>;

    fn mul(self, rhs: T) -> Self::Output {
        let mut b = self.b;
        for e in b.iter_mut().take(N) {
            *e *= rhs;
        }

        Self::Output { b }
    }
}

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
        assert_eq!(vec_complex.magnitude(), 5_f64.sqrt().into());
    }
}
