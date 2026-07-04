//! Fixed-length column vectors over the algebraic tower.
//!
//! [`Vector<T, N>`] is a stack-only `[T; N]` with vector-space operations.
//! Structural operations (add/sub/neg, scalar multiplication, cross
//! products) need only `T: Ring`, so integer vectors are first-class; the
//! norm-dependent operations (`magnitude`, `normalize`, `dot`) require
//! `T: Scalar`.

use crate::algebra::{Field, Monoid, Ring};
use crate::real::Real;
use crate::scalar::Scalar;

use super::matrix::Matrix;
use core::ops::{Add, Div, Index, IndexMut, Mul, Neg, Sub};

/// A fixed-length vector of `N` components of type `T`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Vector<T, const N: usize> {
    /// The components, in order.
    pub b: [T; N],
}

impl<T, const N: usize> Vector<T, N> {
    /// Builds a vector from its component array.
    pub const fn new(b: [T; N]) -> Self {
        Self { b }
    }
}

impl<T, const N: usize> From<[T; N]> for Vector<T, N> {
    fn from(b: [T; N]) -> Self {
        Self { b }
    }
}

/// Component access by position. Panics on out-of-range indices, like a
/// slice.
impl<T, const N: usize> Index<usize> for Vector<T, N> {
    type Output = T;

    fn index(&self, i: usize) -> &T {
        &self.b[i]
    }
}

impl<T, const N: usize> IndexMut<usize> for Vector<T, N> {
    fn index_mut(&mut self, i: usize) -> &mut T {
        &mut self.b[i]
    }
}

/// The default vector is [`Vector::ZERO`].
impl<T: Ring, const N: usize> Default for Vector<T, N> {
    fn default() -> Self {
        Self::ZERO
    }
}

// Structural operations need only a `Ring` (integer vectors are first-class,
// matching the `Matrix` relaxation in §5.6); the norm-based operations below
// stay `Scalar`-bound.
impl<T: Ring, const N: usize> Vector<T, N> {
    /// The zero vector (the additive identity).
    pub const ZERO: Vector<T, N> = Self { b: [T::ZERO; N] };

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
}

impl<T: Scalar, const N: usize> Vector<T, N> {
    /// The **hermitian** dot product `Σ aᵢ·conj(bᵢ)`: the second operand is
    /// conjugated, so `v.dot(&v)` equals `|v|²` with zero imaginary part
    /// even for complex component types. For real scalars `conj` is the
    /// identity and this is the ordinary Euclidean dot product.
    pub fn dot(&self, rhs: &Self) -> T {
        self.b
            .iter()
            .zip(rhs.b.iter())
            .fold(T::ZERO, |acc, (&a, &b)| acc + a * b.conj())
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

impl<T: Ring> Vector<T, 2> {
    /// The 2-D (scalar) cross product `a₀b₁ − a₁b₀`: the signed area of the
    /// parallelogram the two vectors span (the z-component of the 3-D cross
    /// product with zero-extended inputs).
    pub fn cross(&self, rhs: &Vector<T, 2>) -> T {
        self.b[0] * rhs.b[1] - self.b[1] * rhs.b[0]
    }
}

impl<T: Ring> Vector<T, 3> {
    /// The 3-D cross product `a × b`: the vector orthogonal to both inputs
    /// with `|a × b|` the parallelogram area, right-hand-rule oriented.
    pub fn cross(&self, rhs: &Vector<T, 3>) -> Vector<T, 3> {
        Vector::new([
            self.b[1] * rhs.b[2] - self.b[2] * rhs.b[1],
            self.b[2] * rhs.b[0] - self.b[0] * rhs.b[2],
            self.b[0] * rhs.b[1] - self.b[1] * rhs.b[0],
        ])
    }
}

// Writes straight to the `Formatter` (no allocation) so it works in `no_std`.
impl<T: core::fmt::Display, const N: usize> core::fmt::Display for Vector<T, N> {
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

impl<T: Ring, const N: usize> Add<Vector<T, N>> for Vector<T, N> {
    type Output = Vector<T, N>;

    fn add(self, rhs: Vector<T, N>) -> Self::Output {
        let mut b = self.b;
        for (i, e) in b.iter_mut().enumerate().take(N) {
            *e += rhs.b[i];
        }

        Self::Output { b }
    }
}

impl<T: Ring, const N: usize> Sub<Vector<T, N>> for Vector<T, N> {
    type Output = Vector<T, N>;

    fn sub(self, rhs: Vector<T, N>) -> Self::Output {
        let mut b = self.b;
        for (i, e) in b.iter_mut().enumerate().take(N) {
            *e -= rhs.b[i];
        }

        Self::Output { b }
    }
}

impl<T: Ring, const N: usize> Neg for Vector<T, N> {
    type Output = Vector<T, N>;

    fn neg(mut self) -> Self::Output {
        for e in self.b.iter_mut() {
            *e = -*e;
        }
        self
    }
}

impl<T: Ring, const N: usize> Mul<T> for Vector<T, N> {
    type Output = Vector<T, N>;

    fn mul(self, rhs: T) -> Self::Output {
        let mut b = self.b;
        for e in b.iter_mut().take(N) {
            *e *= rhs;
        }

        Self::Output { b }
    }
}

// Componentwise division by a scalar. `Field`, not `Ring`: division is the
// one structural op a plain ring cannot supply.
impl<T: Field, const N: usize> Div<T> for Vector<T, N> {
    type Output = Vector<T, N>;

    fn div(self, rhs: T) -> Self::Output {
        let mut b = self.b;
        for e in b.iter_mut().take(N) {
            *e /= rhs;
        }

        Self::Output { b }
    }
}

/// Scalar-on-the-left multiplication, stamped per concrete scalar type.
/// Coherence forbids the blanket `impl<T: Ring> Mul<Vector<T, N>> for T`
/// (`T` is a bare type parameter in the `impl` head), so each supported
/// scalar gets a concrete impl.
macro_rules! impl_scalar_vector_mul {
    ($($s:ty),+ $(,)?) => {
        $(
            impl<const N: usize> Mul<Vector<$s, N>> for $s {
                type Output = Vector<$s, N>;

                fn mul(self, rhs: Vector<$s, N>) -> Self::Output {
                    let mut b = rhs.b;
                    for e in b.iter_mut().take(N) {
                        *e *= self;
                    }

                    Self::Output { b }
                }
            }
        )+
    };
}

impl_scalar_vector_mul!(f32, f64, crate::complex::c32, crate::complex::c64);

// Some simple tests
#[cfg(test)]
mod tests {
    use crate::complex::{c32, c64};

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
    fn test_cross_3d() {
        // The canonical right-handed frame: x × y = z, y × z = x, z × x = y.
        let x = Vector::new([1., 0., 0.]);
        let y = Vector::new([0., 1., 0.]);
        let z = Vector::new([0., 0., 1.]);
        assert_eq!(x.cross(&y), z);
        assert_eq!(y.cross(&z), x);
        assert_eq!(z.cross(&x), y);

        // Anticommutativity and self-annihilation.
        let a = Vector::new([1., 2., 3.]);
        let b = Vector::new([4., 5., 6.]);
        assert_eq!(a.cross(&b), Vector::new([-3., 6., -3.]));
        assert_eq!(a.cross(&b), -(b.cross(&a)));
        assert_eq!(a.cross(&a), Vector::ZERO);

        // Orthogonality to both inputs.
        assert_eq!(a.cross(&b).dot(&a), 0.0);
        assert_eq!(a.cross(&b).dot(&b), 0.0);

        // Integer vectors work too (Ring bound, not Scalar).
        let ai = Vector::new([1i64, 2, 3]);
        let bi = Vector::new([4i64, 5, 6]);
        assert_eq!(ai.cross(&bi), Vector::new([-3, 6, -3]));
    }

    #[test]
    fn test_dot_real_and_hermitian() {
        let a = Vector::new([1., 2., 3.]);
        let b = Vector::new([4., -5., 6.]);
        assert_eq!(a.dot(&b), 4. - 10. + 18.);

        // dot(v, v) == |v|² for real vectors...
        assert_eq!(a.dot(&a), a.magnitude() * a.magnitude());

        // ...and for complex vectors, thanks to the conjugation on the
        // second operand: the imaginary part is exactly zero.
        let v = Vector::new([c64::new(1.0, 2.0), c64::new(-3.0, 0.5)]);
        let d = v.dot(&v);
        assert_eq!(d, c64::new(1.0 + 4.0 + 9.0 + 0.25, 0.0));

        // The hermitian form is conjugate-symmetric, not symmetric:
        // dot(a, b) == conj(dot(b, a)).
        let w = Vector::new([c64::new(0.0, 1.0), c64::new(2.0, -1.0)]);
        assert_eq!(v.dot(&w), w.dot(&v).conj());
    }

    #[test]
    fn test_index_default_from() {
        let mut v: Vector<f64, 3> = [1., 2., 3.].into();
        assert_eq!(v[0], 1.);
        assert_eq!(v[2], 3.);
        v[1] = 7.;
        assert_eq!(v, Vector::new([1., 7., 3.]));

        assert_eq!(Vector::<f64, 4>::default(), Vector::ZERO);
        assert_eq!(Vector::<i32, 2>::default(), Vector::new([0, 0]));
    }

    #[test]
    fn test_scalar_div_and_left_mul() {
        let v = Vector::new([2., 4., 6.]);
        assert_eq!(v / 2., Vector::new([1., 2., 3.]));

        // Left multiplication for every stamped scalar type.
        assert_eq!(2.0_f64 * Vector::new([1., 2.]), Vector::new([2., 4.]));
        assert_eq!(2.0_f32 * Vector::new([1_f32, 2.]), Vector::new([2., 4.]));
        let i = c64::new(0.0, 1.0);
        assert_eq!(
            i * Vector::new([c64::new(1.0, 0.0), c64::new(0.0, 1.0)]),
            Vector::new([c64::new(0.0, 1.0), c64::new(-1.0, 0.0)])
        );
        let j = c32::new(0.0, 1.0);
        assert_eq!(
            j * Vector::new([c32::new(2.0, 0.0)]),
            Vector::new([c32::new(0.0, 2.0)])
        );

        // Complex vector divided by a complex scalar (Smith division inside).
        let v = Vector::new([c64::new(2.0, 2.0)]);
        assert_eq!(v / c64::new(1.0, 1.0), Vector::new([c64::new(2.0, 0.0)]));
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
