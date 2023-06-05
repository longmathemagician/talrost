use core::fmt::Debug;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::float::Float;

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Complex<F>
where
    F: Float,
{
    re: F,
    im: F,
}

impl<F> Complex<F>
where
    F: Float,
{
    const NAN: Self = Self {
        re: F::NAN,
        im: F::NAN,
    };

    const ZERO: Self = Self {
        re: F::ZERO,
        im: F::ZERO,
    };

    const ONE: Self = Self {
        re: F::ONE,
        im: F::ZERO,
    };

    pub fn new(re: F, im: F) -> Self {
        Self { re, im }
    }

    pub fn magnitude(&self) -> F {
        (self.re.powi(2) + self.im.powi(2)).powi(2)
    }

    pub fn sqrt(&self) -> Self {
        if self.re == F::ZERO && self.im == F::ZERO {
            Self::ZERO
        } else {
            let mdl = (self.re * self.re + self.im * self.im).sqrt();
            let arg = self.im.atan2(self.re);
            let sq_mdl = mdl.sqrt();
            let harg = arg / (F::ONE + F::ONE);
            let re = sq_mdl * harg.cos();
            let im = sq_mdl * harg.sin();
            Self { re, im }
        }
    }

    pub fn normalize(&self) -> Self {
        let mag = self.magnitude();
        Self {
            re: self.re / mag,
            im: self.im / mag,
        }
    }

    pub fn powi(&self, power: i32) -> Self {
        let mut result = Self::ONE;
        for _ in 0..power {
            result *= *self;
        }
        result
    }
}

// Implement core::fmt::Display for Complex<F>
impl<F> core::fmt::Display for Complex<F>
where
    F: Float,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.im < F::ZERO {
            write!(f, "{} - {}i", self.re, -self.im)
        } else {
            write!(f, "{} + {}i", self.re, self.im)
        }
    }
}

// Implement std::iter::Sum for Complex<F>
impl<F> std::iter::Sum for Complex<F>
where
    F: Float,
{
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |a, b| a + b)
    }
}

// Implement From for &str to Complex<F>
impl<F> From<&str> for Complex<F>
where
    F: Float + std::str::FromStr,
{
    fn from(s: &str) -> Self {
        let mut re = String::new();
        let mut im = String::new();
        let mut is_re = true;
        for c in s.chars() {
            match c {
                ' ' => continue,
                '+' => {
                    is_re = false;
                    continue;
                }
                'i' => break,
                _ => {
                    if is_re {
                        re.push(c);
                    } else {
                        im.push(c);
                    }
                }
            }
        }
        Self::new(
            re.parse::<F>().unwrap_or(F::ZERO),
            im.parse::<F>().unwrap_or(F::ZERO),
        )
    }
}

// Implement From for (F, F) to Complex<F>
impl<F> From<(F, F)> for Complex<F>
where
    F: Float,
{
    fn from(value: (F, F)) -> Self {
        Self::new(value.0, value.1)
    }
}

// Implement From for [F, F] to Complex<F>
impl<F> From<[F; 2]> for Complex<F>
where
    F: Float,
{
    fn from(value: [F; 2]) -> Self {
        Self::new(value[0], value[1])
    }
}

// // Implement Into for f32 to Complex<f32>
// impl Into<Complex<f32>> for f32 {
//     fn into(self) -> Complex<f32> {
//         Complex::new((self, 0.0))
//     }
// }

// // Implement Into for f64 to Complex<f64>
// impl Into<Complex<f64>> for f64 {
//     fn into(self) -> Complex<f64> {
//         Complex::new((self, 0.0))
//     }
// }

// Implement core::ops::Add for Complex<F>
impl<F> Add for Complex<F>
where
    F: Float,
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

// Implement core::ops::Add for Complex<F> where RHS is F
impl<F> Add<F> for Complex<F>
where
    F: Float,
{
    type Output = Self;
    fn add(self, rhs: F) -> Self::Output {
        Self {
            re: self.re + rhs,
            im: self.im,
        }
    }
}

// Implement core::ops::Sub for Complex<F>
impl<F> Sub for Complex<F>
where
    F: Float,
{
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

// Implement core::ops::Sub for Complex<F> where RHS is F
impl<F> Sub<F> for Complex<F>
where
    F: Float,
{
    type Output = Self;
    fn sub(self, rhs: F) -> Self::Output {
        Self {
            re: self.re - rhs,
            im: self.im,
        }
    }
}

// Implement core::ops::Mul for Complex<F>
impl<F> Mul for Complex<F>
where
    F: Float,
{
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

// Implement core::ops::Mul for Complex<F> where RHS is F
impl<F> Mul<F> for Complex<F>
where
    F: Float,
{
    type Output = Self;
    fn mul(self, rhs: F) -> Self::Output {
        Self {
            re: self.re * rhs,
            im: self.im * rhs,
        }
    }
}

// Implement core::ops::Div for Complex<F>
impl<F> Div for Complex<F>
where
    F: Float,
{
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        let denom = rhs.re.powi(2) + rhs.im.powi(2);
        Self {
            re: (self.re * rhs.re + self.im * rhs.im) / denom,
            im: (self.im * rhs.re - self.re * rhs.im) / denom,
        }
    }
}

// Implement core::ops::Div for Complex<F> where RHS is F
impl<F> Div<F> for Complex<F>
where
    F: Float,
{
    type Output = Self;
    fn div(self, rhs: F) -> Self::Output {
        Self {
            re: self.re / rhs,
            im: self.im / rhs,
        }
    }
}

// Implement core::ops::AddAssign for Complex<F>
impl<F> AddAssign for Complex<F>
where
    F: Float,
{
    fn add_assign(&mut self, rhs: Self) {
        self.re += rhs.re;
        self.im += rhs.im;
    }
}

// Implement core::ops::AddAssign for Complex<F> where RHS is F
impl<F> AddAssign<F> for Complex<F>
where
    F: Float,
{
    fn add_assign(&mut self, rhs: F) {
        self.re += rhs;
    }
}

// Implement core::ops::SubAssign for Complex<F>
impl<F> SubAssign for Complex<F>
where
    F: Float,
{
    fn sub_assign(&mut self, rhs: Self) {
        self.re -= rhs.re;
        self.im -= rhs.im;
    }
}

// Implement core::ops::SubAssign for Complex<F> where RHS is F
impl<F> SubAssign<F> for Complex<F>
where
    F: Float,
{
    fn sub_assign(&mut self, rhs: F) {
        self.re -= rhs;
    }
}

// Implement core::ops::MulAssign for Complex<F>
impl<F> MulAssign for Complex<F>
where
    F: Float,
{
    fn mul_assign(&mut self, rhs: Self) {
        let re = self.re * rhs.re - self.im * rhs.im;
        let im = self.re * rhs.im + self.im * rhs.re;
        self.re = re;
        self.im = im;
    }
}

// Implement core::ops::MulAssign for Complex<F> where RHS is F
impl<F> MulAssign<F> for Complex<F>
where
    F: Float,
{
    fn mul_assign(&mut self, rhs: F) {
        self.re *= rhs;
        self.im *= rhs;
    }
}

// Implement core::ops::DivAssign for Complex<F>
impl<F> DivAssign for Complex<F>
where
    F: Float,
{
    fn div_assign(&mut self, rhs: Self) {
        let denom = rhs.re * rhs.re + rhs.im * rhs.im;
        let re = (self.re * rhs.re + self.im * rhs.im) / denom;
        let im = (self.im * rhs.re - self.re * rhs.im) / denom;
        self.re = re;
        self.im = im;
    }
}

// Implement core::ops::DivAssign for Complex<F> where RHS is F
impl<F> DivAssign<F> for Complex<F>
where
    F: Float,
{
    fn div_assign(&mut self, rhs: F) {
        self.re /= rhs;
        self.im /= rhs;
    }
}

// Implement core::ops::Neg for Complex<F>
impl<F> Neg for Complex<F>
where
    F: Float,
{
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self {
            re: -self.re,
            im: -self.im,
        }
    }
}

impl<F> From<F> for Complex<F>
where
    F: Float,
{
    fn from(value: F) -> Self {
        Self {
            re: value,
            im: F::ZERO,
        }
    }
}

pub trait NumberTrait
where
    Self: Sized
        + Copy
        + Add<Output = Self>
        + Sub<Output = Self>
        + Mul<Output = Self>
        + Div<Output = Self>
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + Neg<Output = Self>
        + PartialOrd
        + PartialEq
        + core::fmt::Display
        + Debug,
{
    type NumberType: Sized
        + Copy
        + Add<Output = Self>
        + Sub<Output = Self>
        + Mul<Output = Self>
        + Div<Output = Self>
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + Neg<Output = Self>
        + PartialOrd
        + PartialEq
        + core::fmt::Display
        + Debug;
    const NAN: Self::NumberType;
    const ZERO: Self::NumberType;
    const ONE: Self::NumberType;
    fn new(value: Self::NumberType) -> Self::NumberType;
    fn abs(&self) -> Self::NumberType;
    fn powi(&self, power: i32) -> Self::NumberType;
    fn sqrt(&self) -> Self::NumberType;
}

impl<F> NumberTrait for F
where
    F: Float,
{
    type NumberType = F;
    const NAN: Self::NumberType = F::NAN;
    const ZERO: Self::NumberType = F::ZERO;
    const ONE: Self::NumberType = F::ONE;
    fn new(value: Self::NumberType) -> Self::NumberType {
        value
    }
    fn abs(&self) -> Self::NumberType {
        F::abs(self)
    }
    fn powi(&self, power: i32) -> Self::NumberType {
        F::powi(self, power)
    }
    fn sqrt(&self) -> Self::NumberType {
        F::sqrt(self)
    }
}

impl<F> NumberTrait for Complex<F>
where
    F: Float,
{
    type NumberType = Complex<F>;
    const NAN: Self::NumberType = Complex::<F>::NAN;
    const ZERO: Self::NumberType = Complex::<F>::ZERO;
    const ONE: Self::NumberType = Complex::<F>::ONE;
    fn new(value: Self::NumberType) -> Self::NumberType {
        value
    }
    fn abs(&self) -> Self::NumberType {
        Complex::<F>::new(self.re.abs(), self.im.abs())
    }
    fn powi(&self, power: i32) -> Self::NumberType {
        Complex::<F>::powi(self, power)
    }
    fn sqrt(&self) -> Self::NumberType {
        self.sqrt()
    }
}

// Implement core::fmt::Display for N: Number
impl<N> core::fmt::Display for Number<N>
where
    N: NumberTrait,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#?}", self.0)
    }
}

// // Implement core::ops::Add for Number<N> where Self and RHS are both N: Number
// impl<N> Add for Number<N>
// where
//     N: Number,
// {
//     type Output = Self::NumberType<N>;
//     fn add(self, rhs: Self) -> Self::Output {
//         Number::<N>(N::new(N::NumberType::add(self.0, rhs.0)))
//     }
// }

// Implement core::ops::Add for Number<N> where Self and RHS are both F
impl<F> Add for Number<F>
where
    F: Float,
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Number::<F>(F::add(self.0, rhs.0))
    }
}

// Implement core::ops::Add for Number<N> where Self and RHS are both Complex<F>
impl<F> Add for Number<Complex<F>>
where
    F: Float,
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Number::<Complex<F>>(Complex::add(self.0, rhs.0))
    }
}

// Implement core::ops::Add for Number<N> where Self is F and RHS is Complex<F>
impl<F> Add<Number<Complex<F>>> for Number<F>
where
    F: Float,
{
    type Output = Number<Complex<F>>;
    fn add(self, rhs: Number<Complex<F>>) -> Self::Output {
        Number::<Complex<F>>(Complex::<F>::add(Complex::<F>::from(self.0), rhs.0))
    }
}

// Implement core::ops::Add for Number<N> where Self is Complex<F> and RHS is F
impl<F> Add<Number<F>> for Number<Complex<F>>
where
    F: Float,
{
    type Output = Number<Complex<F>>;
    fn add(self, rhs: Number<F>) -> Self::Output {
        Number::<Complex<F>>(Complex::add(self.0, rhs.0))
    }
}

// Implement core::ops::AddAssign for Number<N> where Self is N: Number and RHS is N: Number
impl<N> AddAssign for Number<N>
where
    N: NumberTrait<NumberType = N>,
{
    fn add_assign(&mut self, rhs: Self) {
        self.0 = N::new(N::NumberType::add(self.0, rhs.0));
    }
}

// Implement core::ops::AddAssign for Number<N> where Self is NumberTrait<NumberType = F> and RHS is NumberTrait<NumberType = Complex<F>>
impl<F> AddAssign<Number<Complex<F>>> for Number<F>
where
    F: Float,
{
    fn add_assign(&mut self, _rhs: Number<Complex<F>>) {
        todo!()
    }
}

// Implement core::ops::AddAssign for Number<N> where Self is NumberTrait<NumberType = Complex<F>> and RHS is NumberTrait<NumberType = F>
impl<F> AddAssign<Number<F>> for Number<Complex<F>>
where
    F: Float,
{
    fn add_assign(&mut self, rhs: Number<F>) {
        self.0 = Complex::add(self.0, rhs.0);
    }
}

// Ignore camel case
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Number<N: NumberTrait>(N::NumberType);

impl<N: NumberTrait> Number<N> {
    // Generic new function
    pub fn new(value: N::NumberType) -> Self {
        Self(N::new(value))
    }
}

// From F to Number<F>
impl<F> From<F> for Number<F>
where
    F: Float,
{
    fn from(value: F) -> Self {
        Self::new(value)
    }
}

// From Complex<F> to Number<Complex<F>>
impl<F> From<Complex<F>> for Number<Complex<F>>
where
    F: Float,
{
    fn from(value: Complex<F>) -> Self {
        Self::new(value)
    }
}

// From (F, F) to Number<c32>
impl<F> From<(F, F)> for Number<Complex<F>>
where
    F: Float,
{
    fn from(value: (F, F)) -> Self {
        Self::new(Complex::new(value.0, value.1))
    }
}

// From [F, F] to Number<c32>
impl<F> From<[F; 2]> for Number<Complex<F>>
where
    F: Float,
{
    fn from(value: [F; 2]) -> Self {
        Self::new(Complex::new(value[0], value[1]))
    }
}

#[allow(non_camel_case_types)]
pub type n64<N> = Number<N>;
#[allow(non_camel_case_types)]
pub type n32<N> = Number<N>;

#[allow(non_camel_case_types)]
pub type n32f = Number<f32>;

#[allow(non_camel_case_types)]
pub type n32c = Number<c32>;

#[allow(non_camel_case_types)]
pub type c64 = Complex<f64>;
#[allow(non_camel_case_types)]
pub type c32 = Complex<f32>;

// Implement core::ops::Add for Complex<f64> where Self is f64
impl Add<Complex<f64>> for f64 {
    type Output = Complex<f64>;
    fn add(self, rhs: Complex<f64>) -> Self::Output {
        Self::Output {
            re: self + rhs.re,
            im: rhs.im,
        }
    }
}

// Implement core::ops::Sub for Complex<f64> where Self is f64
impl Sub<Complex<f64>> for f64 {
    type Output = Complex<f64>;
    fn sub(self, rhs: Complex<f64>) -> Self::Output {
        Self::Output {
            re: self - rhs.re,
            im: -rhs.im,
        }
    }
}

// Implement core::ops::Mul for Complex<f64> where Self is f64
impl Mul<Complex<f64>> for f64 {
    type Output = Complex<f64>;
    fn mul(self, rhs: Complex<f64>) -> Self::Output {
        Self::Output {
            re: self * rhs.re,
            im: self * rhs.im,
        }
    }
}

// Implement core::ops::Div for Complex<f64> where Self is f64
impl Div<Complex<f64>> for f64 {
    type Output = Complex<f64>;
    fn div(self, rhs: Complex<f64>) -> Self::Output {
        let denom = rhs.re.powi(2) + rhs.im.powi(2);
        Self::Output {
            re: (self * rhs.re) / denom,
            im: -(self * rhs.im) / denom,
        }
    }
}

// Implement core::ops::Add for Complex<f32> where Self is f32
impl Add<Complex<f32>> for f32 {
    type Output = Complex<f32>;
    fn add(self, rhs: Complex<f32>) -> Self::Output {
        Self::Output {
            re: self + rhs.re,
            im: rhs.im,
        }
    }
}

// Implement core::ops::Sub for Complex<f32> where Self is f32
impl Sub<Complex<f32>> for f32 {
    type Output = Complex<f32>;
    fn sub(self, rhs: Complex<f32>) -> Self::Output {
        Self::Output {
            re: self - rhs.re,
            im: -rhs.im,
        }
    }
}

// Implement core::ops::Mul for Complex<f32> where Self is f32
impl Mul<Complex<f32>> for f32 {
    type Output = Complex<f32>;
    fn mul(self, rhs: Complex<f32>) -> Self::Output {
        Self::Output {
            re: self * rhs.re,
            im: self * rhs.im,
        }
    }
}

// Implement core::ops::Div for Complex<f32> where Self is f32
impl Div<Complex<f32>> for f32 {
    type Output = Complex<f32>;
    fn div(self, rhs: Complex<f32>) -> Self::Output {
        let denom = rhs.re.powi(2) + rhs.im.powi(2);
        Self::Output {
            re: (self * rhs.re) / denom,
            im: -(self * rhs.im) / denom,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_c32_addition() {
        let mut a: f32 = 0.25;
        let mut b: c32 = [0.75, 0.66].into();

        assert_eq!(a + a, 0.5);
        assert_eq!(a + b, [1.0, 0.66].into());
        assert_eq!(b + a, [1.0, 0.66].into());
        assert_eq!(b + b, [1.5, 1.32].into());

        a += a;
        assert_eq!(a, 0.5);

        b += a;
        assert_eq!(b, [1.25, 0.66].into());

        b += b;
        assert_eq!(b, [2.5, 1.32].into());
    }

    #[test]
    fn test_c64_addition() {
        let mut a: f64 = 0.25;
        let mut b: c64 = [0.75, 0.66].into();

        assert_eq!(a + a, 0.5);
        assert_eq!(a + b, [1.0, 0.66].into());
        assert_eq!(b + a, [1.0, 0.66].into());
        assert_eq!(b + b, [1.5, 1.32].into());

        a += a;
        assert_eq!(a, 0.5);

        b += a;
        assert_eq!(b, [1.25, 0.66].into());

        b += b;
        assert_eq!(b, [2.5, 1.32].into());
    }

    #[test]
    fn test_c32_subtraction() {
        let mut a: f32 = 0.25;
        let mut b: c32 = [0.75, 0.66].into();

        assert_eq!(a - a, 0.0);
        assert_eq!(a - b, [-0.5, -0.66].into());
        assert_eq!(b - a, [0.5, 0.66].into());
        assert_eq!(b - b, [0.0, 0.0].into());

        a -= 0.5 * a;
        assert_eq!(a, 0.125);

        b -= a;
        assert_eq!(b, [0.625, 0.66].into());

        b -= b;
        assert_eq!(b, [0.0, 0.0].into());
    }

    #[test]
    fn test_c64_subtraction() {
        let mut a: f64 = 0.25;
        let mut b: c64 = [0.75, 0.66].into();

        assert_eq!(a - a, 0.0);
        assert_eq!(a - b, [-0.5, -0.66].into());
        assert_eq!(b - a, [0.5, 0.66].into());
        assert_eq!(b - b, [0.0, 0.0].into());

        a -= 0.5 * a;
        assert_eq!(a, 0.125);

        b -= a;
        assert_eq!(b, [0.625, 0.66].into());

        b -= b;
        assert_eq!(b, [0.0, 0.0].into());
    }

    #[test]
    fn test_c32_multiplication() {
        let mut a: f32 = 3.0;
        let mut b: c32 = [7.0, 13.0].into();

        assert_eq!(a * a, 9.0);
        assert_eq!(a * b, [21.0, 39.0].into());
        assert_eq!(b * a, [21.0, 39.0].into());
        assert_eq!(b * b, [-120.0, 182.0].into());

        a *= a;
        assert_eq!(a, 9.0);

        b *= a;
        assert_eq!(b, [63.0, 117.0].into());

        b *= b;
        assert_eq!(b, [-9720.0, 14742.0].into());
    }

    #[test]
    fn test_c64_multiplication() {
        let mut a: f64 = 3.0;
        let mut b: c64 = [7.0, 13.0].into();

        assert_eq!(a * a, 9.0);
        assert_eq!(a * b, [21.0, 39.0].into());
        assert_eq!(b * a, [21.0, 39.0].into());
        assert_eq!(b * b, [-120.0, 182.0].into());

        a *= a;
        assert_eq!(a, 9.0);

        b *= a;
        assert_eq!(b, [63.0, 117.0].into());

        b *= b;
        assert_eq!(b, [-9720.0, 14742.0].into());
    }

    #[test]
    fn test_c32_division() {
        let mut a: f32 = 24.0;
        let mut b: c32 = [12.0, 240.0].into();

        assert_eq!(a / a, 1.0);
        assert_eq!(a / b, [2.0 / 401.0, -40.0 / 401.0].into());
        assert_eq!(b / a, [0.5, 10.0].into());
        assert_eq!(b / b, [1.0, 0.0].into());

        a /= 0.5 * a;
        assert_eq!(a, 2.0);

        b /= a;
        assert_eq!(b, [6.0, 120.0].into());

        b /= b;
        assert_eq!(b, [1.0, 0.0].into());
    }

    #[test]
    fn test_c32_negation() {
        let a: c32 = [3.0, 7.0].into();
        let b: c32 = [-3.0, -7.0].into();

        assert_eq!(-a, [-3.0, -7.0].into());
        assert_eq!(-b, [3.0, 7.0].into());
    }

    #[test]
    fn test_c32_powi() {
        let a: c32 = [3.0, 7.0].into();

        assert_eq!(a.powi(0), [1.0, 0.0].into());
        assert_eq!(a.powi(1), [3.0, 7.0].into());
        assert_eq!(a.powi(2), [-40.0, 42.0].into());
        assert_eq!(a.powi(3), [-414.0, -154.0].into());
        assert_eq!(a.powi(4), [-164.0, -3360.0].into());

        let b: c32 = [0.0, 0.0].into();
        assert_eq!(b.powi(0), [1.0, 0.0].into());
        assert_eq!(b.powi(1), [0.0, 0.0].into());
        assert_eq!(b.powi(2), [0.0, 0.0].into());
        assert_eq!(b.powi(3), [0.0, 0.0].into());

        let c: c32 = [0.0, 1.0].into();
        assert_eq!(c.powi(0), [1.0, 0.0].into());
        assert_eq!(c.powi(1), [0.0, 1.0].into());
        assert_eq!(c.powi(2), [-1.0, 0.0].into());
        assert_eq!(c.powi(3), [0.0, -1.0].into());
        assert_eq!(c.powi(4), [1.0, 0.0].into());

        let d: c32 = [-11.0, -47.0].into();
        assert_eq!(d.powi(0), [1.0, 0.0].into());
        assert_eq!(d.powi(1), [-11.0, -47.0].into());
        assert_eq!(d.powi(2), [-2088.0, 1034.0].into());
        assert_eq!(d.powi(3), [71566.0, 86762.0].into());
        assert_eq!(d.powi(4), [3290588.0, -4317984.0].into());
    }

    #[test]
    fn test_f32_sqrt() {
        let a: f32 = 2.0;
        assert_eq!(a.sqrt(), 2_f32.sqrt());

        let b: f32 = 0.0;
        assert_eq!(b.sqrt(), 0_f32.sqrt());

        let c: f32 = -2.0;
        assert_eq!(c.sqrt().is_nan(), true);

        let d: f32 = 1.0;
        assert_eq!(d.sqrt(), 1_f32.sqrt());
    }

    #[test]
    fn test_c32_sqrt() {
        let a: c32 = [0.0, 0.0].into();
        assert_eq!(a.sqrt(), [0.0, 0.0].into());

        let b: c32 = [1.0, 0.0].into();
        assert_eq!(b.sqrt(), [1.0, 0.0].into());

        let c: c32 = [0.0, 1.0].into();
        assert_eq!(c.sqrt(), [0.7071067811865476, 0.7071067811865476].into());

        let d: c32 = [1.0, 1.0].into();
        assert_eq!(d.sqrt(), [1.09868411346781, 0.45508986056222733].into());

        let e: c32 = [1.0, -1.0].into();
        assert_eq!(e.sqrt(), [1.09868411346781, -0.45508986056222733].into());
    }
}
