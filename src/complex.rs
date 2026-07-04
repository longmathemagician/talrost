use crate::algebra::*;
use crate::element::Element;
use crate::real::Real;
use core::fmt::Debug;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

// No `PartialOrd`: lexicographic order on ℂ is mathematically meaningless.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex<F>
where
    F: Real,
{
    pub re: F,
    pub im: F,
}

impl<F> Complex<F>
where
    F: Real,
{
    #[allow(dead_code)]
    const NAN: Self = Self {
        re: F::NAN,
        im: F::NAN,
    };

    #[allow(dead_code)]
    const INFINITY: Self = Self {
        re: F::INFINITY,
        im: F::INFINITY,
    };

    #[allow(dead_code)]
    const EPSILON: Self = Self {
        re: F::EPSILON,
        im: F::ZERO,
    };

    const ZERO: Self = Self {
        re: F::ZERO,
        im: F::ZERO,
    };

    const ONE: Self = Self {
        re: F::ONE,
        im: F::ZERO,
    };

    #[allow(dead_code)]
    #[allow(non_upper_case_globals)]
    const i: Self = Self {
        re: F::ZERO,
        im: F::ONE,
    };

    #[allow(dead_code)]
    const J: Self = Self::i;

    pub fn new(re: F, im: F) -> Self {
        Self { re, im }
    }

    pub fn magnitude(&self) -> F {
        (self.re.powi(2) + self.im.powi(2)).sqrt()
    }

    /// The argument (phase angle) of `z`, in `(-π, π]`.
    pub fn arg(self) -> F {
        self.im.atan2(self.re)
    }

    /// Builds `r·e^(iθ) = r·(cos θ + i sin θ)`.
    pub fn from_polar(r: F, theta: F) -> Self {
        let (s, c) = theta.sin_cos();
        Self::new(r * c, r * s)
    }

    /// The complex exponential `e^z = e^re·(cos im + i sin im)`.
    pub fn exp(self) -> Self {
        Self::from_polar(self.re.exp(), self.im)
    }

    /// The principal natural logarithm, `ln|z| + i·arg z`.
    ///
    /// `ln|z|` is computed on components scaled by `max(|re|, |im|)` so it
    /// stays finite where `re² + im²` would overflow or underflow (the same
    /// regime Smith's division algorithm protects).
    pub fn ln(self) -> Self {
        let (a, b) = (self.re.abs(), self.im.abs());
        let m = if a >= b { a } else { b };
        if m == F::ZERO {
            return Self::new(F::NEG_INFINITY, self.arg());
        }
        let (x, y) = (self.re / m, self.im / m);
        let half = F::ONE / F::from_u32(2);
        let ln_mod = m.ln() + (x * x + y * y).ln() * half;
        Self::new(ln_mod, self.arg())
    }

    /// Raises `z` to a real power via the polar form:
    /// `z^n = |z|^n · e^(i·n·arg z)`, with `|z|^n = e^(n·ln|z|)` computed
    /// through the overflow-safe [`Complex::ln`].
    pub fn powf(self, n: F) -> Self {
        if self == Self::ZERO {
            // 0^n: 1 for n == 0 (the empty product), 0 for n > 0, and +inf
            // for n < 0 via exp(-inf · n).
            if n == F::ZERO {
                return Self::ONE;
            }
            return Self::from_polar((F::NEG_INFINITY * n).exp(), F::ZERO);
        }
        let w = self.ln();
        Self::from_polar((w.re * n).exp(), w.im * n)
    }

    /// Raises `z` to an integer power by exponentiation-by-squaring
    /// (`O(log n)` complex multiplies). Negative exponents go through
    /// [`Field::recip`] first.
    pub fn powi(self, n: i32) -> Self {
        let mut base = if n < 0 { Field::recip(self) } else { self };
        let mut exp = n.unsigned_abs();
        let mut acc = Self::ONE;
        while exp > 0 {
            if exp & 1 == 1 {
                acc *= base;
            }
            base *= base;
            exp >>= 1;
        }
        acc
    }

    /// The `k`-th of the `n` `n`-th roots of unity, `e^(2πik/n)`.
    ///
    /// Start solutions of binomial systems are radius-scaled roots of unity;
    /// this walks them without any allocation. `k` is taken mod nothing —
    /// values `>= n` simply wrap around the circle.
    pub fn nth_root_of_unity(k: u32, n: u32) -> Self {
        debug_assert!(n > 0, "nth_root_of_unity: n must be positive");
        let theta = F::TAU * F::from_u32(k) / F::from_u32(n);
        Self::from_polar(F::ONE, theta)
    }

    pub fn sqrt(self) -> Self {
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

}

/// `(a + bi) / (c + di)` by Smith's algorithm (1962): scale by the larger of
/// `|c|`, `|d|` so the intermediate products stay near the magnitude of the
/// result. The textbook form divides by `c² + d²`, which overflows/underflows
/// exactly in the near-singular regimes path tracking visits (components
/// around `1e±300` in `f64`), even though the quotient itself is
/// representable.
#[inline]
fn smith_div<F: Real>(a: F, b: F, c: F, d: F) -> (F, F) {
    if c.abs() >= d.abs() {
        let r = d / c;
        let den = c + d * r;
        ((a + b * r) / den, (b - a * r) / den)
    } else {
        let r = c / d;
        let den = c * r + d;
        ((a * r + b) / den, (b * r - a) / den)
    }
}

// Complex numbers form a field but are neither ordered nor integer-like:
// they implement the algebraic stack (through `Field`) plus `Scalar` (in
// `crate::scalar`), and nothing from the `Natural`/`Integer`/`Real` families.
// These impls are generic over F so that `impl<F: Real> Scalar for Complex<F>`
// can rely on `Complex<F>: Field` for every real component type.
impl<F: Real> Element for Complex<F> {}

impl<F: Real> Monoid for Complex<F> {
    const ZERO: Self = Self {
        re: F::ZERO,
        im: F::ZERO,
    };
}

impl<F: Real> Group for Complex<F> {}

impl<F: Real> Semiring for Complex<F> {
    const ONE: Self = Self {
        re: F::ONE,
        im: F::ZERO,
    };
}

impl<F: Real> Ring for Complex<F> {}

impl<F: Real> Field for Complex<F> {
    fn recip(self) -> Self {
        // Through Smith division (`1 / z`), not `conj(z) / |z|²`: the latter
        // overflows/underflows for components around 1e±300 in f64.
        Self::ONE / self
    }
}

// Real coefficients evaluated at complex points: one of the concrete
// `Algebra` instances (the blanket impl separately gives
// `Complex<F>: Algebra<Complex<F>>`).
impl<F: Real> Algebra<F> for Complex<F> {}

// Implement core::fmt::Display for Complex<F>
impl<F> core::fmt::Display for Complex<F>
where
    F: Real + core::fmt::Display,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.im < F::ZERO {
            write!(f, "{} - {}i", self.re, -self.im)
        } else {
            write!(f, "{} + {}i", self.re, self.im)
        }
    }
}

// Implement core::iter::Sum for Complex<F> (required by `Monoid`)
impl<F> core::iter::Sum for Complex<F>
where
    F: Real,
{
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |a, b| a + b)
    }
}

/// Error returned when parsing a string into a [`Complex`] fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParseComplexError;

impl core::fmt::Display for ParseComplexError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("invalid complex number syntax")
    }
}

impl core::error::Error for ParseComplexError {}

// Implement core::str::FromStr for Complex<F>.
// Accepts "a + bi", "a - bi", "a", "bi", "-a - bi" (and bare "i"/"-i"), with
// tolerant whitespace around the tokens; round-trips `Display` output.
// Core-only slice parsing: no allocation, works in `no_std` without `alloc`.
impl<F> core::str::FromStr for Complex<F>
where
    F: Real + core::str::FromStr,
{
    type Err = ParseComplexError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        if s.is_empty() {
            return Err(ParseComplexError);
        }
        let Some(body) = s.strip_suffix('i') else {
            // Purely real: "a"
            let re = s.parse::<F>().map_err(|_| ParseComplexError)?;
            return Ok(Self::new(re, F::ZERO));
        };
        let body = body.trim_end();

        // Split at the last '+'/'-' that is neither the leading sign (index 0;
        // `body` starts with a non-space because `s` was trimmed) nor an
        // exponent sign (directly preceded by 'e'/'E', as in "1e-3").
        let bytes = body.as_bytes();
        let mut split = None;
        for (idx, &b) in bytes.iter().enumerate().skip(1) {
            if (b == b'+' || b == b'-') && !matches!(bytes[idx - 1], b'e' | b'E') {
                split = Some(idx);
            }
        }
        let (re, sign, magnitude) = match split {
            Some(idx) => {
                let re = body[..idx]
                    .trim_end()
                    .parse::<F>()
                    .map_err(|_| ParseComplexError)?;
                let sign = if bytes[idx] == b'-' { -F::ONE } else { F::ONE };
                (re, sign, body[idx + 1..].trim_start())
            }
            // No separator: the whole body is the (signed) imaginary part.
            None => match bytes.first() {
                Some(b'+') => (F::ZERO, F::ONE, body[1..].trim_start()),
                Some(b'-') => (F::ZERO, -F::ONE, body[1..].trim_start()),
                _ => (F::ZERO, F::ONE, body),
            },
        };
        let im = if magnitude.is_empty() {
            sign // bare "i", "-i", "a + i", "a - i"
        } else {
            sign * magnitude.parse::<F>().map_err(|_| ParseComplexError)?
        };
        Ok(Self::new(re, im))
    }
}

// Implement From for (F, F) to Complex<F>
impl<F> From<(F, F)> for Complex<F>
where
    F: Real,
{
    fn from(value: (F, F)) -> Self {
        Self::new(value.0, value.1)
    }
}

// Implement From for [F, F] to Complex<F>
impl<F> From<[F; 2]> for Complex<F>
where
    F: Real,
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
    F: Real,
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
    F: Real,
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
    F: Real,
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
    F: Real,
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
    F: Real,
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
    F: Real,
{
    type Output = Self;
    fn mul(self, rhs: F) -> Self::Output {
        Self {
            re: self.re * rhs,
            im: self.im * rhs,
        }
    }
}

// Implement core::ops::Div for Complex<F>, via Smith's algorithm.
impl<F> Div for Complex<F>
where
    F: Real,
{
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        let (re, im) = smith_div(self.re, self.im, rhs.re, rhs.im);
        Self { re, im }
    }
}

// Implement core::ops::Div for Complex<F> where RHS is F
impl<F> Div<F> for Complex<F>
where
    F: Real,
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
    F: Real,
{
    fn add_assign(&mut self, rhs: Self) {
        self.re += rhs.re;
        self.im += rhs.im;
    }
}

// Implement core::ops::AddAssign for Complex<F> where RHS is F
impl<F> AddAssign<F> for Complex<F>
where
    F: Real,
{
    fn add_assign(&mut self, rhs: F) {
        self.re += rhs;
    }
}

// Implement core::ops::SubAssign for Complex<F>
impl<F> SubAssign for Complex<F>
where
    F: Real,
{
    fn sub_assign(&mut self, rhs: Self) {
        self.re -= rhs.re;
        self.im -= rhs.im;
    }
}

// Implement core::ops::SubAssign for Complex<F> where RHS is F
impl<F> SubAssign<F> for Complex<F>
where
    F: Real,
{
    fn sub_assign(&mut self, rhs: F) {
        self.re -= rhs;
    }
}

// Implement core::ops::MulAssign for Complex<F>
impl<F> MulAssign for Complex<F>
where
    F: Real,
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
    F: Real,
{
    fn mul_assign(&mut self, rhs: F) {
        self.re *= rhs;
        self.im *= rhs;
    }
}

// Implement core::ops::DivAssign for Complex<F>, via Smith's algorithm.
impl<F> DivAssign for Complex<F>
where
    F: Real,
{
    fn div_assign(&mut self, rhs: Self) {
        let (re, im) = smith_div(self.re, self.im, rhs.re, rhs.im);
        self.re = re;
        self.im = im;
    }
}

// Implement core::ops::DivAssign for Complex<F> where RHS is F
impl<F> DivAssign<F> for Complex<F>
where
    F: Real,
{
    fn div_assign(&mut self, rhs: F) {
        self.re /= rhs;
        self.im /= rhs;
    }
}

// Implement core::ops::Neg for Complex<F>
impl<F> Neg for Complex<F>
where
    F: Real,
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
    F: Real,
{
    fn from(value: F) -> Self {
        Self {
            re: value,
            im: F::ZERO,
        }
    }
}

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

// Implement core::ops::Div for Complex<f64> where Self is f64, via Smith's
// algorithm.
impl Div<Complex<f64>> for f64 {
    type Output = Complex<f64>;
    fn div(self, rhs: Complex<f64>) -> Self::Output {
        let (re, im) = smith_div(self, 0.0, rhs.re, rhs.im);
        Self::Output { re, im }
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

// Implement core::ops::Div for Complex<f32> where Self is f32, via Smith's
// algorithm.
impl Div<Complex<f32>> for f32 {
    type Output = Complex<f32>;
    fn div(self, rhs: Complex<f32>) -> Self::Output {
        let (re, im) = smith_div(self, 0.0, rhs.re, rhs.im);
        Self::Output { re, im }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        assert_eq!(c64::ZERO, c64::new(0.0, 0.0));
        assert_eq!(c64::ONE, c64::new(1.0, 0.0));
        assert_eq!(c64::i, c64::new(0.0, 1.0));
        assert_eq!(c64::J, c64::new(0.0, 1.0));
    }

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

    /// |a - b| <= tol componentwise.
    fn approx_c32(a: c32, b: c32, tol: f32) -> bool {
        (a.re - b.re).abs() <= tol && (a.im - b.im).abs() <= tol
    }

    fn approx_c64(a: c64, b: c64, tol: f64) -> bool {
        (a.re - b.re).abs() <= tol && (a.im - b.im).abs() <= tol
    }

    #[test]
    fn test_c32_division() {
        // Complex-by-complex quotients go through Smith's algorithm, whose
        // roundings differ from the textbook form by an ulp or so; compare
        // with a tolerance. Division by a *real* stays componentwise-exact.
        let mut a: f32 = 24.0;
        let mut b: c32 = [12.0, 240.0].into();

        assert_eq!(a / a, 1.0);
        assert!(approx_c32(
            a / b,
            [2.0 / 401.0, -40.0 / 401.0].into(),
            1e-9
        ));
        assert_eq!(b / a, [0.5, 10.0].into());
        assert!(approx_c32(b / b, [1.0, 0.0].into(), 1e-6));

        a /= 0.5 * a;
        assert_eq!(a, 2.0);

        b /= a;
        assert_eq!(b, [6.0, 120.0].into());

        b /= b;
        assert!(approx_c32(b, [1.0, 0.0].into(), 1e-6));
    }

    #[test]
    fn test_smith_division_exact_cases() {
        // Quotients whose Smith intermediates are exact stay exact.
        let z = c64::new(1.0, 1.0);
        let w = c64::new(0.5, 0.5);
        assert_eq!(z / w, c64::new(2.0, 0.0));

        let mut q = z;
        q /= w;
        assert_eq!(q, c64::new(2.0, 0.0));

        // Division by a purely real divisor is componentwise.
        assert_eq!(c64::new(3.0, -4.5) / c64::new(1.5, 0.0), c64::new(2.0, -3.0));

        // Multiplicative round trip.
        let n = c64::new(-3.0, 7.0);
        let d = c64::new(2.0, -5.0);
        assert!(approx_c64((n / d) * d, n, 1e-14));
    }

    #[test]
    fn test_smith_division_extreme_magnitudes() {
        // The textbook form computes re² + im² = 1e600 → inf (or 1e-600 → 0)
        // and returns garbage; Smith's algorithm is exact here.
        let big = c64::new(1e300, 1e300);
        assert_eq!(big / big, c64::new(1.0, 0.0));

        let tiny = c64::new(1e-300, 1e-300);
        assert_eq!(tiny / tiny, c64::new(1.0, 0.0));

        // Mixed magnitudes: (1e300 + 1e300 i) / (1e300 i) = 1 - i.
        let d = c64::new(0.0, 1e300);
        assert_eq!(big / d, c64::new(1.0, -1.0));

        // f64-LHS division at the same extremes.
        let q = 1.0 / big;
        assert!(q.re.is_finite() && q.im.is_finite());
        assert!(approx_c64(q * big, c64::new(1.0, 0.0), 1e-15));

        let q = 1.0 / tiny;
        assert!(q.re.is_finite() && q.im.is_finite());
        assert!(approx_c64(q * tiny, c64::new(1.0, 0.0), 1e-15));

        // f32-LHS division near the f32 overflow boundary.
        let big32 = c32::new(1e38, 1e38);
        let q32 = 1.0_f32 / big32;
        assert!(q32.re.is_finite() && q32.im.is_finite());
        let round = q32 * big32;
        assert!((round.re - 1.0).abs() < 1e-6 && round.im.abs() < 1e-6);

        // recip goes through Smith too (conj/|z|² overflows here).
        let r = Field::recip(big);
        assert!(r.re.is_finite() && r.im.is_finite());
        assert!(approx_c64(r * big, c64::new(1.0, 0.0), 1e-15));
    }

    #[test]
    fn test_exp_i_pi() {
        // Euler: e^(iπ) = -1.
        let z = c64::new(0.0, core::f64::consts::PI).exp();
        assert!(approx_c64(z, c64::new(-1.0, 0.0), 1e-15));

        // e^0 = 1, e^(iπ/2) = i.
        assert_eq!(c64::new(0.0, 0.0).exp(), c64::new(1.0, 0.0));
        let i = c64::new(0.0, core::f64::consts::FRAC_PI_2).exp();
        assert!(approx_c64(i, c64::i, 1e-15));

        // exp(a + b) == exp(a)·exp(b).
        let a = c64::new(0.3, -1.2);
        let b = c64::new(-0.7, 0.4);
        assert!(approx_c64((a + b).exp(), a.exp() * b.exp(), 1e-15));
    }

    #[test]
    fn test_ln_exp_round_trip() {
        // ln∘exp is the identity only inside the principal strip |im| ≤ π.
        for z in [
            c64::new(0.5, -0.3),
            c64::new(-1.0, 2.0),
            c64::new(3.0, -2.5),
            c64::new(0.0, 1.0),
        ] {
            assert!(approx_c64(z.exp().ln(), z, 1e-14));
            assert!(approx_c64(z.ln().exp(), z, 1e-14));
        }
        // Outside the strip the argument wraps by 2π.
        let z = c64::new(3.0, 4.0);
        let w = z.exp().ln();
        assert!((w.re - 3.0).abs() < 1e-14);
        assert!((w.im - (4.0 - core::f64::consts::TAU)).abs() < 1e-14);

        // ln(1) = 0, ln(e) = 1, ln(i) = iπ/2.
        assert_eq!(c64::new(1.0, 0.0).ln(), c64::new(0.0, 0.0));
        assert!(approx_c64(
            c64::new(core::f64::consts::E, 0.0).ln(),
            c64::new(1.0, 0.0),
            1e-15
        ));
        assert!(approx_c64(
            c64::i.ln(),
            c64::new(0.0, core::f64::consts::FRAC_PI_2),
            1e-15
        ));

        // ln stays finite where |z|² overflows/underflows.
        let big = c64::new(1e300, 1e300);
        let w = big.ln();
        assert!(w.re.is_finite());
        assert!((w.re - (1e300_f64.ln() + 0.5 * 2.0_f64.ln())).abs() < 1e-12);
        let tiny = c64::new(1e-300, 0.0);
        assert!((tiny.ln().re - 1e-300_f64.ln()).abs() < 1e-12);

        // ln(0) = -inf + 0i (principal).
        let zero_ln = c64::new(0.0, 0.0).ln();
        assert_eq!(zero_ln.re, f64::NEG_INFINITY);
        assert_eq!(zero_ln.im, 0.0);
    }

    #[test]
    fn test_arg() {
        use core::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};
        assert_eq!(c64::new(1.0, 0.0).arg(), 0.0);
        assert_eq!(c64::new(0.0, 1.0).arg(), FRAC_PI_2);
        assert_eq!(c64::new(-1.0, 0.0).arg(), PI);
        assert_eq!(c64::new(0.0, -1.0).arg(), -FRAC_PI_2);
        assert!((c64::new(1.0, 1.0).arg() - FRAC_PI_4).abs() < 1e-15);
    }

    #[test]
    fn test_from_polar() {
        let z = c64::from_polar(2.0, core::f64::consts::FRAC_PI_2);
        assert!(approx_c64(z, c64::new(0.0, 2.0), 1e-15));
        assert_eq!(c64::from_polar(3.0, 0.0), c64::new(3.0, 0.0));

        // Round trip through magnitude/arg.
        let w = c64::new(-3.0, 4.0);
        let back = c64::from_polar(w.magnitude(), w.arg());
        assert!(approx_c64(back, w, 1e-14));
    }

    #[test]
    fn test_powf() {
        // Square/square-root of a positive real.
        assert!(approx_c64(c64::new(4.0, 0.0).powf(0.5), c64::new(2.0, 0.0), 1e-14));
        assert!(approx_c64(c64::new(2.0, 0.0).powf(10.0), c64::new(1024.0, 0.0), 1e-11));

        // i^2 = -1 through the polar form.
        assert!(approx_c64(c64::i.powf(2.0), c64::new(-1.0, 0.0), 1e-15));

        // powf agrees with powi on integer exponents.
        let z = c64::new(1.2, -0.7);
        assert!(approx_c64(z.powf(3.0), z.powi(3), 1e-14));
        assert!(approx_c64(z.powf(-2.0), z.powi(-2), 1e-14));

        // 0^n edges.
        assert_eq!(c64::new(0.0, 0.0).powf(2.0), c64::new(0.0, 0.0));
        assert_eq!(c64::new(0.0, 0.0).powf(0.0), c64::new(1.0, 0.0));
    }

    #[test]
    fn test_powi_negative_exponents() {
        // Exponentiation by squaring handles negative powers via recip.
        let z = c64::new(0.0, 2.0); // 1/z² = 1/(-4) = -0.25
        assert!(approx_c64(z.powi(-2), c64::new(-0.25, 0.0), 1e-15));

        let w = c64::new(3.0, -4.0);
        assert!(approx_c64(w.powi(-1) * w, c64::new(1.0, 0.0), 1e-15));
        assert!(approx_c64(w.powi(-3) * w.powi(3), c64::new(1.0, 0.0), 1e-12));
        assert_eq!(w.powi(0), c64::new(1.0, 0.0));
    }

    #[test]
    fn test_powi_large_exponent() {
        // 2^30 by squaring: exact in f64.
        let two = c64::new(2.0, 0.0);
        assert_eq!(two.powi(30), c64::new(1073741824.0, 0.0));
        // i^4k round trip.
        assert!(approx_c64(c64::i.powi(40), c64::new(1.0, 0.0), 1e-14));
    }

    #[test]
    fn test_nth_roots_of_unity() {
        // k = 0 is exactly 1.
        assert_eq!(c64::nth_root_of_unity(0, 5), c64::new(1.0, 0.0));

        // 4th roots: 1, i, -1, -i.
        assert!(approx_c64(c64::nth_root_of_unity(1, 4), c64::i, 1e-15));
        assert!(approx_c64(c64::nth_root_of_unity(2, 4), c64::new(-1.0, 0.0), 1e-15));
        assert!(approx_c64(c64::nth_root_of_unity(3, 4), c64::new(0.0, -1.0), 1e-15));

        // Each n-th root raised to the n comes back to 1; the full set sums
        // to zero (n > 1).
        for n in [2_u32, 3, 5, 7] {
            let mut sum = c64::new(0.0, 0.0);
            for k in 0..n {
                let w = c64::nth_root_of_unity(k, n);
                assert!((w.magnitude() - 1.0).abs() < 1e-15);
                assert!(approx_c64(w.powi(n as i32), c64::new(1.0, 0.0), 1e-13));
                sum += w;
            }
            assert!(sum.magnitude() < 1e-13);
        }

        // f32 flavor.
        let w = c32::nth_root_of_unity(1, 3);
        assert!((w.powi(3).re - 1.0).abs() < 1e-5);
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
    fn test_powi_squares() {
        // Ported from the deleted `Number` trait tests.
        assert_eq!(c32::new(10.0, 5.0).powi(2), c32::new(75.0, 100.0));
        assert_eq!(c64::new(1.0, 1.0).powi(2), c64::new(0.0, 2.0));
    }

    #[test]
    fn test_magnitude() {
        // Regression: magnitude computed (re^2 + im^2)^2 == |z|^4 instead of sqrt.
        let z = c64::new(3.0, 4.0);
        assert_eq!(z.magnitude(), 5.0);

        let w = c32::new(3.0, 4.0);
        assert_eq!(w.magnitude(), 5.0);
    }

    #[test]
    fn test_normalize() {
        let z = c64::new(3.0, 4.0);
        let n = z.normalize();
        assert!((n.magnitude() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_field_recip() {
        // Regression: `Field::recip` recursed infinitely for Complex (stack overflow).
        assert_eq!(Field::recip(c64::new(2.0, 0.0)), c64::new(0.5, 0.0));
        assert_eq!(Field::recip(c64::new(3.0, 4.0)), c64::new(0.12, -0.16));
    }

    #[test]
    fn test_parse() {
        assert_eq!("2 + 0i".parse::<c64>().unwrap(), c64::new(2.0, 0.0));
        assert_eq!("3 + 4i".parse::<c64>().unwrap(), c64::new(3.0, 4.0));
        assert_eq!("3 - 4i".parse::<c64>().unwrap(), c64::new(3.0, -4.0));
        assert_eq!("-3 - 4i".parse::<c64>().unwrap(), c64::new(-3.0, -4.0));
        assert_eq!("2.5".parse::<c64>().unwrap(), c64::new(2.5, 0.0));
        assert_eq!("-2.5".parse::<c64>().unwrap(), c64::new(-2.5, 0.0));
        assert_eq!("4i".parse::<c64>().unwrap(), c64::new(0.0, 4.0));
        assert_eq!("-4i".parse::<c64>().unwrap(), c64::new(0.0, -4.0));
        assert_eq!("i".parse::<c64>().unwrap(), c64::new(0.0, 1.0));
        assert_eq!("  3+4i ".parse::<c64>().unwrap(), c64::new(3.0, 4.0));
        assert_eq!("1e-3 + 2e-4i".parse::<c64>().unwrap(), c64::new(1e-3, 2e-4));
        assert_eq!("3 + 4i".parse::<c32>().unwrap(), c32::new(3.0, 4.0));

        assert!("".parse::<c64>().is_err());
        assert!("   ".parse::<c64>().is_err());
        assert!("3 & 4i".parse::<c64>().is_err());
        assert!("banana".parse::<c64>().is_err());
    }

    #[test]
    fn test_parse_display_round_trip() {
        // Display prints "a - bi" for negative imaginary parts; parsing must
        // round-trip it (the old From<&str> yielded 0 + 0i for such strings).
        for z in [
            c64::new(3.0, 4.0),
            c64::new(3.0, -4.0),
            c64::new(-3.0, -4.0),
            c64::new(-3.0, 4.0),
            c64::new(0.0, -1.5),
            c64::new(2.0, 0.0),
        ] {
            assert_eq!(z.to_string().parse::<c64>().unwrap(), z);
        }
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
