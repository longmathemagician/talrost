use crate::float::*;
use core::ops::*;

macro_rules! create_complex {
    ($complex: ident, $base_type: ty, $complex_trait: ident) => {
        trait $complex_trait:
            Sized + Copy + Clone + Add<Output = $base_type> + Into<$base_type>
        {
        }
        impl $complex_trait for $base_type {}

        #[allow(non_camel_case_types)]
        #[derive(Clone, Copy, PartialEq, PartialOrd, Debug)]
        pub struct $complex([$base_type; 2]);

        impl $complex {
            pub fn new(c: [$base_type; 2]) -> Self {
                $complex(c)
            }

            pub fn re(&self) -> $base_type {
                self.0[0]
            }

            pub fn im(&self) -> $base_type {
                self.0[1]
            }

            pub fn mag(&self) -> $base_type {
                (self.re().powi(2) + self.im().powi(2)).sqrt()
            }
        }

        impl Float for $complex {
            const NAN: Self = $complex([<$base_type>::NAN, <$base_type>::NAN]);

            const ZERO: Self = $complex([<$base_type>::ZERO, <$base_type>::ZERO]);

            const ONE: Self = $complex([<$base_type>::ONE, <$base_type>::ZERO]);

            fn abs(&self) -> Self {
                $complex([self.mag(), 0.0])
            }

            fn powi(&self, power: i32) -> Self {
                let mut result = $complex([1.0, 0.0]);
                for _ in 0..power {
                    result *= *self;
                }
                result
            }
        }

        // Implement core::display
        impl core::fmt::Display for $complex {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                if self.0[1] < 0.0 {
                    write!(f, "({} - {}i)", self.0[0], -self.0[1])
                } else {
                    write!(f, "({} + {}i)", self.0[0], self.0[1])
                }
            }
        }

        // Implement Addition
        impl Add for $complex {
            // Add for Self
            type Output = Self;

            fn add(self, rhs: Self) -> Self::Output {
                $complex([self.0[0] + rhs.0[0], self.0[1] + rhs.0[1]])
            }
        }
        impl<U: $complex_trait> Add<U> for $complex {
            // Add for $num_trait
            type Output = Self;

            fn add(self, rhs: U) -> Self::Output {
                $complex([self.0[0] + rhs.into(), self.0[1]])
            }
        }
        impl Add<$complex> for $base_type {
            // Add for $base_type
            type Output = $complex;

            fn add(self, rhs: $complex) -> Self::Output {
                $complex([self + rhs.0[0], rhs.0[1]])
            }
        }
        impl AddAssign for $complex {
            // AddAssign for Self
            fn add_assign(&mut self, rhs: Self) {
                self.0[0] += rhs.0[0];
                self.0[1] += rhs.0[1];
            }
        }
        impl<U: $complex_trait> AddAssign<U> for $complex {
            // AddAssign for $num_trait
            fn add_assign(&mut self, rhs: U) {
                self.0[0] += rhs.into();
            }
        }
        // impl AddAssign<$complex> for $base_type {
        //     // AddAssign for $base_type
        //     // CAN'T DO THIS BECAUSE OF THE IMPLICIT CONVERSION
        //     fn add_assign(&mut self, rhs: $complex) {
        //         *self += rhs.0[0];
        //     }
        // }

        // Implement Subtraction
        impl Sub for $complex {
            // Sub for Self
            type Output = Self;

            fn sub(self, rhs: Self) -> Self::Output {
                $complex([self.0[0] - rhs.0[0], self.0[1] - rhs.0[1]])
            }
        }
        impl<U: $complex_trait> Sub<U> for $complex {
            // Sub for $num_trait
            type Output = Self;

            fn sub(self, rhs: U) -> Self::Output {
                $complex([self.0[0] - rhs.into(), self.0[1]])
            }
        }
        impl Sub<$complex> for $base_type {
            // Sub for $base_type
            type Output = $complex;

            fn sub(self, rhs: $complex) -> Self::Output {
                $complex([self - rhs.0[0], -rhs.0[1]])
            }
        }
        impl SubAssign for $complex {
            // SubAssign for Self
            fn sub_assign(&mut self, rhs: Self) {
                self.0[0] -= rhs.0[0];
                self.0[1] -= rhs.0[1];
            }
        }
        impl<U: $complex_trait> SubAssign<U> for $complex {
            // SubAssign for $num_trait
            fn sub_assign(&mut self, rhs: U) {
                self.0[0] -= rhs.into();
            }
        }
        // impl SubAssign<$complex> for $base_type {
        //     // SubAssign for $base_type
        //     // CAN'T DO THIS BECAUSE OF THE IMPLICIT CONVERSION
        //     fn sub_assign(&mut self, rhs: $complex) {
        //         *self -= rhs.0[0];
        //     }
        // }

        // Implement Multiplication
        impl Mul for $complex {
            // Mul for Self
            type Output = Self;

            fn mul(self, rhs: Self) -> Self::Output {
                $complex([
                    self.0[0] * rhs.0[0] - self.0[1] * rhs.0[1],
                    self.0[0] * rhs.0[1] + self.0[1] * rhs.0[0],
                ])
            }
        }
        impl<U: $complex_trait> Mul<U> for $complex {
            // Mul for $num_trait
            type Output = Self;

            fn mul(self, rhs: U) -> Self::Output {
                $complex([self.0[0] * rhs.into(), self.0[1] * rhs.into()])
            }
        }
        impl Mul<$complex> for $base_type {
            // Mul for $base_type
            type Output = $complex;

            fn mul(self, rhs: $complex) -> Self::Output {
                $complex([self * rhs.0[0], self * rhs.0[1]])
            }
        }
        impl MulAssign for $complex {
            // MulAssign for Self
            fn mul_assign(&mut self, rhs: Self) {
                let re = self.0[0] * rhs.0[0] - self.0[1] * rhs.0[1];
                let im = self.0[0] * rhs.0[1] + self.0[1] * rhs.0[0];
                self.0[0] = re;
                self.0[1] = im;
            }
        }
        impl<U: $complex_trait> MulAssign<U> for $complex {
            // MulAssign for $num_trait
            fn mul_assign(&mut self, rhs: U) {
                self.0[0] *= rhs.into();
                self.0[1] *= rhs.into();
            }
        }
        // impl MulAssign<$complex> for $base_type {
        //     // MulAssign for $base_type
        //     // CAN'T DO THIS BECAUSE OF THE IMPLICIT CONVERSION
        //     fn mul_assign(&mut self, rhs: $complex) {
        //         *self *= rhs.0[0];
        //     }
        // }

        // Implement Division
        impl Div for $complex {
            // Div for Self
            type Output = Self;

            fn div(self, rhs: Self) -> Self::Output {
                let denom = rhs.0[0] * rhs.0[0] + rhs.0[1] * rhs.0[1];
                $complex([
                    (self.0[0] * rhs.0[0] + self.0[1] * rhs.0[1]) / denom,
                    (self.0[1] * rhs.0[0] - self.0[0] * rhs.0[1]) / denom,
                ])
            }
        }
        impl<U: $complex_trait> Div<U> for $complex {
            // Div for $num_trait
            type Output = Self;

            fn div(self, rhs: U) -> Self::Output {
                $complex([self.0[0] / rhs.into(), self.0[1] / rhs.into()])
            }
        }
        impl Div<$complex> for $base_type {
            // Div for $base_type
            type Output = $complex;

            fn div(self, rhs: $complex) -> Self::Output {
                let denom = rhs.0[0] * rhs.0[0] + rhs.0[1] * rhs.0[1];
                $complex([self * rhs.0[0] / denom, -self * rhs.0[1] / denom])
            }
        }
        impl DivAssign for $complex {
            // DivAssign for Self
            fn div_assign(&mut self, rhs: Self) {
                let denom = rhs.0[0] * rhs.0[0] + rhs.0[1] * rhs.0[1];
                let re = (self.0[0] * rhs.0[0] + self.0[1] * rhs.0[1]) / denom;
                let im = (self.0[1] * rhs.0[0] - self.0[0] * rhs.0[1]) / denom;
                self.0[0] = re;
                self.0[1] = im;
            }
        }
        impl<U: $complex_trait> DivAssign<U> for $complex {
            // DivAssign for $num_trait
            fn div_assign(&mut self, rhs: U) {
                self.0[0] /= rhs.into();
                self.0[1] /= rhs.into();
            }
        }
        // impl DivAssign<$complex> for $base_type {
        //     // DivAssign for $base_type
        //     // CAN'T DO THIS BECAUSE OF THE IMPLICIT CONVERSION
        //     fn div_assign(&mut self, rhs: $complex) {
        //         let denom = rhs.0[0] * rhs.0[0] + rhs.0[1] * rhs.0[1];
        //         *self *= rhs.0[0] / denom;
        //     }
        // }

        // Implement Negation
        impl Neg for $complex {
            type Output = Self;

            fn neg(self) -> Self::Output {
                $complex([-self.0[0], -self.0[1]])
            }
        }

        // Implement From<$base_type> for $complex
        impl From<$base_type> for $complex {
            fn from(re: $base_type) -> Self {
                $complex([re, 0.0])
            }
        }

        // Implement From<[$base_type; 2]> for $complex
        impl From<[$base_type; 2]> for $complex {
            fn from(c: [$base_type; 2]) -> Self {
                $complex(c)
            }
        }

        // Implement From<($base_type, $base_type)> for $complex
        impl From<($base_type, $base_type)> for $complex {
            fn from(c: ($base_type, $base_type)) -> Self {
                $complex([c.0, c.1])
            }
        }

        // Implement From<&str> for $complex
        // Example: "2 + 0i" -> [2.0, 0.0]
        impl From<&str> for $complex {
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
                $complex([
                    re.parse::<$base_type>().unwrap(),
                    im.parse::<$base_type>().unwrap(),
                ])
            }
        }
    };
}

create_complex!(c32, f32, Complex32);
create_complex!(c64, f64, Complex64);

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
}
