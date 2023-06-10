use crate::natural::Natural;

pub trait Number {
    type Type: Natural;
    fn new(value: Self::Type) -> Self::Type;
}

impl<T: Natural> Number for T {
    type Type = T;
    fn new(value: Self::Type) -> Self::Type {
        value
    }
}

#[cfg(test)]
mod tests {
    use crate::complex::Complex;

    use super::*;

    fn test_numbers_pow2<N: Number + Natural>(n: N, answer: N) {
        assert_eq!(n.powi(2), answer);
    }

    #[test]
    fn test_numbers() {
        let unsigned_32 = 16_u32;
        test_numbers_pow2(unsigned_32, 256_u32);

        let unsigned_64 = 32_u64;
        test_numbers_pow2(unsigned_64, 1024_u64);

        let signed_32 = 64_i32;
        test_numbers_pow2(signed_32, 4096_i32);

        let signed_64 = 128_i64;
        test_numbers_pow2(signed_64, 16384_i64);

        let float_32 = 256_f32;
        test_numbers_pow2(float_32, 65536_f32);

        let float_64 = 512_f64;
        test_numbers_pow2(float_64, 262144_f64);

        let complex_32 = Complex::new(10_f32, 5_f32);
        test_numbers_pow2(complex_32, Complex::new(75_f32, 100_f32));

        let complex_64 = Complex::new(1_f64, 1_f64);
        test_numbers_pow2(complex_64, Complex::new(0_f64, 2_f64));
    }
}
