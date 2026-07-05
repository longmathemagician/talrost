//! Algebra law checks: each tower trait's axioms verified on fixed sample
//! triples, macro-stamped per concrete type.
//!
//! Samples are chosen to be exactly representable (small integers and
//! dyadic rationals), so every semiring/ring law can be asserted with `==`
//! even for floats — associativity and distributivity of IEEE-754 arithmetic
//! only fail through rounding, and these inputs never round. The one
//! genuinely inexact law, `x·recip(x) == ONE`, is checked to a per-type
//! tolerance through a caller-supplied closeness predicate.

use talrost::algebra::{Field, Monoid, Semiring};
use talrost::complex::{c32, c64};
use talrost::dual::Dual;

/// Semiring laws: additive/multiplicative identities, associativity,
/// commutativity, and distributivity on a fixed sample triple.
macro_rules! semiring_laws {
    ($name:ident, $t:ty, [$a:expr, $b:expr, $c:expr]) => {
        #[test]
        fn $name() {
            let (a, b, c): ($t, $t, $t) = ($a, $b, $c);

            // Identities.
            assert_eq!(a + <$t>::ZERO, a, "a + 0 == a");
            assert_eq!(<$t>::ZERO + a, a, "0 + a == a");
            assert_eq!(a * <$t>::ONE, a, "a · 1 == a");
            assert_eq!(<$t>::ONE * a, a, "1 · a == a");
            assert_eq!(a * <$t>::ZERO, <$t>::ZERO, "a · 0 == 0");

            // Associativity.
            assert_eq!((a + b) + c, a + (b + c), "(a + b) + c == a + (b + c)");
            assert_eq!((a * b) * c, a * (b * c), "(a · b) · c == a · (b · c)");

            // Commutativity.
            assert_eq!(a + b, b + a, "a + b == b + a");
            assert_eq!(a * b, b * a, "a · b == b · a");

            // Distributivity.
            assert_eq!(a * (b + c), a * b + a * c, "a · (b + c) == a·b + a·c");
            assert_eq!((a + b) * c, a * c + b * c, "(a + b) · c == a·c + b·c");
        }
    };
}

/// Group laws: additive inverses on each element of the sample triple.
macro_rules! group_laws {
    ($name:ident, $t:ty, [$a:expr, $b:expr, $c:expr]) => {
        #[test]
        fn $name() {
            for x in [$a, $b, $c] as [$t; 3] {
                assert_eq!(x + (-x), <$t>::ZERO, "x + (−x) == 0");
                assert_eq!(x - x, <$t>::ZERO, "x − x == 0");
                assert_eq!(-(-x), x, "−(−x) == x");
            }
        }
    };
}

/// Field laws: `x·recip(x) ≈ ONE` (up to the closeness predicate `$close`)
/// and division as multiplication by the reciprocal, on each nonzero sample.
macro_rules! field_laws {
    ($name:ident, $t:ty, [$a:expr, $b:expr, $c:expr], $close:expr) => {
        #[test]
        fn $name() {
            let close: fn($t, $t) -> bool = $close;
            for x in [$a, $b, $c] as [$t; 3] {
                assert!(close(x * x.recip(), <$t>::ONE), "x · x⁻¹ ≈ 1");
                assert!(close(x.recip() * x, <$t>::ONE), "x⁻¹ · x ≈ 1");
                assert!(close(x / x, <$t>::ONE), "x / x ≈ 1");
            }
            // Division against multiplication on a mixed pair.
            let (a, b): ($t, $t) = ($a, $b);
            assert!(close((a * b) / b, a), "(a·b)/b ≈ a");
        }
    };
}

// --- u32: a semiring (no additive inverses, no division) ------------------

semiring_laws!(semiring_u32, u32, [2, 3, 5]);

// --- i32: a ring ------------------------------------------------------------

semiring_laws!(semiring_i32, i32, [-2, 3, 5]);
group_laws!(group_i32, i32, [-2, 3, 5]);

// --- f32 / f64: fields (dyadic samples keep the ring laws exact) ----------

semiring_laws!(semiring_f32, f32, [-2.0, 0.5, 4.0]);
group_laws!(group_f32, f32, [-2.0, 0.5, 4.0]);
field_laws!(field_f32, f32, [-2.0, 0.5, 3.0], |x, y| (x - y).abs()
    <= f32::EPSILON);

semiring_laws!(semiring_f64, f64, [-2.0, 0.5, 4.0]);
group_laws!(group_f64, f64, [-2.0, 0.5, 4.0]);
field_laws!(field_f64, f64, [-2.0, 0.5, 3.0], |x, y| (x - y).abs()
    <= f64::EPSILON);

// --- c32 / c64: fields ------------------------------------------------------

semiring_laws!(
    semiring_c32,
    c32,
    [c32::new(1.0, 2.0), c32::new(-3.0, 0.5), c32::new(0.0, -4.0)]
);
group_laws!(
    group_c32,
    c32,
    [c32::new(1.0, 2.0), c32::new(-3.0, 0.5), c32::new(0.0, -4.0)]
);
field_laws!(
    field_c32,
    c32,
    [c32::new(1.0, 2.0), c32::new(-3.0, 0.5), c32::new(0.0, -4.0)],
    |x, y| (x - y).magnitude() <= 4.0 * f32::EPSILON
);

semiring_laws!(
    semiring_c64,
    c64,
    [c64::new(1.0, 2.0), c64::new(-3.0, 0.5), c64::new(0.0, -4.0)]
);
group_laws!(
    group_c64,
    c64,
    [c64::new(1.0, 2.0), c64::new(-3.0, 0.5), c64::new(0.0, -4.0)]
);
field_laws!(
    field_c64,
    c64,
    [c64::new(1.0, 2.0), c64::new(-3.0, 0.5), c64::new(0.0, -4.0)],
    |x, y| (x - y).magnitude() <= 4.0 * f64::EPSILON
);

// --- Dual<f64>: a ring, and a field wherever the standard part is nonzero --

const D_A: Dual<f64> = Dual { val: 2.0, der: 3.0 };
const D_B: Dual<f64> = Dual {
    val: -1.0,
    der: 0.5,
};
const D_C: Dual<f64> = Dual {
    val: 4.0,
    der: -2.0,
};

semiring_laws!(semiring_dual_f64, Dual<f64>, [D_A, D_B, D_C]);
group_laws!(group_dual_f64, Dual<f64>, [D_A, D_B, D_C]);
field_laws!(field_dual_f64, Dual<f64>, [D_A, D_B, D_C], |x, y| {
    (x.val - y.val).abs() <= 4.0 * f64::EPSILON && (x.der - y.der).abs() <= 4.0 * f64::EPSILON
});
