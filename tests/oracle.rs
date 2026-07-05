//! Property-based oracle tests: talrost's arithmetic checked against
//! independent implementations (`num-complex`, `nalgebra`) and against
//! self-checkable identities (residuals, round-trips) on proptest-generated
//! inputs.

use nalgebra::{Matrix4, Matrix5, SMatrix};
use num_complex::Complex64;
use proptest::prelude::*;

use talrost::complex::c64;
use talrost::dual::Dual;
use talrost::lattice::smith_normal_form;
use talrost::matrix::Matrix;
use talrost::polynomial::Polynomial;
use talrost::scalar::Scalar;
use talrost::vector::Vector;

fn nc(z: c64) -> Complex64 {
    Complex64::new(z.re, z.im)
}

/// Componentwise `|a - b| <= tol` for complex values.
fn close_c(a: c64, b: Complex64, tol: f64) -> bool {
    (a.re - b.re).abs() <= tol && (a.im - b.im).abs() <= tol
}

/// A moderate-magnitude complex number (all four ring ops well inside range).
fn moderate_c64() -> impl Strategy<Value = c64> {
    (-1e6..1e6f64, -1e6..1e6f64).prop_map(|(re, im)| c64::new(re, im))
}

/// A complex number `m·2^e` with mantissa components in ±[0.5, 2] and a
/// shared power-of-two exponent up to ±950 — deep in the regime where the
/// textbook `re² + im²` division overflows or underflows.
fn extreme_c64() -> impl Strategy<Value = c64> {
    (
        prop::sample::select(vec![-1.0f64, 1.0]),
        0.5..2.0f64,
        prop::sample::select(vec![-1.0f64, 1.0]),
        0.5..2.0f64,
        -950..950i32,
    )
        .prop_map(|(sr, mr, si, mi, e)| {
            let s = 2.0f64.powi(e);
            c64::new(sr * mr * s, si * mi * s)
        })
}

/// Largest power-of-two exponent among the components of `z`.
fn exponent_of(z: c64) -> i32 {
    let m = z.re.abs().max(z.im.abs());
    m.log2().floor() as i32
}

proptest! {
    // ------------------------------------------------------------------
    // Complex ring ops vs num-complex.
    // ------------------------------------------------------------------

    #[test]
    fn complex_add_sub_mul_match_num_complex(a in moderate_c64(), b in moderate_c64()) {
        // Same expressions, same roundings: these must agree exactly.
        prop_assert_eq!(nc(a + b), nc(a) + nc(b));
        prop_assert_eq!(nc(a - b), nc(a) - nc(b));
        prop_assert_eq!(nc(a * b), nc(a) * nc(b));
    }

    #[test]
    fn complex_div_matches_num_complex_moderate(a in moderate_c64(), b in moderate_c64()) {
        prop_assume!(b.norm() > 1e-3);
        let q = a / b;
        let r = nc(a) / nc(b);
        // Smith division rounds differently from the textbook form; compare
        // with a relative tolerance.
        let tol = 1e-12 * r.norm().max(1e-300);
        prop_assert!(close_c(q, r, tol), "{:?} vs {:?}", q, r);
    }

    #[test]
    fn complex_div_extreme_magnitudes(a in extreme_c64(), b in extreme_c64()) {
        // Keep the quotient itself representable; everything else is fair
        // game (the textbook form dies when |b|² over/underflows, around
        // |b| ≳ 2^512 or ≲ 2^-512 — well inside this strategy's range).
        let (ka, kb) = (exponent_of(a), exponent_of(b));
        prop_assume!((ka - kb).abs() < 900);

        let q = a / b;

        // Oracle: scale both operands to ~1 by exact powers of two, divide
        // with num-complex where the textbook form is safe, scale back.
        let sa = 2.0f64.powi(-ka);
        let sb = 2.0f64.powi(-kb);
        let r0 = Complex64::new(a.re * sa, a.im * sa) / Complex64::new(b.re * sb, b.im * sb);
        let back = 2.0f64.powi(ka - kb);
        let r = Complex64::new(r0.re * back, r0.im * back);

        prop_assert!(q.re.is_finite() && q.im.is_finite(), "Smith division overflowed: {:?}", q);
        let tol = 1e-12 * r.norm();
        prop_assert!(close_c(q, r, tol), "{:?} vs {:?}", q, r);
    }

    #[test]
    fn complex_exp_matches_num_complex(re in -30.0..30.0f64, im in -30.0..30.0f64) {
        let z = c64::new(re, im);
        let ours = z.exp();
        let reference = nc(z).exp();
        let tol = 1e-12 * reference.norm();
        prop_assert!(close_c(ours, reference, tol), "{:?} vs {:?}", ours, reference);
    }

    #[test]
    fn complex_ln_matches_num_complex(a in moderate_c64()) {
        prop_assume!(a.norm() > 1e-9);
        let ours = a.ln();
        let reference = nc(a).ln();
        prop_assert!(close_c(ours, reference, 1e-12), "{:?} vs {:?}", ours, reference);
    }

    // ------------------------------------------------------------------
    // Matrix numerics vs nalgebra (f64, 4×4 and 5×5).
    // ------------------------------------------------------------------

    #[test]
    fn matmul_4x4_matches_nalgebra(a in prop::array::uniform16(-100.0..100.0f64),
                                   b in prop::array::uniform16(-100.0..100.0f64)) {
        let (ta, tb) = (to_talrost_4(&a), to_talrost_4(&b));
        let (na, nb) = (to_nalgebra_4(&a), to_nalgebra_4(&b));
        let ours = ta * tb;
        let reference = na * nb;
        for i in 0..4 {
            for j in 0..4 {
                let tol = 1e-10 * reference[(i, j)].abs().max(1.0);
                prop_assert!((ours[(i, j)] - reference[(i, j)]).abs() <= tol);
            }
        }
    }

    #[test]
    fn matmul_5x5_matches_nalgebra(a in prop::array::uniform25(-100.0..100.0f64),
                                   b in prop::array::uniform25(-100.0..100.0f64)) {
        let (ta, tb) = (to_talrost_5(&a), to_talrost_5(&b));
        let (na, nb) = (to_nalgebra_5(&a), to_nalgebra_5(&b));
        let ours = ta * tb;
        let reference = na * nb;
        for i in 0..5 {
            for j in 0..5 {
                let tol = 1e-10 * reference[(i, j)].abs().max(1.0);
                prop_assert!((ours[(i, j)] - reference[(i, j)]).abs() <= tol);
            }
        }
    }

    #[test]
    fn determinant_4x4_matches_nalgebra(a in prop::array::uniform16(-3.0..3.0f64)) {
        let ours = to_talrost_4(&a).determinant();
        let reference = to_nalgebra_4(&a).determinant();
        let tol = 1e-9 * reference.abs().max(1.0);
        prop_assert!((ours - reference).abs() <= tol, "{} vs {}", ours, reference);
    }

    #[test]
    fn determinant_5x5_matches_nalgebra(a in prop::array::uniform25(-3.0..3.0f64)) {
        let ours = to_talrost_5(&a).determinant();
        let reference = to_nalgebra_5(&a).determinant();
        let tol = 1e-9 * reference.abs().max(1.0);
        prop_assert!((ours - reference).abs() <= tol, "{} vs {}", ours, reference);
    }

    #[test]
    fn inverse_4x4_matches_nalgebra(a in prop::array::uniform16(-3.0..3.0f64)) {
        let na = to_nalgebra_4(&a);
        // Filter near-singular inputs: both libraries would amplify noise.
        prop_assume!(na.determinant().abs() > 1e-2);
        let reference = na.try_inverse().unwrap();
        let ours = to_talrost_4(&a).inverse().unwrap();
        let scale = reference.norm();
        for i in 0..4 {
            for j in 0..4 {
                prop_assert!((ours[(i, j)] - reference[(i, j)]).abs() <= 1e-8 * scale);
            }
        }
    }

    #[test]
    fn inverse_5x5_matches_nalgebra(a in prop::array::uniform25(-3.0..3.0f64)) {
        let na = to_nalgebra_5(&a);
        prop_assume!(na.determinant().abs() > 1e-2);
        let reference = na.try_inverse().unwrap();
        let ours = to_talrost_5(&a).inverse().unwrap();
        let scale = reference.norm();
        for i in 0..5 {
            for j in 0..5 {
                prop_assert!((ours[(i, j)] - reference[(i, j)]).abs() <= 1e-8 * scale);
            }
        }
    }

    // ------------------------------------------------------------------
    // Root-finder residuals: |p(root)| must be small relative to the
    // polynomial's magnitude at the root.
    // ------------------------------------------------------------------

    #[test]
    fn quadratic_roots_have_small_residuals(c0 in -10.0..10.0f64,
                                            c1 in -10.0..10.0f64,
                                            lead in 0.5..10.0f64,
                                            sign in prop::bool::ANY) {
        let c2 = if sign { lead } else { -lead };
        let p = Polynomial::new([c0, c1, c2]);
        for &r in p.roots(1e-12).as_slice() {
            prop_assert!(p.eval(r).abs() <= 1e-9 * poly_scale(&p.c, r), "residual at {}", r);
        }
    }

    #[test]
    fn cubic_roots_have_small_residuals(c0 in -10.0..10.0f64,
                                        c1 in -10.0..10.0f64,
                                        c2 in -10.0..10.0f64,
                                        lead in 0.5..10.0f64,
                                        sign in prop::bool::ANY) {
        let c3 = if sign { lead } else { -lead };
        let p = Polynomial::new([c0, c1, c2, c3]);
        for &r in p.roots(1e-10).as_slice() {
            prop_assert!(p.eval(r).abs() <= 1e-6 * poly_scale(&p.c, r), "residual at {}", r);
        }
    }

    #[test]
    fn quartic_roots_have_small_residuals(c0 in -10.0..10.0f64,
                                          c1 in -10.0..10.0f64,
                                          c2 in -10.0..10.0f64,
                                          c3 in -10.0..10.0f64,
                                          lead in 0.5..10.0f64,
                                          sign in prop::bool::ANY) {
        let c4 = if sign { lead } else { -lead };
        let p = Polynomial::new([c0, c1, c2, c3, c4]);
        for &r in p.roots(1e-10).as_slice() {
            prop_assert!(p.eval(r).abs() <= 1e-6 * poly_scale(&p.c, r), "residual at {}", r);
        }
    }

    // ------------------------------------------------------------------
    // LU solve residuals.
    // ------------------------------------------------------------------

    #[test]
    fn lu_solve_residual_is_small(a in prop::array::uniform16(-1.0..1.0f64),
                                  b in prop::array::uniform4(-1.0..1.0f64)) {
        let m = to_talrost_4(&a);
        prop_assume!(to_nalgebra_4(&a).determinant().abs() > 1e-3);
        let rhs = Vector::new(b);
        let x = m.solve(&rhs).unwrap();
        let residual = m * x - rhs;
        let scale = x.magnitude().max(rhs.magnitude()).max(1.0);
        prop_assert!(residual.magnitude() <= 1e-9 * scale,
                     "|Ax - b| = {}", residual.magnitude());
    }

    // ------------------------------------------------------------------
    // Dual-number derivatives vs central finite differences.
    // ------------------------------------------------------------------

    #[test]
    fn dual_derivative_matches_finite_differences(c in prop::array::uniform6(-5.0..5.0f64),
                                                  x in -2.0..2.0f64) {
        let p = Polynomial::new(c);
        let ad = p.eval_at(Dual::variable(x)).der;

        let h = 1e-6;
        let fd = (p.eval(x + h) - p.eval(x - h)) / (2.0 * h);

        // Derivative magnitude scale: Σ |i·cᵢ|·max(1,|x|)^(i-1).
        let m = x.abs().max(1.0);
        let scale: f64 = c
            .iter()
            .enumerate()
            .map(|(i, ci)| (i as f64 * ci).abs() * m.powi(i as i32 - 1))
            .sum::<f64>()
            .max(1.0);
        prop_assert!((ad - fd).abs() <= 1e-4 * scale, "AD {} vs FD {}", ad, fd);
    }

    // ------------------------------------------------------------------
    // Smith normal form invariants on random small integer matrices.
    // ------------------------------------------------------------------

    #[test]
    fn snf_random_3x3(entries in prop::array::uniform9(-9i64..=9)) {
        let a = Matrix::<i64, 3, 3>::new([
            [entries[0], entries[1], entries[2]],
            [entries[3], entries[4], entries[5]],
            [entries[6], entries[7], entries[8]],
        ]);
        check_snf_invariants_3x3(&a)?;
    }
}

/// `Σ |cᵢ|·max(1,|r|)^i`: a magnitude scale for the polynomial near `r`, so
/// residual tolerances are relative, not absolute.
fn poly_scale(c: &[f64], r: f64) -> f64 {
    let m = r.abs().max(1.0);
    c.iter()
        .enumerate()
        .map(|(i, ci)| ci.abs() * m.powi(i as i32))
        .sum::<f64>()
        .max(1.0)
}

fn to_talrost_4(a: &[f64; 16]) -> Matrix<f64, 4, 4> {
    let mut e = [[0.0; 4]; 4];
    for i in 0..4 {
        e[i].copy_from_slice(&a[4 * i..4 * i + 4]);
    }
    Matrix::new(e)
}

fn to_nalgebra_4(a: &[f64; 16]) -> Matrix4<f64> {
    Matrix4::from_row_slice(a)
}

fn to_talrost_5(a: &[f64; 25]) -> Matrix<f64, 5, 5> {
    let mut e = [[0.0; 5]; 5];
    for i in 0..5 {
        e[i].copy_from_slice(&a[5 * i..5 * i + 5]);
    }
    Matrix::new(e)
}

fn to_nalgebra_5(a: &[f64; 25]) -> Matrix5<f64> {
    SMatrix::<f64, 5, 5>::from_row_slice(a)
}

/// Exact 3×3 integer determinant in i128 (no rounding, no overflow at these
/// magnitudes).
fn det3_i128(m: &Matrix<i64, 3, 3>) -> i128 {
    let e = |i: usize, j: usize| m.e[i][j] as i128;
    e(0, 0) * (e(1, 1) * e(2, 2) - e(1, 2) * e(2, 1))
        - e(0, 1) * (e(1, 0) * e(2, 2) - e(1, 2) * e(2, 0))
        + e(0, 2) * (e(1, 0) * e(2, 1) - e(1, 1) * e(2, 0))
}

fn check_snf_invariants_3x3(a: &Matrix<i64, 3, 3>) -> Result<(), TestCaseError> {
    let (u, s, v) = smith_normal_form(a);

    // U·A·V == S, exactly, over the integers.
    prop_assert_eq!(u * *a * v, s);

    // U and V are unimodular.
    prop_assert_eq!(det3_i128(&u).abs(), 1);
    prop_assert_eq!(det3_i128(&v).abs(), 1);

    // S is diagonal with non-negative entries in a divisibility chain.
    for i in 0..3 {
        for j in 0..3 {
            if i != j {
                prop_assert_eq!(s.e[i][j], 0);
            }
        }
    }
    for t in 0..3 {
        prop_assert!(s.e[t][t] >= 0);
        if t + 1 < 3 && s.e[t + 1][t + 1] != 0 {
            prop_assert!(s.e[t][t] != 0 && s.e[t + 1][t + 1] % s.e[t][t] == 0);
        }
    }
    Ok(())
}
