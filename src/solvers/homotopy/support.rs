//! Supports of sparse polynomial systems and the generic liftings that
//! induce their mixed subdivisions.

use crate::algebra::Ring;
use crate::mvpoly::{MPoly, MSystem, Monomial};
use crate::real::Real;

/// The support of a sparse polynomial: a finite **set** of exponent vectors
/// `A ⊂ ℤ^NV`.
///
/// Duplicates are dropped on construction (first appearance wins), so
/// [`Support::points`] is a genuine set; the surviving order is preserved,
/// which keeps downstream cell enumeration deterministic.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Support<const NV: usize> {
    points: Vec<Monomial<NV>>,
}

impl<const NV: usize> Support<NV> {
    /// Builds a support from a list of exponent vectors, dropping duplicates
    /// (first appearance wins).
    pub fn new(points: &[Monomial<NV>]) -> Self {
        let mut uniq: Vec<Monomial<NV>> = Vec::with_capacity(points.len());
        for p in points {
            if !uniq.contains(p) {
                uniq.push(*p);
            }
        }
        Self { points: uniq }
    }

    /// Extracts the support of an existing [`MPoly`].
    ///
    /// Padding rule: [`MSystem`] rows are padded to a common term count with
    /// **zero-coefficient** terms whose exponents are arbitrary garbage, so
    /// every term whose coefficient equals `T::ZERO` is skipped — the same
    /// convention `MPoly`'s `Display` impl uses to hide padding. Duplicate
    /// monomials among the surviving terms are collapsed.
    pub fn from_mpoly<T, const TERMS: usize>(poly: &MPoly<T, NV, TERMS>) -> Self
    where
        T: Ring + PartialEq,
    {
        let mut uniq: Vec<Monomial<NV>> = Vec::with_capacity(TERMS);
        for (c, m) in poly.coeffs.iter().zip(poly.support.iter()) {
            if *c != T::ZERO && !uniq.contains(m) {
                uniq.push(*m);
            }
        }
        Self { points: uniq }
    }

    /// Extracts the support of every equation of an [`MSystem`], in
    /// equation order (padding skipped as in [`Support::from_mpoly`]).
    pub fn from_msystem<T, const NEQ: usize, const MAXT: usize>(
        system: &MSystem<T, NV, NEQ, MAXT>,
    ) -> [Self; NEQ]
    where
        T: Ring + PartialEq,
    {
        core::array::from_fn(|i| Self::from_mpoly(&system.polys[i]))
    }

    /// The exponent vectors: deduplicated, in first-appearance order.
    pub fn points(&self) -> &[Monomial<NV>] {
        &self.points
    }

    /// The number of distinct exponent vectors.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// `true` if the support has no points.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

/// A deterministic 64-bit linear congruential generator with Knuth's MMIX
/// constants: `state ← 6364136223846793005·state + 1442695040888963407
/// (mod 2^64)`. Each draw advances the state once and maps its **top 24
/// bits** — the highest-quality bits of an LCG — to a multiple of `2⁻²⁴` in
/// `[0, 1)`, a mapping that is exact in `f32` and `f64` alike.
struct Lcg {
    state: u64,
}

impl Lcg {
    const MUL: u64 = 6364136223846793005;
    const INC: u64 = 1442695040888963407;

    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_unit<F: Real>(&mut self) -> F {
        self.state = self.state.wrapping_mul(Self::MUL).wrapping_add(Self::INC);
        let bits = (self.state >> 40) as u32; // top 24 bits
        F::from_u32(bits) / F::from_u32(1 << 24)
    }
}

/// Per-point lift values `ω : A → ℝ` for one support.
///
/// The lifting induces the mixed subdivision: point `a` is lifted to
/// `(a, ω(a))` and the lower hull of the Minkowski sum of the lifted
/// supports projects back down onto the cells. **Reproducibility matters**:
/// the same seed always produces the same values, hence the same subdivision
/// and the same mixed cells.
#[derive(Clone, Debug, PartialEq)]
pub struct Lifting<F> {
    values: Vec<F>,
}

impl<F: Real> Lifting<F> {
    /// Wraps explicit lift values (one per support point, in the support's
    /// point order).
    pub fn from_values(values: &[F]) -> Self {
        Self {
            values: values.to_vec(),
        }
    }

    /// `len` pseudo-random lift values in `[0, 1)`, deterministically
    /// derived from `seed`: same `(len, seed)` ⇒ same values.
    ///
    /// The generator is a 64-bit LCG with Knuth's MMIX constants
    /// (multiplier `6364136223846793005`, increment `1442695040888963407`);
    /// each value is the state's top 24 bits scaled to a multiple of `2⁻²⁴`
    /// in `[0, 1)` — exact in both `f32` and `f64`.
    pub fn random(len: usize, seed: u64) -> Self {
        let mut lcg = Lcg::new(seed);
        Self {
            values: (0..len).map(|_| lcg.next_unit()).collect(),
        }
    }

    /// The lift values, in the support's point order.
    pub fn values(&self) -> &[F] {
        &self.values
    }

    /// The number of lift values.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// `true` if there are no values.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// One lifting per support, all drawn from a single LCG stream seeded with
/// `seed` (the generator documented at [`Lifting::random`]): support `i`
/// consumes the next `supports[i].len()` values. Same seed ⇒ identical
/// liftings ⇒ identical subdivision and cells; different seeds give
/// (generically) different subdivisions whose cell volumes still sum to the
/// same mixed volume.
pub fn random_liftings<F: Real, const NV: usize>(
    supports: &[Support<NV>; NV],
    seed: u64,
) -> [Lifting<F>; NV] {
    let mut lcg = Lcg::new(seed);
    core::array::from_fn(|i| Lifting {
        values: (0..supports[i].len()).map(|_| lcg.next_unit()).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::complex::c64;

    #[test]
    fn support_dedupes_preserving_order() {
        let pts = [
            Monomial::new([1, 0]),
            Monomial::new([0, 1]),
            Monomial::new([1, 0]), // duplicate
            Monomial::new([0, 0]),
        ];
        let s = Support::new(&pts);
        assert_eq!(s.len(), 3);
        assert!(!s.is_empty());
        assert_eq!(
            s.points(),
            &[
                Monomial::new([1, 0]),
                Monomial::new([0, 1]),
                Monomial::new([0, 0])
            ]
        );

        assert!(Support::<2>::new(&[]).is_empty());
    }

    #[test]
    fn support_from_mpoly_skips_padding() {
        // Zero-coefficient terms are padding (their exponents are garbage)
        // and must not leak into the support.
        let p = MPoly::<f64, 2, 4>::new(
            [1.0, 0.0, 2.0, 0.0],
            [
                Monomial::new([1, 0]),
                Monomial::new([9, 9]), // padding: garbage exponents
                Monomial::new([0, 1]),
                Monomial::new([0, 0]), // padding
            ],
        );
        let s = Support::from_mpoly(&p);
        assert_eq!(s.points(), &[Monomial::new([1, 0]), Monomial::new([0, 1])]);
    }

    #[test]
    fn support_from_msystem_per_equation() {
        let c = c64::new;
        let f1 = MPoly::new(
            [c(1.0, 0.0), c(2.0, 0.0), c(0.0, 0.0)],
            [
                Monomial::new([1, 0]),
                Monomial::new([0, 0]),
                Monomial::new([7, 7]), // padding
            ],
        );
        let f2 = MPoly::new(
            [c(0.0, 1.0), c(-1.0, 0.0), c(3.0, 0.0)],
            [
                Monomial::new([0, 1]),
                Monomial::new([1, 1]),
                Monomial::new([0, 0]),
            ],
        );
        let sys = MSystem::new([f1, f2]);
        let sup = Support::from_msystem(&sys);
        assert_eq!(
            sup[0].points(),
            &[Monomial::new([1, 0]), Monomial::new([0, 0])]
        );
        assert_eq!(sup[1].len(), 3);
    }

    #[test]
    fn lifting_random_is_deterministic_and_in_unit_interval() {
        let a = Lifting::<f64>::random(16, 12345);
        let b = Lifting::<f64>::random(16, 12345);
        let c = Lifting::<f64>::random(16, 54321);
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(a.len(), 16);
        assert!(a.values().iter().all(|&v| (0.0..1.0).contains(&v)));

        // The values are multiples of 2^-24 (exact in f32 too), and not all
        // equal.
        for &v in a.values() {
            assert_eq!(v * (1u32 << 24) as f64, (v * (1u32 << 24) as f64).floor());
        }
        assert!(a.values().iter().any(|&v| v != a.values()[0]));

        let explicit = Lifting::from_values(&[0.25, 0.5]);
        assert_eq!(explicit.values(), &[0.25, 0.5]);
        assert!(Lifting::<f64>::from_values(&[]).is_empty());
    }

    #[test]
    fn random_liftings_match_supports_and_seed() {
        let s1 = Support::<2>::new(&[
            Monomial::new([0, 0]),
            Monomial::new([1, 0]),
            Monomial::new([0, 1]),
        ]);
        let s2 = Support::<2>::new(&[Monomial::new([0, 0]), Monomial::new([1, 1])]);
        let supports = [s1, s2];

        let l = random_liftings::<f64, 2>(&supports, 7);
        assert_eq!(l[0].len(), 3);
        assert_eq!(l[1].len(), 2);

        // One continuous stream: reproducible per seed.
        let l2 = random_liftings::<f64, 2>(&supports, 7);
        assert_eq!(l, l2);
        let l3 = random_liftings::<f64, 2>(&supports, 8);
        assert_ne!(l, l3);
    }
}
