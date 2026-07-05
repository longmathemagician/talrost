//! Fine mixed-cell enumeration and exact mixed volumes.
//!
//! For supports `A_1..A_n ⊂ ℤⁿ` lifted by `ω_1..ω_n`, a **fine mixed cell**
//! is a tuple of edges `(a_i, b_i) ⊆ A_i` — one per support — admitting an
//! `α ∈ ℝⁿ` with, for every `i`:
//!
//! - `⟨b_i − a_i, α⟩ = ω_i(a_i) − ω_i(b_i)` (the two lifted edge points are
//!   level), and
//! - `⟨a_i, α⟩ + ω_i(a_i) < ⟨c, α⟩ + ω_i(c)` for every other `c ∈ A_i`
//!   (the level pair is strictly minimal in its lifted support).
//!
//! Geometrically `α` is the horizontal part of an inner normal to a lower
//! facet of the lifted Minkowski sum; the cells tile `A_1 + … + A_n` and
//! their normalized volumes `|det V|` sum to the mixed volume — Bernstein's
//! generic root count on `(ℂ*)ⁿ`.

use crate::lattice::smith_normal_form;
use crate::matrix::Matrix;
use crate::mvpoly::Monomial;
use crate::real::Real;
use crate::vector::Vector;

use super::support::{random_liftings, Lifting, Support};

/// The lifting failed its genericity requirement: while testing a candidate
/// cell, some lifted point landed inside the strict-minimality tolerance
/// band (see [`mixed_cells`]), so the induced subdivision cannot be trusted
/// to be fine. Nothing is guessed: re-lift with a different seed and retry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GenericityError {
    /// The index of the support whose strict-minimality inequality was
    /// ambiguous.
    pub support: usize,
}

impl core::fmt::Display for GenericityError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "lifting is not generic (near-tie in support {}): re-lift with a new seed",
            self.support
        )
    }
}

impl core::error::Error for GenericityError {}

/// Relative genericity tolerance, `1e-9`.
///
/// A strict-minimality margin `m = (⟨c, α⟩ + ω(c)) − (⟨a, α⟩ + ω(a))` is
/// compared against the band `1e-9 · max(1, |level_a|, |level_c|)`: margins
/// above the band pass, margins below its negation reject the candidate
/// tuple, and margins inside the band are ambiguous. `f64` is the intended
/// lifting scalar; the tolerance is below `f32::EPSILON`, so `f32` liftings
/// get correspondingly weaker tie detection.
fn tolerance<F: Real>() -> F {
    F::ONE / F::from_u32(1_000_000_000)
}

/// Exact conversion of a small integer (an exponent or exponent difference)
/// to `F`. Exponents beyond ±2²⁴ would round in `f32` and are rejected by
/// `debug_assert` — no sparse system in this crate's scope gets anywhere
/// near that.
fn small_int<F: Real>(k: i64) -> F {
    debug_assert!(
        k.unsigned_abs() <= 1 << 24,
        "exponent too large for exact float conversion"
    );
    let mag = F::from_u32(k.unsigned_abs() as u32);
    if k < 0 {
        -mag
    } else {
        mag
    }
}

/// One fine mixed cell of the lifted subdivision: an edge `(a_i, b_i)` per
/// support, the lower-facet normal `α` certifying it, and the integer edge
/// matrix `V` whose binomial system `x^V = β` the cell's start solutions
/// satisfy (see [`super::start`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MixedCell<F: Real, const NV: usize> {
    /// The cell's edges: `edges[i] = (a_i, b_i)`, both points of support
    /// `i`, oriented by the support's point order (`a_i` first).
    pub edges: [(Monomial<NV>, Monomial<NV>); NV],
    /// The cell normal `α`: the vector making every lifted edge pair level
    /// and strictly below the rest of its lifted support.
    pub normal: [F; NV],
    /// The edge matrix `V`: row `i` is `b_i − a_i`.
    pub edge_matrix: Matrix<i64, NV, NV>,
}

impl<F: Real, const NV: usize> MixedCell<F, NV> {
    /// The cell's normalized volume `|det V|`, computed **exactly** as the
    /// product of the Smith-normal-form diagonal of `V`
    /// ([`crate::lattice::smith_normal_form`]; the diagonal is non-negative
    /// by construction) — integer arithmetic throughout, no `f64`
    /// determinant rounding. This is also the number of start solutions the
    /// cell contributes.
    pub fn volume(&self) -> u64 {
        let (_, s, _) = smith_normal_form(&self.edge_matrix);
        (0..NV).map(|i| s.e[i][i] as u64).product()
    }
}

/// Enumerates the fine mixed cells of the subdivision that `liftings`
/// induces on `supports`.
///
/// # Algorithm and complexity
///
/// Naive enumeration — correct first, fast later: every tuple of candidate
/// edges (one unordered point pair per support) is tested by assembling the
/// `n×n` level system with rows `b_i − a_i` and right-hand side
/// `ω_i(a_i) − ω_i(b_i)`, solving for `α` with [`Matrix::solve`], and
/// verifying the strict-minimality inequalities. That is
/// `Π_i C(|A_i|, 2) ≤ Π_i |A_i|²` LU solves — fine for the small fixed
/// systems this crate targets, hopeless for large polytopes.
///
/// Singular tuples are skipped, and their singularity is decided **exactly
/// over ℤ** (Smith normal form of the integer edge-difference matrix) before
/// the floating-point solve: f64 LU rounds its rational elimination
/// multipliers, so an integer-singular tuple can factor with a ~1e-16 pivot
/// instead of an exact zero and produce a garbage normal that falsely trips
/// the genericity check (see the inline comment; katsura-3 exhibits this on
/// every lifting).
///
/// # Genericity
///
/// A generic lifting makes ties measure-zero, but they are checked, not
/// assumed: if a strict-minimality margin lands inside the relative
/// tolerance band (`1e-9`, see the module source) **and** the candidate
/// tuple is not already decisively rejected by another inequality, the
/// enumeration aborts with [`GenericityError`] instead of silently guessing
/// — re-lift with a different seed. (A near-tie on a tuple that some other
/// point decisively rejects is harmless — no resolution of the tie could
/// make that tuple a cell — so it does not abort.)
///
/// # Panics
///
/// Panics if some `liftings[i]` does not have exactly one value per point
/// of `supports[i]`.
pub fn mixed_cells<F: Real, const NV: usize>(
    supports: &[Support<NV>; NV],
    liftings: &[Lifting<F>; NV],
) -> Result<Vec<MixedCell<F, NV>>, GenericityError> {
    for (i, (sup, lift)) in supports.iter().zip(liftings.iter()).enumerate() {
        assert!(
            sup.len() == lift.len(),
            "lifting {} has {} values for {} support points",
            i,
            lift.len(),
            sup.len()
        );
    }

    let mut cells = Vec::new();
    if NV == 0 {
        return Ok(cells);
    }

    // Candidate edges per support: all unordered point pairs, as index
    // pairs (j, k) with j < k into the support's point list.
    let edge_lists: [Vec<(usize, usize)>; NV] = core::array::from_fn(|i| {
        let n = supports[i].len();
        let mut edges = Vec::with_capacity(n * n.saturating_sub(1) / 2);
        for j in 0..n {
            for k in (j + 1)..n {
                edges.push((j, k));
            }
        }
        edges
    });
    if edge_lists.iter().any(Vec::is_empty) {
        return Ok(cells); // a support with < 2 points admits no edge: no cells
    }

    let tol = tolerance::<F>();
    let mut idx = [0usize; NV]; // odometer over edge tuples
    'tuples: loop {
        // Assemble the level system: row i = b_i − a_i, rhs_i = ω(a_i) − ω(b_i)
        // — both as exact integers (the candidate edge matrix `V`) and as `F`.
        let mut m = Matrix::<F, NV, NV>::ZERO;
        let mut v = Matrix::<i64, NV, NV>::ZERO;
        let mut rhs = Vector::<F, NV>::ZERO;
        for i in 0..NV {
            let (ja, jb) = edge_lists[i][idx[i]];
            let a = &supports[i].points()[ja];
            let b = &supports[i].points()[jb];
            for (((mv, vv), &ea), &eb) in m.e[i]
                .iter_mut()
                .zip(v.e[i].iter_mut())
                .zip(a.exps.iter())
                .zip(b.exps.iter())
            {
                *vv = eb as i64 - ea as i64;
                *mv = small_int(*vv);
            }
            rhs.b[i] = liftings[i].values()[ja] - liftings[i].values()[jb];
        }

        // Exact singularity gate. An **exactly** singular integer tuple must
        // be decided in integer arithmetic: f64 LU elimination rounds its
        // rational multipliers, so an integer-singular matrix can come out
        // of the factorization with a ~1e-16 pivot instead of an exact zero,
        // "solve" to a garbage normal of magnitude ~1e16, and then trip the
        // relative-tolerance genericity check — a false `GenericityError` on
        // *every* seed (found by the Phase 10 benchmark suite: katsura-3's
        // doubled support points produce exactly such tuples, e.g. the edge
        // pair (2e₃, 2e₂) combined with three edges whose differences span
        // the same rank-3 sublattice). Singularity over ℤ is decided exactly
        // by the Smith normal form: any zero diagonal entry means the tuple
        // admits no level normal (or no unique one) and is skipped — the
        // same treatment `Matrix::solve` gives an exact zero pivot.
        let (_, s, _) = smith_normal_form(&v);
        let singular = (0..NV).any(|i| s.e[i][i] == 0);

        if let Some(alpha) = (!singular).then(|| m.solve(&rhs)).flatten() {
            // Verify strict minimality of every edge pair within its own
            // lifted support.
            let mut ambiguous: Option<usize> = None;
            let mut rejected = false;
            'verify: for i in 0..NV {
                let (ja, jb) = edge_lists[i][idx[i]];
                let pts = supports[i].points();
                let lifts = liftings[i].values();
                let level = |p: &Monomial<NV>, lift: F| -> F {
                    let mut acc = lift;
                    for (&e, &al) in p.exps.iter().zip(alpha.b.iter()) {
                        acc += small_int::<F>(e as i64) * al;
                    }
                    acc
                };
                let base = level(&pts[ja], lifts[ja]);
                for (c, (p, &lift)) in pts.iter().zip(lifts.iter()).enumerate() {
                    if c == ja || c == jb {
                        continue;
                    }
                    let val = level(p, lift);
                    let margin = val - base;
                    let band = tol * F::ONE.max(base.abs()).max(val.abs());
                    if margin < -band {
                        rejected = true; // decisively not a lower edge
                        break 'verify;
                    } else if margin <= band && ambiguous.is_none() {
                        // Near-tie: fatal only if the tuple survives every
                        // other inequality.
                        ambiguous = Some(i);
                    }
                }
            }
            if !rejected {
                if let Some(support) = ambiguous {
                    return Err(GenericityError { support });
                }
                // Accept: record the edges, the normal, and the integer edge
                // matrix `v` assembled above.
                let mut edges = [(Monomial::new([0; NV]), Monomial::new([0; NV])); NV];
                for (i, edge) in edges.iter_mut().enumerate() {
                    let (ja, jb) = edge_lists[i][idx[i]];
                    *edge = (supports[i].points()[ja], supports[i].points()[jb]);
                }
                cells.push(MixedCell {
                    edges,
                    normal: alpha.b,
                    edge_matrix: v,
                });
            }
        }

        // Advance the odometer (first support fastest).
        let mut d = 0;
        loop {
            idx[d] += 1;
            if idx[d] < edge_lists[d].len() {
                break;
            }
            idx[d] = 0;
            d += 1;
            if d == NV {
                break 'tuples;
            }
        }
    }

    Ok(cells)
}

/// The mixed volume of `supports`: lift with `seed` ([`random_liftings`]),
/// enumerate the fine mixed cells, and sum their exact volumes
/// ([`MixedCell::volume`]). By Bernstein's theorem this is the generic root
/// count on the torus `(ℂ*)ⁿ`. Fails with [`GenericityError`] when the
/// seed's lifting is degenerate — retry with another seed.
pub fn mixed_volume<F: Real, const NV: usize>(
    supports: &[Support<NV>; NV],
    seed: u64,
) -> Result<u64, GenericityError> {
    let liftings = random_liftings::<F, NV>(supports, seed);
    Ok(mixed_cells(supports, &liftings)?
        .iter()
        .map(MixedCell::volume)
        .sum())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simplex2() -> Support<2> {
        Support::new(&[
            Monomial::new([0, 0]),
            Monomial::new([1, 0]),
            Monomial::new([0, 1]),
        ])
    }

    /// All six exponent vectors with |e| ≤ 2: a dense conic's support.
    fn conic() -> Support<2> {
        Support::new(&[
            Monomial::new([0, 0]),
            Monomial::new([1, 0]),
            Monomial::new([0, 1]),
            Monomial::new([2, 0]),
            Monomial::new([1, 1]),
            Monomial::new([0, 2]),
        ])
    }

    /// The sparse pair with mixed volume 2 (< Bézout 4).
    fn sparse_pair() -> [Support<2>; 2] {
        [
            Support::new(&[
                Monomial::new([0, 0]),
                Monomial::new([1, 0]),
                Monomial::new([1, 1]),
            ]),
            Support::new(&[
                Monomial::new([0, 0]),
                Monomial::new([0, 1]),
                Monomial::new([1, 1]),
            ]),
        ]
    }

    /// Checks the defining property of each cell directly against the
    /// lifting: edge level, everything else strictly above.
    fn check_cells_valid(
        supports: &[Support<2>; 2],
        liftings: &[Lifting<f64>; 2],
        cells: &[MixedCell<f64, 2>],
    ) {
        for cell in cells {
            for i in 0..2 {
                let (a, b) = cell.edges[i];
                let lift_of = |m: &Monomial<2>| -> f64 {
                    let j = supports[i].points().iter().position(|p| p == m).unwrap();
                    liftings[i].values()[j]
                };
                let level = |m: &Monomial<2>| -> f64 {
                    m.exps[0] as f64 * cell.normal[0]
                        + m.exps[1] as f64 * cell.normal[1]
                        + lift_of(m)
                };
                assert!((level(&a) - level(&b)).abs() < 1e-9, "edge not level");
                for p in supports[i].points() {
                    if *p != a && *p != b {
                        assert!(level(p) > level(&a) + 1e-9, "edge not strictly minimal");
                    }
                }
                // Edge matrix rows really are b − a.
                for j in 0..2 {
                    assert_eq!(cell.edge_matrix.e[i][j], (b.exps[j] - a.exps[j]) as i64);
                }
            }
        }
    }

    #[test]
    fn two_unit_simplices_mixed_volume_one() {
        // Two generic linear forms: exactly one cell, mixed volume 1.
        let supports = [simplex2(), simplex2()];
        let liftings = random_liftings::<f64, 2>(&supports, 1);
        let cells = mixed_cells(&supports, &liftings).unwrap();
        assert_eq!(cells.len(), 1);
        assert_eq!(cells[0].volume(), 1);
        assert_eq!(mixed_volume::<f64, 2>(&supports, 1).unwrap(), 1);
        check_cells_valid(&supports, &liftings, &cells);
    }

    #[test]
    fn dense_conics_mixed_volume_four() {
        // Two dense degree-2 curves: Bézout = Bernstein = 4.
        let supports = [conic(), conic()];
        assert_eq!(mixed_volume::<f64, 2>(&supports, 2).unwrap(), 4);
        // The cells are genuine cells of the induced subdivision.
        let liftings = random_liftings::<f64, 2>(&supports, 2);
        let cells = mixed_cells(&supports, &liftings).unwrap();
        assert_eq!(cells.iter().map(MixedCell::volume).sum::<u64>(), 4);
        check_cells_valid(&supports, &liftings, &cells);
    }

    #[test]
    fn sparse_pair_beats_bezout() {
        // Eliminating one variable from the trinomial pair leaves a
        // quadratic: 2 torus roots, half the Bézout bound of 4.
        let supports = sparse_pair();
        assert_eq!(mixed_volume::<f64, 2>(&supports, 3).unwrap(), 2);
    }

    #[test]
    fn three_linear_supports_mixed_volume_one() {
        // {0, e1, e2, e3} three times: one generic 3×3 linear system, MV 1.
        let simplex3 = Support::<3>::new(&[
            Monomial::new([0, 0, 0]),
            Monomial::new([1, 0, 0]),
            Monomial::new([0, 1, 0]),
            Monomial::new([0, 0, 1]),
        ]);
        let supports = [simplex3.clone(), simplex3.clone(), simplex3];
        let liftings = random_liftings::<f64, 3>(&supports, 4);
        let cells = mixed_cells(&supports, &liftings).unwrap();
        assert_eq!(cells.len(), 1);
        assert_eq!(cells[0].volume(), 1);
        assert_eq!(mixed_volume::<f64, 3>(&supports, 4).unwrap(), 1);
    }

    #[test]
    fn same_seed_same_cells_different_seed_same_volume() {
        let supports = sparse_pair();

        // Same seed ⇒ bit-identical cells.
        let a = mixed_cells(&supports, &random_liftings::<f64, 2>(&supports, 11)).unwrap();
        let b = mixed_cells(&supports, &random_liftings::<f64, 2>(&supports, 11)).unwrap();
        assert_eq!(a, b);

        // Different seeds ⇒ possibly different subdivisions, same sum.
        for seed in [5, 17, 99, 12345, 987654321] {
            assert_eq!(mixed_volume::<f64, 2>(&supports, seed).unwrap(), 2);
        }
        // And for the conics: the volume is an invariant of the supports.
        let conics = [conic(), conic()];
        for seed in [6, 28, 496] {
            assert_eq!(mixed_volume::<f64, 2>(&conics, seed).unwrap(), 4);
        }
    }

    #[test]
    fn degenerate_lifting_is_a_first_class_error() {
        // The all-zero lifting puts every point on one plane: the very
        // first solvable tuple has its remaining point exactly level, which
        // must surface as a GenericityError, not as a guessed subdivision.
        let supports = [simplex2(), simplex2()];
        let liftings = [
            Lifting::from_values(&[0.0, 0.0, 0.0]),
            Lifting::from_values(&[0.0, 0.0, 0.0]),
        ];
        let err = mixed_cells(&supports, &liftings).unwrap_err();
        assert_eq!(err, GenericityError { support: 0 });
        assert!(err.to_string().contains("new seed"));
    }

    #[test]
    fn harmless_tie_on_a_rejected_tuple_is_not_an_error() {
        // One-variable support {0, 3, 1, 2} lifted to (0, 0, −10, 0): the
        // lower hull is (0,0)–(1,−10)–(3,0), giving the two genuine cells
        // {0,1} (length 1) and {1,3} (length 2). The candidate edge {0,2}
        // sees exponent 3 *exactly level* with it (a tie inside the band)
        // but exponent 1 decisively below — the tuple is rejected outright,
        // so the tie is harmless and must NOT abort enumeration.
        let support = Support::<1>::new(&[
            Monomial::new([0]),
            Monomial::new([3]),
            Monomial::new([1]),
            Monomial::new([2]),
        ]);
        let lifting = Lifting::from_values(&[0.0, 0.0, -10.0, 0.0]);
        let cells = mixed_cells(&[support], &[lifting]).unwrap();
        assert_eq!(cells.len(), 2);
        let vols: Vec<u64> = cells.iter().map(MixedCell::volume).collect();
        let mut sorted = vols.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![1, 2]); // the interval [0, 3] has length 3
        for cell in &cells {
            let (a, b) = cell.edges[0];
            assert!(
                (a.exps[0], b.exps[0]) == (0, 1) || (a.exps[0], b.exps[0]) == (3, 1),
                "unexpected cell edge {:?}",
                cell.edges[0]
            );
        }
    }

    /// Regression (Phase 10): katsura-3's supports contain doubled simplex
    /// points (2e_i), which admit candidate edge tuples whose integer level
    /// matrix is **exactly** singular — e.g. support-0 edge (2e₃, 2e₂) with
    /// three more edges spanning the same rank-3 sublattice. Before the
    /// exact SNF singularity gate, f64 LU "solved" those tuples to a garbage
    /// normal (~1e16) whose relative level ties falsely tripped the
    /// genericity check, so *every* seed returned `GenericityError`. The
    /// enumeration must succeed and the mixed volume (a lifting invariant)
    /// must be identical across seeds.
    #[test]
    fn katsura_3_singular_tuples_are_skipped_exactly() {
        // Supports of katsura-3 in the (n+1)-unknown convention (see
        // examples/bench_suite.rs): u0²+2u1²+2u2²+2u3²−u0,
        // 2u2u3+2u1u2+2u0u1−u1, 2u1u3+2u0u2+u1²−u2, u0+2u1+2u2+2u3−1.
        let supports: [Support<4>; 4] = [
            Support::new(&[
                Monomial::new([0, 0, 0, 2]),
                Monomial::new([0, 0, 2, 0]),
                Monomial::new([0, 2, 0, 0]),
                Monomial::new([2, 0, 0, 0]),
                Monomial::new([1, 0, 0, 0]),
            ]),
            Support::new(&[
                Monomial::new([0, 0, 1, 1]),
                Monomial::new([0, 1, 1, 0]),
                Monomial::new([1, 1, 0, 0]),
                Monomial::new([0, 1, 0, 0]),
            ]),
            Support::new(&[
                Monomial::new([0, 1, 0, 1]),
                Monomial::new([1, 0, 1, 0]),
                Monomial::new([0, 2, 0, 0]),
                Monomial::new([0, 0, 1, 0]),
            ]),
            Support::new(&[
                Monomial::new([1, 0, 0, 0]),
                Monomial::new([0, 1, 0, 0]),
                Monomial::new([0, 0, 1, 0]),
                Monomial::new([0, 0, 0, 1]),
                Monomial::new([0, 0, 0, 0]),
            ]),
        ];
        let mv = mixed_volume::<f64, 4>(&supports, 1).unwrap();
        for seed in [2, 3, 17, 2026] {
            assert_eq!(mixed_volume::<f64, 4>(&supports, seed).unwrap(), mv);
        }
    }

    #[test]
    fn supports_too_small_yield_no_cells() {
        // A single-point support has no edges: no cells, mixed volume 0.
        let supports = [Support::<2>::new(&[Monomial::new([0, 0])]), simplex2()];
        let liftings = random_liftings::<f64, 2>(&supports, 5);
        assert!(mixed_cells(&supports, &liftings).unwrap().is_empty());
        assert_eq!(mixed_volume::<f64, 2>(&supports, 5).unwrap(), 0);
    }

    #[test]
    #[should_panic(expected = "lifting 1 has")]
    fn mismatched_lifting_length_panics() {
        let supports = [simplex2(), simplex2()];
        let liftings = [
            Lifting::from_values(&[0.1, 0.2, 0.3]),
            Lifting::from_values(&[0.1, 0.2]), // one short
        ];
        let _ = mixed_cells(&supports, &liftings);
    }
}
