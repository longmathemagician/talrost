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
//!
//! Two enumerators share one exact per-tuple decision procedure:
//! [`mixed_cells`] is the production LP-pruned depth-first search (the
//! simplified DEMiCs scheme of Mizutani–Takeda–Kojima, see its docs), and
//! [`mixed_cells_naive`] is the exhaustive reference implementation kept as
//! the correctness oracle.

use crate::lattice::smith_normal_form;
use crate::matrix::Matrix;
use crate::mvpoly::Monomial;
use crate::real::Real;
use crate::vector::Vector;

use super::lp::{max_delta, DeltaMax};
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

/// Panics unless every `liftings[i]` has exactly one value per point of
/// `supports[i]` — the shared precondition of both enumerators.
fn assert_lifting_lengths<F: Real, const NV: usize>(
    supports: &[Support<NV>; NV],
    liftings: &[Lifting<F>; NV],
) {
    for (i, (sup, lift)) in supports.iter().zip(liftings.iter()).enumerate() {
        assert!(
            sup.len() == lift.len(),
            "lifting {} has {} values for {} support points",
            i,
            lift.len(),
            sup.len()
        );
    }
}

/// All candidate edges of each support: every unordered point pair, as
/// index pairs `(j, k)` with `j < k` into the support's point list.
fn candidate_edges<const NV: usize>(supports: &[Support<NV>; NV]) -> [Vec<(usize, usize)>; NV] {
    core::array::from_fn(|i| {
        let n = supports[i].len();
        let mut edges = Vec::with_capacity(n * n.saturating_sub(1) / 2);
        for j in 0..n {
            for k in (j + 1)..n {
                edges.push((j, k));
            }
        }
        edges
    })
}

/// Decides one full candidate edge tuple **exactly** — the single code path
/// both enumerators funnel every accepted cell through, so their outputs
/// are bit-for-bit comparable. `pairs[i]` is support `i`'s candidate edge
/// as a point-index pair `(ja, jb)`.
///
/// Steps, in order:
///
/// 1. Assemble the level system: row `i = b_i − a_i`, right-hand side
///    `ω_i(a_i) − ω_i(b_i)` — both as exact integers (the candidate edge
///    matrix `V`) and as `F`.
/// 2. The exact singularity gate. An **exactly** singular integer tuple
///    must be decided in integer arithmetic: f64 LU elimination rounds its
///    rational multipliers, so an integer-singular matrix can come out of
///    the factorization with a ~1e-16 pivot instead of an exact zero,
///    "solve" to a garbage normal of magnitude ~1e16, and then trip the
///    relative-tolerance genericity check — a false [`GenericityError`] on
///    *every* seed (found by the Phase 10 benchmark suite: katsura-3's
///    doubled support points produce exactly such tuples, e.g. the edge
///    pair (2e₃, 2e₂) combined with three edges whose differences span the
///    same rank-3 sublattice). Singularity over ℤ is decided exactly by the
///    Smith normal form: any zero diagonal entry means the tuple admits no
///    level normal (or no unique one) and is skipped — the same treatment
///    [`Matrix::solve`] gives an exact zero pivot.
/// 3. Solve for the normal `α` and verify strict minimality of every edge
///    pair within its own lifted support, with the relative tolerance band
///    of [`tolerance`]. Decisive violation → `Ok(None)`; a near-tie on a
///    tuple no other point rejects → `Err(GenericityError)`; all margins
///    clear → `Ok(Some(cell))`.
fn decide_tuple<F: Real, const NV: usize>(
    supports: &[Support<NV>; NV],
    liftings: &[Lifting<F>; NV],
    pairs: &[(usize, usize); NV],
) -> Result<Option<MixedCell<F, NV>>, GenericityError> {
    let tol = tolerance::<F>();
    let mut m = Matrix::<F, NV, NV>::ZERO;
    let mut v = Matrix::<i64, NV, NV>::ZERO;
    let mut rhs = Vector::<F, NV>::ZERO;
    for i in 0..NV {
        let (ja, jb) = pairs[i];
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

    let (_, s, _) = smith_normal_form(&v);
    let singular = (0..NV).any(|i| s.e[i][i] == 0);

    let Some(alpha) = (!singular).then(|| m.solve(&rhs)).flatten() else {
        return Ok(None);
    };

    // Verify strict minimality of every edge pair within its own lifted
    // support (in original support order, so the GenericityError support
    // index is enumeration-order independent).
    let mut ambiguous: Option<usize> = None;
    for i in 0..NV {
        let (ja, jb) = pairs[i];
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
                return Ok(None); // decisively not a lower edge
            } else if margin <= band && ambiguous.is_none() {
                // Near-tie: fatal only if the tuple survives every other
                // inequality.
                ambiguous = Some(i);
            }
        }
    }
    if let Some(support) = ambiguous {
        return Err(GenericityError { support });
    }

    // Accept: record the edges, the normal, and the integer edge matrix.
    let mut edges = [(Monomial::new([0; NV]), Monomial::new([0; NV])); NV];
    for (i, edge) in edges.iter_mut().enumerate() {
        let (ja, jb) = pairs[i];
        *edge = (supports[i].points()[ja], supports[i].points()[jb]);
    }
    Ok(Some(MixedCell {
        edges,
        normal: alpha.b,
        edge_matrix: v,
    }))
}

/// Enumerates the fine mixed cells by **exhaustive tuple scan** — the
/// reference implementation and correctness oracle for [`mixed_cells`],
/// which must produce the identical cell set.
///
/// # Algorithm and complexity
///
/// Every tuple of candidate edges (one unordered point pair per support) is
/// tested by the exact per-tuple decision procedure (level-system LU solve
/// behind an exact ℤ Smith-normal-form singularity gate, then the
/// strict-minimality inequalities — see the module source). That is
/// `Π_i C(|A_i|, 2) ≤ Π_i |A_i|²` LU solves: fine for small fixed systems,
/// hopeless past cyclic-6 (BENCHMARKS.md records the measured wall). Use
/// [`mixed_cells`] — same signature, same result, LP-pruned search —
/// everywhere except when an independent cross-check is the point.
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
pub fn mixed_cells_naive<F: Real, const NV: usize>(
    supports: &[Support<NV>; NV],
    liftings: &[Lifting<F>; NV],
) -> Result<Vec<MixedCell<F, NV>>, GenericityError> {
    assert_lifting_lengths(supports, liftings);

    let mut cells = Vec::new();
    if NV == 0 {
        return Ok(cells);
    }
    let edge_lists = candidate_edges(supports);
    if edge_lists.iter().any(Vec::is_empty) {
        return Ok(cells); // a support with < 2 points admits no edge: no cells
    }

    let mut idx = [0usize; NV]; // odometer over edge tuples
    'tuples: loop {
        let pairs: [(usize, usize); NV] = core::array::from_fn(|i| edge_lists[i][idx[i]]);
        if let Some(cell) = decide_tuple(supports, liftings, &pairs)? {
            cells.push(cell);
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

/// One support's candidate edge with its precomputed LP constraint rows.
struct EdgeData<F, const NV: usize> {
    /// The edge as a point-index pair `(ja, jb)`, `ja < jb`.
    pair: (usize, usize),
    /// The level equality `⟨b − a, α⟩ = ω(a) − ω(b)`.
    eq: ([F; NV], F),
    /// The strict-minimality inequalities, one per other point `c`:
    /// `⟨c − a, α⟩ − δ ≥ ω(a) − ω(c)`.
    ins: Vec<([F; NV], F)>,
}

impl<F: Real, const NV: usize> EdgeData<F, NV> {
    /// Builds the constraint rows of edge `(ja, jb)` of one support.
    fn new(support: &Support<NV>, lifting: &Lifting<F>, ja: usize, jb: usize) -> Self {
        let pts = support.points();
        let lifts = lifting.values();
        let diff = |from: usize, to: usize| -> [F; NV] {
            let mut row = [F::ZERO; NV];
            for ((r, &ef), &et) in row
                .iter_mut()
                .zip(pts[from].exps.iter())
                .zip(pts[to].exps.iter())
            {
                *r = small_int(et as i64 - ef as i64);
            }
            row
        };
        let mut ins = Vec::with_capacity(pts.len().saturating_sub(2));
        for c in 0..pts.len() {
            if c != ja && c != jb {
                ins.push((diff(ja, c), lifts[ja] - lifts[c]));
            }
        }
        Self {
            pair: (ja, jb),
            eq: (diff(ja, jb), lifts[ja] - lifts[jb]),
            ins,
        }
    }
}

/// A tree node's LP verdict, classified against the genericity band.
enum NodeClass {
    /// Strictly feasible with margin above the band: descend.
    Viable,
    /// Decisively infeasible (margin below the band's negation): prune the
    /// subtree.
    Pruned,
    /// The optimal margin sits inside the band — a near-tie the lifting
    /// cannot be trusted to resolve. Also used for a stalled LP, which must
    /// never be read as a pruning verdict.
    Ambiguous,
}

/// Classifies an LP optimum against the genericity band (see
/// [`tolerance`]; the naive scan's band is *relative* to the level
/// magnitudes, the LP margin is compared against the absolute `1e-9` floor
/// of that band — levels here satisfy `max(1, …) ≥ 1`. Accepted cells are
/// unaffected by the difference: final acceptance always runs the exact
/// relative-band check in `decide_tuple`; the bands only decide where a
/// *degenerate* lifting aborts, and the [`GenericityError`] re-lift
/// contract absorbs that).
fn classify<F: Real>(opt: DeltaMax<F>, band: F) -> NodeClass {
    match opt {
        DeltaMax::Unbounded => NodeClass::Viable,
        DeltaMax::Bounded(d) if d > band => NodeClass::Viable,
        DeltaMax::Bounded(d) if d < -band => NodeClass::Pruned,
        DeltaMax::Bounded(_) | DeltaMax::Stalled => NodeClass::Ambiguous,
        DeltaMax::Infeasible => NodeClass::Pruned,
    }
}

/// The depth-first search state shared down the recursion.
struct Search<'a, F: Real, const NV: usize> {
    supports: &'a [Support<NV>; NV],
    liftings: &'a [Lifting<F>; NV],
    /// Per support (original index), the viable edges with their LP rows.
    viable: &'a [Vec<EdgeData<F, NV>>; NV],
    /// The static support visit order (ascending viable-edge count).
    order: &'a [usize; NV],
    /// The genericity band, [`tolerance`].
    band: F,
    /// Accepted cells, in DFS order.
    cells: Vec<MixedCell<F, NV>>,
}

impl<F: Real, const NV: usize> Search<'_, F, NV> {
    /// Visits every viable edge of the support at `depth` (in `order`),
    /// pruning by the feasibility LP over all constraints chosen so far and
    /// deciding full tuples exactly. `chosen` is indexed by *original*
    /// support index; `eqs`/`ins` accumulate the LP rows of depths
    /// `0..depth` and are restored before returning.
    fn dfs(
        &mut self,
        depth: usize,
        chosen: &mut [(usize, usize); NV],
        eqs: &mut Vec<([F; NV], F)>,
        ins: &mut Vec<([F; NV], F)>,
    ) -> Result<(), GenericityError> {
        let s = self.order[depth];
        for e in 0..self.viable[s].len() {
            chosen[s] = self.viable[s][e].pair;
            if depth == NV - 1 {
                // Full depth: the exact decision path (SNF singularity
                // gate, LU normal solve, relative-band minimality check) —
                // identical to the naive scan's, so accepted cells match it
                // bit for bit.
                if let Some(cell) = decide_tuple(self.supports, self.liftings, chosen)? {
                    self.cells.push(cell);
                }
                continue;
            }

            let ins_mark = ins.len();
            eqs.push(self.viable[s][e].eq);
            ins.extend_from_slice(&self.viable[s][e].ins);
            // Depth 0's node LP is exactly the pre-filter LP already run
            // for this edge — skip it.
            let verdict = if depth == 0 {
                NodeClass::Viable
            } else {
                classify(max_delta::<F, NV>(eqs, ins), self.band)
            };
            let result = match verdict {
                NodeClass::Viable => self.dfs(depth + 1, chosen, eqs, ins),
                NodeClass::Pruned => Ok(()),
                NodeClass::Ambiguous => Err(GenericityError { support: s }),
            };
            eqs.pop();
            ins.truncate(ins_mark);
            result?;
        }
        Ok(())
    }
}

/// Enumerates the fine mixed cells of the subdivision that `liftings`
/// induces on `supports` — the production enumerator, used by
/// [`mixed_volume`] and the [`super::driver::solve`] driver.
///
/// # Algorithm
///
/// An LP-pruned depth-first search over one-edge-per-support choices — a
/// simplified variant of the **DEMiCs** dynamic enumeration of Mizutani,
/// Takeda & Kojima (*Dynamic enumeration of all mixed cells*, Discrete
/// Comput. Geom. 37, 2007):
///
/// 1. **Pre-filter**: an edge `(a, b)` of `A_i` is individually viable iff
///    the feasibility LP with its own level equality and `A_i`'s own
///    strict-minimality inequalities is strictly feasible (the private
///    `lp` module: maximize the uniform margin `δ`); decisively infeasible
///    edges are discarded up front.
/// 2. **Static ordering**: supports are visited in ascending viable-edge
///    count (ties by index), so the narrowest choices constrain the search
///    earliest. (DEMiCs re-orders *dynamically* per subtree and memoizes
///    one-point relation tables; both are possible refinements here.)
/// 3. **Depth-first search**: at depth `k`, each viable edge of support
///    `k` is tested by the feasibility LP over **all** level equalities of
///    the edges chosen at depths `1..k` plus those same supports'
///    strict-minimality inequalities. Infeasible ⇒ the whole subtree is
///    pruned; a margin inside the genericity band ⇒ [`GenericityError`]
///    (same re-lift contract as the naive scan); at full depth the exact
///    decision procedure of [`mixed_cells_naive`] accepts the cell (Smith
///    normal form singularity gate, LU solve for the normal `α`,
///    relative-band strict-minimality check), so accepted cells are
///    **identical** to the naive enumerator's, normals bit for bit.
///
/// # Complexity
///
/// Worst case still exponential (it is an enumeration of a possibly
/// exponential cell set), but the LP pruning collapses the practical cost:
/// each pruned interior node removes an entire `Π C(|A_j|, 2)` subproduct
/// of candidate tuples, which is the DEMiCs idea. BENCHMARKS.md records
/// measured naive-vs-LP times; cyclic-7 drops from a ~294 s projection to
/// well under a second.
///
/// # Genericity
///
/// Same contract as [`mixed_cells_naive`]: near-ties inside the tolerance
/// band abort with [`GenericityError`] instead of guessing — re-lift with a
/// new seed. Degenerate liftings may abort at an interior node (before a
/// full tuple exists), and the interior band is the absolute `1e-9` floor
/// of the naive scan's relative band (see the module source); on generic
/// liftings, where margins clear the band by orders of magnitude, both
/// enumerators accept and reject identical tuples.
///
/// # Panics
///
/// Panics if some `liftings[i]` does not have exactly one value per point
/// of `supports[i]`.
pub fn mixed_cells<F: Real, const NV: usize>(
    supports: &[Support<NV>; NV],
    liftings: &[Lifting<F>; NV],
) -> Result<Vec<MixedCell<F, NV>>, GenericityError> {
    assert_lifting_lengths(supports, liftings);

    if NV == 0 {
        return Ok(Vec::new());
    }
    let edge_lists = candidate_edges(supports);
    if edge_lists.iter().any(Vec::is_empty) {
        return Ok(Vec::new()); // a support with < 2 points admits no edge
    }

    let band = tolerance::<F>();

    // Pre-filter each support's edges by their own-support LP. Edges whose
    // margin lands inside the band are *kept*: deciding them here would be
    // premature — deeper constraints may reject them decisively (the
    // harmless-tie case, which must not abort), and if nothing does, the
    // interior-node LP or the exact full-depth check raises the error with
    // naive-equivalent semantics.
    let viable: [Vec<EdgeData<F, NV>>; NV] = core::array::from_fn(|i| {
        edge_lists[i]
            .iter()
            .filter_map(|&(ja, jb)| {
                let edge = EdgeData::new(&supports[i], &liftings[i], ja, jb);
                match classify(
                    max_delta::<F, NV>(core::slice::from_ref(&edge.eq), &edge.ins),
                    band,
                ) {
                    NodeClass::Viable | NodeClass::Ambiguous => Some(edge),
                    NodeClass::Pruned => None,
                }
            })
            .collect()
    });
    if viable.iter().any(Vec::is_empty) {
        return Ok(Vec::new()); // some support has no viable edge: no cells
    }

    // Static support order: ascending viable-edge count, ties by index.
    let mut order: [usize; NV] = core::array::from_fn(|i| i);
    order.sort_by_key(|&i| (viable[i].len(), i));

    let mut search = Search {
        supports,
        liftings,
        viable: &viable,
        order: &order,
        band,
        cells: Vec::new(),
    };
    let mut chosen = [(0usize, 0usize); NV];
    let mut eqs: Vec<([F; NV], F)> = Vec::with_capacity(NV);
    let mut ins: Vec<([F; NV], F)> = Vec::new();
    search.dfs(0, &mut chosen, &mut eqs, &mut ins)?;
    Ok(search.cells)
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
