//! Oracle tests for the DEMiCs-style mixed-cell enumerator: on every system
//! the naive exhaustive scan can handle, `mixed_cells` (LP-pruned tree
//! search) must produce the **identical** cell set as `mixed_cells_naive`
//! (the reference implementation) — same count, same edge tuples, same
//! integer edge matrices, same exact volumes and mixed volume, and normals
//! equal within 1e-12 (they take the same final LU code path, so they are
//! expected to match bit for bit; 1e-12 is the documented contract).
//!
//! Systems: the crate's calibration pairs (trinomial, conic), cyclic-3/4/5,
//! katsura-3/4, noon-3, eco-4/5, and seeded randomized support sets. The
//! support builders mirror `examples/bench_suite.rs` (monomial structure
//! only — coefficients don't matter for enumeration).

#![cfg(feature = "std")]

use talrost::mvpoly::Monomial;
use talrost::solvers::homotopy::{
    mixed_cells, mixed_cells_naive, random_liftings, MixedCell, Support,
};

/// Sorts a cell list into a canonical (edge-exponent lexicographic) order.
fn canonicalize<const NV: usize>(mut cells: Vec<MixedCell<f64, NV>>) -> Vec<MixedCell<f64, NV>> {
    let key = |c: &MixedCell<f64, NV>| -> Vec<i32> {
        let mut k = Vec::with_capacity(2 * NV * NV);
        for (a, b) in &c.edges {
            k.extend_from_slice(&a.exps);
            k.extend_from_slice(&b.exps);
        }
        k
    };
    cells.sort_by_key(key);
    cells
}

/// Runs both enumerators on `supports` with the seed's lifting and asserts
/// identical results; returns the (agreed) mixed volume.
fn assert_enumerators_agree<const NV: usize>(
    name: &str,
    supports: &[Support<NV>; NV],
    seed: u64,
) -> u64 {
    let liftings = random_liftings::<f64, NV>(supports, seed);
    let naive = mixed_cells_naive(supports, &liftings)
        .unwrap_or_else(|e| panic!("{name}, seed {seed}: naive enumerator failed: {e}"));
    let fast = mixed_cells(supports, &liftings)
        .unwrap_or_else(|e| panic!("{name}, seed {seed}: tree-search enumerator failed: {e}"));
    assert_eq!(
        naive.len(),
        fast.len(),
        "{name}, seed {seed}: cell count mismatch"
    );

    let naive = canonicalize(naive);
    let fast = canonicalize(fast);
    for (a, b) in naive.iter().zip(fast.iter()) {
        assert_eq!(a.edges, b.edges, "{name}, seed {seed}: edge tuple mismatch");
        assert_eq!(
            a.edge_matrix, b.edge_matrix,
            "{name}, seed {seed}: edge matrix mismatch"
        );
        assert_eq!(
            a.volume(),
            b.volume(),
            "{name}, seed {seed}: cell volume mismatch"
        );
        for (na, nb) in a.normal.iter().zip(b.normal.iter()) {
            assert!(
                (na - nb).abs() <= 1e-12,
                "{name}, seed {seed}: normal mismatch {na} vs {nb}"
            );
        }
    }

    let mv_naive: u64 = naive.iter().map(MixedCell::volume).sum();
    let mv_fast: u64 = fast.iter().map(MixedCell::volume).sum();
    assert_eq!(
        mv_naive, mv_fast,
        "{name}, seed {seed}: mixed volume mismatch"
    );
    mv_naive
}

/// Both enumerators across two seeds, plus the published/oracle-pinned
/// mixed volume.
fn check_system<const NV: usize>(name: &str, supports: &[Support<NV>; NV], expect_mv: u64) {
    for seed in [1, 2] {
        let mv = assert_enumerators_agree(name, supports, seed);
        assert_eq!(mv, expect_mv, "{name}, seed {seed}: mixed volume");
    }
}

// ---------------------------------------------------------------------------
// Support builders (monomial structure of the bench_suite systems).
// ---------------------------------------------------------------------------

fn support_from_exps<const NV: usize>(exps: &[[i32; NV]]) -> Support<NV> {
    let monos: Vec<Monomial<NV>> = exps.iter().map(|&e| Monomial::new(e)).collect();
    Support::new(&monos)
}

/// cyclic-n: equation k (1 ≤ k < n) has the n cyclic degree-k products;
/// equation n is `x_1…x_n − 1` (2 points).
fn cyclic_supports<const NV: usize>() -> [Support<NV>; NV] {
    core::array::from_fn(|eq| {
        let k = eq + 1;
        if k < NV {
            let pts: Vec<[i32; NV]> = (0..NV)
                .map(|i| {
                    let mut e = [0i32; NV];
                    for j in i..i + k {
                        e[j % NV] += 1;
                    }
                    e
                })
                .collect();
            support_from_exps(&pts)
        } else {
            support_from_exps(&[[1; NV], [0; NV]])
        }
    })
}

/// katsura-n in the (n+1)-unknown convention (`NV = n + 1`); see
/// `examples/bench_suite.rs` for the equations.
fn katsura_supports<const NV: usize>() -> [Support<NV>; NV] {
    let n = (NV - 1) as i32;
    core::array::from_fn(|eq| {
        if (eq as i32) < n {
            let m = eq as i32;
            let mut pts: Vec<[i32; NV]> = Vec::new();
            for l in -n..=n {
                if (m - l).abs() <= n {
                    let mut e = [0i32; NV];
                    e[l.unsigned_abs() as usize] += 1;
                    e[(m - l).unsigned_abs() as usize] += 1;
                    pts.push(e);
                }
            }
            let mut lin = [0i32; NV];
            lin[m as usize] = 1;
            pts.push(lin);
            support_from_exps(&pts)
        } else {
            let mut pts: Vec<[i32; NV]> = (0..NV)
                .map(|i| {
                    let mut e = [0i32; NV];
                    e[i] = 1;
                    e
                })
                .collect();
            pts.push([0i32; NV]);
            support_from_exps(&pts)
        }
    })
}

/// noon-n: equation i is `x_i·Σ_{j≠i} x_j² − 1.1·x_i + 1`.
fn noon_supports<const NV: usize>() -> [Support<NV>; NV] {
    core::array::from_fn(|i| {
        let mut pts: Vec<[i32; NV]> = Vec::new();
        for j in 0..NV {
            if j != i {
                let mut e = [0i32; NV];
                e[i] = 1;
                e[j] = 2;
                pts.push(e);
            }
        }
        let mut lin = [0i32; NV];
        lin[i] = 1;
        pts.push(lin);
        pts.push([0i32; NV]);
        support_from_exps(&pts)
    })
}

/// eco-n (PHCpack formulation); see `examples/bench_suite.rs`.
fn eco_supports<const NV: usize>() -> [Support<NV>; NV] {
    core::array::from_fn(|eq| {
        let k = eq + 1;
        if k < NV {
            let mut pts: Vec<[i32; NV]> = Vec::new();
            let mut e = [0i32; NV];
            e[k - 1] += 1;
            e[NV - 1] += 1;
            pts.push(e);
            for i in 1..NV - k {
                let mut e = [0i32; NV];
                e[i - 1] += 1;
                e[i + k - 1] += 1;
                e[NV - 1] += 1;
                pts.push(e);
            }
            pts.push([0i32; NV]);
            support_from_exps(&pts)
        } else {
            let mut pts: Vec<[i32; NV]> = (0..NV - 1)
                .map(|i| {
                    let mut e = [0i32; NV];
                    e[i] = 1;
                    e
                })
                .collect();
            pts.push([0i32; NV]);
            support_from_exps(&pts)
        }
    })
}

// ---------------------------------------------------------------------------
// The oracle tests.
// ---------------------------------------------------------------------------

#[test]
fn trinomial_pair_agrees() {
    let supports = [
        support_from_exps(&[[0, 0], [1, 0], [1, 1]]),
        support_from_exps(&[[0, 0], [0, 1], [1, 1]]),
    ];
    check_system("trinomial", &supports, 2);
}

#[test]
fn dense_conic_agrees() {
    let conic = support_from_exps(&[[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2]]);
    check_system("conic", &[conic.clone(), conic], 4);
}

#[test]
fn cyclic_3_4_5_agree() {
    check_system("cyclic-3", &cyclic_supports::<3>(), 6);
    check_system("cyclic-4", &cyclic_supports::<4>(), 16);
    check_system("cyclic-5", &cyclic_supports::<5>(), 70);
}

#[test]
fn katsura_3_agrees() {
    check_system("katsura-3", &katsura_supports::<4>(), 6);
}

#[test]
fn katsura_4_agrees() {
    // The largest naive-tractable system (~135 000 candidate tuples).
    check_system("katsura-4", &katsura_supports::<5>(), 12);
}

#[test]
fn noon_3_agrees() {
    check_system("noon-3", &noon_supports::<3>(), 21);
}

#[test]
fn eco_4_5_agree() {
    check_system("eco-4", &eco_supports::<4>(), 4);
    check_system("eco-5", &eco_supports::<5>(), 8);
}

/// Randomized support sets (seeded, reproducible): 3-variable systems with
/// exponents drawn from {0, 1, 2}³, four distinct points per support. No
/// published mixed volume exists for these, so the assertion is pure
/// naive-vs-tree-search agreement — across three point-set seeds and two
/// lifting seeds each.
#[test]
fn randomized_supports_agree() {
    // The MMIX LCG used by `Lifting::random`, reused here so the point sets
    // are deterministic without new dependencies.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self, bound: i32) -> i32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 40) % bound as u64) as i32
        }
    }

    for master_seed in [11u64, 42, 2026] {
        let mut lcg = Lcg(master_seed);
        let supports: [Support<3>; 3] = core::array::from_fn(|_| {
            let mut pts: Vec<[i32; 3]> = Vec::new();
            while pts.len() < 4 {
                let p = [lcg.next(3), lcg.next(3), lcg.next(3)];
                if !pts.contains(&p) {
                    pts.push(p);
                }
            }
            support_from_exps(&pts)
        });
        for seed in [1, 7] {
            let name = format!("random-{master_seed}");
            let mv = assert_enumerators_agree(&name, &supports, seed);
            // Same supports, different lifting: the mixed volume is a
            // lifting invariant, so the two seeds must agree with each
            // other (checked here) and with the naive scan (checked above).
            let mv2 = assert_enumerators_agree(&name, &supports, seed + 100);
            assert_eq!(mv, mv2, "{name}: mixed volume must be seed-invariant");
        }
    }
}
