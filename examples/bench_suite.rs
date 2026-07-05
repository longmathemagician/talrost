//! Standard-systems benchmark suite for the polyhedral homotopy solver.
//!
//! Runs the named small systems of the polynomial-systems literature —
//! cyclic-n, katsura-n, noon-n, eco-n — plus the crate's own calibration
//! pairs end-to-end through the full pipeline (supports → lifting → mixed
//! cells → binomial starts → path tracking) and prints one table row per
//! system: mixed volume, cell/path counts, converged paths, offline and
//! tracking wall time, per-path cost, worst verified residual, and honest
//! failure tallies.
//!
//! Run with:
//!
//! ```text
//! cargo run --release --example bench_suite          # human-readable table
//! cargo run --release --example bench_suite -- --csv # machine-readable CSV
//! ```
//!
//! Expected root counts are pinned against the exact sympy oracle in
//! `tools/oracle-sympy/` (Groebner bases over Q); published mixed volumes
//! (cyclic-5 = 70, noon-3 = 21) are asserted. See BENCHMARKS.md for the
//! recorded results, methodology, and the literature context.
//!
//! Times are medians of 3 end-to-end repetitions per system (same seed —
//! the pipeline is deterministic, so the repetitions only smooth scheduler
//! noise). Offline = lifting + cell enumeration + homotopy/start
//! construction; tracking = every path of every cell.

// This example is also a test target (`test = true` in Cargo.toml) so that
// `bench_suite_root_counts` runs under plain `cargo test`; in that mode the
// harness ignores `main` and the printing helpers, so silence dead_code.
#![cfg_attr(test, allow(dead_code))]

use std::env;
use std::time::Instant;

use talrost::complex::c64;
use talrost::mvpoly::{MPoly, MSystem, Monomial};
use talrost::solvers::homotopy::{
    mixed_cells, random_liftings, start_solutions, track_path, CellHomotopy, GenericityError,
    Lifting, MixedCell, PathResult, PathStatus, Support, TrackOptions,
};

fn z(re: f64) -> c64 {
    c64::new(re, 0.0)
}

/// Accumulates `re · mono` into positionally-paired coefficient/monomial
/// arrays, merging repeated monomials — the shared primitive of the
/// programmatic system builders below. Slots beyond `used` stay zero
/// (padding, per the [`MPoly`] convention).
fn add_term<const NV: usize, const MAXT: usize>(
    coeffs: &mut [c64; MAXT],
    monos: &mut [Monomial<NV>; MAXT],
    used: &mut usize,
    mono: Monomial<NV>,
    re: f64,
) {
    for (c, m) in coeffs.iter_mut().zip(monos.iter()).take(*used) {
        if *m == mono {
            *c += z(re);
            return;
        }
    }
    assert!(*used < MAXT, "MAXT too small for this system");
    coeffs[*used] = z(re);
    monos[*used] = mono;
    *used += 1;
}

/// cyclic-n (Björck's cyclic n-roots problem), the standard benchmark family:
///
/// ```text
/// f_k = Σ_{i=0}^{n-1}  Π_{j=i}^{i+k-1} x_{j mod n}     (k = 1 .. n-1)
/// f_n = x_0·x_1·…·x_{n-1} − 1
/// ```
///
/// cyclic-3 has mixed volume 6 (the six permutations of the cube roots of
/// unity); cyclic-5's published mixed volume / root count is 70. cyclic-4 is
/// the family's classic degenerate member: its solution set is
/// positive-dimensional — the two curves (a, b, −a, −b) with ab = ±1 —
/// verified exactly by the sympy oracle, so no path can end on an isolated
/// root. Instantiate as `cyclic::<N, N>()` (the densest equation, f_1, has
/// exactly n terms).
fn cyclic<const NV: usize, const MAXT: usize>() -> MSystem<c64, NV, NV, MAXT> {
    let zero = Monomial::new([0i32; NV]);
    let mut polys = [MPoly::new([z(0.0); MAXT], [zero; MAXT]); NV];
    for k in 1..NV {
        let mut coeffs = [z(0.0); MAXT];
        let mut monos = [zero; MAXT];
        let mut used = 0;
        for i in 0..NV {
            let mut e = [0i32; NV];
            for j in i..i + k {
                e[j % NV] += 1;
            }
            add_term(&mut coeffs, &mut monos, &mut used, Monomial::new(e), 1.0);
        }
        polys[k - 1] = MPoly::new(coeffs, monos);
    }
    let mut coeffs = [z(0.0); MAXT];
    let mut monos = [zero; MAXT];
    let mut used = 0;
    add_term(
        &mut coeffs,
        &mut monos,
        &mut used,
        Monomial::new([1; NV]),
        1.0,
    );
    add_term(&mut coeffs, &mut monos, &mut used, zero, -1.0);
    polys[NV - 1] = MPoly::new(coeffs, monos);
    MSystem::new(polys)
}

/// katsura-n (the magnetism problem), in the **(n+1)-unknown convention**
/// `u_0 … u_n` — root count 2^n, confirmed by the sympy oracle (dim of the
/// quotient ring; note that naming conventions differ across the literature,
/// some call this system katsura-(n+1)):
///
/// ```text
/// Σ_{l=-n}^{n} u_{|l|}·u_{|m-l|}  =  u_m      (m = 0 .. n-1; terms with
///                                              |m-l| > n are dropped)
/// u_0 + 2·(u_1 + … + u_n)  =  1
/// ```
///
/// Instantiate as `katsura::<{n+1}, {n+2}>()`: the m = 0 equation has n+2
/// distinct terms (u_0², 2u_1², …, 2u_n², −u_0), the widest row.
fn katsura<const NV: usize, const MAXT: usize>() -> MSystem<c64, NV, NV, MAXT> {
    assert!(NV >= 2);
    let n = (NV - 1) as i32;
    let zero = Monomial::new([0i32; NV]);
    let mut polys = [MPoly::new([z(0.0); MAXT], [zero; MAXT]); NV];
    for m in 0..n {
        let mut coeffs = [z(0.0); MAXT];
        let mut monos = [zero; MAXT];
        let mut used = 0;
        for l in -n..=n {
            if (m - l).abs() <= n {
                let mut e = [0i32; NV];
                e[l.unsigned_abs() as usize] += 1;
                e[(m - l).unsigned_abs() as usize] += 1;
                add_term(&mut coeffs, &mut monos, &mut used, Monomial::new(e), 1.0);
            }
        }
        let mut e = [0i32; NV];
        e[m as usize] = 1;
        add_term(&mut coeffs, &mut monos, &mut used, Monomial::new(e), -1.0);
        polys[m as usize] = MPoly::new(coeffs, monos);
    }
    let mut coeffs = [z(0.0); MAXT];
    let mut monos = [zero; MAXT];
    let mut used = 0;
    for i in 0..NV {
        let mut e = [0i32; NV];
        e[i] = 1;
        let w = if i == 0 { 1.0 } else { 2.0 };
        add_term(&mut coeffs, &mut monos, &mut used, Monomial::new(e), w);
    }
    add_term(&mut coeffs, &mut monos, &mut used, zero, -1.0);
    polys[NV - 1] = MPoly::new(coeffs, monos);
    MSystem::new(polys)
}

/// noon-n (Noonburg's neural-network system, coefficient 1.1 as in the
/// PHCpack demo database):
///
/// ```text
/// x_i·(Σ_{j≠i} x_j²) − 1.1·x_i + 1 = 0        (i = 1 .. n)
/// ```
///
/// noon-3's published root count is 21, all on the torus (x_i = 0 leaves
/// 1 = 0), and its mixed volume is exactly 21 — a BKK-exact system.
/// Instantiate as `noon::<N, {N+1}>()` (n−1 cubic terms + linear + constant).
fn noon<const NV: usize, const MAXT: usize>() -> MSystem<c64, NV, NV, MAXT> {
    let zero = Monomial::new([0i32; NV]);
    let mut polys = [MPoly::new([z(0.0); MAXT], [zero; MAXT]); NV];
    for (i, poly) in polys.iter_mut().enumerate() {
        let mut coeffs = [z(0.0); MAXT];
        let mut monos = [zero; MAXT];
        let mut used = 0;
        for j in 0..NV {
            if j != i {
                let mut e = [0i32; NV];
                e[i] = 1;
                e[j] = 2;
                add_term(&mut coeffs, &mut monos, &mut used, Monomial::new(e), 1.0);
            }
        }
        let mut e = [0i32; NV];
        e[i] = 1;
        add_term(&mut coeffs, &mut monos, &mut used, Monomial::new(e), -1.1);
        add_term(&mut coeffs, &mut monos, &mut used, zero, 1.0);
        *poly = MPoly::new(coeffs, monos);
    }
    MSystem::new(polys)
}

/// eco-n (Morgan's economics problem, PHCpack formulation):
///
/// ```text
/// (x_k + Σ_{i=1}^{n-k-1} x_i·x_{i+k})·x_n − k = 0    (k = 1 .. n-1)
/// x_1 + x_2 + … + x_{n-1} + 1 = 0
/// ```
///
/// The k-th equation's constant −k keeps x_n (and x_{n-1}, via k = n−1) off
/// zero, but the sympy oracle confirms all solutions lie on the torus and
/// counts them exactly (eco-4: 4, eco-5: 8). Instantiate as
/// `eco::<N, N>()` (the k = 1 equation and the linear equation both have n
/// terms).
fn eco<const NV: usize, const MAXT: usize>() -> MSystem<c64, NV, NV, MAXT> {
    assert!(NV >= 3);
    let zero = Monomial::new([0i32; NV]);
    let mut polys = [MPoly::new([z(0.0); MAXT], [zero; MAXT]); NV];
    for k in 1..NV {
        let mut coeffs = [z(0.0); MAXT];
        let mut monos = [zero; MAXT];
        let mut used = 0;
        // x_k · x_n  (1-indexed) = e[k-1] + e[NV-1].
        let mut e = [0i32; NV];
        e[k - 1] += 1;
        e[NV - 1] += 1;
        add_term(&mut coeffs, &mut monos, &mut used, Monomial::new(e), 1.0);
        // x_i · x_{i+k} · x_n for i = 1 .. n-k-1.
        for i in 1..NV - k {
            let mut e = [0i32; NV];
            e[i - 1] += 1;
            e[i + k - 1] += 1;
            e[NV - 1] += 1;
            add_term(&mut coeffs, &mut monos, &mut used, Monomial::new(e), 1.0);
        }
        add_term(&mut coeffs, &mut monos, &mut used, zero, -(k as f64));
        polys[k - 1] = MPoly::new(coeffs, monos);
    }
    let mut coeffs = [z(0.0); MAXT];
    let mut monos = [zero; MAXT];
    let mut used = 0;
    for i in 0..NV - 1 {
        let mut e = [0i32; NV];
        e[i] = 1;
        add_term(&mut coeffs, &mut monos, &mut used, Monomial::new(e), 1.0);
    }
    add_term(&mut coeffs, &mut monos, &mut used, zero, 1.0);
    polys[NV - 1] = MPoly::new(coeffs, monos);
    MSystem::new(polys)
}

/// The sparse trinomial pair from the driver tests — the crate's smallest
/// calibration row (mixed volume 2 < Bézout 4):
/// `1 − 3x + xy = 0`, `2 + y + xy = 0`.
fn trinomial() -> MSystem<c64, 2, 2, 3> {
    let f1 = MPoly::new(
        [z(1.0), z(-3.0), z(1.0)],
        [
            Monomial::new([0, 0]),
            Monomial::new([1, 0]),
            Monomial::new([1, 1]),
        ],
    );
    let f2 = MPoly::new(
        [z(2.0), z(1.0), z(1.0)],
        [
            Monomial::new([0, 0]),
            Monomial::new([0, 1]),
            Monomial::new([1, 1]),
        ],
    );
    MSystem::new([f1, f2])
}

/// The dense generic conic pair from the driver tests (fixed complex
/// coefficients; mixed volume = Bézout = 4) — the dense calibration row.
fn conic() -> MSystem<c64, 2, 2, 6> {
    let c = c64::new;
    let monos = [
        Monomial::new([0, 0]),
        Monomial::new([1, 0]),
        Monomial::new([0, 1]),
        Monomial::new([2, 0]),
        Monomial::new([1, 1]),
        Monomial::new([0, 2]),
    ];
    let f1 = MPoly::new(
        [
            c(1.1, 0.3),
            c(-0.7, 0.9),
            c(0.5, -1.3),
            c(2.0, 0.1),
            c(-1.4, -0.8),
            c(0.6, 1.7),
        ],
        monos,
    );
    let f2 = MPoly::new(
        [
            c(-0.9, 1.2),
            c(1.8, -0.4),
            c(0.3, 0.7),
            c(-1.1, -1.6),
            c(0.8, 0.2),
            c(1.5, -0.5),
        ],
        monos,
    );
    MSystem::new([f1, f2])
}

// --------------------------------------------------------------------------
// The benchmark framework.
// --------------------------------------------------------------------------

/// End-to-end repetitions per system (medians reported).
const REPS: usize = 3;

/// Empirical cell-enumeration cost model for the pre-run frontier estimate:
/// one candidate edge tuple costs about `TUPLE_NS_PER_NV3 · NV³` ns (exact
/// SNF singularity gate + NV×NV LU solve + strict-minimality verification).
/// Measured on the BENCHMARKS.md machine: katsura-4's 135 000 five-variable
/// tuples took ~157 ms ≈ 9.3 ns·NV³ each, cyclic-5's 10 000 took
/// ~6.1 ns·NV³ each; 10 is the conservative round-up. Order of magnitude
/// only — the gate has a wide budget.
const TUPLE_NS_PER_NV3: f64 = 10.0;

/// Systems whose projected enumeration exceeds this budget are excluded
/// with a printed note — the enumeration frontier documented in
/// BENCHMARKS.md.
const ENUM_BUDGET_S: f64 = 60.0;

/// One finished table row.
struct Row {
    name: &'static str,
    nv: usize,
    mv: u64,
    cells: usize,
    paths: usize,
    converged: usize,
    offline_ms: f64,
    tracking_ms: f64,
    max_residual: f64,
    failures: String,
    note: &'static str,
}

/// A system either ran, or was excluded by the enumeration-frontier gate.
enum Outcome {
    Ran(Row),
    Excluded {
        name: &'static str,
        nv: usize,
        tuples: f64,
        projected_s: f64,
    },
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// `‖F(x)‖∞` — the verified-residual metric of the table.
fn residual_inf<const NV: usize, const MAXT: usize>(
    system: &MSystem<c64, NV, NV, MAXT>,
    x: &[c64; NV],
) -> f64 {
    system
        .eval(x)
        .iter()
        .fold(0.0f64, |m, r| m.max(r.magnitude()))
}

/// Non-converged statuses tallied as `"2 diverged; 4 min-step"`, or `"-"`.
fn failure_summary(statuses: &[PathStatus]) -> String {
    let mut out = String::new();
    for (status, label) in [
        (PathStatus::MinStepReached, "min-step"),
        (PathStatus::MaxStepsReached, "max-steps"),
        (PathStatus::SingularJacobian, "singular"),
        (PathStatus::Diverged, "diverged"),
    ] {
        let n = statuses.iter().filter(|s| **s == status).count();
        if n > 0 {
            if !out.is_empty() {
                out.push_str("; ");
            }
            out.push_str(&format!("{} {}", n, label));
        }
    }
    if out.is_empty() {
        out.push('-');
    }
    out
}

type OfflineData<const NV: usize> = ([Lifting<f64>; NV], Vec<MixedCell<f64, NV>>);

/// Lift with `seed` and enumerate cells, retrying once with `seed + 1` on a
/// degenerate draw — the same policy as the `solve()` driver.
fn lift_or_retry<const NV: usize>(supports: &[Support<NV>; NV], seed: u64) -> OfflineData<NV> {
    let attempt = |s: u64| -> Result<OfflineData<NV>, GenericityError> {
        let liftings = random_liftings::<f64, NV>(supports, s);
        let cells = mixed_cells(supports, &liftings)?;
        Ok((liftings, cells))
    };
    attempt(seed)
        .or_else(|_| attempt(seed.wrapping_add(1)))
        .expect("lifting degenerate for two consecutive seeds")
}

/// Runs one system end-to-end (REPS times, medians) and builds its row.
/// `expect_mv` pins the published/oracle-verified mixed volume — a mismatch
/// is a solver bug and panics.
fn run_system<const NV: usize, const MAXT: usize>(
    name: &'static str,
    system: &MSystem<c64, NV, NV, MAXT>,
    seed: u64,
    expect_mv: Option<u64>,
    note: &'static str,
) -> Outcome {
    eprintln!("[bench_suite] running {} ...", name);
    let supports = Support::from_msystem(system);

    // Enumeration-frontier gate: the naive cell enumeration visits
    // Π C(|A_i|, 2) candidate edge tuples; estimate before running.
    let tuples: f64 = supports
        .iter()
        .map(|s| {
            let n = s.len() as f64;
            n * (n - 1.0) / 2.0
        })
        .product();
    let projected_s = tuples * ((NV * NV * NV) as f64) * TUPLE_NS_PER_NV3 * 1e-9;
    if projected_s > ENUM_BUDGET_S {
        return Outcome::Excluded {
            name,
            nv: NV,
            tuples,
            projected_s,
        };
    }

    let options = TrackOptions::default();
    let mut offline_ms = Vec::with_capacity(REPS);
    let mut tracking_ms = Vec::with_capacity(REPS);
    let mut results: Vec<PathResult<f64, NV>> = Vec::new();
    let mut mv = 0u64;
    let mut n_cells = 0usize;

    for _ in 0..REPS {
        // Offline: lifting, cell enumeration, homotopy + start construction.
        let t0 = Instant::now();
        let (liftings, cells) = lift_or_retry(&supports, seed);
        let jobs: Vec<(CellHomotopy<f64, NV, MAXT>, Vec<[c64; NV]>)> = cells
            .iter()
            .map(|cell| {
                (
                    CellHomotopy::new(system, &supports, &liftings, cell),
                    start_solutions(system, cell),
                )
            })
            .collect();
        offline_ms.push(t0.elapsed().as_secs_f64() * 1e3);

        // Online: track every start of every cell.
        let t1 = Instant::now();
        let mut res = Vec::new();
        for (homotopy, starts) in &jobs {
            for start in starts {
                res.push(track_path(homotopy, *start, &options));
            }
        }
        tracking_ms.push(t1.elapsed().as_secs_f64() * 1e3);

        mv = cells.iter().map(MixedCell::volume).sum();
        n_cells = cells.len();
        results = res;
    }

    if let Some(want) = expect_mv {
        assert_eq!(
            mv, want,
            "{}: computed mixed volume {} != expected {}",
            name, mv, want
        );
    }

    let converged = results
        .iter()
        .filter(|p| p.status == PathStatus::Converged)
        .count();
    // Worst residual over the *converged* endpoints (NaN → no row entry).
    let max_residual = results
        .iter()
        .filter(|p| p.status == PathStatus::Converged)
        .map(|p| residual_inf(system, &p.point))
        .fold(f64::NAN, f64::max);
    let statuses: Vec<PathStatus> = results.iter().map(|p| p.status).collect();

    Outcome::Ran(Row {
        name,
        nv: NV,
        mv,
        cells: n_cells,
        paths: results.len(),
        converged,
        offline_ms: median(offline_ms),
        tracking_ms: median(tracking_ms),
        max_residual,
        failures: failure_summary(&statuses),
        note,
    })
}

fn print_table(outcomes: &[Outcome]) {
    println!(
        "talrost polyhedral homotopy benchmark suite ({} build; single-threaded f64, RK4 predictor, no endgames)",
        if cfg!(debug_assertions) {
            "debug -- numbers are meaningless, use --release"
        } else {
            "release"
        },
    );
    println!("{:-<120}", "");
    println!(
        "{:<10} {:>3} {:>4} {:>6} {:>6} {:>5} {:>11} {:>10} {:>9} {:>10}  failures",
        "system",
        "nv",
        "mv",
        "cells",
        "paths",
        "conv",
        "offline ms",
        "track ms",
        "us/path",
        "max resid",
    );
    println!("{:-<120}", "");
    for outcome in outcomes {
        match outcome {
            Outcome::Ran(r) => {
                let per_path = if r.paths > 0 {
                    r.tracking_ms * 1e3 / r.paths as f64
                } else {
                    0.0
                };
                let resid = if r.max_residual.is_nan() {
                    "-".to_string()
                } else {
                    format!("{:.1e}", r.max_residual)
                };
                println!(
                    "{:<10} {:>3} {:>4} {:>6} {:>6} {:>5} {:>11.3} {:>10.3} {:>9.1} {:>10}  {}",
                    r.name,
                    r.nv,
                    r.mv,
                    r.cells,
                    r.paths,
                    r.converged,
                    r.offline_ms,
                    r.tracking_ms,
                    per_path,
                    resid,
                    r.failures
                );
            }
            Outcome::Excluded {
                name,
                nv,
                tuples,
                projected_s,
            } => {
                println!(
                    "{:<10} {:>3}  excluded: ~{:.1e} candidate edge tuples, projected ~{:.0} s enumeration (> {:.0} s budget)",
                    name, nv, tuples, projected_s, ENUM_BUDGET_S
                );
            }
        }
    }
    println!("{:-<120}", "");
    for outcome in outcomes {
        if let Outcome::Ran(r) = outcome {
            if !r.note.is_empty() {
                println!("note: {:<10} {}", r.name, r.note);
            }
        }
    }
}

fn print_csv(outcomes: &[Outcome]) {
    println!(
        "system,nv,mixed_volume,cells,paths,converged,offline_ms,tracking_ms,us_per_path,max_residual,failures,note"
    );
    for outcome in outcomes {
        match outcome {
            Outcome::Ran(r) => {
                let per_path = if r.paths > 0 {
                    r.tracking_ms * 1e3 / r.paths as f64
                } else {
                    0.0
                };
                let resid = if r.max_residual.is_nan() {
                    String::new()
                } else {
                    format!("{:.3e}", r.max_residual)
                };
                println!(
                    "{},{},{},{},{},{},{:.3},{:.3},{:.2},{},{},{}",
                    r.name,
                    r.nv,
                    r.mv,
                    r.cells,
                    r.paths,
                    r.converged,
                    r.offline_ms,
                    r.tracking_ms,
                    per_path,
                    resid,
                    r.failures,
                    // Commas would break the line-oriented schema.
                    r.note.replace(',', ";")
                );
            }
            Outcome::Excluded {
                name,
                nv,
                tuples,
                projected_s,
            } => {
                println!(
                    "{},{},,,,,,,,,excluded,projected ~{:.0} s enumeration for {:.1e} tuples",
                    name, nv, projected_s, tuples
                );
            }
        }
    }
}

fn main() {
    let csv = env::args().any(|a| a == "--csv");

    // Expected mixed volumes: published values where the literature pins one
    // (cyclic-5: 70, noon-3: 21), otherwise the value computed by this
    // solver and cross-checked against the sympy oracle's exact solution
    // counts (see tools/oracle-sympy/README.md and BENCHMARKS.md).
    let outcomes = vec![
        run_system(
            "trinomial",
            &trinomial(),
            3,
            Some(2),
            "calibration row; oracle: 2 distinct roots, both found",
        ),
        run_system(
            "conic",
            &conic(),
            4,
            Some(4),
            "calibration row; oracle: 4 distinct roots, all found",
        ),
        run_system(
            "cyclic-3",
            &cyclic::<3, 3>(),
            1,
            Some(6),
            "oracle: 6 distinct torus roots (permutations of the cube roots of unity)",
        ),
        run_system(
            "cyclic-4",
            &cyclic::<4, 4>(),
            1,
            Some(16),
            "DEGENERATE target (oracle): positive-dimensional solution set (two curves); \
             isolated-root convergence is impossible and failures here are honest reporting. \
             MV 16 is seed-invariant (the generic root count of these supports)",
        ),
        run_system(
            "cyclic-5",
            &cyclic::<5, 5>(),
            1,
            Some(70),
            "published mixed volume / root count 70",
        ),
        run_system(
            "katsura-3",
            &katsura::<4, 5>(),
            1,
            Some(6),
            "oracle: 8 distinct affine roots, exactly 6 on the torus = MV; \
             the 2 off-torus roots are invisible to Bernstein's count",
        ),
        run_system(
            "katsura-4",
            &katsura::<5, 6>(),
            1,
            Some(12),
            "oracle: 16 distinct affine roots, exactly 12 on the torus = MV",
        ),
        run_system(
            "noon-3",
            &noon::<3, 4>(),
            1,
            Some(21),
            "oracle: 21 distinct torus roots; published count 21; BKK-exact",
        ),
        run_system(
            "eco-4",
            &eco::<4, 4>(),
            1,
            Some(4),
            "oracle: 4 distinct roots, all on the torus",
        ),
        run_system(
            "eco-5",
            &eco::<5, 5>(),
            1,
            Some(8),
            "oracle: 8 distinct roots, all on the torus",
        ),
        // The enumeration frontier, demonstrated: cyclic-7 (published root
        // count 924) projects to C(7,2)^6 ≈ 8.6e7 candidate tuples — hours
        // of naive enumeration — and is excluded by the pre-run gate. See
        // BENCHMARKS.md.
        run_system(
            "cyclic-7",
            &cyclic::<7, 7>(),
            1,
            None,
            "excluded by the enumeration-frontier gate",
        ),
    ];

    if csv {
        print_csv(&outcomes);
    } else {
        print_table(&outcomes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use talrost::solvers::homotopy::solve;

    /// Solves `system` with the driver and asserts the mixed volume, the
    /// converged-path count, and residual < 1e-8 on every converged
    /// endpoint.
    fn assert_counts<const NV: usize, const MAXT: usize>(
        name: &str,
        system: &MSystem<c64, NV, NV, MAXT>,
        seed: u64,
        mv: u64,
        converged: usize,
    ) {
        let report = solve(system, seed, &TrackOptions::default()).unwrap();
        assert_eq!(report.mixed_volume, mv, "{}: mixed volume", name);
        assert_eq!(
            report.converged_count(),
            converged,
            "{}: converged paths",
            name
        );
        for p in report
            .paths
            .iter()
            .filter(|p| p.status == PathStatus::Converged)
        {
            let r = residual_inf(system, &p.point);
            assert!(r < 1e-8, "{}: residual {:.3e} too large", name, r);
        }
    }

    /// The small suite systems reach exactly the root counts the sympy
    /// oracle proved (tools/oracle-sympy/README.md), with verified
    /// residuals: trinomial 2, conic 4, cyclic-3 6, noon-3 21, eco-4 4, and
    /// katsura-3 6 — its mixed volume equals its *torus* root count (the
    /// oracle counts 8 affine roots, 2 of them with zero coordinates, which
    /// Bernstein's theorem does not count and no path targets).
    #[test]
    fn bench_suite_root_counts() {
        assert_counts("trinomial", &trinomial(), 3, 2, 2);
        assert_counts("conic", &conic(), 4, 4, 4);
        assert_counts("cyclic-3", &cyclic::<3, 3>(), 1, 6, 6);
        assert_counts("noon-3", &noon::<3, 4>(), 1, 21, 21);
        assert_counts("katsura-3", &katsura::<4, 5>(), 1, 6, 6);
        assert_counts("eco-4", &eco::<4, 4>(), 1, 4, 4);
    }
}
