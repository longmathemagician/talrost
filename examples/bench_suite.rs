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
//! cargo run --release --example bench_suite           # human-readable table
//! cargo run --release --example bench_suite -- --csv  # machine-readable CSV
//! cargo run --release --example bench_suite -- --enum # naive-vs-DEMiCs
//!                                                     # enumeration compare
//! ```
//!
//! Expected root counts are pinned against the exact sympy oracle in
//! `tools/oracle-sympy/` (Groebner bases over Q); published mixed volumes
//! (cyclic-5 = 70, cyclic-6 = 156, cyclic-7 = 924, noon-3 = 21) are
//! asserted. See BENCHMARKS.md for the recorded results, methodology, and
//! the literature context.
//!
//! Times are medians of 3 end-to-end repetitions per system (same seed —
//! the pipeline is deterministic, so the repetitions only smooth scheduler
//! noise). Offline = lifting + cell enumeration + homotopy/start
//! construction; tracking = every path of every cell.
//!
//! `--enum` measures cell **enumeration only**, side by side: the naive
//! `Π C(|A_i|, 2)` scan (`mixed_cells_naive`, measured fresh where its
//! projected time fits the budget, projected otherwise) against the
//! LP-pruned DEMiCs-style tree search (`mixed_cells`), including cyclic-8
//! (mixed volume 2560), which only the tree search can reach.

// This example is also a test target (`test = true` in Cargo.toml) so that
// `bench_suite_root_counts` runs under plain `cargo test`; in that mode the
// harness ignores `main` and the printing helpers, so silence dead_code.
#![cfg_attr(test, allow(dead_code))]

use std::env;
use std::time::Instant;

use talrost::complex::c64;
use talrost::mvpoly::{MPoly, MSystem, Monomial};
use talrost::solvers::homotopy::{
    mixed_cells, mixed_cells_naive, random_liftings, start_solutions, track_path, CellHomotopy,
    GenericityError, Lifting, MixedCell, PathResult, PathStatus, Support, TrackOptions,
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

/// {x² − 2x + 1, y − x}: mixed volume 2, and the target's only root is the
/// **double root** (1, 1) — the Cauchy-endgame calibration row (both paths
/// must finish `conv-singular` with winding 2; the row makes the endgame's
/// cost visible next to the regular systems).
fn double_root() -> MSystem<c64, 2, 2, 3> {
    let f1 = MPoly::new(
        [z(1.0), z(-2.0), z(1.0)],
        [
            Monomial::new([0, 0]),
            Monomial::new([1, 0]),
            Monomial::new([2, 0]),
        ],
    );
    let f2 = MPoly::new(
        [z(0.0), z(-1.0), z(1.0)],
        [
            Monomial::new([0, 0]),
            Monomial::new([1, 0]),
            Monomial::new([0, 1]),
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

/// Empirical cost model for the **naive** enumerator, used only to decide
/// whether the `--enum` mode measures it or prints a projection: one
/// candidate edge tuple costs about `TUPLE_NS_PER_NV3 · NV³` ns (exact SNF
/// singularity gate + NV×NV LU solve + strict-minimality verification).
/// Measured on the BENCHMARKS.md machine: katsura-4's 135 000 five-variable
/// tuples took ~157 ms ≈ 9.3 ns·NV³ each, cyclic-5's 10 000 took
/// ~6.1 ns·NV³ each; 10 is the conservative round-up. Order of magnitude
/// only.
const TUPLE_NS_PER_NV3: f64 = 10.0;

/// `--enum` measures the naive scan only when its projection fits this
/// budget (per repetition); beyond it the projection is printed instead.
const NAIVE_BUDGET_S: f64 = 60.0;

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

/// Non-root statuses tallied as `"2 diverged; 4 min-step"`, or `"-"`.
/// `ConvergedSingular` is a success (counted in the `conv` column), not a
/// failure.
fn failure_summary(statuses: &[PathStatus]) -> String {
    let mut out = String::new();
    for (status, label) in [
        (PathStatus::MinStepReached, "min-step"),
        (PathStatus::MaxStepsReached, "max-steps"),
        (PathStatus::SingularJacobian, "singular"),
        (PathStatus::Diverged, "diverged"),
        (PathStatus::EndgameFailed, "endgame-fail"),
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
/// is a solver bug and panics. `allow_endgame` marks the systems with
/// singular/degenerate targets; on every other row a single
/// `endgame_entered` path panics — the benchmark doubles as the
/// suite-scale proof (1000+ paths, cyclic-5/6/7 included) that healthy
/// paths never trigger the Cauchy endgame.
fn run_system<const NV: usize, const MAXT: usize>(
    name: &'static str,
    system: &MSystem<c64, NV, NV, MAXT>,
    seed: u64,
    expect_mv: Option<u64>,
    allow_endgame: bool,
    note: &'static str,
) -> Row {
    eprintln!("[bench_suite] running {} ...", name);
    let supports = Support::from_msystem(system);

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

    // Root-bearing paths: plain Converged plus Cauchy-endgame singular
    // endpoints (the latter only ever appear for singular targets — the
    // invariant below plus the regular rows' "conv == paths" keep the
    // endgame honest at benchmark scale).
    let converged = results.iter().filter(|p| p.status.is_root()).count();
    for p in &results {
        assert_eq!(
            p.endgame_entered,
            matches!(
                p.status,
                PathStatus::ConvergedSingular { .. } | PathStatus::EndgameFailed
            ),
            "{}: endgame_entered flag out of sync with the path status",
            name
        );
        assert!(
            allow_endgame || !p.endgame_entered,
            "{}: the Cauchy endgame triggered on a regular system",
            name
        );
    }
    // Worst residual over the root endpoints (NaN → no row entry).
    let max_residual = results
        .iter()
        .filter(|p| p.status.is_root())
        .map(|p| residual_inf(system, &p.point))
        .fold(f64::NAN, f64::max);
    let statuses: Vec<PathStatus> = results.iter().map(|p| p.status).collect();

    Row {
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
    }
}

/// One row of the `--enum` naive-vs-DEMiCs comparison table.
struct EnumRow {
    name: &'static str,
    nv: usize,
    /// Naive candidate-tuple count `Π C(|A_i|, 2)`.
    tuples: f64,
    /// Measured naive median, or `None` when only projected.
    naive_ms: Option<f64>,
    /// The cost-model projection for the naive scan, seconds.
    projected_s: f64,
    /// Measured DEMiCs-style tree-search median.
    demics_ms: f64,
    cells: usize,
    mv: u64,
}

/// Measures **enumeration only** (lifting excluded from the timed region;
/// it is microseconds) for both enumerators. The naive scan is measured
/// when its projection fits [`NAIVE_BUDGET_S`], and its output is asserted
/// identical in count and mixed volume to the tree search's — a live
/// oracle check at benchmark scale.
fn enum_compare<const NV: usize, const MAXT: usize>(
    name: &'static str,
    system: &MSystem<c64, NV, NV, MAXT>,
    seed: u64,
    expect_mv: Option<u64>,
    reps: usize,
) -> EnumRow {
    eprintln!("[bench_suite] enumerating {} ...", name);
    let supports = Support::from_msystem(system);
    let tuples: f64 = supports
        .iter()
        .map(|s| {
            let n = s.len() as f64;
            n * (n - 1.0) / 2.0
        })
        .product();
    let projected_s = tuples * ((NV * NV * NV) as f64) * TUPLE_NS_PER_NV3 * 1e-9;

    let liftings = random_liftings::<f64, NV>(&supports, seed);
    let mut demics_ms = Vec::with_capacity(reps);
    let mut cells: Vec<MixedCell<f64, NV>> = Vec::new();
    for _ in 0..reps {
        let t0 = Instant::now();
        cells = mixed_cells(&supports, &liftings).expect("lifting degenerate; change the seed");
        demics_ms.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    let mv: u64 = cells.iter().map(MixedCell::volume).sum();
    if let Some(want) = expect_mv {
        assert_eq!(
            mv, want,
            "{}: computed mixed volume {} != expected {}",
            name, mv, want
        );
    }

    let naive_ms = (projected_s <= NAIVE_BUDGET_S).then(|| {
        let mut times = Vec::with_capacity(reps);
        let mut naive: Vec<MixedCell<f64, NV>> = Vec::new();
        for _ in 0..reps {
            let t0 = Instant::now();
            naive = mixed_cells_naive(&supports, &liftings).expect("naive enumeration failed");
            times.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        assert_eq!(naive.len(), cells.len(), "{}: cell count mismatch", name);
        let naive_mv: u64 = naive.iter().map(MixedCell::volume).sum();
        assert_eq!(naive_mv, mv, "{}: naive/DEMiCs mixed volume mismatch", name);
        median(times)
    });

    EnumRow {
        name,
        nv: NV,
        tuples,
        naive_ms,
        projected_s,
        demics_ms: median(demics_ms),
        cells: cells.len(),
        mv,
    }
}

fn print_enum_table(rows: &[EnumRow]) {
    println!(
        "mixed-cell enumeration: naive tuple scan vs LP-pruned tree search ({} build; medians, same seed/lifting)",
        if cfg!(debug_assertions) {
            "debug -- numbers are meaningless, use --release"
        } else {
            "release"
        },
    );
    println!("{:-<100}", "");
    println!(
        "{:<10} {:>3} {:>9} {:>16} {:>12} {:>9} {:>6} {:>5}",
        "system", "nv", "tuples", "naive ms", "demics ms", "speedup", "cells", "mv",
    );
    println!("{:-<100}", "");
    for r in rows {
        let (naive, speedup) = match r.naive_ms {
            Some(ms) => (format!("{:.3}", ms), format!("{:.1}x", ms / r.demics_ms)),
            None => (format!("~{:.0} s (proj)", r.projected_s), "-".to_string()),
        };
        println!(
            "{:<10} {:>3} {:>9.1e} {:>16} {:>12.3} {:>9} {:>6} {:>5}",
            r.name, r.nv, r.tuples, naive, r.demics_ms, speedup, r.cells, r.mv,
        );
    }
    println!("{:-<100}", "");
    println!("naive column: measured when the Π C(|A_i|,2)·NV³ cost model fits {NAIVE_BUDGET_S:.0} s, else projected.");
}

fn print_table(rows: &[Row]) {
    println!(
        "talrost polyhedral homotopy benchmark suite ({} build; single-threaded f64, RK4 predictor, Cauchy endgame)",
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
    for r in rows {
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
    println!("{:-<120}", "");
    for r in rows {
        if !r.note.is_empty() {
            println!("note: {:<10} {}", r.name, r.note);
        }
    }
}

fn print_csv(rows: &[Row]) {
    println!(
        "system,nv,mixed_volume,cells,paths,converged,offline_ms,tracking_ms,us_per_path,max_residual,failures,note"
    );
    for r in rows {
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
}

/// The `--enum` mode: enumeration-only, naive vs tree search, out to
/// cyclic-8 (which only the tree search reaches; single repetition there —
/// it is the longest row and the point is its order of magnitude).
fn run_enum_compare() {
    let rows = vec![
        enum_compare("trinomial", &trinomial(), 3, Some(2), REPS),
        enum_compare("conic", &conic(), 4, Some(4), REPS),
        enum_compare("cyclic-3", &cyclic::<3, 3>(), 1, Some(6), REPS),
        enum_compare("cyclic-4", &cyclic::<4, 4>(), 1, Some(16), REPS),
        enum_compare("cyclic-5", &cyclic::<5, 5>(), 1, Some(70), REPS),
        enum_compare("cyclic-6", &cyclic::<6, 6>(), 1, Some(156), REPS),
        enum_compare("cyclic-7", &cyclic::<7, 7>(), 1, Some(924), REPS),
        enum_compare("cyclic-8", &cyclic::<8, 8>(), 1, Some(2560), 1),
        enum_compare("katsura-3", &katsura::<4, 5>(), 1, Some(6), REPS),
        enum_compare("katsura-4", &katsura::<5, 6>(), 1, Some(12), REPS),
        enum_compare("noon-3", &noon::<3, 4>(), 1, Some(21), REPS),
        enum_compare("eco-4", &eco::<4, 4>(), 1, Some(4), REPS),
        enum_compare("eco-5", &eco::<5, 5>(), 1, Some(8), REPS),
    ];
    print_enum_table(&rows);
}

fn main() {
    let csv = env::args().any(|a| a == "--csv");
    if env::args().any(|a| a == "--enum") {
        run_enum_compare();
        return;
    }

    // Expected mixed volumes: published values where the literature pins one
    // (cyclic-5: 70, cyclic-6: 156, cyclic-7: 924, noon-3: 21), otherwise
    // the value computed by this solver and cross-checked against the sympy
    // oracle's exact solution counts (see tools/oracle-sympy/README.md and
    // BENCHMARKS.md).
    let rows = vec![
        run_system(
            "trinomial",
            &trinomial(),
            3,
            Some(2),
            false,
            "calibration row; oracle: 2 distinct roots, both found",
        ),
        run_system(
            "conic",
            &conic(),
            4,
            Some(4),
            false,
            "calibration row; oracle: 4 distinct roots, all found",
        ),
        run_system(
            "dbl-root",
            &double_root(),
            1,
            Some(2),
            true,
            "SINGULAR target: the only root (1,1) is double; both paths finish through the \
             Cauchy endgame as conv-singular with winding 2 (endgame cost row)",
        ),
        run_system(
            "cyclic-3",
            &cyclic::<3, 3>(),
            1,
            Some(6),
            false,
            "oracle: 6 distinct torus roots (permutations of the cube roots of unity)",
        ),
        run_system(
            "cyclic-4",
            &cyclic::<4, 4>(),
            1,
            Some(16),
            true,
            "DEGENERATE target (oracle): positive-dimensional solution set (two curves); \
             no isolated roots exist. The Cauchy endgame lands the paths pairwise on genuine \
             curve points (winding 2, residuals ~1e-15) — locally indistinguishable from \
             double roots; classifying them needs witness sets (deferred). MV 16 is \
             seed-invariant (the generic root count of these supports)",
        ),
        run_system(
            "cyclic-5",
            &cyclic::<5, 5>(),
            1,
            Some(70),
            false,
            "published mixed volume / root count 70",
        ),
        run_system(
            "cyclic-6",
            &cyclic::<6, 6>(),
            1,
            Some(156),
            false,
            "published mixed volume / root count 156; past the naive enumeration frontier",
        ),
        run_system(
            "cyclic-7",
            &cyclic::<7, 7>(),
            1,
            Some(924),
            false,
            "published mixed volume / root count 924; naive enumeration projected ~294 s",
        ),
        run_system(
            "katsura-3",
            &katsura::<4, 5>(),
            1,
            Some(6),
            false,
            "oracle: 8 distinct affine roots, exactly 6 on the torus = MV; \
             the 2 off-torus roots are invisible to Bernstein's count",
        ),
        run_system(
            "katsura-4",
            &katsura::<5, 6>(),
            1,
            Some(12),
            false,
            "oracle: 16 distinct affine roots, exactly 12 on the torus = MV",
        ),
        run_system(
            "noon-3",
            &noon::<3, 4>(),
            1,
            Some(21),
            false,
            "oracle: 21 distinct torus roots; published count 21; BKK-exact",
        ),
        run_system(
            "eco-4",
            &eco::<4, 4>(),
            1,
            Some(4),
            false,
            "oracle: 4 distinct roots, all on the torus",
        ),
        run_system(
            "eco-5",
            &eco::<5, 5>(),
            1,
            Some(8),
            false,
            "oracle: 8 distinct roots, all on the torus",
        ),
    ];

    if csv {
        print_csv(&rows);
    } else {
        print_table(&rows);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use talrost::solvers::homotopy::solve;

    /// Solves `system` with the driver and asserts the mixed volume, the
    /// converged-path count, residual < 1e-8 on every converged endpoint —
    /// and, on every path, that the Cauchy endgame **never triggered**
    /// (`endgame_entered` is the tracker's explicit flag): these systems
    /// have only regular isolated roots, so a single endgame entry would
    /// mean the trigger heuristics misfire on healthy paths.
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
        assert!(
            report.paths.iter().all(|p| !p.endgame_entered),
            "{}: the endgame triggered on a regular system",
            name
        );
        assert_eq!(report.singular_count(), 0, "{}: singular paths", name);
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

    /// The double-root calibration row: both paths finish through the
    /// Cauchy endgame with winding 2 on the double root (1, 1) — the
    /// benchmark-side twin of the driver-level endgame tests.
    #[test]
    fn bench_suite_double_root_row() {
        let system = double_root();
        let report = solve(&system, 1, &TrackOptions::default()).unwrap();
        assert_eq!(report.mixed_volume, 2);
        assert_eq!(report.converged_count(), 0);
        assert_eq!(report.singular_count(), 2);
        assert!(report
            .paths
            .iter()
            .all(|p| p.status == PathStatus::ConvergedSingular { winding: 2 }));
        assert_eq!(
            report.multiplicity_of(&[z(1.0), z(1.0)], 1e-4),
            2,
            "the double root must be hit by both paths"
        );
    }

    /// cyclic-4, the suite's **positive-dimensional** negative control.
    ///
    /// The oracle proves its solution set is two curves
    /// `(a, b, −a, −b)` with `ab = ±1` — there are *no isolated roots*, so
    /// no path may report plain `Converged` (that would fabricate a
    /// regular root where none exists). What actually happens, and why it
    /// is the honest outcome:
    ///
    /// - the 16 paths approach the curves pairwise as square-root branches
    ///   and the Cauchy endgame closes each pair's loop at **winding 2**;
    /// - every computed endpoint **genuinely lies on the solution set**
    ///   (the structural oracle below verifies the `(a, b, −a, −b)`,
    ///   `ab = ±1` form to ~1e-8 and the residuals to 1e-10): nothing is
    ///   fabricated — these are true solutions of the system, just not
    ///   isolated ones;
    /// - what the endgame *cannot* decide is isolatedness: a winding-2
    ///   landing on a curve is locally indistinguishable from an isolated
    ///   double root (identical Puiseux data on any circle). Telling them
    ///   apart needs global information — witness sets / a local dimension
    ///   test — which is explicitly deferred (see HOMOTOPY.md). Until
    ///   then, `ConvergedSingular` documents exactly this caveat, and the
    ///   pinned assertions here make any behavior change loud.
    #[test]
    fn bench_suite_cyclic_4_positive_dimensional() {
        let system = cyclic::<4, 4>();
        let report = solve(&system, 1, &TrackOptions::default()).unwrap();
        assert_eq!(report.mixed_volume, 16);
        assert_eq!(report.paths.len(), 16);
        // No isolated roots exist: no path may claim a regular one.
        assert_eq!(report.converged_count(), 0);
        assert_eq!(report.singular_count(), 16);

        for p in &report.paths {
            assert_eq!(p.status, PathStatus::ConvergedSingular { winding: 2 });
            assert!(p.endgame_entered);
            // Verified residual: the endpoint solves the system.
            let r = residual_inf(&system, &p.point);
            assert!(r < 1e-10, "cyclic-4 endpoint residual {:.3e}", r);
            // Structural oracle: the endpoint lies on one of the two
            // curves (a, b, −a, −b), ab = ±1 — i.e. (ab)² = 1.
            let [a, b, c, d] = p.point;
            assert!((c + a).magnitude() < 1e-8, "y2 != -y0");
            assert!((d + b).magnitude() < 1e-8, "y3 != -y1");
            let ab2 = (a * b) * (a * b);
            assert!(
                (ab2 - z(1.0)).magnitude() < 1e-8,
                "(ab)^2 = {} is not 1",
                ab2
            );
        }

        // The pairs land two-on-one-point (winding 2 ⇔ 2 paths per
        // landing): 8 distinct landing points, each of multiplicity 2 —
        // exactly the local double-root picture the caveat is about.
        let distinct = report.distinct_solutions(1e-6);
        assert_eq!(distinct.len(), 8);
        for s in &distinct {
            assert_eq!(report.multiplicity_of(s, 1e-6), 2);
        }
    }
}
