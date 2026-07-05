# Benchmarks — the polyhedral homotopy solver vs. the standard systems

Phase 10 of the solver work built `examples/bench_suite.rs`: the named
small systems of the polynomial-systems literature (cyclic-n, katsura-n,
noon-n, eco-n) plus the crate's own calibration pairs, run end-to-end, with
every expected count pinned against the **exact sympy oracle** in
[`tools/oracle-sympy/`](tools/oracle-sympy/README.md) (Gröbner bases over ℚ
— quotient dimensions, radicality, torus counts), so nothing in this file
rests on a solver output validating itself. Phase 11 replaced the naive
mixed-cell enumeration with a DEMiCs-style LP-pruned tree search
(`src/solvers/homotopy/cells.rs`, `lp.rs`) and moved the suite's frontier
from cyclic-5 to cyclic-7 end-to-end (cyclic-8 enumeration-only).

```sh
cargo run --release --example bench_suite           # the main table below
cargo run --release --example bench_suite -- --csv  # same data, CSV
cargo run --release --example bench_suite -- --enum # naive-vs-DEMiCs table
```

## 1. Methodology

**Machine** (this container): Intel(R) Xeon(R) Processor @ 2.80GHz
(4 vCPUs), Linux 6.18. Compiler: `rustc 1.98.0-nightly (4c9d2bfe4
2026-07-01)`, `cargo run --release`, default codegen flags (no
`target-cpu=native`, no fat LTO). One process, **one thread**.

**What is measured.** Per system, medians of 3 end-to-end repetitions (the
pipeline is deterministic per seed; repetition only smooths scheduler
noise):

- *offline ms* — lifting, fine mixed-cell enumeration (LP-pruned
  DEMiCs-style tree search with the exact SNF singularity gate),
  cell-homotopy construction, and binomial start solutions;
- *track ms* — every path of every cell: RK4 predictor (Davidenko tangent
  solves) + Newton corrector, corrector-informed step control, γ-twisted
  Huber–Sturmfels coefficient paths, terminal Newton polish;
- *max resid* — the largest `‖F(x)‖∞` over all converged endpoints,
  i.e. every reported root is residual-verified against the target system;
- *failures* — non-converged paths by status, never hidden.

**The fairness asymmetry, stated both ways.** talrost does **less work per
path** than the literature systems: fixed `f64` precision, no adaptive
precision, a single fixed-radius Cauchy endgame (Phase 12) as the only
rescue machinery — no power-series endgame, no endgames for roots at/near
infinity, no certification, no multithreading. That makes its per-path
times look good. But a path that an adaptive-precision tracker with the
full endgame arsenal would save can still be an honest
`min-step`/`singular`/`diverged` here. Comparisons with full-featured
solvers are therefore *scope* comparisons, not solver-quality rankings, in
both directions.

## 2. Results (recorded from this machine, 2026-07-05; Phase 12 endgame)

```text
system      nv   mv  cells  paths  conv  offline ms   track ms   us/path  max resid  failures
----------------------------------------------------------------------------------------------
trinomial    2    2      2      2     2       0.006      0.156      78.1    1.1e-16  -
conic        2    4      2      4     4       0.040      0.620     155.1    5.3e-15  -
dbl-root     2    2      2      2     2       0.004      0.286     143.0    5.9e-31  -
cyclic-3     3    6      2      6     6       0.011      0.700     116.7    1.2e-16  -
cyclic-4     4   16      4     16    16       0.118     17.940    1121.3    2.0e-15  -
cyclic-5     5   70     14     70    70       2.364     84.479    1206.8    1.4e-15  -
cyclic-6     6  156     22    156   156      35.234    391.248    2508.0    2.2e-15  -
cyclic-7     7  924    116    924   924     689.120   4099.401    4436.6    6.2e-15  -
katsura-3    4    6      2      6     6       0.797      3.678     613.0    2.2e-16  -
katsura-4    5   12      4     12    12       7.499     14.850    1237.5    2.2e-16  -
noon-3       3   21      4     21    21       0.106      5.420     258.1    5.0e-16  -
eco-4        4    4      4      4     4       0.097      1.797     449.1    4.5e-15  -
eco-5        5    8      6      8     8       0.874      4.468     558.5    5.9e-15  -
```

(`conv` counts root-bearing paths: plain converged plus Cauchy-endgame
singular endpoints. The `dbl-root` and `cyclic-4` rows are the endgame
rows — see below; every other row is asserted endgame-free at run time.)

Whole suite (3 repetitions of everything, tracking included): **~16 s**.

Reading guide, with the oracle verdicts
([details](tools/oracle-sympy/README.md)):

- **cyclic-3** (MV 6), **cyclic-5** (MV 70 — the published value),
  **cyclic-6** (MV 156), **cyclic-7** (MV 924 — both published values;
  **all 156 and all 924 paths converge** with residuals ≤ 6.2e-15),
  **noon-3** (MV 21 — the published value), **eco-4/eco-5** (4 and 8
  roots, all on the torus), **trinomial/conic** (2 and 4): all paths
  converge, every endpoint residual-verified, converged counts equal the
  published/oracle-verified exact root counts.
- **katsura-3 / katsura-4** settle the naming-convention question: in the
  (n+1)-unknown formulation used here, the oracle proves 2ⁿ distinct affine
  roots (8, 16) of which 2 (resp. 4) lie **off the torus** (some coordinate
  exactly 0). Bernstein's theorem counts torus roots only, so the mixed
  volumes are 6 and 12 — and the tracker converges on exactly 6 and 12
  paths with `~1e-16` residuals. The off-torus roots are structurally
  invisible to a polyhedral homotopy without compactification; that is a
  scope boundary, not a tracking failure.
- **dbl-root** (`{x² − 2x + 1, y − x}`, added in Phase 12) is the
  Cauchy-endgame cost row: its only root (1, 1) is **double**, both paths
  finish `conv-singular` with winding 2, and the endpoint lands ~1e-15
  from the true root (plain Newton stalls at the `√ε ≈ 1e-8` attainable
  accuracy for a double root — the 5.9e-31 residual is `(x−1)²` at
  `x − 1 ≈ 1e-15`). The per-path cost (~2× the regular trinomial row)
  shows the price of the walk-out plus two closure loops.
- **cyclic-4** is the suite's deliberate degenerate row: the oracle proves
  the solution set is **positive-dimensional** (two curves,
  `(a, b, −a, −b)` with `ab = ±1`). The mixed volume of its supports is 16
  (seed-invariant; the root count of a *generic* system with those
  monomials), so 16 paths are tracked. Since Phase 12 the Cauchy endgame
  finishes all 16 pairwise at **winding 2** on points that genuinely lie
  on the curves (the pinned test verifies the `(a, b, −a, −b)`, `ab = ±1`
  form and 1e-15 residuals): nothing is fabricated, but winding certifies
  local branch structure, **not isolatedness** — a winding-2 landing on a
  curve is locally indistinguishable from an isolated double root, and
  separating the two needs witness sets (deferred). The row stays in the
  table as the positive-dimensional control.

## 3. Findings

### 3.1 Naive scan vs. LP-pruned tree search (Phase 11), measured side by side

Enumeration only (`--enum`; same seed, same lifting, medians; "naive" =
`mixed_cells_naive`, the retained reference implementation; "demics" =
`mixed_cells`, the production LP-pruned search; both measured fresh on this
machine, 2026-07-05):

```text
system      nv    tuples         naive ms    demics ms   speedup  cells    mv
------------------------------------------------------------------------------
trinomial    2     9.0e0            0.001        0.004      0.2x      2     2
conic        2     2.2e2            0.022        0.047      0.5x      2     4
cyclic-3     3     9.0e0            0.002        0.008      0.2x      2     6
cyclic-4     4     2.2e2            0.060        0.091      0.7x      4    16
cyclic-5     5     1.0e4            5.148        2.285      2.3x     14    70
cyclic-6     6     7.6e5          557.643       36.164     15.4x     22   156
cyclic-7     7     8.6e7    ~294 s (proj)      698.671         -    116   924
cyclic-8     8    1.3e10  ~69084 s (proj)    10443.986         -    266  2560
katsura-3    4     3.6e3            1.415        0.774      1.8x      2     6
katsura-4    5     1.4e5           73.922        7.891      9.4x      4    12
noon-3       3     2.2e2            0.051        0.070      0.7x      4    21
eco-4        4     1.1e2            0.029        0.069      0.4x      4     4
eco-5        5     1.8e3            0.963        0.629      1.5x      6     8
```

The pattern is exactly the theory: on trivially small systems (≤ a few
hundred candidate tuples) the LP setup overhead makes the tree search a
fraction of a *millisecond* slower — irrelevant — while every system past
~10⁴ tuples flips hard the other way, because one pruned interior node
deletes an entire `Π C(|A_j|, 2)` block of tuples. katsura-4's end-to-end
offline column collapsed from 113 ms (Phase 10, naive) to ~8 ms; cyclic-7
went from a ~294 s projection (excluded from the Phase 10 suite) to 0.7 s
measured; cyclic-8 from a ~19 *hour* projection to 10.4 s. The naive and
tree-search cell sets are asserted identical (count + mixed volume in the
`--enum` run itself; full edge/matrix/normal/volume equality in
`tests/cells_oracle.rs`).

### 3.2 The enumeration frontier, relocated

Phase 10's wall was the naive `Π C(|A_i|, 2)` scan: cyclic-7 was excluded
(~294 s projected) and katsura-4's offline time (113 ms) dwarfed its
tracking (25 ms). Phase 11's LP-pruned search moves the wall out by roughly
three orders of magnitude:

- **cyclic-7** now runs end-to-end in the suite: 0.70 s enumeration,
  4.0 s tracking of all 924 paths — offline is again a minority of the
  total, and no row in the suite has offline > tracking.
- **cyclic-8** (MV 2560, 266 fine cells) enumerates in **10.4 s**,
  measured — against a ~19 hour naive projection. It stays out of the
  default suite only because tracking 2560 paths at NV = 8 would multiply
  the suite's wall time for no additional verification value (`--enum`
  covers it).
- The projected next wall: each additional cyclic level multiplies the
  edge choices per support by ~1.3× and deepens every LP, and the measured
  growth (cyclic-6 → 7 → 8: 36 ms → 0.70 s → 10.4 s, ~15–19× per level)
  puts *this implementation's* cyclic-9 enumeration in the ~3-minute range
  and cyclic-10 near an hour. The remaining gap to DEMiCs itself (which
  reaches cyclic-10+ in seconds) is dynamic support re-ordering, one-point
  relation tables, and warm-started LPs — noted as future refinements in
  `cells.rs`; the from-scratch dense two-phase simplex per node is this
  implementation's dominant cost.

Tracking remains nowhere near its wall (~4.4 ms/path at NV = 7,
single-threaded, trivially parallelizable across paths — deliberately not
done yet).

### 3.3 The Phase-10 exact-singularity gate still does its job

katsura-3/4's doubled support points admit candidate edge tuples whose
integer level matrix is **exactly singular over ℤ**; f64 LU sees them as
~1e-16-pivot "solvable" and produces garbage normals that falsely trip the
genericity check (the Phase 10 bug). The gate — Smith normal form of the
integer edge matrix before any float solve — is unchanged in Phase 11 and
sits in the one shared per-tuple decision procedure both enumerators call,
so the tree search inherits the fix verbatim
(`katsura_3_singular_tuples_are_skipped_exactly` plus the naive-equality
oracle tests cover it).

### 3.4 Per-path costs

µs/path grows with NV roughly as the LU/Jacobian cost times the step
count: ~80–170 µs (NV = 2–3) → ~1.2 ms (NV = 5) → ~2.5 ms (NV = 6) →
~4.4 ms (NV = 7). Residuals stay at 1e-15/1e-16 across the suite — the
terminal Newton polish against the exact target coefficients does its job.

## 4. Literature context (different hardware, different scope)

The standard modern reference point is **HomotopyContinuation.jl**
(P. Breiding, S. Timme, *HomotopyContinuation.jl: A package for homotopy
continuation in Julia*, ICMS 2018, LNCS 10931, pp. 458–465). The same named
families (cyclic-n, katsura-n) are its canonical benchmark systems, and the
paper reports substantial speedups over the older standard packages Bertini
and PHCpack on them — for the concrete figures **see the paper** (no
numbers are restated here: they were measured on different hardware, with
adaptive precision, a full endgame arsenal, and path-level parallelism in
scope — talrost has only the fixed-precision tracker plus the Phase 12
Cauchy endgame — and quoting them next to the table above without those
qualifiers would be misleading in talrost's favor or against it depending
on the row).

The mixed-cell enumeration is a simplified variant of **DEMiCs**
(T. Mizutani, A. Takeda, M. Kojima, *Dynamic enumeration of all mixed
cells*, Discrete Comput. Geom. 37, 2007): the same one-edge-per-support
LP-pruned tree search, without the paper's dynamic ordering and relation
tables (see `src/solvers/homotopy/cells.rs` for what is and isn't
implemented).

For a same-hardware comparison, run the untested-here harness in
[`tools/bench-external/`](tools/bench-external/README.md): it solves the
identical systems with HomotopyContinuation.jl (`start_system =
:polyhedral`, single-threaded, JIT-warmed) and emits the same CSV schema as
`bench_suite -- --csv`, so the outputs join directly. It could not be
executed in this development container — the egress proxy 403-blocks the
Julia CDN (`julialang-s3.julialang.org`, `pkg.julialang.org`); the README
there records the exact failure and the install/run commands for an
unrestricted machine.

## 5. Reproducing

```sh
# Tier A: the tables and CSV above
cargo run --release --example bench_suite
cargo run --release --example bench_suite -- --csv
cargo run --release --example bench_suite -- --enum

# the oracle-verified small-system root counts as a test
cargo test bench_suite_root_counts

# naive-vs-DEMiCs identical-cell-set oracle tests
cargo test --test cells_oracle

# Tier C: the exact ground truth (sympy; see tools/oracle-sympy/README.md)
cd tools/oracle-sympy && python3 -m venv .venv && .venv/bin/pip install sympy
.venv/bin/python oracle.py

# Tier B: the external head-to-head (needs an unrestricted machine)
# see tools/bench-external/README.md
```
