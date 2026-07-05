# Benchmarks — the polyhedral homotopy solver vs. the standard systems

Phase 10 of the solver work: `examples/bench_suite.rs` runs the named small
systems of the polynomial-systems literature (cyclic-n, katsura-n, noon-n,
eco-n) plus the crate's own calibration pairs end-to-end and prints the
table recorded below. Every expected count is pinned against the **exact
sympy oracle** in [`tools/oracle-sympy/`](tools/oracle-sympy/README.md)
(Gröbner bases over ℚ — quotient dimensions, radicality, torus counts), so
nothing in this file rests on a solver output validating itself.

```sh
cargo run --release --example bench_suite          # the table below
cargo run --release --example bench_suite -- --csv # same data, CSV
```

## 1. Methodology

**Machine** (this container): Intel(R) Xeon(R) Processor @ 2.80GHz
(4 vCPUs), Linux 6.18. Compiler: `rustc 1.98.0-nightly (4c9d2bfe4
2026-07-01)`, `cargo run --release`, default codegen flags (no
`target-cpu=native`, no fat LTO). One process, **one thread**.

**What is measured.** Per system, medians of 3 end-to-end repetitions (the
pipeline is deterministic per seed; repetition only smooths scheduler
noise):

- *offline ms* — lifting, fine mixed-cell enumeration (naive tuple scan
  with the exact SNF singularity gate), cell-homotopy construction, and
  binomial start solutions;
- *track ms* — every path of every cell: RK4 predictor (Davidenko tangent
  solves) + Newton corrector, corrector-informed step control, γ-twisted
  Huber–Sturmfels coefficient paths, terminal Newton polish;
- *max resid* — the largest `‖F(x)‖∞` over all converged endpoints,
  i.e. every reported root is residual-verified against the target system;
- *failures* — non-converged paths by status, never hidden.

**The fairness asymmetry, stated both ways.** talrost does **less work per
path** than the literature systems: fixed `f64` precision, no adaptive
precision, no endgames (singular endpoints and roots at/near infinity are
reported as failures, not resolved), no certification, no multithreading.
That makes its per-path times look good. But it also has **no rescue
machinery**: a path that an adaptive-precision tracker with endgames would
save is an honest `min-step`/`singular`/`diverged` here. Comparisons with
full-featured solvers are therefore *scope* comparisons, not
solver-quality rankings, in both directions.

## 2. Results (recorded from this machine, 2026-07-05)

```text
system      nv   mv  cells  paths  conv  offline ms   track ms   us/path  max resid  failures
----------------------------------------------------------------------------------------------
trinomial    2    2      2      2     2       0.003      0.339     169.4    1.1e-16  -
conic        2    4      2      4     4       0.040      1.736     434.0    5.3e-15  -
cyclic-3     3    6      2      6     6       0.009      1.008     168.0    1.2e-16  -
cyclic-4     4   16      4     16     0       0.163     21.183    1323.9          -  16 min-step
cyclic-5     5   70     14     70    70       7.858    167.105    2387.2    1.4e-15  -
katsura-3    4    6      2      6     6       2.060      6.383    1063.8    2.2e-16  -
katsura-4    5   12      4     12    12     112.838     25.402    2116.9    2.2e-16  -
noon-3       3   21      4     21    21       0.092      8.051     383.4    5.0e-16  -
eco-4        4    4      4      4     4       0.073      2.674     668.5    4.5e-15  -
eco-5        5    8      6      8     8       1.602      6.976     872.0    5.9e-15  -
cyclic-7     7   excluded: ~8.6e7 candidate edge tuples, projected ~294 s enumeration (> 60 s budget)
```

Reading guide, with the oracle verdicts
([details](tools/oracle-sympy/README.md)):

- **cyclic-3** (MV 6), **cyclic-5** (MV 70 — the published value), **noon-3**
  (MV 21 — the published value), **eco-4/eco-5** (4 and 8 roots, all on the
  torus), **trinomial/conic** (2 and 4): all paths converge, every endpoint
  residual-verified, converged counts equal the oracle's exact distinct
  root counts.
- **katsura-3 / katsura-4** settle the naming-convention question: in the
  (n+1)-unknown formulation used here, the oracle proves 2ⁿ distinct affine
  roots (8, 16) of which 2 (resp. 4) lie **off the torus** (some coordinate
  exactly 0). Bernstein's theorem counts torus roots only, so the mixed
  volumes are 6 and 12 — and the tracker converges on exactly 6 and 12
  paths with `~1e-16` residuals. The off-torus roots are structurally
  invisible to a polyhedral homotopy without compactification; that is a
  scope boundary, not a tracking failure.
- **cyclic-4** is the suite's deliberate degenerate row: the oracle proves
  the solution set is **positive-dimensional** (two curves,
  `(a, b, −a, −b)` with `ab = ±1`). The mixed volume of its supports is 16
  (seed-invariant; the root count of a *generic* system with those
  monomials), so 16 paths are tracked — and all 16 end `min-step` near
  `t = 1`, where the Jacobian degenerates on approach to the solution
  curves. With no endgames this is exactly the honest outcome; the row
  stays in the table as a negative control.

## 3. Findings

### 3.1 The suite found a real solver bug (fixed in this phase)

katsura-3 initially failed cell enumeration with `GenericityError` on
*every* seed. Cause: its supports contain doubled simplex points (`2eᵢ`),
which admit candidate edge tuples whose integer level matrix is **exactly
singular over ℤ**. f64 LU rounds its rational elimination multipliers, so
the factorization ended with a ~1e-16 pivot instead of an exact zero,
"solved" to a garbage normal of magnitude ~1e16, and the relative-tolerance
genericity check then saw ties everywhere — a false degeneracy report no
re-lift could escape. Fix: `mixed_cells` now decides tuple singularity
**exactly** (Smith normal form of the integer edge matrix,
`src/solvers/homotopy/cells.rs`) before the floating-point solve;
regression-tested with katsura-3's supports across seeds
(`katsura_3_singular_tuples_are_skipped_exactly`).

### 3.2 The enumeration frontier

Cell enumeration is the naive `Π_i C(|A_i|, 2)` tuple scan (documented as
"correct first, fast later" in `cells.rs`), and it — not path tracking — is
this solver's scaling wall:

- measured per-tuple cost on this machine ≈ 6–10 ns · NV³ (katsura-4's
  135 000 five-variable tuples ≈ 113 ms offline; cyclic-5's 10 000 ≈ 8 ms);
- katsura-4 is the first system where offline (113 ms) dwarfs online
  (25 ms for all 12 paths);
- **cyclic-7** (published root count 924) projects to
  `C(7,2)⁶ ≈ 8.6·10⁷` tuples ≈ **294 s** and is excluded by the suite's
  pre-run 60 s gate — the row is printed with that projection instead of
  numbers. cyclic-6 (`15⁵ ≈ 7.6·10⁵` tuples, ~1.6 s projected) would still
  fit; cyclic-8+ and katsura-6+ do not.

The literature crosses this frontier with dedicated mixed-cell codes
(DEMiCs' dynamic enumeration, MixedVol-style algorithms), which reach
cyclic-10 and beyond; that is the natural Phase 11 work item if larger
systems matter. Path tracking itself is nowhere near its wall: 70 paths of
cyclic-5 in ~167 ms single-threaded, and tracking parallelizes trivially
across paths (which this crate deliberately does not do yet).

### 3.3 Per-path costs

µs/path grows with NV roughly as the LU/Jacobian cost times the step count:
~170 µs (NV = 2–3) → ~1–2.1 ms (NV = 4–5). Residuals stay at 1e-15/1e-16
across the suite — the terminal Newton polish against the exact target
coefficients does its job.

## 4. Literature context (different hardware, different scope)

The standard modern reference point is **HomotopyContinuation.jl**
(P. Breiding, S. Timme, *HomotopyContinuation.jl: A package for homotopy
continuation in Julia*, ICMS 2018, LNCS 10931, pp. 458–465). The same named
families (cyclic-n, katsura-n) are its canonical benchmark systems, and the
paper reports substantial speedups over the older standard packages Bertini
and PHCpack on them — for the concrete figures **see the paper** (no
numbers are restated here: they were measured on different hardware, with
adaptive precision, endgames, and path-level parallelism in scope, none of
which talrost has, and quoting them next to the table above without those
qualifiers would be misleading in talrost's favor or against it depending
on the row).

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
# Tier A: the table and CSV above
cargo run --release --example bench_suite
cargo run --release --example bench_suite -- --csv

# the oracle-verified small-system root counts as a test
cargo test bench_suite_root_counts

# Tier C: the exact ground truth (sympy; see tools/oracle-sympy/README.md)
cd tools/oracle-sympy && python3 -m venv .venv && .venv/bin/pip install sympy
.venv/bin/python oracle.py

# Tier B: the external head-to-head (needs an unrestricted machine)
# see tools/bench-external/README.md
```
