# bench-external — HomotopyContinuation.jl head-to-head harness

`hc_bench.jl` solves the **same systems** as talrost's
`cargo run --release --example bench_suite -- --csv` with
[HomotopyContinuation.jl](https://www.juliahomotopycontinuation.org/)
(Breiding & Timme, *HomotopyContinuation.jl: A package for homotopy
continuation in Julia*, ICMS 2018) and prints the **same CSV schema**, so
the two outputs can be diffed or joined directly.

## This harness has NOT been run in the talrost development container

The container's egress proxy allows only crates.io / PyPI / npm / the Go
module proxy. Both Julia download hosts are blocked — verified at the time
of writing:

```text
$ curl https://pkg.julialang.org
curl: (56) CONNECT tunnel failed, response 403
$ curl https://julialang-s3.julialang.org/bin/versions.json
curl: (56) CONNECT tunnel failed, response 403
```

So neither Julia nor the package could be installed here, and the script is
**syntactically careful but unexecuted**. Run it on an unrestricted machine;
if the HomotopyContinuation.jl API has drifted, fix the script (the talrost
numbers in BENCHMARKS.md do not depend on it).

## Running on an unrestricted machine

1. Install Julia via [juliaup](https://github.com/JuliaLang/juliaup):

   ```sh
   curl -fsSL https://install.julialang.org | sh
   ```

2. Install the package (one-off; from `julia`'s Pkg prompt, entered with
   `]`):

   ```text
   ]add HomotopyContinuation
   ```

   or non-interactively:

   ```sh
   julia -e 'using Pkg; Pkg.add("HomotopyContinuation")'
   ```

3. Run the harness **single-threaded** (talrost's tracker is
   single-threaded; HomotopyContinuation.jl parallelizes across paths by
   default, which would make wall times incomparable):

   ```sh
   JULIA_NUM_THREADS=1 julia hc_bench.jl > hc_results.csv
   ```

4. Produce the matching talrost CSV on the same machine:

   ```sh
   cargo run --release --example bench_suite -- --csv > talrost_results.csv
   ```

## Reading the results honestly

The comparison is **not** apples-to-apples even on one machine — see the
"fairness asymmetry" section of BENCHMARKS.md. In short:

- talrost does *less* work per path: fixed f64, no endgames, no adaptive
  precision, no certification, RK4 + Newton only;
- talrost also has no rescue path: an ill-conditioned stretch that
  HomotopyContinuation.jl survives via adaptive precision / endgames is an
  honest failure (`min-step` / `singular`) for talrost;
- HomotopyContinuation.jl's first `solve` includes JIT compilation — the
  script reports the *second* run, but talrost has no analogous warm-up;
- HomotopyContinuation.jl does not split offline (lift + cells + starts)
  from online (tracking) time, so its `offline_ms`/`cells`/`mixed_volume`
  CSV fields are left empty and `tracking_ms` holds the whole `solve` wall
  time.

`cyclic-7` (root count 924) is included in the Julia list although talrost's
suite excludes it: it lies beyond talrost's current naive-enumeration
frontier (projected ≈ 294 s of cell enumeration; see BENCHMARKS.md), and the
row exists to make that scale gap visible in the joined data.
