# talrost bench-external harness: HomotopyContinuation.jl on the same
# benchmark systems as `cargo run --release --example bench_suite`, printing
# the same line-oriented CSV schema.
#
# *** UNTESTED IN THE TALROST DEVELOPMENT CONTAINER ***
# The container's egress proxy allows only crates.io / PyPI / npm / the Go
# proxy; the Julia CDN (https://julialang-s3.julialang.org and
# https://pkg.julialang.org) answers "CONNECT tunnel failed, response 403",
# so neither Julia nor the package could be installed or run here. The code
# below is kept deliberately simple (@var, System, solve, nsolutions,
# residual checks) and syntactically careful, but it has NOT been executed —
# run it on an unrestricted machine per README.md and treat any API drift
# as a bug in this script, not in the recorded talrost numbers.
#
# Usage (see README.md):
#     JULIA_NUM_THREADS=1 julia hc_bench.jl > hc_results.csv
#
# Schema (identical to `bench_suite -- --csv`):
#     system,nv,mixed_volume,cells,paths,converged,offline_ms,tracking_ms,
#     us_per_path,max_residual,failures,note
# HomotopyContinuation.jl does not split offline/online timing and does not
# expose its mixed-cell count, so `mixed_volume`, `cells` and `offline_ms`
# are left empty; `tracking_ms` is the full `solve` wall time (second run,
# after JIT warm-up) and `paths` is the number of tracked paths.

using HomotopyContinuation
using Printf

"cyclic-n: f_k = Σ_i Π_{j=i}^{i+k-1} x_{j mod n} (k = 1..n-1), Πx - 1."
function cyclic_system(n)
    @var x[1:n]
    eqs = [sum(prod(x[mod(j, n)+1] for j in (i-1):(i+k-2)) for i in 1:n) for k in 1:n-1]
    push!(eqs, prod(x) - 1)
    System(eqs; variables = collect(x))
end

"katsura-n, (n+1)-unknown convention u_0..u_n (root count 2^n)."
function katsura_system(n)
    @var u[1:n+1]  # uu[k] stands for the mathematical u_{k-1}
    uu = collect(u)
    eqs = [
        sum(uu[abs(l)+1] * uu[abs(m - l)+1] for l in -n:n if abs(m - l) <= n) - uu[m+1]
        for m in 0:n-1
    ]
    push!(eqs, uu[1] + 2 * sum(uu[2:end]) - 1)
    System(eqs; variables = uu)
end

"noon-n (Noonburg), coefficient 1.1: x_i·Σ_{j≠i} x_j² − 1.1·x_i + 1."
function noon_system(n)
    @var x[1:n]
    eqs = [x[i] * sum(x[j]^2 for j in 1:n if j != i) - 1.1 * x[i] + 1 for i in 1:n]
    System(eqs; variables = collect(x))
end

"eco-n (Morgan's economics, PHCpack formulation)."
function eco_system(n)
    @var x[1:n]
    eqs = [
        (x[k] + sum((x[i] * x[i+k] for i in 1:n-k-1); init = 0)) * x[n] - k
        for k in 1:n-1
    ]
    push!(eqs, sum(x[1:n-1]) + 1)
    System(eqs; variables = collect(x))
end

"The talrost calibration trinomial pair: 1 − 3x + xy, 2 + y + xy."
function trinomial_system()
    @var x y
    System([1 - 3x + x * y, 2 + y + x * y]; variables = [x, y])
end

"The talrost dense complex conic pair (same fixed coefficients)."
function conic_system()
    @var x y
    f1 = (1.1 + 0.3im) + (-0.7 + 0.9im) * x + (0.5 - 1.3im) * y +
         (2.0 + 0.1im) * x^2 + (-1.4 - 0.8im) * x * y + (0.6 + 1.7im) * y^2
    f2 = (-0.9 + 1.2im) + (1.8 - 0.4im) * x + (0.3 + 0.7im) * y +
         (-1.1 - 1.6im) * x^2 + (0.8 + 0.2im) * x * y + (1.5 - 0.5im) * y^2
    System([f1, f2]; variables = [x, y])
end

"Max ∞-norm residual of F over the found solutions (NaN when none)."
function max_residual(F, sols)
    isempty(sols) && return NaN
    maximum(maximum(abs, F(s)) for s in sols)
end

function run_system(name, F; note = "")
    n = length(variables(F))
    # First solve pays Julia's JIT cost; the second is the reported number.
    # The polyhedral start system is the apples-to-apples choice against
    # talrost's Huber–Sturmfels pipeline.
    result = solve(F; start_system = :polyhedral, show_progress = false)
    t = @elapsed result = solve(F; start_system = :polyhedral, show_progress = false)
    paths = ntracked(result)
    conv = nsolutions(result)
    sols = solutions(result)
    resid = max_residual(F, sols)
    failures = paths - conv == 0 ? "-" : string(paths - conv, " not-converged")
    us_per_path = paths == 0 ? 0.0 : 1e6 * t / paths
    @printf(
        "%s,%d,,,%d,%d,,%.3f,%.2f,%s,%s,%s\n",
        name, n, paths, conv, 1e3 * t, us_per_path,
        isnan(resid) ? "" : @sprintf("%.3e", resid),
        failures, note
    )
end

function main()
    println(
        "system,nv,mixed_volume,cells,paths,converged,offline_ms," *
        "tracking_ms,us_per_path,max_residual,failures,note",
    )
    run_system("trinomial", trinomial_system())
    run_system("conic", conic_system())
    run_system("cyclic-3", cyclic_system(3))
    run_system("cyclic-4", cyclic_system(4); note = "positive-dimensional target")
    run_system("cyclic-5", cyclic_system(5))
    run_system("katsura-3", katsura_system(3))
    run_system("katsura-4", katsura_system(4))
    run_system("noon-3", noon_system(3))
    run_system("eco-4", eco_system(4))
    run_system("eco-5", eco_system(5))
    # Beyond talrost's current enumeration frontier (see BENCHMARKS.md) but
    # easy for HomotopyContinuation.jl — included for scale context:
    run_system("cyclic-7", cyclic_system(7); note = "beyond talrost enumeration frontier")
end

main()
