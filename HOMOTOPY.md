# Homotopy tower — implementation notes (Phases 5–6)

*Design target: a polyhedral homotopy continuation solver (Huber–Sturmfels) built on talrost.
These notes turn the design discussion into an implementable spec. Phase 5 is the tower
refactor; Phase 6 is polish/CI/docs and documents the finished API.*

## Context: what the solver consumes

Pipeline: **(offline)** from the support sets A_i ⊂ ℤⁿ of a sparse polynomial system,
compute a mixed subdivision induced by a generic lifting, extract mixed cells, and solve one
binomial start system per cell via Smith normal form of its exponent matrix; **(online)**
track each start solution along H(x, t) through ℂⁿ with a predictor–corrector loop — the
Newton corrector needs ∂H/∂x (an n×n complex Jacobian) and a linear solve per iteration;
the predictor needs ∂H/∂t. The offline result depends only on monomial structure, never on
coefficient values, so for embedded targets the cells/start data can be baked at compile
time and the device only runs the tracker. The tower must make both halves expressible with
stack-only, monomorphized code; the solver layer itself (mixed-volume LP, cell enumeration,
tracker, endgames) is NOT part of these phases.

## Phase 5 — tower refactor

### 5.1 Flip univariate `Polynomial` to ascending coefficient order  *(approved)*

`c[i]` = coefficient of `x^i`. Horner folds from the top coefficient down
(`iter().rev()`); codegen is identical (fully unrolls). Display still prints
highest-degree-first (presentation is independent of storage). Update Blinn/Yuksel
coefficient indexing, all constructors/tests/demo. Document the convention loudly on the
type. Rationale: index meaning becomes independent of `N`, matching the multivariate types
(5.8) where ascending exponent semantics is the only sane choice, and making
derivative/deflation index math trivial.

### 5.2 `Algebra<T>` — the single evaluation abstraction

```rust
pub trait Algebra<T: Ring>: Ring + Mul<T, Output = Self> + Add<T, Output = Self> + From<T> {}
impl<T: Ring> Algebra<T> for T {}
impl<F: Real> Algebra<F> for Complex<F> {}
// Dual/DualN instances in 5.3
```

(No overlap: the blanket gives `Complex<F>: Algebra<Complex<F>>`; the explicit impl gives
`Complex<F>: Algebra<F>` — different trait parameterizations.)

`Polynomial::eval_at<X: Algebra<T>>(&self, x: X) -> X` — one Horner body (plain mul+add)
covering: plain eval (X = T), real coefficients at complex points (Aberth/Durand–Kerner,
γ-trick), derivatives via duals, and later matrices. The monomorphic `eval()` keeps its
`mul_add_fast` fast path; document the relationship.

### 5.3 Dual numbers — forward AD as ring elements (new src/dual.rs)

```rust
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Dual<T>  { pub val: T, pub der: T }                 // a + a′ε, ε² = 0
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct DualN<T, const K: usize> { pub val: T, pub der: [T; K] } // vector-mode forward AD
```

- Constructors: `Dual::variable(x)` (der = ONE), `Dual::constant(x)`;
  `DualN::variable(x, k)` (der = e_k), `DualN::constant(x)`.
- Tower impls: Element, Monoid, Group, Semiring (ONE = (ONE, ZERO)), Ring for `T: Ring`;
  Field for `T: Field` with `recip = (1/v, −d/v²)` — document that division requires a
  nonzero standard part (same pragmatism as floats vs. the Field axioms).
- Mixed-scalar ops: `Mul<T>/Add<T>/Sub<T>/Div<T> for Dual<T>` (coherence: `T` cannot unify
  with `Dual<T>` — occurs check — so these coexist with the `Self`-RHS ops; if the compiler
  disagrees, fall back to concrete impls via macro). Targeted nested-composition impls so
  `Dual<Complex<F>>: Algebra<Complex<F>>` and (via explicit `Mul<F>/Add<F>/From<F>` impls)
  `Dual<Complex<F>>: Algebra<F>`.
- `Algebra` instances: `Dual<T>: Algebra<T>`, `DualN<T, K>: Algebra<T>`.
- Chain-rule lifts on `Dual<F: Real>` as inherent methods (NOT a full Real impl): sqrt,
  sin, cos, sin_cos, tan, exp, ln, powi, mul_add. DualN likewise where cheap.
- Tests: ring/field identities; `p.eval_at(Dual::variable(x))` equals the analytic
  derivative for a known cubic; DualN Jacobian of a small 2-variable system vs
  hand-computed; one nested `Dual<Complex<f64>>` case.

Point of the design: every generic routine in the crate differentiates itself.
`DualN<Complex<f64>, NV>` gives a Jacobian row per equation in one sweep (Newton
corrector); the same `Dual` in the t-slot gives ∂H/∂t (predictor). Structural derivatives
(5.8) are the fast path; duals are the general mechanism.

### 5.4 Real/Complex completeness

- `Real` additions: consts PI, TAU, E; methods exp, ln, powf, signum, min, max, clamp
  (std + libm backends, matching the Phase 4 feature structure).
- `Complex` additions: exp, ln, arg, powf(F), from_polar(r, θ), by-value powi via
  exponentiation-by-squaring (deprecate/remove the &self loop version),
  roots-of-unity helper (`nth_root_of_unity(k, n)` or a no_std-friendly iterator) — start
  solutions of binomial systems are radius-scaled roots of unity.
- Replace textbook complex division (re²+im² denominator) with **Smith's algorithm** in
  Div/DivAssign and the f32/f64-LHS impls — the textbook form overflows/underflows exactly
  in the near-singular regimes path tracking visits. Tests: exp(iπ) ≈ −1, ln∘exp
  round-trip, division at 1e±300 magnitudes where the textbook form fails.

### 5.5 LU factorization as the linear-solve primitive

```rust
pub struct Lu<T: Scalar, const N: usize> { /* packed LU, pivot vec, permutation sign */ }
impl<T: Scalar, const N: usize> Matrix<T, N, N> {
    pub fn lu(&self) -> Option<Lu<T, N>>;                     // partial pivoting by norm_sqr
    pub fn solve(&self, b: &Vector<T, N>) -> Option<Vector<T, N>>;
}
impl Lu { pub fn solve(&self, b: &Vector<T, N>) -> Vector<T, N>; pub fn determinant(&self) -> T; }
```

Reimplement `determinant` (N>3 arm) and `inverse` on top of `Lu` so there is one pivoting
code path. The corrector solves J·Δx = −H every iteration; `inverse()` is the wrong
primitive. Tests: real+complex solves vs known solutions, singular → None, determinant
consistency with the closed forms.

### 5.6 Ring-relaxed containers + the `lattice` module

- Split Matrix/Vector impl blocks: construction, ZERO/IDENTITY, add/sub/neg, scalar mul,
  transpose, and matrix multiply need only `T: Ring` (matmul accumulation uses plain
  mul+add in the Ring impl; the Scalar/specialization kernels keep mul_add_fast).
  Norms, determinant, inverse, lu/solve stay `T: Scalar`. Motivation: exponent matrices are
  integer matrices — currently unrepresentable because Matrix demands a Field.
- New src/lattice.rs (the module stub already exists commented-out in lib.rs): Smith normal
  form and Hermite normal form over `Matrix<i64, M, N>` returning (U, S, V) with unimodular
  U, V — the closed-form solve of binomial start systems needs SNF of the cell's exponent
  matrix. Keep the algorithm simple (gcd row/col reduction); document entry-growth caveats.
  Tests: known SNFs, U·A·V == S, |det U| == |det V| == 1.

### 5.7 `eval_with_error` (univariate, T: Real)

Horner with Higham's running error bound: alongside the value, accumulate
μ_k = μ_{k−1}·|x| + |acc_k|; return `(value, bound)` with bound ≈ (2·deg+1)·u·μ (u = half
EPSILON). This turns solver stopping rules from arbitrary `tol` into "|p(x)| is below its
own evaluation noise". Test: bound dominates the true error on an ill-conditioned case
(e.g. expanded (x−1)^6 near x = 1, f32 eval vs f64 reference).

### 5.8 Multivariate sparse polynomials (new src/mvpoly.rs)

```rust
pub struct Monomial<const NV: usize> { pub exps: [i32; NV] }   // i32: Laurent after torus transforms
pub struct MPoly<T, const NV: usize, const TERMS: usize> {
    pub coeffs: [T; TERMS],
    pub support: [Monomial<NV>; TERMS],
}
pub struct MSystem<T, const NV: usize, const NEQ: usize, const MAXT: usize> {
    pub polys: [MPoly<T, NV, MAXT>; NEQ],                      // rows padded to MAXT with zero coeffs
}
```

- Padding to MAXT is the deliberate answer to heterogeneous term counts vs const generics
  (small waste, uniform type, stack-only).
- `MPoly::eval_at<X: Algebra<T>>(&self, x: &[X; NV]) -> X` — per-term coeff·∏xᵢ^eᵢ with
  exponentiation-by-squaring; `debug_assert!(exps >= 0)` in plain eval (negative exponents
  arrive only after torus transforms, which live in the solver layer; a Field-bounded
  Laurent eval can come later).
- `partial(&self, j) -> Self` — structural derivative: coeff·eⱼ, eⱼ−1; term count is
  preserved under padding, so the type doesn't change. Zero-coefficient terms are the
  padding mechanism working as intended.
- `eval_grad(&self, x: &[T; NV]) -> (T, [T; NV])` via `DualN<T, NV>`;
  `MSystem::eval(&self, x: &[X; NV]) -> [X; NEQ]` and
  `eval_jacobian(&self, x) -> ([T; NEQ], Matrix<T, NEQ, NV>)`.
- Display via direct `write!` (no alloc). Tests: known 2-var system values, structural
  partial ≡ DualN gradient, padded-row behavior.

### Phase 5 acceptance

The full Phase 4 feature matrix stays green with zero warnings: `cargo test` (pinned
nightly), `rustup run stable cargo test`, `cargo test --features specialization`,
`cargo build --no-default-features --features libm` (and with specialization). Demo gains a
short dual-number/eval_at showcase.

## Phase 6 — polish, tests, CI, docs (absorbs the old "Phase 5" list)

1. Container API round-out: `Vector::dot` (hermitian: Σ aᵢ·conj(bᵢ) — document the
   convention; makes dot(v,v) == |v|² real for complex), 3-D `cross` returning a vector
   (keep the 2-D scalar cross), Index/IndexMut, Default, From<[T; N]>, scalar Div, Neg;
   Matrix Index/IndexMut, Sub/Neg, `Mul<Vector<T, N>> -> Vector<T, M>`; scalar×container
   impls stamped by macro for f32/f64/c32/c64 (coherence blocks the blanket form).
2. Algebra law-check macro (test-only): identities, associativity/commutativity/
   distributivity on fixed samples, x + (−x) == ZERO, x·recip(x) ≈ ONE; instantiate for
   u32, i32, f32, f64, c32, c64, Dual<f64>.
3. Property/oracle tests: dev-deps proptest + num-complex + nalgebra if fetchable through
   the proxy (else a small LCG fallback with self-checkable oracles): complex ops vs
   num-complex incl. Smith-division extremes; matmul/determinant/inverse vs nalgebra;
   |p(root)| small for solver outputs; LU solve residuals; Dual derivatives vs central
   finite differences; SNF unimodularity.
4. Codegen guard: a script (tools/check_codegen.sh) that compiles a #[no_mangle] probe
   using Polynomial::eval/eval_at and asserts no `call` instruction lands in the emitted
   asm on x86-64, with and without `-C target-feature=+fma` (this is what would have caught
   the 5.7× mul_add libm cliff automatically). Wire into CI as an advisory job.
5. Docs: `#![warn(missing_docs)]`, `#![forbid(unsafe_code)]`, doc comments on all public
   items, crate-level example oriented toward the homotopy use case (eval_at + DualN
   Jacobian teaser). `cargo doc --no-deps` warning-free.
6. CI (.github/workflows/ci.yml): fmt --check, clippy -D warnings, stable + pinned-nightly
   test lanes, feature matrix (default / no-default+libm / specialization), thumbv7em
   no_std build if the target installs, codegen guard job.
7. README rewrite: the Real/Scalar/Algebra tower, Dual/DualN, Roots, ascending coefficient
   convention, row-major Matrix, LU, lattice, feature flags, and a roadmap section naming
   the polyhedral homotopy goal. Honest experimental-status disclaimer stays.
8. Micro-benchmarks as ignored tests or an example (std::time, no criterion dependency):
   Horner eval, matmul naive-vs-kernels, a mock corrector step (eval_jacobian + lu + solve).

## Status (Phases 7–9) — the solver itself

The solver layer that Phases 5–6 declared out of scope now exists in
`solvers::homotopy` (std-only; the tower underneath stays `no_std`-clean).

**Implemented:**

- Phase 7 (offline): `Support`/`Lifting` (seeded LCG liftings,
  reproducible), naive fine mixed-cell enumeration with an explicit
  genericity check (`GenericityError`, never a guessed subdivision), exact
  cell volumes and `mixed_volume` via Smith normal form, and closed-form
  binomial start solutions (`binomial_solutions`/`start_solutions`).
- Phase 8 (online): `CellHomotopy` (per-cell levels `e_{i,a}`, normalized by
  the global `t ↦ t^(1/e_min)` reparametrization so the smallest positive
  level is 1 — raw lifted levels are too stiff to track), the
  allocation-free predictor–corrector `track_path` (Euler tangent + Newton,
  corrector-only first step to dodge the `t^{e−1}` singularity at `t = 0`,
  step doubling/halving, honest `PathStatus` reporting), and the `solve()`
  driver (`SolveReport` with raw paths, `solutions()`,
  `distinct_solutions(tol)`, one automatic re-lift on a degenerate lifting).
- Phase 9 (polish):
  - **Predictors** — `TrackOptions::predictor` selects `Predictor::Euler`,
    `::Rk2` (midpoint), or `::Rk4` (classical), each stage a fresh
    Jacobian + LU tangent solve of the Davidenko ODE. RK4 is the
    benchmarked default (~20× fewer steps and ~8× less wall time than
    Euler on the conic pair and cyclic-3; table in the
    `Predictor::default` doc comment). `TrackOptions` deliberately stays
    exhaustive (documented on the type): pre-1.0, field additions are an
    accepted breaking change and `..Default::default()` construction stays
    available to callers.
  - **Corrector-informed step control** — accepted steps adapt `dt` by the
    observed Newton effort (1 iteration → `×grow`; 2 → hold; converged on
    the `max_newton`-th → `×0.8`), rejections still halve; documented on
    `TrackOptions`, with a no-regression test against the Phase 8 rule's
    measured 419 total conic steps (Phase 9 defaults: 186).
  - **The γ-twist** — the Phase 9 headline finding: cyclic-3 (real,
    symmetric, maximally non-generic coefficients) folds on the
    discriminant mid-path — the textbook coefficient paths `c·tᵉ` never
    leave the real slice, where the discriminant has real codimension 1,
    so all six paths died pairwise (conjugate collisions) at one interior
    `t` for every seed tried. Fix: every non-edge term is rotated by the
    endpoint-preserving phase `exp(iγe(1−t))` (a homotopy-level gamma
    trick; `γ = ln 2` fixed for reproducibility,
    `CellHomotopy::with_gamma` for explicit control, `γ = 0` = textbook).
    With the twist, cyclic-3 tracks 6/6 on every seed and predictor.
  - **Diagnostics** — `Lu::pivot_ratio()` (min/max pivot-norm ratio,
    documented as a singularity-proximity hint, not a condition number)
    surfaces as `PathResult::pivot_ratio` from the final polished Newton
    solve; `SolveReport` gains `converged_count()`, `failed_paths()`,
    `real_solutions(tol)` (filtering, never zeroing imaginary parts), and
    allocation-free `Display` impls for `PathStatus` and `SolveReport`.
  - **Validation** — cyclic-3 end-to-end with the structural oracle (every
    solution a permutation of `(1, ω, ω̄)`, |coord| = 1, sum = 0,
    product = 1); the trinomial pair end-to-end over `Complex<f32>`
    (loosened tolerances — the solver is genuinely `Real`-generic); a
    proptest lane (`tests/homotopy_prop.rs`, 32 cases) with random complex
    coefficients on the fixed trinomial supports asserting MV = 2 and
    verified residuals on every converged path.

**Deferred:**

- endgames (singular endpoints, roots at infinity — such paths currently
  just report `MinStepReached`/`SingularJacobian`/`Diverged`);
- non-fine mixed cells (cells with more than two points per support);
- torus transforms / Laurent tracking (the `i32` exponents and
  `Ring`-relaxed containers are ready for them);
- parameter homotopies and coefficient-path (cheater's) homotopies;
- baked offline data for `no_std` targets (the tracker is already
  allocation-free by construction, so only the gating moves).
