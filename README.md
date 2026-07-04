# talrost — a mathematics library for embedded scientific computation

_Note: This library is experimental and in no way ready for even limited use.
Use [num-traits](https://crates.io/crates/num-traits),
[num-complex](https://crates.io/crates/num-complex), and
[nalgebra](https://crates.io/crates/nalgebra) if you need a proper math
library._

Talrost is a stack-only, allocation-free numerical tower being built toward a
**polyhedral homotopy continuation solver** for sparse polynomial systems.
Every type is `Copy`, every size is a const generic, there is no `unsafe`
(`#![forbid(unsafe_code)]`), and the whole crate — containers, solvers,
automatic differentiation — compiles for `no_std` embedded targets such as
`thumbv7em-none-eabihf`.

Design documents: [REVIEW.md](REVIEW.md) (the architecture review that set
the direction) and [HOMOTOPY.md](HOMOTOPY.md) (the implementation notes for
the homotopy tower).

All code snippets below are taken from [`examples/demo.rs`](examples/demo.rs)
and the test suite; run `cargo run --example demo` to execute them.

## The numerical tower

Algebraic structure comes first, as a tower of traits in `talrost::algebra`
(rooted at `talrost::element::Element`):

```text
Element  →  Monoid (+, ZERO)  →  Group (−)  →  Semiring (×, ONE)
         →  Ring  →  Field (÷, recip)
```

Concrete families implement exactly the prefix they honestly satisfy:

| Family | Types | Reaches |
|---|---|---|
| `natural::Natural` | `u8`…`u128`, `usize` | `Semiring` (no `−`) |
| `integer::Integer` | `i8`…`i128`, `isize` | `Ring` (no `÷`) |
| `real::Real` | `f32`, `f64` | `Field`, ordered, IEEE-754 consts + libm/std math |
| `complex::Complex<F>` | `c32`, `c64` | `Field` (deliberately **not** `Real`: ℂ has no order) |
| `dual::Dual`/`DualN` | over any `Ring`/`Field` | `Ring`/`Field` |

Two cross-cutting traits complete the picture:

- `scalar::Scalar` — a field with a *real-valued* norm (`norm_sqr`, `norm`,
  `conj`, `mul_add`/`mul_add_fast`). Implemented by every `Real` type (with
  `Scalar::Real = Self`) and by `Complex<F>` (with `Scalar::Real = F`), so a
  complex vector's magnitude is an `f64`, not a complex number.
- `algebra::Algebra<T>` — the single evaluation abstraction: an algebra over
  the ring `T`. One generic Horner/term loop evaluates `T`-coefficient
  polynomials at plain points (`X = T`), complex points of real polynomials,
  or dual numbers (derivatives). Generic code bounds on the weakest structure
  it needs; integer matrices, for example, are first-class because matrix
  multiplication only demands `Ring`.

## Complex arithmetic

`c32`/`c64` with **Smith's division algorithm** instead of the textbook
`(c² + d²)`-denominator form — quotients stay accurate at the `1e±300`
magnitudes near-singular path tracking visits — plus the polar toolkit
(`from_polar`, `arg`, `exp`, `ln`, `powf`, `powi`, `nth_root_of_unity`) that
generates binomial start solutions:

```rust
use talrost::complex::c64;

let a: f64 = 0.25;
let mut b: c64 = (0.75, 1.0).into();
b += a;
b /= c64::new(0.5, 0.5);
assert_eq!(b, "2 + 0i".parse().unwrap());

// Division goes through Smith's algorithm: it stays exact where the
// textbook (re² + im²) form overflows to garbage.
let big = c64::new(1e300, 1e300);
assert_eq!(big / big, c64::new(1.0, 0.0));

// Polar helpers: binomial start solutions are radius-scaled roots of unity.
let w = c64::nth_root_of_unity(1, 4); // e^(2πi/4) = i
assert!((w - c64::new(0.0, 1.0)).magnitude() < 1e-15);
```

## Polynomials

> **⚠️ Coefficient order changed in 0.2.0:** `Polynomial<T, N>` now stores
> coefficients in **ascending** power order — `c[i]` is the coefficient of
> `x^i`, so `Polynomial::new([0.0, -14.0, 5.0, 1.0])` is
> `x³ + 5x² − 14x + 0`. The index of a coefficient names its power
> independently of `N`, matching the multivariate types. `Display` still
> prints highest-degree-first.

Three evaluators, one convention:

- `eval(x)` — monomorphic Horner with `mul_add_fast` (hardware FMA where the
  target has it, plain mul+add elsewhere — never a software-fma libm call);
- `eval_at(x)` — the generic `Algebra<T>` Horner body: complex points, dual
  numbers, anything that is an algebra over the coefficient ring;
- `eval_with_error(x)` — Horner plus Higham's running error bound, so a
  solver can stop when `|p(x)|` is below its own evaluation noise instead of
  below an arbitrary tolerance.

Root finding for degrees 1–4 returns `Roots`, a counted, ascending prefix of
a fixed array (no NaN sentinels leaking into user code):

```rust
use talrost::polynomial::Polynomial;

let tol = f64::EPSILON;

// p(x) = x^3 + 5x^2 - 14x, has roots -7, 0, 2.
// Coefficients are stored ascending: c[i] multiplies x^i.
let p = Polynomial::new([0.0, -14.0, 5.0, 1.0]);

let y = p.eval(4.0);
assert_eq!(y, 88.0); // p(4) = 88

// Roots contract: ascending order, len() == number of real roots found.
let r = p.roots(tol);
assert_eq!(r.len(), 3);
assert_eq!(r.as_slice(), &[-7.0, 0.0, 2.0]);
assert_eq!(r[0], -7.0); // indexing works via Deref

// A quadratic with complex roots has no real roots at all.
let q = Polynomial::new([1.0, -3.0, 5.0]);
assert!(q.roots(tol).is_empty());
```

The solver modules (`solvers::blinn`, `solvers::yuksel` — Blinn's
homogeneous closed forms and Cem Yuksel's HPG 2022 finder, both generic over
`Real`) remain available with their native output conventions.

## Automatic differentiation

`Dual<T>` (one derivative slot) and `DualN<T, K>` (K slots, vector mode) are
ring elements: seed a variable, run any generic computation, read the
derivative. No symbolic work, no truncation error.

```rust
use talrost::{dual::Dual, polynomial::Polynomial};

// p(x) = x^3 - 2x + 5, ascending storage; p'(x) = 3x^2 - 2.
let p = Polynomial::new([5.0, -2.0, 0.0, 1.0]);
let d = p.eval_at(Dual::variable(2.0));
assert_eq!(d.val, p.eval(2.0));
assert_eq!(d.der, 10.0);
```

Chain-rule lifts (`sqrt`, `sin_cos`, `exp`, `ln`, `powi`, `mul_add`, …) are
provided as inherent methods on `Dual<F: Real>` and `DualN<F: Real, K>`, and
nested compositions like `Dual<Complex<f64>>` work — that is ∂/∂t along a
complex path, the homotopy predictor's t-derivative.

## Vectors and matrices

`Vector<T, N>` and row-major `Matrix<T, ROWS, COLS>` split their APIs by
bound: structural operations (add/sub/neg, scalar mul, transpose, matrix
multiply, cross products) need only `T: Ring` — integer matrices are
first-class — while norms, `determinant`, `inverse`, and `lu`/`solve` need
`T: Scalar`. Square-only operations live on `Matrix<T, N, N>`, so a
non-square determinant is a *compile* error.

```rust
use talrost::{matrix::Matrix, vector::Vector};

let v1 = Vector::new([1., 2., 3.]);
let v2 = Vector::new([4., 5., 6.]);

// Dot products are hermitian (the second operand is conjugated), so
// dot(v, v) == |v|² is real even for complex vectors.
assert_eq!(v1.dot(&v2), 32.);
assert_eq!(v1.cross(&v2), Vector::new([-3., 6., -3.]));
assert_eq!(v1.magnitude(), 14_f64.sqrt());

// Matrix × vector, and the LU-backed one-shot solve.
let a = Matrix::new([[2., 1.], [1., 3.]]);
let x = Vector::new([1., -2.]);
let b = a * x;
let solved = a.solve(&b).unwrap();
assert!((solved - x).magnitude() < 1e-12);

// Factor once, reuse for many right-hand sides (Newton corrector shape);
// determinant and inverse ride on the same factorization.
let lu = a.lu().unwrap();
let x2 = lu.solve(&Vector::new([1., 0.]));
assert_eq!(a.determinant(), 5.0);
assert_eq!(a.transpose(), Matrix::new([[2., 1.], [1., 3.]]));
```

`Lu` uses partial pivoting by `norm_sqr`, so the same code path factors real
and complex matrices; `determinant` (N > 3) and `inverse` are built on it.

## Integer lattices

`lattice::smith_normal_form` and `lattice::hermite_normal_form` reduce
`Matrix<i64, M, N>` exponent matrices — the offline half of a polyhedral
homotopy, where the Smith form solves each mixed cell's binomial start
system in closed form:

```rust
use talrost::{lattice, matrix::Matrix};

// U·A·V == S with unimodular U, V; |det A| = ∏ Sᵢᵢ = the number of start
// solutions of the binomial system x^A = b.
let a = Matrix::<i64, 2, 2>::new([[3, 1], [1, 3]]);
let (u, s, v) = lattice::smith_normal_form(&a);
assert_eq!(u * a * v, s);
assert_eq!(s, Matrix::new([[1, 0], [0, 8]])); // 8 start solutions
```

## Multivariate systems

`mvpoly` provides sparse polynomials with compile-time shape: `Monomial<NV>`
exponent vectors, `MPoly<T, NV, TERMS>`, and `MSystem<T, NV, NEQ, MAXT>`
(rows padded to `MAXT` terms with zero coefficients — the stack-only answer
to heterogeneous term counts). `eval_jacobian` produces values and the full
`NEQ×NV` Jacobian in one `DualN` sweep:

```rust
use talrost::{matrix::Matrix, mvpoly::{MPoly, MSystem, Monomial}, vector::Vector};

// f1 = x^2 + y^2 - 5,  f2 = xy - 2      (root at (2, 1))
// J = [[2x, 2y], [y, x]].
let f1 = MPoly::<f64, 2, 3>::new(
    [1.0, 1.0, -5.0],
    [Monomial::new([2, 0]), Monomial::new([0, 2]), Monomial::new([0, 0])],
);
let f2 = MPoly::<f64, 2, 3>::new(
    [1.0, -2.0, 0.0],
    [Monomial::new([1, 1]), Monomial::new([0, 0]), Monomial::new([0, 0])],
);
let sys = MSystem::new([f1, f2]);

let (vals, jac) = sys.eval_jacobian(&[2.0, 1.0]);
assert_eq!(vals, [0.0, 0.0]);
assert_eq!(jac, Matrix::new([[4.0, 2.0], [1.0, 2.0]]));

// A Newton corrector step: solve J·dx = -f at a perturbed point via LU.
let x0 = [2.1, 0.9];
let (h, j) = sys.eval_jacobian(&x0);
let dx = j.solve(&Vector::new([-h[0], -h[1]])).unwrap();
```

Structural derivatives (`MPoly::partial`) are the allocation-free fast path;
the `DualN` machinery is the general mechanism, and the two are tested
against each other.

## Feature flags

| Feature | Default | Toolchain | What it does |
|---|---|---|---|
| `std` | ✓ | stable | Float math (`sqrt`, `sin`, `fma`, …) via the standard library. |
| `libm` | | stable | Float math via the [`libm`](https://crates.io/crates/libm) crate for `no_std` builds. If both are enabled, `std` wins. |
| `specialization` | | nightly | Dispatches fixed-size matrix-multiply kernels (Strassen 2×2, Laderman 3×3, AlphaTensor-style 4×4) via `min_specialization`. Without it the naive FMA loop is used everywhere. |

The crate builds on **stable by default** (`rust-version = "1.94"`); the
pinned nightly in `rust-toolchain.toml` is only required for the optional
`specialization` feature. For embedded targets:

```sh
cargo build --no-default-features --features libm --target thumbv7em-none-eabihf
```

Everything — containers, root finders, dual numbers, SNF — is `no_std`
compatible; only the float math backend changes.

## Benchmarks and codegen guard

`cargo run --release --example bench` runs dependency-free micro-benchmarks
(Horner evaluation, `eval_at` at a complex point, 4×4 matmul, a mock
corrector step); re-run with `--features specialization` on nightly to
compare the matmul kernels. `tools/check_codegen.sh` compiles a probe crate
and fails if any `call` instruction lands inside the hot polynomial
evaluation paths — the guard that catches `mul_add` silently falling back to
a software-fma libm call (a 5.7× regression when it happened).

## Roadmap

The destination is a **polyhedral homotopy continuation solver**
(Huber–Sturmfels) for sparse polynomial systems, split as:

- **Offline** (host, or build time): from the support sets of the target
  system, compute a lifted mixed subdivision, enumerate mixed cells, and
  solve one binomial start system per cell via the Smith normal form of its
  exponent matrix (`lattice` is this phase's foundation). The offline result
  depends only on monomial structure — for embedded targets the cells/start
  data can be baked in at compile time.
- **Online** (device): track each start solution along `H(x, t)` with a
  predictor–corrector loop — `DualN<Complex<f64>, NV>` Jacobians for the
  Newton corrector, `Dual` in the t-slot for the predictor, `Lu::solve` per
  iteration, `eval_with_error`-style stopping rules.

The tower described above exists so that both halves are expressible with
stack-only, monomorphized code. The solver layer itself (mixed-volume LP,
cell enumeration, tracker, endgames) is future work.
