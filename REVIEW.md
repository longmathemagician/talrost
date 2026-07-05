# Talrost design review and improvement proposals

*Reviewed July 2026 against `rustc 1.98.0-nightly (c397dae80 2026-07-02)`. All defects and
feature-status claims below were verified by compiling and running code on that toolchain.*

Talrost's core idea — an algebraic trait tower feeding const-generic `Polynomial`, `Vector`,
and `Matrix` types with size-exact APIs — is sound and worth pursuing. The main problems are:

1. Several **correctness bugs**, two of which silently return wrong numbers.
2. A **numerical tower that is inverted**: `f64` implements `Integer`, `Complex` implements
   `Float`/`PartialOrd`, and half of those impls are `todo!()` landmines.
3. A hard dependency on **`generic_const_exprs`, which the Rust project has abandoned** in
   favor of `min_generic_const_args` — the crate should migrate off it.
4. Feature gates that are **stale** (three are stable, one is unused, one lint no longer
   exists) and one (`min_specialization`) that is declared but never actually used — while
   there is a genuinely good use for it in the matrix kernels.

Each section below is ordered roughly by priority.

---

## 1. Confirmed defects (P0 — fix before anything else)

### 1.1 `Matrix` kernel dispatch uses bitwise `&` instead of `&&`

`src/matrix.rs:83` (and `:99`, `:154`, plus `determinant` at `:47`, `:49`):

```rust
if M == 2 & N & O {        // parses as  M == (2 & N & O)
```

`2 & N & O` is a *bitwise AND*. The guard is true whenever the bit-pattern happens to
collide, not when all three dimensions are 2. Verified failure:

```rust
let a = Matrix::<f64, 2, 3>::new([[1., 2.], [3., 4.], [5., 6.]]);
let b = Matrix::<f64, 2, 2>::new([[1., 0.], [0., 1.]]);   // identity
let c = a * b;   // M=2, N=3, O=2 → 2 & 3 & 2 == 2 → Strassen 2×2 runs
// c = [[1,2],[3,4],[0,0]]  — third row silently zeroed
```

Likewise `Matrix::<f64, 2, 6>::determinant()` compiles and returns a number for a
non-square matrix. The minimal fix is `M == 2 && N == 2 && O == 2` etc.; the structural fix
(making non-square `determinant` *uncompilable*) is in §3.4.

### 1.2 `Complex::magnitude` returns |z|⁴

`src/complex.rs:77`:

```rust
pub fn magnitude(&self) -> F {
    (self.re.powi(2) + self.im.powi(2)).powi(2)   // should be .sqrt()
}
```

Verified: `c64::new(3.0, 4.0).magnitude() == 625.0` (expected `5.0`). `normalize()` divides
by this value, so it is wrong too. The existing tests don't catch it because
`Vector::magnitude` computes its own sum-of-squares and calls `Float::sqrt`.

### 1.3 `Field::recip` for `Complex` is an infinite recursion

`impl_field!` expands to:

```rust
fn recip(self) -> Self { Self::recip(self) }
```

For `f32`/`f64` the *inherent* `recip` wins method resolution, so this works. `Complex` has
no inherent `recip`, so `Self::recip` resolves to the trait method itself. Verified:
`Field::recip(c64::new(2.0, 0.0))` overflows the stack. The `#[allow(unconditional_recursion)]`
in the macro is suppressing a real compiler diagnosis. Fix: give `Field` no body (force each
impl to write it) or default to `Self::ONE / self`, and implement complex reciprocal
properly (`conj(z) / |z|²`).

### 1.4 `From<&str> for Complex` cannot parse negative components and swallows errors

The parser only splits on `'+'`, so `"3 - 4i"` accumulates `"3-4"` as the real part, fails
to parse, and `unwrap_or(ZERO)` turns it into `0 + 0i` (verified). Meanwhile `Display`
prints negative imaginary parts as `"a - bi"`, so `Display` output does not round-trip
through `From<&str>`. Also, a `From` impl that silently maps garbage to zero is an API
hazard. Proposal: implement `core::str::FromStr` with a real error type, delete
`From<&str>`, and update `assert_eq!(b, "2 + 0i".into())`-style tests to
`"2 + 0i".parse().unwrap()`.

### 1.5 `Polynomial::roots` for cubics returns NaN (one test already fails)

`polynomial.rs:66` has the `4 => yuksel::roots_cubic` arm commented out (the Yuksel solver
is `f64`-only, so it doesn't fit the generic signature), so cubic root requests fall through
to the NaN arm. `cargo test` fails today on `roots_3_generic`. The structural fix is §3.5.

### 1.6 Smaller but real

- **`Natural::powi(power: i32)` casts to `u32`** (`natural.rs:27`, `integer.rs:18`):
  `2u32.powi(-1)` becomes `pow(4294967295)` — overflow panic in debug, garbage semantics in
  release. Take `u32` for non-fields, or define negative powers only where `recip` exists.
- **`Natural::BITS: Self`** is `0.0` for floats (self-acknowledged "Totally wrong") — it
  should be an associated `u32`, not `Self`.
- **`Complex` derives `PartialOrd`** — lexicographic order on ℂ is mathematically
  meaningless and only exists to satisfy `Natural: PartialOrd`. Symptom of §2.
- **Root ordering contract is inconsistent**: Yuksel returns ascending roots, Blinn returns
  descending (the tests encode both). Pick one contract (ascending) and document it.
- **`eval` is `todo!()` for N > 5**: a generic Horner loop should be the fallback (§3.5).
- **35 compiler warnings**, including `unused_mut`, unused variables, and non-snake-case
  locals. In solver code where uppercase names mirror the paper's notation, use a scoped
  `#[allow(non_snake_case)]` with a comment; fix the rest.

---

## 2. The numerical tower is upside-down (P1)

The current subtyping chain is:

```text
Element ← Monoid ← Group ← Semiring ← Ring ← Field
Natural: Semiring + PartialOrd
Integer: Natural + Ring        // and f32/f64 implement it
Float:   Natural + Field       // and Complex<F> implements it
```

Reading `impl Integer for f64` as a proposition — "f64 is an integer" — shows the problem:
the tower is expressing *capability inclusion* ("anything a u32 can do, an f64 can do") but
naming it as *set inclusion*, which runs in exactly the opposite direction (ℕ ⊂ ℤ ⊂ ℝ ⊂ ℂ).
The consequences are concrete, not aesthetic:

- `Complex` is forced to implement `PartialOrd` (bogus), `MIN`/`MAX` (bogus), `BITS`
  (bogus), and nine `Float` methods as `todo!()` panics (`abs`, `floor`, `ceil`, `sin`,
  `cos`, `tan`, `atan2`, `sin_cos`, `cbrt`, `mul_add`, `copysign`).
- `Float::MIN` for `f64` is −1.8·10³⁰⁸ under a trait named `Natural`.
- `abs()` returns `Self`, so a complex modulus — a *real* number — cannot be expressed.

### Proposed tower

Keep the algebraic layer (§2.1 tweaks aside) and rebuild the numeric layer around a
`Real`/`Scalar` split, which is the same conclusion `num-complex` and `nalgebra`
(`RealField`/`ComplexField`) converged on:

```rust
/// Ordered field with float semantics. f32, f64 (later f16/f128).
pub trait Real: Field + PartialOrd {
    const EPSILON: Self;
    const INFINITY: Self;
    const NAN: Self;
    const MANTISSA_DIGITS: u32;   // u32, not Self
    fn abs(self) -> Self;
    fn floor(self) -> Self;  fn ceil(self) -> Self;
    fn sqrt(self) -> Self;   fn cbrt(self) -> Self;
    fn sin_cos(self) -> (Self, Self);
    fn atan2(self, other: Self) -> Self;
    fn mul_add(self, a: Self, b: Self) -> Self;
    fn copysign(self, sign: Self) -> Self;
    fn is_nan(self) -> bool; fn is_finite(self) -> bool;
    // ...
}

/// What Vector/Matrix/Polynomial actually need: a field with a real-valued norm.
pub trait Scalar: Field {
    type Real: Real;
    fn norm_sqr(self) -> Self::Real;              // cheap, exact
    fn norm(self) -> Self::Real { self.norm_sqr().sqrt() }
    fn conj(self) -> Self;
}

impl<F: Real> Scalar for F { type Real = F; ... } // conj = id, norm = abs
impl<F: Real> Scalar for Complex<F> { type Real = F; ... }
```

What this buys, concretely:

- `Complex` stops lying: no `PartialOrd`, no `MIN/MAX/BITS`, no `todo!()` panics reachable
  through generic code. (Complex `sin`/`cos`/`exp` can be added later as inherent methods —
  they are well-defined, unlike complex `floor`.)
- `Vector::magnitude` returns `T::Real`, so a complex vector's magnitude is an `f64`, not a
  `Complex` that happens to have zero imaginary part. The existing
  `vec_complex.magnitude() == 5_f64.sqrt().into()` test becomes the more honest
  `== 5_f64.sqrt()`.
- `Natural`/`Integer` become what they are — the unsigned/signed machine-integer families
  (`Semiring + Ord` and `Ring + Ord` respectively) — and floats stop implementing them.

### 2.1 Algebraic layer tweaks

- **`Group::Neg` (capital N) is redundant** with the `Neg<Output = Self>` supertrait bound
  and violates naming conventions. Delete the method; `-x` and `x.neg()` already work.
- **Document the additive convention.** `Monoid` is silently *additive* monoid; either
  rename (`AdditiveMonoid`) or document. Adding `core::iter::Sum` to `Monoid` and
  `Product` to `Semiring` costs nothing and unlocks `.sum()` in generic code (the
  hand-rolled `Sum for Complex` impl generalizes to a blanket default).
- **Macro hygiene:** `impl_group!` re-implements `Monoid` itself, so `impl_monoid!` +
  `impl_group!` on the same type would conflict. Make each macro implement exactly one
  trait and provide one `impl_scalar!`/`stack_*` convenience macro per numeric family that
  composes them.
- **`Element: Display`** is a heavy bound for an embedded-targeted crate (pulls formatting
  machinery into every generic). Keep `Debug`, move `Display` bounds to the `Display` impls
  that need them.
- Optional, cheap, and useful for solver correctness: marker traits `CommutativeRing`, and
  doc-contracts on identities (`ZERO + x == x`, etc.) testable with a macro-generated
  law-check test suite per implementing type.

### 2.2 Delete `Number`

```rust
pub trait Number { type Type: Natural; fn new(v: Self::Type) -> Self::Type; }
impl<T: Natural> Number for T { ... }
```

This is an identity function wearing a trait costume, and it infects every signature in the
crate with `T: Number<Type = T> + Float`. Every such bound is exactly equivalent to
`T: Float`. Removing it deletes ~20 `where` clauses.

---

## 3. Const-generics strategy: get off `generic_const_exprs` (P1)

**Status check (verified July 2026):** `generic_const_exprs` is still marked incomplete,
and the Rust project has effectively declared it dead — the active work is
`min_generic_const_args` ("stabilizable prototype" project goal, 2024h2) and the
[Full Const Generics 2026 project goal](https://rust-lang.github.io/rust-project-goals/2026/const-generics.html).
Nothing shaped like `[T; N + 0_usize.pow(N as u32 - 1) - 1]` is on any stabilization path.
A crate whose *public API* (`Polynomial::roots` return type) depends on it will break with
nightly churn and can never ride toward stable.

The good news: everything talrost uses it for can be expressed better without it.

### 3.1 Replace the runtime `match N` + NaN-padded arrays with per-size inherent impls

Inherent impls on *concrete* const arguments are **stable Rust** and do not overlap:

```rust
impl<T: Real> Polynomial<T, 3> {
    pub fn roots(&self, tol: T) -> Roots<T, 2> { /* quadratic */ }
}
impl<T: Real> Polynomial<T, 4> {
    pub fn roots(&self, tol: T) -> Roots<T, 3> { /* cubic */ }
}
impl<T: Real> Polynomial<T, 5> {
    pub fn roots(&self, tol: T) -> Roots<T, 4> { /* quartic */ }
}
```

This simultaneously:

- deletes the `N + 0_usize.pow(N as u32 - 1) - 1` hack (it encodes `max(N-1, 1)`);
- deletes the `[(); N]:` bounds;
- deletes the runtime `match N` with its `todo!()` and silently-NaN arms (each degree gets
  a real implementation or *doesn't compile*, instead of failing at runtime — the currently
  failing `roots_3_generic` test is exactly this failure mode);
- gives exact-size return types per degree with no dead slots.

### 3.2 Return a counted `Roots` type, not NaN sentinels

NaN-as-"no root" breaks `assert_eq!`/`PartialEq`, can't be distinguished from a genuinely
NaN computation, and forces every caller to re-scan the array. A `heapless`-style bounded
vec is `no_std`-friendly and self-describing:

```rust
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Roots<T, const MAX: usize> { buf: [T; MAX], len: usize }

impl<T: Copy, const MAX: usize> Roots<T, MAX> {
    pub fn as_slice(&self) -> &[T] { &self.buf[..self.len] }
    pub fn len(&self) -> usize { self.len }
}
// + Deref<Target = [T]>, IntoIterator, FromIterator-ish push
```

### 3.3 Degree-generic pieces that remain generic

`eval` needs no per-size code at all — Horner's method, which is also fewer operations and
better conditioned than the current powers-of-x form:

```rust
pub fn eval(&self, x: T) -> T {
    self.c.iter().skip(1)
        .fold(self.c[0], |acc, &k| acc.mul_add(x, k))
}
```

Monomorphization fully unrolls this for each `N`; the hand-written `eval_quadratic`/
`eval_cubic`/`eval_quartic` and the `match N` dispatcher can all be deleted (keep one
benchmark to confirm codegen parity). A `derivative()` returning `Polynomial<T, N-1>` is
the one API that genuinely wants arithmetic in a const position — per-size inherent impls
cover degrees ≤ 5 today; revisit when `min_generic_const_args` lands.

### 3.4 Matrices: encode squareness in the type

The identity/determinant/inverse family only exists for square matrices, and the current
code checks this at runtime (`identity()` has a `const`-context `panic!()`; `determinant`
mis-dispatches per §1.1). Inherent impls on `Matrix<T, N, N>` make wrong dimensions
*uncompilable*:

```rust
impl<T: Scalar, const N: usize> Matrix<T, N, N> {
    pub const IDENTITY: Self = { /* diagonal ONE */ };
    pub fn determinant(&self) -> T { ... }     // 2, 3 via specialization or match
    pub fn inverse(&self) -> Option<Self> { ... }
}
```

`transpose` — currently `todo!()` — is expressible today with plain const generics:

```rust
impl<T: Scalar, const M: usize, const N: usize> Matrix<T, M, N> {
    pub fn transpose(&self) -> Matrix<T, N, M> { ... }
}
```

Also strongly recommended: flip the parameter order to `Matrix<T, const ROWS, const COLS>`.
Today `Matrix::<f32, 2, 3>` is 3 rows × 2 columns (`e: [[T; M]; N]`), the reverse of
universal m×n convention, which will bite every new user and already makes `Mul`'s
signature (`Matrix<T, M, N> × Matrix<T, O, M>`) hard to audit.

---

## 4. Specialization: current state and where it actually helps (P2)

**Status check (verified July 2026):**

- Full `specialization` (RFC 1210) remains **unsound** — the lifetime-dependence hole is
  still open on the [tracking issue #31844](https://github.com/rust-lang/rust/issues/31844)
  — and the std-dev guide's policy is unchanged: *only* `min_specialization` should be
  used. There is no stabilization timeline for either.
- `min_specialization` compiles fine on 1.98.0-nightly. Restrictions: the specializing impl
  must be "always applicable" (no new lifetime constraints, no specializing on lifetimes),
  and only items marked `default` can be overridden.
- **Verified on this exact nightly:** `min_specialization` *does* allow specializing a
  const-generic impl with concrete const values:

  ```rust
  impl<const M: usize, const N: usize> Kernel for Mat<M, N> {
      default fn which(&self) -> &'static str { "generic" }
  }
  impl Kernel for Mat<2, 2> {                    // ← accepted, dispatches correctly
      fn which(&self) -> &'static str { "2x2" }
  }
  ```

That last point is the one genuinely load-bearing use of specialization for talrost:

### 4.1 Matrix multiply kernels as specialized impls

Move the Strassen (2×2), Laderman (3×3), and AlphaTensor-style (4×4) kernels out of the
`if`-chain into an internal kernel trait:

```rust
trait Gemm<Rhs> { type Output; fn gemm(self, rhs: Rhs) -> Self::Output; }

impl<T: Scalar, const M: usize, const N: usize, const O: usize>
    Gemm<Matrix<T, O, M>> for Matrix<T, M, N>
{
    type Output = Matrix<T, O, N>;
    default fn gemm(self, x: Matrix<T, O, M>) -> Matrix<T, O, N> { /* triple loop */ }
}

impl<T: Scalar> Gemm<Matrix<T, 2, 2>> for Matrix<T, 2, 2> {
    fn gemm(self, x: Matrix<T, 2, 2>) -> Matrix<T, 2, 2> { /* Strassen */ }
}
// likewise 3×3, 4×4; `Mul` delegates to `Gemm::gemm`
```

This makes the dimension logic *type-checked* (a 2×3 input can never reach the 2×2 kernel —
the bug class of §1.1 becomes unrepresentable), keeps `Mul` as the single user-facing
operator, and lets each kernel carry different bounds if needed (e.g., an FMA-using kernel
requiring `Real`).

Two honest caveats to record in the code:

- **A stable fallback exists and should land first**: fixing §1.1 to
  `if M == 2 && N == 2 && O == 2` is correct and free — after monomorphization the
  condition is a compile-time constant and LLVM deletes the dead branches. Specialization
  is the better *architecture*, not a performance unlock.
- **Question the kernels themselves.** Strassen/Laderman trade multiplications for many
  additions; at 2×2–4×4 with scalar f64 they are typically *slower* than the naive loop on
  modern hardware, and numerically less stable (relevant for a scientific-computing crate).
  The 47-multiplication AlphaTensor result is for ℤ/2ℤ, not floats. Benchmark before
  keeping; consider gating them behind a `fast-kernels` cargo feature and defaulting to
  naive + `mul_add`.

### 4.2 Places that look like specialization but aren't

- **Per-type overrides of trait methods** (e.g., an optimized `sin_cos` for `f64`): trait
  impls per concrete type already do this. No specialization needed.
- **Per-degree polynomial APIs**: inherent impls on concrete const args (§3.1) are stable.
- **Scalar × Vector (`2. * v`)**: currently implemented only for `f64`. The blanket
  `impl<T: Scalar, const N: usize> Mul<Vector<T, N>> for T` is *not* writable — coherence
  forbids a foreign trait impl whose self type is a bare type parameter — and
  specialization does not lift orphan-rule restrictions. The standard answer is a macro
  stamping impls for `f32, f64, c32, c64` (what `num-traits` does). Do that for `Vector`
  and `Matrix` both.
- If/when the crate wants `From<F> for Complex<F>`-style blanket conversions interacting
  with generic impls, that's where `min_specialization` may reappear; avoid designing for
  it until forced.

---

## 5. Feature-gate and toolchain hygiene (P0, mechanical)

Verified on 1.98.0-nightly:

| Gate in `lib.rs` | Status today | Action |
|---|---|---|
| `associated_type_bounds` | **stable since 1.79** (warns) | delete |
| `const_float_bits_conv` | **stable since 1.83** (warns) | delete |
| `generic_arg_infer` | **stable since 1.89** (warns) | delete |
| `more_qualified_paths` | declared, **unused** (warns) | delete |
| `#![allow(soft_unstable)]` | lint **removed from rustc** (warns) | delete |
| `min_specialization` | still nightly; **currently unused** | delete now; reintroduce with §4.1 |
| `generic_const_exprs` | incomplete, project abandoned it | remove per §3; until then keep |

After §3 lands, the crate is `generic_const_exprs`-free; after §4.1, the only gate is
`min_specialization` — a deliberate, single, sound-subset dependency. If instead you take
the stable `match` route, **talrost compiles on stable Rust**, which is worth a lot for an
embedded-facing crate (pin nightly only in CI lanes that test the specialized kernels).

Also pin the toolchain: `rust-toolchain.toml` says `channel = "nightly"`, which means every
fresh clone gets a different compiler. Use a dated pin (e.g. `nightly-2026-07-02`) and bump
it on a schedule.

### Nightly features worth *watching* (not adopting yet)

- **`min_generic_const_args` / Full Const Generics (2026 project goal)** — will eventually
  give `Polynomial<T, N>::derivative() -> Polynomial<T, N-1>` and friends a sound basis.
- **Const traits (RFC 3762, `const_trait_impl`)** — would let `T::ZERO`-style arithmetic
  run in `const fn` generic over `T`, enabling `const` polynomial evaluation and
  `IDENTITY` construction without per-type tricks.
- **`portable_simd`** — natural fit for `Vector`/`Matrix` inner loops once APIs settle;
  keep behind a cargo feature.
- **`f16`/`f128`** — cheap to support through the `Real` macro once their std methods
  stabilize; interesting for the embedded audience.

---

## 6. Embedded/`no_std` roadmap (P2 — it's the crate's stated purpose)

The README promises embedded-readiness; the code currently requires `std` in avoidable ways:

1. **Cargo features**: `default = ["std"]`; a `libm` feature supplying `sin`/`sqrt`/… for
   `no_std` builds (this is exactly the commented-out `libm = "0.2"` dependency). The
   `Real` impl macro picks `std` intrinsics or `libm` per cfg.
2. **Stop allocating in `Display`**: `Vector`/`Matrix`/`Polynomial` `fmt` build `String`s
   with `format!` and even use `.pop()` to strip separators. Write straight to the
   `Formatter` (`write!` between separator logic) — no alloc, no `std`, less code.
3. `#![no_std]` + `#[cfg(feature = "std")] extern crate std;` at the root; import from
   `core::` throughout (several files use `std::ops` where `core::ops` works).
4. `display.rs::format_f64` is dead code (`#[allow(dead_code)]`), `f64`-only, and
   documented to panic for widths < 7. Either finish it as a real fixed-width formatter
   behind a feature or delete it.

---

## 7. API polish and process (P3)

- **Solver architecture**: `yuksel` is free functions over `f64` only; `blinn` is a
  `PhantomData` struct generic over `T`. Unify: generic free functions per module over
  `T: Real`, plus a `CubicSolver`-style trait if pluggable solver choice is wanted
  (`p.roots(tol)` uses the default; `p.roots_with::<Yuksel>(tol)` overrides). Genericizing
  Yuksel over `Real` also un-blocks §1.5, and `Blinn`'s repeated
  `T::ONE + T::ONE + T::ONE` constants fall out of a `Real::from_u16(3)`-ish constructor or
  associated `TWO`/`THREE` consts.
- **`Polynomial::from` vs `new`**: two identical constructors, one shadowing the `From`
  trait idiom. Keep `new` (const), add real `From<[T; N]>`.
- **Missing basics** users will reach for immediately: `Vector::dot`, 3-D `cross`,
  `Index`/`IndexMut` on all three container types, `Default`, `From<[T; N]>`,
  scalar-division for vectors, `Matrix × Vector`.
- **Testing**: add property tests (`proptest`) comparing against `num-complex`/`nalgebra`
  as dev-dependencies — §1.1 and §1.2 would both have been caught by a one-line
  round-trip/oracle property. Add the algebra law-check macro from §2.1. Turn on
  `#![warn(missing_docs)]` and `#![forbid(unsafe_code)]` (the crate uses no `unsafe` —
  advertise it).
- **CI**: a GitHub Actions matrix over {pinned nightly, stable-if-§4-stays-optional} ×
  {default, `no_std`+`libm`} running `fmt`, `clippy -D warnings`, `test`. The tree
  currently has 35 warnings and a failing test; CI is what keeps them at zero.

---

## Suggested sequencing

| Phase | Content |
|---|---|
| 1 (bugs) | §1.1–§1.6, §5 gate cleanup, fix warnings, add regression tests |
| 2 (tower) | §2: `Real`/`Scalar` split, delete `Number`, algebra macro cleanup |
| 3 (const generics) | §3: per-size inherent impls, `Roots<T, MAX>`, Horner `eval`, square-matrix impls, drop `generic_const_exprs` |
| 4 (kernels + no_std) | §4.1 behind a feature (or stable `match`), §6 `no_std`+`libm`, benchmarks |
| 5 (polish) | §7 API additions, property tests, CI, docs |

Phases 1–3 leave the crate smaller, stable-compilable (if kernels use `match`), and with
every currently-wrong answer either fixed or unrepresentable.
