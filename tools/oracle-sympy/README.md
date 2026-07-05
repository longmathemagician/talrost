# oracle-sympy — exact solution-count oracle (Tier C)

`oracle.py` computes **exact** ground-truth data for the benchmark systems
of `examples/bench_suite.rs`, using sympy Gröbner bases over ℚ (Gaussian
rationals ℚ(i) for the complex conic). Per system it reports:

- **zero-dimensionality** (finitely many solutions or not);
- **dim C[x]/I** — the quotient-ring dimension = number of affine complex
  solutions counted *with multiplicity* (standard-monomial count under the
  grevlex leading-term ideal);
- **radicality**, by Seidenberg's criterion (the minimal polynomial of every
  coordinate function on C[x]/I is squarefree ⟺ the zero-dimensional ideal
  is radical; minimal polynomials are found by exact linear algebra over the
  standard-monomial basis). Radical ⟹ the distinct count equals the
  dimension;
- the **torus count** (solutions with all coordinates ≠ 0 — the only roots
  Bernstein's theorem and a polyhedral homotopy's mixed volume count). For
  radical ideals: `dim C[x]/(I + ⟨x₁⋯x_n⟩)` counts exactly the distinct
  off-torus solutions (every local ring of a radical zero-dimensional
  quotient is ℂ). Otherwise the Rabinowitsch localization
  `dim C[x,t]/(I + ⟨t·x₁⋯x_n − 1⟩)` counts torus solutions with
  multiplicity;
- for cyclic-3, the six exact solutions.

## Why `solve_poly_system` is not trusted for counting

On katsura-3, `sympy.solvers.polysys.solve_poly_system` silently returned
**2 of the 8** solutions: its lex back-substitution drops roots it cannot
express by radicals (the lex eliminant factors as
`u3·(3u3 − 1)·(irreducible sextic)` — the sextic's six roots vanished
without a warning). All counts here therefore come from quotient
dimensions, which are exact and complete.

## Setup and run (verified in the talrost dev container, 2026-07-05)

```sh
cd tools/oracle-sympy
python3 -m venv .venv          # .venv is gitignored
.venv/bin/pip install sympy    # sympy 1.14.0 at the time of recording
.venv/bin/python oracle.py     # all systems (~4 min; cyclic-5's 70-dim
                               # minimal-polynomial elimination is the pole)
.venv/bin/python oracle.py katsura-4 eco-5   # or a subset
```

Timeouts (SIGALRM, 600 s per Gröbner computation) turn blowups into
recorded findings instead of hangs. One was hit and worked around: the
6-variable Rabinowitsch basis for katsura-4's torus count exceeded 600 s,
which is why the radical-case hyperplane count above exists — it finishes in
seconds and gives the same answer on every system where both run
(cross-checked on katsura-3: 6 of 8 by both methods).

## Recorded results (this container)

| system    | zero-dim | dim C[x]/I | radical (distinct = dim) | torus count |
|-----------|----------|-----------:|--------------------------|------------:|
| cyclic-3  | yes      |          6 | yes                      |      6 of 6 |
| cyclic-4  | **NO**   |          — | — (positive-dimensional) |           — |
| cyclic-5  | yes      |         70 | yes                      |    70 of 70 |
| katsura-3 | yes      |          8 | yes                      |  **6 of 8** |
| katsura-4 | yes      |         16 | yes                      | **12 of 16**|
| noon-3    | yes      |         21 | yes                      |    21 of 21 |
| eco-4     | yes      |          4 | yes                      |      4 of 4 |
| eco-5     | yes      |          8 | yes                      |      8 of 8 |
| trinomial | yes      |          2 | yes                      |      2 of 2 |
| conic     | yes      |          4 | yes                      |      4 of 4 |

Notes:

- **cyclic-4** is *not* zero-dimensional: substituting `(a, b, −a, −b)`
  reduces the system to `a²b² − 1 = 0` — two curves of solutions
  (`ab = ±1`), verified by exact substitution in the oracle. There is no
  finite root count to benchmark against; the suite reports its 16 paths
  (the seed-invariant mixed volume of the supports, i.e. the root count of a
  *generic* system with cyclic-4's monomials) as honestly failing.
- **katsura-3/-4** settle the convention question: in the (n+1)-unknown
  formulation used here (`u_0..u_n`), katsura-n has exactly 2ⁿ distinct
  affine solutions (8 and 16), of which 2 (resp. 4) have some coordinate
  exactly 0. The polyhedral homotopy's mixed volume — 6 (resp. 12) — equals
  the *torus* count exactly, and the suite's converged path counts match it.
- **eco-4/eco-5** (PHCpack formulation with constants −k): exactly 4 and 8
  distinct solutions, all on the torus, matching the computed mixed volumes
  and converged counts. No solutions at infinity affect the affine counts;
  the mixed volume here is exact for the torus roots.
- The **cyclic-3** solutions are the six permutations of
  `(1, ω, ω̄)`, `ω = e^{2πi/3}`, matching the structural oracle in the Rust
  test suite.

The oracle is deliberately **not** part of the Rust build or CI — it is a
manual, recorded ground-truth tool.
