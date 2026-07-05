#!/usr/bin/env python3
"""Exact solution-count oracle for the talrost benchmark suite (Tier C).

Analyzes the benchmark systems exactly with sympy Groebner bases over the
rationals (Gaussian rationals for the complex conic) and prints, per system:

  - whether the ideal is zero-dimensional (finitely many solutions);
  - ``dim C[x]/I``: the quotient-ring dimension as a C-vector space (the
    standard-monomial count under the grevlex leading-term ideal), which is
    the number of affine complex solutions counted WITH multiplicity;
  - whether the ideal is RADICAL, decided by Seidenberg's criterion: a
    zero-dimensional ideal is radical iff the minimal polynomial of every
    coordinate function x_i on C[x]/I is squarefree (the minimal polynomials
    are computed by exact linear algebra over the standard-monomial basis).
    For a radical ideal the distinct-solution count equals dim C[x]/I;
  - the TORUS solution count (all coordinates nonzero) via the Rabinowitsch
    trick: dim C[x,t]/(I + <t*x_1*...*x_n - 1>) counts exactly the solutions
    off the coordinate hyperplanes, with multiplicity. This is the count a
    polyhedral homotopy's mixed volume bounds (Bernstein);
  - for cyclic-3: the six exact solutions.

``solve_poly_system`` is deliberately NOT used for counting: on katsura-3 it
silently returned 2 of the 8 solutions (it drops roots its back-substitution
cannot express by radicals — the lex eliminant of katsura-3 factors as
u3 * (3*u3 - 1) * <irreducible sextic>, and the sextic's six roots were
dropped without a warning). Counts here come from quotient dimensions only.

This oracle is ground truth for examples/bench_suite.rs and BENCHMARKS.md.
It is deliberately NOT part of the Rust build or CI — run it manually:

    python3 -m venv .venv
    .venv/bin/pip install sympy
    .venv/bin/python oracle.py            # all systems
    .venv/bin/python oracle.py cyclic-4   # one system

Verified outputs from this container are recorded in README.md next to this
script.
"""

import signal
import sys
from contextlib import contextmanager

from sympy import (
    I,
    Integer,
    Poly,
    Rational,
    degree,
    diff,
    gcd,
    groebner,
    prod,
    symbols,
)
from sympy.solvers.polysys import solve_poly_system


class Timeout(Exception):
    pass


@contextmanager
def time_limit(seconds):
    """SIGALRM-based timeout (Unix only) so a Groebner blowup cannot hang
    the oracle; the caller reports the timeout as a finding."""

    def handler(signum, frame):
        raise Timeout()

    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


# ---------------------------------------------------------------------------
# System definitions — these must match examples/bench_suite.rs exactly.
# ---------------------------------------------------------------------------


def cyclic(n):
    """cyclic-n: f_k = sum_i prod_{j=i}^{i+k-1} x_{j mod n} (k = 1..n-1),
    f_n = x_0*...*x_{n-1} - 1."""
    x = symbols(f"x0:{n}")
    polys = []
    for k in range(1, n):
        f = 0
        for i in range(n):
            term = 1
            for j in range(i, i + k):
                term *= x[j % n]
            f += term
        polys.append(f)
    polys.append(prod(x) - 1)
    return polys, x


def katsura(n):
    """katsura-n in the (n+1)-unknown convention u_0..u_n (root count 2^n):
    for m = 0..n-1:  sum_{l=-n}^{n} u_{|l|} * u_{|m-l|}  =  u_m
    (terms with |m-l| > n dropped), plus  u_0 + 2*sum_{l=1}^{n} u_l = 1."""
    u = symbols(f"u0:{n + 1}")
    polys = []
    for m in range(n):
        f = -u[m]
        for l in range(-n, n + 1):
            if abs(m - l) <= n:
                f += u[abs(l)] * u[abs(m - l)]
        polys.append(f)
    polys.append(u[0] + 2 * sum(u[1:]) - 1)
    return polys, u


def noon(n):
    """noon-n (Noonburg's neural-network system, coefficient 1.1 as in the
    PHCpack demo database): x_i * (sum_{j != i} x_j^2) - 1.1*x_i + 1 = 0."""
    x = symbols(f"x1:{n + 1}")
    c = Rational(11, 10)
    polys = []
    for i in range(n):
        s = sum(x[j] ** 2 for j in range(n) if j != i)
        polys.append(x[i] * s - c * x[i] + 1)
    return polys, x


def eco(n):
    """eco-n (Morgan's economics system, PHCpack formulation):
    (x_k + sum_{i=1}^{n-k-1} x_i * x_{i+k}) * x_n - k = 0,  k = 1..n-1,
    sum_{i=1}^{n-1} x_i + 1 = 0."""
    x = symbols(f"x1:{n + 1}")
    polys = []
    for k in range(1, n):
        s = x[k - 1]
        for i in range(1, n - k):
            s += x[i - 1] * x[i + k - 1]
        polys.append(s * x[n - 1] - k)
    polys.append(sum(x[: n - 1]) + 1)
    return polys, x


def trinomial():
    """The calibration trinomial pair from the driver tests (MV 2):
    1 - 3x + xy = 0,  2 + y + xy = 0."""
    x, y = symbols("x y")
    return [1 - 3 * x + x * y, 2 + y + x * y], (x, y)


def conic():
    """The dense complex conic pair from the driver tests (MV 4). The float
    coefficients are exact decimals, entered as Gaussian rationals."""
    x, y = symbols("x y")

    def c(re, im):
        return Rational(re) + Rational(im) * I

    f1 = (
        c("1.1", "0.3")
        + c("-0.7", "0.9") * x
        + c("0.5", "-1.3") * y
        + c("2.0", "0.1") * x**2
        + c("-1.4", "-0.8") * x * y
        + c("0.6", "1.7") * y**2
    )
    f2 = (
        c("-0.9", "1.2")
        + c("1.8", "-0.4") * x
        + c("0.3", "0.7") * y
        + c("-1.1", "-1.6") * x**2
        + c("0.8", "0.2") * x * y
        + c("1.5", "-0.5") * y**2
    )
    return [f1, f2], (x, y)


# ---------------------------------------------------------------------------
# Exact counting machinery.
# ---------------------------------------------------------------------------


def standard_monomials(G, gens):
    """The monomials outside the grevlex leading-term ideal of G (zero-dim
    only). Their count is dim C[x]/I = #solutions with multiplicity."""
    lms = [p.monoms(order="grevlex")[0] for p in G.polys]
    n = len(gens)
    if any(sum(lm) == 0 for lm in lms):
        return []  # the unit ideal: empty variety, dim 0
    # Zero-dimensionality guarantees a pure power of each variable among the
    # leading monomials; those powers bound the standard-monomial box.
    box = []
    for i in range(n):
        d = min(
            (lm[i] for lm in lms if lm[i] > 0 and sum(lm) == lm[i]),
            default=None,
        )
        assert d is not None, "not zero-dimensional?"
        box.append(d)

    def divides(a, b):
        return all(ai <= bi for ai, bi in zip(a, b))

    out = []
    idx = [0] * n
    while True:
        if not any(divides(lm, tuple(idx)) for lm in lms):
            out.append(tuple(idx))
        j = 0
        while j < n:
            idx[j] += 1
            if idx[j] < box[j]:
                break
            idx[j] = 0
            j += 1
        if j == n:
            return out


# Exact Gaussian-rational arithmetic on (re, im) pairs of sympy Rationals.
# Plain sympy *expression* arithmetic does not auto-canonicalize over Q(i)
# (products/quotients stay unevaluated), so structural zero tests fail and
# the elimination below swells without terminating; pairs of Rationals stay
# canonical at every step. Over Q the imaginary parts are simply 0.


def _c_sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def _c_mul(a, b):
    return (a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])


def _c_div(a, b):
    d = b[0] ** 2 + b[1] ** 2
    return ((a[0] * b[0] + a[1] * b[1]) / d, (a[1] * b[0] - a[0] * b[1]) / d)


def _c_expr(a):
    return a[0] + a[1] * I


def nf_vector(G, expr, gens, index):
    """The normal form of expr modulo G as a dict over the standard-monomial
    basis (index: monomial tuple -> position), coefficients as exact
    (re, im) Rational pairs."""
    r = G.reduce(expr)[1]
    if r == 0:
        return {}
    p = Poly(r, *gens)
    out = {}
    for m, c in zip(p.monoms(), p.coeffs()):
        re, im = c.as_real_imag()
        out[index[m]] = (Rational(re), Rational(im))
    return out


def coordinate_minimal_polynomial(G, gens, i, basis):
    """The minimal polynomial of the coordinate function x_i on C[x]/I,
    found by exact incremental Gaussian elimination on the normal forms of
    1, x_i, x_i^2, ... over the standard-monomial basis. Returns a Poly in
    a fresh symbol T."""
    index = {m: j for j, m in enumerate(basis)}
    xi = gens[i]
    czero, cone = (Rational(0), Rational(0)), (Rational(1), Rational(0))
    # Echelon rows: pivot position -> (vector dict, power-combination list).
    rows = {}
    cur = Integer(1)  # normal form of x_i^k as an expression
    for k in range(len(basis) + 1):
        vec = nf_vector(G, cur, gens, index)
        comb = [czero] * (len(basis) + 2)
        comb[k] = cone
        # Reduce against the echelon rows.
        while vec:
            piv = max(vec)
            if piv not in rows:
                break
            rvec, rcomb = rows[piv]
            factor = _c_div(vec[piv], rvec[piv])
            for pos, c in rvec.items():
                nv = _c_sub(vec.get(pos, czero), _c_mul(factor, c))
                if nv == czero:
                    vec.pop(pos, None)
                else:
                    vec[pos] = nv
            comb = [_c_sub(cc, _c_mul(factor, rc)) for cc, rc in zip(comb, rcomb)]
        if not vec:
            # Dependency: sum_j comb[j] * x_i^j = 0 in the quotient.
            T = symbols("_T")
            return Poly(sum(_c_expr(c) * T**j for j, c in enumerate(comb)), T)
        rows[max(vec)] = (vec, comb)
        cur = G.reduce(xi * cur)[1]
    raise AssertionError("no dependency within dim+1 powers — impossible")


def is_squarefree(mp):
    """Squarefree test for a univariate Poly: gcd(m, m') is constant."""
    g = gcd(mp.as_expr(), diff(mp.as_expr(), mp.gens[0]))
    return degree(g, mp.gens[0]) == 0


def analyze(
    name,
    polys,
    gens,
    list_solutions=False,
    symmetric=False,
    torus_reason=None,
    gb_timeout=600,
):
    """Full exact analysis of one system; see the module docstring for what
    each printed line means. `symmetric=True` computes the coordinate
    minimal polynomial once (the system is invariant under a variable
    permutation acting transitively, so all coordinates share it).
    `torus_reason` skips the Rabinowitsch count with an analytic argument."""
    print(f"== {name} ({len(gens)} unknowns, {len(polys)} equations)")
    try:
        with time_limit(gb_timeout):
            # field=True: coefficients live in QQ (QQ_I for the conic), so
            # the normal-form linear algebra below can divide.
            G = groebner(polys, *gens, order="grevlex", field=True)
    except Timeout:
        print(f"   groebner: TIMEOUT ({gb_timeout} s) — no exact count available")
        print()
        return
    if not G.is_zero_dimensional:
        print("   zero-dimensional: NO — positive-dimensional solution set,")
        print("   infinitely many solutions; no meaningful finite root count.")
        print()
        return
    basis = standard_monomials(G, gens)
    dim = len(basis)
    print("   zero-dimensional: yes")
    print(f"   dim C[x]/I (affine solutions with multiplicity): {dim}")

    # Radicality (Seidenberg): every coordinate minimal polynomial squarefree.
    radical = None
    try:
        with time_limit(gb_timeout):
            check = range(1) if symmetric else range(len(gens))
            square = [
                is_squarefree(coordinate_minimal_polynomial(G, gens, i, basis))
                for i in check
            ]
        radical = all(square)
        note = " (by symmetry, one coordinate checked)" if symmetric else ""
        if radical:
            print(f"   radical: yes{note} — all {dim} solutions are DISTINCT")
        else:
            print("   radical: NO — some solutions are multiple; the distinct")
            print(f"   count is strictly below {dim}.")
    except Timeout:
        print(f"   radical test: TIMEOUT ({gb_timeout} s) — skipped")

    # Torus count (all coordinates nonzero — the count Bernstein's theorem
    # and a polyhedral homotopy's mixed volume see).
    if torus_reason is not None:
        print(f"   torus solutions: {dim} of {dim} — {torus_reason}")
    elif radical:
        # For a *radical* zero-dimensional I every local ring of C[x]/I is
        # C, so dim C[x]/(I + <x_1*...*x_n>) counts exactly the DISTINCT
        # solutions with some zero coordinate (each contributes 1; on the
        # torus the product is a unit and contributes 0). Same-variable-count
        # Groebner basis — far cheaper than the (n+1)-variable Rabinowitsch
        # saturation, which times sympy out on katsura-4.
        try:
            with time_limit(gb_timeout):
                Gz = groebner(
                    list(polys) + [prod(gens)], *gens, order="grevlex", field=True
                )
                off = len(standard_monomials(Gz, gens))
            torus = dim - off
            print(f"   torus solutions (all coordinates nonzero): {torus} of {dim}")
            if off > 0:
                print(
                    f"   -> {off} solution(s) lie OFF the torus (some"
                    " coordinate is exactly 0): a polyhedral"
                )
                print(
                    "      homotopy's mixed volume does not count these"
                    " (Bernstein counts torus roots only)."
                )
        except Timeout:
            print(f"   torus count: TIMEOUT ({gb_timeout} s) — skipped")
    else:
        # Radicality unknown/false: count with multiplicity through the
        # Rabinowitsch trick, dim C[x,t]/(I + <t*prod(x) - 1>) — the
        # localization of C[x]/I inverting prod(x).
        t = symbols("_t")
        sat = list(polys) + [t * prod(gens) - 1]
        try:
            with time_limit(gb_timeout):
                Gs = groebner(sat, *gens, t, order="grevlex", field=True)
                torus = len(standard_monomials(Gs, tuple(gens) + (t,)))
            print(
                f"   torus solutions (all coordinates nonzero, with"
                f" multiplicity): {torus} of {dim}"
            )
        except Timeout:
            print(f"   torus count: TIMEOUT ({gb_timeout} s) — skipped")

    if list_solutions:
        try:
            with time_limit(120):
                sols = solve_poly_system(polys, *gens)
            print(f"   solve_poly_system lists {len(sols)} (complete iff = dim):")
            for s in sols:
                print(f"     {tuple(s)}")
        except (Timeout, NotImplementedError) as e:
            print(f"   solve_poly_system: skipped ({type(e).__name__})")
    print()


def cyclic4_structure():
    """The known positive-dimensional structure of cyclic-4: the curves
    (a, b, -a, -b) with (ab)^2 = 1. Verified by exact substitution."""
    a, b = symbols("a b")
    polys, x = cyclic(4)
    subs = {x[0]: a, x[1]: b, x[2]: -a, x[3]: -b}
    residuals = [p.subs(subs).expand() for p in polys]
    print("   structure check: substituting (a, b, -a, -b) into cyclic-4 gives")
    print(f"     {residuals}")
    print("   i.e. the first three equations vanish identically and the last")
    print("   becomes a^2*b^2 - 1 = 0: two curves of solutions (ab = +/-1),")
    print("   confirming the positive-dimensional verdict above.")
    print()


CYCLIC_TORUS = "the last equation x_0*...*x_{n-1} = 1 forces every coordinate nonzero"
NOON_TORUS = (
    "x_i = 0 in equation i leaves 1 = 0: no solution has a zero coordinate"
)

SYSTEMS = {
    "cyclic-3": lambda: analyze(
        "cyclic-3",
        *cyclic(3),
        list_solutions=True,
        symmetric=True,
        torus_reason=CYCLIC_TORUS,
    ),
    "cyclic-4": lambda: (analyze("cyclic-4", *cyclic(4)), cyclic4_structure()),
    "cyclic-5": lambda: analyze(
        "cyclic-5", *cyclic(5), symmetric=True, torus_reason=CYCLIC_TORUS
    ),
    "katsura-3": lambda: analyze("katsura-3", *katsura(3)),
    "katsura-4": lambda: analyze("katsura-4", *katsura(4)),
    "noon-3": lambda: analyze(
        "noon-3", *noon(3), symmetric=True, torus_reason=NOON_TORUS
    ),
    "eco-4": lambda: analyze("eco-4", *eco(4)),
    "eco-5": lambda: analyze("eco-5", *eco(5)),
    "trinomial": lambda: analyze("trinomial", *trinomial()),
    "conic": lambda: analyze("conic", *conic()),
}


def main():
    names = sys.argv[1:] or list(SYSTEMS)
    for name in names:
        if name not in SYSTEMS:
            print(f"unknown system {name!r}; known: {', '.join(SYSTEMS)}")
            return 1
        SYSTEMS[name]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
