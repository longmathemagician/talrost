#!/usr/bin/env bash
# Codegen guard: compile the probe crate (tools/codegen-probe) to assembly
# and FAIL if any call instruction lands inside the probed function bodies.
#
# The probes wrap Polynomial::<f64, 4>::eval and ::eval_at::<f64>; both must
# inline down to straight-line mul/add (or FMA) arithmetic. A `call` in
# either body means an optimization cliff — most likely `mul_add` falling
# back to a libm software-fma call, the regression this script exists to
# catch (it cost 5.7× on Horner evaluation before it was found by hand).
#
# The check runs twice: once for the default target configuration and once
# with `-C target-feature=+fma`, because the two configurations lower
# `mul_add_fast` through different code paths.

set -euo pipefail

cd "$(dirname "$0")/codegen-probe"

PROBES=(
    probe_poly_eval_f64_n4
    probe_poly_eval_at_f64_n4
)

# Extract the body of `$2` from the assembly file `$1` (from its label to
# the following .Lfunc_end marker) and fail if a machine call appears in it.
# Tail calls (`jmp some_function@PLT`) count: a libm fallback can be emitted
# as a tail jump in a small wrapper. Jumps to local labels (.L*) are normal
# control flow and are ignored.
check_symbol() {
    local asm="$1" sym="$2"

    # LLVM's MergeFunctions pass may emit identical probes as symbol
    # aliases ("a = b"); follow the alias chain to the real body.
    local alias hops=0
    while alias=$(sed -n "s/^$sym = \(.*\)\$/\1/p" "$asm") && [[ -n "$alias" ]]; do
        echo "  note: $sym is an alias for $alias"
        sym="$alias"
        hops=$((hops + 1))
        if [[ "$hops" -gt 4 ]]; then
            echo "FAIL: alias chain too deep for $2" >&2
            return 1
        fi
    done

    local body
    body=$(awk -v sym="$sym" '
        $0 == sym ":" { inside = 1; next }
        inside && /^\.Lfunc_end/ { inside = 0 }
        inside { print }
    ' "$asm")
    if [[ -z "$body" ]]; then
        echo "FAIL: symbol $sym not found in $asm" >&2
        return 1
    fi
    local bad
    bad=$(grep -E '^\s*(call|jmp\s+[A-Za-z_])' <<<"$body" || true)
    if [[ -n "$bad" ]]; then
        echo "FAIL: call instruction(s) inside $sym:" >&2
        echo "$bad" >&2
        return 1
    fi
    echo "  OK: $sym is call-free"
}

run_check() {
    local rustflags="$1" label="$2"
    echo "== codegen check: $label =="
    rm -f target/release/deps/codegen_probe*.s
    # Cargo keeps per-RUSTFLAGS fingerprints, so switching flag sets can
    # short-circuit to "fresh" without re-emitting the .s; force codegen.
    touch src/lib.rs
    RUSTFLAGS="$rustflags" cargo rustc --release --quiet -- --emit asm
    local asm
    asm=$(ls target/release/deps/codegen_probe*.s 2>/dev/null | head -n 1)
    if [[ -z "$asm" ]]; then
        echo "FAIL: no assembly emitted (expected target/release/deps/codegen_probe*.s)" >&2
        return 1
    fi
    local status=0
    for sym in "${PROBES[@]}"; do
        check_symbol "$asm" "$sym" || status=1
    done
    return "$status"
}

status=0
run_check "" "default target" || status=1
run_check "-C target-feature=+fma" "-C target-feature=+fma" || status=1

if [[ "$status" -ne 0 ]]; then
    echo "codegen guard: FAILED" >&2
    exit 1
fi
echo "codegen guard: all probes call-free in both configurations"
