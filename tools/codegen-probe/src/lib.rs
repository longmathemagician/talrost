//! Codegen probe for tools/check_codegen.sh.
//!
//! Each `#[unsafe(no_mangle)]` wrapper pins a hot talrost entry point under a
//! stable symbol name so the guard script can locate its body in the emitted
//! assembly and assert that no `call` instruction appears — i.e. that the
//! whole evaluation inlined down to straight-line arithmetic.
//!
//! This is the check that would have caught the 5.7× `mul_add` → libm
//! software-fma cliff: on an x86-64 target without `+fma`, a fused `mul_add`
//! compiles to a `call fma@PLT` in exactly these bodies.

use talrost::polynomial::Polynomial;

/// `Polynomial::<f64, 4>::eval` (Horner with `mul_add_fast`): must compile
/// to mul/add — or FMA instructions under `-C target-feature=+fma` — with
/// no libm calls.
#[unsafe(no_mangle)]
pub extern "C" fn probe_poly_eval_f64_n4(c: &[f64; 4], x: f64) -> f64 {
    Polynomial::new(*c).eval(x)
}

/// `Polynomial::<f64, 4>::eval_at::<f64>` (the generic `Algebra` Horner
/// body, plain mul+add): must also stay call-free.
#[unsafe(no_mangle)]
pub extern "C" fn probe_poly_eval_at_f64_n4(c: &[f64; 4], x: f64) -> f64 {
    Polynomial::new(*c).eval_at(x)
}
