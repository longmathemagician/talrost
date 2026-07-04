// TODO: Backfill embedded support and disable std
// #![no_main]
// #![no_std]
// #[cfg(debug_assertions)]
// #[panic_handler]
// fn panic(_info: &core::panic::PanicInfo) -> ! {
//     loop {}
// }
// #[cfg(not(debug_assertions))]
// extern crate panic_semihosting;

// The algebraic tower.
pub mod algebra;
pub mod element;

// Numeric families: unsigned/signed machine integers, IEEE floats, and the
// Scalar abstraction (field with a real-valued norm) over floats and complex.
pub mod complex;
pub mod integer;
pub mod natural;
pub mod real;
pub mod scalar;

// Containers and solvers.
mod display;
pub mod matrix;
pub mod polynomial;
pub mod roots;
pub mod solvers;
pub mod vector;
