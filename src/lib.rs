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
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

pub mod algebra;
pub mod complex;
mod display;
pub mod element;
pub mod float;
pub mod integer;
// pub mod lattice;
pub mod matrix;
pub mod natural;
pub mod number;
pub mod polynomial;
pub mod solvers;
pub mod vector;
