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
#![feature(associated_type_bounds)]
#![feature(generic_const_exprs)]
#![feature(generic_arg_infer)]
#![feature(more_qualified_paths)]
#![feature(const_float_bits_conv)]
// #![feature(negative_impls)]
#![feature(min_specialization)]
#![allow(incomplete_features)]
#![allow(soft_unstable)]

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
