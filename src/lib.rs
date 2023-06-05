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
#![allow(incomplete_features)]
#![allow(soft_unstable)]

// pub mod complex;
mod float;
pub mod matrix;
pub mod number;
pub mod polynomial;
pub mod vector;
