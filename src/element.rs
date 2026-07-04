use core::fmt::Debug;

/// The base requirement for members of the algebraic tower: a small, copyable,
/// debuggable value. `Display` is deliberately *not* required here; formatting
/// bounds live on the `Display` impls that actually format elements.
pub trait Element: Sized + Copy + Clone + Debug {}
