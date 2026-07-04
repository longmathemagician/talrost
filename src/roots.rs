//! A counted, bounded collection of real roots.
//!
//! Root finders know an upper bound on the number of real roots at compile
//! time (`MAX = degree`) but only discover the actual count at runtime.
//! `Roots` stores that count explicitly instead of padding a fixed array with
//! NaN sentinels — NaN-as-"no root" breaks `PartialEq`, is indistinguishable
//! from a genuinely-NaN computation, and forces every caller to re-scan the
//! array.

use core::ops::Deref;

/// Up to `MAX` roots, with the live count in `len`. Only the first `len`
/// entries are meaningful; everything (equality, iteration, indexing, `Debug`)
/// operates on that live prefix.
#[derive(Clone, Copy)]
pub struct Roots<T, const MAX: usize> {
    buf: [T; MAX],
    len: usize,
}

impl<T, const MAX: usize> Roots<T, MAX> {
    /// Solver-facing constructor: the first `len` entries of `buf` are the
    /// roots found. Panics if `len > MAX`.
    pub fn from_buf(buf: [T; MAX], len: usize) -> Self {
        assert!(len <= MAX, "Roots::from_buf: len {} exceeds MAX {}", len, MAX);
        Self { buf, len }
    }

    /// The live prefix: only the roots actually found.
    pub fn as_slice(&self) -> &[T] {
        &self.buf[..self.len]
    }

    /// Number of roots found.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<T, const MAX: usize> Deref for Roots<T, MAX> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<'a, T, const MAX: usize> IntoIterator for &'a Roots<T, MAX> {
    type Item = &'a T;
    type IntoIter = core::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}

// Equality compares only the live prefixes; dead slots never participate.
impl<T: PartialEq, const MAX: usize> PartialEq for Roots<T, MAX> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

// Debug shows only the live prefix, like a slice.
impl<T: core::fmt::Debug, const MAX: usize> core::fmt::Debug for Roots<T, MAX> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_list().entries(self.as_slice()).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn len_and_slice() {
        let r = Roots::from_buf([1.0, 2.0, 0.0], 2);
        assert_eq!(r.len(), 2);
        assert!(!r.is_empty());
        assert_eq!(r.as_slice(), &[1.0, 2.0]);

        let e = Roots::from_buf([0.0_f64; 3], 0);
        assert_eq!(e.len(), 0);
        assert!(e.is_empty());
        assert_eq!(e.as_slice(), &[] as &[f64]);
    }

    #[test]
    fn eq_ignores_dead_slots() {
        let a = Roots::from_buf([1.0, 2.0, 99.0], 2);
        let b = Roots::from_buf([1.0, 2.0, -1.0], 2);
        assert_eq!(a, b);

        let c = Roots::from_buf([1.0, 2.0, 3.0], 3);
        assert_ne!(a, c);
    }

    #[test]
    fn deref_and_iteration() {
        let r = Roots::from_buf([3.0, 4.0, 0.0, 0.0], 2);
        // Indexing and slice methods via Deref.
        assert_eq!(r[0], 3.0);
        assert_eq!(r[1], 4.0);
        assert_eq!(r.first(), Some(&3.0));
        assert_eq!(r.iter().count(), 2);

        // IntoIterator for &Roots.
        let mut sum = 0.0;
        for x in &r {
            sum += x;
        }
        assert_eq!(sum, 7.0);
    }

    #[test]
    fn debug_shows_live_prefix_only() {
        let r = Roots::from_buf([1.0, 99.0], 1);
        assert_eq!(format!("{:?}", r), "[1.0]");
    }

    #[test]
    #[should_panic(expected = "exceeds MAX")]
    fn from_buf_rejects_overlong_len() {
        let _ = Roots::from_buf([0.0; 2], 3);
    }
}
