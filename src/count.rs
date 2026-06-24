//! Counter-width abstraction for histogram bucket counts.
//!
//! The [`Count`] trait abstracts over the bucket counter width. It is
//! implemented for `u32` and `u64`. The trait is sealed: it cannot be
//! implemented outside this crate.
//!
//! [`AtomicCount`] is the matching atomic-primitive trait, mapped via the
//! [`Count::Atomic`] associated type. It is implemented for `AtomicU32`
//! (paired with `u32`) and `AtomicU64` (paired with `u64`).

use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};

mod private {
    pub trait Sealed {}
}

/// A counter type usable for histogram bucket counts.
///
/// Sealed: implemented only for `u32` and `u64` inside this crate.
pub trait Count: private::Sealed + Copy + Default + Eq + Ord + std::fmt::Debug + 'static {
    /// The atomic-primitive counterpart used by `AtomicHistogram<Self>`.
    type Atomic: AtomicCount<Value = Self>;

    /// The additive identity for this counter type.
    const ZERO: Self;
    /// The multiplicative identity (used by `increment`).
    const ONE: Self;

    fn wrapping_add(self, other: Self) -> Self;
    fn wrapping_sub(self, other: Self) -> Self;
    fn checked_add(self, other: Self) -> Option<Self>;
    fn checked_sub(self, other: Self) -> Option<Self>;

    /// Widen to `u128` for partial-sum aggregation.
    fn as_u128(self) -> u128;
    /// Narrow from `u64`. Returns `None` if `v` exceeds the range of `Self`.
    fn try_from_u64(v: u64) -> Option<Self>;
}

/// Atomic counterpart of a [`Count`] type.
///
/// Sealed: implemented only for `AtomicU32` and `AtomicU64` inside this crate.
pub trait AtomicCount: private::Sealed {
    type Value: Count<Atomic = Self>;

    fn new(v: Self::Value) -> Self;
    fn load_relaxed(&self) -> Self::Value;
    fn store_relaxed(&self, v: Self::Value);
    fn fetch_add_relaxed(&self, v: Self::Value);
    fn swap_relaxed(&self, v: Self::Value) -> Self::Value;
}

impl private::Sealed for u32 {}
impl private::Sealed for u64 {}
impl private::Sealed for AtomicU32 {}
impl private::Sealed for AtomicU64 {}

impl Count for u32 {
    type Atomic = AtomicU32;
    const ZERO: Self = 0;
    const ONE: Self = 1;

    #[inline]
    fn wrapping_add(self, other: Self) -> Self {
        u32::wrapping_add(self, other)
    }
    #[inline]
    fn wrapping_sub(self, other: Self) -> Self {
        u32::wrapping_sub(self, other)
    }
    #[inline]
    fn checked_add(self, other: Self) -> Option<Self> {
        u32::checked_add(self, other)
    }
    #[inline]
    fn checked_sub(self, other: Self) -> Option<Self> {
        u32::checked_sub(self, other)
    }
    #[inline]
    fn as_u128(self) -> u128 {
        self as u128
    }
    #[inline]
    fn try_from_u64(v: u64) -> Option<Self> {
        u32::try_from(v).ok()
    }
}

impl Count for u64 {
    type Atomic = AtomicU64;
    const ZERO: Self = 0;
    const ONE: Self = 1;

    #[inline]
    fn wrapping_add(self, other: Self) -> Self {
        u64::wrapping_add(self, other)
    }
    #[inline]
    fn wrapping_sub(self, other: Self) -> Self {
        u64::wrapping_sub(self, other)
    }
    #[inline]
    fn checked_add(self, other: Self) -> Option<Self> {
        u64::checked_add(self, other)
    }
    #[inline]
    fn checked_sub(self, other: Self) -> Option<Self> {
        u64::checked_sub(self, other)
    }
    #[inline]
    fn as_u128(self) -> u128 {
        self as u128
    }
    #[inline]
    fn try_from_u64(v: u64) -> Option<Self> {
        Some(v)
    }
}

impl AtomicCount for AtomicU32 {
    type Value = u32;
    #[inline]
    fn new(v: u32) -> Self {
        AtomicU32::new(v)
    }
    #[inline]
    fn load_relaxed(&self) -> u32 {
        self.load(Ordering::Relaxed)
    }
    #[inline]
    fn store_relaxed(&self, v: u32) {
        self.store(v, Ordering::Relaxed)
    }
    #[inline]
    fn fetch_add_relaxed(&self, v: u32) {
        self.fetch_add(v, Ordering::Relaxed);
    }
    #[inline]
    fn swap_relaxed(&self, v: u32) -> u32 {
        self.swap(v, Ordering::Relaxed)
    }
}

impl AtomicCount for AtomicU64 {
    type Value = u64;
    #[inline]
    fn new(v: u64) -> Self {
        AtomicU64::new(v)
    }
    #[inline]
    fn load_relaxed(&self) -> u64 {
        self.load(Ordering::Relaxed)
    }
    #[inline]
    fn store_relaxed(&self, v: u64) {
        self.store(v, Ordering::Relaxed)
    }
    #[inline]
    fn fetch_add_relaxed(&self, v: u64) {
        self.fetch_add(v, Ordering::Relaxed);
    }
    #[inline]
    fn swap_relaxed(&self, v: u64) -> u64 {
        self.swap(v, Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u32_const_values() {
        assert_eq!(<u32 as Count>::ZERO, 0u32);
        assert_eq!(<u32 as Count>::ONE, 1u32);
    }

    #[test]
    fn u64_const_values() {
        assert_eq!(<u64 as Count>::ZERO, 0u64);
        assert_eq!(<u64 as Count>::ONE, 1u64);
    }

    #[test]
    fn u32_wrapping_arithmetic() {
        assert_eq!(<u32 as Count>::wrapping_add(u32::MAX, 1), 0);
        assert_eq!(<u32 as Count>::wrapping_sub(0, 1), u32::MAX);
    }

    #[test]
    fn u32_checked_arithmetic() {
        assert_eq!(<u32 as Count>::checked_add(u32::MAX, 1), None);
        assert_eq!(<u32 as Count>::checked_sub(0u32, 1), None);
        assert_eq!(<u32 as Count>::checked_add(1u32, 1), Some(2));
    }

    #[test]
    fn try_from_u64_narrowing() {
        assert_eq!(<u32 as Count>::try_from_u64(42), Some(42u32));
        assert_eq!(
            <u32 as Count>::try_from_u64(u32::MAX as u64),
            Some(u32::MAX)
        );
        assert_eq!(<u32 as Count>::try_from_u64(u32::MAX as u64 + 1), None);
        assert_eq!(<u64 as Count>::try_from_u64(u64::MAX), Some(u64::MAX));
    }

    #[test]
    fn as_u128_widening() {
        assert_eq!(<u32 as Count>::as_u128(u32::MAX), u32::MAX as u128);
        assert_eq!(<u64 as Count>::as_u128(u64::MAX), u64::MAX as u128);
    }

    #[test]
    fn atomic_u32_basic() {
        let a = <AtomicU32 as AtomicCount>::new(0);
        a.fetch_add_relaxed(5);
        assert_eq!(a.load_relaxed(), 5);
        let prev = a.swap_relaxed(10);
        assert_eq!(prev, 5);
        assert_eq!(a.load_relaxed(), 10);
    }

    #[test]
    fn atomic_u64_basic() {
        let a = <AtomicU64 as AtomicCount>::new(0);
        a.fetch_add_relaxed(5);
        assert_eq!(a.load_relaxed(), 5);
    }
}
