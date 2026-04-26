# u32 Bucket Counts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `u32` as a first-class counter width across all four histogram variants via generics with a `C: Count = u64` default, preserving backward compatibility for all existing call sites.

**Architecture:** Introduce a sealed `Count` trait (with paired `AtomicCount`) that abstracts over `u32`/`u64`. Generalize `Histogram`, `AtomicHistogram`, `SparseHistogram`, `CumulativeROHistogram` to `<C: Count = u64>` so existing code continues to infer `<u64>`. Add `From` (widening) and `TryFrom` (narrowing, including direct cross-variant + narrow combined paths) conversions to support the recording → snapshot → analytics pipeline.

**Tech Stack:** Rust 2024 edition, `core::sync::atomic::{AtomicU32, AtomicU64}`, `criterion` for bench, `serde`/`schemars` (optional features), `cargo test` / `cargo bench`.

**Spec:** [docs/superpowers/specs/2026-04-26-u32-bucket-counts-design.md](../specs/2026-04-26-u32-bucket-counts-design.md)

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/count.rs` | **Create** | `Count` and `AtomicCount` sealed traits + impls for `u32`, `u64`, `AtomicU32`, `AtomicU64`. |
| `src/lib.rs` | Modify | Re-export `Count`, `AtomicCount`. Add module-level rustdoc for counter width and recommended pipeline. |
| `src/standard.rs` | Modify | Generalize `Histogram` → `Histogram<C: Count = u64>`. Generalize `Iter`, `SampleQuantiles`, `From<&SparseHistogram<C>>` impls. Add u32-specific tests. |
| `src/atomic.rs` | Modify | Generalize `AtomicHistogram` → `AtomicHistogram<C: Count = u64>`. Concrete-typed `drain` impls for `<u64>` and `<u32>` (per cfg). Add u32-specific tests. |
| `src/sparse.rs` | Modify | Generalize `SparseHistogram` → `SparseHistogram<C: Count = u64>`. Generalize `From<&Histogram<C>>` and `SampleQuantiles`. Add u32-specific tests. |
| `src/cumulative.rs` | Modify | Generalize `CumulativeROHistogram` → `CumulativeROHistogram<C: Count = u64>`. Generalize `From` impls and `SampleQuantiles`. Add u32-specific tests. |
| `src/conversions.rs` | **Create** | All cross-width and cross-variant + narrowing conversion impls (`From` for widening, `TryFrom` for narrowing). Keeps the conversion matrix in one auditable place. |
| `benches/histogram.rs` | Modify | Add `Histogram<u32>` and `AtomicHistogram<u32>` bench groups alongside existing `<u64>` cases. |
| `README.md` | Modify | New "Counter Width" and "Recommended Pipeline" sections. |
| `Cargo.toml` | Modify | Bump version to next alpha (`1.3.0-alpha.0` or higher revision if main has already advanced). |

---

## Task 1: Add `Count` and `AtomicCount` sealed traits

**Files:**
- Create: `src/count.rs`
- Modify: `src/lib.rs` (add module declaration and re-exports)

- [ ] **Step 1: Create `src/count.rs` with the sealed trait definitions**

Write to `src/count.rs`:

```rust
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
pub trait Count:
    private::Sealed + Copy + Default + Eq + Ord + std::fmt::Debug + 'static
{
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
    fn wrapping_add(self, other: Self) -> Self { u32::wrapping_add(self, other) }
    #[inline]
    fn wrapping_sub(self, other: Self) -> Self { u32::wrapping_sub(self, other) }
    #[inline]
    fn checked_add(self, other: Self) -> Option<Self> { u32::checked_add(self, other) }
    #[inline]
    fn checked_sub(self, other: Self) -> Option<Self> { u32::checked_sub(self, other) }
    #[inline]
    fn as_u128(self) -> u128 { self as u128 }
    #[inline]
    fn try_from_u64(v: u64) -> Option<Self> { u32::try_from(v).ok() }
}

impl Count for u64 {
    type Atomic = AtomicU64;
    const ZERO: Self = 0;
    const ONE: Self = 1;

    #[inline]
    fn wrapping_add(self, other: Self) -> Self { u64::wrapping_add(self, other) }
    #[inline]
    fn wrapping_sub(self, other: Self) -> Self { u64::wrapping_sub(self, other) }
    #[inline]
    fn checked_add(self, other: Self) -> Option<Self> { u64::checked_add(self, other) }
    #[inline]
    fn checked_sub(self, other: Self) -> Option<Self> { u64::checked_sub(self, other) }
    #[inline]
    fn as_u128(self) -> u128 { self as u128 }
    #[inline]
    fn try_from_u64(v: u64) -> Option<Self> { Some(v) }
}

impl AtomicCount for AtomicU32 {
    type Value = u32;
    #[inline]
    fn new(v: u32) -> Self { AtomicU32::new(v) }
    #[inline]
    fn load_relaxed(&self) -> u32 { self.load(Ordering::Relaxed) }
    #[inline]
    fn store_relaxed(&self, v: u32) { self.store(v, Ordering::Relaxed) }
    #[inline]
    fn fetch_add_relaxed(&self, v: u32) { self.fetch_add(v, Ordering::Relaxed); }
    #[inline]
    fn swap_relaxed(&self, v: u32) -> u32 { self.swap(v, Ordering::Relaxed) }
}

impl AtomicCount for AtomicU64 {
    type Value = u64;
    #[inline]
    fn new(v: u64) -> Self { AtomicU64::new(v) }
    #[inline]
    fn load_relaxed(&self) -> u64 { self.load(Ordering::Relaxed) }
    #[inline]
    fn store_relaxed(&self, v: u64) { self.store(v, Ordering::Relaxed) }
    #[inline]
    fn fetch_add_relaxed(&self, v: u64) { self.fetch_add(v, Ordering::Relaxed); }
    #[inline]
    fn swap_relaxed(&self, v: u64) -> u64 { self.swap(v, Ordering::Relaxed) }
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
        assert_eq!(<u32 as Count>::try_from_u64(u32::MAX as u64), Some(u32::MAX));
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
```

- [ ] **Step 2: Wire the module into `src/lib.rs`**

Edit `src/lib.rs` — add `mod count;` to the module list (alphabetical) and add `pub use count::{Count, AtomicCount};` to the re-export block.

Find the existing module list (after the `//! # Background` doc comment):

```rust
mod atomic;
mod bucket;
mod config;
mod cumulative;
mod errors;
mod quantile;
mod sparse;
mod standard;
```

Replace with:

```rust
mod atomic;
mod bucket;
mod config;
mod count;
mod cumulative;
mod errors;
mod quantile;
mod sparse;
mod standard;
```

Find the existing re-exports:

```rust
pub use atomic::AtomicHistogram;
pub use bucket::Bucket;
pub use config::Config;
pub use cumulative::CumulativeROHistogram;
pub use errors::Error;
pub use quantile::{Quantile, QuantilesResult, SampleQuantiles};
pub use sparse::SparseHistogram;
pub use standard::Histogram;
```

Replace with:

```rust
pub use atomic::AtomicHistogram;
pub use bucket::Bucket;
pub use config::Config;
pub use count::{AtomicCount, Count};
pub use cumulative::CumulativeROHistogram;
pub use errors::Error;
pub use quantile::{Quantile, QuantilesResult, SampleQuantiles};
pub use sparse::SparseHistogram;
pub use standard::Histogram;
```

- [ ] **Step 3: Build and run tests**

Run: `cargo test count::`
Expected: 8 tests pass (the unit tests in the new module).

Run: `cargo build`
Expected: clean build, no warnings.

- [ ] **Step 4: Commit**

```bash
git add src/count.rs src/lib.rs
git commit -m "add Count and AtomicCount sealed traits

Introduces the counter-width abstraction used by the upcoming generic
histogram types. Implemented for u32 and u64 with their atomic counterparts."
```

---

## Task 2: Generalize `Histogram` to `Histogram<C: Count = u64>`

**Files:**
- Modify: `src/standard.rs` (whole-file refactor)

This is the largest task. It generalizes the dense `Histogram` type and all its methods, including the `SampleQuantiles` impl, the iterator, and the `From<&SparseHistogram>` impl. Existing tests (which use the default `<u64>`) must continue passing.

- [ ] **Step 1: Read the current file**

Run: read `src/standard.rs` end-to-end so you know exactly what's there.

- [ ] **Step 2: Apply the generalization**

Replace the contents of `src/standard.rs` (preserve test functions; the impl-block changes are mechanical):

Key changes to make (a complete diff would be hundreds of lines — apply these patterns throughout):

a) Imports — add `use crate::Count;` near the top.

b) Struct: `pub struct Histogram` → `pub struct Histogram<C: Count = u64>` with `buckets: Box<[C]>`.

c) `impl Histogram` → `impl<C: Count> Histogram<C>` for all method blocks.

d) Constructors:
   - `with_config`: `vec![0; n]` → `vec![C::ZERO; n]`.
   - `from_buckets`: parameter `buckets: Vec<u64>` → `buckets: Vec<C>`.

e) `increment`: `self.add(value, 1)` → `self.add(value, C::ONE)`.

f) `add`: signature changes to `count: C`. The body line `self.buckets[index].wrapping_add(count)` already works because we're calling the trait method (it dispatches to `C::wrapping_add`).

g) `as_slice` / `as_mut_slice`: return `&[C]` / `&mut [C]`.

h) `checked_add`/`wrapping_add`/`checked_sub`/`wrapping_sub`: signature takes `&Histogram<C>`, returns `Result<Histogram<C>, Error>`. Bodies are unchanged because we call trait methods on `*this` and `*other`.

i) `downsample`: same — returns `Histogram<C>`. The `histogram.add(val, *n)?` line works as-is (`*n` is `C`).

j) `config`: unchanged (returns `Config`, which is non-generic).

k) `iter`: returns `Iter<'_, C>`.

l) Deprecated `percentile`/`percentiles`: keep as-is, but add `<C: Count>` bound on the impl block. Body uses `SampleQuantiles::quantiles` which we'll generalize next.

m) `SampleQuantiles for Histogram`: change to `impl<C: Count> SampleQuantiles for Histogram<C>`. In the body:
   - `let total_count: u128 = self.buckets.iter().map(|v| *v as u128).sum();` → `... .map(|v| v.as_u128()).sum();`
   - In the loop: `*count > 0` → `*count != C::ZERO` (where `count` is the loop variable). Wait — looking at the actual code, the comparison is `count > 0`. Replace with `*count != C::ZERO`. Actually re-checking: the variable is `count` referring to `&u64`, comparison is `*count > 0` in some places, `count > 0` in others. Convert all numeric literal comparisons to `C::ZERO`.
   - When constructing `Bucket { count: self.buckets[min_idx], ... }`: widen via `count: self.buckets[min_idx].as_u128() as u64`. Same for max_idx and inside the quantile loop.
   - The `partial_sum` uses `self.buckets[bucket_idx] as u128` → `self.buckets[bucket_idx].as_u128()`.

n) `Iter` struct: `pub struct Iter<'a, C: Count = u64>` with field `histogram: &'a Histogram<C>`. The `next` impl widens to `u64` for the `Bucket`: `count: self.histogram.buckets[self.index].as_u128() as u64`.

o) `IntoIterator for &'a Histogram` → `impl<'a, C: Count> IntoIterator for &'a Histogram<C>` with `IntoIter = Iter<'a, C>`.

p) `From<&SparseHistogram> for Histogram` → `impl<C: Count> From<&SparseHistogram<C>> for Histogram<C>`.

The `#[cfg(target_pointer_width = "64")] fn size()` test gates on `Histogram` (default `<u64>`) — keep it as-is, it tests the default-instantiation size.

- [ ] **Step 3: Verify the file compiles**

Run: `cargo build`
Expected: clean build. If borrow-checker complains about `count: self.buckets[i].as_u128() as u64`, the issue is usually parens/precedence — explicit `(self.buckets[i].as_u128()) as u64`.

- [ ] **Step 4: Run all existing tests**

Run: `cargo test --lib`
Expected: all existing tests in `standard::tests` pass (they exercise the `<u64>` default path). All other module tests pass too.

- [ ] **Step 5: Add u32-specific tests**

Append to `src/standard.rs` inside `mod tests`:

```rust
#[cfg(target_pointer_width = "64")]
#[test]
fn size_u32() {
    assert_eq!(std::mem::size_of::<Histogram<u32>>(), 48);
}

#[test]
fn increment_u32() {
    let mut h = Histogram::<u32>::new(7, 64).unwrap();
    h.increment(5).unwrap();
    h.increment(5).unwrap();
    h.increment(5).unwrap();
    let result = h.quantile(0.5).unwrap().unwrap();
    let q = Quantile::new(0.5).unwrap();
    assert_eq!(result.get(&q).unwrap().count(), 3u64);
}

#[test]
fn add_u32_wraps_at_max() {
    let mut h = Histogram::<u32>::new(2, 4).unwrap();
    h.add(1, u32::MAX).unwrap();
    // wrapping_add: u32::MAX + 1 = 0
    h.add(1, 1).unwrap();
    assert_eq!(h.as_slice()[1], 0u32);
}

#[test]
fn checked_add_u32_overflow() {
    let mut h1 = Histogram::<u32>::new(1, 3).unwrap();
    let mut h2 = Histogram::<u32>::new(1, 3).unwrap();
    h1.as_mut_slice()[0] = u32::MAX;
    h2.as_mut_slice()[0] = 1;
    assert_eq!(h1.checked_add(&h2), Err(Error::Overflow));
}

#[test]
fn wrapping_add_u32_wraps() {
    let mut h1 = Histogram::<u32>::new(1, 3).unwrap();
    let mut h2 = Histogram::<u32>::new(1, 3).unwrap();
    h1.as_mut_slice()[0] = u32::MAX;
    h2.as_mut_slice()[0] = 1;
    let r = h1.wrapping_add(&h2).unwrap();
    assert_eq!(r.as_slice()[0], 0u32);
}

#[test]
fn checked_sub_u32_underflow() {
    let mut h1 = Histogram::<u32>::new(1, 3).unwrap();
    let mut h2 = Histogram::<u32>::new(1, 3).unwrap();
    h1.as_mut_slice()[0] = 1;
    h2.as_mut_slice()[0] = 2;
    assert_eq!(h1.checked_sub(&h2), Err(Error::Underflow));
}

#[test]
fn from_buckets_u32() {
    let buckets: Vec<u32> = vec![0; Config::new(2, 4).unwrap().total_buckets()];
    let h = Histogram::<u32>::from_buckets(2, 4, buckets).unwrap();
    assert_eq!(h.as_slice().len(), 12);
}

#[test]
fn quantiles_u32_match_u64() {
    let mut h32 = Histogram::<u32>::new(7, 64).unwrap();
    let mut h64 = Histogram::<u64>::new(7, 64).unwrap();
    for v in 1..=100u64 {
        h32.increment(v).unwrap();
        h64.increment(v).unwrap();
    }
    let q32 = h32.quantiles(&[0.5, 0.9, 0.99]).unwrap().unwrap();
    let q64 = h64.quantiles(&[0.5, 0.9, 0.99]).unwrap().unwrap();
    for (k32, k64) in q32.entries().iter().zip(q64.entries().iter()) {
        assert_eq!(k32.0, k64.0);
        assert_eq!(k32.1.range(), k64.1.range());
        assert_eq!(k32.1.count(), k64.1.count());
    }
}

#[test]
fn iter_u32_widens_count_to_u64() {
    let mut h = Histogram::<u32>::new(2, 4).unwrap();
    h.add(1, 5u32).unwrap();
    let bucket = h.iter().find(|b| b.count() > 0).unwrap();
    // Bucket.count() is u64 regardless of C
    let count: u64 = bucket.count();
    assert_eq!(count, 5);
}
```

- [ ] **Step 6: Run new tests**

Run: `cargo test --lib standard::tests::`
Expected: all existing tests pass plus 8 new u32 tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/standard.rs
git commit -m "generalize Histogram to Histogram<C: Count = u64>

Existing call sites continue to infer Histogram<u64> via the default.
Bucket.count remains u64 regardless of C; counts are widened via
Count::as_u128() at the API boundary."
```

---

## Task 3: Generalize `AtomicHistogram` to `AtomicHistogram<C: Count = u64>`

**Files:**
- Modify: `src/atomic.rs`

- [ ] **Step 1: Read the current file**

Run: read `src/atomic.rs` end-to-end.

- [ ] **Step 2: Apply the generalization**

Replace the file with the generalized version. Key changes:

a) Imports: replace `use core::sync::atomic::{AtomicU64, Ordering};` with `use crate::{AtomicCount, Count};` (atomic primitives now come through the trait). The `Ordering` import is no longer needed because atomic operations go through `AtomicCount` methods.

b) Struct:
```rust
pub struct AtomicHistogram<C: Count = u64> {
    config: Config,
    buckets: Box<[C::Atomic]>,
}
```

c) Constructors:
```rust
impl<C: Count> AtomicHistogram<C> {
    pub fn new(grouping_power: u8, max_value_power: u8) -> Result<Self, Error> {
        let config = Config::new(grouping_power, max_value_power)?;
        Ok(Self::with_config(&config))
    }

    pub fn with_config(config: &Config) -> Self {
        let mut buckets = Vec::with_capacity(config.total_buckets());
        buckets.resize_with(config.total_buckets(), || <C::Atomic as AtomicCount>::new(C::ZERO));
        Self { config: *config, buckets: buckets.into() }
    }

    pub fn increment(&self, value: u64) -> Result<(), Error> {
        self.add(value, C::ONE)
    }

    pub fn add(&self, value: u64, count: C) -> Result<(), Error> {
        let index = self.config.value_to_index(value)?;
        self.buckets[index].fetch_add_relaxed(count);
        Ok(())
    }

    pub fn config(&self) -> Config { self.config }

    pub fn load(&self) -> Histogram<C> {
        let buckets: Vec<C> = self.buckets.iter().map(|b| b.load_relaxed()).collect();
        Histogram { config: self.config, buckets: buckets.into() }
    }
}
```

Note: `Histogram { config, buckets }` requires the fields to be reachable. They are `pub(crate)` already in `standard.rs`. Good.

d) `drain` — concrete-typed impl blocks per cfg. Since we cannot place `cfg(target_has_atomic = ...)` on a generic impl predicated on `C`, split into two concrete impls:

```rust
#[cfg(target_has_atomic = "64")]
impl AtomicHistogram<u64> {
    /// Drains the bucket values into a new `Histogram<u64>`.
    ///
    /// Resets all bucket values to zero. Uses `AtomicU64::swap`. Available
    /// only on platforms that support 64-bit atomics.
    pub fn drain(&self) -> Histogram<u64> {
        let buckets: Vec<u64> = self.buckets.iter().map(|b| b.swap_relaxed(0)).collect();
        Histogram { config: self.config, buckets: buckets.into() }
    }
}

#[cfg(target_has_atomic = "32")]
impl AtomicHistogram<u32> {
    /// Drains the bucket values into a new `Histogram<u32>`.
    ///
    /// Resets all bucket values to zero. Uses `AtomicU32::swap`. Available
    /// only on platforms that support 32-bit atomics (more widely supported
    /// than 64-bit).
    pub fn drain(&self) -> Histogram<u32> {
        let buckets: Vec<u32> = self.buckets.iter().map(|b| b.swap_relaxed(0)).collect();
        Histogram { config: self.config, buckets: buckets.into() }
    }
}
```

e) Debug impl:
```rust
impl<C: Count> std::fmt::Debug for AtomicHistogram<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AtomicHistogram")
            .field("config", &self.config)
            .finish()
    }
}
```

- [ ] **Step 3: Run all existing tests**

Run: `cargo test --lib atomic::`
Expected: all existing atomic tests pass (they exercise default `<u64>`).

- [ ] **Step 4: Add u32-specific tests**

Append to `src/atomic.rs` inside `mod tests`:

```rust
#[cfg(target_pointer_width = "64")]
#[test]
fn size_u32() {
    assert_eq!(std::mem::size_of::<AtomicHistogram<u32>>(), 48);
}

#[test]
fn increment_u32_load() {
    let h = AtomicHistogram::<u32>::new(7, 64).unwrap();
    for v in 0..=100u64 {
        h.increment(v).unwrap();
    }
    let snap = h.load();
    let result = snap.quantile(0.5).unwrap().unwrap();
    let q = Quantile::new(0.5).unwrap();
    assert_eq!(result.get(&q).unwrap().end(), 50);
}

#[cfg(target_has_atomic = "32")]
#[test]
fn drain_u32() {
    let h = AtomicHistogram::<u32>::new(7, 64).unwrap();
    for v in 0..=100u64 {
        h.increment(v).unwrap();
    }
    let snap = h.drain();
    let result = snap.quantile(0.5).unwrap().unwrap();
    let q = Quantile::new(0.5).unwrap();
    assert_eq!(result.get(&q).unwrap().end(), 50);
    // After drain, the recorder is empty
    let snap2 = h.load();
    assert_eq!(snap2.quantile(0.5).unwrap(), None);
}

#[test]
fn add_u32_wraps_at_max() {
    let h = AtomicHistogram::<u32>::new(2, 4).unwrap();
    h.add(1, u32::MAX).unwrap();
    h.add(1, 1).unwrap();
    let snap = h.load();
    assert_eq!(snap.as_slice()[1], 0u32);
}
```

- [ ] **Step 5: Run new tests**

Run: `cargo test --lib atomic::`
Expected: all existing tests plus 4 new u32 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/atomic.rs
git commit -m "generalize AtomicHistogram to AtomicHistogram<C: Count = u64>

drain() is split into concrete-typed impl blocks for u32 and u64 because
cfg cannot gate generic impls on type parameters. AtomicHistogram<u32>::drain
is available on more targets than AtomicHistogram<u64>::drain."
```

---

## Task 4: Generalize `SparseHistogram` to `SparseHistogram<C: Count = u64>`

**Files:**
- Modify: `src/sparse.rs`

- [ ] **Step 1: Read the current file**

Run: read `src/sparse.rs` end-to-end.

- [ ] **Step 2: Apply the generalization**

Key changes:

a) Imports: add `use crate::Count;`.

b) Struct:
```rust
pub struct SparseHistogram<C: Count = u64> {
    pub(crate) config: Config,
    pub(crate) index: Vec<u32>,
    pub(crate) count: Vec<C>,
}
```

c) `impl SparseHistogram` → `impl<C: Count> SparseHistogram<C>` for the main impl block.

d) `from_parts(config, index, count)`: signature takes `count: Vec<C>`. The check `if c == 0` becomes `if c == C::ZERO`.

e) `into_parts(self)` returns `(Config, Vec<u32>, Vec<C>)`.

f) `count(&self) -> &[C]`.

g) `add_bucket(&mut self, idx: u32, n: C)`: parameter type `n: C`; check `if n != C::ZERO`.

h) `checked_add(&self, h: &SparseHistogram<C>)`: returns `Result<SparseHistogram<C>, Error>`. Body uses `Count::checked_add` already (via trait method dispatch on `v1.checked_add(v2)`).

i) `wrapping_add` / `wrapping_sub` / `checked_sub`: same signature changes, bodies unchanged.

j) `downsample(&self, grouping_power: u8) -> Result<SparseHistogram<C>, Error>`. Body:
   - `aggregating_count: u64` → `aggregating_count: C`
   - `aggregating_count.wrapping_add(*n)` → `aggregating_count.wrapping_add(*n)` (already trait dispatch)
   - Initial value `0u64` → `C::ZERO`.

k) `iter()` returns `Iter<'_, C>`.

l) `SampleQuantiles for SparseHistogram` → `impl<C: Count> SampleQuantiles for SparseHistogram<C>`. Body:
   - `let total_count: u128 = self.count.iter().map(|v| *v as u128).sum();` → `... map(|v| v.as_u128()).sum();`
   - Bucket construction: `count: self.count[idx]` → `count: self.count[idx].as_u128() as u64` (in min, max, and inside the loop).
   - `partial_sum` calculation: same `as u128` → `.as_u128()` translation.

m) `Iter`: `pub struct Iter<'a, C: Count = u64> { index: usize, histogram: &'a SparseHistogram<C> }`. The `next` impl widens count via `.as_u128() as u64`.

n) `IntoIterator for &'a SparseHistogram` → `impl<'a, C: Count> IntoIterator for &'a SparseHistogram<C>` with `IntoIter = Iter<'a, C>`.

o) `From<&Histogram> for SparseHistogram` → `impl<C: Count> From<&Histogram<C>> for SparseHistogram<C>`. The body's `if *n > 0` becomes `if *n != C::ZERO`.

- [ ] **Step 3: Run all existing tests**

Run: `cargo test --lib sparse::`
Expected: all existing tests pass.

- [ ] **Step 4: Add u32-specific tests**

Append to `src/sparse.rs` inside `mod tests`:

```rust
#[test]
fn from_parts_u32() {
    let config = Config::new(7, 32).unwrap();
    let h = SparseHistogram::<u32>::from_parts(config, vec![1, 3, 5], vec![6u32, 12, 7]).unwrap();
    assert_eq!(h.index(), &[1, 3, 5]);
    assert_eq!(h.count(), &[6u32, 12, 7]);
}

#[test]
fn checked_add_u32_overflow() {
    let config = Config::new(7, 32).unwrap();
    let h1 = SparseHistogram::<u32>::from_parts(config, vec![1], vec![u32::MAX]).unwrap();
    let h2 = SparseHistogram::<u32>::from_parts(config, vec![1], vec![1u32]).unwrap();
    assert_eq!(h1.checked_add(&h2), Err(Error::Overflow));
}

#[test]
fn wrapping_add_u32_wraps() {
    let config = Config::new(7, 32).unwrap();
    let h1 = SparseHistogram::<u32>::from_parts(config, vec![1], vec![u32::MAX]).unwrap();
    let h2 = SparseHistogram::<u32>::from_parts(config, vec![1], vec![1u32]).unwrap();
    let h = h1.wrapping_add(&h2).unwrap();
    // Wraps to 0; add_bucket skips zero-count entries
    assert!(h.index().is_empty());
}

#[test]
fn from_histogram_u32() {
    let mut h = Histogram::<u32>::new(7, 64).unwrap();
    h.increment(1).unwrap();
    h.increment(5).unwrap();
    h.increment(100).unwrap();
    let s = SparseHistogram::<u32>::from(&h);
    assert_eq!(s.count().len(), 3);
}

#[test]
fn quantiles_u32_match_u64() {
    let mut h32 = Histogram::<u32>::new(4, 10).unwrap();
    let mut h64 = Histogram::<u64>::new(4, 10).unwrap();
    for v in 1..1024u64 {
        h32.increment(v).unwrap();
        h64.increment(v).unwrap();
    }
    let s32 = SparseHistogram::<u32>::from(&h32);
    let s64 = SparseHistogram::<u64>::from(&h64);
    let q32 = s32.quantile(0.5).unwrap().unwrap();
    let q64 = s64.quantile(0.5).unwrap().unwrap();
    let q = Quantile::new(0.5).unwrap();
    assert_eq!(q32.get(&q).unwrap().range(), q64.get(&q).unwrap().range());
}
```

- [ ] **Step 5: Run new tests**

Run: `cargo test --lib sparse::`
Expected: all existing tests plus 5 new u32 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/sparse.rs
git commit -m "generalize SparseHistogram to SparseHistogram<C: Count = u64>

Bucket counts in QuantilesResult continue to be widened to u64 via
Count::as_u128 at the API boundary."
```

---

## Task 5: Generalize `CumulativeROHistogram` to `CumulativeROHistogram<C: Count = u64>`

**Files:**
- Modify: `src/cumulative.rs`

- [ ] **Step 1: Read the current file**

Run: read `src/cumulative.rs` end-to-end.

- [ ] **Step 2: Apply the generalization**

Key changes:

a) Imports: add `use crate::Count;`.

b) Struct:
```rust
pub struct CumulativeROHistogram<C: Count = u64> {
    config: Config,
    index: Vec<u32>,
    count: Vec<C>,
}
```

c) `CACHE_LINE_U64S` constant — rename to `CACHE_LINE_ENTRIES` and compute via `64 / std::mem::size_of::<C>()`. Since this is now C-dependent, make it a const fn or compute at use:

```rust
const fn cache_line_entries<C>() -> usize {
    64 / std::mem::size_of::<C>()
}
```

Use as `cache_line_entries::<C>()` in `find_quantile_position`.

d) `impl CumulativeROHistogram` → `impl<C: Count> CumulativeROHistogram<C>` for the main block.

e) `from_parts(config, index, count: Vec<C>)`: validation:
   - `if c == 0` → `if c == C::ZERO`.
   - `if c < p` (non-decreasing check): keep, `<` works for `Ord`.

f) `into_parts(self) -> (Config, Vec<u32>, Vec<C>)`.

g) `count(&self) -> &[C]`.

h) `total_count(&self) -> u64`: widen via `.as_u128() as u64`:

```rust
pub fn total_count(&self) -> u64 {
    self.count.last().map(|c| c.as_u128() as u64).unwrap_or(0)
}
```

i) `bucket_quantile_range`: replace `as f64` with `.as_u128() as f64`:

```rust
let total = self.count.last().copied()?.as_u128() as f64;
// ...
let lower = if bucket_idx == 0 {
    0.0
} else {
    self.count[bucket_idx - 1].as_u128() as f64 / total
};
let upper = self.count[bucket_idx].as_u128() as f64 / total;
```

j) `iter_with_quantiles`: same f64 conversion via `.as_u128() as f64`.

k) `iter()` returns `Iter<'_, C>`.

l) `individual_count` private fn — keep returning `u64` but compute through `C` first:

```rust
fn individual_count(&self, position: usize) -> u64 {
    if position == 0 {
        self.count[0].as_u128() as u64
    } else {
        // self.count[position] >= self.count[position - 1] by invariant,
        // so wrapping_sub yields the correct difference.
        self.count[position].wrapping_sub(self.count[position - 1]).as_u128() as u64
    }
}
```

m) `find_quantile_position`: use `cache_line_entries::<C>()` and `c.as_u128() as u128 >= target`:

```rust
fn find_quantile_position(&self, target: u128) -> usize {
    if self.count.len() <= cache_line_entries::<C>() {
        for (i, c) in self.count.iter().enumerate() {
            if c.as_u128() >= target {
                return i;
            }
        }
        self.count.len() - 1
    } else {
        let pos = self.count.partition_point(|c| c.as_u128() < target);
        pos.min(self.count.len() - 1)
    }
}
```

n) `SampleQuantiles for CumulativeROHistogram` → `impl<C: Count> SampleQuantiles for CumulativeROHistogram<C>`. Body:
   - `*self.count.last().unwrap() as u128` → `self.count.last().unwrap().as_u128()`.
   - `self.count[0]` in min Bucket → `self.count[0].as_u128() as u64`.
   - Use `self.individual_count(last)` for max bucket count (already u64).
   - Use `self.individual_count(pos)` for quantile-result bucket counts (already u64).

o) `Iter` and `QuantileRangeIter`: parameterize over `<'a, C: Count>`. The `next` impls use `self.histogram.individual_count(i)` (already u64) for `Bucket.count`.

p) `IntoIterator for &'a CumulativeROHistogram` → `impl<'a, C: Count> IntoIterator for &'a CumulativeROHistogram<C>`.

q) `From<&Histogram> for CumulativeROHistogram` → `impl<C: Count> From<&Histogram<C>> for CumulativeROHistogram<C>`. Body:
   - `running_sum: u64` → `running_sum: C`.
   - `running_sum = running_sum.wrapping_add(n)` → already trait dispatch, works.
   - `if n > 0` → `if n != C::ZERO`.
   - `count.push(running_sum)` works for any `C: Copy`.
   - Initial `running_sum = 0u64` → `running_sum = C::ZERO`.

r) `From<&SparseHistogram> for CumulativeROHistogram` → `impl<C: Count> From<&SparseHistogram<C>> for CumulativeROHistogram<C>`. Same `running_sum: C` change.

- [ ] **Step 3: Run all existing tests**

Run: `cargo test --lib cumulative::`
Expected: all existing tests pass.

- [ ] **Step 4: Add u32-specific tests**

Append to `src/cumulative.rs` inside `mod tests`:

```rust
#[test]
fn from_histogram_u32() {
    let mut h = Histogram::<u32>::new(7, 64).unwrap();
    h.increment(1).unwrap();
    h.increment(1).unwrap();
    h.increment(5).unwrap();
    h.increment(100).unwrap();
    let croh = CumulativeROHistogram::<u32>::from(&h);
    assert_eq!(croh.index().len(), 3);
    assert_eq!(croh.count(), &[2u32, 3, 4]);
    assert_eq!(croh.total_count(), 4);
}

#[test]
fn from_parts_u32() {
    let config = Config::new(7, 32).unwrap();
    let croh =
        CumulativeROHistogram::<u32>::from_parts(config, vec![1, 3, 5], vec![6u32, 18, 25])
            .unwrap();
    assert_eq!(croh.total_count(), 25);
}

#[test]
fn quantiles_u32_match_u64() {
    let mut h32 = Histogram::<u32>::new(4, 10).unwrap();
    let mut h64 = Histogram::<u64>::new(4, 10).unwrap();
    for v in 1..1024u64 {
        h32.increment(v).unwrap();
        h64.increment(v).unwrap();
    }
    let c32 = CumulativeROHistogram::<u32>::from(&h32);
    let c64 = CumulativeROHistogram::<u64>::from(&h64);
    let qs = &[0.0, 0.5, 0.99, 1.0];
    let r32 = c32.quantiles(qs).unwrap().unwrap();
    let r64 = c64.quantiles(qs).unwrap().unwrap();
    for ((q32, _), (q64, _)) in r32.entries().iter().zip(r64.entries().iter()) {
        assert_eq!(q32, q64);
    }
}

#[test]
fn individual_count_u32() {
    let config = Config::new(7, 32).unwrap();
    let croh =
        CumulativeROHistogram::<u32>::from_parts(config, vec![1, 3, 5], vec![10u32, 40, 100])
            .unwrap();
    let buckets: Vec<_> = croh.iter().collect();
    assert_eq!(buckets[0].count(), 10);
    assert_eq!(buckets[1].count(), 30);
    assert_eq!(buckets[2].count(), 60);
}
```

- [ ] **Step 5: Run new tests**

Run: `cargo test --lib cumulative::`
Expected: all existing tests plus 4 new u32 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/cumulative.rs
git commit -m "generalize CumulativeROHistogram to CumulativeROHistogram<C: Count = u64>

individual_count and total_count continue to return u64 (widened via
Count::as_u128) so the public API does not gain a generic Bucket type."
```

---

## Task 6: Cross-width same-variant `From` (widening) impls

**Files:**
- Create: `src/conversions.rs`
- Modify: `src/lib.rs` (add `mod conversions;`)

All cross-width and combined cross-variant + narrowing conversions live in one module so the conversion matrix is reviewable in one place.

- [ ] **Step 1: Create `src/conversions.rs` with widening impls**

Write to `src/conversions.rs`:

```rust
//! Cross-width and combined cross-variant + narrowing conversions
//! between histogram variants.
//!
//! - Widening (`u32` → `u64`) is infallible and exposed via `From`.
//! - Narrowing (`u64` → `u32`) is fallible (`Err(Error::Overflow)`) and
//!   exposed via `TryFrom`. Direct cross-variant + narrowing paths
//!   (e.g. `Histogram<u64>` → `CumulativeROHistogram<u32>`) are also
//!   provided for the snapshot pipeline.

use crate::{
    AtomicCount, AtomicHistogram, Count, CumulativeROHistogram, Error,
    Histogram, SparseHistogram,
};

// ---------------- Widening (u32 -> u64) ----------------

impl From<&Histogram<u32>> for Histogram<u64> {
    fn from(h: &Histogram<u32>) -> Self {
        let buckets: Vec<u64> = h.as_slice().iter().map(|&c| c as u64).collect();
        Histogram::from_buckets(
            h.config().grouping_power(),
            h.config().max_value_power(),
            buckets,
        )
        .expect("widening preserves bucket count")
    }
}

impl From<&AtomicHistogram<u32>> for AtomicHistogram<u64> {
    fn from(h: &AtomicHistogram<u32>) -> Self {
        // Snapshot the source via load(), widen, then materialize as atomic.
        let snapshot = h.load(); // Histogram<u32>
        let widened: Histogram<u64> = (&snapshot).into();
        let out = AtomicHistogram::<u64>::with_config(&widened.config());
        for (i, &c) in widened.as_slice().iter().enumerate() {
            // Direct slot writes via the atomic primitive.
            out.add_at_index(i, c);
        }
        out
    }
}

impl From<&SparseHistogram<u32>> for SparseHistogram<u64> {
    fn from(h: &SparseHistogram<u32>) -> Self {
        let widened: Vec<u64> = h.count().iter().map(|&c| c as u64).collect();
        SparseHistogram::<u64>::from_parts(h.config(), h.index().to_vec(), widened)
            .expect("widening preserves invariants")
    }
}

impl From<&CumulativeROHistogram<u32>> for CumulativeROHistogram<u64> {
    fn from(h: &CumulativeROHistogram<u32>) -> Self {
        let widened: Vec<u64> = h.count().iter().map(|&c| c as u64).collect();
        CumulativeROHistogram::<u64>::from_parts(h.config(), h.index().to_vec(), widened)
            .expect("widening preserves invariants")
    }
}
```

Note the `AtomicHistogram` widening above uses a helper `add_at_index` we don't yet have. Add a `pub(crate)` helper to `AtomicHistogram` for this purpose:

In `src/atomic.rs`, add inside `impl<C: Count> AtomicHistogram<C>`:

```rust
pub(crate) fn add_at_index(&self, index: usize, count: C) {
    self.buckets[index].fetch_add_relaxed(count);
}
```

- [ ] **Step 2: Wire the module into `src/lib.rs`**

Add `mod conversions;` to the module list in `src/lib.rs` (no re-exports needed — the impls are picked up by trait coherence).

- [ ] **Step 3: Build and run existing tests**

Run: `cargo build`
Expected: clean.

Run: `cargo test --lib`
Expected: all prior tests still pass.

- [ ] **Step 4: Add widening tests**

Append to `src/conversions.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn widen_histogram() {
        let mut h32 = Histogram::<u32>::new(7, 32).unwrap();
        h32.add(1, 1234u32).unwrap();
        h32.add(1000, 5678u32).unwrap();
        let h64: Histogram<u64> = (&h32).into();
        assert_eq!(h64.config(), h32.config());
        // Compare slice values widened
        for (a, b) in h64.as_slice().iter().zip(h32.as_slice().iter()) {
            assert_eq!(*a, *b as u64);
        }
    }

    #[test]
    fn widen_sparse() {
        let config = crate::Config::new(7, 32).unwrap();
        let s32 = SparseHistogram::<u32>::from_parts(config, vec![1, 3], vec![10u32, 20]).unwrap();
        let s64: SparseHistogram<u64> = (&s32).into();
        assert_eq!(s64.count(), &[10u64, 20]);
        assert_eq!(s64.index(), &[1u32, 3]);
    }

    #[test]
    fn widen_cumulative() {
        let config = crate::Config::new(7, 32).unwrap();
        let c32 =
            CumulativeROHistogram::<u32>::from_parts(config, vec![1, 3], vec![10u32, 30]).unwrap();
        let c64: CumulativeROHistogram<u64> = (&c32).into();
        assert_eq!(c64.count(), &[10u64, 30]);
    }

    #[cfg(target_has_atomic = "32")]
    #[cfg(target_has_atomic = "64")]
    #[test]
    fn widen_atomic_histogram() {
        let h32 = AtomicHistogram::<u32>::new(7, 32).unwrap();
        h32.add(5, 100u32).unwrap();
        h32.add(50, 200u32).unwrap();
        let h64: AtomicHistogram<u64> = (&h32).into();
        let snap = h64.load();
        let total: u64 = snap.as_slice().iter().sum();
        assert_eq!(total, 300);
    }
}
```

- [ ] **Step 5: Run tests**

Run: `cargo test --lib conversions::`
Expected: 4 widening tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/conversions.rs src/lib.rs src/atomic.rs
git commit -m "add cross-width widening From impls

u32 -> u64 widening for Histogram, AtomicHistogram, SparseHistogram,
CumulativeROHistogram. Widening is infallible."
```

---

## Task 7: Cross-width same-variant `TryFrom` (narrowing) impls

**Files:**
- Modify: `src/conversions.rs`

- [ ] **Step 1: Add narrowing impls**

Append to `src/conversions.rs` (above the test module):

```rust
// ---------------- Narrowing (u64 -> u32) ----------------

impl TryFrom<&Histogram<u64>> for Histogram<u32> {
    type Error = Error;

    fn try_from(h: &Histogram<u64>) -> Result<Self, Error> {
        let mut narrowed: Vec<u32> = Vec::with_capacity(h.as_slice().len());
        for &c in h.as_slice() {
            narrowed.push(u32::try_from(c).map_err(|_| Error::Overflow)?);
        }
        Histogram::<u32>::from_buckets(
            h.config().grouping_power(),
            h.config().max_value_power(),
            narrowed,
        )
    }
}

impl TryFrom<&SparseHistogram<u64>> for SparseHistogram<u32> {
    type Error = Error;

    fn try_from(h: &SparseHistogram<u64>) -> Result<Self, Error> {
        let mut narrowed: Vec<u32> = Vec::with_capacity(h.count().len());
        for &c in h.count() {
            narrowed.push(u32::try_from(c).map_err(|_| Error::Overflow)?);
        }
        SparseHistogram::<u32>::from_parts(h.config(), h.index().to_vec(), narrowed)
    }
}

impl TryFrom<&CumulativeROHistogram<u64>> for CumulativeROHistogram<u32> {
    type Error = Error;

    fn try_from(h: &CumulativeROHistogram<u64>) -> Result<Self, Error> {
        // Cumulative-only optimization: the last (max) cumulative value bounds
        // every prefix sum. If it fits in u32, every entry fits in u32.
        if let Some(&last) = h.count().last() {
            if u32::try_from(last).is_err() {
                return Err(Error::Overflow);
            }
        }
        let narrowed: Vec<u32> = h.count().iter().map(|&c| c as u32).collect();
        CumulativeROHistogram::<u32>::from_parts(h.config(), h.index().to_vec(), narrowed)
    }
}
```

- [ ] **Step 2: Add narrowing tests**

Append to the `mod tests` block in `src/conversions.rs`:

```rust
#[test]
fn narrow_histogram_success() {
    let mut h64 = Histogram::<u64>::new(7, 32).unwrap();
    h64.add(1, 100u64).unwrap();
    h64.add(1000, 200u64).unwrap();
    let h32: Histogram<u32> = (&h64).try_into().unwrap();
    assert_eq!(h32.as_slice()[1], 100u32);
}

#[test]
fn narrow_histogram_overflow() {
    let mut h64 = Histogram::<u64>::new(2, 4).unwrap();
    h64.add(1, (u32::MAX as u64) + 1).unwrap();
    let r: Result<Histogram<u32>, _> = (&h64).try_into();
    assert_eq!(r, Err(Error::Overflow));
}

#[test]
fn narrow_sparse_overflow() {
    let config = crate::Config::new(7, 32).unwrap();
    let s64 =
        SparseHistogram::<u64>::from_parts(config, vec![1], vec![(u32::MAX as u64) + 1]).unwrap();
    let r: Result<SparseHistogram<u32>, _> = (&s64).try_into();
    assert_eq!(r, Err(Error::Overflow));
}

#[test]
fn narrow_cumulative_checks_total_only() {
    let config = crate::Config::new(7, 32).unwrap();
    // Total exceeds u32::MAX -> fail.
    let c64 = CumulativeROHistogram::<u64>::from_parts(
        config,
        vec![1, 3],
        vec![100u64, (u32::MAX as u64) + 1],
    )
    .unwrap();
    let r: Result<CumulativeROHistogram<u32>, _> = (&c64).try_into();
    assert_eq!(r, Err(Error::Overflow));

    // Total fits -> succeed (every prefix is necessarily smaller).
    let c64_ok =
        CumulativeROHistogram::<u64>::from_parts(config, vec![1, 3], vec![100u64, 200]).unwrap();
    let c32: CumulativeROHistogram<u32> = (&c64_ok).try_into().unwrap();
    assert_eq!(c32.total_count(), 200);
}

#[test]
fn round_trip_widen_then_narrow() {
    let mut h32 = Histogram::<u32>::new(7, 32).unwrap();
    h32.add(5, 1234u32).unwrap();
    h32.add(50, 5678u32).unwrap();
    let h64: Histogram<u64> = (&h32).into();
    let h32_back: Histogram<u32> = (&h64).try_into().unwrap();
    assert_eq!(h32.as_slice(), h32_back.as_slice());
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test --lib conversions::`
Expected: all widening + 5 new narrowing tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/conversions.rs
git commit -m "add cross-width narrowing TryFrom impls

u64 -> u32 narrowing for Histogram, SparseHistogram, CumulativeROHistogram.
Cumulative narrowing checks only the total count (last cumulative value)
since it bounds every prefix sum."
```

---

## Task 8: Cross-variant + narrowing combined `TryFrom` impls

**Files:**
- Modify: `src/conversions.rs`

These provide the direct snapshot-pipeline paths called out in the spec.

- [ ] **Step 1: Add combined narrowing impls**

Append to `src/conversions.rs` (above the test module):

```rust
// -------- Cross-variant + narrowing (u64 -> u32) --------

/// Direct path for the snapshot pipeline:
/// `Histogram<u64>` (delta) → `CumulativeROHistogram<u32>`.
///
/// Single pass: accumulate non-zero buckets, fail with `Error::Overflow`
/// if the running total ever exceeds `u32::MAX`.
impl TryFrom<&Histogram<u64>> for CumulativeROHistogram<u32> {
    type Error = Error;

    fn try_from(h: &Histogram<u64>) -> Result<Self, Error> {
        let mut index: Vec<u32> = Vec::new();
        let mut count: Vec<u32> = Vec::new();
        let mut running: u64 = 0;
        for (i, &n) in h.as_slice().iter().enumerate() {
            if n > 0 {
                running = running
                    .checked_add(n)
                    .ok_or(Error::Overflow)?;
                if running > u32::MAX as u64 {
                    return Err(Error::Overflow);
                }
                index.push(i as u32);
                count.push(running as u32);
            }
        }
        CumulativeROHistogram::<u32>::from_parts(h.config(), index, count)
    }
}

impl TryFrom<&Histogram<u64>> for SparseHistogram<u32> {
    type Error = Error;

    fn try_from(h: &Histogram<u64>) -> Result<Self, Error> {
        let mut index: Vec<u32> = Vec::new();
        let mut count: Vec<u32> = Vec::new();
        for (i, &n) in h.as_slice().iter().enumerate() {
            if n > 0 {
                count.push(u32::try_from(n).map_err(|_| Error::Overflow)?);
                index.push(i as u32);
            }
        }
        SparseHistogram::<u32>::from_parts(h.config(), index, count)
    }
}

impl TryFrom<&SparseHistogram<u64>> for CumulativeROHistogram<u32> {
    type Error = Error;

    fn try_from(h: &SparseHistogram<u64>) -> Result<Self, Error> {
        let mut running: u64 = 0;
        let mut count: Vec<u32> = Vec::with_capacity(h.count().len());
        for &n in h.count() {
            running = running.checked_add(n).ok_or(Error::Overflow)?;
            if running > u32::MAX as u64 {
                return Err(Error::Overflow);
            }
            count.push(running as u32);
        }
        CumulativeROHistogram::<u32>::from_parts(h.config(), h.index().to_vec(), count)
    }
}
```

- [ ] **Step 2: Add tests**

Append to the `mod tests` block in `src/conversions.rs`:

```rust
#[test]
fn histogram_u64_to_cumulative_u32() {
    let mut h = Histogram::<u64>::new(7, 32).unwrap();
    h.add(1, 100u64).unwrap();
    h.add(50, 200u64).unwrap();
    h.add(1000, 300u64).unwrap();
    let croh: CumulativeROHistogram<u32> = (&h).try_into().unwrap();
    assert_eq!(croh.total_count(), 600);
    assert_eq!(croh.count().len(), 3);
}

#[test]
fn histogram_u64_to_cumulative_u32_overflow() {
    let mut h = Histogram::<u64>::new(2, 4).unwrap();
    h.add(0, 3_000_000_000u64).unwrap();
    h.add(1, 2_000_000_000u64).unwrap(); // running > u32::MAX
    let r: Result<CumulativeROHistogram<u32>, _> = (&h).try_into();
    assert_eq!(r, Err(Error::Overflow));
}

#[test]
fn histogram_u64_to_sparse_u32() {
    let mut h = Histogram::<u64>::new(7, 32).unwrap();
    h.add(1, 100u64).unwrap();
    h.add(1000, 200u64).unwrap();
    let s: SparseHistogram<u32> = (&h).try_into().unwrap();
    assert_eq!(s.count().iter().map(|&c| c as u64).sum::<u64>(), 300);
}

#[test]
fn histogram_u64_to_sparse_u32_overflow() {
    let mut h = Histogram::<u64>::new(2, 4).unwrap();
    h.add(1, (u32::MAX as u64) + 1).unwrap();
    let r: Result<SparseHistogram<u32>, _> = (&h).try_into();
    assert_eq!(r, Err(Error::Overflow));
}

#[test]
fn sparse_u64_to_cumulative_u32() {
    let config = crate::Config::new(7, 32).unwrap();
    let s = SparseHistogram::<u64>::from_parts(config, vec![1, 3], vec![100u64, 200]).unwrap();
    let c: CumulativeROHistogram<u32> = (&s).try_into().unwrap();
    assert_eq!(c.count(), &[100u32, 300]);
}

#[test]
fn direct_path_matches_two_step() {
    // Build a moderately complex Histogram<u64>.
    let mut h = Histogram::<u64>::new(4, 10).unwrap();
    for v in 1..1024u64 {
        h.increment(v).unwrap();
    }

    // Direct: Histogram<u64> -> CumulativeROHistogram<u32>
    let direct: CumulativeROHistogram<u32> = (&h).try_into().unwrap();

    // Two-step: Histogram<u64> -> CumulativeROHistogram<u64> -> CumulativeROHistogram<u32>
    let mid: CumulativeROHistogram<u64> = (&h).into();
    let two_step: CumulativeROHistogram<u32> = (&mid).try_into().unwrap();

    assert_eq!(direct.count(), two_step.count());
    assert_eq!(direct.index(), two_step.index());
}

#[test]
fn snapshot_pipeline_end_to_end() {
    use crate::AtomicHistogram;
    let recorder = AtomicHistogram::<u64>::new(7, 64).unwrap();

    // Window 1
    for v in 1..=50u64 {
        recorder.increment(v).unwrap();
    }
    let snap_t0 = recorder.load();

    // Window 2
    for v in 1..=50u64 {
        recorder.increment(v).unwrap();
    }
    let snap_t1 = recorder.load();

    let delta = snap_t1.checked_sub(&snap_t0).unwrap();
    let analytic: CumulativeROHistogram<u32> = (&delta).try_into().unwrap();

    assert_eq!(analytic.total_count(), 50);
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test --lib conversions::`
Expected: all prior conversion tests + 7 new combined-conversion tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/conversions.rs
git commit -m "add cross-variant + narrowing TryFrom impls for snapshot pipeline

Direct paths Histogram<u64> -> CumulativeROHistogram<u32>,
Histogram<u64> -> SparseHistogram<u32>, and SparseHistogram<u64> ->
CumulativeROHistogram<u32>. Each runs in a single pass; the cumulative
target checks only the running total against u32::MAX."
```

---

## Task 9: Update benchmarks for u32

**Files:**
- Modify: `benches/histogram.rs`

- [ ] **Step 1: Add u32 bench groups**

Replace the contents of `benches/histogram.rs`:

```rust
use criterion::{Criterion, Throughput, criterion_group, criterion_main};

macro_rules! benchmark {
    ($name:tt, $histogram:ident, $c:ident) => {
        let mut group = $c.benchmark_group($name);
        group.throughput(Throughput::Elements(1));
        group.bench_function("increment/1", |b| b.iter(|| $histogram.increment(1)));
        group.bench_function("increment/max", |b| {
            b.iter(|| $histogram.increment(u64::MAX))
        });

        group.finish();
    };
}

fn histogram_u64(c: &mut Criterion) {
    let mut histogram = histogram::Histogram::<u64>::new(7, 64).unwrap();
    benchmark!("histogram/u64", histogram, c);
}

fn histogram_u32(c: &mut Criterion) {
    let mut histogram = histogram::Histogram::<u32>::new(7, 64).unwrap();
    benchmark!("histogram/u32", histogram, c);
}

fn atomic_u64(c: &mut Criterion) {
    let histogram = histogram::AtomicHistogram::<u64>::new(7, 64).unwrap();
    benchmark!("atomic_histogram/u64", histogram, c);
}

fn atomic_u32(c: &mut Criterion) {
    let histogram = histogram::AtomicHistogram::<u32>::new(7, 64).unwrap();
    benchmark!("atomic_histogram/u32", histogram, c);
}

criterion_group!(
    benches,
    histogram_u64,
    histogram_u32,
    atomic_u64,
    atomic_u32
);
criterion_main!(benches);
```

- [ ] **Step 2: Verify benches compile**

Run: `cargo bench --no-run`
Expected: clean build of the bench binary.

- [ ] **Step 3: Commit**

```bash
git add benches/histogram.rs
git commit -m "add u32 benchmark cases for Histogram and AtomicHistogram

Confirms no regression on the u64 path and gives baseline numbers for
the new u32 instantiations."
```

---

## Task 10: Documentation updates

**Files:**
- Modify: `README.md`
- Modify: `src/lib.rs`
- Modify: `src/standard.rs` (per-type rustdoc one-liner)
- Modify: `src/atomic.rs` (per-type rustdoc one-liner)
- Modify: `src/sparse.rs` (per-type rustdoc one-liner)
- Modify: `src/cumulative.rs` (per-type rustdoc one-liner)
- Modify: `src/config.rs` (footnote on memory table)

- [ ] **Step 1: Update `README.md`**

Insert two new sections between the existing "Histogram Types" and "Features" sections.

After the existing `## Histogram Types` block, insert:

```markdown
## Counter Width

All four histogram types are generic over a counter width `C: Count`,
defaulted to `u64`. Existing code (`Histogram::new(7, 64)`) keeps working
unchanged. To opt into `u32` counters — halving memory and on-the-wire
size at the cost of a 4.3 billion count ceiling per bucket — annotate the
type:

```rust
use histogram::Histogram;
let mut h = Histogram::<u32>::new(7, 64).unwrap();
```

The `Count` trait is sealed: only `u32` and `u64` are supported.

Conversions:

- **Widening** (`u32` → `u64`) is infallible and exposed via `From`.
- **Narrowing** (`u64` → `u32`) is fallible (`Err(Overflow)`) and exposed
  via `TryFrom`. Direct cross-variant + narrowing paths
  (`Histogram<u64>` → `CumulativeROHistogram<u32>`, etc.) are also provided.

## Recommended Pipeline

Pick the counter width based on the *role* the histogram plays in your data flow:

- **Recording — `AtomicHistogram<u64>` (or `Histogram<u64>`).** Use `u64`
  for the long-running, continuously-updated histogram. Counts here are
  unbounded over the lifetime of the process; `u64` heads off any
  practical risk of overflow.

- **Snapshot delta — `Histogram<u64>`, then narrowed.** When you take
  periodic snapshots and compute a delta with `checked_sub`, the delta
  covers only the activity in one window. Window counts are typically
  much smaller than lifetime counts, which is exactly when narrowing pays
  off.

- **Read-only analytics — `CumulativeROHistogram<u32>`.** This is the
  recommended storage and query format for completed snapshots. The
  cumulative-prefix-sum representation gives you O(log n) quantile queries
  via binary search, while `u32` counts halve the on-the-wire and on-disk
  size versus `u64`. Narrowing is checked once against the *total count*
  (cheaper than per-bucket), and any total ≤ ~4.3B fits.

```rust
use histogram::{AtomicHistogram, CumulativeROHistogram, Histogram};

// Recording: u64, atomic, long-lived
let recorder = AtomicHistogram::<u64>::new(7, 64).unwrap();
# let snap_t0 = recorder.load();

// Snapshot pipeline (run periodically)
let snap_t1 = recorder.load();                              // Histogram<u64>
let delta = snap_t1.checked_sub(&snap_t0).unwrap();         // Histogram<u64> — small counts
let analytic: CumulativeROHistogram<u32> =
    CumulativeROHistogram::<u32>::try_from(&delta).unwrap(); // narrow + cumulative in one pass
// analytic is now ready to ship/store/query
```

If you don't take snapshots — i.e., you query the recording histogram
directly — just stay on `u64` everywhere. The narrowing optimization is
specifically for the snapshot/delta pattern.
```

- [ ] **Step 2: Update `src/lib.rs` module rustdoc**

Mirror the same two sections in the crate-level `//!` rustdoc block. Place them after the existing `# Example` section and before `# Background`.

Find the existing `//! # Background` line in `src/lib.rs` and insert before it:

```rust
//! # Counter Width
//!
//! All four histogram types are generic over a counter width `C: Count`,
//! defaulted to `u64`. To opt into `u32` counters (halving memory at the
//! cost of a 4.3-billion-count ceiling per bucket), annotate the type:
//!
//! ```
//! use histogram::Histogram;
//! let mut h = Histogram::<u32>::new(7, 64).unwrap();
//! ```
//!
//! The [`Count`] trait is sealed: only `u32` and `u64` are supported.
//!
//! Widening (`u32` → `u64`) is infallible (`From`); narrowing
//! (`u64` → `u32`) is fallible (`TryFrom`, returning [`Error::Overflow`]).
//! Direct cross-variant + narrowing paths support the snapshot pipeline.
//!
//! # Recommended Pipeline
//!
//! Pick the counter width based on the *role* the histogram plays:
//!
//! - **Recording — `AtomicHistogram<u64>` or `Histogram<u64>`.** Counts
//!   are unbounded over the lifetime of the process; `u64` is the safe
//!   choice.
//! - **Snapshot delta — `Histogram<u64>`, then narrowed.** Compute the
//!   delta with `checked_sub`, then `TryFrom` into the analytics type.
//! - **Read-only analytics — `CumulativeROHistogram<u32>`.** Halved size,
//!   O(log n) quantile queries, total-count check is cheaper than
//!   per-bucket.
//!
//! ```
//! use histogram::{AtomicHistogram, CumulativeROHistogram, Histogram};
//!
//! let recorder = AtomicHistogram::<u64>::new(7, 64).unwrap();
//! # let snap_t0 = recorder.load();
//! let snap_t1 = recorder.load();
//! let delta = snap_t1.checked_sub(&snap_t0).unwrap();
//! let analytic: CumulativeROHistogram<u32> =
//!     CumulativeROHistogram::<u32>::try_from(&delta).unwrap();
//! ```
```

- [ ] **Step 3: Update the `# Types` list in `src/lib.rs` rustdoc**

Replace the existing types list with:

```rust
//! # Types
//!
//! - [`Histogram<C>`] — standard histogram with non-atomic counters of
//!   width `C` (defaults to `u64`). Use for single-threaded recording and
//!   percentile queries.
//! - [`AtomicHistogram<C>`] — atomic histogram for concurrent recording
//!   (defaults to `u64`). Take a snapshot with [`AtomicHistogram::load`]
//!   or [`AtomicHistogram::drain`] to query percentiles.
//! - [`SparseHistogram<C>`] — compact representation storing only
//!   non-zero buckets (defaults to `u64`). Useful for serialization and
//!   storage.
//! - [`CumulativeROHistogram<C>`] — read-only histogram with cumulative
//!   counts for fast quantile queries via binary search (defaults to
//!   `u64`).
```

- [ ] **Step 4: Add per-type rustdoc one-liners**

For each of the four histogram types, add a one-line note to the existing rustdoc just above the `pub struct` declaration. Specifically:

In `src/standard.rs`, find:
```rust
/// A histogram that uses plain 64bit counters for each bucket.
```
Replace with:
```rust
/// A histogram that uses plain counters for each bucket.
///
/// Generic over counter width `C`. Defaults to `u64`. See the crate-level
/// docs for guidance on choosing between `u32` and `u64`.
```

In `src/atomic.rs`, find:
```rust
/// A histogram that uses atomic 64bit counters for each bucket.
///
/// Unlike the non-atomic variant, it cannot be used directly to report
/// percentiles. Instead, a snapshot must be taken which captures the state of
/// the histogram at a point in time.
```
Replace with:
```rust
/// A histogram that uses atomic counters for each bucket.
///
/// Generic over counter width `C`. Defaults to `u64`. See the crate-level
/// docs for guidance on choosing between `u32` and `u64`.
///
/// Unlike the non-atomic variant, it cannot be used directly to report
/// percentiles. Instead, a snapshot must be taken which captures the state of
/// the histogram at a point in time.
```

In `src/sparse.rs`, find the existing `/// A sparse, columnar representation...` block and append after the existing description:
```rust
///
/// Generic over counter width `C`. Defaults to `u64`. See the crate-level
/// docs for guidance on choosing between `u32` and `u64`.
```

In `src/cumulative.rs`, find the existing `/// A read-only, cumulative histogram...` block and append after the last paragraph (before `#[derive...]`):
```rust
///
/// Generic over counter width `C`. Defaults to `u64`. See the crate-level
/// docs for guidance on choosing between `u32` and `u64`.
```

- [ ] **Step 5: Add memory-table footnote in `src/config.rs`**

In `src/config.rs`, immediately after the existing memory table (the `///` block ending in `|    12 | .025% | 160 KiB | 672 KiB | 1.7 MiB |`), add:

```rust
/// Halve all sizes for histograms instantiated with `<u32>` counters.
```

- [ ] **Step 6: Build documentation locally**

Run: `cargo doc --no-deps`
Expected: clean build, no broken intra-doc links.

Run: `cargo test --doc`
Expected: doctests pass (the new examples in lib.rs and README run via doctest).

- [ ] **Step 7: Run the full test suite once more**

Run: `cargo test`
Expected: all unit tests + doc tests pass.

- [ ] **Step 8: Bump version in `Cargo.toml`**

Find the current version in `Cargo.toml` (`1.2.0` per the spec capture). Change to:

```toml
version = "1.3.0-alpha.0"
```

If the alpha revision has already been bumped on `main` since the spec was written, set to `1.3.0-alpha.<next>` per the project versioning rule.

- [ ] **Step 9: Commit**

```bash
git add README.md src/lib.rs src/standard.rs src/atomic.rs src/sparse.rs src/cumulative.rs src/config.rs Cargo.toml
git commit -m "$(cat <<'EOF'
add docs and bump version for u32 bucket counts

- README: new "Counter Width" and "Recommended Pipeline" sections
- lib.rs: matching rustdoc with the snapshot pipeline guidance
- per-type rustdoc: note generic-over-C with u64 default
- config.rs: footnote on the memory table
- Cargo.toml: bump to 1.3.0-alpha.0 per project versioning rule

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Final Verification

- [ ] **Step 1: Full test suite**

Run: `cargo test`
Expected: every unit test, integration test, and doctest passes.

- [ ] **Step 2: Bench compile**

Run: `cargo bench --no-run`
Expected: bench binary builds cleanly.

- [ ] **Step 3: Lint check**

Run: `cargo clippy --all-targets -- -D warnings`
Expected: no warnings.

- [ ] **Step 4: Doc build**

Run: `cargo doc --no-deps`
Expected: clean build of public-facing docs.

- [ ] **Step 5: Confirm with the user before opening a PR.** Per the project's CLAUDE.md, do not push or open a PR without explicit permission.

---

## Self-Review Notes

Spec coverage check (run after writing the plan):

- ✅ Section 1 (Count trait, AtomicCount, sealed) → Task 1.
- ✅ Section 2 (Histogram<C>) → Task 2.
- ✅ Section 2 (AtomicHistogram<C>, drain cfg-split) → Task 3.
- ✅ Section 2 (SparseHistogram<C>) → Task 4.
- ✅ Section 2 (CumulativeROHistogram<C>) → Task 5.
- ✅ Section 2 (Bucket stays u64) → enforced in every generalization task via `.as_u128() as u64`.
- ✅ Section 3 (same-width cross-variant From) → already generalized in Tasks 4 and 5.
- ✅ Section 3 (cross-width widening From, same-variant) → Task 6.
- ✅ Section 3 (cross-width narrowing TryFrom, same-variant) → Task 7.
- ✅ Section 3 (cross-variant + narrowing TryFrom) → Task 8.
- ✅ Section 5 (docs + recommended pipeline) → Task 10.
- ✅ Testing (existing + u32 instantiations + width-specific tests) → distributed across Tasks 1–8.
- ✅ Bench updates → Task 9.
- ✅ Versioning → Task 10 step 8.
- ✅ No deprecation work, no Cargo.toml feature changes — both are explicitly out of scope.

Type-consistency check:

- `Count::ZERO`, `Count::ONE`, `Count::wrapping_add`, `Count::checked_add`, `Count::wrapping_sub`, `Count::checked_sub`, `Count::as_u128`, `Count::try_from_u64` referenced consistently across all tasks.
- `AtomicCount::new`, `load_relaxed`, `store_relaxed`, `fetch_add_relaxed`, `swap_relaxed` consistent.
- `pub(crate) fn add_at_index` introduced in Task 6 step 1 (added to `AtomicHistogram` for the widening path).
- `Bucket.count` is `u64` throughout (never `C`).
- `total_count` returns `u64` for both `Histogram` (via `QuantilesResult.total_count: u128` — unchanged) and `CumulativeROHistogram` (returns `u64`).
