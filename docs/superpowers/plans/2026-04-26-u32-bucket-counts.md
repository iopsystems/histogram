# u32 Bucket Counts Implementation Plan (v2 — named-sibling/macro approach)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `u32` as a sibling counter width across all four histogram variants by introducing named concrete sibling types (`Histogram32`, `AtomicHistogram32`, `SparseHistogram32`, `CumulativeROHistogram32`) generated from a shared declarative macro per variant. Existing types stay byte-identical to today — no source breakage for any user.

**Architecture:** A single `macro_rules!` macro per variant takes a type name, an iterator name, and a count primitive (and, for atomic, the atomic primitive plus the corresponding non-atomic histogram name). It emits the full struct definition, impl block, iterator types, and trait impls. The macro is invoked twice per variant — once for the existing type names and once for the new `*32` siblings. The internal vocabulary uses the sealed `Count` / `AtomicCount` traits already shipped in Task 1.

**Tech Stack:** Rust 2024 edition, `core::sync::atomic::{AtomicU32, AtomicU64}`, `criterion` for bench, `serde`/`schemars` (optional features), `cargo test` / `cargo bench`.

**Spec:** [docs/superpowers/specs/2026-04-26-u32-bucket-counts-design.md](../specs/2026-04-26-u32-bucket-counts-design.md) (v2, revised 2026-04-26)

**Prerequisite (already done):** Task 1 added `Count` and `AtomicCount` sealed traits in `src/count.rs` with re-exports from `src/lib.rs`. Commits `bba6635` (initial) and `26776af` (rustfmt).

**Cross-task dependency note:** Some `*32` types reference each other (e.g., `Histogram32` mentions `SparseHistogram32`). The plan handles this by commenting out the `*32` macro invocations in Tasks 2 and 3 and uncommenting them in Task 4 once all referenced types exist. The intermediate commits (Task 2 / Task 3) are intentionally u64-only.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/count.rs` | (already shipped) | `Count` and `AtomicCount` sealed traits + impls. |
| `src/lib.rs` | Modify | Re-export new `*32` types. Add module-level rustdoc. |
| `src/standard.rs` | Modify (whole-file macroification) | `define_histogram!` macro; invoke for `Histogram` and `Histogram32`. |
| `src/atomic.rs` | Modify (whole-file macroification) | `define_atomic_histogram!` macro; invoke for `AtomicHistogram` and `AtomicHistogram32`. Concrete-type `drain` blocks per cfg. |
| `src/sparse.rs` | Modify (whole-file macroification) | `define_sparse_histogram!` macro; invoke for both. |
| `src/cumulative.rs` | Modify (whole-file macroification) | `define_cumulative_histogram!` macro; invoke for both. |
| `src/conversions.rs` | **Create** | Cross-width and cross-variant + narrowing conversion impls. |
| `benches/histogram.rs` | Modify | Add `Histogram32` / `AtomicHistogram32` bench groups. |
| `README.md` | Modify | New "Counter Width" and "Recommended Pipeline" sections. |
| `Cargo.toml` | Modify | Bump version to next alpha (`1.3.0-alpha.<n>`). |

---

## Task 2: Macroify `Histogram` and prepare `Histogram32`

**Files:**
- Modify: `src/standard.rs` (whole-file refactor — wrap existing impl in macro)

This is the first variant macroification. Subsequent tasks (3, 4, 5) follow the same pattern.

**Important deprecation note:** the macroification drops the deprecated `percentile` / `percentiles` inherent methods (already marked `#[deprecated]` since v1.0). Users who relied on them migrate to `quantile`/`quantiles` (same shape, different name) or to the `SampleQuantiles` trait. Document in CHANGELOG via Task 10.

- [ ] **Step 1: Read the current file end-to-end**

Read `src/standard.rs` so you know exactly what's there. The existing file has a `Histogram` struct, an `impl Histogram` block, an `impl SampleQuantiles for Histogram` block, an `Iter` struct + `Iterator` / `ExactSizeIterator` / `FusedIterator` impls, an `IntoIterator for &Histogram` impl, an `impl From<&SparseHistogram> for Histogram` impl, and a `mod tests`.

- [ ] **Step 2: Replace the file contents with the macro**

Write `src/standard.rs`:

```rust
use std::collections::BTreeMap;

use crate::quantile::{Quantile, QuantilesResult, SampleQuantiles};
use crate::{Bucket, Config, Count, Error, SparseHistogram};
// SparseHistogram32 reference is added in Task 4; for now Histogram32 is commented out.

macro_rules! define_histogram {
    ($name:ident, $iter:ident, $sparse:ident, $count:ty) => {
        /// A histogram that uses plain counters for each bucket.
        #[derive(Clone, Debug, PartialEq, Eq)]
        #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
        #[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
        pub struct $name {
            pub(crate) config: Config,
            pub(crate) buckets: Box<[$count]>,
        }

        impl $name {
            pub fn new(grouping_power: u8, max_value_power: u8) -> Result<Self, Error> {
                let config = Config::new(grouping_power, max_value_power)?;
                Ok(Self::with_config(&config))
            }

            pub fn with_config(config: &Config) -> Self {
                let buckets: Box<[$count]> =
                    vec![<$count as Count>::ZERO; config.total_buckets()].into();
                Self { config: *config, buckets }
            }

            pub fn from_buckets(
                grouping_power: u8,
                max_value_power: u8,
                buckets: Vec<$count>,
            ) -> Result<Self, Error> {
                let config = Config::new(grouping_power, max_value_power)?;
                if config.total_buckets() != buckets.len() {
                    return Err(Error::IncompatibleParameters);
                }
                Ok(Self { config, buckets: buckets.into() })
            }

            pub fn increment(&mut self, value: u64) -> Result<(), Error> {
                self.add(value, <$count as Count>::ONE)
            }

            pub fn add(&mut self, value: u64, count: $count) -> Result<(), Error> {
                let index = self.config.value_to_index(value)?;
                self.buckets[index] = self.buckets[index].wrapping_add(count);
                Ok(())
            }

            pub fn as_slice(&self) -> &[$count] { &self.buckets }
            pub fn as_mut_slice(&mut self) -> &mut [$count] { &mut self.buckets }

            pub fn downsample(&self, grouping_power: u8) -> Result<Self, Error> {
                if grouping_power >= self.config.grouping_power() {
                    return Err(Error::IncompatibleParameters);
                }
                let mut histogram = Self::new(grouping_power, self.config.max_value_power())?;
                for (i, n) in self.as_slice().iter().enumerate() {
                    if *n != <$count as Count>::ZERO {
                        let val = self.config.index_to_lower_bound(i);
                        histogram.add(val, *n)?;
                    }
                }
                Ok(histogram)
            }

            pub fn checked_add(&self, other: &Self) -> Result<Self, Error> {
                if self.config != other.config {
                    return Err(Error::IncompatibleParameters);
                }
                let mut result = self.clone();
                for (this, other) in result.buckets.iter_mut().zip(other.buckets.iter()) {
                    *this = this.checked_add(*other).ok_or(Error::Overflow)?;
                }
                Ok(result)
            }

            pub fn wrapping_add(&self, other: &Self) -> Result<Self, Error> {
                if self.config != other.config {
                    return Err(Error::IncompatibleParameters);
                }
                let mut result = self.clone();
                for (this, other) in result.buckets.iter_mut().zip(other.buckets.iter()) {
                    *this = this.wrapping_add(*other);
                }
                Ok(result)
            }

            pub fn checked_sub(&self, other: &Self) -> Result<Self, Error> {
                if self.config != other.config {
                    return Err(Error::IncompatibleParameters);
                }
                let mut result = self.clone();
                for (this, other) in result.buckets.iter_mut().zip(other.buckets.iter()) {
                    *this = this.checked_sub(*other).ok_or(Error::Underflow)?;
                }
                Ok(result)
            }

            pub fn wrapping_sub(&self, other: &Self) -> Result<Self, Error> {
                if self.config != other.config {
                    return Err(Error::IncompatibleParameters);
                }
                let mut result = self.clone();
                for (this, other) in result.buckets.iter_mut().zip(other.buckets.iter()) {
                    *this = this.wrapping_sub(*other);
                }
                Ok(result)
            }

            pub fn iter(&self) -> $iter<'_> {
                $iter { index: 0, histogram: self }
            }

            pub fn config(&self) -> Config { self.config }

            // Inherent quantile / quantiles forwarders. Inherent dispatch
            // works without trait-resolution ambiguity because the type
            // is concrete (no generic parameter).
            pub fn quantiles(&self, quantiles: &[f64]) -> Result<Option<QuantilesResult>, Error> {
                <Self as SampleQuantiles>::quantiles(self, quantiles)
            }

            pub fn quantile(&self, quantile: f64) -> Result<Option<QuantilesResult>, Error> {
                <Self as SampleQuantiles>::quantile(self, quantile)
            }
        }

        impl SampleQuantiles for $name {
            fn quantiles(&self, quantiles: &[f64]) -> Result<Option<QuantilesResult>, Error> {
                for q in quantiles {
                    if !(0.0..=1.0).contains(q) {
                        return Err(Error::InvalidQuantile);
                    }
                }
                let total_count: u128 = self.buckets.iter().map(|v| v.as_u128()).sum();
                if total_count == 0 { return Ok(None); }

                let mut sorted: Vec<Quantile> = quantiles
                    .iter().map(|&q| Quantile::new(q).unwrap()).collect();
                sorted.sort();
                sorted.dedup();

                let mut min_idx = None;
                let mut max_idx = None;
                for (i, count) in self.buckets.iter().enumerate() {
                    if *count != <$count as Count>::ZERO {
                        if min_idx.is_none() { min_idx = Some(i); }
                        max_idx = Some(i);
                    }
                }
                let min_idx = min_idx.unwrap();
                let max_idx = max_idx.unwrap();

                let min = Bucket {
                    count: self.buckets[min_idx].as_u128() as u64,
                    range: self.config.index_to_range(min_idx),
                };
                let max = Bucket {
                    count: self.buckets[max_idx].as_u128() as u64,
                    range: self.config.index_to_range(max_idx),
                };

                let mut bucket_idx = 0;
                let mut partial_sum = self.buckets[bucket_idx].as_u128();
                let mut entries = BTreeMap::new();

                for quantile in &sorted {
                    let count = std::cmp::max(
                        1, (quantile.as_f64() * total_count as f64).ceil() as u128,
                    );
                    loop {
                        if partial_sum >= count {
                            entries.insert(*quantile, Bucket {
                                count: self.buckets[bucket_idx].as_u128() as u64,
                                range: self.config.index_to_range(bucket_idx),
                            });
                            break;
                        }
                        if bucket_idx == (self.buckets.len() - 1) { break; }
                        bucket_idx += 1;
                        partial_sum += self.buckets[bucket_idx].as_u128();
                    }
                }

                Ok(Some(QuantilesResult::new(entries, total_count, min, max)))
            }
        }

        impl<'a> IntoIterator for &'a $name {
            type Item = Bucket;
            type IntoIter = $iter<'a>;
            fn into_iter(self) -> Self::IntoIter {
                $iter { index: 0, histogram: self }
            }
        }

        pub struct $iter<'a> {
            index: usize,
            histogram: &'a $name,
        }
        impl Iterator for $iter<'_> {
            type Item = Bucket;
            fn next(&mut self) -> Option<Bucket> {
                if self.index >= self.histogram.buckets.len() { return None; }
                let bucket = Bucket {
                    count: self.histogram.buckets[self.index].as_u128() as u64,
                    range: self.histogram.config.index_to_range(self.index),
                };
                self.index += 1;
                Some(bucket)
            }
        }
        impl ExactSizeIterator for $iter<'_> {
            fn len(&self) -> usize {
                self.histogram.buckets.len() - self.index
            }
        }
        impl std::iter::FusedIterator for $iter<'_> {}

        impl From<&$sparse> for $name {
            fn from(other: &$sparse) -> Self {
                let mut histogram = $name::with_config(&other.config);
                for (index, count) in other.index.iter().zip(other.count.iter()) {
                    histogram.buckets[*index as usize] = *count;
                }
                histogram
            }
        }
    };
}

define_histogram!(Histogram, Iter, SparseHistogram, u64);
// define_histogram!(Histogram32, Iter32, SparseHistogram32, u32);  // uncommented in Task 4

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngExt;

    // ===== Existing u64 tests preserved =====

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn size() {
        assert_eq!(std::mem::size_of::<Histogram>(), 48);
    }

    #[test]
    fn quantiles() {
        let mut histogram = Histogram::new(7, 64).unwrap();
        assert_eq!(histogram.quantile(0.5).unwrap(), None);
        for i in 0..=100 {
            let _ = histogram.increment(i);
        }
        assert_eq!(histogram.quantile(0.5).unwrap().unwrap().get(&Quantile::new(0.5).unwrap()).unwrap().end(), 50);
        assert_eq!(histogram.quantile(0.99).unwrap().unwrap().get(&Quantile::new(0.99).unwrap()).unwrap().end(), 99);
        assert_eq!(histogram.quantile(-1.0), Err(Error::InvalidQuantile));
        assert_eq!(histogram.quantile(1.01), Err(Error::InvalidQuantile));
    }

    #[test]
    fn min() {
        let mut histogram = Histogram::new(7, 64).unwrap();
        assert_eq!(histogram.quantile(0.0).unwrap(), None);
        let _ = histogram.increment(10);
        assert_eq!(histogram.quantile(0.0).unwrap().unwrap().get(&Quantile::new(0.0).unwrap()).unwrap().end(), 10);
        let _ = histogram.increment(4);
        assert_eq!(histogram.quantile(0.0).unwrap().unwrap().get(&Quantile::new(0.0).unwrap()).unwrap().end(), 4);
    }

    #[test]
    fn downsample() {
        let mut histogram = Histogram::new(8, 32).unwrap();
        let mut vals: Vec<u64> = Vec::with_capacity(10000);
        use rand::SeedableRng;
        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
        for _ in 0..vals.capacity() {
            let v: u64 = rng.random_range(1..2_u64.pow(histogram.config.max_value_power() as u32));
            vals.push(v);
            let _ = histogram.increment(v);
        }
        vals.sort();
        let h = histogram.clone();
        let grouping_power = histogram.config.grouping_power();
        for factor in 1..grouping_power {
            let error = histogram.config.error();
            for p in &[0.5, 0.9, 0.99, 0.999, 1.0] {
                let v = vals[((*p * (vals.len() as f64)) as usize) - 1];
                let q = histogram.quantile(*p).unwrap().unwrap();
                let vhist = q.get(&Quantile::new(*p).unwrap()).unwrap().end();
                let e = (v.abs_diff(vhist) as f64) * 100.0 / (v as f64);
                assert!(e < error);
            }
            histogram = h.downsample(grouping_power - factor).unwrap();
        }
    }

    fn build_histograms() -> (Histogram, Histogram, Histogram, Histogram) {
        let mut h1 = Histogram::new(1, 3).unwrap();
        let mut h2 = Histogram::new(1, 3).unwrap();
        let mut h3 = Histogram::new(1, 3).unwrap();
        let h4 = Histogram::new(7, 32).unwrap();
        for i in 0..h1.config().total_buckets() {
            h1.as_mut_slice()[i] = 1;
            h2.as_mut_slice()[i] = 1;
            h3.as_mut_slice()[i] = u64::MAX;
        }
        (h1, h2, h3, h4)
    }

    #[test]
    fn checked_add() {
        let (h, h_good, h_overflow, h_mismatch) = build_histograms();
        assert_eq!(h.checked_add(&h_mismatch), Err(Error::IncompatibleParameters));
        let r = h.checked_add(&h_good).unwrap();
        assert_eq!(r.as_slice(), &[2, 2, 2, 2, 2, 2]);
        assert_eq!(h.checked_add(&h_overflow), Err(Error::Overflow));
    }

    #[test]
    fn wrapping_add() {
        let (h, h_good, h_overflow, h_mismatch) = build_histograms();
        assert_eq!(h.wrapping_add(&h_mismatch), Err(Error::IncompatibleParameters));
        let r = h.wrapping_add(&h_good).unwrap();
        assert_eq!(r.as_slice(), &[2, 2, 2, 2, 2, 2]);
        let r = h.wrapping_add(&h_overflow).unwrap();
        assert_eq!(r.as_slice(), &[0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn checked_sub() {
        let (h, h_good, h_overflow, h_mismatch) = build_histograms();
        assert_eq!(h.checked_sub(&h_mismatch), Err(Error::IncompatibleParameters));
        let r = h.checked_sub(&h_good).unwrap();
        assert_eq!(r.as_slice(), &[0, 0, 0, 0, 0, 0]);
        assert_eq!(h.checked_sub(&h_overflow), Err(Error::Underflow));
    }

    #[test]
    fn wrapping_sub() {
        let (h, h_good, h_overflow, h_mismatch) = build_histograms();
        assert_eq!(h.wrapping_sub(&h_mismatch), Err(Error::IncompatibleParameters));
        let r = h.wrapping_sub(&h_good).unwrap();
        assert_eq!(r.as_slice(), &[0, 0, 0, 0, 0, 0]);
        let r = h.wrapping_sub(&h_overflow).unwrap();
        assert_eq!(r.as_slice(), &[2, 2, 2, 2, 2, 2]);
    }

    #[test]
    fn from_buckets() {
        let mut histogram = Histogram::new(8, 32).unwrap();
        for i in 0..=100 {
            let _ = histogram.increment(i);
        }
        let buckets = histogram.as_slice();
        let constructed = Histogram::from_buckets(8, 32, buckets.to_vec()).unwrap();
        assert!(constructed == histogram);
    }
}
```

Note: the existing `mod tests` block in the original file referenced the deprecated `percentile`/`percentiles` methods. The replacement above maps those tests to use `quantile`/`quantiles` (same semantic content, the new name). The `from_buckets`, arithmetic (`checked_add`/`wrapping_add`/`checked_sub`/`wrapping_sub`), `downsample`, `min`, and `quantiles` tests are preserved.

- [ ] **Step 3: Update `src/quantile.rs` and `src/lib.rs` to use new method names**

The existing tests in `src/quantile.rs` and the doctest in `src/lib.rs` already use `quantile` / `quantiles` (introduced in v1.2). They should compile against the macro-emitted concrete `Histogram` without any turbofish — verify with `cargo test --doc` and `cargo test --lib quantile::`.

If anything fails, the failing call site will need to be updated to use the new method name. There should be nothing to change in practice because the existing tests already use trait names matching the macro emission.

- [ ] **Step 4: Build + run existing tests**

Run: `cargo build`
Expected: clean.

Run: `cargo test --lib`
Expected: existing tests still pass (`Histogram` is byte-identical in shape to before the macroification).

- [ ] **Step 5: cargo fmt**

Run: `cargo fmt && cargo fmt --check`

- [ ] **Step 6: Commit (intermediate — Histogram only)**

```bash
git -C /Users/yao/workspace/histogram/.worktrees/u32-bucket-counts add src/standard.rs
git -C /Users/yao/workspace/histogram/.worktrees/u32-bucket-counts commit -m "macroify Histogram

Wrap Histogram, Iter, SampleQuantiles impl, IntoIterator impl, and
From<&SparseHistogram> impl in define_histogram! macro. Invoke for
Histogram (u64) only; Histogram32 invocation is commented out and
follows in Task 4 once SparseHistogram32 exists.

Drops deprecated percentile/percentiles inherent methods; users
migrate to quantile/quantiles or to the SampleQuantiles trait."
```

(No Co-Authored-By trailer.)

## Task 3: Macroify `AtomicHistogram` and prepare `AtomicHistogram32`

**Files:**
- Modify: `src/atomic.rs`

- [ ] **Step 1: Read the current file**

- [ ] **Step 2: Replace contents with the macro**

```rust
use crate::config::Config;
use crate::{AtomicCount, Count, Error, Histogram};
// Histogram32 reference is used after Task 4 uncomments the second invocation.
use core::sync::atomic::{AtomicU32, AtomicU64};

macro_rules! define_atomic_histogram {
    ($name:ident, $count:ty, $atomic:ty, $hist:ident) => {
        /// A histogram that uses atomic counters for each bucket.
        ///
        /// Unlike the non-atomic variant, it cannot be used directly to report
        /// percentiles. Instead, a snapshot must be taken which captures the
        /// state of the histogram at a point in time.
        pub struct $name {
            pub(crate) config: Config,
            pub(crate) buckets: Box<[$atomic]>,
        }

        impl $name {
            pub fn new(grouping_power: u8, max_value_power: u8) -> Result<Self, Error> {
                let config = Config::new(grouping_power, max_value_power)?;
                Ok(Self::with_config(&config))
            }

            pub fn with_config(config: &Config) -> Self {
                let mut buckets = Vec::with_capacity(config.total_buckets());
                buckets.resize_with(config.total_buckets(), || {
                    <$atomic as AtomicCount>::new(<$count as Count>::ZERO)
                });
                Self { config: *config, buckets: buckets.into() }
            }

            pub fn increment(&self, value: u64) -> Result<(), Error> {
                self.add(value, <$count as Count>::ONE)
            }

            pub fn add(&self, value: u64, count: $count) -> Result<(), Error> {
                let index = self.config.value_to_index(value)?;
                self.buckets[index].fetch_add_relaxed(count);
                Ok(())
            }

            pub fn config(&self) -> Config { self.config }

            pub fn load(&self) -> $hist {
                let buckets: Vec<$count> =
                    self.buckets.iter().map(|b| b.load_relaxed()).collect();
                $hist { config: self.config, buckets: buckets.into() }
            }
        }

        impl std::fmt::Debug for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct(stringify!($name))
                    .field("config", &self.config)
                    .finish()
            }
        }
    };
}

define_atomic_histogram!(AtomicHistogram, u64, AtomicU64, Histogram);
// define_atomic_histogram!(AtomicHistogram32, u32, AtomicU32, Histogram32);  // uncommented in Task 4

#[cfg(target_has_atomic = "64")]
impl AtomicHistogram {
    /// Drains the bucket values into a new `Histogram`.
    pub fn drain(&self) -> Histogram {
        let buckets: Vec<u64> = self.buckets.iter().map(|b| b.swap_relaxed(0)).collect();
        Histogram { config: self.config, buckets: buckets.into() }
    }
}

// AtomicHistogram32 drain block — uncomment in Task 4.
// #[cfg(target_has_atomic = "32")]
// impl AtomicHistogram32 {
//     pub fn drain(&self) -> Histogram32 {
//         let buckets: Vec<u32> = self.buckets.iter().map(|b| b.swap_relaxed(0)).collect();
//         Histogram32 { config: self.config, buckets: buckets.into() }
//     }
// }

#[cfg(test)]
mod tests {
    use crate::*;

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn size() {
        assert_eq!(std::mem::size_of::<AtomicHistogram>(), 48);
    }

    #[cfg(target_has_atomic = "64")]
    #[test]
    fn drain() {
        let histogram = AtomicHistogram::new(7, 64).unwrap();
        for i in 0..=100 {
            let _ = histogram.increment(i);
        }
        let snapshot = histogram.drain();
        let result = snapshot.quantile(0.50).unwrap().unwrap();
        assert_eq!(
            result.get(&Quantile::new(0.50).unwrap()).unwrap().end(),
            50,
        );
        histogram.increment(1000).unwrap();
        let snapshot = histogram.drain();
        let result = snapshot.quantile(0.50).unwrap().unwrap();
        assert_eq!(
            result.get(&Quantile::new(0.50).unwrap()).unwrap().end(),
            1003,
        );
    }

    #[test]
    fn quantiles() {
        let histogram = AtomicHistogram::new(7, 64).unwrap();
        let qs = [0.25, 0.50, 0.75, 0.90, 0.99];
        assert_eq!(histogram.load().quantiles(&qs).unwrap(), None);
        for i in 0..=100 {
            let _ = histogram.increment(i);
            let result = histogram.load().quantile(0.0).unwrap().unwrap();
            assert_eq!(result.get(&Quantile::new(0.0).unwrap()).unwrap().end(), 0);
            let result = histogram.load().quantile(1.0).unwrap().unwrap();
            assert_eq!(result.get(&Quantile::new(1.0).unwrap()).unwrap().end(), i);
        }
        for q in qs {
            let result = histogram.load().quantile(q).unwrap().unwrap();
            let bucket = result.get(&Quantile::new(q).unwrap()).unwrap();
            assert_eq!(bucket.end(), (q * 100.0) as u64);
        }
    }
}
```

- [ ] **Step 3: Build + run existing tests**

Run: `cargo build && cargo test --lib atomic::`
Expected: existing atomic tests still pass.

- [ ] **Step 4: cargo fmt**

- [ ] **Step 5: Commit**

```bash
git -C /Users/yao/workspace/histogram/.worktrees/u32-bucket-counts add src/atomic.rs
git -C /Users/yao/workspace/histogram/.worktrees/u32-bucket-counts commit -m "macroify AtomicHistogram

Wrap AtomicHistogram in define_atomic_histogram! macro. Invoke for
AtomicHistogram (u64) only; AtomicHistogram32 invocation and drain
impl follow in Task 4 once Histogram32 exists in standard.rs."
```

## Task 4: Macroify `SparseHistogram` and finalize `Histogram32` / `AtomicHistogram32`

**Files:**
- Modify: `src/sparse.rs`
- Modify: `src/standard.rs` (uncomment second macro invocation; add SparseHistogram32 import)
- Modify: `src/atomic.rs` (uncomment second macro invocation + AtomicHistogram32 drain impl)

This task closes the `Histogram` ↔ `SparseHistogram` cycle for both u32 and u64 families.

- [ ] **Step 1: Read `src/sparse.rs` end-to-end**

- [ ] **Step 2: Replace `src/sparse.rs` with the macro**

```rust
use std::collections::BTreeMap;
use crate::quantile::{Quantile, QuantilesResult, SampleQuantiles};
use crate::{Bucket, Config, Count, Error, Histogram, Histogram32};

macro_rules! define_sparse_histogram {
    ($name:ident, $iter:ident, $hist:ident, $count:ty) => {
        /// A sparse, columnar representation of a histogram.
        #[derive(Clone, Debug, PartialEq, Eq)]
        #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
        #[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
        pub struct $name {
            pub(crate) config: Config,
            pub(crate) index: Vec<u32>,
            pub(crate) count: Vec<$count>,
        }

        impl $name {
            pub fn new(grouping_power: u8, max_value_power: u8) -> Result<Self, Error> {
                let config = Config::new(grouping_power, max_value_power)?;
                Ok(Self::with_config(&config))
            }

            pub fn with_config(config: &Config) -> Self {
                Self { config: *config, index: Vec::new(), count: Vec::new() }
            }

            pub fn from_parts(
                config: Config,
                index: Vec<u32>,
                count: Vec<$count>,
            ) -> Result<Self, Error> {
                if index.len() != count.len() {
                    return Err(Error::IncompatibleParameters);
                }
                let total_buckets = config.total_buckets();
                let mut prev = None;
                for &idx in &index {
                    if idx as usize >= total_buckets { return Err(Error::OutOfRange); }
                    if let Some(p) = prev { if idx <= p { return Err(Error::IncompatibleParameters); } }
                    prev = Some(idx);
                }
                for &c in &count {
                    if c == <$count as Count>::ZERO {
                        return Err(Error::IncompatibleParameters);
                    }
                }
                Ok(Self { config, index, count })
            }

            pub fn into_parts(self) -> (Config, Vec<u32>, Vec<$count>) {
                (self.config, self.index, self.count)
            }

            pub fn config(&self) -> Config { self.config }
            pub fn index(&self) -> &[u32] { &self.index }
            pub fn count(&self) -> &[$count] { &self.count }

            fn add_bucket(&mut self, idx: u32, n: $count) {
                if n != <$count as Count>::ZERO {
                    self.index.push(idx);
                    self.count.push(n);
                }
            }

            #[allow(clippy::comparison_chain)]
            pub fn checked_add(&self, h: &Self) -> Result<Self, Error> {
                if self.config != h.config { return Err(Error::IncompatibleParameters); }
                let mut histogram = Self::with_config(&self.config);
                let (mut i, mut j) = (0, 0);
                while i < self.index.len() && j < h.index.len() {
                    let (k1, v1) = (self.index[i], self.count[i]);
                    let (k2, v2) = (h.index[j], h.count[j]);
                    if k1 == k2 {
                        let v = v1.checked_add(v2).ok_or(Error::Overflow)?;
                        histogram.add_bucket(k1, v);
                        (i, j) = (i + 1, j + 1);
                    } else if k1 < k2 {
                        histogram.add_bucket(k1, v1); i += 1;
                    } else {
                        histogram.add_bucket(k2, v2); j += 1;
                    }
                }
                if i < self.index.len() {
                    histogram.index.extend(&self.index[i..]);
                    histogram.count.extend(&self.count[i..]);
                }
                if j < h.index.len() {
                    histogram.index.extend(&h.index[j..]);
                    histogram.count.extend(&h.count[j..]);
                }
                Ok(histogram)
            }

            #[allow(clippy::comparison_chain)]
            pub fn wrapping_add(&self, h: &Self) -> Result<Self, Error> {
                if self.config != h.config { return Err(Error::IncompatibleParameters); }
                let mut histogram = Self::with_config(&self.config);
                let (mut i, mut j) = (0, 0);
                while i < self.index.len() && j < h.index.len() {
                    let (k1, v1) = (self.index[i], self.count[i]);
                    let (k2, v2) = (h.index[j], h.count[j]);
                    if k1 == k2 {
                        histogram.add_bucket(k1, v1.wrapping_add(v2));
                        (i, j) = (i + 1, j + 1);
                    } else if k1 < k2 {
                        histogram.add_bucket(k1, v1); i += 1;
                    } else {
                        histogram.add_bucket(k2, v2); j += 1;
                    }
                }
                if i < self.index.len() {
                    histogram.index.extend(&self.index[i..]);
                    histogram.count.extend(&self.count[i..]);
                }
                if j < h.index.len() {
                    histogram.index.extend(&h.index[j..]);
                    histogram.count.extend(&h.count[j..]);
                }
                Ok(histogram)
            }

            #[allow(clippy::comparison_chain)]
            pub fn checked_sub(&self, h: &Self) -> Result<Self, Error> {
                if self.config != h.config { return Err(Error::IncompatibleParameters); }
                let mut histogram = Self::with_config(&self.config);
                let (mut i, mut j) = (0, 0);
                while i < self.index.len() && j < h.index.len() {
                    let (k1, v1) = (self.index[i], self.count[i]);
                    let (k2, v2) = (h.index[j], h.count[j]);
                    if k1 == k2 {
                        let v = v1.checked_sub(v2).ok_or(Error::Underflow)?;
                        if v != <$count as Count>::ZERO { histogram.add_bucket(k1, v); }
                        (i, j) = (i + 1, j + 1);
                    } else if k1 < k2 {
                        histogram.add_bucket(k1, v1); i += 1;
                    } else {
                        return Err(Error::InvalidSubset);
                    }
                }
                if j < h.index.len() { return Err(Error::InvalidSubset); }
                if i < self.index.len() {
                    histogram.index.extend(&self.index[i..]);
                    histogram.count.extend(&self.count[i..]);
                }
                Ok(histogram)
            }

            #[allow(clippy::comparison_chain)]
            pub fn wrapping_sub(&self, h: &Self) -> Result<Self, Error> {
                if self.config != h.config { return Err(Error::IncompatibleParameters); }
                let mut histogram = Self::with_config(&self.config);
                let (mut i, mut j) = (0, 0);
                while i < self.index.len() && j < h.index.len() {
                    let (k1, v1) = (self.index[i], self.count[i]);
                    let (k2, v2) = (h.index[j], h.count[j]);
                    if k1 == k2 {
                        histogram.add_bucket(k1, v1.wrapping_sub(v2));
                        (i, j) = (i + 1, j + 1);
                    } else if k1 < k2 {
                        histogram.add_bucket(k1, v1); i += 1;
                    } else {
                        return Err(Error::InvalidSubset);
                    }
                }
                if i < self.index.len() {
                    histogram.index.extend(&self.index[i..]);
                    histogram.count.extend(&self.count[i..]);
                }
                if j < h.index.len() { return Err(Error::InvalidSubset); }
                Ok(histogram)
            }

            pub fn downsample(&self, grouping_power: u8) -> Result<Self, Error> {
                if grouping_power >= self.config.grouping_power() {
                    return Err(Error::IncompatibleParameters);
                }
                let config = Config::new(grouping_power, self.config.max_value_power())?;
                let mut histogram = Self::with_config(&config);
                let mut aggregating_idx: u32 = 0;
                let mut aggregating_count: $count = <$count as Count>::ZERO;
                for (idx, n) in self.index.iter().zip(self.count.iter()) {
                    let new_idx = config
                        .value_to_index(self.config.index_to_lower_bound(*idx as usize))?
                        as u32;
                    if new_idx == aggregating_idx {
                        aggregating_count = aggregating_count.wrapping_add(*n);
                        continue;
                    }
                    histogram.add_bucket(aggregating_idx, aggregating_count);
                    aggregating_idx = new_idx;
                    aggregating_count = *n;
                }
                histogram.add_bucket(aggregating_idx, aggregating_count);
                Ok(histogram)
            }

            pub fn iter(&self) -> $iter<'_> {
                $iter { index: 0, histogram: self }
            }

            pub fn quantiles(&self, quantiles: &[f64]) -> Result<Option<QuantilesResult>, Error> {
                <Self as SampleQuantiles>::quantiles(self, quantiles)
            }
            pub fn quantile(&self, quantile: f64) -> Result<Option<QuantilesResult>, Error> {
                <Self as SampleQuantiles>::quantile(self, quantile)
            }
        }

        impl SampleQuantiles for $name {
            fn quantiles(&self, quantiles: &[f64]) -> Result<Option<QuantilesResult>, Error> {
                for q in quantiles {
                    if !(0.0..=1.0).contains(q) {
                        return Err(Error::InvalidQuantile);
                    }
                }
                let total_count: u128 = self.count.iter().map(|v| v.as_u128()).sum();
                if total_count == 0 { return Ok(None); }

                let mut sorted: Vec<Quantile> = quantiles
                    .iter().map(|&q| Quantile::new(q).unwrap()).collect();
                sorted.sort();
                sorted.dedup();

                let min = Bucket {
                    count: self.count[0].as_u128() as u64,
                    range: self.config.index_to_range(self.index[0] as usize),
                };
                let last = self.index.len() - 1;
                let max = Bucket {
                    count: self.count[last].as_u128() as u64,
                    range: self.config.index_to_range(self.index[last] as usize),
                };

                let mut idx = 0;
                let mut partial_sum = self.count[0].as_u128();
                let mut entries = BTreeMap::new();

                for quantile in &sorted {
                    let count = std::cmp::max(
                        1, (quantile.as_f64() * total_count as f64).ceil() as u128,
                    );
                    loop {
                        if partial_sum >= count {
                            entries.insert(*quantile, Bucket {
                                count: self.count[idx].as_u128() as u64,
                                range: self.config.index_to_range(self.index[idx] as usize),
                            });
                            break;
                        }
                        if idx == (self.index.len() - 1) { break; }
                        idx += 1;
                        partial_sum += self.count[idx].as_u128();
                    }
                }
                Ok(Some(QuantilesResult::new(entries, total_count, min, max)))
            }
        }

        impl<'a> IntoIterator for &'a $name {
            type Item = Bucket;
            type IntoIter = $iter<'a>;
            fn into_iter(self) -> Self::IntoIter {
                $iter { index: 0, histogram: self }
            }
        }

        pub struct $iter<'a> {
            index: usize,
            histogram: &'a $name,
        }
        impl Iterator for $iter<'_> {
            type Item = Bucket;
            fn next(&mut self) -> Option<Bucket> {
                if self.index >= self.histogram.index.len() { return None; }
                let bucket = Bucket {
                    count: self.histogram.count[self.index].as_u128() as u64,
                    range: self.histogram
                        .config
                        .index_to_range(self.histogram.index[self.index] as usize),
                };
                self.index += 1;
                Some(bucket)
            }
        }
        impl ExactSizeIterator for $iter<'_> {
            fn len(&self) -> usize {
                self.histogram.index.len() - self.index
            }
        }
        impl std::iter::FusedIterator for $iter<'_> {}

        impl From<&$hist> for $name {
            fn from(histogram: &$hist) -> Self {
                let mut index = Vec::new();
                let mut count = Vec::new();
                for (idx, n) in histogram.as_slice().iter().enumerate() {
                    if *n != <$count as Count>::ZERO {
                        index.push(idx as u32);
                        count.push(*n);
                    }
                }
                Self { config: histogram.config(), index, count }
            }
        }
    };
}

define_sparse_histogram!(SparseHistogram, SparseIter, Histogram, u64);
define_sparse_histogram!(SparseHistogram32, SparseIter32, Histogram32, u32);

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use rand::RngExt;
    use crate::standard::Histogram;

    // ===== Existing tests preserved (u64) =====
    // [keep the existing checked_add, wrapping_add, checked_sub, wrapping_sub,
    //  wrapping_add_overflow, percentiles (renamed quantiles), min,
    //  compare_histograms, snapshot, downsample tests verbatim — they all
    //  reference SparseHistogram (u64) which still exists]

    // Add new u32-targeted tests at the end:

    #[test]
    fn from_parts_u32() {
        let config = Config::new(7, 32).unwrap();
        let h = SparseHistogram32::from_parts(config, vec![1, 3, 5], vec![6u32, 12, 7]).unwrap();
        assert_eq!(h.index(), &[1, 3, 5]);
        assert_eq!(h.count(), &[6u32, 12, 7]);
    }

    #[test]
    fn checked_add_u32_overflow() {
        let config = Config::new(7, 32).unwrap();
        let h1 = SparseHistogram32::from_parts(config, vec![1], vec![u32::MAX]).unwrap();
        let h2 = SparseHistogram32::from_parts(config, vec![1], vec![1u32]).unwrap();
        assert_eq!(h1.checked_add(&h2), Err(Error::Overflow));
    }

    #[test]
    fn from_histogram_u32() {
        let mut h = Histogram32::new(7, 64).unwrap();
        h.increment(1).unwrap();
        h.increment(5).unwrap();
        h.increment(100).unwrap();
        let s = SparseHistogram32::from(&h);
        assert_eq!(s.count().len(), 3);
    }
}
```

(For the existing u64 tests, copy the bodies verbatim from the original `src/sparse.rs` test module.)

- [ ] **Step 3: In `src/standard.rs`, uncomment the `Histogram32` line**

Find `// define_histogram!(Histogram32, Iter32, SparseHistogram32, u32);  // uncommented in Task 4` and uncomment.

Also update the import line at the top to include `SparseHistogram32`:

```rust
use crate::{Bucket, Config, Count, Error, SparseHistogram, SparseHistogram32};
```

- [ ] **Step 4: In `src/atomic.rs`, uncomment the `AtomicHistogram32` line and the impl block**

Find `// define_atomic_histogram!(AtomicHistogram32, ...)` and the commented-out `impl AtomicHistogram32 { ... }`; uncomment both. Update the import:

```rust
use crate::{AtomicCount, Count, Error, Histogram, Histogram32};
```

- [ ] **Step 5: Build everything**

Run: `cargo build`
Expected: clean. Now `Histogram32`, `AtomicHistogram32`, and `SparseHistogram32` all exist.

Run: `cargo test --lib`
Expected: all existing tests pass.

- [ ] **Step 6: Add u32-targeted tests in `src/standard.rs` and `src/atomic.rs`**

In `src/standard.rs` `mod tests`, append:

```rust
#[cfg(target_pointer_width = "64")]
#[test]
fn size_u32() { assert_eq!(std::mem::size_of::<Histogram32>(), 48); }

#[test]
fn increment_u32() {
    let mut h = Histogram32::new(7, 64).unwrap();
    h.increment(5).unwrap();
    h.increment(5).unwrap();
    h.increment(5).unwrap();
    let result = h.quantile(0.5).unwrap().unwrap();
    let q = Quantile::new(0.5).unwrap();
    assert_eq!(result.get(&q).unwrap().count(), 3u64);
}

#[test]
fn add_u32_wraps_at_max() {
    let mut h = Histogram32::new(2, 4).unwrap();
    h.add(1, u32::MAX).unwrap();
    h.add(1, 1).unwrap();
    assert_eq!(h.as_slice()[1], 0u32);
}

#[test]
fn checked_add_u32_overflow() {
    let mut h1 = Histogram32::new(1, 3).unwrap();
    let mut h2 = Histogram32::new(1, 3).unwrap();
    h1.as_mut_slice()[0] = u32::MAX;
    h2.as_mut_slice()[0] = 1;
    assert_eq!(h1.checked_add(&h2), Err(Error::Overflow));
}

#[test]
fn iter_u32_widens_count_to_u64() {
    let mut h = Histogram32::new(2, 4).unwrap();
    h.add(1, 5u32).unwrap();
    let bucket = h.iter().find(|b| b.count() > 0).unwrap();
    let count: u64 = bucket.count();
    assert_eq!(count, 5);
}
```

In `src/atomic.rs` `mod tests`, append:

```rust
#[cfg(target_pointer_width = "64")]
#[test]
fn size_u32() { assert_eq!(std::mem::size_of::<AtomicHistogram32>(), 48); }

#[cfg(target_has_atomic = "32")]
#[test]
fn drain_u32() {
    let h = AtomicHistogram32::new(7, 64).unwrap();
    for v in 0..=100u64 { h.increment(v).unwrap(); }
    let snap = h.drain();
    let result = snap.quantile(0.5).unwrap().unwrap();
    let q = Quantile::new(0.5).unwrap();
    assert_eq!(result.get(&q).unwrap().end(), 50);
}
```

- [ ] **Step 7: cargo fmt + run all tests**

Run: `cargo fmt && cargo test`
Expected: all unit + doc tests pass, clean fmt.

- [ ] **Step 8: Single commit closes the cycle**

```bash
git -C /Users/yao/workspace/histogram/.worktrees/u32-bucket-counts add src/sparse.rs src/standard.rs src/atomic.rs
git -C /Users/yao/workspace/histogram/.worktrees/u32-bucket-counts commit -m "macroify SparseHistogram and finalize Histogram32 / AtomicHistogram32

Wrap SparseHistogram in define_sparse_histogram! macro and invoke
for SparseHistogram (u64) and SparseHistogram32 (u32). Uncomments
the previously-deferred Histogram32 and AtomicHistogram32 macro
invocations now that SparseHistogram32 exists.

Includes u32-specific tests for Histogram32, AtomicHistogram32,
and SparseHistogram32."
```

## Task 5: Macroify `CumulativeROHistogram` and add `CumulativeROHistogram32`

**Files:**
- Modify: `src/cumulative.rs`

- [ ] **Step 1: Read the current file**

- [ ] **Step 2: Wrap in `define_cumulative_histogram!` and invoke twice**

Pattern follows the same shape as Task 4. Key differences from sparse:

- Counts are cumulative (non-decreasing). `from_parts` validation: counts must be non-decreasing and non-zero.
- `total_count() -> u64` widens via `c.as_u128() as u64`.
- `bucket_quantile_range` and `iter_with_quantiles`: replace `as f64` with `.as_u128() as f64`.
- `find_quantile_position`: cache-line threshold becomes `count_ty`-dependent — declare a const inside the macro body: `const CACHE_LINE_ENTRIES: usize = 64 / std::mem::size_of::<$count>();`.
- `individual_count` private fn returns `u64`: uses `c.wrapping_sub(prev).as_u128() as u64`.
- `From<&$hist>`: cumulative running sum starting from `<$count as Count>::ZERO`; `if n != ZERO` check; `running_sum.wrapping_add(*n)`.
- `From<&$sparse>`: same cumulative pattern.
- Iterator types: `CumulativeIter` / `CumulativeIter32`, `QuantileRangeIter` / `QuantileRangeIter32`.

Macro signature: `define_cumulative_histogram!($name:ident, $iter:ident, $qr_iter:ident, $hist:ident, $sparse:ident, $count:ty)`.

Two invocations:

```rust
define_cumulative_histogram!(
    CumulativeROHistogram, CumulativeIter, QuantileRangeIter,
    Histogram, SparseHistogram, u64
);
define_cumulative_histogram!(
    CumulativeROHistogram32, CumulativeIter32, QuantileRangeIter32,
    Histogram32, SparseHistogram32, u32
);
```

The full macro body mirrors the existing `src/cumulative.rs` impl block with `$count` substituted and the type-dependent const + `.as_u128()` calls applied. Since this is a mechanical translation, follow the existing source carefully — every numeric literal that was `0u64` becomes `<$count as Count>::ZERO`; every `as u128` becomes `.as_u128()`; every `as u64` widening for `total_count`/`individual_count`/`Bucket` becomes `(.as_u128()) as u64`.

- [ ] **Step 3: Add u32-specific tests**

```rust
#[test]
fn from_histogram_u32() {
    let mut h = Histogram32::new(7, 64).unwrap();
    h.increment(1).unwrap();
    h.increment(1).unwrap();
    h.increment(5).unwrap();
    h.increment(100).unwrap();
    let croh = CumulativeROHistogram32::from(&h);
    assert_eq!(croh.index().len(), 3);
    assert_eq!(croh.count(), &[2u32, 3, 4]);
    assert_eq!(croh.total_count(), 4);
}

#[test]
fn from_parts_u32() {
    let config = Config::new(7, 32).unwrap();
    let croh = CumulativeROHistogram32::from_parts(
        config, vec![1, 3, 5], vec![6u32, 18, 25],
    ).unwrap();
    assert_eq!(croh.total_count(), 25);
}

#[test]
fn quantiles_u32_match_u64() {
    let mut h32 = Histogram32::new(4, 10).unwrap();
    let mut h64 = Histogram::new(4, 10).unwrap();
    for v in 1..1024u64 {
        h32.increment(v).unwrap();
        h64.increment(v).unwrap();
    }
    let c32 = CumulativeROHistogram32::from(&h32);
    let c64 = CumulativeROHistogram::from(&h64);
    let qs = &[0.0, 0.5, 0.99, 1.0];
    let r32 = c32.quantiles(qs).unwrap().unwrap();
    let r64 = c64.quantiles(qs).unwrap().unwrap();
    for ((q32, _), (q64, _)) in r32.entries().iter().zip(r64.entries().iter()) {
        assert_eq!(q32, q64);
    }
}
```

- [ ] **Step 4: cargo fmt + run all tests**

Run: `cargo fmt && cargo test`

- [ ] **Step 5: Commit**

```bash
git -C /Users/yao/workspace/histogram/.worktrees/u32-bucket-counts add src/cumulative.rs
git -C /Users/yao/workspace/histogram/.worktrees/u32-bucket-counts commit -m "macroify CumulativeROHistogram and add CumulativeROHistogram32

Wrap in define_cumulative_histogram! macro. Cache-line threshold for
linear-vs-binary search is now count-type-dependent (64 / size_of::<C>).
total_count and individual_count continue to return u64."
```

## Task 6: Cross-width same-variant `From` (widening) impls

**Files:**
- Create: `src/conversions.rs`
- Modify: `src/lib.rs` (add `mod conversions;`, export new types)

- [ ] **Step 1: Create `src/conversions.rs` with widening impls**

```rust
//! Cross-width and combined cross-variant + narrowing conversions
//! between histogram type families.

use crate::{
    AtomicCount, AtomicHistogram, AtomicHistogram32, CumulativeROHistogram,
    CumulativeROHistogram32, Error, Histogram, Histogram32, SparseHistogram,
    SparseHistogram32,
};

// ---------------- Widening (u32 -> u64) ----------------

impl From<&Histogram32> for Histogram {
    fn from(h: &Histogram32) -> Self {
        let buckets: Vec<u64> = h.as_slice().iter().map(|&c| c as u64).collect();
        Histogram::from_buckets(
            h.config().grouping_power(),
            h.config().max_value_power(),
            buckets,
        ).expect("widening preserves bucket count")
    }
}

impl From<&AtomicHistogram32> for AtomicHistogram {
    fn from(h: &AtomicHistogram32) -> Self {
        let snapshot = h.load();
        let widened: Histogram = (&snapshot).into();
        let out = AtomicHistogram::with_config(&widened.config());
        for (i, &c) in widened.as_slice().iter().enumerate() {
            out.buckets[i].fetch_add_relaxed(c);
        }
        out
    }
}

impl From<&SparseHistogram32> for SparseHistogram {
    fn from(h: &SparseHistogram32) -> Self {
        let widened: Vec<u64> = h.count().iter().map(|&c| c as u64).collect();
        SparseHistogram::from_parts(h.config(), h.index().to_vec(), widened)
            .expect("widening preserves invariants")
    }
}

impl From<&CumulativeROHistogram32> for CumulativeROHistogram {
    fn from(h: &CumulativeROHistogram32) -> Self {
        let widened: Vec<u64> = h.count().iter().map(|&c| c as u64).collect();
        CumulativeROHistogram::from_parts(h.config(), h.index().to_vec(), widened)
            .expect("widening preserves invariants")
    }
}
```

- [ ] **Step 2: Wire the module into `src/lib.rs`**

Add `mod conversions;` to the module list. Update re-exports to include new types:

```rust
pub use atomic::{AtomicHistogram, AtomicHistogram32};
pub use cumulative::{CumulativeROHistogram, CumulativeROHistogram32};
pub use sparse::{SparseHistogram, SparseHistogram32};
pub use standard::{Histogram, Histogram32};
```

- [ ] **Step 3: Add widening tests in `src/conversions.rs`**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::Config;

    #[test]
    fn widen_histogram() {
        let mut h32 = Histogram32::new(7, 32).unwrap();
        h32.add(1, 1234u32).unwrap();
        h32.add(1000, 5678u32).unwrap();
        let h64: Histogram = (&h32).into();
        assert_eq!(h64.config(), h32.config());
        for (a, b) in h64.as_slice().iter().zip(h32.as_slice().iter()) {
            assert_eq!(*a, *b as u64);
        }
    }

    #[test]
    fn widen_sparse() {
        let config = Config::new(7, 32).unwrap();
        let s32 = SparseHistogram32::from_parts(config, vec![1, 3], vec![10u32, 20]).unwrap();
        let s64: SparseHistogram = (&s32).into();
        assert_eq!(s64.count(), &[10u64, 20]);
        assert_eq!(s64.index(), &[1u32, 3]);
    }

    #[test]
    fn widen_cumulative() {
        let config = Config::new(7, 32).unwrap();
        let c32 = CumulativeROHistogram32::from_parts(
            config, vec![1, 3], vec![10u32, 30],
        ).unwrap();
        let c64: CumulativeROHistogram = (&c32).into();
        assert_eq!(c64.count(), &[10u64, 30]);
    }

    #[cfg(target_has_atomic = "32")]
    #[cfg(target_has_atomic = "64")]
    #[test]
    fn widen_atomic_histogram() {
        let h32 = AtomicHistogram32::new(7, 32).unwrap();
        h32.add(5, 100u32).unwrap();
        h32.add(50, 200u32).unwrap();
        let h64: AtomicHistogram = (&h32).into();
        let snap = h64.load();
        let total: u64 = snap.as_slice().iter().sum();
        assert_eq!(total, 300);
    }
}
```

- [ ] **Step 4: Run + commit**

Run: `cargo fmt && cargo test`

```bash
git -C /Users/yao/workspace/histogram/.worktrees/u32-bucket-counts add src/conversions.rs src/lib.rs
git -C /Users/yao/workspace/histogram/.worktrees/u32-bucket-counts commit -m "add cross-width widening From impls

u32 -> u64 widening for all four histogram families. Infallible."
```

## Task 7: Cross-width same-variant `TryFrom` (narrowing) impls

**Files:**
- Modify: `src/conversions.rs`

- [ ] **Step 1: Add narrowing impls (above the test module)**

```rust
// ---------------- Narrowing (u64 -> u32) ----------------

impl TryFrom<&Histogram> for Histogram32 {
    type Error = Error;
    fn try_from(h: &Histogram) -> Result<Self, Error> {
        let mut narrowed: Vec<u32> = Vec::with_capacity(h.as_slice().len());
        for &c in h.as_slice() {
            narrowed.push(u32::try_from(c).map_err(|_| Error::Overflow)?);
        }
        Histogram32::from_buckets(
            h.config().grouping_power(),
            h.config().max_value_power(),
            narrowed,
        )
    }
}

impl TryFrom<&SparseHistogram> for SparseHistogram32 {
    type Error = Error;
    fn try_from(h: &SparseHistogram) -> Result<Self, Error> {
        let mut narrowed: Vec<u32> = Vec::with_capacity(h.count().len());
        for &c in h.count() {
            narrowed.push(u32::try_from(c).map_err(|_| Error::Overflow)?);
        }
        SparseHistogram32::from_parts(h.config(), h.index().to_vec(), narrowed)
    }
}

impl TryFrom<&CumulativeROHistogram> for CumulativeROHistogram32 {
    type Error = Error;
    fn try_from(h: &CumulativeROHistogram) -> Result<Self, Error> {
        if let Some(&last) = h.count().last() {
            if u32::try_from(last).is_err() {
                return Err(Error::Overflow);
            }
        }
        let narrowed: Vec<u32> = h.count().iter().map(|&c| c as u32).collect();
        CumulativeROHistogram32::from_parts(h.config(), h.index().to_vec(), narrowed)
    }
}
```

- [ ] **Step 2: Add narrowing tests in `mod tests`**

```rust
#[test]
fn narrow_histogram_success() {
    let mut h64 = Histogram::new(7, 32).unwrap();
    h64.add(1, 100u64).unwrap();
    h64.add(1000, 200u64).unwrap();
    let h32: Histogram32 = (&h64).try_into().unwrap();
    assert_eq!(h32.as_slice()[1], 100u32);
}

#[test]
fn narrow_histogram_overflow() {
    let mut h64 = Histogram::new(2, 4).unwrap();
    h64.add(1, (u32::MAX as u64) + 1).unwrap();
    let r: Result<Histogram32, _> = (&h64).try_into();
    assert_eq!(r, Err(Error::Overflow));
}

#[test]
fn narrow_sparse_overflow() {
    let config = Config::new(7, 32).unwrap();
    let s64 = SparseHistogram::from_parts(
        config, vec![1], vec![(u32::MAX as u64) + 1],
    ).unwrap();
    let r: Result<SparseHistogram32, _> = (&s64).try_into();
    assert_eq!(r, Err(Error::Overflow));
}

#[test]
fn narrow_cumulative_checks_total_only() {
    let config = Config::new(7, 32).unwrap();
    let c64 = CumulativeROHistogram::from_parts(
        config, vec![1, 3], vec![100u64, (u32::MAX as u64) + 1],
    ).unwrap();
    let r: Result<CumulativeROHistogram32, _> = (&c64).try_into();
    assert_eq!(r, Err(Error::Overflow));

    let c64_ok = CumulativeROHistogram::from_parts(
        config, vec![1, 3], vec![100u64, 200],
    ).unwrap();
    let c32: CumulativeROHistogram32 = (&c64_ok).try_into().unwrap();
    assert_eq!(c32.total_count(), 200);
}

#[test]
fn round_trip_widen_then_narrow() {
    let mut h32 = Histogram32::new(7, 32).unwrap();
    h32.add(5, 1234u32).unwrap();
    h32.add(50, 5678u32).unwrap();
    let h64: Histogram = (&h32).into();
    let h32_back: Histogram32 = (&h64).try_into().unwrap();
    assert_eq!(h32.as_slice(), h32_back.as_slice());
}
```

- [ ] **Step 3: Run + commit**

Run: `cargo fmt && cargo test`

```bash
git -C /Users/yao/workspace/histogram/.worktrees/u32-bucket-counts add src/conversions.rs
git -C /Users/yao/workspace/histogram/.worktrees/u32-bucket-counts commit -m "add cross-width narrowing TryFrom impls

u64 -> u32 narrowing for Histogram, SparseHistogram, CumulativeROHistogram.
Cumulative narrowing checks only the total count (last cumulative
value) since it bounds every prefix sum."
```

## Task 8: Cross-variant + narrowing combined `TryFrom` impls

**Files:**
- Modify: `src/conversions.rs`

- [ ] **Step 1: Add direct paths**

```rust
// -------- Cross-variant + narrowing (u64 -> u32) --------

impl TryFrom<&Histogram> for CumulativeROHistogram32 {
    type Error = Error;
    fn try_from(h: &Histogram) -> Result<Self, Error> {
        let mut index: Vec<u32> = Vec::new();
        let mut count: Vec<u32> = Vec::new();
        let mut running: u64 = 0;
        for (i, &n) in h.as_slice().iter().enumerate() {
            if n > 0 {
                running = running.checked_add(n).ok_or(Error::Overflow)?;
                if running > u32::MAX as u64 { return Err(Error::Overflow); }
                index.push(i as u32);
                count.push(running as u32);
            }
        }
        CumulativeROHistogram32::from_parts(h.config(), index, count)
    }
}

impl TryFrom<&Histogram> for SparseHistogram32 {
    type Error = Error;
    fn try_from(h: &Histogram) -> Result<Self, Error> {
        let mut index: Vec<u32> = Vec::new();
        let mut count: Vec<u32> = Vec::new();
        for (i, &n) in h.as_slice().iter().enumerate() {
            if n > 0 {
                count.push(u32::try_from(n).map_err(|_| Error::Overflow)?);
                index.push(i as u32);
            }
        }
        SparseHistogram32::from_parts(h.config(), index, count)
    }
}

impl TryFrom<&SparseHistogram> for CumulativeROHistogram32 {
    type Error = Error;
    fn try_from(h: &SparseHistogram) -> Result<Self, Error> {
        let mut running: u64 = 0;
        let mut count: Vec<u32> = Vec::with_capacity(h.count().len());
        for &n in h.count() {
            running = running.checked_add(n).ok_or(Error::Overflow)?;
            if running > u32::MAX as u64 { return Err(Error::Overflow); }
            count.push(running as u32);
        }
        CumulativeROHistogram32::from_parts(h.config(), h.index().to_vec(), count)
    }
}
```

- [ ] **Step 2: Add tests**

```rust
#[test]
fn histogram_to_cumulative32() {
    let mut h = Histogram::new(7, 32).unwrap();
    h.add(1, 100u64).unwrap();
    h.add(50, 200u64).unwrap();
    h.add(1000, 300u64).unwrap();
    let croh: CumulativeROHistogram32 = (&h).try_into().unwrap();
    assert_eq!(croh.total_count(), 600);
    assert_eq!(croh.count().len(), 3);
}

#[test]
fn histogram_to_cumulative32_overflow() {
    let mut h = Histogram::new(2, 4).unwrap();
    h.add(0, 3_000_000_000u64).unwrap();
    h.add(1, 2_000_000_000u64).unwrap();
    let r: Result<CumulativeROHistogram32, _> = (&h).try_into();
    assert_eq!(r, Err(Error::Overflow));
}

#[test]
fn histogram_to_sparse32() {
    let mut h = Histogram::new(7, 32).unwrap();
    h.add(1, 100u64).unwrap();
    h.add(1000, 200u64).unwrap();
    let s: SparseHistogram32 = (&h).try_into().unwrap();
    assert_eq!(s.count().iter().map(|&c| c as u64).sum::<u64>(), 300);
}

#[test]
fn sparse_to_cumulative32() {
    let config = Config::new(7, 32).unwrap();
    let s = SparseHistogram::from_parts(config, vec![1, 3], vec![100u64, 200]).unwrap();
    let c: CumulativeROHistogram32 = (&s).try_into().unwrap();
    assert_eq!(c.count(), &[100u32, 300]);
}

#[test]
fn direct_path_matches_two_step() {
    let mut h = Histogram::new(4, 10).unwrap();
    for v in 1..1024u64 { h.increment(v).unwrap(); }
    let direct: CumulativeROHistogram32 = (&h).try_into().unwrap();
    let mid: CumulativeROHistogram = (&h).into();
    let two_step: CumulativeROHistogram32 = (&mid).try_into().unwrap();
    assert_eq!(direct.count(), two_step.count());
    assert_eq!(direct.index(), two_step.index());
}

#[test]
fn snapshot_pipeline_end_to_end() {
    let recorder = AtomicHistogram::new(7, 64).unwrap();
    for v in 1..=50u64 { recorder.increment(v).unwrap(); }
    let snap_t0 = recorder.load();
    for v in 1..=50u64 { recorder.increment(v).unwrap(); }
    let snap_t1 = recorder.load();
    let delta = snap_t1.checked_sub(&snap_t0).unwrap();
    let analytic: CumulativeROHistogram32 = (&delta).try_into().unwrap();
    assert_eq!(analytic.total_count(), 50);
}
```

- [ ] **Step 3: Run + commit**

```bash
git -C /Users/yao/workspace/histogram/.worktrees/u32-bucket-counts add src/conversions.rs
git -C /Users/yao/workspace/histogram/.worktrees/u32-bucket-counts commit -m "add cross-variant + narrowing TryFrom impls for snapshot pipeline

Direct paths Histogram -> CumulativeROHistogram32, Histogram ->
SparseHistogram32, SparseHistogram -> CumulativeROHistogram32. Each
is a single pass."
```

## Task 9: Update benchmarks

**Files:**
- Modify: `benches/histogram.rs`

- [ ] **Step 1: Replace contents**

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
    let mut histogram = histogram::Histogram::new(7, 64).unwrap();
    benchmark!("histogram/u64", histogram, c);
}
fn histogram_u32(c: &mut Criterion) {
    let mut histogram = histogram::Histogram32::new(7, 64).unwrap();
    benchmark!("histogram/u32", histogram, c);
}
fn atomic_u64(c: &mut Criterion) {
    let histogram = histogram::AtomicHistogram::new(7, 64).unwrap();
    benchmark!("atomic_histogram/u64", histogram, c);
}
fn atomic_u32(c: &mut Criterion) {
    let histogram = histogram::AtomicHistogram32::new(7, 64).unwrap();
    benchmark!("atomic_histogram/u32", histogram, c);
}

criterion_group!(benches, histogram_u64, histogram_u32, atomic_u64, atomic_u32);
criterion_main!(benches);
```

- [ ] **Step 2: Verify + commit**

Run: `cargo bench --no-run`

```bash
git -C /Users/yao/workspace/histogram/.worktrees/u32-bucket-counts add benches/histogram.rs
git -C /Users/yao/workspace/histogram/.worktrees/u32-bucket-counts commit -m "add u32 benchmark cases for Histogram and AtomicHistogram"
```

## Task 10: Documentation + version bump

**Files:**
- Modify: `README.md`
- Modify: `src/lib.rs` (rustdoc + Types section)
- Modify: `src/config.rs` (memory-table footnote)
- Modify: `Cargo.toml` (version → `1.3.0-alpha.0` or next alpha)

- [ ] **Step 1: Update `README.md`**

After the existing "Histogram Types" section, insert:

```markdown
## Counter Width

All four histogram types ship in two flavors:

- **u64-counter family** (`Histogram`, `AtomicHistogram`, `SparseHistogram`, `CumulativeROHistogram`): the default. Counts up to 2^64 − 1 per bucket.
- **u32-counter siblings** (`Histogram32`, `AtomicHistogram32`, `SparseHistogram32`, `CumulativeROHistogram32`): half the memory and serialization size, counts up to 2^32 − 1 per bucket.

Pick the family based on the memory/range tradeoff. Conversions:

- **Widening** (`u32` → `u64`) is infallible (`From`).
- **Narrowing** (`u64` → `u32`) is fallible (`TryFrom`, returns `Err(Overflow)`). Direct cross-variant + narrowing paths support the snapshot pipeline.

## Recommended Pipeline

Pick the histogram type based on the *role* it plays in your data flow:

- **Recording — `AtomicHistogram` (or `Histogram`).** Use u64-counter types for the long-running, continuously-updated histogram. Counts here are unbounded over the lifetime of the process; `u64` heads off any practical risk of overflow.
- **Snapshot delta — `Histogram`, then narrowed.** When you take periodic snapshots and compute a delta with `checked_sub`, the delta covers only the activity in one window. Use `Histogram::checked_sub` to compute the delta, then `TryFrom` to narrow into a `*32` type.
- **Read-only analytics — `CumulativeROHistogram32`.** This is the recommended storage and query format for completed snapshots. The cumulative-prefix-sum representation gives you O(log n) quantile queries via binary search, while `u32` counts halve the on-the-wire and on-disk size versus `u64`. Narrowing is checked once against the *total count* (cheaper than per-bucket), and any total ≤ ~4.3B fits.

```rust
use histogram::{AtomicHistogram, CumulativeROHistogram32, Histogram};

let recorder = AtomicHistogram::new(7, 64).unwrap();
# let snap_t0 = recorder.load();
let snap_t1 = recorder.load();
let delta = snap_t1.checked_sub(&snap_t0).unwrap();
let analytic: CumulativeROHistogram32 =
    CumulativeROHistogram32::try_from(&delta).unwrap();
```

If you don't take snapshots — i.e., you query the recording histogram directly — just stay on the u64 types everywhere. The narrowing optimization is specifically for the snapshot/delta pattern.

For JavaScript-frontend plotting specifically, prefer `CumulativeROHistogram32` over a hypothetical f32-backed alternative: `u32` is exact up to ~4.3B (vs f32 exact only to ~16M), and cumulative-monotonicity is structurally preserved (no rounding-induced plateau artifacts in ECDF rendering).
```

Update the existing "Histogram Types" section to add four `*32` bullets.

- [ ] **Step 2: Update `src/lib.rs` module rustdoc**

Mirror the "Counter Width" and "Recommended Pipeline" sections in the crate-level `//!` rustdoc. Update the `# Types` list to mention all eight types. The doctest example may need a `# let snap_t0 = recorder.load();` hidden line to compile.

- [ ] **Step 3: Per-type rustdoc**

The macro emits a generic doc comment for each type. To customize per-invocation, modify each macro to accept a leading `#[doc = "..."]` attribute. Quick approach: extract the type-specific doc into a separate `/// ...` comment placed immediately before the `pub struct $name { ... }` line inside the macro body, and parameterize the doc via a `$doc:literal` macro argument.

If that complicates the macro, a simpler approach: post-macro, write a separate `impl <type> { /* no methods */ }` block with module-level rustdoc comments that document the type's purpose. The `*32` types each get a one-paragraph rustdoc covering counter width, memory tradeoff, overflow ceiling, and a pointer to the conversion API.

- [ ] **Step 4: Memory-table footnote in `src/config.rs`**

After the existing memory table block, append:

```rust
/// Halve all sizes for `*32` histograms (`Histogram32`, `AtomicHistogram32`,
/// `SparseHistogram32`, `CumulativeROHistogram32`).
```

- [ ] **Step 5: Build docs locally**

Run: `cargo doc --no-deps`
Expected: clean build, no broken intra-doc links.

Run: `cargo test --doc`
Expected: doctests pass.

- [ ] **Step 6: Bump version in `Cargo.toml`**

Set `version = "1.3.0-alpha.0"` (or next alpha if `main` has advanced).

- [ ] **Step 7: Commit (final commit retains Co-Authored-By trailer per user instruction)**

```bash
git -C /Users/yao/workspace/histogram/.worktrees/u32-bucket-counts add README.md src/lib.rs src/config.rs Cargo.toml
git -C /Users/yao/workspace/histogram/.worktrees/u32-bucket-counts commit -m "$(cat <<'EOF'
add docs and bump version for u32 bucket counts

- README: new "Counter Width" and "Recommended Pipeline" sections
- lib.rs: matching rustdoc with the snapshot pipeline guidance
- per-type rustdoc: note the *32 sibling family
- config.rs: footnote on the memory table
- Cargo.toml: bump to 1.3.0-alpha.0

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Final Verification

- [ ] **Full test suite:** `cargo test` — every unit / integration / doctest passes.
- [ ] **Bench compile:** `cargo bench --no-run` — clean.
- [ ] **Lint:** `cargo clippy --all-targets -- -D warnings` — no warnings.
- [ ] **Docs:** `cargo doc --no-deps` — clean.
- [ ] **Confirm with user before opening PR** per CLAUDE.md.

---

## Self-Review Notes

**Spec coverage:**
- ✅ Count trait (Task 1, already shipped)
- ✅ Histogram macro pair → Task 2 + Task 4 step 3
- ✅ AtomicHistogram macro pair → Task 3 + Task 4 step 4
- ✅ SparseHistogram macro pair → Task 4
- ✅ CumulativeROHistogram macro pair → Task 5
- ✅ Same-width cross-variant From impls → Tasks 2, 4, 5 (inside the macros)
- ✅ Cross-width widening From → Task 6
- ✅ Cross-width narrowing TryFrom → Task 7
- ✅ Cross-variant + narrowing TryFrom → Task 8
- ✅ Bench updates → Task 9
- ✅ Documentation + version bump → Task 10

**Type-name consistency:** struct names (`Histogram`, `Histogram32`, etc.), iterator names (`Iter` / `Iter32`, `SparseIter` / `SparseIter32`, `CumulativeIter` / `CumulativeIter32`, `QuantileRangeIter` / `QuantileRangeIter32`), macro names (`define_histogram!`, `define_atomic_histogram!`, `define_sparse_histogram!`, `define_cumulative_histogram!`) used consistently across tasks.

**Cross-task dependency hazard:** Tasks 2 and 3 reference types that don't exist until Task 4 completes. Plan handles this with commented-out invocations; Task 4 step 3/4 uncomments them.

**Deprecation notice:** Task 2 drops the deprecated `percentile`/`percentiles` inherent methods. Documented in Task 10 commit and CHANGELOG.
