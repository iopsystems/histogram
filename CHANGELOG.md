# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- `from_parts_unchecked` on `CumulativeROHistogramRef` /
  `CumulativeROHistogram32Ref` / `SparseHistogramRef` /
  `SparseHistogram32Ref` now runs the same validation as `from_parts`
  inside a `debug_assert!`. Debug builds catch invariant violations at
  the call site; release builds are unchanged (validation is elided).

## [1.3.1] - 2026-04-29

### Added

- `CumulativeROHistogramRef<'a>` / `CumulativeROHistogram32Ref<'a>` and
  `SparseHistogramRef<'a>` / `SparseHistogram32Ref<'a>` — borrowed views over
  histogram storage that mirror the read-only API surface of the owned types.
  Lets columnar consumers compute quantiles directly off `&[u32]` slices
  without per-snapshot allocation or revalidation.
- `from_parts_unchecked` constructor on each ref type for hot-path use where
  invariants are already known to hold; `from_parts` validates as before.
- `as_ref()` on `CumulativeROHistogram*` and `SparseHistogram*` returning the
  matching ref view; `From<&Owned> for OwnedRef<'_>` for trait-based conversion.
- `SampleQuantiles` impl on each ref type.

## [1.3.0] - 2026-04-29

### Added

- `Histogram32`, `AtomicHistogram32`, `SparseHistogram32`, and
  `CumulativeROHistogram32` — u32-counter sibling types for all four histogram
  variants, generated from shared declarative macros
- Sealed `Count` and `AtomicCount` traits abstracting the bucket counter width
  (implemented for `u32`/`u64` and their atomic counterparts)
- Cross-width widening `From` conversions (u32 → u64) for all four histogram
  families
- Cross-width narrowing `TryFrom` conversions (u64 → u32) for `Histogram`,
  `SparseHistogram`, and `CumulativeROHistogram`
- Cross-variant + narrowing `TryFrom` paths for the snapshot pipeline:
  `Histogram` → `CumulativeROHistogram32`, `Histogram` → `SparseHistogram32`,
  and `SparseHistogram` → `CumulativeROHistogram32`
- u32 benchmark cases for `Histogram` and `AtomicHistogram`
- `rezolus_memory` example comparing `Histogram`, `SparseHistogram`, and
  `CumulativeROHistogram` memory footprints from Rezolus parquet recordings
- README "Counter Width" and "Recommended Pipeline" sections, with matching
  rustdoc in `lib.rs`

### Changed

- `CumulativeROHistogram` cache-line threshold for linear-vs-binary search is
  now count-type-dependent (`64 / size_of::<C>()`); `total_count` and
  `individual_count` continue to return `u64`

## [1.2.0] - 2026-04-22

### Added

- `CumulativeROHistogram`, a read-only histogram variant with cumulative
  (prefix-sum) counts that enables O(log n) quantile lookups via binary search
  (with a linear-scan fallback for cache-line-sized data)
- `From<&Histogram>` and `From<&SparseHistogram>` conversions for constructing
  a `CumulativeROHistogram`
- Quantile range query methods on `CumulativeROHistogram` for analytics use
  cases

## [1.1.0] - 2026-04-06

### Added

- `SampleQuantiles` trait with `quantiles()` and `quantile()` methods, implemented
  for both `Histogram` and `SparseHistogram`
- `QuantilesResult` struct returning a `BTreeMap<Quantile, Bucket>` with
  `total_count`, `min`, and `max` bucket metadata
- `Quantile` newtype wrapping validated `f64` in `0.0..=1.0`, implementing
  `Ord`/`Eq` for use as a `BTreeMap` key
- `Error::InvalidQuantile` variant for quantile validation errors

### Deprecated

- `Histogram::percentiles()` and `Histogram::percentile()` — use `SampleQuantiles`
  trait methods instead
- `SparseHistogram::percentiles()` and `SparseHistogram::percentile()` — use
  `SampleQuantiles` trait methods instead
- `Error::InvalidPercentile` — use `Error::InvalidQuantile` instead

## [1.0.0] - 2026-03-20

First release with changelog.
