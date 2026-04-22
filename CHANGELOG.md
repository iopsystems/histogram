# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
