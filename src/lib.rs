//! This crate provides histogram implementations that are conceptually similar
//! to HdrHistogram, with modifications to the bucket construction and indexing
//! algorithms that we believe provide a simpler implementation and more
//! efficient runtime compared to the reference implementation of HdrHistogram.
//!
//! # Types
//!
//! - [`Histogram`] — standard histogram with `u64` counters. Use for
//!   single-threaded recording and percentile queries.
//! - [`Histogram32`] — like [`Histogram`] but with `u32` counters.
//! - [`AtomicHistogram`] — atomic histogram for concurrent recording. Take a
//!   snapshot with [`AtomicHistogram::load`] or [`AtomicHistogram::drain`] to
//!   query percentiles.
//! - [`AtomicHistogram32`] — like [`AtomicHistogram`] but with `u32` counters.
//! - [`SparseHistogram`] — compact representation storing only non-zero
//!   buckets. Useful for serialization and storage.
//! - [`SparseHistogram32`] — like [`SparseHistogram`] but with `u32` counters.
//! - [`CumulativeROHistogram`] — read-only histogram with cumulative counts
//!   for fast quantile queries via binary search.
//! - [`CumulativeROHistogram32`] — like [`CumulativeROHistogram`] but with
//!   `u32` counters.
//!
//! # Example
//!
//! ```
//! use histogram::{Histogram, Quantile};
//!
//! let mut h = Histogram::new(7, 64).unwrap();
//!
//! for value in 1..=100 {
//!     h.increment(value).unwrap();
//! }
//!
//! // Quantiles use the 0.0..=1.0 scale
//! let report = h.quantiles(&[0.5, 0.99]).unwrap().unwrap();
//! // quantiles() shares scan work and returns Result<Option<QuantilesResult>, Error>
//! // outer unwrap: quantile value is valid
//! // inner unwrap: histogram is non-empty
//!
//! let p50 = report.get(&Quantile::new(0.5).unwrap()).unwrap();
//! let p99 = report.get(&Quantile::new(0.99).unwrap()).unwrap();
//! println!("p50: {}-{}", p50.start(), p50.end());
//! println!("p99: {}-{}", p99.start(), p99.end());
//! ```
//!
//! # Workflow and counter width
//!
//! Choose writer ownership first: per-writer dense histograms require handoff and
//! aggregation; shared atomic histograms may contend. Batch a report's quantiles
//! to share scans. Convert completed data to cumulative snapshots when repeated
//! binary-search queries justify construction cost. Dense recording keeps no
//! cached total or bounds; reporting derives them when needed.
//!
//! u32 halves counter storage, not indices, object overhead, or necessarily wire
//! size. Dense, atomic, and sparse counts must fit **per bucket**; cumulative prefix
//! counts must fit **in total**. Recording wraps on overflow. Choose width from
//! count bounds and reset cadence; u32 can also suit direct recording and reporting.
//! Dense queries widen sums to u128. Widening u32 to u64 is infallible; checked
//! narrowing uses `TryFrom`. Ensure totals fit when constructing cumulative prefixes.
//!
//! # Reusing reporting storage
//!
//! [`Histogram::reset`] clears counts in existing storage.
//! [`Histogram::checked_add_assign`] aggregates compatible histograms without
//! allocating and leaves counts unchanged on overflow or configuration mismatch.
//! It validates all additions before a second pass applies them.
//!
//! [`AtomicHistogram::load_into`] and [`AtomicHistogram::drain_into`] overwrite a
//! compatible dense destination without allocating. Each bucket is observed or
//! captured/cleared individually; no instantaneous histogram-wide boundary or
//! publication of unrelated data is provided. Coordinate writers externally if an
//! exact interval boundary is required. The 32-bit types provide the same APIs.
//!
//! ```
//! use histogram::{AtomicHistogram, Histogram};
//! let recorder = AtomicHistogram::new(7, 32).unwrap();
//! let mut window = Histogram::with_config(&recorder.config());
//! let mut aggregate = Histogram::with_config(&recorder.config());
//! recorder.increment(100).unwrap();
//! recorder.drain_into(&mut window).unwrap();
//! aggregate.checked_add_assign(&window).unwrap();
//! assert_eq!(aggregate.quantile_bucket(0.99).unwrap().unwrap().count(), 1);
//! ```
//!
//! # Allocation-free bucket queries
//!
//! Dense histograms and owned/borrowed cumulative snapshots support scalar bucket
//! queries and batches into caller-provided storage without allocating results.
//! Dense batches take O(B + Q²) work for B buckets and Q requests, intended for
//! small reporting batches. Cumulative batches take O(Q log K) for K stored buckets.
//! Requests need not be sorted, and duplicates preserve their input positions.
//! The existing [`SampleQuantiles`] API remains available for a result map and
//! min/max/total metadata.
//!
//! ```
//! use histogram::{Bucket, CumulativeROHistogram32, Histogram};
//!
//! let mut recorder = Histogram::new(7, 32).unwrap();
//! recorder.increment(100).unwrap();
//! let snapshot = CumulativeROHistogram32::try_from(&recorder).unwrap();
//! let bucket = snapshot.quantile_bucket(0.99).unwrap().unwrap();
//! assert_eq!(bucket.count(), 1);
//!
//! let mut output: [Option<Bucket>; 3] = std::array::from_fn(|_| None);
//! snapshot.as_ref().quantile_buckets_into(&[0.99, 0.5, 0.99], &mut output).unwrap();
//! assert_eq!(output[0], output[2]);
//! ```
//!
//! # Compacting retained snapshots
//!
//! Owned sparse and cumulative histograms provide an opt-in `shrink_to_fit()`
//! method to reduce spare vector capacity after construction. It preserves
//! observations, query results, cached means, and serialization. Compaction may
//! reallocate/move the vectors; exact capacity and lower process RSS are not
//! guaranteed. Conversions do not compact automatically.
//!
//! ```
//! use histogram::{CumulativeROHistogram32, Histogram, SparseHistogram};
//!
//! let mut recorder = Histogram::new(10, 30).unwrap();
//! recorder.increment(1000).unwrap();
//! let mut sparse = SparseHistogram::from(&recorder);
//! let mut cumulative = CumulativeROHistogram32::try_from(&recorder).unwrap();
//! let mean = cumulative.mean();
//! sparse.shrink_to_fit();
//! cumulative.shrink_to_fit();
//! assert_eq!(cumulative.mean(), mean);
//! assert_eq!(cumulative.total_count(), 1);
//! ```
//!
//! # Transforming retained snapshots
//!
//! Cumulative snapshots and their borrowed views provide `checked_add` and
//! `downsample`, returning new owned snapshots without changing their inputs.
//! Addition requires identical configurations/widths and checks combined total
//! overflow. Downsampling requires a strictly lower grouping power and preserves
//! maximum value power. Both recompute the mean from output bucket midpoints;
//! this is an estimate, not recovery of exact raw-observation moments.
//!
//! ```
//! use histogram::{CumulativeROHistogram32, Histogram};
//! let mut a = Histogram::new(10, 30).unwrap();
//! let mut b = Histogram::new(7, 30).unwrap();
//! a.increment(1000).unwrap();
//! b.increment(2000).unwrap();
//! let a = CumulativeROHistogram32::try_from(&a).unwrap().downsample(7).unwrap();
//! let b = CumulativeROHistogram32::try_from(&b).unwrap();
//! let summary = a.as_ref().checked_add(&b.as_ref()).unwrap();
//! assert_eq!(summary.total_count(), 2);
//! assert!(summary.quantile_bucket(0.99).unwrap().is_some());
//! ```
//!
//! # Background
//! Please see: <https://h2histogram.org>

mod atomic;
mod bucket;
mod config;
mod conversions;
mod count;
mod cumulative;
mod errors;
mod quantile;
mod sparse;
mod standard;

pub use atomic::{AtomicHistogram, AtomicHistogram32};
pub use bucket::Bucket;
pub use config::Config;
pub use count::{AtomicCount, Count};
pub use cumulative::{
    CumulativeROHistogram, CumulativeROHistogram32, CumulativeROHistogram32Ref,
    CumulativeROHistogramRef,
};
pub use errors::Error;
pub use quantile::{Quantile, QuantilesResult, SampleQuantiles};
pub use sparse::{SparseHistogram, SparseHistogram32, SparseHistogram32Ref, SparseHistogramRef};
pub use standard::{Histogram, Histogram32};
