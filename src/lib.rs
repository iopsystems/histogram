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
//! let r50 = h.quantile(0.5).unwrap().unwrap();
//! let r99 = h.quantile(0.99).unwrap().unwrap();
//! // quantile() returns Result<Option<QuantilesResult>, Error>
//! // outer unwrap: quantile value is valid
//! // inner unwrap: histogram is non-empty
//!
//! let p50 = r50.get(&Quantile::new(0.5).unwrap()).unwrap();
//! let p99 = r99.get(&Quantile::new(0.99).unwrap()).unwrap();
//! println!("p50: {}-{}", p50.start(), p50.end());
//! println!("p99: {}-{}", p99.start(), p99.end());
//! ```
//!
//! # Counter Width
//!
//! All four histogram types ship in two flavors:
//!
//! - u64-counter family ([`Histogram`], [`AtomicHistogram`],
//!   [`SparseHistogram`], [`CumulativeROHistogram`]): the default.
//! - u32-counter siblings ([`Histogram32`], [`AtomicHistogram32`],
//!   [`SparseHistogram32`], [`CumulativeROHistogram32`]): half the memory
//!   and serialization size; counts up to 2^32 − 1 per bucket.
//!
//! Conversions: widening (`u32` → `u64`) is infallible (`From`); narrowing
//! (`u64` → `u32`) is fallible (`TryFrom`, returns [`Error::Overflow`]).
//! Direct cross-variant + narrowing paths support the snapshot pipeline.
//!
//! # Recommended Pipeline
//!
//! Pick the histogram type based on the *role* it plays:
//!
//! - **Recording — `AtomicHistogram` or `Histogram`.** Counts are unbounded
//!   over the lifetime of the process; `u64` is the safe choice.
//! - **Snapshot delta — `Histogram`, then narrowed.** Compute the delta with
//!   `checked_sub`, then `TryFrom` into the analytics type.
//! - **Read-only analytics — `CumulativeROHistogram32`.** Halved size, O(log n)
//!   quantile queries, total-count check is cheaper than per-bucket.
//!
//! ```
//! use histogram::{AtomicHistogram, CumulativeROHistogram32, Histogram};
//!
//! let recorder = AtomicHistogram::new(7, 64).unwrap();
//! # let snap_t0 = recorder.load();
//! let snap_t1 = recorder.load();
//! let delta = snap_t1.checked_sub(&snap_t0).unwrap();
//! let analytic: CumulativeROHistogram32 =
//!     CumulativeROHistogram32::try_from(&delta).unwrap();
//! ```
//!
//! # Allocation-free analytical queries
//!
//! Owned and borrowed cumulative histograms support scalar bucket queries and
//! batches into caller-provided storage, without allocating query results.
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
