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
pub use cumulative::{CumulativeROHistogram, CumulativeROHistogram32};
pub use errors::Error;
pub use quantile::{Quantile, QuantilesResult, SampleQuantiles};
pub use sparse::{SparseHistogram, SparseHistogram32};
pub use standard::{Histogram, Histogram32};
