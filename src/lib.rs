//! This crate provides histogram implementations that are conceptually similar
//! to HdrHistogram, with modifications to the bucket construction and indexing
//! algorithms that we believe provide a simpler implementation and more
//! efficient runtime compared to the reference implementation of HdrHistogram.
//!
//! # Types
//!
//! - [`Histogram`] — standard histogram with `u64` counters. Use for
//!   single-threaded recording and percentile queries.
//! - [`AtomicHistogram`] — atomic histogram for concurrent recording. Take a
//!   snapshot with [`AtomicHistogram::load`] or [`AtomicHistogram::drain`] to
//!   query percentiles.
//! - [`SparseHistogram`] — compact representation storing only non-zero
//!   buckets. Useful for serialization and storage.
//! - [`CumulativeROHistogram`] — read-only histogram with cumulative counts
//!   for fast quantile queries via binary search.
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
//! # Background
//! Please see: <https://h2histogram.org>

mod atomic;
mod bucket;
mod config;
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
pub use cumulative::CumulativeROHistogram;
pub use errors::Error;
pub use quantile::{Quantile, QuantilesResult, SampleQuantiles};
pub use sparse::{SparseHistogram, SparseHistogram32};
pub use standard::{Histogram, Histogram32};
