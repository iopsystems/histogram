# histogram

A collection of histogram data structures for Rust, providing standard, atomic,
and sparse variants. Like HDRHistogram, values are stored in quantized buckets,
but the bucket construction and indexing algorithm are modified for fast
increments and lookups.

## Getting Started

```
cargo add histogram
```

## Usage

```rust
use histogram::Histogram;

// Create a histogram with grouping power 7 and max value power 64.
let mut histogram = Histogram::new(7, 64).unwrap();

// Record some values.
for i in 1..=100 {
    histogram.increment(i).unwrap();
}

// Query percentiles using the 0.0..=1.0 scale.
let median = histogram.percentile(0.5).unwrap().unwrap();
let p99 = histogram.percentile(0.99).unwrap().unwrap();
// percentile() returns Result<Option<Bucket>, Error>
// outer unwrap: percentile value is valid
// inner unwrap: histogram is non-empty

println!("median: {}", median.end());
println!("p99: {}", p99.end());
```

## Histogram Types

- **Histogram** -- Standard histogram with plain 64-bit counters. Best for
  single-threaded use.
- **AtomicHistogram** -- Uses atomic 64-bit counters, allowing concurrent
  recording from multiple threads. Take a snapshot via `load()` or `drain()`
  to query percentiles.
- **SparseHistogram** -- Columnar representation that only stores non-zero
  buckets. Ideal for serialization and storage when most buckets are empty.
- **CumulativeROHistogram** -- Read-only histogram with cumulative counts for
  fast O(log n) quantile queries via binary search.

All four types ship with a `*32` sibling (`Histogram32`, `AtomicHistogram32`,
`SparseHistogram32`, `CumulativeROHistogram32`) that uses 32-bit counters.

## Counter Width

All four histogram types ship in two flavors:

- **u64-counter family** (`Histogram`, `AtomicHistogram`, `SparseHistogram`, `CumulativeROHistogram`): the default. Counts up to 2^64 − 1 per bucket.
- **u32-counter siblings** (`Histogram32`, `AtomicHistogram32`, `SparseHistogram32`, `CumulativeROHistogram32`): half the memory and serialization size, counts up to 2^32 − 1 per bucket.

Pick the family based on the memory/range tradeoff. Conversions:

- **Widening** (`u32` → `u64`) is infallible (`From`).
- **Narrowing** (`u64` → `u32`) is fallible (`TryFrom`, returns `Err(Overflow)`). Direct cross-variant + narrowing paths support the snapshot pipeline.

## Recommended Pipeline

Pick the histogram type based on the *role* it plays in your data flow:

- **Recording — `AtomicHistogram` (or `Histogram`).** Use the u64-counter types for the long-running, continuously-updated histogram. Counts here are unbounded over the lifetime of the process; `u64` heads off any practical risk of overflow.
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

## Features

- `serde` -- Enables `Serialize` and `Deserialize` for histogram types.
- `schemars` -- Enables JSON Schema generation (implies `serde`).

## Documentation

- [API Documentation](https://docs.rs/histogram)
- [Crates.io](https://crates.io/crates/histogram)
- [Repository](https://github.com/iopsystems/histogram)

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your
option.
