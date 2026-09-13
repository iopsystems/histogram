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
use histogram::{Histogram, Quantile};

// Create a histogram with grouping power 7 and max value power 64.
let mut histogram = Histogram::new(7, 64).unwrap();

// Record some values.
for i in 1..=100 {
    histogram.increment(i).unwrap();
}

// Query quantiles using the 0.0..=1.0 scale.
let r50 = histogram.quantile(0.5).unwrap().unwrap();
let r99 = histogram.quantile(0.99).unwrap().unwrap();
// quantile() returns Result<Option<QuantilesResult>, Error>
// outer unwrap: quantile value is valid
// inner unwrap: histogram is non-empty

let median = r50.get(&Quantile::new(0.5).unwrap()).unwrap();
let p99 = r99.get(&Quantile::new(0.99).unwrap()).unwrap();

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

## Allocation-free analytical queries

For repeated reads of completed snapshots, the owned and borrowed cumulative
types provide allocation-free bucket queries. Both `u64` and `u32` variants support
these methods; construction/conversion is a separate cost.

```rust
use histogram::{Bucket, CumulativeROHistogram, Histogram};

let mut recorder = Histogram::new(7, 32).unwrap();
for value in [100, 200, 300] {
    recorder.increment(value).unwrap();
}
let snapshot = CumulativeROHistogram::from(&recorder);
let p99 = snapshot.quantile_bucket(0.99).unwrap().unwrap();
println!("p99: {}..={}", p99.start(), p99.end());

let requests = [0.99, 0.5, 0.99];
let mut output: [Option<Bucket>; 3] = std::array::from_fn(|_| None);
let written = snapshot.as_ref().quantile_buckets_into(&requests, &mut output).unwrap();
assert_eq!(written, 3);
assert_eq!(output[0], output[2]);
```

Batch requests may be unsorted and may repeat. Results preserve request order,
and buckets include their individual counts. Empty histograms return/write `None`.
Invalid quantiles or insufficient output capacity return an error without changing
the output; extra output slots are left untouched. Empty requests write nothing.
The existing `quantile()`/`quantiles()` APIs still provide a sorted result map,
total count and min/max metadata when those are needed.

## Compacting retained snapshots

Sparse and cumulative snapshots use vectors that can retain spare capacity after
construction. For snapshots you intend to keep, call `shrink_to_fit()` explicitly:

```rust
use histogram::{CumulativeROHistogram32, Histogram};

let mut recorder = Histogram::new(10, 30).unwrap();
recorder.increment(1000).unwrap();
let mut retained = CumulativeROHistogram32::try_from(&recorder).unwrap();
retained.shrink_to_fit();
```

All four owned sparse/cumulative types support this method. It preserves counts,
quantiles, cached means, and serialization. It can reallocate and move the backing
vectors, so budget the one-time cost at the retention boundary. Short-lived
snapshots can skip it; conversions keep their existing allocation behavior.

Slice lengths describe logical payload, not allocated capacity. Use the vectors
returned by `into_parts()` to inspect their capacities. Like `Vec::shrink_to_fit`,
this method does not guarantee exact capacity or that the allocator returns freed
memory to the operating system. It changes neither precision nor counter width.

## Transforming retained cumulative snapshots

Owned cumulative histograms and their borrowed views support `checked_add` and
`downsample`, returning new owned cumulative snapshots. These operations work on
individual counts derived from adjacent prefixes. Inputs stay unchanged.

```rust
use histogram::{CumulativeROHistogram32, Histogram};

let mut first = Histogram::new(10, 30).unwrap();
let mut second = Histogram::new(7, 30).unwrap();
first.increment(1000).unwrap();
second.increment(2000).unwrap();
let first = CumulativeROHistogram32::try_from(&first).unwrap();
let second = CumulativeROHistogram32::try_from(&second).unwrap();
// Choose a common geometry explicitly before merging.
let first = first.downsample(7).unwrap();
let mut summary = first.as_ref().checked_add(&second.as_ref()).unwrap();
assert_eq!(summary.total_count(), 2);
summary.shrink_to_fit(); // Optional retention decision after transformation.
let p99 = summary.quantile_bucket(0.99).unwrap();
```

Addition requires matching configurations and counter widths. It rejects a
combined total above `u32::MAX` or `u64::MAX`, including disjoint buckets whose
individual counts fit. Use `CumulativeROHistogram::from(&narrow)` to widen before
merging; narrow the result with `CumulativeROHistogram32::try_from(&wide)` only
when its total fits. Different maximum value powers are rejected. Downsampling
requires a strictly smaller grouping power and preserves range, width and total.

For n and m stored input buckets and k occupied output buckets, addition takes
O(n + m) time and O(k) output space; downsampling takes O(n) time and O(k) output
space. Both validate input indices/prefixes, omit zero individual counts and
allocate two growing vectors without a dense intermediate. Inputs coexist with
the output and any allocator-internal reallocation overlap. Output vectors may
retain spare capacity; compaction is separate.

The output mean is recomputed from output bucket midpoints. Downsampling can
change this estimate; neither operation recovers exact raw-observation means or
preserves externally supplied cached moments. Serialized representation is unchanged.

For many windows, repeated `checked_add` calls rescan the growing accumulator
and allocate each result. A balanced reduction or an explicitly owned dense
accumulator populated from `snapshot.iter()` may suit different occupancy and
window counts. Iterator buckets contain individual counts; `snapshot.count()`
contains prefixes and must not be treated as independent bucket counts. When
using a dense accumulator, check the combined total before wrapping recorder
operations or conversion. Benchmark these application choices separately from
downstream quantile reads with `cargo bench --bench cumulative_transforms`.
That benchmark uses fully overlapping windows at 8 or 2,048 occupied buckets;
partially overlapping windows can grow the output and change the tradeoff.

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
