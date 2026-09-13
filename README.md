# histogram

A collection of histogram data structures for Rust, providing standard, atomic,
and sparse variants. Like HDRHistogram, values are stored in quantized buckets,
but the bucket construction and indexing algorithm are modified for fast
increments and lookups.

## Getting Started

```sh
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
let report = histogram.quantiles(&[0.5, 0.99]).unwrap().unwrap();
// quantiles() shares scan work and returns Result<Option<QuantilesResult>, Error>
// outer unwrap: quantile value is valid
// inner unwrap: histogram is non-empty

let median = report.get(&Quantile::new(0.5).unwrap()).unwrap();
let p99 = report.get(&Quantile::new(0.99).unwrap()).unwrap();

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

## Choosing a representation

| Phase | Starting point | Cost to consider |
| --- | --- | --- |
| Update | `Histogram` owned by one writer, or `AtomicHistogram` when writers must share | Update rate, recorder count, counter storage, and contention |
| Report | Batch quantiles on the dense recorder or a completed snapshot | Count scanning, publication/handoff, reset, and aggregation |
| Analytics | Convert a completed window to `CumulativeROHistogram` for repeated queries; use sparse storage when appropriate | Conversion cost, subsequent read count, retained capacity, and transforms |

Thread ownership is an application decision. Per-writer dense recorders require
handoff and aggregation; shared atomic recorders incur atomic operations and may
contend. Writers and readers can live in different services. Choose based on the
whole workflow, not the fastest isolated query.

Dense recording maintains bucket counts without cached totals or bounds. Reporting
derives that metadata when requested, keeping the recording path small. Use one
batch for a report's quantiles instead of separate scans. A cumulative snapshot
pays construction once for binary-search reads; it need not pay off for one report.

## Counter Width

Both counter families have the same value range and bucket precision. The limit
that matters depends on the representation:

| Representation | u32 limit | u64 limit |
| --- | --- | --- |
| Dense, atomic, sparse individual counts | 2^32 − 1 per bucket | 2^64 − 1 per bucket |
| Cumulative prefix counts | 2^32 − 1 observations in total | 2^64 − 1 observations in total |

Recording uses wrapping arithmetic. Choose a width that fits the activity between
resets, or the lifetime count when retaining cumulative metrics. u64 provides more
headroom; a bounded reporting interval may safely use u32 even for direct queries.
Dense queries widen sums to u128, so their total can exceed a single counter's limit.

u32 halves the **counter array** size. It does not halve sparse/cumulative indices,
object overhead, spare capacity, or necessarily serialized size. Widening u32 to
u64 is infallible; checked narrowing uses `TryFrom` and returns `Error::Overflow`
when the destination count limit is exceeded. Ensure the total fits when converting
individual counts into cumulative prefixes, even at the same width.

For completed windows, checked conversion from a u64 dense histogram to
`CumulativeROHistogram32` is useful when the total fits u32. Construction and
retained storage remain separate costs from querying. Merge/downsample operations
can create subsequent analytical summaries without modifying the inputs.

## Reusing reporting storage

Dense recorders offer `reset()` to clear counts without reallocating and
`checked_add_assign()` to aggregate compatible workers/windows in place. Checked
aggregation validates every addition before changing the destination; overflow or
incompatible geometry leaves it unchanged. This requires two bucket passes.

Atomic recorders offer `load_into()` and `drain_into()` to overwrite an existing,
compatible dense histogram without allocating. Both overwrite zero buckets too;
`drain_into()` also clears each source bucket as it captures it. Configuration
mismatch leaves both source and destination unchanged.

```rust
use histogram::{AtomicHistogram, Bucket, Histogram};

let recorder = AtomicHistogram::new(7, 32).unwrap();
let mut window = Histogram::with_config(&recorder.config());
let mut combined = Histogram::with_config(&recorder.config());
let mut output: [Option<Bucket>; 2] = std::array::from_fn(|_| None);

recorder.increment(100).unwrap();
recorder.drain_into(&mut window).unwrap();
window.quantile_buckets_into(&[0.5, 0.99], &mut output).unwrap();
combined.checked_add_assign(&window).unwrap();
// Reuse window as the destination for the next drain after consumers finish.
```

Atomic loads and drains visit buckets individually; they do not establish one
instantaneous histogram-wide reporting boundary or publish unrelated application
data. Coordinate writers if an exact boundary is required. A separate snapshot
followed by clearing while writers continue can lose observations. An owned dense
recorder can instead be handed off or rotated, then reset when consumers finish.

Lifetime cumulative metrics need not reset. Compatible bucket snapshots can be
subtracted to obtain interval counts if the recorder has not reset or overflowed;
subtracting percentile values does not produce an interval percentile.

## Allocation-free bucket queries

Dense histograms and owned/borrowed cumulative snapshots provide
`quantile_bucket()` and `quantile_buckets_into()` for both counter widths.
Dense batches share a forward rank scan, with O(B + Q²) work for B buckets and
Q requests; the allocation-free request ordering is intended for small reports.
Cumulative snapshots use binary searches, with O(Q log K) work for K stored buckets.
Construction/conversion is a separate cost.

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
