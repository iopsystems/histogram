# u32 Bucket Counts — Design

**Date:** 2026-04-26
**Status:** Design approved (revised), pending implementation plan rewrite
**Scope:** Add `u32` as a sibling counter width across all four histogram variants via named concrete types generated from a shared macro template.

## Revision history

**v1 (initial, 2026-04-26):** Generic-with-default approach (`Histogram<C: Count = u64>`). Approved, partial implementation begun.

**v2 (current, 2026-04-26):** Switched to named-sibling concrete types (`Histogram` + `Histogram32`, etc.) generated from a shared declarative macro. Reason: Rust's type-default fallback (rust-lang/rust#27336) does not fire when a generic parameter is constrained by a sealed trait that has multiple implementors. Existing call sites like `let h = Histogram::new(7, 64).unwrap(); h.quantile(0.5);` fail to compile under v1 with `error[E0283]: type annotations needed`, breaking backwards compatibility for every existing user. The macro/named-sibling approach eliminates the type ambiguity at the source — `Histogram` stays a fully concrete type (byte-identical to today), and `Histogram32` is a separate concrete sibling.

## Motivation

The crate currently uses `u64` bucket counts everywhere. For the snapshot/delta pattern common in observability — long-running recorder sampled periodically, delta shipped for storage and analytics — `u64` is wasteful: window-scoped delta counts almost always fit in `u32`. Halving counter width halves the dominant memory and serialization cost of dense histograms (e.g., 6.4 MiB → 3.2 MiB at `grouping_power=14`).

Pattern-matching the broader observability ecosystem confirms named-sibling is the right shape:

- **HdrHistogram-Java** (the closest analogue, ~15 years of production use): `Histogram` (long), `IntCountsHistogram` (int), `ShortCountsHistogram` (short), `DoubleHistogram` (double) — separate top-level classes per counter representation. This crate is essentially a base-2 HdrHistogram, so the design lineage is direct.
- **Rust `std::sync::atomic`:** `AtomicU8` / `AtomicU16` / `AtomicU32` / `AtomicU64` — discrete named types per width, no genericity.
- **Crossbeam channels:** `bounded` / `unbounded` are separate constructors, not a generic capacity parameter.

Counter-width is a discrete operational choice (memory vs overflow ceiling), not an open type abstraction (any `T` satisfying bounds). Discrete choices warrant discrete names.

The named-sibling approach also positions the crate cleanly for future widths (e.g., `HistogramF64` for probabilistic-counting use cases — see "Counter representation table" below) as additional macro invocations rather than as a trait redesign.

## Use case (canonical pipeline)

```
AtomicHistogram                 // long-running recorder, u64 counts
    │
    │ load() / drain()
    ▼
Histogram                       // snapshot at t1, u64 counts
    │
    │ checked_sub(&snapshot_t0)
    ▼
Histogram                       // delta, u64 counts (small in practice)
    │
    │ TryFrom (narrow + cumulative)
    ▼
CumulativeROHistogram32         // shipped, stored, queried; u32 counts
```

Read-only analytics (including JS frontends for plotting) consume `CumulativeROHistogram32` and benefit from both halved size and O(log n) quantile queries via binary search. For plotting specifically, `u32` is preferred over `f32` of the same wire size: u32 is exact up to ~4.3B (vs f32 exact only to ~16M), and cumulative-monotonicity is structurally preserved (no f32-rounding plateau artifacts in ECDF rendering).

## Counter representation table

| Type | Range Limit | Relative Error |
|---|---|---|
| 8-bit unsigned integer | 2^8 − 1 | exact |
| 16-bit unsigned integer | 2^16 − 1 | exact |
| **32-bit unsigned integer** | **2^32 − 1** | **exact** |
| **64-bit unsigned integer** | **2^64 − 1** | **exact** |
| 16-bit (half) float | 65519 | 2^−10 |
| 32-bit (single) float | ≈ 2^128 | 2^−23 |
| 64-bit (double) float | ≈ 2^1024 | 2^−52 |

This PR delivers the **bold rows only** (`u32` and `u64`). The macro architecture is built so that integer-counter widths (`u8`, `u16`) and float-counter representations (`f32`, `f64`) can be added later as straight macro invocations without a redesign. Float-counter representations have meaningfully different operational semantics (saturation rather than wrapping, no atomic counterpart in `std`, relative-error trade-off) and are intentionally out of scope here; the named-sibling architecture admits them naturally if demand surfaces.

## Approach

**Named concrete sibling types generated from a shared declarative macro.** Existing types stay byte-identical to today (no source breakage); new u32 siblings live alongside.

```rust
// Existing types — unchanged, byte-identical:
pub struct Histogram { ... }                   // u64 counts
pub struct AtomicHistogram { ... }             // AtomicU64 counts
pub struct SparseHistogram { ... }             // Vec<u64> counts
pub struct CumulativeROHistogram { ... }       // Vec<u64> counts

// New u32 siblings:
pub struct Histogram32 { ... }                 // u32 counts
pub struct AtomicHistogram32 { ... }           // AtomicU32 counts
pub struct SparseHistogram32 { ... }           // Vec<u32> counts
pub struct CumulativeROHistogram32 { ... }     // Vec<u32> counts
```

A single declarative macro per variant (`define_histogram!`, `define_atomic_histogram!`, `define_sparse_histogram!`, `define_cumulative_histogram!`) takes the type name and the count primitive (and, for atomic, the atomic primitive), and emits the full type definition and impl block. Each macro is invoked twice — once for the existing type names (unchanged), once for the `*32` siblings.

The internal vocabulary the macro uses is the sealed `Count` / `AtomicCount` trait pair from Task 1. The traits remain a **purely internal** abstraction: they are public so the macro can name them in trait bounds (sealed-trait pattern requires the trait itself be reachable), but no public histogram API exposes them in user-facing signatures.

### Why this works for backwards compatibility

`Histogram::new(7, 64).unwrap().quantile(0.5)` continues to compile unchanged because `Histogram` is a fully concrete type — there is no type parameter to infer, no trait-resolution ambiguity, no E0283 error. The only effect on existing users is the addition of new optional types in the public surface; no migration required.

### Considered and rejected

- **Generics with `<C: Count = u64>` default** (v1 of this spec): broken by rust-lang/rust#27336. The default fallback does not fire when the type variable has multiple satisfying types, so `Histogram::new(...).quantile(...)` produces an E0283 error and forces every existing user to add turbofish or a binding annotation.
- **Type aliases** (e.g., `type Histogram32 = Histogram<u32>;`): same problem — alias resolution preserves the underlying generic, so the inference issue follows.
- **Hand-rolled parallel types** (no macro): would work but duplicates ~600 lines per width. The declarative macro keeps the source of truth single without paying the inference tax.

## Counter trait (internal)

The sealed `Count` and `AtomicCount` traits from Task 1 stay as-is. They are the macro's internal vocabulary. Their public surface lets users name the trait in `where` clauses if they implement their own helpers over the histogram types — useful for downstream code that needs to be width-generic — but the histogram types themselves never expose `<C: Count>` in their signatures.

```rust
mod private { pub trait Sealed {} }

pub trait Count: private::Sealed
                 + Copy + Default + Eq + Ord
                 + std::fmt::Debug + 'static {
    type Atomic: AtomicCount<Value = Self>;
    const ZERO: Self;
    const ONE: Self;
    fn wrapping_add(self, other: Self) -> Self;
    fn wrapping_sub(self, other: Self) -> Self;
    fn checked_add(self, other: Self) -> Option<Self>;
    fn checked_sub(self, other: Self) -> Option<Self>;
    fn as_u128(self) -> u128;
    fn try_from_u64(v: u64) -> Option<Self>;
}

pub trait AtomicCount: private::Sealed {
    type Value: Count<Atomic = Self>;
    fn new(v: Self::Value) -> Self;
    fn load_relaxed(&self) -> Self::Value;
    fn store_relaxed(&self, v: Self::Value);
    fn fetch_add_relaxed(&self, v: Self::Value);
    fn swap_relaxed(&self, v: Self::Value) -> Self::Value;
}

impl Count for u32 { /* ... */ }
impl Count for u64 { /* ... */ }
impl AtomicCount for AtomicU32 { /* ... */ }
impl AtomicCount for AtomicU64 { /* ... */ }
```

(Already shipped in Task 1; no further changes here.)

## Macro design

A single declarative macro per variant. Each invocation receives the type name and the count primitive (and, for atomic, the atomic primitive). The macro emits the full type definition and impl block.

Sketch (real implementation will be longer):

```rust
macro_rules! define_histogram {
    ($name:ident, $iter:ident, $count:ty) => {
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
                let buckets: Box<[$count]> = vec![<$count as Count>::ZERO; config.total_buckets()].into();
                Self { config: *config, buckets }
            }

            pub fn from_buckets(grouping_power: u8, max_value_power: u8, buckets: Vec<$count>) -> Result<Self, Error> { ... }

            pub fn increment(&mut self, value: u64) -> Result<(), Error> {
                self.add(value, <$count as Count>::ONE)
            }

            pub fn add(&mut self, value: u64, count: $count) -> Result<(), Error> { ... }

            pub fn as_slice(&self) -> &[$count] { &self.buckets }
            pub fn as_mut_slice(&mut self) -> &mut [$count] { &mut self.buckets }

            pub fn checked_add(&self, other: &Self) -> Result<Self, Error> { ... }
            pub fn wrapping_add(&self, other: &Self) -> Result<Self, Error> { ... }
            pub fn checked_sub(&self, other: &Self) -> Result<Self, Error> { ... }
            pub fn wrapping_sub(&self, other: &Self) -> Result<Self, Error> { ... }

            pub fn downsample(&self, grouping_power: u8) -> Result<Self, Error> { ... }
            pub fn iter(&self) -> $iter<'_> { $iter { index: 0, histogram: self } }
            pub fn config(&self) -> Config { self.config }

            pub fn quantiles(&self, quantiles: &[f64]) -> Result<Option<QuantilesResult>, Error> { ... }
            pub fn quantile(&self, quantile: f64) -> Result<Option<QuantilesResult>, Error> { ... }
        }

        impl SampleQuantiles for $name { ... }

        pub struct $iter<'a> { index: usize, histogram: &'a $name }
        impl Iterator for $iter<'_> { ... }
        impl ExactSizeIterator for $iter<'_> { ... }
        impl std::iter::FusedIterator for $iter<'_> {}

        impl<'a> IntoIterator for &'a $name {
            type Item = Bucket;
            type IntoIter = $iter<'a>;
            fn into_iter(self) -> Self::IntoIter { $iter { index: 0, histogram: self } }
        }
    };
}

define_histogram!(Histogram, Iter, u64);
define_histogram!(Histogram32, Iter32, u32);
```

Method bodies use `<$count as Count>::ZERO`, `<$count as Count>::ONE`, and trait methods (`x.wrapping_add(y)`) — same arithmetic primitives as v1, but now closed over a concrete `$count` per macro invocation, so no inference problem.

The iterator type name is also macro-parameterized (`Iter` for u64, `Iter32` for u32) so each histogram type has its own concrete iterator. Same for the iterator types in the other three variants.

Macros for the other three variants follow the same shape, parameterized by their type-specific needs.

## Histogram type details

### `Histogram` and `Histogram32`

Public surface (per type, identical method names across the pair):
- Constructors: `new(grouping_power, max_value_power)`, `with_config(&Config)`, `from_buckets(grouping_power, max_value_power, Vec<count_ty>)`.
- Recording: `increment(value: u64)`, `add(value: u64, count: count_ty)`.
- Direct access: `as_slice() -> &[count_ty]`, `as_mut_slice() -> &mut [count_ty]`.
- Inter-histogram arithmetic: `checked_add` / `wrapping_add` / `checked_sub` / `wrapping_sub` (each takes `&Self`, returns `Result<Self, Error>`).
- Reshape: `downsample(grouping_power) -> Result<Self, Error>`.
- Iteration: `iter() -> Iter<'_>` for `Histogram`, `iter() -> Iter32<'_>` for `Histogram32` (concrete iterator type per histogram type).
- Quantile queries: inherent `quantiles(&[f64])` / `quantile(f64)` plus `impl SampleQuantiles for $name`.
- Misc: `config() -> Config`.

`Bucket.count` stays `u64` for both. Internally, when a `Histogram32` constructs a `Bucket` from a `u32` count, it widens via `Count::as_u128() as u64`. This keeps `Bucket` and `QuantilesResult` non-generic — quantile results from u32 and u64 histograms are the same type, easing downstream consumption.

### `AtomicHistogram` and `AtomicHistogram32`

Public surface mirrors the non-atomic variants minus inter-histogram arithmetic. `load()` returns `Histogram` for the u64 variant, `Histogram32` for the u32 variant. `drain()` is gated on `target_has_atomic = "64"` (for `AtomicHistogram`) or `target_has_atomic = "32"` (for `AtomicHistogram32`); the `*32` variant is consequently available on more targets.

### `SparseHistogram` and `SparseHistogram32`

Columnar `(Vec<u32> indices, Vec<count_ty> counts)`. Constructors `from_parts`, `into_parts`, `with_config`. `SampleQuantiles` inherent + trait methods. Cross-variant `From` impls between same-width pairs (e.g., `From<&Histogram> for SparseHistogram`, `From<&Histogram32> for SparseHistogram32`) — same-width conversions are infallible.

### `CumulativeROHistogram` and `CumulativeROHistogram32`

Read-only cumulative form. Constructors `from_parts`, `into_parts`. `total_count() -> u64` (widened via `Count::as_u128`). `bucket_quantile_range`, `iter_with_quantiles`, `find_quantile_position` (binary search vs linear scan based on bucket count fitting in a cache line; the cache-line-fit threshold is `count_ty`-dependent — `64 / size_of::<count_ty>()`).

`individual_count` private helper continues to return `u64` regardless of width.

### Same-width cross-variant conversions

Generated alongside the macro invocations or in `conversions.rs`:

```rust
// u64 family — these already exist today; bodies are unchanged.
impl From<&Histogram>          for SparseHistogram          { ... }
impl From<&Histogram>          for CumulativeROHistogram    { ... }
impl From<&SparseHistogram>    for Histogram                { ... }
impl From<&SparseHistogram>    for CumulativeROHistogram    { ... }

// u32 family — new.
impl From<&Histogram32>        for SparseHistogram32        { ... }
impl From<&Histogram32>        for CumulativeROHistogram32  { ... }
impl From<&SparseHistogram32>  for Histogram32              { ... }
impl From<&SparseHistogram32>  for CumulativeROHistogram32  { ... }
```

Eight impls total (four per family). The u64 family stays exactly as today — no body changes.

## Cross-width and cross-variant conversions

All cross-width and combined conversions live in `src/conversions.rs` for auditability.

### Cross-width same-variant widening (`From`, infallible)

```rust
impl From<&Histogram32>              for Histogram               { ... }
impl From<&AtomicHistogram32>        for AtomicHistogram         { ... }
impl From<&SparseHistogram32>        for SparseHistogram         { ... }
impl From<&CumulativeROHistogram32>  for CumulativeROHistogram   { ... }
```

Each maps `u32 → u64` casts over the count slice/`Vec`; config and indices copy verbatim.

### Cross-width same-variant narrowing (`TryFrom`, fallible)

```rust
impl TryFrom<&Histogram>               for Histogram32              { /* checks every bucket */ }
impl TryFrom<&SparseHistogram>         for SparseHistogram32        { /* checks every non-zero bucket */ }
impl TryFrom<&CumulativeROHistogram>   for CumulativeROHistogram32  { /* checks last cumulative count only */ }
```

For `CumulativeROHistogram`, only the final cumulative value (the total count) needs checking. Any total ≤ `u32::MAX` implies every individual bucket ≤ `u32::MAX`. Strictly cheaper than per-bucket checking.

`AtomicHistogram` narrowing is intentionally not offered. The natural pipeline is `atomic.load() / drain() → Histogram → TryFrom → Histogram32`.

### Cross-variant + narrowing combined (`TryFrom`)

Direct paths supporting the snapshot pipeline:

```rust
impl TryFrom<&Histogram>        for CumulativeROHistogram32  { ... }
//   single pass: accumulate non-zero buckets, check final total ≤ u32::MAX

impl TryFrom<&Histogram>        for SparseHistogram32        { ... }
//   single pass: copy non-zero buckets, per-bucket check

impl TryFrom<&SparseHistogram>  for CumulativeROHistogram32  { ... }
//   single pass: cumulative running sum, check final total
```

All return `Err(Error::Overflow)` on failure.

Cross-variant + widening combined paths are intentionally **not** added (e.g., `From<&Histogram32> for CumulativeROHistogram`). The two-step `Histogram32 → Histogram → CumulativeROHistogram` is fine since both steps are infallible and cheap.

## Overflow semantics

`*32` types follow the same overflow semantics as the existing types:

- **Single-bucket increment / `add`:** silent `wrapping_add`. Symmetric across widths. Callers who pick `*32` are explicitly making a memory/range tradeoff and own the consequence.
- **Inter-histogram operations:** `checked_add` / `checked_sub` return `Err(Overflow)` / `Err(Underflow)`; `wrapping_add` / `wrapping_sub` wrap silently.
- **Narrowing conversions (`TryFrom`):** return `Err(Error::Overflow)` if any bucket (or, for cumulative, the total) exceeds `u32::MAX`. No silent saturating cast.

## Documentation updates

### `README.md`

Two new sections inserted between "Histogram Types" and "Features":

1. **Counter Width.** Describes the `*32` sibling family, when to use it, and the `From` / `TryFrom` conversion API.
2. **Recommended Pipeline.** The canonical snapshot/delta pattern with sample code (see "Recommended Pipeline guidance" below).

The existing "Histogram Types" section gains four new bullets describing the `*32` siblings.

### `src/lib.rs` module rustdoc

Mirror the same two sections in the crate-level `//!` rustdoc so they appear on docs.rs. Add a "Types" subsection with all eight types listed.

### `Config` rustdoc

The existing memory table shows `u64`-only sizes. Add a footnote: "Halve all sizes for `*32` histograms (`Histogram32`, `AtomicHistogram32`, `SparseHistogram32`, `CumulativeROHistogram32`)."

### Per-type rustdoc

Each `*32` type gets a one-paragraph rustdoc covering: the counter width, the memory tradeoff, the overflow ceiling, and a pointer to the cross-width conversion API. Each existing `u64` type gets a one-line addition mentioning the `*32` sibling.

### Recommended Pipeline guidance (verbatim)

> **Counter width and the recommended snapshot pipeline**
>
> Pick the histogram type based on the *role* it plays in your data flow:
>
> - **Recording — `AtomicHistogram` (or `Histogram`).** Use `u64`-counter types for the long-running, continuously-updated histogram. Counts here are unbounded over the lifetime of the process; `u64` heads off any practical risk of overflow.
>
> - **Snapshot delta — `Histogram`, then narrowed.** When you take periodic snapshots and compute a delta with `checked_sub`, the delta covers only the activity in one window. Window counts are typically much smaller than lifetime counts, which is exactly when narrowing pays off. Use `Histogram::checked_sub` to compute the delta, then `TryFrom` to narrow into a `*32` type.
>
> - **Read-only analytics — `CumulativeROHistogram32`.** This is the recommended storage and query format for completed snapshots. The cumulative-prefix-sum representation gives you O(log n) quantile queries via binary search, while `u32` counts halve the on-the-wire and on-disk size versus `u64`. Narrowing is checked once against the *total count* (cheaper than per-bucket), and any total ≤ ~4.3B fits.
>
> ```rust
> use histogram::{AtomicHistogram, CumulativeROHistogram32, Histogram};
>
> // Recording: u64, atomic, long-lived
> let recorder = AtomicHistogram::new(7, 64)?;
>
> // Snapshot pipeline (run periodically)
> let snap_t1 = recorder.load();                            // Histogram
> let delta = snap_t1.checked_sub(&snap_t0)?;               // Histogram — small counts
> let analytic: CumulativeROHistogram32 =
>     CumulativeROHistogram32::try_from(&delta)?;            // narrow + cumulative in one pass
> // analytic is now ready to ship/store/query
> ```
>
> If you don't take snapshots — i.e., you query the recording histogram directly — just stay on the `u64` types everywhere. The narrowing optimization is specifically for the snapshot/delta pattern.
>
> For JavaScript-frontend plotting specifically, prefer `CumulativeROHistogram32` over a hypothetical f32-backed alternative: `u32` is exact up to ~4.3B (vs f32 exact only to ~16M), and cumulative-monotonicity is structurally preserved (no rounding-induced plateau artifacts in ECDF rendering).

## Testing

### Existing tests

Continue to exercise the `u64`-counter types unchanged. No call-site modifications required because `Histogram`, `AtomicHistogram`, etc. are byte-identical to today.

### New tests

For each `*32` variant, add tests parallel to the existing tests where width-relevant behavior is exercised. The macro guarantees behavioral parity, so we just need spot-coverage rather than full duplication.

Specific new tests:

- **Wrapping arithmetic at `u32::MAX`** for `Histogram32::add` and inter-histogram `wrapping_add` / `wrapping_sub`.
- **Checked arithmetic at `u32::MAX`** for inter-histogram `checked_add` / `checked_sub`.
- **Round-trip widening:** `Histogram32 → Histogram → TryFrom → Histogram32` equals original.
- **Narrowing failure:** build a `Histogram` with a bucket exceeding `u32::MAX`, assert `TryFrom<&Histogram> for Histogram32` returns `Err(Overflow)`. Equivalent for sparse and cumulative.
- **Cumulative narrowing total-check semantics:** build a `CumulativeROHistogram` where individual buckets fit `u32` but the cumulative total exceeds `u32::MAX`, assert it fails. (Validates the total-only check.)
- **Direct cross-variant + narrow:** `Histogram → CumulativeROHistogram32` direct path matches the two-step path.
- **`size_of` assertions** for `Histogram32`, `AtomicHistogram32`, etc.
- **`target_has_atomic = "32"` cfg gate** on the `AtomicHistogram32::drain` test.

### Benchmarks

Extend the existing macro-driven bench in `benches/histogram.rs` to add `Histogram32` and `AtomicHistogram32` cases alongside the existing `Histogram` / `AtomicHistogram` cases. Confirms no regression on the existing path and gives baseline numbers for the new types.

## Migration / breaking-change analysis

**Zero source breakage.** All four existing types (`Histogram`, `AtomicHistogram`, `SparseHistogram`, `CumulativeROHistogram`) stay byte-identical to today. Existing user code compiles unchanged. No type annotations, no turbofish, no migration guide needed.

The public surface gains:
- 4 new types (`Histogram32`, `AtomicHistogram32`, `SparseHistogram32`, `CumulativeROHistogram32`).
- Cross-width `From` / `TryFrom` impls.
- Sealed `Count` and `AtomicCount` traits exposed in re-exports (used internally by the macro; public so users can name them in `where` clauses if they want to write width-generic helpers).

**Versioning:** per `CLAUDE.md`, the implementation PR uses an alpha revision (`<next-version>-alpha.<n>`) bumping the alpha revision relative to what's on `main`. The final release version is set by a separate release PR.

**Deprecation:** nothing existing is deprecated. The already-deprecated `percentile` / `percentiles` methods on the existing types stay deprecated; the `*32` types do *not* gain those deprecated methods (they're new types and we don't bring forward the old API surface).

**Feature flags:** `serde` / `schemars` derives are emitted by the macro per type, so all eight types get them when the feature is enabled. No feature-flag changes needed.

## Out of scope

- `u8` / `u16` / `f32` / `f64` counter widths. The macro architecture admits them as future invocations; not delivered here.
- A `saturating_increment` / `saturating_add` API. Wrapping-symmetric is the chosen default; can be added non-breakingly later if demand surfaces.
- Changes to `Bucket` or `QuantilesResult` shape — both stay non-generic with `u64` count fields.
- Cross-variant + widening combined `From` impls (covered above).

## Dependency graph (implementation order)

1. **Count trait** — `src/count.rs`. Already shipped (Task 1, commits `bba6635` + `26776af`).
2. **Histogram macro** — `src/standard.rs` rewritten as a macro emission, invoked twice (`Histogram`, `Histogram32`).
3. **AtomicHistogram macro** — `src/atomic.rs`, invoked twice.
4. **SparseHistogram macro** — `src/sparse.rs`, invoked twice. Includes same-width cross-variant `From` impls.
5. **CumulativeROHistogram macro** — `src/cumulative.rs`, invoked twice. Includes same-width cross-variant `From` impls.
6. **Cross-width `From` (widening) impls** — `src/conversions.rs`.
7. **Cross-width `TryFrom` (narrowing) impls + cross-variant + narrowing combined impls** — `src/conversions.rs`.
8. **Bench updates** — `benches/histogram.rs`.
9. **Documentation** — `README.md`, `src/lib.rs`, per-type rustdoc, `src/config.rs` footnote, `Cargo.toml` version bump.
