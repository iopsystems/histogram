# u32 Bucket Counts — Design

**Date:** 2026-04-26
**Status:** Design approved, pending implementation plan
**Scope:** Add `u32` as a first-class counter width across all four histogram variants.

## Motivation

The crate currently uses `u64` bucket counts everywhere. For the snapshot/delta pattern common in observability — where a long-running recorder is sampled periodically and the delta between samples is shipped for storage and analytics — `u64` is wasteful: window-scoped delta counts almost always fit in a `u32`. Halving counter width halves the dominant memory and serialization cost of dense histograms (e.g., 6.4 MiB → 3.2 MiB at `grouping_power=14`).

Symmetric `u32` support is added across all four variants (`Histogram`, `AtomicHistogram`, `SparseHistogram`, `CumulativeROHistogram`) so counter width is a first-class library parameter rather than a bolted-on variant.

## Use case (canonical pipeline)

```
AtomicHistogram<u64>            // long-running recorder
    │
    │ load() / drain()
    ▼
Histogram<u64>                  // snapshot at t1
    │
    │ checked_sub(&snapshot_t0)
    ▼
Histogram<u64>                  // delta (small counts)
    │
    │ TryFrom (narrow + cumulative)
    ▼
CumulativeROHistogram<u32>      // shipped, stored, queried
```

The narrowing happens once per snapshot window. Read-only analytics consume `CumulativeROHistogram<u32>` and benefit from both halved size and O(log n) quantile queries via binary search.

## Approach

**Generics with a defaulted type parameter.** Each existing type becomes generic over a counter type `C: Count`, with `C` defaulted to `u64`:

```rust
pub struct Histogram<C: Count = u64> { ... }
pub struct AtomicHistogram<C: Count = u64> { ... }
pub struct SparseHistogram<C: Count = u64> { ... }
pub struct CumulativeROHistogram<C: Count = u64> { ... }
```

The default keeps existing call sites (`Histogram::new(7, 64)`) compiling unchanged with no source modification — they continue to infer `Histogram<u64>`. Users who want narrower counters opt in via `Histogram::<u32>::new(...)` or a binding annotation.

Generics are zero-overhead at runtime: monomorphization emits a separate, fully-specialized copy per concrete instantiation. No vtable, no boxing, no dynamic dispatch.

Considered and rejected:
- **Parallel `Histogram32` types** (manual or macro-generated): adds four new type names per width, fragments the conversion matrix, more docs surface for no functional gain over generics.
- **Open trait** (allow downstream `impl Count for u16`): committed-to-stability surface we don't need today; sealing leaves the door open for `u8`/`u16` in-crate without it being a breaking change.

## Counter trait

A new `count.rs` module exposes a sealed `Count` trait (and a paired `AtomicCount`) abstracting over `u32` and `u64`.

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
    type Value: Count;
    fn new(v: Self::Value) -> Self;
    fn load_relaxed(&self) -> Self::Value;
    fn store_relaxed(&self, v: Self::Value);
    fn fetch_add_relaxed(&self, v: Self::Value);
    fn swap_relaxed(&self, v: Self::Value) -> Self::Value;
}

impl private::Sealed for u32 {}
impl private::Sealed for u64 {}
impl private::Sealed for std::sync::atomic::AtomicU32 {}
impl private::Sealed for std::sync::atomic::AtomicU64 {}

impl Count for u32 { type Atomic = AtomicU32; /* ... */ }
impl Count for u64 { type Atomic = AtomicU64; /* ... */ }
impl AtomicCount for AtomicU32 { type Value = u32; /* ... */ }
impl AtomicCount for AtomicU64 { type Value = u64; /* ... */ }
```

The `Sealed` super-trait pattern restricts `Count` and `AtomicCount` to types defined in this crate. The `Atomic` associated type is the only clean way to map `Count → atomic primitive` for `AtomicHistogram<C>` without adding a second generic parameter.

`as_u128` is the upcast hook used by quantile computation for partial sums (preserves the existing `total_count: u128` contract). `try_from_u64` is the downcast hook used in the narrowing path.

## Histogram type changes

All four variants gain `<C: Count = u64>`. Method bodies are structurally unchanged; `u64` literals in arithmetic become trait calls.

### `Histogram<C>`

```rust
pub struct Histogram<C: Count = u64> {
    pub(crate) config: Config,
    pub(crate) buckets: Box<[C]>,
}

impl<C: Count> Histogram<C> {
    pub fn new(grouping_power: u8, max_value_power: u8) -> Result<Self, Error> { ... }
    pub fn with_config(config: &Config) -> Self { ... }
    pub fn from_buckets(grouping_power: u8, max_value_power: u8, buckets: Vec<C>) -> Result<Self, Error> { ... }

    pub fn increment(&mut self, value: u64) -> Result<(), Error> {
        self.add(value, C::ONE)
    }

    pub fn add(&mut self, value: u64, count: C) -> Result<(), Error> {
        let index = self.config.value_to_index(value)?;
        self.buckets[index] = self.buckets[index].wrapping_add(count);
        Ok(())
    }

    pub fn checked_add(&self, other: &Histogram<C>) -> Result<Histogram<C>, Error> { ... }
    pub fn wrapping_add(&self, other: &Histogram<C>) -> Result<Histogram<C>, Error> { ... }
    pub fn checked_sub(&self, other: &Histogram<C>) -> Result<Histogram<C>, Error> { ... }
    pub fn wrapping_sub(&self, other: &Histogram<C>) -> Result<Histogram<C>, Error> { ... }
    pub fn downsample(&self, grouping_power: u8) -> Result<Histogram<C>, Error> { ... }
    pub fn as_slice(&self) -> &[C] { ... }
    pub fn as_mut_slice(&mut self) -> &mut [C] { ... }
}
```

### `AtomicHistogram<C>`

```rust
pub struct AtomicHistogram<C: Count = u64> {
    config: Config,
    buckets: Box<[C::Atomic]>,
}

impl<C: Count> AtomicHistogram<C> {
    pub fn increment(&self, value: u64) -> Result<(), Error> {
        self.add(value, C::ONE)
    }
    pub fn add(&self, value: u64, count: C) -> Result<(), Error> {
        let index = self.config.value_to_index(value)?;
        self.buckets[index].fetch_add_relaxed(count);
        Ok(())
    }
    pub fn load(&self) -> Histogram<C> { ... }
    #[cfg(target_has_atomic = "32")]   // for C = u32
    #[cfg(target_has_atomic = "64")]   // for C = u64
    pub fn drain(&self) -> Histogram<C> { ... }
}
```

The `target_has_atomic` cfg gate becomes per-instantiation. Rust does not support `cfg` on impl blocks predicated on a generic type parameter, so the implementation uses two concretely-typed impl blocks for `drain`:

```rust
#[cfg(target_has_atomic = "64")]
impl AtomicHistogram<u64> { pub fn drain(&self) -> Histogram<u64> { ... } }

#[cfg(target_has_atomic = "32")]
impl AtomicHistogram<u32> { pub fn drain(&self) -> Histogram<u32> { ... } }
```

`AtomicHistogram<u32>::drain` is available on more targets than `AtomicHistogram<u64>::drain` because `AtomicU32` is more widely supported.

### `SparseHistogram<C>` and `CumulativeROHistogram<C>`

```rust
pub struct SparseHistogram<C: Count = u64> {
    pub(crate) config: Config,
    pub(crate) index: Vec<u32>,
    pub(crate) count: Vec<C>,
}

pub struct CumulativeROHistogram<C: Count = u64> {
    config: Config,
    index: Vec<u32>,
    count: Vec<C>,
}
```

All existing methods generalize over `C`. `from_parts`, `into_parts`, `count()`, etc. take/return `Vec<C>` / `&[C]`.

### `Bucket` and `QuantilesResult` stay non-generic

`Bucket.count` remains `u64`. Internally we widen via `C::as_u128() as u64` (always safe for `u32 → u64`). This keeps `QuantilesResult` non-generic — quantile results from `Histogram<u32>` and `Histogram<u64>` are the same type, simplifying downstream consumers that work with quantile output.

### `SampleQuantiles` impl

```rust
impl<C: Count> SampleQuantiles for Histogram<C> { ... }
impl<C: Count> SampleQuantiles for SparseHistogram<C> { ... }
impl<C: Count> SampleQuantiles for CumulativeROHistogram<C> { ... }
```

Internally the partial-sum loop uses `C::as_u128()`, preserving the `total_count: u128` field on `QuantilesResult`. No change to the trait signature.

## Cross-width and cross-variant conversions

### Same-width cross-variant `From` (existing, generalized)

```rust
impl<C: Count> From<&Histogram<C>> for SparseHistogram<C> { ... }
impl<C: Count> From<&Histogram<C>> for CumulativeROHistogram<C> { ... }
impl<C: Count> From<&SparseHistogram<C>> for Histogram<C> { ... }
impl<C: Count> From<&SparseHistogram<C>> for CumulativeROHistogram<C> { ... }
```

### Same-variant cross-width widening (`From`, infallible)

```rust
impl From<&Histogram<u32>> for Histogram<u64> { ... }
impl From<&AtomicHistogram<u32>> for AtomicHistogram<u64> { ... }
impl From<&SparseHistogram<u32>> for SparseHistogram<u64> { ... }
impl From<&CumulativeROHistogram<u32>> for CumulativeROHistogram<u64> { ... }
```

Each maps `u32 → u64` casts over the count slice/`Vec`; config and indices copy verbatim.

### Same-variant cross-width narrowing (`TryFrom`, fallible)

```rust
impl TryFrom<&Histogram<u64>> for Histogram<u32> { /* checks every bucket */ }
impl TryFrom<&SparseHistogram<u64>> for SparseHistogram<u32> { /* checks every non-zero bucket */ }
impl TryFrom<&CumulativeROHistogram<u64>> for CumulativeROHistogram<u32> { /* checks last cumulative count only */ }
```

For `CumulativeROHistogram`, only the final cumulative value (the total count) needs checking — any total ≤ `u32::MAX` implies every individual bucket ≤ `u32::MAX`. This is strictly cheaper than per-bucket checking.

`AtomicHistogram<u64>` narrowing is intentionally not offered: the natural pipeline goes `atomic.load()` / `drain() → Histogram<u64> → TryFrom → Histogram<u32>`, which avoids a fresh atomic-allocation snapshot.

### Cross-variant + narrowing combined (`TryFrom`)

Direct paths supporting the snapshot pipeline:

```rust
impl TryFrom<&Histogram<u64>> for CumulativeROHistogram<u32> { ... }
//   single pass: accumulate non-zero buckets, check final total ≤ u32::MAX

impl TryFrom<&Histogram<u64>> for SparseHistogram<u32> { ... }
//   single pass: copy non-zero buckets, per-bucket check

impl TryFrom<&SparseHistogram<u64>> for CumulativeROHistogram<u32> { ... }
//   single pass: cumulative running sum, check final total
```

All return `Err(Error::Overflow)` on failure.

Cross-variant + widening combined paths are intentionally **not** added (e.g., `From<&Histogram<u32>> for CumulativeROHistogram<u64>`). The two-step `Histogram<u32> → Histogram<u64> → CumulativeROHistogram<u64>` is fine since both steps are infallible and cheap.

## Overflow semantics

`u32` instantiations follow the same overflow semantics as `u64`:

- **Single-bucket increment / `add`:** silent `wrapping_add`. Symmetric with `u64`. Callers who pick `u32` are explicitly making a memory/range tradeoff and own the consequence.
- **Inter-histogram operations:** `checked_add` / `checked_sub` return `Err(Overflow)` / `Err(Underflow)`; `wrapping_add` / `wrapping_sub` wrap silently. Identical to `u64` semantics today.
- **Narrowing conversions:** `TryFrom` returns `Err(Overflow)` if any bucket (or, for cumulative, the total) exceeds `u32::MAX`. No silent saturating cast — distorts quantiles invisibly.

## Documentation updates

### `README.md`

Two new sections inserted between "Histogram Types" and "Features":

1. **Counter Width.** Describes the `<C>` parameter, default `u64`, available `u32` opt-in, and the conversion API. Updated example shows `Histogram::<u32>::new(...)` form briefly.

2. **Recommended Pipeline.** The canonical snapshot/delta pattern with full sample code (see "Recommended Pipeline guidance" below).

### `src/lib.rs` module rustdoc

Mirror the same two sections in the crate-level rustdoc so they appear on docs.rs. Update the `# Types` list: each entry mentions the `<C>` parameter and `u64` default. The existing usage example stays unchanged — it relies on the default.

### `Config` rustdoc

The existing memory table shows `u64`-only sizes. Add a footnote: "Halve all sizes for histograms instantiated with `<u32>` counters."

### Per-type rustdoc

`Histogram<C>`, `AtomicHistogram<C>`, `SparseHistogram<C>`, `CumulativeROHistogram<C>` each gain a one-line note: "Generic over counter width `C`. Defaults to `u64`. See crate-level docs for guidance."

`Count` and `AtomicCount` traits get full module rustdoc explaining sealed-ness and supported widths (`u32`, `u64`).

### Recommended Pipeline guidance (verbatim)

> **Counter width and the recommended snapshot pipeline**
>
> Pick the counter width based on the *role* the histogram plays in your data flow:
>
> - **Recording — `AtomicHistogram<u64>` (or `Histogram<u64>`).** Use `u64` for the long-running, continuously-updated histogram. Counts here are unbounded over the lifetime of the process; `u64` heads off any practical risk of overflow.
>
> - **Snapshot delta — `Histogram<u64>`, then narrowed.** When you take periodic snapshots and compute a delta with `checked_sub`, the delta covers only the activity in one window. Window counts are typically much smaller than lifetime counts, which is exactly when narrowing pays off. Use `Histogram<u64>::checked_sub` to compute the delta, then `TryFrom` to narrow.
>
> - **Read-only analytics — `CumulativeROHistogram<u32>`.** This is the recommended storage and query format for completed snapshots. The cumulative-prefix-sum representation gives you O(log n) quantile queries via binary search, while `u32` counts halve the on-the-wire and on-disk size versus `u64`. Narrowing is checked once against the *total count* (cheaper than per-bucket), and any total ≤ ~4.3B fits.
>
> ```rust
> use histogram::{AtomicHistogram, CumulativeROHistogram, Histogram};
>
> // Recording: u64, atomic, long-lived
> let recorder = AtomicHistogram::<u64>::new(7, 64)?;
>
> // Snapshot pipeline (run periodically)
> let snap_t1 = recorder.load();                              // Histogram<u64>
> let delta = snap_t1.checked_sub(&snap_t0)?;                 // Histogram<u64> — small counts
> let analytic: CumulativeROHistogram<u32> =
>     CumulativeROHistogram::<u32>::try_from(&delta)?;        // narrow + cumulative in one pass
> // analytic is now ready to ship/store/query
> ```
>
> If you don't take snapshots — i.e., you query the recording histogram directly — just stay on `u64` everywhere. The narrowing optimization is specifically for the snapshot/delta pattern.

## Testing

### Existing tests

Continue to exercise the `u64` path unchanged via the `C = u64` default. No call-site modifications required.

### New tests (parallel to existing for `<u32>`)

For each variant, add a `<u32>` instantiation of every existing test. A generic test helper macro is acceptable to avoid duplication.

Specific new tests:

- **Wrapping arithmetic at `u32::MAX`** for `Histogram<u32>::add` and inter-histogram `wrapping_add` / `wrapping_sub`.
- **Checked arithmetic at `u32::MAX`** for inter-histogram `checked_add` (returns `Err(Overflow)`) and `checked_sub` (returns `Err(Underflow)`).
- **Round-trip widening:** `Histogram<u32> → Histogram<u64> → TryFrom → Histogram<u32>` equals original.
- **Narrowing failure:** build a `Histogram<u64>` with a bucket exceeding `u32::MAX`, assert `TryFrom` returns `Err(Overflow)`. Equivalent for sparse and cumulative.
- **Cumulative narrowing total-check semantics:** build a `CumulativeROHistogram<u64>` where individual buckets fit `u32` but the cumulative total exceeds `u32::MAX`, assert it fails. (Documents and validates that cumulative checks the total.)
- **Direct cross-variant + narrow:** `Histogram<u64> → CumulativeROHistogram<u32>` round-trip equals two-step (`Histogram<u64> → CumulativeROHistogram<u64> → CumulativeROHistogram<u32>`) path.
- **`size_of` assertions** for `Histogram<u32>`, `AtomicHistogram<u32>`, etc., paralleling the existing `size_of::<Histogram>() == 48` assertion.
- **`target_has_atomic = "32"` cfg gate** on the `AtomicHistogram<u32>::drain` test.

### Benchmarks

Extend the existing macro-driven bench in `benches/histogram.rs` to add `Histogram<u32>` and `AtomicHistogram<u32>` instantiations alongside the existing `<u64>` cases. Confirms no regression on the `u64` path and gives baseline numbers for `u32`.

## Migration / breaking-change analysis

- **Public API shape:** `Histogram` and friends gain a defaulted type parameter `C: Count = u64`. Type inference for `let h = Histogram::new(7, 64).unwrap()` continues to resolve `Histogram<u64>`. No source breakage expected for normal call sites.
- **Soft-breaking case:** downstream code with `impl SomeTrait for Histogram` would need to specify `Histogram<u64>` (or generalize over `C`). Should be flagged in CHANGELOG.
- **Versioning:** per `CLAUDE.md`, the implementation PR uses an alpha revision (`<next-version>-alpha.<n>`) bumping the alpha revision relative to what is on `main`. The final release version is set by a separate release PR.
- **Deprecation:** nothing existing is deprecated by this work. The already-deprecated `percentile`/`percentiles` methods stay deprecated as-is.
- **Feature flags:** `serde` and `schemars` feature derives become bounded (`where C: Serialize + Deserialize<'de>` / `where C: JsonSchema`). Standard pattern; no feature-flag changes.

## Out of scope

Explicitly not included in this work:

- `u8` and `u16` counter widths. Sealed trait makes adding them in-crate non-breaking later if demand appears.
- Cross-variant + widening combined `From` impls (covered above).
- A `saturating_increment` / `saturating_add` API. The wrapping-symmetric design is the agreed default; if user demand surfaces, this can be added non-breakingly in a follow-up.
- Changes to `Bucket` or `QuantilesResult` shape — both stay non-generic with `u64` count fields.

## Dependency graph (implementation order)

1. `count.rs` module — `Count` and `AtomicCount` sealed traits, impls for `u32`/`u64`/`AtomicU32`/`AtomicU64`.
2. `Histogram<C>` — generalize, plus all method impls and tests.
3. `AtomicHistogram<C>` — generalize, plus per-instantiation `target_has_atomic` cfg.
4. `SparseHistogram<C>` — generalize.
5. `CumulativeROHistogram<C>` — generalize.
6. Cross-width `From` / `TryFrom` impls (per-variant).
7. Cross-variant + narrowing `TryFrom` impls.
8. Bench updates.
9. Documentation updates (README, lib.rs, per-type rustdoc).
