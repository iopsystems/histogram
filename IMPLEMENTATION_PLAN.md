# Implementation plan: borrowed-view types for percentile queries

## Goal

Let columnar consumers (e.g. metriken-query) compute percentiles directly off
`&[u32]` slices without per-snapshot `Vec` allocation or revalidation. For a
1k-snapshot × 4-percentile query this drops ~2k transient allocations.

Add borrowed-view siblings to the existing read-only types:

- `CumulativeROHistogramRef<'a>` / `CumulativeROHistogram32Ref<'a>`
- `SparseHistogramRef<'a>` / `SparseHistogram32Ref<'a>`

Each mirrors the API surface of its owned counterpart so consumers can use
either interchangeably.

## Files to change

- `src/cumulative.rs` — add ref types via the existing macro
- `src/sparse.rs` — add ref types via the existing macro
- `src/lib.rs` — re-export the new types
- `Cargo.toml` — bump version to `1.3.1-alpha.0` (per CLAUDE.md, feature PRs use
  `<version>-alpha.<revision>` relative to latest on `main`, which is `1.3.0`)
- `CHANGELOG.md` — add entries under `[Unreleased]`

## Design

### Struct layout

```rust
#[derive(Clone, Copy, Debug)]
pub struct CumulativeROHistogram32Ref<'a> {
    config: Config,
    index: &'a [u32],
    count: &'a [u32],
}
```

`Config` is already `Copy` (`src/config.rs:61`), so `Ref` is `Copy` — passing it
around is cheap and the borrow checker stays happy.

Same shape for the `u64` variant and for the two sparse variants (sparse uses
non-cumulative counts but is otherwise identical).

### Constructors

Two constructors per ref type:

```rust
/// Validates length match, ascending indices, in-range indices,
/// non-decreasing counts, no zero counts. Mirrors `from_parts` on the
/// owned type. Use this at trust boundaries.
pub fn from_parts(
    config: Config,
    index: &'a [u32],
    count: &'a [u32],
) -> Result<Self, Error> { ... }

/// Skips validation. Caller is responsible for upholding the invariants.
/// Use this in hot paths where the data already came from a validated
/// source (e.g. snapshots produced by this crate).
pub fn from_parts_unchecked(
    config: Config,
    index: &'a [u32],
    count: &'a [u32],
) -> Self { ... }
```

The `_unchecked` variant is what realizes the optimization — it skips the O(n)
validation pass that `CumulativeROHistogram32::from_parts` does today
(`src/cumulative.rs:48-89`).

To DRY the validation across owned and borrowed forms, factor it out:

```rust
impl<'a> CumulativeROHistogram32Ref<'a> {
    fn validate(
        config: &Config,
        index: &[u32],
        count: &[u32],
    ) -> Result<(), Error> {
        // body lifted from existing $name::from_parts
    }
}
```

Then `CumulativeROHistogram32::from_parts` calls
`CumulativeROHistogram32Ref::validate(&config, &index, &count)?` before
constructing the owned struct.

### Method surface

Mirror the existing read-only API on each ref type:

**Cumulative ref** (matches `src/cumulative.rs:96-217`):
- `config(&self) -> Config`
- `index(&self) -> &'a [u32]`
- `count(&self) -> &'a [u32]`
- `len(&self) -> usize`
- `is_empty(&self) -> bool`
- `total_count(&self) -> u64`
- `bucket_quantile_range(&self, bucket_idx: usize) -> Option<(f64, f64)>`
- `iter(&self) -> CumulativeIter<'a>`
- `iter_with_quantiles(&self) -> QuantileRangeIter<'a>`
- `quantiles(&self, &[f64]) -> Result<Option<QuantilesResult>, Error>`
- `quantile(&self, f64) -> Result<Option<QuantilesResult>, Error>`
- private `individual_count` and `find_quantile_position`

**Sparse ref** (matches `src/sparse.rs:86-340`, query subset only):
- `config`, `index`, `count`, `len`/`is_empty`
- `iter(&self) -> SparseIter<'a>`
- `quantiles` / `quantile`

Skip `checked_add`/`wrapping_add`/`checked_sub`/`wrapping_sub`/`downsample` on
the sparse ref — those return owned values and are not relevant to the
no-allocation percentile path.

### Shared iterators

Today the iterator structs hold `&'a $name` (`src/cumulative.rs:291-293,
328-332`, `src/sparse.rs:427-430`). Refactor them to hold the slices directly:

```rust
pub struct CumulativeIter<'a> {
    position: usize,
    config: Config,
    index: &'a [u32],
    count: &'a [u32],
}
```

Both owned and ref `iter()` methods can then return the same iterator type:

```rust
// owned
pub fn iter(&self) -> CumulativeIter<'_> {
    CumulativeIter { position: 0, config: self.config, index: &self.index, count: &self.count }
}

// ref
pub fn iter(&self) -> CumulativeIter<'a> {
    CumulativeIter { position: 0, config: self.config, index: self.index, count: self.count }
}
```

The iterator types are `pub` in their module but **not** re-exported from
`lib.rs`, so consumers can't name them — changing the field layout is not a
public API break. The `Iterator` / `ExactSizeIterator` / `FusedIterator` impls
stay unchanged.

### `as_ref()` on owned types

```rust
impl CumulativeROHistogram32 {
    pub fn as_ref(&self) -> CumulativeROHistogram32Ref<'_> {
        CumulativeROHistogram32Ref::from_parts_unchecked(
            self.config, &self.index, &self.count,
        )
    }
}
```

Use this to DRY the owned type's quantile methods — they all delegate:

```rust
pub fn quantiles(&self, q: &[f64]) -> Result<Option<QuantilesResult>, Error> {
    self.as_ref().quantiles(q)
}
pub fn quantile(&self, q: f64) -> Result<Option<QuantilesResult>, Error> {
    self.as_ref().quantile(q)
}
pub fn bucket_quantile_range(&self, i: usize) -> Option<(f64, f64)> {
    self.as_ref().bucket_quantile_range(i)
}
pub fn total_count(&self) -> u64 {
    self.as_ref().total_count()
}
```

`SampleQuantiles for $name` likewise delegates to `self.as_ref().quantiles(q)`.

### Trait impls on the ref

```rust
impl SampleQuantiles for CumulativeROHistogram32Ref<'_> { ... }

impl<'a> From<&'a CumulativeROHistogram32> for CumulativeROHistogram32Ref<'a> {
    fn from(h: &'a CumulativeROHistogram32) -> Self {
        Self::from_parts_unchecked(h.config(), h.index(), h.count())
    }
}

impl<'a> IntoIterator for CumulativeROHistogram32Ref<'a> {
    type Item = Bucket;
    type IntoIter = CumulativeIter<'a>;
    fn into_iter(self) -> Self::IntoIter { self.iter() }
}

impl<'a, 'b> IntoIterator for &'a CumulativeROHistogram32Ref<'b> {
    type Item = Bucket;
    type IntoIter = CumulativeIter<'b>;
    fn into_iter(self) -> Self::IntoIter { self.iter() }
}
```

### Macro changes

Both `cumulative.rs` and `sparse.rs` already use macros to generate the u32 and
u64 variants. Extend each macro's parameter list with `$ref_name:ident` and
emit the ref struct + its impls inside the same expansion.

```rust
define_cumulative_histogram!(
    CumulativeROHistogram, CumulativeROHistogramRef,
    CumulativeIter, QuantileRangeIter,
    Histogram, SparseHistogram, u64
);
define_cumulative_histogram!(
    CumulativeROHistogram32, CumulativeROHistogram32Ref,
    CumulativeIter32, QuantileRangeIter32,
    Histogram32, SparseHistogram32, u32
);
```

Same shape for `define_sparse_histogram!` — add a `$ref_name` parameter.

## `lib.rs` re-exports

```rust
pub use cumulative::{
    CumulativeROHistogram, CumulativeROHistogram32,
    CumulativeROHistogramRef, CumulativeROHistogram32Ref,
};
pub use sparse::{
    SparseHistogram, SparseHistogram32,
    SparseHistogramRef, SparseHistogram32Ref,
};
```

## Tests

Add a test module per file (or extend existing) covering:

1. **Validation parity:** `Ref::from_parts` returns the same `Err` variants as
   the owned `from_parts` for each invalid input (length mismatch, OOR index,
   non-ascending indices, duplicate indices, non-monotone counts, zero count).
2. **Quantile parity:** for a histogram populated via the standard path, the
   `Ref` (built from slices) and the owned struct return identical
   `QuantilesResult` for the same `&[f64]`.
3. **`from_parts_unchecked` happy path:** quantiles match.
4. **`From<&Owned>` round trip:** `Ref::from(&owned).quantiles(q)` matches
   `owned.quantiles(q)`.
5. **Iterators agree:** `Ref::iter().collect()` equals `owned.iter().collect()`.
6. **`bucket_quantile_range` parity** (cumulative only).
7. **Empty / single-sample edge cases.**
8. **u32 + u64 symmetry:** run the same scenarios on both width variants.

## CHANGELOG

Under `[Unreleased]`:

```
### Added

- `CumulativeROHistogramRef<'a>` / `CumulativeROHistogram32Ref<'a>` and
  `SparseHistogramRef<'a>` / `SparseHistogram32Ref<'a>` — borrowed views over
  histogram storage that mirror the read-only API surface of the owned types.
  Lets columnar consumers compute quantiles directly off `&[u32]` slices
  without per-snapshot allocation or revalidation.
- `from_parts_unchecked` constructor on each ref type for hot-path use where
  invariants are already known to hold; `from_parts` validates as before.
- `as_ref()` on `CumulativeROHistogram*` and `SparseHistogram*` returning the
  matching ref view; `From<&Owned> for OwnedRef<'_>` for trait-based conversion.
- `SampleQuantiles` impl on each ref type.
```

No `### Changed` entry needed — the iterator field-layout refactor is internal
(types aren't re-exported from `lib.rs`), and the owned types' public methods
are unchanged.

## Cargo.toml

```
version = "1.3.1-alpha.0"
```

(`main` is at `1.3.0`; per `CLAUDE.md`, feature PRs use `-alpha.<revision>`.)

## Verification

- `cargo test` — full suite must pass, including the new tests
- `cargo clippy --all-targets --all-features -- -D warnings`
- `cargo doc --no-deps` — confirm the new types render with their docs

## Out of scope

- Adding equivalent ref types for `Histogram`/`AtomicHistogram`. Those are the
  recording side; the optimization target is the read-only analytics path.
- Mutation methods (`checked_add` etc.) on the sparse ref — the ref is a
  read-only view; mutating ops belong on the owned type.
- A free `quantiles_from_parts` function — superseded by the ref type, which
  covers the same use case plus the rest of the read-only API.

## Commit / PR

- Branch: `claude/optimize-percentile-queries-fgONU`
- One commit (or split: "refactor iterators to hold slices" + "add Ref view
  types" + "version + changelog" if reviewer prefers smaller commits)
- Open as a **draft** PR against `main` once pushed
