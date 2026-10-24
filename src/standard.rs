use std::collections::BTreeMap;

use crate::quantile::{Quantile, QuantilesResult, SampleQuantiles};
use crate::{Bucket, Config, Count, Error, SparseHistogram, SparseHistogram32};

/// Counters per quantile-scan step: four AVX2 lanes of `u64`, so one
/// horizontal reduction serves 32 counters instead of 8.
const SCAN_CHUNK: usize = 32;
/// What one report's bounds-and-total pass established.
#[derive(Clone, Copy)]
struct Scan {
    min_idx: usize,
    max_idx: usize,
    total: u128,
    /// Some counter in the occupied range is at or above 2⁵⁹: word-sized chunk
    /// sums are not trusted and every counter is widened.
    wide: bool,
}

macro_rules! define_histogram {
    ($name:ident, $iter:ident, $sparse:ident, $count:ty $(, #[$merge_hint:meta])?) => {
        /// A histogram that uses plain counters for each bucket.
        #[derive(Clone, Debug, PartialEq, Eq)]
        #[cfg_attr(feature = "serde", derive(serde::Serialize))]
        #[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
        pub struct $name {
            pub(crate) config: Config,
            pub(crate) buckets: Box<[$count]>,
        }

        #[cfg(feature = "serde")]
        impl<'de> serde::Deserialize<'de> for $name {
            fn deserialize<D: serde::Deserializer<'de>>(
                deserializer: D,
            ) -> Result<Self, D::Error> {
                // Use the original type name and field order on the wire.
                #[derive(serde::Deserialize)]
                struct $name {
                    config: Config,
                    buckets: Vec<$count>,
                }
                let wire = <$name as serde::Deserialize>::deserialize(deserializer)?;
                Self::from_buckets(
                    wire.config.grouping_power(),
                    wire.config.max_value_power(),
                    wire.buckets,
                )
                .map_err(serde::de::Error::custom)
            }
        }

        impl $name {
            /// Construct a new histogram from the provided parameters.
            pub fn new(grouping_power: u8, max_value_power: u8) -> Result<Self, Error> {
                let config = Config::new(grouping_power, max_value_power)?;
                Ok(Self::with_config(&config))
            }

            /// Creates a new histogram using a provided [`crate::Config`].
            pub fn with_config(config: &Config) -> Self {
                let buckets: Box<[$count]> =
                    vec![<$count as Count>::ZERO; config.total_buckets()].into();
                Self {
                    config: *config,
                    buckets,
                }
            }

            /// Creates a new histogram from a config and a vector of buckets.
            pub fn from_buckets(
                grouping_power: u8,
                max_value_power: u8,
                buckets: Vec<$count>,
            ) -> Result<Self, Error> {
                let config = Config::new(grouping_power, max_value_power)?;
                if config.total_buckets() != buckets.len() {
                    return Err(Error::IncompatibleParameters);
                }
                Ok(Self {
                    config,
                    buckets: buckets.into(),
                })
            }

            /// Increment the counter for the bucket corresponding to `value` by one.
            /// Uses wrapping arithmetic on overflow.
            pub fn increment(&mut self, value: u64) -> Result<(), Error> {
                self.add(value, <$count as Count>::ONE)
            }

            /// Add `count` to the bucket containing `value`. Uses wrapping
            /// arithmetic on overflow.
            pub fn add(&mut self, value: u64, count: $count) -> Result<(), Error> {
                let index = self.config.value_to_index(value)?;
                self.buckets[index] = self.buckets[index].wrapping_add(count);
                Ok(())
            }

            /// Reset all bucket counts to zero, preserving the configuration and
            /// existing counter allocation for reuse in another reporting interval.
            ///
            /// This takes time proportional to the number of buckets and does not
            /// allocate. The histogram reports no observations until values are
            /// recorded again.
            pub fn reset(&mut self) {
                self.buckets.fill(<$count as Count>::ZERO);
            }

            /// Get a reference to the raw counters.
            pub fn as_slice(&self) -> &[$count] {
                &self.buckets
            }

            /// Get a mutable reference to the raw counters.
            pub fn as_mut_slice(&mut self) -> &mut [$count] {
                &mut self.buckets
            }

            /// Returns a new histogram with a reduced grouping power. The reduced
            /// grouping power should lie in the range (0..existing grouping power).
            ///
            /// Returns an error if the requested grouping power is not less than the current grouping power.
            ///
            /// The difference in grouping powers determines how much histogram size
            /// is reduced by, with every step approximately halving the total
            /// number of buckets (and hence total size of the histogram), while
            /// doubling the relative error.
            ///
            /// This works by iterating over every bucket in the existing histogram
            /// and inserting the contained values into the new histogram. While we
            /// do not know the exact values of the data points (only that they lie
            /// within the bucket's range), it does not matter since the bucket is
            /// not split during downsampling and any value can be used.
            pub fn downsample(&self, grouping_power: u8) -> Result<Self, Error> {
                if grouping_power >= self.config.grouping_power() {
                    return Err(Error::IncompatibleParameters);
                }
                let mut histogram = Self::new(grouping_power, self.config.max_value_power())?;
                for (i, n) in self.as_slice().iter().enumerate() {
                    if *n != <$count as Count>::ZERO {
                        let val = self.config.index_to_lower_bound(i);
                        histogram.add(val, *n)?;
                    }
                }
                Ok(histogram)
            }

            /// Adds the other histogram to this histogram and returns the result as a
            /// new histogram.
            ///
            /// An error is returned if the two histograms have incompatible parameters
            /// or if there is an overflow.
            pub fn checked_add(&self, other: &Self) -> Result<Self, Error> {
                if self.config != other.config {
                    return Err(Error::IncompatibleParameters);
                }
                let mut result = self.clone();
                for (this, other) in result.buckets.iter_mut().zip(other.buckets.iter()) {
                    *this = this.checked_add(*other).ok_or(Error::Overflow)?;
                }
                Ok(result)
            }

            /// Sums compatible histograms into one new, independently owned histogram.
            ///
            /// All configurations are checked before allocating or adding counts.
            /// An empty input returns [`Error::IncompatibleParameters`] because
            /// there is no configuration for the result. A single input is cloned.
            /// Repeated references are allowed and count as repeated inputs.
            ///
            /// Returns [`Error::IncompatibleParameters`] if any configurations differ,
            /// or [`Error::Overflow`] if any individual bucket sum exceeds the counter
            /// limit. The total across different buckets may exceed that limit.
            /// Inputs are never modified, including when an error is returned.
            ///
            /// Clones the first input once, then combines addition and overflow
            /// detection in one pass over each remaining input's buckets. A failing
            /// partial result is discarded. Overflow is checked after each input,
            /// so an overflowing input is scanned to the end before returning.
            /// For an existing destination that must retain its allocation, use
            /// [`Self::checked_add_assign`] instead.
            ///
            /// On x86/x86-64, AVX2 is selected automatically when supported by the
            /// CPU and operating system. Other machines use the portable kernel.
            /// No compiler flags or higher minimum CPU requirement are needed.
            ///
            /// ```
            /// use histogram::Histogram;
            ///
            /// let mut first = Histogram::new(7, 30)?;
            /// let mut second = Histogram::new(7, 30)?;
            /// first.add(100, 2)?;
            /// second.add(100, 3)?;
            /// let combined = Histogram::checked_sum(&[&first, &second])?;
            /// assert_eq!(combined, first.checked_add(&second)?);
            /// # Ok::<(), histogram::Error>(())
            /// ```
            #[inline]
            pub fn checked_sum(inputs: &[&Self]) -> Result<Self, Error> {
                // TODO: Benchmark AVX-512 against AVX2 on compatible hardware before
                // preferring it here, including small/large merges and recording
                // interleaved with reporting to assess frequency effects. AVX-512
                // target_feature support requires Rust 1.89; our MSRV is 1.85.
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                if inputs.len() > 1 && std::is_x86_feature_detected!("avx2") {
                    // SAFETY: detection checks CPU and OS support for AVX2 before
                    // entering this target-feature function. Selection is per batch,
                    // not per bucket or source histogram.
                    return unsafe { Self::checked_sum_avx2(inputs) };
                }
                Self::checked_sum_portable(inputs)
            }

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            #[target_feature(enable = "avx2")]
            unsafe fn checked_sum_avx2(inputs: &[&Self]) -> Result<Self, Error> {
                Self::checked_sum_portable(inputs)
            }

            // Inline the same arithmetic into each target-feature context so the
            // compiler can vectorize it using that context's supported instructions.
            #[inline(always)]
            fn checked_sum_portable(inputs: &[&Self]) -> Result<Self, Error> {
                let first = inputs.first().ok_or(Error::IncompatibleParameters)?;
                if inputs.iter().any(|input| input.config != first.config) {
                    return Err(Error::IncompatibleParameters);
                }
                let mut result = (*first).clone();
                for input in &inputs[1..] {
                    let mut overflow = false;
                    for (dst, &src) in result.buckets.iter_mut().zip(input.buckets.iter()) {
                        let (sum, carry) = dst.overflowing_add(src);
                        *dst = sum;
                        overflow |= carry;
                    }
                    if overflow {
                        return Err(Error::Overflow);
                    }
                }
                Ok(result)
            }

            /// Adds a compatible histogram into this histogram without allocating.
            ///
            /// Configurations must match. Returns [`Error::IncompatibleParameters`]
            /// or [`Error::Overflow`] without changing any counts. Overflow is
            /// checked per bucket, not against the sum of all counts.
            ///
            /// Takes two passes over the buckets: one to validate all additions,
            /// then one to apply them. Configuration and counter storage are preserved.
            $(#[$merge_hint])?
            pub fn checked_add_assign(&mut self, other: &Self) -> Result<(), Error> {
                if self.config != other.config {
                    return Err(Error::IncompatibleParameters);
                }
                if <$count>::BITS == 32 && self.buckets.len() >= 8 {
                    // Reducing u32 overflow flags allows vectorized validation.
                    // Eight counters match the observed 128-bit vector lowering;
                    // shorter arrays avoid reduction overhead via scalar checks.
                    let overflow = self
                        .buckets
                        .iter()
                        .zip(other.buckets.iter())
                        .fold(false, |overflow, (&a, &b)| {
                            overflow | a.overflowing_add(b).1
                        });
                    if overflow {
                        return Err(Error::Overflow);
                    }
                } else {
                    for (this, other) in self.buckets.iter().zip(other.buckets.iter()) {
                        this.checked_add(*other).ok_or(Error::Overflow)?;
                    }
                }
                for (this, other) in self.buckets.iter_mut().zip(other.buckets.iter()) {
                    *this = this.wrapping_add(*other);
                }
                Ok(())
            }

            /// Adds the other histogram to this histogram and returns the result as a
            /// new histogram.
            ///
            /// An error is returned if the two histograms have incompatible parameters.
            pub fn wrapping_add(&self, other: &Self) -> Result<Self, Error> {
                if self.config != other.config {
                    return Err(Error::IncompatibleParameters);
                }
                let mut result = self.clone();
                for (this, other) in result.buckets.iter_mut().zip(other.buckets.iter()) {
                    *this = this.wrapping_add(*other);
                }
                Ok(result)
            }

            /// Subtracts the other histogram from this histogram and returns the result
            /// as a new histogram.
            ///
            /// An error is returned if the two histograms have incompatible parameters
            /// or if there is an overflow.
            pub fn checked_sub(&self, other: &Self) -> Result<Self, Error> {
                if self.config != other.config {
                    return Err(Error::IncompatibleParameters);
                }
                let mut result = self.clone();
                for (this, other) in result.buckets.iter_mut().zip(other.buckets.iter()) {
                    *this = this.checked_sub(*other).ok_or(Error::Underflow)?;
                }
                Ok(result)
            }

            /// Subtracts the other histogram from this histogram and returns the result
            /// as a new histogram.
            ///
            /// An error is returned if the two histograms have incompatible parameters.
            pub fn wrapping_sub(&self, other: &Self) -> Result<Self, Error> {
                if self.config != other.config {
                    return Err(Error::IncompatibleParameters);
                }
                let mut result = self.clone();
                for (this, other) in result.buckets.iter_mut().zip(other.buckets.iter()) {
                    *this = this.wrapping_sub(*other);
                }
                Ok(result)
            }

            /// Returns an iterator across the histogram.
            pub fn iter(&self) -> $iter<'_> {
                $iter {
                    index: 0,
                    histogram: self,
                }
            }

            /// Returns the bucket configuration of the histogram.
            pub fn config(&self) -> Config {
                self.config
            }

            /// Compute quantiles for the given values.
            ///
            /// Each value in `quantiles` must be in `0.0..=1.0`. Returns
            /// `Err(Error::InvalidQuantile)` if any value is out of range,
            /// `Ok(None)` if the histogram is empty.
            ///
            /// This is an inherent forwarder for [`SampleQuantiles::quantiles`].
            pub fn quantiles(&self, quantiles: &[f64]) -> Result<Option<QuantilesResult>, Error> {
                <Self as SampleQuantiles>::quantiles(self, quantiles)
            }

            /// Compute a single quantile.
            ///
            /// The quantile must be in `0.0..=1.0`. Returns
            /// `Err(Error::InvalidQuantile)` if out of range, `Ok(None)` if the
            /// histogram is empty.
            ///
            /// This is an inherent forwarder for [`SampleQuantiles::quantile`].
            pub fn quantile(&self, quantile: f64) -> Result<Option<QuantilesResult>, Error> {
                <Self as SampleQuantiles>::quantile(self, quantile)
            }

            /// Returns one quantile bucket without allocating a result map.
            ///
            /// Uses nearest rank with quantiles in `0.0..=1.0`; zero selects the
            /// first occupied bucket and one the last. Returns `Ok(None)` for an
            /// empty histogram or [`Error::InvalidQuantile`] for an invalid request.
            /// The bucket contains its individual count and inclusive value range.
            /// Interior ranks use `ceil(quantile * total as f64)`, clamped to the
            /// observation count; very large totals can incur floating-point rounding.
            ///
            /// Counts are scanned at query time; recording maintains no cached
            /// totals or bounds. Use [`Self::quantile_buckets_into`] to share scan
            /// work across a report's quantiles, or a cumulative snapshot for
            /// repeated reads of completed data.
            pub fn quantile_bucket(&self, quantile: f64) -> Result<Option<Bucket>, Error> {
                let mut output = [None];
                self.quantile_buckets_into(&[quantile], &mut output)?;
                Ok(output[0].take())
            }

            /// Writes quantile buckets into caller-owned storage without allocating.
            ///
            /// Requests may be unsorted and repeated; results preserve their order.
            /// Returns the number of slots written. An empty histogram writes
            /// `None`; an empty request writes nothing. Extra output slots are untouched.
            /// Buckets have the semantics of [`Self::quantile_bucket`].
            ///
            /// Validates every quantile before checking output capacity. Returns
            /// [`Error::InvalidQuantile`] or [`Error::InsufficientOutputCapacity`]
            /// without modifying output, including for empty histograms.
            ///
            /// Derives bounds and the exact total in one chunked pass, then shares one
            /// forward rank scan whose 32-counter steps vectorize.
            /// Selecting requests in rank order without scratch allocation takes
            /// O(Q²) work for Q requests, in addition to O(B) bucket work. Intended
            /// for small reporting batches; cumulative snapshots suit repeated analytics.
            pub fn quantile_buckets_into(
                &self,
                quantiles: &[f64],
                output: &mut [Option<Bucket>],
            ) -> Result<usize, Error> {
                if quantiles.iter().any(|q| !(0.0..=1.0).contains(q)) {
                    return Err(Error::InvalidQuantile);
                }
                if output.len() < quantiles.len() {
                    return Err(Error::InsufficientOutputCapacity {
                        required: quantiles.len(),
                        available: output.len(),
                    });
                }
                let slots = &mut output[..quantiles.len()];
                slots.fill(None);
                if quantiles.is_empty() {
                    return Ok(0);
                }
                let Some(scan) = self.occupied_total() else {
                    return Ok(quantiles.len());
                };
                let (max_idx, total) = (scan.max_idx, scan.total);
                let mut next = scan.min_idx;
                let mut partial = 0u128;
                // Unfilled slots identify requests still to process. No per-record
                // state, temporary vectors, or changes to the input are needed.
                while let Some(q) = quantiles
                    .iter()
                    .zip(slots.iter())
                    .filter(|(_, slot)| slot.is_none())
                    .map(|(&q, _)| q)
                    .min_by(f64::total_cmp)
                {
                    if q == 1.0 {
                        next = max_idx + 1;
                        partial = total;
                    } else {
                        self.advance_to_rank(&scan, &mut next, &mut partial, Self::rank_target(q, total));
                    }
                    let index = next - 1;
                    let bucket = Bucket {
                        count: self.buckets[index].as_u128() as u64,
                        range: self.config.index_to_range(index),
                    };
                    for (&request, slot) in quantiles.iter().zip(slots.iter_mut()) {
                        if request == q {
                            *slot = Some(bucket.clone());
                        }
                    }
                }
                Ok(quantiles.len())
            }

            /// Machine-word sum of one chunk: a single vectorized reduction. Exact
            /// only while every counter in the occupied range is below 2⁵⁹, which
            /// `occupied_total` establishes once per report; otherwise it wraps.
            #[inline(always)]
            fn chunk_sum(chunk: &[$count; SCAN_CHUNK]) -> u64 {
                // Wrapping: the total pass computes this speculatively and discards
                // it when the OR guard reports a wide counter, so it must not trap
                // under overflow checks; the walk only calls it when the guard passed.
                chunk.iter().fold(0u64, |acc, c| acc.wrapping_add(c.as_u64()))
            }

            /// Whether a chunk holds any observation: one OR reduction.
            #[inline(always)]
            fn chunk_occupied(chunk: &[$count; SCAN_CHUNK]) -> bool {
                chunk.iter().fold(0u64, |acc, c| acc | c.as_u64()) != 0
            }

            /// First and last occupied bucket, the exact total, and whether any
            /// counter in the occupied range is at or above 2⁵⁹ (`wide`). Empty
            /// chunks outside the occupied range are only tested for occupancy.
            /// The total pass sums chunks in `u64` while OR-ing every counter into
            /// four lane accumulators; one reduction of those at the end decides
            /// whether the word-sized sums were exact, and the rare wide case is
            /// re-summed in `u128`. Returns `None` when empty. Recording keeps no
            /// totals or bounds, so a report derives them here.
            fn occupied_total(&self) -> Option<Scan> {
                let buckets: &[$count] = &self.buckets;
                let zero = <$count as Count>::ZERO;
                let n = buckets.len();
                let mut lo = 0;
                while lo + SCAN_CHUNK <= n {
                    let chunk: &[$count; SCAN_CHUNK] = buckets[lo..lo + SCAN_CHUNK].try_into().unwrap();
                    if Self::chunk_occupied(chunk) {
                        break;
                    }
                    lo += SCAN_CHUNK;
                }
                let min_idx = lo + buckets[lo..].iter().position(|c| *c != zero)?;
                let mut hi = n;
                while hi >= min_idx + SCAN_CHUNK {
                    let chunk: &[$count; SCAN_CHUNK] = buckets[hi - SCAN_CHUNK..hi].try_into().unwrap();
                    if Self::chunk_occupied(chunk) {
                        break;
                    }
                    hi -= SCAN_CHUNK;
                }
                let max_idx = buckets[..hi].iter().rposition(|c| *c != zero).unwrap();
                let occupied = &buckets[min_idx..=max_idx];
                let chunks = occupied.chunks_exact(SCAN_CHUNK);
                // Four OR lanes fit one vector register, so the guard adds one
                // vector OR per four counters and no per-chunk reduction.
                let mut high = [0u64; 4];
                let mut total: u128 = chunks.remainder().iter().map(|c| c.as_u128()).sum();
                for c in chunks.remainder() {
                    high[0] |= c.as_u64();
                }
                for chunk in chunks {
                    let chunk: &[$count; SCAN_CHUNK] = chunk.try_into().unwrap();
                    for group in chunk.chunks_exact(4) {
                        for (acc, c) in high.iter_mut().zip(group.iter()) {
                            *acc |= c.as_u64();
                        }
                    }
                    total += Self::chunk_sum(chunk) as u128;
                }
                let wide = high.iter().fold(0u64, |acc, h| acc | h) >> 59 != 0;
                if wide {
                    total = occupied.iter().map(|c| c.as_u128()).sum();
                }
                Some(Scan { min_idx, max_idx, total, wide })
            }

            /// Advances the exclusive prefix (`next`, `partial`: the sum of
            /// `buckets[min_idx..next]`) until the inclusive prefix reaches
            /// `target`, leaving `next - 1` on the selected bucket. Chunks that
            /// cannot contain the target are skipped with word-sized sums; in the
            /// wide case every counter is widened individually.
            #[inline]
            fn advance_to_rank(&self, scan: &Scan, next: &mut usize, partial: &mut u128, target: u128) {
                let buckets: &[$count] = &self.buckets;
                let (mut i, mut acc) = (*next, *partial);
                if !scan.wide {
                    while acc < target && i + SCAN_CHUNK <= scan.max_idx + 1 {
                        let chunk: &[$count; SCAN_CHUNK] = buckets[i..i + SCAN_CHUNK].try_into().unwrap();
                        let subtotal = Self::chunk_sum(chunk) as u128;
                        if acc + subtotal >= target {
                            break;
                        }
                        acc += subtotal;
                        i += SCAN_CHUNK;
                    }
                }
                while acc < target && i <= scan.max_idx {
                    acc += buckets[i].as_u128();
                    i += 1;
                }
                *next = i;
                *partial = acc;
            }

            /// Nearest rank for `quantile` over `total` observations, clamped to the
            /// observation count so floating-point rounding at very large totals
            /// cannot request a rank past the last bucket.
            #[inline]
            fn rank_target(quantile: f64, total: u128) -> u128 {
                ((quantile * total as f64).ceil() as u128).clamp(1, total)
            }
        }

        impl SampleQuantiles for $name {
            fn quantiles(&self, quantiles: &[f64]) -> Result<Option<QuantilesResult>, Error> {
                for q in quantiles {
                    if !(0.0..=1.0).contains(q) {
                        return Err(Error::InvalidQuantile);
                    }
                }
                let Some(scan) = self.occupied_total() else {
                    return Ok(None);
                };
                let (min_idx, max_idx, total_count) = (scan.min_idx, scan.max_idx, scan.total);
                let mut sorted: Vec<Quantile> = quantiles
                    .iter()
                    .map(|&q| Quantile::new(q).unwrap())
                    .collect();
                sorted.sort();
                sorted.dedup();

                let bucket_at = |index: usize| Bucket {
                    count: self.buckets[index].as_u128() as u64,
                    range: self.config.index_to_range(index),
                };
                let min = bucket_at(min_idx);
                let max = bucket_at(max_idx);

                let mut next = min_idx;
                let mut partial = 0u128;
                let mut entries = BTreeMap::new();
                for quantile in &sorted {
                    if quantile.as_f64() == 1.0 {
                        next = max_idx + 1;
                        partial = total_count;
                    } else {
                        self.advance_to_rank(&scan, &mut next, &mut partial, Self::rank_target(quantile.as_f64(), total_count));
                    }
                    entries.insert(*quantile, bucket_at(next - 1));
                }

                Ok(Some(QuantilesResult::new(entries, total_count, min, max)))
            }
        }

        impl<'a> IntoIterator for &'a $name {
            type Item = Bucket;
            type IntoIter = $iter<'a>;
            fn into_iter(self) -> Self::IntoIter {
                $iter {
                    index: 0,
                    histogram: self,
                }
            }
        }

        /// An iterator across the histogram buckets.
        pub struct $iter<'a> {
            index: usize,
            histogram: &'a $name,
        }

        impl Iterator for $iter<'_> {
            type Item = Bucket;
            fn next(&mut self) -> Option<Bucket> {
                if self.index >= self.histogram.buckets.len() {
                    return None;
                }
                let bucket = Bucket {
                    count: self.histogram.buckets[self.index].as_u128() as u64,
                    range: self.histogram.config.index_to_range(self.index),
                };
                self.index += 1;
                Some(bucket)
            }
        }

        impl ExactSizeIterator for $iter<'_> {
            fn len(&self) -> usize {
                self.histogram.buckets.len() - self.index
            }
        }

        impl std::iter::FusedIterator for $iter<'_> {}

        // Requires: `$sparse.count` is `Vec<$count>` (the same count type as `$name`).
        // Task 4 enforces this when defining SparseHistogram32 alongside Histogram32.
        impl From<&$sparse> for $name {
            fn from(other: &$sparse) -> Self {
                let mut histogram = $name::with_config(&other.config);
                for (index, count) in other.index.iter().zip(other.count.iter()) {
                    histogram.buckets[*index as usize] = *count;
                }
                histogram
            }
        }
    };
}

define_histogram!(Histogram, Iter, SparseHistogram, u64);
define_histogram!(Histogram32, Iter32, SparseHistogram32, u32, #[inline]);

// Deprecated forwarding methods — only on the u64 variant to avoid proliferating
// deprecated APIs onto the new u32 type. Mirrors the same pattern in sparse.rs.
impl Histogram {
    /// Return a collection of percentiles from this histogram.
    ///
    /// Each percentile should be in the inclusive range `0.0..=1.0`. For
    /// example, the 50th percentile (median) can be found using `0.5`.
    ///
    /// The results will be sorted by the percentile.
    #[deprecated(note = "Use the SampleQuantiles trait")]
    #[allow(deprecated)]
    pub fn percentiles(&self, percentiles: &[f64]) -> Result<Option<Vec<(f64, Bucket)>>, Error> {
        Ok(SampleQuantiles::quantiles(self, percentiles)
            .map_err(|e| match e {
                Error::InvalidQuantile => Error::InvalidPercentile,
                other => other,
            })?
            .map(|qr| {
                qr.entries()
                    .iter()
                    .map(|(q, b)| (q.as_f64(), b.clone()))
                    .collect()
            }))
    }

    /// Return a single percentile from this histogram.
    ///
    /// The percentile should be in the inclusive range `0.0..=1.0`. For
    /// example, the 50th percentile (median) can be found using `0.5`.
    #[deprecated(note = "Use the SampleQuantiles trait")]
    pub fn percentile(&self, percentile: f64) -> Result<Option<Bucket>, Error> {
        #[allow(deprecated)]
        self.percentiles(&[percentile])
            .map(|v| v.map(|x| x.first().unwrap().1.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngExt;

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn size() {
        assert_eq!(std::mem::size_of::<Histogram>(), 48);
    }

    #[test]
    // Tests quantiles (replaces the deprecated percentile/percentiles tests).
    fn quantiles() {
        let mut histogram = Histogram::new(7, 64).unwrap();

        assert_eq!(histogram.quantile(0.5).unwrap(), None);
        assert_eq!(histogram.quantiles(&[0.5, 0.9, 0.99, 0.999]).unwrap(), None);

        for i in 0..=100 {
            let _ = histogram.increment(i);
            let r = histogram.quantile(0.0).unwrap().unwrap();
            assert_eq!(
                r.get(&Quantile::new(0.0).unwrap()),
                Some(&Bucket {
                    count: 1,
                    range: 0..=0,
                })
            );
            let r = histogram.quantile(1.0).unwrap().unwrap();
            assert_eq!(
                r.get(&Quantile::new(1.0).unwrap()),
                Some(&Bucket {
                    count: 1,
                    range: i..=i,
                })
            );
        }
        let r = histogram.quantile(0.0).unwrap().unwrap();
        assert_eq!(r.get(&Quantile::new(0.0).unwrap()).unwrap().end(), 0);
        let r = histogram.quantile(0.25).unwrap().unwrap();
        assert_eq!(r.get(&Quantile::new(0.25).unwrap()).unwrap().end(), 25);
        let r = histogram.quantile(0.50).unwrap().unwrap();
        assert_eq!(r.get(&Quantile::new(0.50).unwrap()).unwrap().end(), 50);
        let r = histogram.quantile(0.75).unwrap().unwrap();
        assert_eq!(r.get(&Quantile::new(0.75).unwrap()).unwrap().end(), 75);
        let r = histogram.quantile(0.90).unwrap().unwrap();
        assert_eq!(r.get(&Quantile::new(0.90).unwrap()).unwrap().end(), 90);
        let r = histogram.quantile(0.99).unwrap().unwrap();
        assert_eq!(r.get(&Quantile::new(0.99).unwrap()).unwrap().end(), 99);
        let r = histogram.quantile(0.999).unwrap().unwrap();
        assert_eq!(r.get(&Quantile::new(0.999).unwrap()).unwrap().end(), 100);

        assert_eq!(histogram.quantile(-1.0), Err(Error::InvalidQuantile));
        assert_eq!(histogram.quantile(1.01), Err(Error::InvalidQuantile));

        let _ = histogram.increment(1024);
        let r = histogram.quantile(0.999).unwrap().unwrap();
        assert_eq!(
            r.get(&Quantile::new(0.999).unwrap()),
            Some(&Bucket {
                count: 1,
                range: 1024..=1031,
            })
        );
    }

    #[test]
    fn min() {
        let mut histogram = Histogram::new(7, 64).unwrap();

        assert_eq!(histogram.quantile(0.0).unwrap(), None);

        let _ = histogram.increment(10);
        let r = histogram.quantile(0.0).unwrap().unwrap();
        assert_eq!(r.get(&Quantile::new(0.0).unwrap()).unwrap().end(), 10);

        let _ = histogram.increment(4);
        let r = histogram.quantile(0.0).unwrap().unwrap();
        assert_eq!(r.get(&Quantile::new(0.0).unwrap()).unwrap().end(), 4);
    }

    #[test]
    fn downsample() {
        let mut histogram = Histogram::new(8, 32).unwrap();
        let mut vals: Vec<u64> = Vec::with_capacity(10000);
        use rand::SeedableRng;
        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);

        for _ in 0..vals.capacity() {
            let v: u64 = rng.random_range(1..2_u64.pow(histogram.config.max_value_power() as u32));
            vals.push(v);
            let _ = histogram.increment(v);
        }
        vals.sort();

        let mut percentiles: Vec<f64> = Vec::with_capacity(109);
        for i in 20..99 {
            percentiles.push(i as f64 / 100.0);
        }
        let mut tail = vec![
            0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999, 0.9999, 1.0,
        ];
        percentiles.append(&mut tail);

        let h = histogram.clone();
        let grouping_power = histogram.config.grouping_power();
        for factor in 1..grouping_power {
            let error = histogram.config.error();

            for p in &percentiles {
                let v = vals[((*p * (vals.len() as f64)) as usize) - 1];
                let q = histogram.quantile(*p).unwrap().unwrap();
                let vhist = q.get(&Quantile::new(*p).unwrap()).unwrap().end();
                let e = (v.abs_diff(vhist) as f64) * 100.0 / (v as f64);
                assert!(e < error);
            }

            histogram = h.downsample(grouping_power - factor).unwrap();
        }
    }

    fn build_histograms() -> (Histogram, Histogram, Histogram, Histogram) {
        let mut h1 = Histogram::new(1, 3).unwrap();
        let mut h2 = Histogram::new(1, 3).unwrap();
        let mut h3 = Histogram::new(1, 3).unwrap();
        let h4 = Histogram::new(7, 32).unwrap();

        for i in 0..h1.config().total_buckets() {
            h1.as_mut_slice()[i] = 1;
            h2.as_mut_slice()[i] = 1;
            h3.as_mut_slice()[i] = u64::MAX;
        }

        (h1, h2, h3, h4)
    }

    #[test]
    fn checked_add() {
        let (h, h_good, h_overflow, h_mismatch) = build_histograms();
        assert_eq!(
            h.checked_add(&h_mismatch),
            Err(Error::IncompatibleParameters)
        );
        let r = h.checked_add(&h_good).unwrap();
        assert_eq!(r.as_slice(), &[2, 2, 2, 2, 2, 2]);
        assert_eq!(h.checked_add(&h_overflow), Err(Error::Overflow));
    }

    #[test]
    fn wrapping_add() {
        let (h, h_good, h_overflow, h_mismatch) = build_histograms();
        assert_eq!(
            h.wrapping_add(&h_mismatch),
            Err(Error::IncompatibleParameters)
        );
        let r = h.wrapping_add(&h_good).unwrap();
        assert_eq!(r.as_slice(), &[2, 2, 2, 2, 2, 2]);
        let r = h.wrapping_add(&h_overflow).unwrap();
        assert_eq!(r.as_slice(), &[0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn checked_sub() {
        let (h, h_good, h_overflow, h_mismatch) = build_histograms();
        assert_eq!(
            h.checked_sub(&h_mismatch),
            Err(Error::IncompatibleParameters)
        );
        let r = h.checked_sub(&h_good).unwrap();
        assert_eq!(r.as_slice(), &[0, 0, 0, 0, 0, 0]);
        assert_eq!(h.checked_sub(&h_overflow), Err(Error::Underflow));
    }

    #[test]
    fn wrapping_sub() {
        let (h, h_good, h_overflow, h_mismatch) = build_histograms();
        assert_eq!(
            h.wrapping_sub(&h_mismatch),
            Err(Error::IncompatibleParameters)
        );
        let r = h.wrapping_sub(&h_good).unwrap();
        assert_eq!(r.as_slice(), &[0, 0, 0, 0, 0, 0]);
        let r = h.wrapping_sub(&h_overflow).unwrap();
        assert_eq!(r.as_slice(), &[2, 2, 2, 2, 2, 2]);
    }

    #[test]
    fn from_buckets() {
        let mut histogram = Histogram::new(8, 32).unwrap();
        for i in 0..=100 {
            let _ = histogram.increment(i);
        }
        let buckets = histogram.as_slice();
        let constructed = Histogram::from_buckets(8, 32, buckets.to_vec()).unwrap();
        assert!(constructed == histogram);
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn size_u32() {
        assert_eq!(std::mem::size_of::<Histogram32>(), 48);
    }

    #[test]
    fn increment_u32() {
        let mut h = Histogram32::new(7, 64).unwrap();
        h.increment(5).unwrap();
        h.increment(5).unwrap();
        h.increment(5).unwrap();
        let result = h.quantile(0.5).unwrap().unwrap();
        let q = Quantile::new(0.5).unwrap();
        assert_eq!(result.get(&q).unwrap().count(), 3u64);
    }

    #[test]
    fn add_u32_wraps_at_max() {
        let mut h = Histogram32::new(2, 4).unwrap();
        h.add(1, u32::MAX).unwrap();
        h.add(1, 1).unwrap();
        assert_eq!(h.as_slice()[1], 0u32);
    }

    #[test]
    fn checked_add_u32_overflow() {
        let mut h1 = Histogram32::new(1, 3).unwrap();
        let mut h2 = Histogram32::new(1, 3).unwrap();
        h1.as_mut_slice()[0] = u32::MAX;
        h2.as_mut_slice()[0] = 1;
        assert_eq!(h1.checked_add(&h2), Err(Error::Overflow));
    }

    #[test]
    fn iter_u32_widens_count_to_u64() {
        let mut h = Histogram32::new(2, 4).unwrap();
        h.add(1, 5u32).unwrap();
        let bucket = h.iter().find(|b| b.count() > 0).unwrap();
        let count: u64 = bucket.count();
        assert_eq!(count, 5);
    }
}

#[cfg(test)]
#[path = "standard_sum_tests.rs"]
mod sum_backend_tests;
