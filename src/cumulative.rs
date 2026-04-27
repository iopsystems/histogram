use std::collections::BTreeMap;

use crate::quantile::{Quantile, QuantilesResult, SampleQuantiles};
use crate::{
    Bucket, Config, Count, Error, Histogram, Histogram32, SparseHistogram, SparseHistogram32,
};

macro_rules! define_cumulative_histogram {
    ($name:ident, $iter:ident, $qr_iter:ident, $hist:ident, $sparse:ident, $count:ty) => {
        /// A read-only, cumulative histogram for fast quantile queries.
        ///
        /// This is a variant of the [`SparseHistogram`] with cumulative counts
        /// (starting from the first bucket) for each bucket that is present.
        ///
        /// Stores only non-zero buckets in columnar form, like [`SparseHistogram`],
        /// but with **cumulative** counts: `count[i]` equals the total number of
        /// observations in buckets `0..=i` (i.e., a running prefix sum). The last
        /// element of `count` equals the total observation count.
        ///
        /// `CumulativeROHistogram` is intended to be read-only—i.e. it shouldn't
        /// accept updates for new observations, because such operations would be
        /// expensive given counts are cumulative. On the other hand, querying
        /// percentiles is cheaper than standard or sparse histograms without
        /// cumulative counts, which can be performed with binary search.
        /// Additional methods to provide the percentile range each or all bucket(s)
        /// represent are implmented to facilitate analytics based on such histograms.
        #[derive(Clone, Debug, PartialEq, Eq)]
        #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
        #[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
        pub struct $name {
            config: Config,
            index: Vec<u32>,
            count: Vec<$count>,
        }

        impl $name {
            /// Creates a cumulative histogram from its raw parts.
            ///
            /// The `count` vector must contain **cumulative** counts (a running prefix
            /// sum of individual bucket counts).
            ///
            /// Returns an error if:
            /// - `index` and `count` have different lengths
            /// - any index is out of range for the config
            /// - the indices are not in strictly ascending order
            /// - the counts are not strictly non-decreasing
            /// - any count is zero
            pub fn from_parts(
                config: Config,
                index: Vec<u32>,
                count: Vec<$count>,
            ) -> Result<Self, Error> {
                if index.len() != count.len() {
                    return Err(Error::IncompatibleParameters);
                }

                let total_buckets = config.total_buckets();
                let mut prev_idx = None;
                for &idx in &index {
                    if idx as usize >= total_buckets {
                        return Err(Error::OutOfRange);
                    }
                    if let Some(p) = prev_idx {
                        if idx <= p {
                            return Err(Error::IncompatibleParameters);
                        }
                    }
                    prev_idx = Some(idx);
                }

                let mut prev_count = None;
                for &c in &count {
                    if c == <$count as Count>::ZERO {
                        return Err(Error::IncompatibleParameters);
                    }
                    if let Some(p) = prev_count {
                        if c < p {
                            return Err(Error::IncompatibleParameters);
                        }
                    }
                    prev_count = Some(c);
                }

                Ok(Self {
                    config,
                    index,
                    count,
                })
            }

            /// Consumes the histogram, returning the config, index, and cumulative
            /// count vectors.
            pub fn into_parts(self) -> (Config, Vec<u32>, Vec<$count>) {
                (self.config, self.index, self.count)
            }

            /// Returns the bucket configuration.
            pub fn config(&self) -> Config {
                self.config
            }

            /// Returns a slice of the non-zero bucket indices.
            pub fn index(&self) -> &[u32] {
                &self.index
            }

            /// Returns a slice of the cumulative bucket counts.
            pub fn count(&self) -> &[$count] {
                &self.count
            }

            /// Returns the total number of observations across all buckets.
            pub fn total_count(&self) -> u64 {
                self.count.last().map(|c| c.as_u128() as u64).unwrap_or(0)
            }

            /// Returns the number of non-zero buckets.
            pub fn len(&self) -> usize {
                self.index.len()
            }

            /// Returns `true` if the histogram contains no observations.
            pub fn is_empty(&self) -> bool {
                self.index.is_empty()
            }

            /// Returns the quantile range `(lower, upper)` for the bucket at
            /// position `bucket_idx` in the sparse representation.
            ///
            /// - `lower` is the fraction of observations strictly before this bucket
            ///   (in `[0.0, 1.0]`).
            /// - `upper` is the fraction of observations at or before this bucket
            ///   (in `[0.0, 1.0]`).
            ///
            /// Returns `None` if the histogram is empty or `bucket_idx` is out of
            /// range.
            pub fn bucket_quantile_range(&self, bucket_idx: usize) -> Option<(f64, f64)> {
                if bucket_idx >= self.count.len() {
                    return None;
                }
                let total = self.count.last().map(|c| c.as_u128() as f64)?;
                if total == 0.0 {
                    return None;
                }
                let lower = if bucket_idx == 0 {
                    0.0
                } else {
                    self.count[bucket_idx - 1].as_u128() as f64 / total
                };
                let upper = self.count[bucket_idx].as_u128() as f64 / total;
                Some((lower, upper))
            }

            /// Returns an iterator yielding `(Bucket, lower_quantile, upper_quantile)`
            /// for each non-zero bucket.
            ///
            /// Each `Bucket` contains the **individual** (non-cumulative) count.
            /// The quantile range `(lower, upper)` indicates the fraction of total
            /// observations before and up to this bucket.
            pub fn iter_with_quantiles(&self) -> $qr_iter<'_> {
                let total = self.count.last().map(|c| c.as_u128() as f64).unwrap_or(0.0);
                $qr_iter {
                    position: 0,
                    histogram: self,
                    total,
                }
            }

            /// Returns an iterator across the non-zero histogram buckets.
            ///
            /// Each `Bucket` contains the **individual** (non-cumulative) count for
            /// that bucket.
            pub fn iter(&self) -> $iter<'_> {
                $iter {
                    position: 0,
                    histogram: self,
                }
            }

            /// Compute quantiles for the given values.
            pub fn quantiles(&self, quantiles: &[f64]) -> Result<Option<QuantilesResult>, Error> {
                <Self as SampleQuantiles>::quantiles(self, quantiles)
            }

            /// Compute a single quantile.
            pub fn quantile(&self, quantile: f64) -> Result<Option<QuantilesResult>, Error> {
                <Self as SampleQuantiles>::quantile(self, quantile)
            }

            /// Returns the individual (non-cumulative) count at the given position.
            fn individual_count(&self, position: usize) -> u64 {
                if position == 0 {
                    self.count[0].as_u128() as u64
                } else {
                    self.count[position]
                        .wrapping_sub(self.count[position - 1])
                        .as_u128() as u64
                }
            }

            /// Find the first position where cumulative count >= target.
            /// Uses linear scan for small slices (fits in one cache line),
            /// binary search otherwise.
            fn find_quantile_position(&self, target: u128) -> usize {
                const CACHE_LINE_ENTRIES: usize = 64 / std::mem::size_of::<$count>();
                if self.count.len() <= CACHE_LINE_ENTRIES {
                    for (i, c) in self.count.iter().enumerate() {
                        if c.as_u128() >= target {
                            return i;
                        }
                    }
                    self.count.len() - 1
                } else {
                    let pos = self.count.partition_point(|c| c.as_u128() < target);
                    pos.min(self.count.len() - 1)
                }
            }
        }

        impl SampleQuantiles for $name {
            fn quantiles(&self, quantiles: &[f64]) -> Result<Option<QuantilesResult>, Error> {
                // Validate all quantile values
                for q in quantiles {
                    if !(0.0..=1.0).contains(q) {
                        return Err(Error::InvalidQuantile);
                    }
                }

                // Empty histogram
                if self.count.is_empty() {
                    return Ok(None);
                }

                let total_count = self.count.last().unwrap().as_u128();
                if total_count == 0 {
                    return Ok(None);
                }

                // Sort and dedup requested quantiles
                let mut sorted: Vec<Quantile> = quantiles
                    .iter()
                    .map(|&q| Quantile::new(q).unwrap())
                    .collect();
                sorted.sort();
                sorted.dedup();

                // min/max from first and last entries
                let min = Bucket {
                    count: self.individual_count(0),
                    range: self.config.index_to_range(self.index[0] as usize),
                };
                let last = self.count.len() - 1;
                let max = Bucket {
                    count: self.individual_count(last),
                    range: self.config.index_to_range(self.index[last] as usize),
                };

                // Find bucket for each quantile
                let mut entries = BTreeMap::new();
                for quantile in &sorted {
                    let target = std::cmp::max(
                        1u128,
                        (quantile.as_f64() * total_count as f64).ceil() as u128,
                    );

                    let pos = self.find_quantile_position(target);

                    entries.insert(
                        *quantile,
                        Bucket {
                            count: self.individual_count(pos),
                            range: self.config.index_to_range(self.index[pos] as usize),
                        },
                    );
                }

                Ok(Some(QuantilesResult::new(entries, total_count, min, max)))
            }
        }

        impl<'a> IntoIterator for &'a $name {
            type Item = Bucket;
            type IntoIter = $iter<'a>;

            fn into_iter(self) -> Self::IntoIter {
                self.iter()
            }
        }

        /// An iterator across the histogram buckets with individual counts.
        pub struct $iter<'a> {
            position: usize,
            histogram: &'a $name,
        }

        impl Iterator for $iter<'_> {
            type Item = Bucket;

            fn next(&mut self) -> Option<Bucket> {
                if self.position >= self.histogram.index.len() {
                    return None;
                }

                let i = self.position;
                let bucket = Bucket {
                    count: self.histogram.individual_count(i),
                    range: self
                        .histogram
                        .config
                        .index_to_range(self.histogram.index[i] as usize),
                };

                self.position += 1;
                Some(bucket)
            }
        }

        impl ExactSizeIterator for $iter<'_> {
            fn len(&self) -> usize {
                self.histogram.index.len() - self.position
            }
        }

        impl std::iter::FusedIterator for $iter<'_> {}

        /// An iterator yielding `(Bucket, lower_quantile, upper_quantile)` for each
        /// non-zero bucket.
        pub struct $qr_iter<'a> {
            position: usize,
            histogram: &'a $name,
            total: f64,
        }

        impl Iterator for $qr_iter<'_> {
            type Item = (Bucket, f64, f64);

            fn next(&mut self) -> Option<Self::Item> {
                if self.position >= self.histogram.index.len() {
                    return None;
                }

                let i = self.position;
                let lower = if i == 0 {
                    0.0
                } else {
                    self.histogram.count[i - 1].as_u128() as f64 / self.total
                };
                let upper = self.histogram.count[i].as_u128() as f64 / self.total;

                let bucket = Bucket {
                    count: self.histogram.individual_count(i),
                    range: self
                        .histogram
                        .config
                        .index_to_range(self.histogram.index[i] as usize),
                };

                self.position += 1;
                Some((bucket, lower, upper))
            }
        }

        impl ExactSizeIterator for $qr_iter<'_> {
            fn len(&self) -> usize {
                self.histogram.index.len() - self.position
            }
        }

        impl std::iter::FusedIterator for $qr_iter<'_> {}

        impl From<&$hist> for $name {
            fn from(histogram: &$hist) -> Self {
                let mut index = Vec::new();
                let mut count = Vec::new();
                let mut running_sum: $count = <$count as Count>::ZERO;

                for (idx, &n) in histogram.as_slice().iter().enumerate() {
                    if n != <$count as Count>::ZERO {
                        running_sum = running_sum.wrapping_add(n);
                        index.push(idx as u32);
                        count.push(running_sum);
                    }
                }

                Self {
                    config: histogram.config(),
                    index,
                    count,
                }
            }
        }

        impl From<&$sparse> for $name {
            fn from(histogram: &$sparse) -> Self {
                let mut running_sum: $count = <$count as Count>::ZERO;
                let cumulative: Vec<$count> = histogram
                    .count()
                    .iter()
                    .map(|&n| {
                        running_sum = running_sum.wrapping_add(n);
                        running_sum
                    })
                    .collect();

                Self {
                    config: histogram.config(),
                    index: histogram.index().to_vec(),
                    count: cumulative,
                }
            }
        }
    };
}

define_cumulative_histogram!(
    CumulativeROHistogram,
    CumulativeIter,
    QuantileRangeIter,
    Histogram,
    SparseHistogram,
    u64
);
define_cumulative_histogram!(
    CumulativeROHistogram32,
    CumulativeIter32,
    QuantileRangeIter32,
    Histogram32,
    SparseHistogram32,
    u32
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_histogram() {
        let mut h = Histogram::new(7, 64).unwrap();
        h.increment(1).unwrap();
        h.increment(1).unwrap();
        h.increment(5).unwrap();
        h.increment(100).unwrap();

        let croh = CumulativeROHistogram::from(&h);
        assert_eq!(croh.config(), h.config());
        // Three distinct buckets: index 1 (count 2), index 5 (count 1), index 100 (count 1)
        assert_eq!(croh.index().len(), 3);
        // Cumulative: [2, 3, 4]
        assert_eq!(croh.count(), &[2, 3, 4]);
        assert_eq!(croh.total_count(), 4);
    }

    #[test]
    fn from_sparse() {
        let config = Config::new(7, 32).unwrap();
        let sparse = SparseHistogram::from_parts(config, vec![1, 3, 5], vec![6, 12, 7]).unwrap();

        let croh = CumulativeROHistogram::from(&sparse);
        assert_eq!(croh.config(), config);
        assert_eq!(croh.index(), &[1, 3, 5]);
        // Cumulative: [6, 18, 25]
        assert_eq!(croh.count(), &[6, 18, 25]);
        assert_eq!(croh.total_count(), 25);
    }

    #[test]
    fn quantiles_match_histogram() {
        let mut h = Histogram::new(4, 10).unwrap();
        for v in 1..1024 {
            h.increment(v).unwrap();
        }

        let sparse = SparseHistogram::from(&h);
        let croh = CumulativeROHistogram::from(&h);

        let quantiles = &[0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 1.0];

        let hr = h.quantiles(quantiles).unwrap().unwrap();
        let sr = sparse.quantiles(quantiles).unwrap().unwrap();
        let cr = croh.quantiles(quantiles).unwrap().unwrap();

        assert_eq!(hr.total_count(), cr.total_count());
        assert_eq!(sr.total_count(), cr.total_count());
        assert_eq!(hr.min().range(), cr.min().range());
        assert_eq!(hr.max().range(), cr.max().range());

        for ((hq, sq), cq) in hr
            .entries()
            .iter()
            .zip(sr.entries().iter())
            .zip(cr.entries().iter())
        {
            assert_eq!(hq.0, cq.0);
            assert_eq!(sq.0, cq.0);
            assert_eq!(hq.1.range(), cq.1.range());
            assert_eq!(sq.1.range(), cq.1.range());
            assert_eq!(hq.1.count(), cq.1.count());
        }
    }

    #[test]
    fn empty_histogram() {
        let h = Histogram::new(7, 64).unwrap();
        let croh = CumulativeROHistogram::from(&h);

        assert!(croh.is_empty());
        assert_eq!(croh.len(), 0);
        assert_eq!(croh.total_count(), 0);
        assert_eq!(croh.quantiles(&[0.5]).unwrap(), None);
        assert_eq!(croh.quantile(0.5).unwrap(), None);
    }

    #[test]
    fn single_sample() {
        let mut h = Histogram::new(7, 64).unwrap();
        h.increment(42).unwrap();

        let croh = CumulativeROHistogram::from(&h);
        assert_eq!(croh.len(), 1);
        assert_eq!(croh.total_count(), 1);

        let result = croh.quantile(0.0).unwrap().unwrap();
        assert_eq!(result.min().end(), 42);

        let result = croh.quantile(1.0).unwrap().unwrap();
        assert_eq!(result.max().end(), 42);

        let result = croh.quantile(0.5).unwrap().unwrap();
        let q = Quantile::new(0.5).unwrap();
        assert_eq!(result.get(&q).unwrap().end(), 42);
    }

    #[test]
    fn from_parts_validation() {
        let config = Config::new(7, 32).unwrap();

        // Mismatched lengths
        assert_eq!(
            CumulativeROHistogram::from_parts(config, vec![1, 2], vec![1]),
            Err(Error::IncompatibleParameters)
        );

        // Out of range index
        assert_eq!(
            CumulativeROHistogram::from_parts(config, vec![u32::MAX], vec![1]),
            Err(Error::OutOfRange)
        );

        // Non-ascending indices
        assert_eq!(
            CumulativeROHistogram::from_parts(config, vec![3, 1], vec![1, 2]),
            Err(Error::IncompatibleParameters)
        );

        // Duplicate indices
        assert_eq!(
            CumulativeROHistogram::from_parts(config, vec![1, 1], vec![1, 2]),
            Err(Error::IncompatibleParameters)
        );

        // Non-non-decreasing counts
        assert_eq!(
            CumulativeROHistogram::from_parts(config, vec![1, 3], vec![5, 3]),
            Err(Error::IncompatibleParameters)
        );

        // Zero count
        assert_eq!(
            CumulativeROHistogram::from_parts(config, vec![1], vec![0]),
            Err(Error::IncompatibleParameters)
        );

        // Valid
        assert!(CumulativeROHistogram::from_parts(config, vec![1, 3, 5], vec![6, 18, 25]).is_ok());

        // Empty is valid
        assert!(CumulativeROHistogram::from_parts(config, vec![], vec![]).is_ok());
    }

    #[test]
    fn quantile_ranges() {
        let config = Config::new(7, 32).unwrap();
        // 3 buckets with individual counts 10, 30, 60 → cumulative [10, 40, 100]
        let croh =
            CumulativeROHistogram::from_parts(config, vec![1, 3, 5], vec![10, 40, 100]).unwrap();

        // Bucket 0: [0.0, 0.1)
        let (lo, hi) = croh.bucket_quantile_range(0).unwrap();
        assert!((lo - 0.0).abs() < f64::EPSILON);
        assert!((hi - 0.1).abs() < f64::EPSILON);

        // Bucket 1: [0.1, 0.4)
        let (lo, hi) = croh.bucket_quantile_range(1).unwrap();
        assert!((lo - 0.1).abs() < f64::EPSILON);
        assert!((hi - 0.4).abs() < f64::EPSILON);

        // Bucket 2: [0.4, 1.0]
        let (lo, hi) = croh.bucket_quantile_range(2).unwrap();
        assert!((lo - 0.4).abs() < f64::EPSILON);
        assert!((hi - 1.0).abs() < f64::EPSILON);

        // Out of range
        assert_eq!(croh.bucket_quantile_range(3), None);

        // Empty histogram
        let empty = CumulativeROHistogram::from_parts(config, vec![], vec![]).unwrap();
        assert_eq!(empty.bucket_quantile_range(0), None);
    }

    #[test]
    fn iter_with_quantiles() {
        let config = Config::new(7, 32).unwrap();
        let croh =
            CumulativeROHistogram::from_parts(config, vec![1, 3, 5], vec![10, 40, 100]).unwrap();

        let items: Vec<_> = croh.iter_with_quantiles().collect();
        assert_eq!(items.len(), 3);

        // Check individual counts
        assert_eq!(items[0].0.count(), 10);
        assert_eq!(items[1].0.count(), 30);
        assert_eq!(items[2].0.count(), 60);

        // Check quantile ranges
        assert!((items[0].1 - 0.0).abs() < f64::EPSILON);
        assert!((items[0].2 - 0.1).abs() < f64::EPSILON);
        assert!((items[1].1 - 0.1).abs() < f64::EPSILON);
        assert!((items[1].2 - 0.4).abs() < f64::EPSILON);
        assert!((items[2].1 - 0.4).abs() < f64::EPSILON);
        assert!((items[2].2 - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn iter_individual_counts() {
        let mut h = Histogram::new(7, 64).unwrap();
        h.increment(1).unwrap();
        h.increment(1).unwrap();
        h.increment(5).unwrap();
        h.increment(100).unwrap();

        let sparse = SparseHistogram::from(&h);
        let croh = CumulativeROHistogram::from(&h);

        let sparse_buckets: Vec<_> = sparse.iter().collect();
        let croh_buckets: Vec<_> = croh.iter().collect();

        assert_eq!(sparse_buckets.len(), croh_buckets.len());
        for (sb, cb) in sparse_buckets.iter().zip(croh_buckets.iter()) {
            assert_eq!(sb.count(), cb.count());
            assert_eq!(sb.range(), cb.range());
        }
    }

    #[test]
    fn into_parts_roundtrip() {
        let config = Config::new(7, 32).unwrap();
        let original =
            CumulativeROHistogram::from_parts(config, vec![1, 3, 5], vec![6, 18, 25]).unwrap();

        let (cfg, idx, cnt) = original.clone().into_parts();
        let reconstructed = CumulativeROHistogram::from_parts(cfg, idx, cnt).unwrap();

        assert_eq!(original, reconstructed);
    }

    #[test]
    fn invalid_quantile_returns_error() {
        let config = Config::new(7, 32).unwrap();
        let croh = CumulativeROHistogram::from_parts(config, vec![1], vec![5]).unwrap();

        assert_eq!(croh.quantiles(&[1.5]), Err(Error::InvalidQuantile));
        assert_eq!(croh.quantiles(&[-0.1]), Err(Error::InvalidQuantile));
    }

    #[test]
    fn from_histogram_u32() {
        let mut h = Histogram32::new(7, 64).unwrap();
        h.increment(1).unwrap();
        h.increment(1).unwrap();
        h.increment(5).unwrap();
        h.increment(100).unwrap();
        let croh = CumulativeROHistogram32::from(&h);
        assert_eq!(croh.index().len(), 3);
        assert_eq!(croh.count(), &[2u32, 3, 4]);
        assert_eq!(croh.total_count(), 4);
    }

    #[test]
    fn from_parts_u32() {
        let config = Config::new(7, 32).unwrap();
        let croh =
            CumulativeROHistogram32::from_parts(config, vec![1, 3, 5], vec![6u32, 18, 25]).unwrap();
        assert_eq!(croh.total_count(), 25);
    }

    #[test]
    fn quantiles_u32_match_u64() {
        let mut h32 = Histogram32::new(4, 10).unwrap();
        let mut h64 = Histogram::new(4, 10).unwrap();
        for v in 1..1024u64 {
            h32.increment(v).unwrap();
            h64.increment(v).unwrap();
        }
        let c32 = CumulativeROHistogram32::from(&h32);
        let c64 = CumulativeROHistogram::from(&h64);
        let qs = &[0.0, 0.5, 0.99, 1.0];
        let r32 = c32.quantiles(qs).unwrap().unwrap();
        let r64 = c64.quantiles(qs).unwrap().unwrap();
        for ((q32, _), (q64, _)) in r32.entries().iter().zip(r64.entries().iter()) {
            assert_eq!(q32, q64);
        }
    }
}
