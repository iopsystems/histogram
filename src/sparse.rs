use std::collections::BTreeMap;

use crate::quantile::{Quantile, QuantilesResult, SampleQuantiles};
use crate::{Bucket, Config, Count, Error, Histogram, Histogram32};

macro_rules! define_sparse_histogram {
    ($name:ident, $iter:ident, $hist:ident, $count:ty) => {
        /// A sparse, columnar representation of a histogram.
        ///
        /// Significantly smaller than the dense form when many buckets are
        /// zero. Each non-zero bucket is stored as a pair `(index[i],
        /// count[i])` where `index[i]` is the bucket index and `count[i]`
        /// is its count, in ascending index order.
        #[derive(Clone, Debug, PartialEq, Eq)]
        #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
        #[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
        pub struct $name {
            pub(crate) config: Config,
            pub(crate) index: Vec<u32>,
            pub(crate) count: Vec<$count>,
        }

        impl $name {
            /// Construct a new histogram from the provided parameters.
            pub fn new(grouping_power: u8, max_value_power: u8) -> Result<Self, Error> {
                let config = Config::new(grouping_power, max_value_power)?;
                Ok(Self::with_config(&config))
            }

            /// Creates a new histogram using a provided [`crate::Config`].
            pub fn with_config(config: &Config) -> Self {
                Self {
                    config: *config,
                    index: Vec::new(),
                    count: Vec::new(),
                }
            }

            /// Creates a sparse histogram from its raw parts.
            ///
            /// Returns an error if:
            /// - `index` and `count` have different lengths
            /// - any index is out of range for the config
            /// - the indices are not in strictly ascending order
            pub fn from_parts(
                config: Config,
                index: Vec<u32>,
                count: Vec<$count>,
            ) -> Result<Self, Error> {
                if index.len() != count.len() {
                    return Err(Error::IncompatibleParameters);
                }

                let total_buckets = config.total_buckets();
                let mut prev = None;
                for &idx in &index {
                    if idx as usize >= total_buckets {
                        return Err(Error::OutOfRange);
                    }
                    if let Some(p) = prev {
                        if idx <= p {
                            return Err(Error::IncompatibleParameters);
                        }
                    }
                    prev = Some(idx);
                }

                for &c in &count {
                    if c == <$count as Count>::ZERO {
                        return Err(Error::IncompatibleParameters);
                    }
                }

                Ok(Self {
                    config,
                    index,
                    count,
                })
            }

            /// Consumes the histogram, returning the config, index, and count vectors.
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

            /// Returns a slice of the bucket counts.
            pub fn count(&self) -> &[$count] {
                &self.count
            }

            /// Helper function to store a bucket in the histogram.
            fn add_bucket(&mut self, idx: u32, n: $count) {
                if n != <$count as Count>::ZERO {
                    self.index.push(idx);
                    self.count.push(n);
                }
            }

            /// Adds the other histogram to this histogram and returns the result as a
            /// new histogram.
            ///
            /// Returns `Err(Error::IncompatibleParameters)` if the configs don't match,
            /// or `Err(Error::Overflow)` if any bucket overflows.
            #[allow(clippy::comparison_chain)]
            pub fn checked_add(&self, h: &Self) -> Result<Self, Error> {
                if self.config != h.config {
                    return Err(Error::IncompatibleParameters);
                }

                let mut histogram = Self::with_config(&self.config);

                let (mut i, mut j) = (0, 0);
                while i < self.index.len() && j < h.index.len() {
                    let (k1, v1) = (self.index[i], self.count[i]);
                    let (k2, v2) = (h.index[j], h.count[j]);

                    if k1 == k2 {
                        let v = v1.checked_add(v2).ok_or(Error::Overflow)?;
                        histogram.add_bucket(k1, v);
                        (i, j) = (i + 1, j + 1);
                    } else if k1 < k2 {
                        histogram.add_bucket(k1, v1);
                        i += 1;
                    } else {
                        histogram.add_bucket(k2, v2);
                        j += 1;
                    }
                }

                if i < self.index.len() {
                    histogram.index.extend(&self.index[i..]);
                    histogram.count.extend(&self.count[i..]);
                }

                if j < h.index.len() {
                    histogram.index.extend(&h.index[j..]);
                    histogram.count.extend(&h.count[j..]);
                }

                Ok(histogram)
            }

            /// Adds the other histogram to this histogram and returns the result as a
            /// new histogram.
            ///
            /// Returns `Err(Error::IncompatibleParameters)` if the configs don't match.
            /// Buckets which have values in both histograms are allowed to wrap.
            #[allow(clippy::comparison_chain)]
            pub fn wrapping_add(&self, h: &Self) -> Result<Self, Error> {
                if self.config != h.config {
                    return Err(Error::IncompatibleParameters);
                }

                let mut histogram = Self::with_config(&self.config);

                // Sort and merge buckets from both histograms
                let (mut i, mut j) = (0, 0);
                while i < self.index.len() && j < h.index.len() {
                    let (k1, v1) = (self.index[i], self.count[i]);
                    let (k2, v2) = (h.index[j], h.count[j]);

                    if k1 == k2 {
                        histogram.add_bucket(k1, v1.wrapping_add(v2));
                        (i, j) = (i + 1, j + 1);
                    } else if k1 < k2 {
                        histogram.add_bucket(k1, v1);
                        i += 1;
                    } else {
                        histogram.add_bucket(k2, v2);
                        j += 1;
                    }
                }

                // Fill remaining values, if any, from the left histogram
                if i < self.index.len() {
                    histogram.index.extend(&self.index[i..self.index.len()]);
                    histogram.count.extend(&self.count[i..self.count.len()]);
                }

                // Fill remaining values, if any, from the right histogram
                if j < h.index.len() {
                    histogram.index.extend(&h.index[j..h.index.len()]);
                    histogram.count.extend(&h.count[j..h.count.len()]);
                }

                Ok(histogram)
            }

            /// Subtracts the other histogram from this histogram and returns the result as a
            /// new histogram.
            ///
            /// Returns `Err(Error::IncompatibleParameters)` if the configs don't match,
            /// `Err(Error::InvalidSubset)` if the other histogram has buckets not present in
            /// this one, or `Err(Error::Underflow)` if any bucket would underflow.
            #[allow(clippy::comparison_chain)]
            pub fn checked_sub(&self, h: &Self) -> Result<Self, Error> {
                if self.config != h.config {
                    return Err(Error::IncompatibleParameters);
                }

                let mut histogram = Self::with_config(&self.config);

                // Sort and merge buckets from both histograms
                let (mut i, mut j) = (0, 0);
                while i < self.index.len() && j < h.index.len() {
                    let (k1, v1) = (self.index[i], self.count[i]);
                    let (k2, v2) = (h.index[j], h.count[j]);

                    if k1 == k2 {
                        let v = v1.checked_sub(v2).ok_or(Error::Underflow)?;
                        if v != <$count as Count>::ZERO {
                            histogram.add_bucket(k1, v);
                        }
                        (i, j) = (i + 1, j + 1);
                    } else if k1 < k2 {
                        histogram.add_bucket(k1, v1);
                        i += 1;
                    } else {
                        // Other histogram has a bucket not present in this histogram
                        return Err(Error::InvalidSubset);
                    }
                }

                // Check that the subset histogram has been consumed
                if j < h.index.len() {
                    return Err(Error::InvalidSubset);
                }

                // Fill remaining buckets, if any, from the superset histogram
                if i < self.index.len() {
                    histogram.index.extend(&self.index[i..self.index.len()]);
                    histogram.count.extend(&self.count[i..self.count.len()]);
                }

                Ok(histogram)
            }

            /// Subtracts the other histogram from this histogram and returns the result
            /// as a new histogram.
            ///
            /// Returns `Err(Error::IncompatibleParameters)` if the configs don't match,
            /// or `Err(Error::InvalidSubset)` if the other histogram has buckets not
            /// present in this one. Buckets are allowed to wrap on underflow.
            #[allow(clippy::comparison_chain)]
            pub fn wrapping_sub(&self, h: &Self) -> Result<Self, Error> {
                if self.config != h.config {
                    return Err(Error::IncompatibleParameters);
                }

                let mut histogram = Self::with_config(&self.config);

                let (mut i, mut j) = (0, 0);
                while i < self.index.len() && j < h.index.len() {
                    let (k1, v1) = (self.index[i], self.count[i]);
                    let (k2, v2) = (h.index[j], h.count[j]);

                    if k1 == k2 {
                        histogram.add_bucket(k1, v1.wrapping_sub(v2));
                        (i, j) = (i + 1, j + 1);
                    } else if k1 < k2 {
                        histogram.add_bucket(k1, v1);
                        i += 1;
                    } else {
                        return Err(Error::InvalidSubset);
                    }
                }

                if i < self.index.len() {
                    histogram.index.extend(&self.index[i..]);
                    histogram.count.extend(&self.count[i..]);
                }

                if j < h.index.len() {
                    return Err(Error::InvalidSubset);
                }

                Ok(histogram)
            }

            /// Returns a new histogram with a reduced grouping power.
            ///
            /// Returns an error if the requested grouping power is not less than the current
            /// grouping power.
            pub fn downsample(&self, grouping_power: u8) -> Result<Self, Error> {
                if grouping_power >= self.config.grouping_power() {
                    return Err(Error::IncompatibleParameters);
                }

                let config = Config::new(grouping_power, self.config.max_value_power())?;
                let mut histogram = Self::with_config(&config);

                let mut aggregating_idx: u32 = 0;
                let mut aggregating_count: $count = <$count as Count>::ZERO;
                for (idx, n) in self.index.iter().zip(self.count.iter()) {
                    let new_idx = config
                        .value_to_index(self.config.index_to_lower_bound(*idx as usize))?
                        as u32;

                    if new_idx == aggregating_idx {
                        aggregating_count = aggregating_count.wrapping_add(*n);
                        continue;
                    }

                    histogram.add_bucket(aggregating_idx, aggregating_count);
                    aggregating_idx = new_idx;
                    aggregating_count = *n;
                }

                histogram.add_bucket(aggregating_idx, aggregating_count);

                Ok(histogram)
            }

            /// Returns an iterator across the non-zero histogram buckets.
            pub fn iter(&self) -> $iter<'_> {
                $iter {
                    index: 0,
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
        }

        impl SampleQuantiles for $name {
            fn quantiles(&self, quantiles: &[f64]) -> Result<Option<QuantilesResult>, Error> {
                // validate all the quantiles
                for q in quantiles {
                    if !(0.0..=1.0).contains(q) {
                        return Err(Error::InvalidQuantile);
                    }
                }

                // get the total count
                let total_count: u128 = self.count.iter().map(|v| v.as_u128()).sum();

                // empty histogram, no quantiles available
                if total_count == 0 {
                    return Ok(None);
                }

                // sort the requested quantiles so we can find them in a single pass
                let mut sorted: Vec<Quantile> = quantiles
                    .iter()
                    .map(|&q| Quantile::new(q).unwrap())
                    .collect();
                sorted.sort();
                sorted.dedup();

                // min/max are the first and last entries in the sparse vectors
                let min = Bucket {
                    count: self.count[0].as_u128() as u64,
                    range: self.config.index_to_range(self.index[0] as usize),
                };
                let last = self.index.len() - 1;
                let max = Bucket {
                    count: self.count[last].as_u128() as u64,
                    range: self.config.index_to_range(self.index[last] as usize),
                };

                // single pass to find all quantile buckets
                let mut idx = 0;
                let mut partial_sum = self.count[0].as_u128();

                let mut entries = BTreeMap::new();

                for quantile in &sorted {
                    let count =
                        std::cmp::max(1, (quantile.as_f64() * total_count as f64).ceil() as u128);

                    loop {
                        if partial_sum >= count {
                            entries.insert(
                                *quantile,
                                Bucket {
                                    count: self.count[idx].as_u128() as u64,
                                    range: self.config.index_to_range(self.index[idx] as usize),
                                },
                            );
                            break;
                        }

                        if idx == (self.index.len() - 1) {
                            break;
                        }

                        idx += 1;
                        partial_sum += self.count[idx].as_u128();
                    }
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

            fn next(&mut self) -> Option<<Self as std::iter::Iterator>::Item> {
                if self.index >= self.histogram.index.len() {
                    return None;
                }

                let bucket = Bucket {
                    count: self.histogram.count[self.index].as_u128() as u64,
                    range: self
                        .histogram
                        .config
                        .index_to_range(self.histogram.index[self.index] as usize),
                };

                self.index += 1;

                Some(bucket)
            }
        }

        impl ExactSizeIterator for $iter<'_> {
            fn len(&self) -> usize {
                self.histogram.index.len() - self.index
            }
        }

        impl std::iter::FusedIterator for $iter<'_> {}

        impl From<&$hist> for $name {
            fn from(histogram: &$hist) -> Self {
                let mut index = Vec::new();
                let mut count = Vec::new();

                for (idx, n) in histogram.as_slice().iter().enumerate() {
                    if *n != <$count as Count>::ZERO {
                        index.push(idx as u32);
                        count.push(*n);
                    }
                }

                Self {
                    config: histogram.config(),
                    index,
                    count,
                }
            }
        }
    };
}

define_sparse_histogram!(SparseHistogram, SparseIter, Histogram, u64);
define_sparse_histogram!(SparseHistogram32, SparseIter32, Histogram32, u32);

// Deprecated forwarding methods — only on the u64 variant to avoid proliferating
// deprecated APIs onto the new u32 type.
impl SparseHistogram {
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
    use rand::RngExt;
    use std::collections::HashMap;

    use super::*;
    use crate::standard::Histogram;

    #[test]
    fn checked_add() {
        let config = Config::new(7, 32).unwrap();

        let h1 = SparseHistogram::from_parts(config, vec![1, 3, 5], vec![6, 12, 7]).unwrap();
        let h2 = SparseHistogram::from_parts(config, vec![2, 3], vec![5, 7]).unwrap();

        let h = h1.checked_add(&h2).unwrap();
        assert_eq!(h.index(), &[1, 2, 3, 5]);
        assert_eq!(h.count(), &[6, 5, 19, 7]);

        // overflow
        let h_max = SparseHistogram::from_parts(config, vec![3], vec![u64::MAX]).unwrap();
        assert_eq!(h1.checked_add(&h_max), Err(Error::Overflow));
    }

    #[test]
    fn wrapping_add() {
        let config = Config::new(7, 32).unwrap();

        let h1 = SparseHistogram::from_parts(config, vec![1, 3, 5], vec![6, 12, 7]).unwrap();
        let h2 = SparseHistogram::with_config(&config);
        let h3 = SparseHistogram::from_parts(config, vec![2, 3, 6, 11, 13], vec![5, 7, 3, 15, 6])
            .unwrap();

        let hdiff = SparseHistogram::new(6, 16).unwrap();
        let h = h1.wrapping_add(&hdiff);
        assert_eq!(h, Err(Error::IncompatibleParameters));

        let h = h1.wrapping_add(&h2).unwrap();
        assert_eq!(h.index(), &[1, 3, 5]);
        assert_eq!(h.count(), &[6, 12, 7]);

        let h = h2.wrapping_add(&h3).unwrap();
        assert_eq!(h.index(), &[2, 3, 6, 11, 13]);
        assert_eq!(h.count(), &[5, 7, 3, 15, 6]);

        let h = h1.wrapping_add(&h3).unwrap();
        assert_eq!(h.index(), &[1, 2, 3, 5, 6, 11, 13]);
        assert_eq!(h.count(), &[6, 5, 19, 7, 3, 15, 6]);
    }

    #[test]
    fn checked_sub() {
        let config = Config::new(7, 32).unwrap();

        let h1 = SparseHistogram::from_parts(config, vec![1, 3, 5], vec![6, 12, 7]).unwrap();

        let hparams = SparseHistogram::new(6, 16).unwrap();
        let h = h1.checked_sub(&hparams);
        assert_eq!(h, Err(Error::IncompatibleParameters));

        let hempty = SparseHistogram::with_config(&config);
        let h = h1.checked_sub(&hempty).unwrap();
        assert_eq!(h.index(), &[1, 3, 5]);
        assert_eq!(h.count(), &[6, 12, 7]);

        let hclone = h1.clone();
        let h = h1.checked_sub(&hclone).unwrap();
        assert!(h.index().is_empty());
        assert!(h.count().is_empty());

        let hlarger = SparseHistogram::from_parts(config, vec![1, 3, 5], vec![4, 13, 7]).unwrap();
        let h = h1.checked_sub(&hlarger);
        assert_eq!(h, Err(Error::Underflow));

        let hmore = SparseHistogram::from_parts(config, vec![1, 5, 7], vec![4, 7, 1]).unwrap();
        let h = h1.checked_sub(&hmore);
        assert_eq!(h, Err(Error::InvalidSubset));

        let hdiff = SparseHistogram::from_parts(config, vec![1, 2, 5], vec![4, 1, 7]).unwrap();
        let h = h1.checked_sub(&hdiff);
        assert_eq!(h, Err(Error::InvalidSubset));

        let hsubset = SparseHistogram::from_parts(config, vec![1, 3], vec![5, 9]).unwrap();
        let h = h1.checked_sub(&hsubset).unwrap();
        assert_eq!(h.index(), &[1, 3, 5]);
        assert_eq!(h.count(), &[1, 3, 7]);
    }

    #[test]
    fn wrapping_sub() {
        let config = Config::new(7, 32).unwrap();

        let h1 = SparseHistogram::from_parts(config, vec![1, 3, 5], vec![6, 12, 7]).unwrap();
        let h2 = SparseHistogram::from_parts(config, vec![1, 3], vec![4, 5]).unwrap();

        let h = h1.wrapping_sub(&h2).unwrap();
        assert_eq!(h.index(), &[1, 3, 5]);
        assert_eq!(h.count(), &[2, 7, 7]);

        // wrapping underflow
        let h3 = SparseHistogram::from_parts(config, vec![1], vec![10]).unwrap();
        let h = h1.wrapping_sub(&h3).unwrap();
        assert_eq!(h.count()[0], 6u64.wrapping_sub(10)); // wraps

        // non-subset returns error
        let h4 = SparseHistogram::from_parts(config, vec![2], vec![1]).unwrap();
        assert_eq!(h1.wrapping_sub(&h4), Err(Error::InvalidSubset));
    }

    #[test]
    fn wrapping_add_overflow() {
        let config = Config::new(7, 32).unwrap();
        let h1 = SparseHistogram::from_parts(config, vec![1], vec![u64::MAX]).unwrap();
        let h2 = SparseHistogram::from_parts(config, vec![1], vec![1]).unwrap();
        let h = h1.wrapping_add(&h2).unwrap();
        // u64::MAX + 1 wraps to 0, add_bucket skips zero-count entries
        assert!(h.index().is_empty());
    }

    #[allow(deprecated)]
    #[test]
    fn percentiles() {
        let mut hstandard = Histogram::new(4, 10).unwrap();
        let hempty = SparseHistogram::from(&hstandard);

        for v in 1..1024 {
            let _ = hstandard.increment(v);
        }

        let hsparse = SparseHistogram::from(&hstandard);
        let percentiles = [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 0.999];
        for percentile in percentiles {
            let bempty = hempty.percentile(percentile).unwrap();
            // Use quantile() on hstandard (deprecated percentile() removed)
            let bstandard = hstandard
                .quantile(percentile)
                .unwrap()
                .map(|r| r.get(&Quantile::new(percentile).unwrap()).unwrap().clone());
            let bsparse = hsparse.percentile(percentile).unwrap();

            assert_eq!(bempty, None);
            assert_eq!(bsparse, bstandard);
        }

        assert_eq!(hempty.percentiles(&percentiles), Ok(None));
        // Compare sparse percentiles against standard quantiles
        let sparse_result = hsparse.percentiles(&percentiles).unwrap().unwrap();
        for (p, bucket) in &sparse_result {
            let q = hstandard.quantile(*p).unwrap().unwrap();
            let standard_bucket = q.get(&Quantile::new(*p).unwrap()).unwrap();
            assert_eq!(bucket, standard_bucket);
        }
    }

    #[allow(deprecated)]
    #[test]
    // Tests percentile used to find min
    fn min() {
        let mut histogram = Histogram::new(7, 64).unwrap();

        let h = SparseHistogram::from(&histogram);
        assert_eq!(h.percentile(0.0).unwrap(), None);

        let _ = histogram.increment(10);
        let h = SparseHistogram::from(&histogram);
        assert_eq!(h.percentile(0.0).map(|b| b.unwrap().end()), Ok(10));

        let _ = histogram.increment(4);
        let h = SparseHistogram::from(&histogram);
        assert_eq!(h.percentile(0.0).map(|b| b.unwrap().end()), Ok(4));
    }

    fn compare_histograms(hstandard: &Histogram, hsparse: &SparseHistogram) {
        assert_eq!(hstandard.config(), hsparse.config());

        let mut buckets: HashMap<u32, u64> = HashMap::new();
        for (idx, count) in hsparse.index().iter().zip(hsparse.count().iter()) {
            let _ = buckets.insert(*idx, *count);
        }

        for (idx, count) in hstandard.as_slice().iter().enumerate() {
            if *count > 0 {
                let v = buckets.get(&(idx as u32)).unwrap();
                assert_eq!(*v, *count);
            }
        }
    }

    #[test]
    fn snapshot() {
        let mut hstandard = Histogram::new(5, 10).unwrap();

        for v in 1..1024 {
            let _ = hstandard.increment(v);
        }

        // Convert to sparse and store buckets in a hash for random lookup
        let hsparse = SparseHistogram::from(&hstandard);
        compare_histograms(&hstandard, &hsparse);
    }

    #[test]
    fn downsample() {
        let mut histogram = Histogram::new(8, 32).unwrap();
        let mut rng = rand::rng();

        // Generate 10,000 values to store in a sorted array and a histogram
        for _ in 0..10000 {
            let v: u64 = rng.random_range(1..2_u64.pow(histogram.config.max_value_power() as u32));
            let _ = histogram.increment(v);
        }

        let hsparse = SparseHistogram::from(&histogram);
        compare_histograms(&histogram, &hsparse);

        // Downsample and check the percentiles lie within error margin
        let grouping_power = histogram.config.grouping_power();
        for factor in 1..grouping_power {
            let reduced_gp = grouping_power - factor;
            let h1 = histogram.downsample(reduced_gp).unwrap();
            let h2 = hsparse.downsample(reduced_gp).unwrap();
            compare_histograms(&h1, &h2);
        }
    }

    // ===== new u32-targeted tests =====

    #[test]
    fn from_parts_u32() {
        let config = Config::new(7, 32).unwrap();
        let h = SparseHistogram32::from_parts(config, vec![1, 3, 5], vec![6u32, 12, 7]).unwrap();
        assert_eq!(h.index(), &[1, 3, 5]);
        assert_eq!(h.count(), &[6u32, 12, 7]);
    }

    #[test]
    fn checked_add_u32_overflow() {
        let config = Config::new(7, 32).unwrap();
        let h1 = SparseHistogram32::from_parts(config, vec![1], vec![u32::MAX]).unwrap();
        let h2 = SparseHistogram32::from_parts(config, vec![1], vec![1u32]).unwrap();
        assert_eq!(h1.checked_add(&h2), Err(Error::Overflow));
    }

    #[test]
    fn from_histogram_u32() {
        use crate::standard::Histogram32;
        let mut h = Histogram32::new(7, 64).unwrap();
        h.increment(1).unwrap();
        h.increment(5).unwrap();
        h.increment(100).unwrap();
        let s = SparseHistogram32::from(&h);
        assert_eq!(s.count().len(), 3);
    }
}
