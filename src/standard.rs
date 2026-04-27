use std::collections::BTreeMap;

use crate::quantile::{Quantile, QuantilesResult, SampleQuantiles};
use crate::{Bucket, Config, Count, Error, SparseHistogram};
// SparseHistogram32 import is added in Task 4 when Histogram32 is uncommented.

macro_rules! define_histogram {
    ($name:ident, $iter:ident, $sparse:ident, $count:ty) => {
        /// A histogram that uses plain counters for each bucket.
        #[derive(Clone, Debug, PartialEq, Eq)]
        #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
        #[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
        pub struct $name {
            pub(crate) config: Config,
            pub(crate) buckets: Box<[$count]>,
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
        }

        impl SampleQuantiles for $name {
            fn quantiles(&self, quantiles: &[f64]) -> Result<Option<QuantilesResult>, Error> {
                for q in quantiles {
                    if !(0.0..=1.0).contains(q) {
                        return Err(Error::InvalidQuantile);
                    }
                }
                let total_count: u128 = self.buckets.iter().map(|v| v.as_u128()).sum();
                if total_count == 0 {
                    return Ok(None);
                }
                let mut sorted: Vec<Quantile> = quantiles
                    .iter()
                    .map(|&q| Quantile::new(q).unwrap())
                    .collect();
                sorted.sort();
                sorted.dedup();

                let mut min_idx = None;
                let mut max_idx = None;
                for (i, count) in self.buckets.iter().enumerate() {
                    if *count != <$count as Count>::ZERO {
                        if min_idx.is_none() {
                            min_idx = Some(i);
                        }
                        max_idx = Some(i);
                    }
                }
                let min_idx = min_idx.unwrap();
                let max_idx = max_idx.unwrap();

                let min = Bucket {
                    count: self.buckets[min_idx].as_u128() as u64,
                    range: self.config.index_to_range(min_idx),
                };
                let max = Bucket {
                    count: self.buckets[max_idx].as_u128() as u64,
                    range: self.config.index_to_range(max_idx),
                };

                let mut bucket_idx = 0;
                let mut partial_sum = self.buckets[bucket_idx].as_u128();
                let mut entries = BTreeMap::new();

                for quantile in &sorted {
                    let count =
                        std::cmp::max(1, (quantile.as_f64() * total_count as f64).ceil() as u128);
                    loop {
                        if partial_sum >= count {
                            entries.insert(
                                *quantile,
                                Bucket {
                                    count: self.buckets[bucket_idx].as_u128() as u64,
                                    range: self.config.index_to_range(bucket_idx),
                                },
                            );
                            break;
                        }
                        if bucket_idx == (self.buckets.len() - 1) {
                            break;
                        }
                        bucket_idx += 1;
                        partial_sum += self.buckets[bucket_idx].as_u128();
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
// define_histogram!(Histogram32, Iter32, SparseHistogram32, u32);  // uncommented in Task 4

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
}
