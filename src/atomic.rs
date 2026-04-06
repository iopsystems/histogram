use crate::{Config, Error, Histogram};
use core::sync::atomic::{AtomicU64, Ordering};

/// A histogram that uses atomic 64bit counters for each bucket.
///
/// Unlike the non-atomic variant, it cannot be used directly to report
/// percentiles. Instead, a snapshot must be taken which captures the state of
/// the histogram at a point in time.
pub struct AtomicHistogram {
    config: Config,
    buckets: Box<[AtomicU64]>,
}

impl AtomicHistogram {
    /// Construct a new atomic histogram from the provided parameters. See the
    /// documentation for [`crate::Config`] to understand their meaning.
    pub fn new(grouping_power: u8, max_value_power: u8) -> Result<Self, Error> {
        let config = Config::new(grouping_power, max_value_power)?;

        Ok(Self::with_config(&config))
    }

    /// Creates a new atomic histogram using a provided [`crate::Config`].
    pub fn with_config(config: &Config) -> Self {
        let mut buckets = Vec::with_capacity(config.total_buckets());
        buckets.resize_with(config.total_buckets(), || AtomicU64::new(0));

        Self {
            config: *config,
            buckets: buckets.into(),
        }
    }

    /// Increment the bucket that contains the value by one.
    pub fn increment(&self, value: u64) -> Result<(), Error> {
        self.add(value, 1)
    }

    /// Add `count` to the bucket that contains the `value`.
    pub fn add(&self, value: u64, count: u64) -> Result<(), Error> {
        let index = self.config.value_to_index(value)?;
        self.buckets[index].fetch_add(count, Ordering::Relaxed);
        Ok(())
    }

    // NOTE: once stabilized, `target_has_atomic_load_store` is more correct. https://github.com/rust-lang/rust/issues/94039
    #[cfg(target_has_atomic = "64")]
    /// Drains the bucket values into a new Histogram
    ///
    /// Unlike [`load`](AtomicHistogram::load), this method will reset all bucket values to zero. This uses [`AtomicU64::swap`] and is not available
    /// on platforms where [`AtomicU64::swap`] is not available.
    pub fn drain(&self) -> Histogram {
        let buckets: Vec<u64> = self
            .buckets
            .iter()
            .map(|bucket| bucket.swap(0, Ordering::Relaxed))
            .collect();

        Histogram {
            config: self.config,
            buckets: buckets.into(),
        }
    }

    /// Returns the bucket configuration of the histogram.
    pub fn config(&self) -> Config {
        self.config
    }

    /// Read the bucket values into a new `Histogram`
    pub fn load(&self) -> Histogram {
        let buckets: Vec<u64> = self
            .buckets
            .iter()
            .map(|bucket| bucket.load(Ordering::Relaxed))
            .collect();

        Histogram {
            config: self.config,
            buckets: buckets.into(),
        }
    }
}

impl std::fmt::Debug for AtomicHistogram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AtomicHistogram")
            .field("config", &self.config)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use crate::*;

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn size() {
        assert_eq!(std::mem::size_of::<AtomicHistogram>(), 48);
    }

    #[cfg(target_has_atomic = "64")]
    #[test]
    /// Tests that drain properly resets buckets to 0
    fn drain() {
        let histogram = AtomicHistogram::new(7, 64).unwrap();
        for i in 0..=100 {
            let _ = histogram.increment(i);
        }
        let snapshot = histogram.drain();
        let result = snapshot.quantile(0.50).unwrap().unwrap();
        assert_eq!(
            result.get(&Quantile::new(0.50).unwrap()),
            Some(&Bucket {
                count: 1,
                range: 50..=50,
            })
        );
        histogram.increment(1000).unwrap();
        // after another drain the previous data is gone
        let snapshot = histogram.drain();
        let result = snapshot.quantile(0.50).unwrap().unwrap();
        assert_eq!(
            result.get(&Quantile::new(0.50).unwrap()),
            Some(&Bucket {
                count: 1,
                range: 1000..=1003,
            })
        );
    }

    #[test]
    fn quantiles() {
        let histogram = AtomicHistogram::new(7, 64).unwrap();
        let qs = [0.25, 0.50, 0.75, 0.90, 0.99];

        // check empty
        assert_eq!(histogram.load().quantiles(&qs).unwrap(), None);
        assert_eq!(histogram.load().quantile(0.5).unwrap(), None);

        // populate and check min/max
        for i in 0..=100 {
            let _ = histogram.increment(i);
            let result = histogram.load().quantile(0.0).unwrap().unwrap();
            assert_eq!(
                result.get(&Quantile::new(0.0).unwrap()),
                Some(&Bucket {
                    count: 1,
                    range: 0..=0,
                })
            );
            let result = histogram.load().quantile(1.0).unwrap().unwrap();
            assert_eq!(
                result.get(&Quantile::new(1.0).unwrap()),
                Some(&Bucket {
                    count: 1,
                    range: i..=i,
                })
            );
        }

        // check individual quantiles
        for q in qs {
            let result = histogram.load().quantile(q).unwrap().unwrap();
            let bucket = result.get(&Quantile::new(q).unwrap()).unwrap();
            assert_eq!(bucket.end(), (q * 100.0) as u64);
        }

        // p99.9 should be 100
        let result = histogram.load().quantile(0.999).unwrap().unwrap();
        assert_eq!(
            result.get(&Quantile::new(0.999).unwrap()).unwrap().end(),
            100
        );

        // invalid quantiles
        assert_eq!(
            histogram.load().quantiles(&[-1.0]),
            Err(Error::InvalidQuantile)
        );
        assert_eq!(
            histogram.load().quantiles(&[1.01]),
            Err(Error::InvalidQuantile)
        );

        // multi-quantile query
        let result = histogram
            .load()
            .quantiles(&[0.5, 0.9, 0.99, 0.999])
            .unwrap()
            .unwrap();
        let values: Vec<(f64, u64)> = result
            .entries()
            .iter()
            .map(|(q, b)| (q.as_f64(), b.end()))
            .collect();
        assert_eq!(values, vec![(0.5, 50), (0.9, 90), (0.99, 99), (0.999, 100)]);

        // after adding a new value, p99.9 shifts
        let _ = histogram.increment(1024);
        let result = histogram.load().quantile(0.999).unwrap().unwrap();
        assert_eq!(
            result.get(&Quantile::new(0.999).unwrap()),
            Some(&Bucket {
                count: 1,
                range: 1024..=1031,
            })
        );
    }
}
