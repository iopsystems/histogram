#[allow(unused_imports)]
// AtomicU32 used by the AtomicHistogram32 invocation enabled in Task 4
use core::sync::atomic::{AtomicU32, AtomicU64};

use crate::config::Config;
use crate::{AtomicCount, Count, Error, Histogram};

macro_rules! define_atomic_histogram {
    ($name:ident, $count:ty, $atomic:ty, $hist:ident) => {
        /// A histogram that uses atomic counters for each bucket.
        ///
        /// Unlike the non-atomic variant, it cannot be used directly to report
        /// percentiles. Instead, a snapshot must be taken which captures the
        /// state of the histogram at a point in time.
        pub struct $name {
            pub(crate) config: Config,
            pub(crate) buckets: Box<[$atomic]>,
        }

        impl $name {
            /// Construct a new atomic histogram from the provided parameters.
            /// See [`crate::Config`] for the meaning of the parameters.
            pub fn new(grouping_power: u8, max_value_power: u8) -> Result<Self, Error> {
                let config = Config::new(grouping_power, max_value_power)?;
                Ok(Self::with_config(&config))
            }

            /// Creates a new atomic histogram using a provided [`crate::Config`].
            pub fn with_config(config: &Config) -> Self {
                let mut buckets = Vec::with_capacity(config.total_buckets());
                buckets.resize_with(config.total_buckets(), || {
                    <$atomic as AtomicCount>::new(<$count as Count>::ZERO)
                });
                Self {
                    config: *config,
                    buckets: buckets.into(),
                }
            }

            /// Increment the bucket that contains `value` by one.
            pub fn increment(&self, value: u64) -> Result<(), Error> {
                self.add(value, <$count as Count>::ONE)
            }

            /// Add `count` to the bucket that contains `value`.
            pub fn add(&self, value: u64, count: $count) -> Result<(), Error> {
                let index = self.config.value_to_index(value)?;
                self.buckets[index].fetch_add_relaxed(count);
                Ok(())
            }

            /// Returns the bucket configuration of the histogram.
            pub fn config(&self) -> Config {
                self.config
            }

            /// Read the bucket values into a new non-atomic histogram snapshot.
            pub fn load(&self) -> $hist {
                let buckets: Vec<$count> = self.buckets.iter().map(|b| b.load_relaxed()).collect();
                $hist {
                    config: self.config,
                    buckets: buckets.into(),
                }
            }
        }

        impl std::fmt::Debug for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct(stringify!($name))
                    .field("config", &self.config)
                    .finish()
            }
        }
    };
}

define_atomic_histogram!(AtomicHistogram, u64, AtomicU64, Histogram);
// define_atomic_histogram!(AtomicHistogram32, u32, AtomicU32, Histogram32);  // uncommented in Task 4

// NOTE: once stabilized, `target_has_atomic_load_store` is more correct.
// https://github.com/rust-lang/rust/issues/94039
#[cfg(target_has_atomic = "64")]
impl AtomicHistogram {
    /// Drains the bucket values into a new `Histogram`.
    ///
    /// Unlike [`load`](AtomicHistogram::load), this method resets all bucket
    /// values to zero. Uses [`AtomicU64::swap`] under the hood and is
    /// available only on platforms that support 64-bit atomics.
    pub fn drain(&self) -> Histogram {
        let buckets: Vec<u64> = self.buckets.iter().map(|b| b.swap_relaxed(0)).collect();
        Histogram {
            config: self.config,
            buckets: buckets.into(),
        }
    }
}

// AtomicHistogram32 drain block — uncommented in Task 4 once Histogram32 exists.
// #[cfg(target_has_atomic = "32")]
// impl AtomicHistogram32 {
//     pub fn drain(&self) -> Histogram32 {
//         let buckets: Vec<u32> = self.buckets.iter().map(|b| b.swap_relaxed(0)).collect();
//         Histogram32 { config: self.config, buckets: buckets.into() }
//     }
// }

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
    /// Tests that drain properly resets buckets to 0.
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

        for q in qs {
            let result = histogram.load().quantile(q).unwrap().unwrap();
            let bucket = result.get(&Quantile::new(q).unwrap()).unwrap();
            assert_eq!(bucket.end(), (q * 100.0) as u64);
        }

        let result = histogram.load().quantile(0.999).unwrap().unwrap();
        assert_eq!(
            result.get(&Quantile::new(0.999).unwrap()).unwrap().end(),
            100
        );

        assert_eq!(
            histogram.load().quantiles(&[-1.0]),
            Err(Error::InvalidQuantile)
        );
        assert_eq!(
            histogram.load().quantiles(&[1.01]),
            Err(Error::InvalidQuantile)
        );

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
