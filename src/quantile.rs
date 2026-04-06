use std::collections::BTreeMap;

use crate::{Bucket, Error};

/// A validated quantile value in the inclusive range `0.0..=1.0`.
///
/// This newtype ensures that only valid quantile values (finite, non-NaN,
/// within `[0.0, 1.0]`) are used as keys in [`QuantilesResult`]. Because
/// NaN and infinity are excluded by construction, `Quantile` safely
/// implements [`Eq`] and [`Ord`].
///
/// # Examples
///
/// ```
/// use histogram::Quantile;
///
/// let q = Quantile::new(0.99).unwrap();
/// assert_eq!(q.as_f64(), 0.99);
///
/// // Invalid values are rejected
/// assert!(Quantile::new(1.5).is_err());
/// assert!(Quantile::new(f64::NAN).is_err());
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Quantile(f64);

impl Quantile {
    /// Create a new `Quantile` from a value in `0.0..=1.0`.
    ///
    /// Returns [`Error::InvalidQuantile`] if the value is outside the
    /// valid range or is NaN/infinite.
    pub fn new(value: f64) -> Result<Self, Error> {
        if (0.0..=1.0).contains(&value) {
            Ok(Self(value))
        } else {
            Err(Error::InvalidQuantile)
        }
    }

    /// Returns the underlying `f64` value.
    pub fn as_f64(self) -> f64 {
        self.0
    }
}

impl TryFrom<f64> for Quantile {
    type Error = Error;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl PartialEq for Quantile {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}

impl Eq for Quantile {}

impl PartialOrd for Quantile {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Quantile {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap()
    }
}

impl std::fmt::Display for Quantile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// The result of a quantile query on a histogram.
///
/// Contains the requested quantile-to-bucket mappings, the total sample
/// count, and the min/max buckets (the first and last non-zero buckets,
/// representing p0 and p100 respectively).
///
/// Entries are stored in a [`BTreeMap`] keyed by [`Quantile`], providing
/// sorted iteration and efficient keyed lookups.
///
/// # Examples
///
/// ```
/// use histogram::{Histogram, SampleQuantiles, Quantile};
///
/// let mut h = Histogram::new(7, 64).unwrap();
/// for v in 1..=100 {
///     h.increment(v).unwrap();
/// }
///
/// let result = h.quantiles(&[0.5, 0.99]).unwrap().unwrap();
/// assert_eq!(result.entries().len(), 2);
/// assert!(result.total_count() > 0);
///
/// // Keyed lookup
/// let q99 = Quantile::new(0.99).unwrap();
/// let bucket = result.get(&q99).unwrap();
/// println!("p99: {}-{}", bucket.start(), bucket.end());
/// ```
#[derive(Debug, PartialEq)]
pub struct QuantilesResult {
    entries: BTreeMap<Quantile, Bucket>,
    total_count: u128,
    min: Bucket,
    max: Bucket,
}

impl QuantilesResult {
    /// Creates a new `QuantilesResult`.
    pub(crate) fn new(
        entries: BTreeMap<Quantile, Bucket>,
        total_count: u128,
        min: Bucket,
        max: Bucket,
    ) -> Self {
        Self {
            entries,
            total_count,
            min,
            max,
        }
    }

    /// Returns the quantile-to-bucket mappings, sorted by quantile.
    pub fn entries(&self) -> &BTreeMap<Quantile, Bucket> {
        &self.entries
    }

    /// Look up the bucket for a specific quantile.
    pub fn get(&self, quantile: &Quantile) -> Option<&Bucket> {
        self.entries.get(quantile)
    }

    /// Returns the total number of observations across all buckets.
    pub fn total_count(&self) -> u128 {
        self.total_count
    }

    /// Returns the minimum bucket (first non-zero bucket, i.e. p0).
    pub fn min(&self) -> &Bucket {
        &self.min
    }

    /// Returns the maximum bucket (last non-zero bucket, i.e. p100).
    pub fn max(&self) -> &Bucket {
        &self.max
    }
}

/// Trait for computing quantiles from histogram data.
///
/// Implemented for [`crate::Histogram`] and [`crate::SparseHistogram`].
/// [`crate::AtomicHistogram`] does not implement this trait — use
/// [`AtomicHistogram::load()`](crate::AtomicHistogram::load) to get a
/// `Histogram` snapshot first.
///
/// # Examples
///
/// ```
/// use histogram::{Histogram, SparseHistogram, SampleQuantiles};
///
/// let mut h = Histogram::new(7, 64).unwrap();
/// h.increment(100).unwrap();
///
/// // Works on both Histogram and SparseHistogram
/// fn show_p99(h: &impl SampleQuantiles) {
///     if let Ok(Some(result)) = h.quantile(0.99) {
///         let bucket = result.entries().values().next().unwrap();
///         println!("p99: {}-{}", bucket.start(), bucket.end());
///     }
/// }
///
/// show_p99(&h);
/// show_p99(&SparseHistogram::from(&h));
/// ```
pub trait SampleQuantiles {
    /// Compute quantiles for the given values.
    ///
    /// Each value in `quantiles` must be in `0.0..=1.0`. Returns
    /// `Err(Error::InvalidQuantile)` if any value is out of range.
    /// Returns `Ok(None)` if the histogram is empty.
    fn quantiles(&self, quantiles: &[f64]) -> Result<Option<QuantilesResult>, Error>;

    /// Compute a single quantile. Convenience wrapper around [`quantiles`](Self::quantiles).
    fn quantile(&self, quantile: f64) -> Result<Option<QuantilesResult>, Error> {
        self.quantiles(&[quantile])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Histogram, SparseHistogram};

    #[test]
    fn quantile_validation() {
        assert!(Quantile::new(0.0).is_ok());
        assert!(Quantile::new(0.5).is_ok());
        assert!(Quantile::new(1.0).is_ok());
        assert!(Quantile::new(-0.1).is_err());
        assert!(Quantile::new(1.1).is_err());
        assert!(Quantile::new(f64::NAN).is_err());
        assert!(Quantile::new(f64::INFINITY).is_err());
    }

    #[test]
    fn quantile_ordering() {
        let q1 = Quantile::new(0.5).unwrap();
        let q2 = Quantile::new(0.99).unwrap();
        assert!(q1 < q2);
        assert_eq!(q1, Quantile::new(0.5).unwrap());
    }

    #[test]
    fn empty_histogram_returns_none() {
        let h = Histogram::new(7, 64).unwrap();
        assert_eq!(h.quantiles(&[0.5, 0.99]).unwrap(), None);
        assert_eq!(h.quantile(0.5).unwrap(), None);

        let s = SparseHistogram::from(&h);
        assert_eq!(s.quantiles(&[0.5, 0.99]).unwrap(), None);
    }

    #[test]
    fn invalid_quantile_returns_error() {
        let h = Histogram::new(7, 64).unwrap();
        assert_eq!(h.quantiles(&[1.5]), Err(Error::InvalidQuantile));
        assert_eq!(h.quantiles(&[-0.1]), Err(Error::InvalidQuantile));
    }

    #[test]
    fn basic_quantiles() {
        let mut h = Histogram::new(7, 64).unwrap();
        for v in 0..=100 {
            h.increment(v).unwrap();
        }

        let result = h.quantiles(&[0.5, 0.9, 0.99]).unwrap().unwrap();
        assert_eq!(result.entries().len(), 3);
        assert_eq!(result.total_count(), 101);

        // min bucket should contain 0
        assert_eq!(result.min().start(), 0);
        // max bucket should contain 100
        assert_eq!(result.max().end(), 100);

        // p50 should be around 50
        let q50 = Quantile::new(0.5).unwrap();
        assert_eq!(result.get(&q50).unwrap().end(), 50);

        // p99 should be around 99
        let q99 = Quantile::new(0.99).unwrap();
        assert_eq!(result.get(&q99).unwrap().end(), 99);
    }

    #[test]
    fn single_sample() {
        let mut h = Histogram::new(7, 64).unwrap();
        h.increment(42).unwrap();

        let result = h.quantile(0.5).unwrap().unwrap();
        assert_eq!(result.total_count(), 1);
        assert_eq!(result.min().end(), 42);
        assert_eq!(result.max().end(), 42);
    }

    #[test]
    fn histogram_and_sparse_agree() {
        let mut h = Histogram::new(4, 10).unwrap();
        for v in 1..1024 {
            h.increment(v).unwrap();
        }
        let s = SparseHistogram::from(&h);

        let quantiles = &[0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999];

        let hr = h.quantiles(quantiles).unwrap().unwrap();
        let sr = s.quantiles(quantiles).unwrap().unwrap();

        assert_eq!(hr.total_count(), sr.total_count());
        assert_eq!(hr.min().range(), sr.min().range());
        assert_eq!(hr.max().range(), sr.max().range());

        for (hq, sq) in hr.entries().iter().zip(sr.entries().iter()) {
            assert_eq!(hq.0, sq.0);
            assert_eq!(hq.1.range(), sq.1.range());
        }
    }

    #[test]
    fn duplicate_quantiles_deduped() {
        let mut h = Histogram::new(7, 64).unwrap();
        h.increment(50).unwrap();

        let result = h.quantiles(&[0.5, 0.5, 0.99, 0.99]).unwrap().unwrap();
        // Duplicates are deduped in the BTreeMap
        assert_eq!(result.entries().len(), 2);
    }
}
