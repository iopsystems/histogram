use core::ops::RangeInclusive;

/// A bucket represents a quantized range of values and a count of observations
/// that fall into that range.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bucket {
    pub(crate) count: u64,
    pub(crate) range: RangeInclusive<u64>,
}

impl Bucket {
    /// Returns the number of observations within the bucket's range.
    #[inline]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Returns the range for the bucket.
    #[inline]
    pub fn range(&self) -> RangeInclusive<u64> {
        self.range.clone()
    }

    /// Returns the inclusive lower bound for the bucket.
    #[inline]
    pub fn start(&self) -> u64 {
        *self.range.start()
    }

    /// Returns the inclusive upper bound for the bucket.
    #[inline]
    pub fn end(&self) -> u64 {
        *self.range.end()
    }
}
