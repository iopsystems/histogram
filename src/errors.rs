use thiserror::Error;

/// Errors returned for histogram construction and operations.
#[non_exhaustive]
#[derive(Error, Debug, PartialEq)]
pub enum Error {
    #[error("max power is too high, check that n <= 64")]
    MaxPowerTooHigh,
    #[error("max power is too low, check that grouping_power < max_value_power")]
    MaxPowerTooLow,
    #[deprecated(since = "1.1.0", note = "Use InvalidQuantile")]
    #[error("invalid percentile, must be in range 0.0..=1.0")]
    InvalidPercentile,
    #[error("invalid quantile, must be in range 0.0..=1.0")]
    InvalidQuantile,
    #[error("the value is outside of the storable range")]
    OutOfRange,
    #[error("the histogram parameters are incompatible")]
    IncompatibleParameters,
    #[error("an overflow occurred")]
    Overflow,
    #[error("an underflow occurred")]
    Underflow,
    #[error("the histogram is not a subset")]
    InvalidSubset,
}
