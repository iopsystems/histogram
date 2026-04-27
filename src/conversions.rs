//! Cross-width and combined cross-variant + narrowing conversions
//! between histogram type families.
//!
//! - **Widening** (`u32` → `u64`) is infallible (`From`).
//! - **Narrowing** (`u64` → `u32`) is fallible (`TryFrom`, returns
//!   [`Error::Overflow`]).
//! - **Cross-variant + narrowing combined paths** (e.g. `Histogram` →
//!   `CumulativeROHistogram32`) are also exposed as `TryFrom` for the
//!   recommended snapshot pipeline.

use crate::{
    AtomicCount, AtomicHistogram, AtomicHistogram32, CumulativeROHistogram,
    CumulativeROHistogram32, Error, Histogram, Histogram32, SparseHistogram, SparseHistogram32,
};

// =================================================================
// Widening (u32 -> u64) — Task 6
// =================================================================

impl From<&Histogram32> for Histogram {
    fn from(h: &Histogram32) -> Self {
        let buckets: Vec<u64> = h.as_slice().iter().map(|&c| c as u64).collect();
        Histogram::from_buckets(
            h.config().grouping_power(),
            h.config().max_value_power(),
            buckets,
        )
        .expect("widening preserves bucket count")
    }
}

impl From<&AtomicHistogram32> for AtomicHistogram {
    fn from(h: &AtomicHistogram32) -> Self {
        // Snapshot via load(), widen, materialize as fresh atomic histogram.
        let snapshot = h.load();
        let widened: Histogram = (&snapshot).into();
        let out = AtomicHistogram::with_config(&widened.config());
        for (i, &c) in widened.as_slice().iter().enumerate() {
            out.buckets[i].fetch_add_relaxed(c);
        }
        out
    }
}

impl From<&SparseHistogram32> for SparseHistogram {
    fn from(h: &SparseHistogram32) -> Self {
        let widened: Vec<u64> = h.count().iter().map(|&c| c as u64).collect();
        SparseHistogram::from_parts(h.config(), h.index().to_vec(), widened)
            .expect("widening preserves invariants")
    }
}

impl From<&CumulativeROHistogram32> for CumulativeROHistogram {
    fn from(h: &CumulativeROHistogram32) -> Self {
        let widened: Vec<u64> = h.count().iter().map(|&c| c as u64).collect();
        CumulativeROHistogram::from_parts(h.config(), h.index().to_vec(), widened)
            .expect("widening preserves invariants")
    }
}

// =================================================================
// Narrowing (u64 -> u32), same variant — Task 7
// =================================================================

impl TryFrom<&Histogram> for Histogram32 {
    type Error = Error;
    fn try_from(h: &Histogram) -> Result<Self, Error> {
        let mut narrowed: Vec<u32> = Vec::with_capacity(h.as_slice().len());
        for &c in h.as_slice() {
            narrowed.push(u32::try_from(c).map_err(|_| Error::Overflow)?);
        }
        Histogram32::from_buckets(
            h.config().grouping_power(),
            h.config().max_value_power(),
            narrowed,
        )
    }
}

impl TryFrom<&SparseHistogram> for SparseHistogram32 {
    type Error = Error;
    fn try_from(h: &SparseHistogram) -> Result<Self, Error> {
        let mut narrowed: Vec<u32> = Vec::with_capacity(h.count().len());
        for &c in h.count() {
            narrowed.push(u32::try_from(c).map_err(|_| Error::Overflow)?);
        }
        SparseHistogram32::from_parts(h.config(), h.index().to_vec(), narrowed)
    }
}

impl TryFrom<&CumulativeROHistogram> for CumulativeROHistogram32 {
    type Error = Error;
    fn try_from(h: &CumulativeROHistogram) -> Result<Self, Error> {
        // Cumulative-only optimization: total bounds every prefix sum.
        // If the last (max) cumulative value fits in u32, every entry fits.
        if let Some(&last) = h.count().last() {
            if u32::try_from(last).is_err() {
                return Err(Error::Overflow);
            }
        }
        let narrowed: Vec<u32> = h.count().iter().map(|&c| c as u32).collect();
        CumulativeROHistogram32::from_parts(h.config(), h.index().to_vec(), narrowed)
    }
}

// =================================================================
// Tests
// =================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Config;

    // ---------- Widening ----------

    #[test]
    fn widen_histogram() {
        let mut h32 = Histogram32::new(7, 32).unwrap();
        h32.add(1, 1234u32).unwrap();
        h32.add(1000, 5678u32).unwrap();
        let h64: Histogram = (&h32).into();
        assert_eq!(h64.config(), h32.config());
        for (a, b) in h64.as_slice().iter().zip(h32.as_slice().iter()) {
            assert_eq!(*a, *b as u64);
        }
    }

    #[test]
    fn widen_sparse() {
        let config = Config::new(7, 32).unwrap();
        let s32 = SparseHistogram32::from_parts(config, vec![1, 3], vec![10u32, 20]).unwrap();
        let s64: SparseHistogram = (&s32).into();
        assert_eq!(s64.count(), &[10u64, 20]);
        assert_eq!(s64.index(), &[1u32, 3]);
    }

    #[test]
    fn widen_cumulative() {
        let config = Config::new(7, 32).unwrap();
        let c32 = CumulativeROHistogram32::from_parts(config, vec![1, 3], vec![10u32, 30]).unwrap();
        let c64: CumulativeROHistogram = (&c32).into();
        assert_eq!(c64.count(), &[10u64, 30]);
    }

    #[cfg(target_has_atomic = "32")]
    #[cfg(target_has_atomic = "64")]
    #[test]
    fn widen_atomic_histogram() {
        let h32 = AtomicHistogram32::new(7, 32).unwrap();
        h32.add(5, 100u32).unwrap();
        h32.add(50, 200u32).unwrap();
        let h64: AtomicHistogram = (&h32).into();
        let snap = h64.load();
        let total: u64 = snap.as_slice().iter().sum();
        assert_eq!(total, 300);
    }

    // ---------- Narrowing (same variant) ----------

    #[test]
    fn narrow_histogram_success() {
        let mut h64 = Histogram::new(7, 32).unwrap();
        h64.add(1, 100u64).unwrap();
        h64.add(1000, 200u64).unwrap();
        let h32: Histogram32 = (&h64).try_into().unwrap();
        assert_eq!(h32.as_slice()[1], 100u32);
    }

    #[test]
    fn narrow_histogram_overflow() {
        let mut h64 = Histogram::new(2, 4).unwrap();
        h64.add(1, (u32::MAX as u64) + 1).unwrap();
        let r: Result<Histogram32, _> = (&h64).try_into();
        assert_eq!(r, Err(Error::Overflow));
    }

    #[test]
    fn narrow_sparse_overflow() {
        let config = Config::new(7, 32).unwrap();
        let s64 =
            SparseHistogram::from_parts(config, vec![1], vec![(u32::MAX as u64) + 1]).unwrap();
        let r: Result<SparseHistogram32, _> = (&s64).try_into();
        assert_eq!(r, Err(Error::Overflow));
    }

    #[test]
    fn narrow_cumulative_checks_total_only() {
        let config = Config::new(7, 32).unwrap();
        let c64 = CumulativeROHistogram::from_parts(
            config,
            vec![1, 3],
            vec![100u64, (u32::MAX as u64) + 1],
        )
        .unwrap();
        let r: Result<CumulativeROHistogram32, _> = (&c64).try_into();
        assert_eq!(r, Err(Error::Overflow));

        let c64_ok =
            CumulativeROHistogram::from_parts(config, vec![1, 3], vec![100u64, 200]).unwrap();
        let c32: CumulativeROHistogram32 = (&c64_ok).try_into().unwrap();
        assert_eq!(c32.total_count(), 200);
    }

    #[test]
    fn round_trip_widen_then_narrow() {
        let mut h32 = Histogram32::new(7, 32).unwrap();
        h32.add(5, 1234u32).unwrap();
        h32.add(50, 5678u32).unwrap();
        let h64: Histogram = (&h32).into();
        let h32_back: Histogram32 = (&h64).try_into().unwrap();
        assert_eq!(h32.as_slice(), h32_back.as_slice());
    }
}
