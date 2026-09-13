use histogram::{Histogram, Histogram32};

macro_rules! reset_contract {
    ($test:ident, $histogram:ty, $count:ty) => {
        #[test]
        fn $test() {
            let mut histogram = <$histogram>::new(7, 64).unwrap();
            let config = histogram.config();
            let pointer = histogram.as_slice().as_ptr();
            let length = histogram.as_slice().len();
            histogram.as_mut_slice().fill(<$count>::MAX);

            histogram.reset();
            assert_eq!(histogram.config(), config);
            assert_eq!(histogram.as_slice().as_ptr(), pointer);
            assert_eq!(histogram.as_slice().len(), length);
            assert!(histogram.as_slice().iter().all(|&count| count == 0));
            assert!(histogram.quantiles(&[0.0, 0.5, 1.0]).unwrap().is_none());

            // Reset is also valid on an already empty histogram.
            histogram.reset();
            assert_eq!(histogram.as_slice().as_ptr(), pointer);
            histogram.increment(0).unwrap();
            histogram.add(u64::MAX, 2).unwrap();
            let report = histogram.quantiles(&[0.0, 1.0]).unwrap().unwrap();
            assert_eq!(report.total_count(), 3);
            assert_eq!(report.min().range(), 0..=0);
            assert_eq!(report.min().count(), 1);
            assert_eq!(report.max().end(), u64::MAX);
            assert_eq!(report.max().count(), 2);
        }
    };
}
reset_contract!(
    reset_u64_preserves_storage_for_next_interval,
    Histogram,
    u64
);
reset_contract!(
    reset_u32_preserves_storage_for_next_interval,
    Histogram32,
    u32
);
