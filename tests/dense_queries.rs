//! Dense report invariants: exact ranks, bounds, wide sums, and mutable counters.
use histogram::{Error, Histogram, Histogram32, Quantile};

macro_rules! check_width {
    ($module:ident, $histogram:ty, $count:ty) => {
        mod $module {
            use super::*;

            #[test]
            fn ranks_at_scan_boundaries_and_zero_runs() {
                // gp5/max6 has one bucket per integer in 0..64.
                let mut h = <$histogram>::new(5, 6).unwrap();
                for (value, count) in [
                    (0, 1),
                    (7, 2),
                    (8, 1),
                    (16, 4),
                    (31, 2),
                    (32, 1),
                    (55, 3),
                    (63, 2),
                ] {
                    h.add(value, count).unwrap();
                }
                let mut qs: Vec<_> = (0..=16).rev().map(|rank| rank as f64 / 16.0).collect();
                qs.extend([0.5, 0.0, 1.0]);
                let result = h.quantiles(&qs).unwrap().unwrap();
                assert_eq!(result.total_count(), 16);
                assert_eq!(result.entries().len(), 17);
                assert_eq!(result.min().range(), 0..=0);
                assert_eq!(result.min().count(), 1);
                assert_eq!(result.max().range(), 63..=63);
                assert_eq!(result.max().count(), 2);
                let expected = [
                    (0, 1),
                    (0, 1),
                    (7, 2),
                    (7, 2),
                    (8, 1),
                    (16, 4),
                    (16, 4),
                    (16, 4),
                    (16, 4),
                    (31, 2),
                    (31, 2),
                    (32, 1),
                    (55, 3),
                    (55, 3),
                    (55, 3),
                    (63, 2),
                    (63, 2),
                ];
                for (rank, (value, count)) in expected.into_iter().enumerate() {
                    let bucket = result
                        .get(&Quantile::new(rank as f64 / 16.0).unwrap())
                        .unwrap();
                    assert_eq!(bucket.range(), value..=value);
                    assert_eq!(bucket.count(), count);
                }
            }

            #[test]
            fn singleton_at_every_position_including_short_tails() {
                // Includes arrays shorter than a scan block and non-multiple lengths.
                for (gp, max) in [(0, 1), (0, 8), (1, 7), (5, 6)] {
                    let mut h = <$histogram>::new(gp, max).unwrap();
                    for index in 0..h.as_slice().len() {
                        h.as_mut_slice().fill(0);
                        h.as_mut_slice()[index] = 3;
                        let result = h.quantiles(&[1.0, 0.5, 0.0]).unwrap().unwrap();
                        let expected = h.iter().nth(index).unwrap();
                        assert_eq!(result.total_count(), 3);
                        assert_eq!(result.min(), &expected);
                        assert_eq!(result.max(), &expected);
                        for bucket in result.entries().values() {
                            assert_eq!(bucket, &expected);
                        }
                    }
                }
            }

            #[test]
            fn reports_match_independently_sorted_observations() {
                let mut seed = 42u64;
                for gp in [0, 3, 7, 10] {
                    let mut h = <$histogram>::new(gp, 16).unwrap();
                    let mut values = Vec::new();
                    for _ in 0..2048 {
                        seed ^= seed << 13;
                        seed ^= seed >> 7;
                        seed ^= seed << 17;
                        let value = seed & 65535;
                        h.increment(value).unwrap();
                        values.push(value);
                    }
                    values.sort_unstable();
                    let qs: Vec<_> = (0..=128).map(|i| i as f64 / 128.0).collect();
                    let result = h.quantiles(&qs).unwrap().unwrap();
                    assert_eq!(result.total_count(), 2048);
                    assert!(result.min().range().contains(&values[0]));
                    assert!(result.max().range().contains(&values[2047]));
                    for (i, q) in qs.into_iter().enumerate() {
                        let bucket = result.get(&Quantile::new(q).unwrap()).unwrap();
                        assert!(bucket.range().contains(&values[(i * 16).max(1) - 1]));
                        assert_eq!(
                            bucket.count(),
                            values.iter().filter(|v| bucket.range().contains(v)).count() as u64
                        );
                    }
                }
            }

            #[test]
            fn empty_requests_validation_and_counter_mutation() {
                let mut h = <$histogram>::new(5, 6).unwrap();
                for populated in [false, true] {
                    if populated {
                        h.add(17, 4).unwrap();
                    }
                    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.01, 1.01] {
                        assert_eq!(h.quantiles(&[0.5, invalid]), Err(Error::InvalidQuantile));
                    }
                    let result = h.quantiles(&[]).unwrap();
                    if populated {
                        let result = result.unwrap();
                        assert!(result.entries().is_empty());
                        assert_eq!(result.total_count(), 4);
                        assert_eq!(result.min().range(), 17..=17);
                        assert_eq!(result.max().range(), 17..=17);
                    } else {
                        assert!(result.is_none());
                    }
                }
                h.as_mut_slice().fill(0);
                assert!(h.quantile(0.5).unwrap().is_none());
                h.as_mut_slice()[63] = 2;
                let result = h.quantile(0.5).unwrap().unwrap();
                assert_eq!(result.total_count(), 2);
                assert_eq!(result.min().range(), 63..=63);
                assert_eq!(result.max().range(), 63..=63);
            }

            #[test]
            fn block_sums_exceed_counter_width() {
                let mut h = <$histogram>::new(5, 6).unwrap();
                let count: $count = 1 << (<$count>::BITS - 1);
                h.as_mut_slice()[1..=16].fill(count);
                let result = h.quantiles(&[0.0, 0.5, 1.0]).unwrap().unwrap();
                assert_eq!(result.total_count(), count as u128 * 16);
                for (q, value) in [(0.0, 1), (0.5, 8), (1.0, 16)] {
                    let bucket = result.get(&Quantile::new(q).unwrap()).unwrap();
                    assert_eq!(bucket.range(), value..=value);
                    assert_eq!(bucket.count(), count as u64);
                }
            }
        }
    };
}
check_width!(wide, Histogram, u64);
check_width!(narrow, Histogram32, u32);
