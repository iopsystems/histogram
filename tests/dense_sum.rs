//! Owned batch aggregation preserves inputs and checks individual bucket sums.
use histogram::Error;

macro_rules! contract {
    ($module:ident, $dense:ty, $count:ty) => {
        mod $module {
            use super::*;
            type Dense = $dense;
            type Count = $count;

            fn check(sources: &[&Dense]) {
                let saved: Vec<Dense> = sources.iter().map(|h| (*h).clone()).collect();
                let mut sums = vec![0u128; sources[0].as_slice().len()];
                for source in sources {
                    for (sum, &count) in sums.iter_mut().zip(source.as_slice()) {
                        *sum += u128::from(count);
                    }
                }
                let result = Dense::checked_sum(sources);
                if sums.iter().any(|&v| v > u128::from(Count::MAX)) {
                    assert_eq!(result, Err(Error::Overflow));
                } else {
                    let result = result.unwrap();
                    assert_eq!(result.config(), sources[0].config());
                    assert_eq!(
                        result
                            .as_slice()
                            .iter()
                            .copied()
                            .map(u128::from)
                            .collect::<Vec<_>>(),
                        sums
                    );
                    for source in sources {
                        assert_ne!(result.as_slice().as_ptr(), source.as_slice().as_ptr());
                    }
                }
                for (source, saved) in sources.iter().zip(&saved) {
                    assert_eq!(*source, saved);
                }
            }

            #[test]
            fn empty_input_has_no_configuration() {
                assert_eq!(Dense::checked_sum(&[]), Err(Error::IncompatibleParameters));
            }

            #[test]
            fn one_input_is_an_independent_clone() {
                let mut input = Dense::new(0, 7).unwrap();
                input.as_mut_slice().fill(Count::MAX);
                let mut result = Dense::checked_sum(&[&input]).unwrap();
                assert_eq!(input, result);
                assert_ne!(input.as_slice().as_ptr(), result.as_slice().as_ptr());
                result.reset();
                assert!(input.as_slice().iter().all(|&v| v == Count::MAX));
            }

            #[test]
            fn same_input_can_be_repeated_and_zero_counts_are_valid() {
                for (gp, max) in [(0, 1), (0, 7), (7, 30), (10, 30)] {
                    let mut input = Dense::new(gp, max).unwrap();
                    check(&[&input, &input, &input]);
                    input.as_mut_slice().fill(7);
                    check(&[&input, &input, &input]);
                }
            }

            #[test]
            fn per_bucket_limit_allows_a_wider_total() {
                let mut a = Dense::new(0, 3).unwrap();
                let mut b = a.clone();
                let mut c = a.clone();
                a.as_mut_slice().fill(Count::MAX - 2);
                b.as_mut_slice().fill(1);
                c.as_mut_slice().fill(1);
                check(&[&a, &b, &c]);
                let result = Dense::checked_sum(&[&a, &b, &c]).unwrap();
                assert!(result.as_slice().iter().all(|&v| v == Count::MAX));
            }

            #[test]
            fn all_configurations_are_checked_before_arithmetic() {
                let mut a = Dense::new(0, 7).unwrap();
                a.as_mut_slice().fill(Count::MAX);
                let before = a.clone();
                // Both mismatched bucket lengths and equal-length configurations
                // must be rejected, even when an earlier pair would overflow.
                for mismatch in [Dense::new(0, 8).unwrap(), Dense::new(1, 4).unwrap()] {
                    for inputs in [&[&a, &a, &mismatch][..], &[&mismatch, &a, &a][..]] {
                        assert_eq!(
                            Dense::checked_sum(inputs),
                            Err(Error::IncompatibleParameters)
                        );
                    }
                    assert_eq!(a, before);
                    assert!(mismatch.as_slice().iter().all(|&v| v == 0));
                }
            }

            #[test]
            fn bounded_and_full_width_inputs_match_wide_oracle() {
                let mut state = 0x9e37_79b9_7f4a_7c15u64;
                for (gp, max) in (1..=64).map(|m| (0, m)).chain([(3, 30), (7, 30), (10, 30)]) {
                    for inputs in [2, 8, 64] {
                        let mut sources = vec![Dense::new(gp, max).unwrap(); inputs];
                        for bounded in [true, false] {
                            for source in &mut sources {
                                for bucket in source.as_mut_slice() {
                                    state ^= state << 13;
                                    state ^= state >> 7;
                                    state ^= state << 17;
                                    *bucket = if bounded {
                                        (state % 65536) as Count
                                    } else {
                                        state as Count
                                    };
                                }
                            }
                            check(&sources.iter().collect::<Vec<_>>());
                        }
                    }
                }
            }

            #[test]
            fn carry_and_late_overflow_at_every_vector_lane_and_tail() {
                let high = 1 << (Count::BITS - 1);
                for (gp, max) in (1..=64).map(|m| (0, m)).chain([(3, 30), (7, 30)]) {
                    let mut a = Dense::new(gp, max).unwrap();
                    let mut b = a.clone();
                    a.as_mut_slice().fill(Count::MAX - 2);
                    b.as_mut_slice().fill(1);
                    check(&[&a, &b, &b]);
                    let before_a = a.clone();
                    let before_b = b.clone();
                    for index in 0..a.as_slice().len() {
                        let mut late = Dense::new(gp, max).unwrap();
                        late.as_mut_slice()[index] = 2;
                        let before_late = late.clone();
                        assert_eq!(Dense::checked_sum(&[&a, &b, &late]), Err(Error::Overflow));
                        assert_eq!(a, before_a);
                        assert_eq!(b, before_b);
                        assert_eq!(late, before_late);
                    }
                    // Exact carries across the sign bit must not be confused
                    // with overflow by vectorized unsigned comparisons.
                    a.as_mut_slice().fill(high - 1);
                    b.as_mut_slice().fill(1);
                    check(&[&a, &b]);
                    a.as_mut_slice().fill(high);
                    b.as_mut_slice().fill(high - 1);
                    check(&[&a, &b]);
                    b.as_mut_slice().fill(high);
                    check(&[&a, &b]);
                }
            }
        }
    };
}

contract!(wide, histogram::Histogram, u64);
contract!(narrow, histogram::Histogram32, u32);
