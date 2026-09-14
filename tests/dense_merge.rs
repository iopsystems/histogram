//! Checked in-place merging must agree with arithmetic wider than either counter.
use histogram::Error;

macro_rules! contract {
    ($module:ident, $dense:ty, $count:ty) => {
        mod $module {
            use super::*;
            type Dense = $dense;
            type Count = $count;

            fn check_merge(mut dst: Dense, src: &Dense) {
                let saved = dst.clone();
                let source_saved = src.clone();
                let pointer = dst.as_slice().as_ptr();
                let sums: Vec<u128> = dst
                    .as_slice()
                    .iter()
                    .zip(src.as_slice())
                    .map(|(&a, &b)| u128::from(a) + u128::from(b))
                    .collect();
                let overflow = sums.iter().any(|&sum| sum > u128::from(Count::MAX));

                let result = dst.checked_add_assign(src);
                if overflow {
                    assert_eq!(result, Err(Error::Overflow));
                    assert_eq!(dst, saved);
                } else {
                    assert_eq!(result, Ok(()));
                    assert_eq!(dst.config(), saved.config());
                    assert_eq!(
                        dst.as_slice()
                            .iter()
                            .copied()
                            .map(u128::from)
                            .collect::<Vec<_>>(),
                        sums
                    );
                }
                assert_eq!(dst.as_slice().as_ptr(), pointer);
                assert_eq!(src, &source_saved);
            }

            fn geometries() -> impl Iterator<Item = (u8, u8)> {
                // gp0 covers every realizable length from 2 through 65, including
                // vector boundaries and tails; larger gp covers realistic layouts.
                (1..=64).map(|max| (0, max)).chain([
                    (1, 2),
                    (1, 4),
                    (4, 9),
                    (4, 16),
                    (7, 32),
                    (10, 32),
                ])
            }

            #[test]
            fn valid_sums_match_wide_oracle_across_lengths_and_geometries() {
                let mut state = 0x9e37_79b9_7f4a_7c15u64;
                for (gp, max) in geometries() {
                    let mut dst = Dense::new(gp, max).unwrap();
                    let mut src = dst.clone();
                    check_merge(dst.clone(), &src);
                    for round in 0..8 {
                        for (a, b) in dst.as_mut_slice().iter_mut().zip(src.as_mut_slice()) {
                            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                            *a = state as Count;
                            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                            *b = if round % 2 == 0 {
                                // Exact MAX sums exercise carries through every bit;
                                // their aggregate deliberately exceeds Count::MAX.
                                Count::MAX - *a
                            } else {
                                *a >>= 1;
                                (state as Count) >> 1
                            };
                        }
                        check_merge(dst.clone(), &src);
                    }
                }
            }

            #[test]
            fn extreme_operands_at_every_position_match_wide_oracle() {
                let high = 1 << (Count::BITS - 1);
                let pairs = [
                    (0, Count::MAX),
                    (Count::MAX, 0),
                    (Count::MAX - 1, 1),
                    (high - 1, 1),
                    (high, high - 1),
                    (high - 1, high),
                    (Count::MAX, 1),
                    (1, Count::MAX),
                    (high, high),
                    (Count::MAX, Count::MAX),
                ];
                for (gp, max) in (1..=64).map(|max| (0, max)).chain([(1, 4), (4, 9), (7, 8)]) {
                    let template = Dense::new(gp, max).unwrap();
                    for position in 0..template.as_slice().len() {
                        for (a, b) in pairs {
                            let mut dst = template.clone();
                            let mut src = template.clone();
                            // Every other bucket would change if the application
                            // pass started before all overflow checks completed.
                            dst.as_mut_slice().fill(3);
                            src.as_mut_slice().fill(5);
                            dst.as_mut_slice()[position] = a;
                            src.as_mut_slice()[position] = b;
                            check_merge(dst, &src);
                        }
                    }
                }
            }

            #[test]
            fn arbitrary_full_width_operands_match_wide_oracle() {
                let mut state = 0xd1b5_4a32_d192_ed03u64;
                for (gp, max) in geometries() {
                    let mut dst = Dense::new(gp, max).unwrap();
                    let mut src = dst.clone();
                    for (a, b) in dst.as_mut_slice().iter_mut().zip(src.as_mut_slice()) {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        *a = state as Count;
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        *b = state as Count;
                    }
                    check_merge(dst, &src);
                }
            }

            #[test]
            fn incompatible_geometry_precedes_overflow_and_preserves_storage() {
                let mut dst = Dense::new(0, 7).unwrap();
                dst.as_mut_slice().fill(Count::MAX);
                let saved = dst.clone();
                let pointer = dst.as_slice().as_ptr();
                for (gp, max) in [(1, 4), (0, 6), (0, 8)] {
                    let mut src = Dense::new(gp, max).unwrap();
                    src.as_mut_slice().fill(1);
                    assert_eq!(
                        dst.checked_add_assign(&src),
                        Err(Error::IncompatibleParameters)
                    );
                    assert_eq!(dst, saved);
                    assert_eq!(dst.as_slice().as_ptr(), pointer);
                }
            }
        }
    };
}

contract!(u32_counts, histogram::Histogram32, u32);
contract!(u64_counts, histogram::Histogram, u64);
