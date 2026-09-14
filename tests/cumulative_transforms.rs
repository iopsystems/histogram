use histogram::{
    Config, CumulativeROHistogram, CumulativeROHistogram32, Error, Histogram, Histogram32,
};

macro_rules! transform_tests {
    ($module:ident, $owned:ident, $view:ident, $dense:ident, $count:ty) => {
        mod $module {
            use super::*;

            fn replay(gp: u8, max: u8, values: &[u64]) -> $owned {
                let mut dense = $dense::new(gp, max).unwrap();
                for &value in values {
                    dense.increment(value).unwrap();
                }
                $owned::from(&dense)
            }

            #[test]
            fn merge_matches_raw_replay_for_empty_disjoint_and_overlapping_windows() {
                let windows: &[&[u64]] = &[&[], &[0], &[1, 1, 17, 255], &[0, 17, 17, 256, 1023]];
                for left in windows {
                    for right in windows {
                        let a = replay(3, 10, left);
                        let b = replay(3, 10, right);
                        let before = (a.clone(), b.clone());
                        let values: Vec<_> = left.iter().chain(right.iter()).copied().collect();
                        let expected = replay(3, 10, &values);
                        let result = a.checked_add(&b).unwrap();
                        assert_eq!(result, expected);
                        assert_eq!(a.as_ref().checked_add(&b.as_ref()).unwrap(), expected);
                        assert_eq!(b.checked_add(&a).unwrap(), expected);
                        for q in [0.0, 0.5, 0.99, 1.0] {
                            assert_eq!(result.quantile_bucket(q), expected.quantile_bucket(q));
                        }
                        assert_eq!((a, b), before);
                    }
                }
            }

            #[test]
            fn downsample_matches_every_value_replayed_at_all_coarser_geometries() {
                let values: Vec<_> = (0..1024).chain((0..1024).step_by(7)).collect();
                for gp in 1..8 {
                    let input = replay(gp, 10, &values);
                    let before = input.clone();
                    for target in 0..gp {
                        let expected = replay(target, 10, &values);
                        assert_eq!(input.downsample(target).unwrap(), expected);
                        assert_eq!(input.as_ref().downsample(target).unwrap(), expected);
                        assert_eq!(
                            replay(gp, 10, &[]).downsample(target).unwrap(),
                            replay(target, 10, &[])
                        );
                    }
                    assert_eq!(input, before);
                }
            }

            #[test]
            fn transforms_cover_u64_value_range_boundaries() {
                let values = [
                    0,
                    1,
                    255,
                    256,
                    257,
                    1 << 32,
                    (1 << 63) - 1,
                    1 << 63,
                    u64::MAX,
                ];
                let a = replay(7, 64, &values);
                assert_eq!(a.downsample(0).unwrap(), replay(0, 64, &values));
                let doubled: Vec<_> = values.into_iter().chain(values).collect();
                assert_eq!(a.checked_add(&a).unwrap(), replay(7, 64, &doubled));
            }

            #[test]
            fn rejects_geometry_mismatch_even_for_empty_inputs() {
                let a = replay(3, 10, &[]);
                for b in [replay(2, 10, &[]), replay(3, 11, &[])] {
                    assert_eq!(a.checked_add(&b), Err(Error::IncompatibleParameters));
                    assert_eq!(
                        a.as_ref().checked_add(&b.as_ref()),
                        Err(Error::IncompatibleParameters)
                    );
                }
                for gp in [3, 4, 255] {
                    assert_eq!(a.downsample(gp), Err(Error::IncompatibleParameters));
                }
            }

            #[test]
            fn total_overflow_is_rejected_even_when_individual_buckets_fit() {
                let config = Config::new(3, 10).unwrap();
                let a = $owned::from_parts(config, vec![1], vec![<$count>::MAX]).unwrap();
                let b = $owned::from_parts(config, vec![2], vec![1]).unwrap();
                let before = (a.clone(), b.clone());
                assert_eq!(a.checked_add(&b), Err(Error::Overflow));
                assert_eq!(a.checked_add(&a), Err(Error::Overflow));
                assert_eq!(a.as_ref().checked_add(&b.as_ref()), Err(Error::Overflow));
                assert_eq!((a, b), before);
                let near = $owned::from_parts(config, vec![1], vec![<$count>::MAX - 1]).unwrap();
                let merged = near.checked_add(&before.1).unwrap();
                assert_eq!(merged.count(), &[<$count>::MAX - 1, <$count>::MAX]);
                assert_eq!(
                    merged.downsample(0).unwrap().total_count(),
                    <$count>::MAX as u64
                );
            }

            #[test]
            fn repeated_prefixes_are_zero_deltas_and_are_omitted_from_output() {
                let config = Config::new(3, 10).unwrap();
                let a = $owned::from_parts(config, vec![1, 2, 3], vec![2, 2, 5]).unwrap();
                let b = $owned::from_parts(config, vec![2, 4], vec![1, 4]).unwrap();
                let merged = a.checked_add(&b).unwrap();
                assert_eq!(merged.index(), &[1, 2, 3, 4]);
                assert_eq!(merged.count(), &[2, 3, 6, 9]);
                let empty = replay(3, 10, &[]);
                assert_eq!(a.checked_add(&empty).unwrap().index(), &[1, 3]);
                assert_eq!(a.downsample(2).unwrap().index(), &[1, 3]);
            }

            #[cfg(feature = "serde")]
            #[test]
            fn transforms_use_validated_midpoint_means_after_deserialization() {
                let original = replay(3, 10, &[17, 17, 35]);
                let mut encoded = serde_json::to_value(&original).unwrap();
                encoded["mean"] = serde_json::json!(123.456);
                let owned: $owned = serde_json::from_value(encoded).unwrap();
                let view = owned.as_ref();
                let empty = replay(3, 10, &[]);
                assert_eq!(
                    view.checked_add(&empty.as_ref()).unwrap().mean(),
                    original.mean()
                );
                let coarse = view.downsample(0).unwrap();
                assert_eq!(coarse.mean(), Some((23.5 * 2.0 + 47.5) / 3.0));
                assert_eq!(view.mean(), original.mean());
            }

            #[test]
            fn many_window_merge_then_downsample_matches_raw_replay() {
                for count in [2, 8, 64] {
                    let mut all = Vec::new();
                    let mut merged = replay(5, 12, &[]);
                    for window in 0..count {
                        let values: Vec<_> =
                            (0..64).map(|i| (i * 53 + window * 197) % 4096).collect();
                        merged = merged.checked_add(&replay(5, 12, &values)).unwrap();
                        all.extend(values);
                    }
                    assert_eq!(merged, replay(5, 12, &all));
                    assert_eq!(merged.downsample(2).unwrap(), replay(2, 12, &all));
                }
            }

            #[cfg(feature = "serde")]
            #[test]
            fn transformed_outputs_roundtrip_through_existing_serialization() {
                let a = replay(3, 10, &[17, 17, 35]);
                for output in [a.checked_add(&a).unwrap(), a.downsample(1).unwrap()] {
                    let bytes = serde_json::to_vec(&output).unwrap();
                    let decoded: $owned = serde_json::from_slice(&bytes).unwrap();
                    assert_eq!(decoded, output);
                }
            }
        }
    };
}

transform_tests!(
    wide,
    CumulativeROHistogram,
    CumulativeROHistogramRef,
    Histogram,
    u64
);
transform_tests!(
    narrow,
    CumulativeROHistogram32,
    CumulativeROHistogram32Ref,
    Histogram32,
    u32
);

#[test]
fn mixed_width_merge_requires_explicit_widening_and_checked_narrowing() {
    let config = Config::new(3, 10).unwrap();
    let narrow = CumulativeROHistogram32::from_parts(config, vec![1], vec![u32::MAX]).unwrap();
    let wide = CumulativeROHistogram::from_parts(config, vec![2], vec![1]).unwrap();
    let result = CumulativeROHistogram::from(&narrow)
        .checked_add(&wide)
        .unwrap();
    assert_eq!(result.total_count(), u32::MAX as u64 + 1);
    assert_eq!(
        CumulativeROHistogram32::try_from(&result),
        Err(Error::Overflow)
    );
}
