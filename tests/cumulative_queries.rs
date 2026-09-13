//! The public query contract, including allocations made by this thread only.
use histogram::{Bucket, Error};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;

struct TrackingAllocator;
thread_local! {
    static ALLOCATIONS: Cell<Option<usize>> = const { Cell::new(None) };
}

fn count_allocation() {
    let _ = ALLOCATIONS.try_with(|counter| {
        if let Some(count) = counter.get() {
            counter.set(Some(count + 1));
        }
    });
}

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count_allocation();
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count_allocation();
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        count_allocation();
        unsafe { System.realloc(ptr, layout, size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: TrackingAllocator = TrackingAllocator;

fn allocations_during<T>(operation: impl FnOnce() -> T) -> (T, usize) {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            ALLOCATIONS.with(|c| c.set(None));
        }
    }
    ALLOCATIONS.with(|c| c.set(Some(0)));
    let reset = Reset;
    let result = operation();
    let count = ALLOCATIONS.with(|c| c.get().unwrap());
    drop(reset);
    (result, count)
}

macro_rules! contract {
    ($module:ident, $dense:ty, $cumulative:ty) => {
        mod $module {
            use super::*;
            type Dense = $dense;
            type Cumulative = $cumulative;

            fn fixture() -> Cumulative {
                let mut h = Dense::new(4, 16).unwrap();
                for v in [0, 0, 1, 3, 3, 3, 10000, 65535] {
                    h.increment(v).unwrap();
                }
                Cumulative::from(&h)
            }

            #[test]
            fn scalar_preserves_rank_range_and_individual_count() {
                let h = fixture();
                let sorted = [0, 0, 1, 3, 3, 3, 10000, 65535];
                for q in [0.0, -0.0, 0.125, 0.5, 0.75, 0.875, 0.99, 1.0] {
                    let old = h.quantile(q).unwrap().unwrap();
                    let expected = old.get(&histogram::Quantile::new(q).unwrap()).unwrap();
                    let bucket = h.quantile_bucket(q).unwrap().unwrap();
                    assert_eq!(&bucket, expected);
                    assert_eq!(
                        h.as_ref().quantile_bucket(q).unwrap().as_ref(),
                        Some(expected)
                    );
                    let rank = ((q * sorted.len() as f64).ceil() as usize).max(1);
                    let exact = sorted[rank - 1];
                    assert!(bucket.start() <= exact && exact <= bucket.end());
                    assert_eq!(
                        bucket.count(),
                        sorted
                            .iter()
                            .filter(|&&v| bucket.range().contains(&v))
                            .count() as u64
                    );
                }
            }

            #[test]
            fn batch_preserves_order_duplicates_and_unused_output() {
                let h = fixture();
                let qs = [0.99, 0.0, 0.5, 0.5, 1.0, -0.0];
                let sentinel = h.quantile_bucket(0.875).unwrap();
                let mut out: [Option<Bucket>; 8] = std::array::from_fn(|_| sentinel.clone());
                assert_eq!(h.quantile_buckets_into(&qs, &mut out).unwrap(), qs.len());
                for (q, value) in qs.into_iter().zip(&out) {
                    assert_eq!(*value, h.quantile_bucket(q).unwrap());
                }
                assert_eq!(&out[6..], &[sentinel.clone(), sentinel]);
                let mut borrowed = std::array::from_fn::<_, 8, _>(|_| None);
                h.as_ref()
                    .quantile_buckets_into(&qs, &mut borrowed)
                    .unwrap();
                assert_eq!(&out[..6], &borrowed[..6]);
            }

            #[test]
            fn empty_histogram_and_empty_requests_have_explicit_output() {
                let populated = fixture();
                let empty = Cumulative::from(&Dense::new(4, 16).unwrap());
                assert_eq!(empty.quantile_bucket(0.5).unwrap(), None);
                assert_eq!(empty.as_ref().quantile_bucket(1.0).unwrap(), None);
                let mut out = [
                    populated.quantile_bucket(0.5).unwrap(),
                    populated.quantile_bucket(0.5).unwrap(),
                ];
                let saved = out.clone();
                assert_eq!(empty.quantile_buckets_into(&[], &mut out).unwrap(), 0);
                assert_eq!(out, saved);
                assert_eq!(
                    empty.quantile_buckets_into(&[0.0, 1.0], &mut out).unwrap(),
                    2
                );
                assert_eq!(out, [None, None]);
                assert_eq!(populated.quantile_buckets_into(&[], &mut []).unwrap(), 0);
            }

            #[test]
            fn invalid_requests_do_not_partially_modify_output() {
                let h = fixture();
                let empty = Cumulative::from(&Dense::new(4, 16).unwrap());
                for q in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.1, 1.1] {
                    for source in [&h, &empty] {
                        assert_eq!(source.quantile_bucket(q), Err(Error::InvalidQuantile));
                        assert_eq!(
                            source.as_ref().quantile_bucket(q),
                            Err(Error::InvalidQuantile)
                        );
                        let mut out = [h.quantile_bucket(1.0).unwrap(), None];
                        let saved = out.clone();
                        assert_eq!(
                            source.quantile_buckets_into(&[0.5, q], &mut out),
                            Err(Error::InvalidQuantile)
                        );
                        assert_eq!(out, saved);
                        assert_eq!(
                            source.as_ref().quantile_buckets_into(&[0.5, q], &mut out),
                            Err(Error::InvalidQuantile)
                        );
                        assert_eq!(out, saved);
                    }
                }
            }

            #[test]
            fn too_small_output_is_an_error_even_for_empty_histograms() {
                let h = fixture();
                let empty = Cumulative::from(&Dense::new(4, 16).unwrap());
                for source in [&h, &empty] {
                    let mut out = [h.quantile_bucket(1.0).unwrap()];
                    let saved = out.clone();
                    assert_eq!(
                        source.quantile_buckets_into(&[0.0, 1.0], &mut out),
                        Err(Error::InsufficientOutputCapacity {
                            required: 2,
                            available: 1
                        })
                    );
                    assert_eq!(out, saved);
                    // Quantile validation takes precedence over capacity validation.
                    assert_eq!(
                        source.quantile_buckets_into(&[f64::NAN], &mut []),
                        Err(Error::InvalidQuantile)
                    );
                }
            }

            #[test]
            fn scalar_and_batch_queries_allocate_nothing() {
                let h = fixture();
                let view = h.as_ref();
                let qs = [0.99, 0.0, 0.5, 0.5, 1.0];
                let mut out: [Option<Bucket>; 5] = std::array::from_fn(|_| None);
                let (old, old_allocs) = allocations_during(|| h.quantiles(black_box(&qs)).unwrap());
                assert!(
                    old.is_some() && old_allocs > 0,
                    "allocation counter must detect the existing API"
                );
                let (_, count) = allocations_during(|| {
                    black_box(h.quantile_bucket(black_box(0.99)).unwrap());
                    black_box(view.quantile_bucket(black_box(0.99)).unwrap());
                    h.quantile_buckets_into(black_box(&qs), black_box(&mut out))
                        .unwrap();
                    view.quantile_buckets_into(black_box(&qs), black_box(&mut out))
                        .unwrap();
                });
                assert_eq!(count, 0);
                assert!(out.iter().all(Option::is_some));
            }

            #[test]
            fn large_snapshots_match_sorted_observations() {
                let mut raw: Vec<u64> = (0..4096u64)
                    .map(|i| i.wrapping_mul(2654435761) % (1 << 30))
                    .collect();
                let mut dense = Dense::new(10, 30).unwrap();
                for &v in &raw {
                    dense.increment(v).unwrap();
                }
                raw.sort_unstable();
                let h = Cumulative::from(&dense);
                assert!(h.count().len() > 64);
                let qs = [1.0, 0.001, 0.125, 0.99, 0.5, 0.999, 0.0, 0.9];
                let mut out: [Option<Bucket>; 8] = std::array::from_fn(|_| None);
                let (_, allocations) = allocations_during(|| {
                    h.quantile_buckets_into(&qs, &mut out).unwrap();
                });
                assert_eq!(allocations, 0);
                for (&q, slot) in qs.iter().zip(&out) {
                    let bucket = slot.as_ref().unwrap();
                    let rank = ((q * raw.len() as f64).ceil() as usize).max(1);
                    assert!(bucket.range().contains(&raw[rank - 1]));
                    let expected = h.quantile(q).unwrap().unwrap();
                    assert_eq!(
                        expected.get(&histogram::Quantile::new(q).unwrap()),
                        Some(bucket)
                    );
                    assert_eq!(
                        h.as_ref().quantile_bucket(q).unwrap().as_ref(),
                        Some(bucket)
                    );
                }
            }

            #[test]
            fn wrapped_zero_total_matches_existing_empty_result() {
                let mut dense = Dense::new(4, 16).unwrap();
                dense.add(1, !0).unwrap();
                dense.add(3, 1).unwrap();
                let h = Cumulative::from(&dense);
                assert!(!h.count().is_empty());
                assert_eq!(*h.count().last().unwrap(), 0);
                let mut out = [fixture().quantile_bucket(0.5).unwrap()];
                for q in [0.0, 0.5, 1.0] {
                    assert!(h.quantile(q).unwrap().is_none());
                    assert_eq!(h.quantile_bucket(q).unwrap(), None);
                    assert_eq!(h.as_ref().quantile_bucket(q).unwrap(), None);
                }
                assert_eq!(h.quantile_buckets_into(&[0.5], &mut out).unwrap(), 1);
                assert_eq!(out, [None]);
                out[0] = fixture().quantile_bucket(0.5).unwrap();
                h.as_ref().quantile_buckets_into(&[0.5], &mut out).unwrap();
                assert_eq!(out, [None]);
            }

            #[test]
            fn empty_and_error_query_paths_allocate_nothing() {
                let h = Cumulative::from(&Dense::new(4, 16).unwrap());
                let mut out: [Option<Bucket>; 2] = [None, None];
                let (_, count) = allocations_during(|| {
                    black_box(h.quantile_bucket(0.5).unwrap());
                    h.quantile_buckets_into(&[0.0, 1.0], &mut out).unwrap();
                    black_box(h.quantile_buckets_into(&[0.0, 1.0], &mut []).unwrap_err());
                    black_box(
                        h.as_ref()
                            .quantile_buckets_into(&[f64::NAN], &mut out)
                            .unwrap_err(),
                    );
                });
                assert_eq!(count, 0);
            }
        }
    };
}

contract!(
    u64_counts,
    histogram::Histogram,
    histogram::CumulativeROHistogram
);

#[test]
fn u64_maximum_total_and_prefix_plateaus_keep_existing_semantics() {
    let config = histogram::Config::new(4, 16).unwrap();
    let h = histogram::CumulativeROHistogram::from_parts(
        config,
        vec![1, 3],
        vec![u64::MAX - 1, u64::MAX],
    )
    .unwrap();
    let max = h.quantile_bucket(1.0).unwrap().unwrap();
    assert_eq!(max.range(), 3..=3);
    assert_eq!(max.count(), 1);
    let median = h.quantile_bucket(0.5).unwrap().unwrap();
    assert_eq!(median.count(), u64::MAX - 1);
    let plateau =
        histogram::CumulativeROHistogram::from_parts(config, vec![1, 3, 7], vec![5, 5, 9]).unwrap();
    assert_eq!(
        plateau.quantile_bucket(0.5).unwrap().unwrap().range(),
        1..=1
    );
    assert_eq!(
        plateau.quantile_bucket(0.6).unwrap().unwrap().range(),
        7..=7
    );
    for source in [&h, &plateau] {
        for q in [0.0, 0.5, 0.6, 1.0] {
            let old = source.quantile(q).unwrap().unwrap();
            assert_eq!(
                source.quantile_bucket(q).unwrap().as_ref(),
                old.get(&histogram::Quantile::new(q).unwrap())
            );
        }
    }
}
contract!(
    u32_counts,
    histogram::Histogram32,
    histogram::CumulativeROHistogram32
);
