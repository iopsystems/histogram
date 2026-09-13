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
    ($module:ident, $dense:ty, $atomic:ty, $count:ty) => {
        mod $module {
            use super::*;
            type Dense = $dense;
            type Atomic = $atomic;

            #[test]
            fn dense_queries_match_observations_and_preserve_request_order() {
                let mut h = Dense::new(7, 30).unwrap();
                let mut raw: Vec<_> = (0..4096u64).map(|i| i * 2654435761 % (1 << 30)).collect();
                for &v in &raw {
                    h.increment(v).unwrap();
                }
                raw.sort_unstable();
                let qs = [0.99, 0.0, 0.5, 0.5, 1.0, -0.0, 0.001, 0.9];
                let mut out: [Option<Bucket>; 9] = std::array::from_fn(|_| None);
                let sentinel = h.quantile_bucket(0.7).unwrap();
                out[8] = sentinel.clone();
                assert_eq!(h.quantile_buckets_into(&qs, &mut out).unwrap(), qs.len());
                let map = h.quantiles(&qs).unwrap().unwrap();
                for (&q, slot) in qs.iter().zip(&out) {
                    let bucket = slot.as_ref().unwrap();
                    let rank = ((q * raw.len() as f64).ceil() as usize).max(1);
                    assert!(bucket.range().contains(&raw[rank - 1]));
                    assert_eq!(map.get(&histogram::Quantile::new(q).unwrap()), Some(bucket));
                    assert_eq!(h.quantile_bucket(q).unwrap(), *slot);
                    assert_eq!(
                        bucket.count(),
                        raw.iter().filter(|v| bucket.range().contains(v)).count() as u64
                    );
                }
                assert_eq!(out[8], sentinel);
            }

            #[test]
            fn query_validation_is_transactional_and_empty_output_is_explicit() {
                let mut h = Dense::new(4, 16).unwrap();
                h.increment(3).unwrap();
                let empty = Dense::new(4, 16).unwrap();
                let sentinel = h.quantile_bucket(0.5).unwrap();
                for source in [&h, &empty] {
                    for q in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.01, 1.01] {
                        let mut out = [sentinel.clone(), None];
                        let saved = out.clone();
                        assert_eq!(source.quantile_bucket(q), Err(Error::InvalidQuantile));
                        assert_eq!(
                            source.quantile_buckets_into(&[0.5, q], &mut out),
                            Err(Error::InvalidQuantile)
                        );
                        assert_eq!(out, saved);
                        assert_eq!(
                            source.quantile_buckets_into(&[q], &mut []),
                            Err(Error::InvalidQuantile)
                        );
                    }
                    let mut out = [sentinel.clone()];
                    assert_eq!(
                        source.quantile_buckets_into(&[0.0, 1.0], &mut out),
                        Err(Error::InsufficientOutputCapacity {
                            required: 2,
                            available: 1
                        })
                    );
                    assert_eq!(out[0], sentinel);
                    assert_eq!(source.quantile_buckets_into(&[], &mut out), Ok(0));
                    assert_eq!(out[0], sentinel);
                }
                let mut out = [sentinel.clone(), sentinel];
                assert_eq!(empty.quantile_buckets_into(&[0.0, 1.0], &mut out), Ok(2));
                assert_eq!(out, [None, None]);
                assert_eq!(empty.quantile_bucket(0.5), Ok(None));
            }

            #[test]
            fn queries_widen_totals_and_observe_raw_mutation_and_reset() {
                let mut h = Dense::new(4, 16).unwrap();
                h.add(3, <$count>::MAX).unwrap();
                h.add(17, <$count>::MAX).unwrap();
                assert_eq!(h.quantile_bucket(0.25).unwrap().unwrap().start(), 3);
                assert_eq!(h.quantile_bucket(0.75).unwrap().unwrap().start(), 17);
                // Interior ranks retain the existing f64 rounding convention.
                let old = h.quantile(0.5).unwrap().unwrap();
                assert_eq!(
                    h.quantile_bucket(0.5).unwrap().as_ref(),
                    old.get(&histogram::Quantile::new(0.5).unwrap())
                );
                assert_eq!(h.quantile_bucket(1.0).unwrap().unwrap().start(), 17);
                h.as_mut_slice().fill(0);
                h.increment(10000).unwrap();
                assert!(
                    h.quantile_bucket(0.0)
                        .unwrap()
                        .unwrap()
                        .range()
                        .contains(&10000)
                );
                h.reset();
                assert_eq!(h.quantile_bucket(0.99), Ok(None));
            }

            #[test]
            fn checked_merge_reuses_storage_and_rejects_errors_before_mutating() {
                let mut dst = Dense::new(4, 16).unwrap();
                dst.increment(0).unwrap();
                let mut src = Dense::new(4, 16).unwrap();
                src.add(5, 3).unwrap();
                let pointer = dst.as_slice().as_ptr();
                let expected = dst.checked_add(&src).unwrap();
                dst.checked_add_assign(&src).unwrap();
                assert_eq!(dst, expected);
                assert_eq!(dst.as_slice().as_ptr(), pointer);
                *dst.as_mut_slice().last_mut().unwrap() = <$count>::MAX;
                *src.as_mut_slice().last_mut().unwrap() = 1;
                let saved = dst.clone();
                assert_eq!(dst.checked_add_assign(&src), Err(Error::Overflow));
                assert_eq!(dst, saved);
                assert_eq!(
                    dst.checked_add_assign(&Dense::new(5, 16).unwrap()),
                    Err(Error::IncompatibleParameters)
                );
                assert_eq!(dst, saved);
            }

            #[test]
            fn snapshots_overwrite_reused_storage_and_drain_only_on_success() {
                let src = Atomic::new(4, 16).unwrap();
                src.add(5, 3).unwrap();
                let mut dst = Dense::new(4, 16).unwrap();
                dst.increment(100).unwrap();
                let pointer = dst.as_slice().as_ptr();
                src.load_into(&mut dst).unwrap();
                assert_eq!(dst, src.load());
                assert_eq!(dst.as_slice().as_ptr(), pointer);
                let mut bad = Dense::new(5, 16).unwrap();
                bad.increment(7).unwrap();
                let saved = bad.clone();
                assert_eq!(src.load_into(&mut bad), Err(Error::IncompatibleParameters));
                assert_eq!(src.drain_into(&mut bad), Err(Error::IncompatibleParameters));
                assert_eq!(bad, saved);
                assert_eq!(src.load(), dst);
                src.drain_into(&mut dst).unwrap();
                assert_eq!(dst.as_slice().as_ptr(), pointer);
                assert_eq!(dst.quantile_bucket(0.5).unwrap().unwrap().count(), 3);
                assert_eq!(src.load().quantile_bucket(0.5), Ok(None));
                src.drain_into(&mut dst).unwrap();
                assert_eq!(dst.quantile_bucket(0.5), Ok(None));
            }

            #[test]
            fn concurrent_drains_conserve_each_bucket() {
                let src = Atomic::new(5, 6).unwrap();
                let mut accumulated = Dense::new(5, 6).unwrap();
                let mut snapshot = Dense::new(5, 6).unwrap();
                std::thread::scope(|scope| {
                    for writer in 0..4 {
                        let src = &src;
                        scope.spawn(move || {
                            for i in 0..2000 {
                                src.increment((i + writer) % 64).unwrap();
                            }
                        });
                    }
                    for _ in 0..64 {
                        src.drain_into(&mut snapshot).unwrap();
                        accumulated.checked_add_assign(&snapshot).unwrap();
                        std::thread::yield_now();
                    }
                });
                src.drain_into(&mut snapshot).unwrap();
                accumulated.checked_add_assign(&snapshot).unwrap();
                let mut expected = Dense::new(5, 6).unwrap();
                for writer in 0..4 {
                    for i in 0..2000 {
                        expected.increment((i + writer) % 64).unwrap();
                    }
                }
                assert_eq!(accumulated, expected);
            }

            #[test]
            fn matching_bucket_lengths_do_not_imply_compatible_geometry() {
                let mut first = Dense::new(0, 7).unwrap();
                let second = Dense::new(1, 4).unwrap();
                assert_eq!(first.as_slice().len(), second.as_slice().len());
                let atomic = Atomic::new(1, 4).unwrap();
                first.increment(0).unwrap();
                let saved = first.clone();
                assert_eq!(
                    first.checked_add_assign(&second),
                    Err(Error::IncompatibleParameters)
                );
                assert_eq!(
                    atomic.load_into(&mut first),
                    Err(Error::IncompatibleParameters)
                );
                assert_eq!(
                    atomic.drain_into(&mut first),
                    Err(Error::IncompatibleParameters)
                );
                assert_eq!(first, saved);
            }

            #[test]
            fn reporting_paths_allocate_nothing() {
                let src = Atomic::new(4, 16).unwrap();
                src.add(5, 3).unwrap();
                let mut dst = src.load();
                let rhs = dst.clone();
                let qs = [1.0, 0.5, 0.0, 0.5];
                let mut out: [Option<Bucket>; 4] = std::array::from_fn(|_| None);
                let (old, count) = allocations_during(|| dst.quantiles(black_box(&qs)).unwrap());
                assert!(old.is_some() && count > 0);
                let (_, count) = allocations_during(|| {
                    black_box(dst.quantile_bucket(black_box(0.99)).unwrap());
                    dst.quantile_buckets_into(black_box(&qs), black_box(&mut out))
                        .unwrap();
                    src.load_into(black_box(&mut dst)).unwrap();
                    src.drain_into(black_box(&mut dst)).unwrap();
                    dst.checked_add_assign(black_box(&rhs)).unwrap();
                    dst.reset();
                    black_box(dst.quantile_bucket(0.5).unwrap());
                    dst.quantile_buckets_into(&qs, &mut out).unwrap();
                    black_box(
                        dst.quantile_buckets_into(&[f64::NAN], &mut out)
                            .unwrap_err(),
                    );
                    black_box(dst.quantile_buckets_into(&qs, &mut []).unwrap_err());
                });
                assert_eq!(count, 0);
            }
        }
    };
}
contract!(
    u64_counts,
    histogram::Histogram,
    histogram::AtomicHistogram,
    u64
);
contract!(
    u32_counts,
    histogram::Histogram32,
    histogram::AtomicHistogram32,
    u32
);

#[test]
fn endpoint_rank_rounding_never_selects_an_empty_tail() {
    let mut h = histogram::Histogram::new(4, 16).unwrap();
    h.add(3, u64::MAX - 1).unwrap();
    h.increment(17).unwrap();
    assert_eq!(h.quantile_bucket(1.0).unwrap().unwrap().start(), 17);
    let mut out = [None, None];
    h.quantile_buckets_into(&[0.5, 1.0], &mut out).unwrap();
    assert_eq!(out[1].as_ref().unwrap().start(), 17);
}
