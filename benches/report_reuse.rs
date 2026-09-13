//! Native API costs on prebuilt histograms; allocation and construction boundaries
//! are explicit. No RNG or request construction occurs in the timed closures.
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use histogram::{AtomicHistogram, AtomicHistogram32, Bucket, Histogram, Histogram32};
use std::hint::black_box;

macro_rules! benchmark_width {
    ($function:ident, $dense:ty, $atomic:ty, $width:literal) => {
        fn $function(c: &mut Criterion) {
            for gp in [7, 10] {
                for shape in ["low8", "high8", "spread8", "full"] {
                    let mut h = <$dense>::new(gp, 30).unwrap();
                    let n = h.as_slice().len();
                    let indices: Vec<_> = match shape {
                        "low8" => (0..8).collect(),
                        "high8" => (n - 8..n).collect(),
                        "spread8" => (0..8).map(|i| i * (n - 1) / 7).collect(),
                        "full" => (0..n).collect(),
                        _ => unreachable!(),
                    };
                    for &index in &indices {
                        h.as_mut_slice()[index] = 512;
                    }
                    let mut group =
                        c.benchmark_group(format!("report_reuse/{}/gp{gp}/{shape}", $width));
                    for (label, qs) in [
                        ("scalar", &[0.99][..]),
                        ("batch5", &[0.99, 0.5, 1.0, 0.9, 0.999][..]),
                    ] {
                        group.bench_with_input(BenchmarkId::new("map", label), &qs, |b, qs| {
                            b.iter(|| black_box(black_box(&h).quantiles(black_box(qs)).unwrap()));
                        });
                        let mut output: [Option<Bucket>; 5] = std::array::from_fn(|_| None);
                        group.bench_with_input(BenchmarkId::new("buckets", label), &qs, |b, qs| {
                            b.iter(|| {
                                if qs.len() == 1 {
                                    black_box(
                                        black_box(&h).quantile_bucket(black_box(qs[0])).unwrap(),
                                    );
                                } else {
                                    black_box(&h)
                                        .quantile_buckets_into(
                                            black_box(qs),
                                            black_box(&mut output),
                                        )
                                        .unwrap();
                                    black_box(&output);
                                }
                            });
                        });
                    }
                    group.finish();
                }
                let atomic = <$atomic>::new(gp, 30).unwrap();
                atomic.add(1000, 4096).unwrap();
                let h = atomic.load();
                let mut destination = <$dense>::with_config(&h.config());
                let mut group =
                    c.benchmark_group(format!("report_reuse/{}/gp{gp}/lifecycle", $width));
                group.bench_function("load_new", |b| {
                    b.iter(|| black_box(black_box(&atomic).load()))
                });
                group.bench_function("load_into", |b| {
                    b.iter(|| {
                        black_box(&atomic)
                            .load_into(black_box(&mut destination))
                            .unwrap();
                        black_box(&destination);
                    })
                });
                group.bench_function("merge_new", |b| {
                    b.iter_batched_ref(
                        || h.clone(),
                        |destination| {
                            let result = black_box(destination).checked_add(black_box(&h)).unwrap();
                            black_box(&result);
                            // Destroy allocating outputs inside the timed closure.
                        },
                        BatchSize::SmallInput,
                    )
                });
                group.bench_function("merge_assign", |b| {
                    b.iter_batched_ref(
                        || h.clone(),
                        |destination| {
                            black_box(&mut *destination)
                                .checked_add_assign(black_box(&h))
                                .unwrap();
                            black_box(destination);
                        },
                        BatchSize::SmallInput,
                    )
                });
                // Fresh populated source is setup, excluded from drain timing.
                group.bench_function("drain_new", |b| {
                    b.iter_batched_ref(
                        || {
                            let source = <$atomic>::new(gp, 30).unwrap();
                            source.add(1000, 4096).unwrap();
                            source
                        },
                        |source| {
                            let result = black_box(source).drain();
                            black_box(&result);
                        },
                        BatchSize::SmallInput,
                    )
                });
                group.bench_function("drain_into", |b| {
                    b.iter_batched_ref(
                        || {
                            let source = <$atomic>::new(gp, 30).unwrap();
                            source.add(1000, 4096).unwrap();
                            source
                        },
                        |source| {
                            black_box(source)
                                .drain_into(black_box(&mut destination))
                                .unwrap();
                            black_box(&destination);
                        },
                        BatchSize::SmallInput,
                    )
                });
                group.finish();
            }
        }
    };
}
benchmark_width!(wide, Histogram, AtomicHistogram, "u64");
benchmark_width!(narrow, Histogram32, AtomicHistogram32, "u32");
criterion_group!(benches, wide, narrow);
criterion_main!(benches);
