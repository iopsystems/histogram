//! Run with `cargo bench --bench cumulative_queries`.
//! Query outputs are normalized to the same buckets; construction is not timed.
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use histogram::{Bucket, Config, CumulativeROHistogram, CumulativeROHistogram32, Quantile};
use std::hint::black_box;

fn queries() -> Vec<[f64; 5]> {
    // Deterministic, varying, unsorted requests built outside the timed regions.
    // Mix a varying quantile with common median/tail requests.
    (0..256)
        .map(|i| {
            let q = (i as f64 + 1.0) / 257.0;
            [q, 0.99, 0.5, 0.9, 1.0]
        })
        .collect()
}

macro_rules! benchmark_width {
    ($function:ident, $histogram:ty, $count:ty, $width:literal) => {
        fn $function(c: &mut Criterion) {
            let config = Config::new(10, 30).unwrap();
            let requests = queries();
            for occupied in [0usize, 1, 8, 4096] {
                for bank_size in [1usize, 2048] {
                    // Avoid a redundant 2048-object empty-bank measurement.
                    if occupied == 0 && bank_size != 1 {
                        continue;
                    }
                    let seed = <$histogram>::from_parts(
                        config,
                        (0..occupied)
                            .map(|i| (i * config.total_buckets() / occupied) as u32)
                            .collect(),
                        (1..=occupied).map(|i| i as $count).collect(),
                    )
                    .unwrap();
                    let bank: Vec<_> = (0..bank_size).map(|_| seed.clone()).collect();
                    let access = if bank_size == 1 {
                        "hot"
                    } else {
                        "rotating2048"
                    };
                    let mut group = c.benchmark_group(format!(
                        "cumulative_queries/{}/{occupied}/{access}",
                        $width
                    ));
                    group.throughput(Throughput::Elements(1)); // One report, not one requested quantile.
                    for query_count in [1usize, 5] {
                        group.bench_with_input(
                            BenchmarkId::new("map", query_count),
                            &query_count,
                            |b, &n| {
                                let mut cursor = 0usize;
                                let mut out: [Option<Bucket>; 5] = std::array::from_fn(|_| None);
                                b.iter(|| {
                                    let h = black_box(&bank[cursor % bank.len()]);
                                    let qs = black_box(
                                        &requests[(cursor / bank.len() + cursor) % requests.len()]
                                            [..n],
                                    );
                                    cursor = cursor.wrapping_add(1);
                                    let result = h.quantiles(qs).unwrap();
                                    for (slot, &q) in out.iter_mut().zip(qs) {
                                        *slot = result.as_ref().map(|r| {
                                            r.get(&Quantile::new(q).unwrap()).unwrap().clone()
                                        });
                                    }
                                    black_box(&out[..n]);
                                });
                            },
                        );
                        group.bench_with_input(
                            BenchmarkId::new("scalar", query_count),
                            &query_count,
                            |b, &n| {
                                let mut cursor = 0usize;
                                let mut out: [Option<Bucket>; 5] = std::array::from_fn(|_| None);
                                b.iter(|| {
                                    let h = black_box(&bank[cursor % bank.len()]);
                                    let qs = black_box(
                                        &requests[(cursor / bank.len() + cursor) % requests.len()]
                                            [..n],
                                    );
                                    cursor = cursor.wrapping_add(1);
                                    for (slot, &q) in out.iter_mut().zip(qs) {
                                        *slot = h.quantile_bucket(q).unwrap();
                                    }
                                    black_box(&out[..n]);
                                });
                            },
                        );
                        group.bench_with_input(
                            BenchmarkId::new("buffer", query_count),
                            &query_count,
                            |b, &n| {
                                let mut cursor = 0usize;
                                let mut out: [Option<Bucket>; 5] = std::array::from_fn(|_| None);
                                b.iter(|| {
                                    let h = black_box(&bank[cursor % bank.len()]);
                                    let qs = black_box(
                                        &requests[(cursor / bank.len() + cursor) % requests.len()]
                                            [..n],
                                    );
                                    cursor = cursor.wrapping_add(1);
                                    h.quantile_buckets_into(qs, &mut out).unwrap();
                                    black_box(&out[..n]);
                                });
                            },
                        );
                    }
                    group.finish();
                }
            }
        }
    };
}

benchmark_width!(u64_queries, CumulativeROHistogram, u64, "u64");
benchmark_width!(u32_queries, CumulativeROHistogram32, u32, "u32");
criterion_group!(benches, u64_queries, u32_queries);
criterion_main!(benches);
