use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use histogram::{CumulativeROHistogram, CumulativeROHistogram32, Histogram, Histogram32};

macro_rules! benchmark_width {
    ($function:ident, $label:literal, $cumulative:ident, $dense:ident, $count:ty) => {
        fn $function(c: &mut Criterion) {
            let mut group = c.benchmark_group($label);
            for occupied in [8, 2048] {
                let mut source = $dense::new(10, 30).unwrap();
                let buckets = source.as_slice().len();
                for i in 0..occupied {
                    source.as_mut_slice()[i * buckets / occupied] = (i % 7 + 1) as $count;
                }
                let snapshot = $cumulative::from(&source);
                for windows in [2, 8, 64] {
                    let inputs = vec![snapshot.clone(); windows];
                    for reports in [1, 100] {
                        for path in ["direct", "dense_adapter"] {
                            let id = BenchmarkId::new(
                                path,
                                format!("{windows}w-{occupied}b-{reports}r"),
                            );
                            group.bench_function(id, |b| {
                                b.iter(|| {
                                    let inputs = black_box(&inputs);
                                    let result = if path == "direct" {
                                        let mut result = inputs[0].clone();
                                        for input in &inputs[1..] {
                                            result = result.checked_add(input).unwrap();
                                        }
                                        result
                                    } else {
                                        let total: u128 =
                                            inputs.iter().map(|h| h.total_count() as u128).sum();
                                        assert!(total <= <$count>::MAX as u128);
                                        let mut result = $dense::with_config(&inputs[0].config());
                                        for input in inputs {
                                            for bucket in input.iter() {
                                                result
                                                    .add(bucket.start(), bucket.count() as $count)
                                                    .unwrap();
                                            }
                                        }
                                        $cumulative::from(&result)
                                    };
                                    for _ in 0..reports {
                                        black_box(result.quantile_bucket(black_box(0.99)).unwrap());
                                    }
                                    black_box(result);
                                });
                            });
                        }
                    }
                }
                group.bench_with_input(
                    BenchmarkId::new("downsample_gp7", occupied),
                    &snapshot,
                    |b, input| {
                        b.iter(|| black_box(black_box(input).downsample(7).unwrap()));
                    },
                );
            }
            group.finish();
        }
    };
}

benchmark_width!(wide, "cumulative64", CumulativeROHistogram, Histogram, u64);
benchmark_width!(
    narrow,
    "cumulative32",
    CumulativeROHistogram32,
    Histogram32,
    u32
);
criterion_group!(benches, wide, narrow);
criterion_main!(benches);
