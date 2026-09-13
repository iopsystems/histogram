//! Full dense reports over prebuilt inputs; recording and request creation are untimed.
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use histogram::{Histogram, Histogram32};
use std::hint::black_box;

fn observations(shape: &str) -> Vec<u64> {
    let mut seed = 42u64;
    let mut unit = || {
        seed = seed.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    };
    (0..4096)
        .map(|_| match shape {
            "low" => 0,
            "high" => (1 << 30) - 1,
            "lognormal" => {
                let z = (-2.0 * unit().max(f64::MIN_POSITIVE).ln()).sqrt()
                    * (std::f64::consts::TAU * unit()).cos();
                ((1000.0f64.ln() + 1.5 * z).exp() as u64).clamp(1, (1 << 30) - 1)
            }
            "log_uniform" => (2.0f64.powf(unit() * 30.0) as u64).clamp(1, (1 << 30) - 1),
            _ => unreachable!(),
        })
        .collect()
}

macro_rules! benchmark_width {
    ($function:ident, $histogram:ty, $width:literal) => {
        fn $function(c: &mut Criterion) {
            let requests: Vec<_> = (0..256)
                .map(|i| [(i as f64 + 1.0) / 512.0, 0.9, 0.99, 0.999, 1.0])
                .collect();
            for gp in [0, 7, 10] {
                for shape in ["low", "high", "lognormal", "log_uniform"] {
                    let mut h = <$histogram>::new(gp, 30).unwrap();
                    for value in observations(shape) {
                        h.increment(value).unwrap();
                    }
                    let mut group =
                        c.benchmark_group(format!("dense_queries/{}/gp{gp}/{shape}", $width));
                    group.throughput(Throughput::Elements(1));
                    for n in [0, 1, 5] {
                        group.bench_with_input(BenchmarkId::new("full_report", n), &n, |b, &n| {
                            let mut cursor = 0usize;
                            b.iter(|| {
                                let qs: &[f64] = if n == 1 {
                                    &[0.99]
                                } else {
                                    &requests[cursor % requests.len()][..n]
                                };
                                cursor = cursor.wrapping_add(1);
                                let result = black_box(&h).quantiles(black_box(qs)).unwrap();
                                // Keep map, total, min and max observable, including destruction.
                                black_box(&result);
                            });
                        });
                    }
                    group.finish();
                }
            }
        }
    };
}
benchmark_width!(wide, Histogram, "u64");
benchmark_width!(narrow, Histogram32, "u32");
criterion_group!(benches, wide, narrow);
criterion_main!(benches);
