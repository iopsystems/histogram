use criterion::{Criterion, Throughput, criterion_group, criterion_main};

macro_rules! benchmark {
    ($name:tt, $histogram:ident, $c:ident) => {
        let mut group = $c.benchmark_group($name);
        group.throughput(Throughput::Elements(1));
        group.bench_function("increment/1", |b| b.iter(|| $histogram.increment(1)));
        group.bench_function("increment/max", |b| {
            b.iter(|| $histogram.increment(u64::MAX))
        });
        group.finish();

        let mut group = $c.benchmark_group(concat!($name, "/stream"));
        group.throughput(Throughput::Elements(STREAM_LEN as u64));
        let stream = mixed_stream();
        group.bench_function("increment/mixed", |b| {
            b.iter(|| {
                for &v in stream.iter() {
                    let _ = $histogram.increment(std::hint::black_box(v));
                }
            })
        });
        group.finish();
    };
}

const STREAM_LEN: usize = 4096;

/// Fixed-seed LCG stream spanning 0..2^40 to exercise both indexing regions.
/// The modulo mapping favors shifts 0..=22 over 23..=40.
/// For gp7, 1,184 of the 4,096 values (29%) fall in the linear region;
/// the rest fall in logarithmic buckets.
fn mixed_stream() -> Vec<u64> {
    let mut seed = 0x9E37_79B9_7F4A_7C15_u64;
    (0..STREAM_LEN)
        .map(|_| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let bits = (seed >> 58) as u32;
            let shift = bits % 41;
            let mantissa = seed >> 20;
            (mantissa % (1u64 << shift.max(1))) | if shift == 0 { 0 } else { 1u64 << (shift - 1) }
        })
        .collect()
}

fn histogram_u64(c: &mut Criterion) {
    let mut histogram = histogram::Histogram::new(7, 64).unwrap();
    benchmark!("histogram/u64", histogram, c);
}

fn histogram_u32(c: &mut Criterion) {
    let mut histogram = histogram::Histogram32::new(7, 64).unwrap();
    benchmark!("histogram/u32", histogram, c);
}

fn atomic_u64(c: &mut Criterion) {
    let histogram = histogram::AtomicHistogram::new(7, 64).unwrap();
    benchmark!("atomic_histogram/u64", histogram, c);
}

fn atomic_u32(c: &mut Criterion) {
    let histogram = histogram::AtomicHistogram32::new(7, 64).unwrap();
    benchmark!("atomic_histogram/u32", histogram, c);
}

criterion_group!(
    benches,
    histogram_u64,
    histogram_u32,
    atomic_u64,
    atomic_u32
);
criterion_main!(benches);
