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

        // A stream that crosses the linear and logarithmic regions in an order
        // the branch predictor cannot learn, so the cost of the region test and
        // the range check shows up instead of being predicted away.
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

/// Values spanning 0..2^40 with a log-uniform spread, from a fixed-seed LCG, so
/// about half fall in the linear region of a gp7 histogram and half spread
/// across the logarithmic buckets.
fn mixed_stream() -> Vec<u64> {
    let mut seed = 0x9E37_79B9_7F4A_7C15_u64;
    (0..STREAM_LEN)
        .map(|_| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let bits = (seed >> 58) as u32; // 0..64
            let shift = bits % 41; // 0..=40
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
