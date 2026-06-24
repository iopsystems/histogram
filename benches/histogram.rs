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
    };
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
