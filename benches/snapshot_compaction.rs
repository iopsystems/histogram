//! Snapshot build/retire cost, with and without opt-in compaction.
//! Dense source construction is excluded; snapshot destruction is included.
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use histogram::{
    CumulativeROHistogram, CumulativeROHistogram32, Histogram, SparseHistogram, SparseHistogram32,
};
use std::hint::black_box;

fn benchmarks(c: &mut Criterion) {
    let geometry = Histogram::new(10, 30).unwrap();
    let values: Vec<_> = geometry.iter().map(|b| b.start()).collect();
    for occupied in [0, 1, 10, 100, 1000, 2423, values.len()] {
        let mut source = Histogram::new(10, 30).unwrap();
        for i in 0..occupied {
            source
                .increment(values[i * values.len() / occupied])
                .unwrap();
        }
        let mut group = c.benchmark_group(format!("snapshot_compaction/{occupied}"));
        macro_rules! bench {
            ($name:literal, $build:expr) => {
                for compact in [false, true] {
                    group.bench_with_input(
                        BenchmarkId::new($name, if compact { "build_compact" } else { "build" }),
                        &compact,
                        |b, &compact| {
                            b.iter(|| {
                                let mut snapshot = $build;
                                if compact {
                                    snapshot.shrink_to_fit();
                                }
                                black_box(snapshot)
                            })
                        },
                    );
                }
            };
        }
        bench!("sparse64", SparseHistogram::from(black_box(&source)));
        bench!(
            "sparse32",
            SparseHistogram32::try_from(black_box(&source)).unwrap()
        );
        bench!(
            "cumulative64",
            CumulativeROHistogram::from(black_box(&source))
        );
        bench!(
            "cumulative32",
            CumulativeROHistogram32::try_from(black_box(&source)).unwrap()
        );
        group.finish();
    }
}

criterion_group!(benches, benchmarks);
criterion_main!(benches);
