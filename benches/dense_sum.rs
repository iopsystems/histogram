//! Fresh owned outputs, prepared inputs, and optional batched reporting.
use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use histogram::{Histogram, Histogram32};

macro_rules! benchmark_width {
    ($function:ident, $label:literal, $dense:ident, $count:ty) => {
        fn $function(c: &mut Criterion) {
            let mut group = c.benchmark_group($label);
            for (gp, max) in [(0, 1), (0, 7), (7, 30), (10, 30)] {
                for windows in [2, 8, 64] {
                    for shape in ["clustered", "full"] {
                        let mut inputs = vec![$dense::new(gp, max).unwrap(); windows];
                        let len = inputs[0].as_slice().len();
                        for (window, input) in inputs.iter_mut().enumerate() {
                            for (i, count) in input.as_mut_slice().iter_mut().enumerate() {
                                if shape == "full" || (len / 2..len / 2 + 16).contains(&i) {
                                    *count = ((i + window * 7) % 31 + 1) as $count;
                                }
                            }
                        }
                        let refs: Vec<_> = inputs.iter().collect();
                        let expected: Vec<u128> = (0..len)
                            .map(|i| inputs.iter().map(|h| u128::from(h.as_slice()[i])).sum())
                            .collect();
                        let run = |method, refs: &[&$dense]| match method {
                            "checked_sum" => $dense::checked_sum(refs).unwrap(),
                            "clone_first" => {
                                let mut result = refs[0].clone();
                                for input in &refs[1..] {
                                    result.checked_add_assign(input).unwrap();
                                }
                                result
                            }
                            "scalar_owned" => {
                                let first = refs[0];
                                assert!(refs.iter().all(|h| h.config() == first.config()));
                                let mut result = first.clone();
                                for input in &refs[1..] {
                                    for (dst, &src) in
                                        result.as_mut_slice().iter_mut().zip(input.as_slice())
                                    {
                                        *dst = dst.checked_add(src).unwrap();
                                    }
                                }
                                result
                            }
                            _ => unreachable!(),
                        };
                        for method in ["checked_sum", "clone_first", "scalar_owned"] {
                            assert_eq!(
                                run(method, &refs)
                                    .as_slice()
                                    .iter()
                                    .copied()
                                    .map(u128::from)
                                    .collect::<Vec<_>>(),
                                expected
                            );
                            for reports in [false, true] {
                                let id = BenchmarkId::new(
                                    method,
                                    format!("gp{gp}-max{max}-{windows}w-{shape}-report{reports}"),
                                );
                                group.bench_function(id, |b| {
                                    b.iter(|| {
                                        let result = run(method, black_box(&refs));
                                        if reports {
                                            black_box(
                                                result
                                                    .quantiles(black_box(&[
                                                        0.0, 0.5, 0.9, 0.99, 1.0,
                                                    ]))
                                                    .unwrap(),
                                            );
                                        }
                                        // The owned output is dropped in the timed iteration.
                                        black_box(result);
                                    });
                                });
                            }
                        }
                    }
                }
            }
            group.finish();
        }
    };
}

benchmark_width!(wide, "dense_sum64", Histogram, u64);
benchmark_width!(narrow, "dense_sum32", Histogram32, u32);
criterion_group!(benches, wide, narrow);
criterion_main!(benches);
