use histogram::{Error, Histogram, Histogram32};
use std::{hint::black_box, time::Instant};

macro_rules! define_probe {
    ($sum:ident, $fused:ident, $run:ident, $hist:ty, $count:ty) => {
        // Experimental implementation using only public APIs. No recorder change.
        fn $sum(sources: &[&$hist]) -> Result<$hist, Error> {
            let first = sources.first().ok_or(Error::IncompatibleParameters)?;
            if sources.iter().any(|h| h.config() != first.config()) {
                return Err(Error::IncompatibleParameters);
            }
            let mut result = (*first).clone();
            for source in &sources[1..] {
                for (dst, &src) in result.as_mut_slice().iter_mut().zip(source.as_slice()) {
                    *dst = dst.checked_add(src).ok_or(Error::Overflow)?;
                }
            }
            Ok(result)
        }
        // The output is private: wrapped intermediate sums cannot escape if
        // any lane overflows. No caller-owned histogram is modified.
        fn $fused(sources: &[&$hist]) -> Result<$hist, Error> {
            let first = sources.first().ok_or(Error::IncompatibleParameters)?;
            if sources.iter().any(|h| h.config() != first.config()) {
                return Err(Error::IncompatibleParameters);
            }
            let mut result = (*first).clone();
            for source in &sources[1..] {
                let mut overflow = false;
                for (dst, &src) in result.as_mut_slice().iter_mut().zip(source.as_slice()) {
                    let (sum, carry) = dst.overflowing_add(src);
                    *dst = sum;
                    overflow |= carry;
                }
                if overflow {
                    return Err(Error::Overflow);
                }
            }
            Ok(result)
        }
        fn $run(
            gp: u8,
            max: u8,
            windows: usize,
            shape: &str,
            method: &str,
            report: bool,
            iterations: u64,
        ) -> u128 {
            let mut sources = Vec::new();
            let n = <$hist>::new(gp, max).unwrap().as_slice().len();
            let mut expected = vec![0u128; n];
            for w in 0..windows {
                let mut h = <$hist>::new(gp, max).unwrap();
                for (i, count) in h.as_mut_slice().iter_mut().enumerate() {
                    if shape == "full" || (i >= n / 2 && i < (n / 2 + 16).min(n)) {
                        *count = (1 + (i + w * 13) % 31) as $count;
                        expected[i] += *count as u128;
                    }
                }
                sources.push(h);
            }
            let refs: Vec<_> = sources.iter().collect();
            let merge = || {
                let refs = black_box(&refs);
                match method {
                    "owned" => $sum(refs).unwrap(),
                    "fused" => $fused(refs).unwrap(),
                    "pairwise_clone" => {
                        let mut out = refs[0].clone();
                        for h in &refs[1..] {
                            out.checked_add_assign(h).unwrap();
                        }
                        out
                    }
                    "pairwise_zero" => {
                        let mut out = <$hist>::with_config(&refs[0].config());
                        for h in refs.iter() {
                            out.checked_add_assign(h).unwrap();
                        }
                        out
                    }
                    "owned_pairwise" => {
                        let mut out = refs[0].checked_add(refs[1]).unwrap();
                        for h in &refs[2..] {
                            out = out.checked_add(h).unwrap();
                        }
                        out
                    }
                    _ => unreachable!(),
                }
            };
            let check = merge();
            assert_eq!(
                check
                    .as_slice()
                    .iter()
                    .map(|&v| v as u128)
                    .collect::<Vec<_>>(),
                expected
            );
            assert_eq!(
                check
                    .quantiles(&[0.0, 0.5, 0.9, 0.99, 1.0])
                    .unwrap()
                    .unwrap()
                    .total_count(),
                expected.iter().sum::<u128>()
            );
            let start = Instant::now();
            for _ in 0..iterations {
                let out = merge();
                if report {
                    black_box(
                        out.quantiles(black_box(&[0.0, 0.5, 0.9, 0.99, 1.0]))
                            .unwrap(),
                    );
                }
                black_box(out); // Owned output is dropped inside timing.
            }
            start.elapsed().as_nanos()
        }
    };
}
define_probe!(sum64, fused64, run64, Histogram, u64);
define_probe!(sum32, fused32, run32, Histogram32, u32);
fn next(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}
fn main() {
    let mut seed = std::env::args()
        .nth(1)
        .unwrap_or("42".into())
        .parse::<u64>()
        .unwrap();
    let target_ms: f64 = std::env::args()
        .nth(2)
        .unwrap_or("10".into())
        .parse()
        .unwrap();
    assert!(seed != 0 && target_ms.is_finite() && target_ms > 0.0);
    let target_ns = (target_ms * 1_000_000.0).max(1.0) as u128;
    let mut cases = Vec::new();
    for width in [32, 64] {
        for (gp, max) in [(0, 1), (0, 7), (7, 30), (10, 30)] {
            for windows in [2, 8, 64] {
                for shape in ["clustered", "full"] {
                    for report in [false, true] {
                        for method in [
                            "fused",
                            "owned",
                            "pairwise_clone",
                            "pairwise_zero",
                            "owned_pairwise",
                        ] {
                            cases.push((width, gp, max, windows, shape, report, method));
                        }
                    }
                }
            }
        }
    }
    for i in (1..cases.len()).rev() {
        let j = next(&mut seed) as usize % (i + 1);
        cases.swap(i, j);
    }
    println!("width,gp,max_power,windows,shape,report,method,iterations,elapsed_ns,ns_per_set");
    for (width, gp, max, windows, shape, report, method) in cases {
        let run = if width == 32 { run32 } else { run64 };
        let mut iters = 1;
        loop {
            let ns = run(gp, max, windows, shape, method, report, iters);
            if ns >= target_ns / 5 || iters >= 1_000_000 {
                iters = ((iters as u128 * target_ns / ns.max(1)).clamp(1, 1_000_000)) as u64;
                break;
            }
            iters *= 10;
        }
        let ns = run(gp, max, windows, shape, method, report, iters);
        println!(
            "{width},{gp},{max},{windows},{shape},{report},{method},{iters},{ns},{:.3}",
            ns as f64 / iters as f64
        );
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    macro_rules! check {
        ($name:ident,$hist:ty,$sum:ident,$count:ty) => {
            #[test]
            fn $name() {
                let mut a = <$hist>::new(0, 2).unwrap();
                a.as_mut_slice().copy_from_slice(&[1, 2, 3]);
                let mut b = <$hist>::new(0, 2).unwrap();
                b.as_mut_slice().copy_from_slice(&[4, 5, 6]);
                let out = $sum(&[&a, &b]).unwrap();
                assert_eq!(out.as_slice(), &[5, 7, 9]);
                assert_eq!($sum(&[&a]).unwrap(), a);
                assert_eq!($sum(&[]), Err(Error::IncompatibleParameters));
                let c = <$hist>::new(1, 2).unwrap();
                assert_eq!($sum(&[&a, &c]), Err(Error::IncompatibleParameters));
                for i in 0..3 {
                    let mut b = b.clone();
                    b.as_mut_slice()[i] = <$count>::MAX;
                    let a0 = a.clone();
                    let b0 = b.clone();
                    assert_eq!($sum(&[&a, &b]), Err(Error::Overflow));
                    assert_eq!(a, a0);
                    assert_eq!(b, b0);
                }
                let mut late = <$hist>::new(0, 2).unwrap();
                late.as_mut_slice()[2] = <$count>::MAX - 8;
                assert_eq!($sum(&[&a, &b, &late]), Err(Error::Overflow));
                assert_eq!(a.as_slice(), &[1, 2, 3]);
                assert_eq!(b.as_slice(), &[4, 5, 6]);
                let mut a = <$hist>::new(0, 2).unwrap();
                a.as_mut_slice().fill(<$count>::MAX);
                let z = <$hist>::new(0, 2).unwrap();
                assert_eq!($sum(&[&a, &z]).unwrap(), a);
            }
        };
    }
    check!(u32_contract, Histogram32, sum32, u32);
    check!(u64_contract, Histogram, sum64, u64);
    check!(fused_u32_contract, Histogram32, fused32, u32);
    check!(fused_u64_contract, Histogram, fused64, u64);
    macro_rules! boundaries {
        ($name:ident, $hist:ty, $fused:ident, $count:ty) => {
            #[test]
            fn $name() {
                let geometries: Vec<_> =
                    (1..=32).map(|m| (0, m)).chain([(3, 30), (7, 30)]).collect();
                for (gp, max) in geometries {
                    let mut a = <$hist>::new(gp, max).unwrap();
                    let mut b = a.clone();
                    a.as_mut_slice().fill(<$count>::MAX - 2);
                    b.as_mut_slice().fill(1);
                    let exact = $fused(&[&a, &b, &b]).unwrap();
                    assert!(exact.as_slice().iter().all(|&v| v == <$count>::MAX));
                    let before_a = a.clone();
                    let before_b = b.clone();
                    for index in 0..a.as_slice().len() {
                        let mut late = <$hist>::new(gp, max).unwrap();
                        late.as_mut_slice()[index] = 2;
                        let before_late = late.clone();
                        assert_eq!($fused(&[&a, &b, &late]), Err(Error::Overflow));
                        assert_eq!(a, before_a);
                        assert_eq!(b, before_b);
                        assert_eq!(late, before_late);
                    }
                }
            }
        };
    }
    boundaries!(fused_u32_vector_boundaries, Histogram32, fused32, u32);
    boundaries!(fused_u64_vector_boundaries, Histogram, fused64, u64);
}
