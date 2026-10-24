//! Dense quantile scans must match a plain reference walk for every counter
//! width, including totals that exceed `u64` and chunks whose counts sum past
//! `u64`, which the vectorized fast path must hand off to wide arithmetic.
use histogram::{Bucket, Histogram, Histogram32, SampleQuantiles};

fn lcg(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *seed >> 11
}

/// Nearest-rank reference: full u128 prefix walk, no chunking.
fn reference(counts: &[u128], q: f64) -> Option<(usize, u128)> {
    let total: u128 = counts.iter().sum();
    if total == 0 {
        return None;
    }
    let target = ((q * total as f64).ceil() as u128).clamp(1, total);
    if q == 1.0 {
        return counts.iter().rposition(|&c| c != 0).map(|i| (i, total));
    }
    let mut partial = 0u128;
    for (i, &c) in counts.iter().enumerate() {
        partial += c;
        if partial >= target {
            return Some((i, total));
        }
    }
    unreachable!()
}

const QUANTILES: [f64; 9] = [0.0, 0.001, 0.25, 0.5, 0.9, 0.99, 0.999, 0.9999, 1.0];

/// The allocation-free `quantile_buckets_into` must select the same buckets as
/// the map API on the same histogram, here with the requests in reverse order.
macro_rules! agree_into {
    ($h:expr) => {{
        let mut reversed = QUANTILES;
        reversed.reverse();
        let mut out: [Option<Bucket>; 9] = std::array::from_fn(|_| None);
        assert_eq!($h.quantile_buckets_into(&reversed, &mut out).unwrap(), 9);
        match $h.quantiles(&QUANTILES).unwrap() {
            None => assert!(out.iter().all(Option::is_none)),
            Some(map) => {
                for (q, slot) in reversed.iter().zip(out.iter()) {
                    let b = slot.as_ref().expect("buckets-into slot filled");
                    let e = map.get(&histogram::Quantile::new(*q).unwrap()).unwrap();
                    assert_eq!(
                        (b.start(), b.end(), b.count()),
                        (e.start(), e.end(), e.count()),
                        "q={q}"
                    );
                }
            }
        }
    }};
}

fn check_u64(gp: u8, counts: Vec<u64>) {
    let h = Histogram::from_buckets(gp, 30, counts.clone()).unwrap();
    agree_into!(h);
    let wide: Vec<u128> = counts.iter().map(|&c| c as u128).collect();
    let ranges: Vec<(u64, u64)> = (&h).into_iter().map(|b| (b.start(), b.end())).collect();
    check(&h, &wide, |i| counts[i], &ranges);
}

fn check_u32(gp: u8, counts: Vec<u32>) {
    let h = Histogram32::from_buckets(gp, 30, counts.clone()).unwrap();
    agree_into!(h);
    let wide: Vec<u128> = counts.iter().map(|&c| c as u128).collect();
    let ranges: Vec<(u64, u64)> = (&h).into_iter().map(|b| (b.start(), b.end())).collect();
    check(&h, &wide, |i| counts[i] as u64, &ranges);
}

fn check<H: SampleQuantiles>(
    h: &H,
    wide: &[u128],
    count_at: impl Fn(usize) -> u64,
    ranges: &[(u64, u64)],
) {
    let total: u128 = wide.iter().sum();
    let result = h.quantiles(&QUANTILES).unwrap();
    if total == 0 {
        assert!(result.is_none());
        return;
    }
    let result = result.unwrap();
    assert_eq!(result.total_count(), total, "total");
    let min_idx = wide.iter().position(|&c| c != 0).unwrap();
    let max_idx = wide.iter().rposition(|&c| c != 0).unwrap();
    assert_eq!(result.min().count(), count_at(min_idx), "min bucket count");
    assert_eq!(result.max().count(), count_at(max_idx), "max bucket count");
    for (q, bucket) in result.entries().iter() {
        let (idx, _) = reference(wide, q.as_f64()).unwrap();
        assert_eq!(
            (bucket.start(), bucket.end()),
            ranges[idx],
            "q={} bucket range",
            q.as_f64()
        );
        assert_eq!(
            bucket.count(),
            count_at(idx),
            "q={} bucket count",
            q.as_f64()
        );
        let min_b: &Bucket = result.min();
        assert!(
            bucket.start() >= min_b.start(),
            "q={} below min",
            q.as_f64()
        );
    }
    // Every requested quantile must be present exactly once.
    assert_eq!(result.entries().len(), QUANTILES.len());
}

#[test]
fn random_u64_histograms_match_reference() {
    let mut seed = 7u64;
    for gp in [0u8, 3, 7, 10] {
        let n = Histogram::new(gp, 30).unwrap().config().total_buckets();
        for density in [1usize, 8, 64, 1000, usize::MAX] {
            let counts: Vec<u64> = (0..n)
                .map(|i| {
                    if density == usize::MAX || lcg(&mut seed) as usize % density == 0 {
                        (lcg(&mut seed) % 5000) * (1 + (i as u64 % 3))
                    } else {
                        0
                    }
                })
                .collect();
            check_u64(gp, counts);
        }
    }
}

#[test]
fn random_u32_histograms_match_reference() {
    let mut seed = 11u64;
    for gp in [0u8, 7, 10] {
        let n = Histogram32::new(gp, 30).unwrap().config().total_buckets();
        for density in [1usize, 16, 500] {
            let counts: Vec<u32> = (0..n)
                .map(|_| {
                    if lcg(&mut seed) as usize % density == 0 {
                        (lcg(&mut seed) % 70000) as u32
                    } else {
                        0
                    }
                })
                .collect();
            check_u32(gp, counts);
        }
    }
}

#[test]
fn u32_counts_at_their_maximum_do_not_overflow_the_scan() {
    let n = Histogram32::new(7, 30).unwrap().config().total_buckets();
    // Every bucket at u32::MAX: the total is far beyond u64 for wide geometries' worth of
    // chunks and the eight-wide chunk sums must stay exact.
    check_u32(7, vec![u32::MAX; n]);
}

#[test]
fn u64_chunks_that_overflow_u64_fall_back_to_wide_arithmetic() {
    let n = Histogram::new(7, 30).unwrap().config().total_buckets();
    // Eight adjacent counts of 2^63 sum to 2^66, past u64: the word-sized chunk
    // sums must be discarded and the whole report taken on the wide path.
    let mut counts = vec![0u64; n];
    for c in counts.iter_mut().skip(100).take(8) {
        *c = 1u64 << 63;
    }
    counts[200] = u64::MAX;
    counts[201] = u64::MAX;
    counts[n - 1] = 3;
    check_u64(7, counts.clone());
    // Mixed chunk: one huge count among small ones.
    let mut mixed = vec![1u64; n];
    mixed[37] = u64::MAX - 5;
    mixed[38] = 1 << 61;
    check_u64(7, mixed);
}

#[test]
fn totals_beyond_u64_are_exact() {
    let n = Histogram::new(0, 30).unwrap().config().total_buckets();
    let counts = vec![u64::MAX; n];
    let h = Histogram::from_buckets(0, 30, counts).unwrap();
    let r = h.quantiles(&[0.5]).unwrap().unwrap();
    assert_eq!(r.total_count(), (u64::MAX as u128) * n as u128);
}

#[test]
fn unsorted_and_repeated_requests_keep_their_order_without_allocating() {
    let mut h = Histogram::new(7, 30).unwrap();
    for v in [1u64, 5, 5, 9, 1000, 1000, 1000, 1 << 29] {
        h.increment(v).unwrap();
    }
    let requests = [0.99, 0.5, 0.99, 0.0, 1.0, 0.5];
    let mut out = [None, None, None, None, None, None];
    assert_eq!(h.quantile_buckets_into(&requests, &mut out).unwrap(), 6);
    let map = h.quantiles(&requests).unwrap().unwrap();
    for (q, slot) in requests.iter().zip(out.iter()) {
        let b = slot.as_ref().unwrap();
        let m = map.get(&histogram::Quantile::new(*q).unwrap()).unwrap();
        assert_eq!(
            (b.start(), b.end(), b.count()),
            (m.start(), m.end(), m.count()),
            "q={q}"
        );
    }
    assert_eq!(
        out[0].as_ref().map(|b| (b.start(), b.end())),
        out[2].as_ref().map(|b| (b.start(), b.end()))
    );
}

#[test]
fn empty_histogram_reports_nothing() {
    let h = Histogram::new(7, 30).unwrap();
    assert!(h.quantiles(&[0.5]).unwrap().is_none());
    let mut populated = Histogram::new(7, 30).unwrap();
    populated.increment(3).unwrap();
    let mut out = [populated.quantile_bucket(0.5).unwrap()];
    assert!(out[0].is_some());
    assert_eq!(h.quantile_buckets_into(&[0.5], &mut out).unwrap(), 1);
    assert!(
        out[0].is_none(),
        "stale output slots are cleared for an empty histogram"
    );
}
