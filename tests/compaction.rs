use histogram::Config;

macro_rules! contract {
    ($module:ident, $hist:ty, $count:ty, $cumulative:expr) => {
        mod $module {
            use super::*;
            type Snapshot = $hist;

            fn oversized(n: usize) -> Snapshot {
                let config = Config::new(7, 20).unwrap();
                let mut index = Vec::with_capacity(n + 4096);
                let mut count: Vec<$count> = Vec::with_capacity(n + 4096);
                for i in 0..n {
                    index.push((i * config.total_buckets() / n) as u32);
                    count.push(if $cumulative { (i + 1) as $count } else { 1 });
                }
                Snapshot::from_parts(config, index, count).unwrap()
            }

            #[test]
            fn releases_spare_capacity_without_changing_observations() {
                let full = Config::new(7, 20).unwrap().total_buckets();
                for n in [0, 1, 10, 100, 1000, full] {
                    let mut h = oversized(n);
                    let before = h.clone();
                    let before_buckets: Vec<_> = h.iter().collect();
                    h.shrink_to_fit();
                    assert_eq!(h, before);
                    assert_eq!(h.iter().collect::<Vec<_>>(), before_buckets);
                    for q in [0.0, 0.5, 0.99, 1.0] {
                        let expected = before.quantile(q).unwrap();
                        let actual = h.quantile(q).unwrap();
                        assert_eq!(actual, expected);
                    }
                    let (_, indices, counts) = h.into_parts();
                    assert_eq!(indices.len(), n);
                    assert_eq!(counts.len(), n);
                    assert!(indices.capacity() >= n && indices.capacity() < n + 4096);
                    assert!(counts.capacity() >= n && counts.capacity() < n + 4096);
                    if n == 0 {
                        assert_eq!((indices.capacity(), counts.capacity()), (0, 0));
                    }
                }
            }

            #[test]
            fn repeated_compaction_preserves_data_and_capacity() {
                let mut h = oversized(100);
                h.shrink_to_fit();
                let (config, indices, counts) = h.into_parts();
                let capacities = (indices.capacity(), counts.capacity());
                let mut h = Snapshot::from_parts(config, indices, counts).unwrap();
                let before = h.clone();
                h.shrink_to_fit();
                assert_eq!(h, before);
                let (_, indices, counts) = h.into_parts();
                assert_eq!((indices.capacity(), counts.capacity()), capacities);
            }

            #[cfg(feature = "serde")]
            #[test]
            fn serialization_is_unchanged() {
                for n in [0, 1, 1000] {
                    let mut h = oversized(n);
                    let before = serde_json::to_vec(&h).unwrap();
                    h.shrink_to_fit();
                    assert_eq!(serde_json::to_vec(&h).unwrap(), before);
                    let restored: Snapshot = serde_json::from_slice(&before).unwrap();
                    assert_eq!(h, restored);
                }
            }
        }
    };
}

contract!(sparse64, histogram::SparseHistogram, u64, false);
contract!(sparse32, histogram::SparseHistogram32, u32, false);
contract!(cumulative64, histogram::CumulativeROHistogram, u64, true);
contract!(cumulative32, histogram::CumulativeROHistogram32, u32, true);

#[test]
fn cached_means_are_preserved_bit_for_bit() {
    let mut h = histogram::Histogram::new(10, 30).unwrap();
    for v in [1, 2, 7, 999, 100000] {
        h.increment(v).unwrap();
    }
    let mut wide = histogram::CumulativeROHistogram::from(&h);
    let mut narrow = histogram::CumulativeROHistogram32::try_from(&h).unwrap();
    let before = (
        wide.mean().map(f64::to_bits),
        narrow.mean().map(f64::to_bits),
    );
    wide.shrink_to_fit();
    narrow.shrink_to_fit();
    assert_eq!(
        (
            wide.mean().map(f64::to_bits),
            narrow.mean().map(f64::to_bits)
        ),
        before
    );
}

#[cfg(feature = "serde")]
#[test]
fn compaction_does_not_recompute_a_deserialized_cached_mean() {
    let mut dense = histogram::Histogram::new(7, 20).unwrap();
    dense.increment(1000).unwrap();
    macro_rules! check {
        ($snapshot:expr, $ty:ty) => {{
            let mut encoded = serde_json::to_value($snapshot).unwrap();
            encoded["mean"] = serde_json::json!(123.456);
            let mut h: $ty = serde_json::from_value(encoded).unwrap();
            h.shrink_to_fit();
            assert_eq!(h.mean().unwrap().to_bits(), 123.456f64.to_bits());
        }};
    }
    check!(
        histogram::CumulativeROHistogram::from(&dense),
        histogram::CumulativeROHistogram
    );
    check!(
        histogram::CumulativeROHistogram32::try_from(&dense).unwrap(),
        histogram::CumulativeROHistogram32
    );
}
