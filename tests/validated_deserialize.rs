#![cfg(feature = "serde")]

use histogram::{
    Config, CumulativeROHistogram, CumulativeROHistogram32, Histogram, Histogram32,
    SparseHistogram, SparseHistogram32,
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Value, json};

// These independent wire fixtures reproduce the original derived field order.
// Keep them separate from the library's constructors/deserialization helpers.
#[derive(Clone, Serialize, Deserialize)]
#[serde(rename = "Config")]
struct LegacyConfig {
    max: u64,
    grouping_power: u8,
    max_value_power: u8,
    cutoff_power: u8,
    cutoff_value: u64,
    lower_bin_count: u32,
    upper_bin_divisions: u32,
    upper_bin_count: u32,
}
fn legacy_config() -> LegacyConfig {
    LegacyConfig {
        max: 7,
        grouping_power: 1,
        max_value_power: 3,
        cutoff_power: 2,
        cutoff_value: 4,
        lower_bin_count: 4,
        upper_bin_divisions: 2,
        upper_bin_count: 2,
    }
}
#[derive(Serialize, Deserialize)]
struct LegacyDense<T> {
    config: LegacyConfig,
    buckets: Box<[T]>,
}
#[derive(Serialize, Deserialize)]
struct LegacySparse<T> {
    config: LegacyConfig,
    index: Vec<u32>,
    count: Vec<T>,
}
#[derive(Serialize, Deserialize)]
struct LegacyCumulative<T> {
    config: LegacyConfig,
    index: Vec<u32>,
    count: Vec<T>,
    mean: Option<f64>,
}

fn assert_legacy_roundtrip<T: DeserializeOwned + Serialize, W: Serialize>(wire: &W) -> T {
    let json = serde_json::to_value(wire).unwrap();
    let actual: T = serde_json::from_value(json.clone()).unwrap();
    assert_eq!(serde_json::to_value(&actual).unwrap(), json);
    let bytes = bincode::serialize(wire).unwrap();
    let actual: T = bincode::deserialize(&bytes).unwrap();
    assert_eq!(bincode::serialize(&actual).unwrap(), bytes);
    actual
}

fn assert_rejected<T: DeserializeOwned>(wire: &Value) {
    assert!(
        serde_json::from_value::<T>(wire.clone()).is_err(),
        "accepted {wire}"
    );
}

#[test]
fn config_legacy_json_and_binary_remain_compatible() {
    let config: Config = assert_legacy_roundtrip(&legacy_config());
    assert_eq!(config.total_buckets(), 6);
    assert_eq!(config.error(), 50.0);
}

#[test]
fn config_rejects_each_inconsistent_derived_field() {
    let valid = serde_json::to_value(legacy_config()).unwrap();
    for (field, bad) in [
        ("max", 9),
        ("grouping_power", 0),
        ("max_value_power", 4),
        ("cutoff_power", 3),
        ("cutoff_value", 8),
        ("lower_bin_count", 5),
        ("upper_bin_divisions", 4),
        ("upper_bin_count", 3),
    ] {
        let mut wire = valid.clone();
        wire[field] = json!(bad);
        assert_rejected::<Config>(&wire);
    }
}

#[test]
fn config_accepts_the_pre_0_11_4_max_encoding() {
    // histogram 0.8.2 through 0.11.3 serialized `max` as `2^max_value_power`.
    let mut wire = legacy_config();
    wire.max = 1 << wire.max_value_power;
    let json = serde_json::to_value(&wire).unwrap();
    let from_json: Config = serde_json::from_value(json).unwrap();
    let from_bincode: Config = bincode::deserialize(&bincode::serialize(&wire).unwrap()).unwrap();
    let current = Config::new(wire.grouping_power, wire.max_value_power).unwrap();
    assert_eq!(from_json, current);
    assert_eq!(from_bincode, current);

    // It re-serializes in the current encoding.
    assert_eq!(serde_json::to_value(from_json).unwrap()["max"], json!(7));

    // A histogram carrying such a config decodes too.
    let dense: Histogram = serde_json::from_value(
        serde_json::to_value(LegacyDense {
            config: wire,
            buckets: vec![2_u64, 3, 5, 0, 0, 0].into(),
        })
        .unwrap(),
    )
    .unwrap();
    assert_eq!(dense.config(), current);
    assert_eq!(dense.as_slice(), &[2, 3, 5, 0, 0, 0]);
}

#[test]
fn full_range_configs_still_round_trip() {
    // max_value_power 64 has no legacy encoding (`max` was `u64::MAX` in
    // every version); the legacy check must not shift by 64.
    for gp in [0, 1, 7, 20] {
        let config = Config::new(gp, 64).unwrap();
        let json = serde_json::to_value(config).unwrap();
        assert_eq!(serde_json::from_value::<Config>(json).unwrap(), config);
        let bytes = bincode::serialize(&config).unwrap();
        assert_eq!(bincode::deserialize::<Config>(&bytes).unwrap(), config);
    }
}

#[test]
fn config_rejects_a_max_that_is_neither_encoding() {
    for max in [0, 6, 9, u64::MAX] {
        let mut wire = legacy_config();
        wire.max = max;
        assert_rejected::<Config>(&serde_json::to_value(&wire).unwrap());
    }
}

#[test]
fn config_rejects_invalid_and_overflowing_geometry_without_panicking() {
    for (gp, max) in [
        (3, 3),
        (0, 0),
        (1, 65),
        (255, 255),
        (63, 64),
        (32, 64),
        (30, 33),
        (27, 64),
    ] {
        let mut wire = legacy_config();
        wire.grouping_power = gp;
        wire.max_value_power = max;
        assert_rejected::<Config>(&serde_json::to_value(&wire).unwrap());
        assert!(bincode::deserialize::<Config>(&bincode::serialize(&wire).unwrap()).is_err());
    }
}

macro_rules! histogram_tests {
    ($module:ident, $dense:ident, $sparse:ident, $cumulative:ident, $count:ty) => {
        mod $module {
            use super::*;

            #[test]
            fn valid_dense_sparse_and_cumulative_legacy_roundtrips() {
                let dense: $dense = assert_legacy_roundtrip(&LegacyDense {
                    config: legacy_config(), buckets: vec![2 as $count, 3, 5, 0, 0, 0].into(),
                });
                assert_eq!(dense.as_slice(), &[2, 3, 5, 0, 0, 0]);
                let sparse: $sparse = assert_legacy_roundtrip(&LegacySparse {
                    config: legacy_config(), index: vec![0, 1, 2], count: vec![2 as $count, 3, 5],
                });
                assert_eq!(sparse, $sparse::from(&dense));
                let cumulative: $cumulative = assert_legacy_roundtrip(&LegacyCumulative {
                    config: legacy_config(), index: vec![0, 1, 2], count: vec![2 as $count, 5, 10], mean: Some(1.3),
                });
                assert_eq!(cumulative, $cumulative::from(&dense));
                assert_eq!(cumulative.mean(), Some(1.3));
                let bucket = cumulative.quantile_bucket(0.5).unwrap().unwrap();
                assert_eq!((bucket.start(), bucket.end(), bucket.count()), (1, 1, 3));
            }

            #[test]
            fn empty_legacy_roundtrips_remain_valid() {
                let _: $dense = assert_legacy_roundtrip(&LegacyDense {
                    config: legacy_config(), buckets: vec![0 as $count; 6].into(),
                });
                let _: $sparse = assert_legacy_roundtrip(&LegacySparse::<$count> {
                    config: legacy_config(), index: vec![], count: vec![],
                });
                let cumulative: $cumulative = assert_legacy_roundtrip(&LegacyCumulative::<$count> {
                    config: legacy_config(), index: vec![], count: vec![], mean: None,
                });
                assert_eq!(cumulative.mean(), None);
            }

            #[test]
            fn dense_rejects_wrong_bucket_lengths_and_invalid_config() {
                for len in [0, 5, 7] {
                    let wire = LegacyDense { config: legacy_config(), buckets: vec![0 as $count; len].into() };
                    assert_rejected::<$dense>(&serde_json::to_value(&wire).unwrap());
                    assert!(bincode::deserialize::<$dense>(&bincode::serialize(&wire).unwrap()).is_err());
                }
                let mut wire = serde_json::to_value(LegacyDense {
                    config: legacy_config(), buckets: vec![0 as $count; 6].into(),
                }).unwrap();
                wire["config"]["max"] = json!(u64::MAX);
                assert_rejected::<$dense>(&wire);
            }

            #[test]
            fn sparse_rejects_malformed_indices_counts_and_config() {
                for (index, count) in [
                    (vec![0, 1], vec![1 as $count]),
                    (vec![6], vec![1]),
                    (vec![u32::MAX], vec![1]),
                    (vec![2, 1], vec![1, 2]),
                    (vec![1, 1], vec![1, 2]),
                    (vec![1], vec![0]),
                ] {
                    let wire = LegacySparse { config: legacy_config(), index, count };
                    assert_rejected::<$sparse>(&serde_json::to_value(&wire).unwrap());
                    assert!(bincode::deserialize::<$sparse>(&bincode::serialize(&wire).unwrap()).is_err());
                }
                let wire = json!({"config": {"max": 7, "grouping_power": 63, "max_value_power": 64, "cutoff_power": 2, "cutoff_value": 4, "lower_bin_count": 4, "upper_bin_divisions": 2, "upper_bin_count": 2}, "index": [], "count": []});
                assert_rejected::<$sparse>(&wire);
            }

            #[test]
            fn cumulative_rejects_malformed_indices_and_prefix_counts() {
                for (index, count) in [
                    (vec![0, 1], vec![1 as $count]),
                    (vec![6], vec![1]),
                    (vec![u32::MAX], vec![1]),
                    (vec![2, 1], vec![1, 2]),
                    (vec![1, 1], vec![1, 2]),
                    (vec![1], vec![0]),
                    (vec![1, 2], vec![2, 1]),
                ] {
                    let wire = LegacyCumulative { config: legacy_config(), index, count, mean: Some(1.0) };
                    assert_rejected::<$cumulative>(&serde_json::to_value(&wire).unwrap());
                    assert!(bincode::deserialize::<$cumulative>(&bincode::serialize(&wire).unwrap()).is_err());
                }
            }

            #[test]
            fn equal_prefix_counts_remain_valid() {
                // The second bucket has zero individual observations. This is
                // legal under from_parts and must not be rejected by serde.
                let h: $cumulative = assert_legacy_roundtrip(&LegacyCumulative {
                    config: legacy_config(), index: vec![1, 2, 3], count: vec![2 as $count, 2, 5], mean: Some(2.2),
                });
                assert_eq!(h.count(), &[2, 2, 5]);
                assert_eq!(h.mean(), Some(2.2));
            }

            #[test]
            fn cumulative_recomputes_forged_missing_and_nonfinite_cached_means() {
                for mean in [Some(-10.0), Some(f64::NAN), Some(f64::INFINITY), None] {
                    let wire = LegacyCumulative {
                        config: legacy_config(), index: vec![0, 1, 2], count: vec![2 as $count, 5, 10], mean,
                    };
                    let bytes = bincode::serialize(&wire).unwrap();
                    let actual: $cumulative = bincode::deserialize(&bytes).unwrap();
                    assert_eq!(actual.mean(), Some(1.3));
                }
                let mut wire = json!({"config": legacy_config(), "index": [0, 1, 2], "count": [2, 5, 10], "mean": -10.0});
                let actual: $cumulative = serde_json::from_value(wire.clone()).unwrap();
                assert_eq!(actual.mean(), Some(1.3));
                wire.as_object_mut().unwrap().remove("mean");
                let actual: $cumulative = serde_json::from_value(wire).unwrap();
                assert_eq!(actual.mean(), Some(1.3));
                let wire = LegacyCumulative::<$count> { config: legacy_config(), index: vec![], count: vec![], mean: Some(42.0) };
                let actual: $cumulative = bincode::deserialize(&bincode::serialize(&wire).unwrap()).unwrap();
                assert_eq!(actual.mean(), None);
            }
        }
    };
}
histogram_tests!(
    u64_counts,
    Histogram,
    SparseHistogram,
    CumulativeROHistogram,
    u64
);
histogram_tests!(
    u32_counts,
    Histogram32,
    SparseHistogram32,
    CumulativeROHistogram32,
    u32
);

// Some formats use the struct name in addition to field order. Exercise the
// deserializer entry point directly so local wire helpers cannot rename it.
struct Named<D> {
    inner: D,
    expected: &'static str,
}
impl<'de, D: serde::Deserializer<'de>> serde::Deserializer<'de> for Named<D> {
    type Error = D::Error;
    fn deserialize_any<V: serde::de::Visitor<'de>>(
        self,
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        self.inner.deserialize_any(visitor)
    }
    fn deserialize_struct<V: serde::de::Visitor<'de>>(
        self,
        name: &'static str,
        fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        assert_eq!(name, self.expected);
        self.inner.deserialize_struct(name, fields, visitor)
    }
    serde::forward_to_deserialize_any! {
        bool i8 i16 i32 i64 u8 u16 u32 u64 f32 f64 char str string bytes
        byte_buf option unit unit_struct newtype_struct seq tuple tuple_struct
        map enum identifier ignored_any
    }
}
fn assert_struct_name<T: Serialize + DeserializeOwned>(value: T, expected: &'static str) {
    use serde::de::IntoDeserializer;
    let value = serde_json::to_value(value).unwrap();
    let _: T = T::deserialize(Named {
        inner: value.into_deserializer(),
        expected,
    })
    .unwrap();
}
#[test]
fn original_serde_struct_names_are_preserved() {
    assert_struct_name(Config::new(1, 3).unwrap(), "Config");
    let dense = Histogram::new(1, 3).unwrap();
    assert_struct_name(SparseHistogram::from(&dense), "SparseHistogram");
    assert_struct_name(CumulativeROHistogram::from(&dense), "CumulativeROHistogram");
    assert_struct_name(dense, "Histogram");
    let dense = Histogram32::new(1, 3).unwrap();
    assert_struct_name(SparseHistogram32::from(&dense), "SparseHistogram32");
    assert_struct_name(
        CumulativeROHistogram32::from(&dense),
        "CumulativeROHistogram32",
    );
    assert_struct_name(dense, "Histogram32");
}
