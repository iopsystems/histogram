use histogram::{Config, Error};

#[test]
fn every_parameter_pair_returns_a_valid_config_or_an_error_without_overflow() {
    for grouping_power in 0..=u8::MAX {
        for max_value_power in 0..=u8::MAX {
            let result = Config::new(grouping_power, max_value_power);
            if max_value_power > 64 {
                assert_eq!(result, Err(Error::MaxPowerTooHigh));
            } else if grouping_power >= max_value_power {
                assert_eq!(result, Err(Error::MaxPowerTooLow));
            } else {
                // The linear range has 2^(gp+1) buckets; every remaining
                // power contributes 2^gp. Calculate in u128 independently.
                let buckets =
                    (u128::from(max_value_power - grouping_power) + 1) * (1_u128 << grouping_power);
                if buckets > u128::from(u32::MAX) {
                    assert_eq!(result, Err(Error::Overflow));
                } else {
                    let config = result.unwrap();
                    assert_eq!(config.total_buckets() as u128, buckets);
                    assert_eq!(config.grouping_power(), grouping_power);
                    assert_eq!(config.max_value_power(), max_value_power);
                }
            }
        }
    }
}
