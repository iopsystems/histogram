//! Exercise both sum backends even when public dispatch always selects AVX2.
use super::{Histogram, Histogram32};
use crate::Error;

macro_rules! backend_contract {
    ($module:ident, $dense:ty, $count:ty) => {
        mod $module {
            use super::*;
            type Dense = $dense;
            type Count = $count;

            fn check_backend(sum: impl Fn(&[&Dense]) -> Result<Dense, Error>) {
                assert_eq!(sum(&[]), Err(Error::IncompatibleParameters));
                let mut a = Dense::new(0, 7).unwrap();
                a.as_mut_slice().fill(Count::MAX);
                let mut singleton = sum(&[&a]).unwrap();
                assert_ne!(singleton.as_slice().as_ptr(), a.as_slice().as_ptr());
                singleton.reset();
                assert!(a.as_slice().iter().all(|&v| v == Count::MAX));
                let mismatch = Dense::new(1, 4).unwrap();
                assert_eq!(a.as_slice().len(), mismatch.as_slice().len());
                assert_eq!(
                    sum(&[&a, &a, &mismatch]),
                    Err(Error::IncompatibleParameters)
                );

                for (gp, max) in (1..=64).map(|max| (0, max)).chain([(3, 30), (7, 30)]) {
                    let mut first = Dense::new(gp, max).unwrap();
                    let mut second = first.clone();
                    assert_eq!(sum(&[&first, &second]).unwrap(), first);
                    first.as_mut_slice().fill(Count::MAX - 2);
                    second.as_mut_slice().fill(1);
                    let result = sum(&[&first, &second, &second]).unwrap();
                    // Every bucket reaches MAX; the total deliberately exceeds it.
                    assert!(result.as_slice().iter().all(|&v| v == Count::MAX));
                    let before_first = first.clone();
                    let before_second = second.clone();
                    for position in 0..first.as_slice().len() {
                        let mut late = Dense::new(gp, max).unwrap();
                        late.as_mut_slice()[position] = 2;
                        let before_late = late.clone();
                        assert_eq!(sum(&[&first, &second, &late]), Err(Error::Overflow));
                        assert_eq!(first, before_first);
                        assert_eq!(second, before_second);
                        assert_eq!(late, before_late);
                    }
                    // Vector unsigned comparisons must handle the sign boundary.
                    let high = 1 << (Count::BITS - 1);
                    first.as_mut_slice().fill(high);
                    second.as_mut_slice().fill(high - 1);
                    let result = sum(&[&first, &second]).unwrap();
                    assert!(result.as_slice().iter().all(|&v| v == Count::MAX));
                    second.as_mut_slice().fill(high);
                    assert_eq!(sum(&[&first, &second]), Err(Error::Overflow));
                }
            }

            #[test]
            fn portable_backend_contract() {
                check_backend(Dense::checked_sum_portable);
            }

            #[test]
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            fn avx2_backend_contract() {
                if !std::is_x86_feature_detected!("avx2") {
                    return;
                }
                // SAFETY: the CPU/OS feature check above guards every invocation.
                check_backend(|inputs| unsafe { Dense::checked_sum_avx2(inputs) });
            }
        }
    };
}

backend_contract!(wide, Histogram, u64);
backend_contract!(narrow, Histogram32, u32);
