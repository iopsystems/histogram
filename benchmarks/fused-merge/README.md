# Fused checked-merge benchmark: Apple Silicon and Linux

This branch packages the experiment for issue #36. It compares fused sum/overflow reduction, scalar owned aggregation, clone-first in-place merging, zero-first in-place merging, and allocating pairwise `checked_add`.

The library is pinned to PR #35 (`2562b8e8cfd328a4a0ef4e7e6143841535f14b0e`), matching the original x86 comparison. The branch intentionally predates the later serde fix: this benchmark does not deserialize anything. No library implementation is modified. `library-pin.json` verifies the actual library source against that revision.

## Run on Apple Silicon

Prerequisites: native Apple Silicon Rust (Rust 1.85 or later), Python 3.9 or later, and the usual Apple command-line linker tools. No third-party Python packages are needed. The first run downloads six locked registry packages; subsequent runs can use `--offline`.

```sh
git fetch origin
git switch --track origin/bench/fused-merge-arm
rustc -vV
python3 benchmarks/fused-merge/run.py
```

`rustc -vV` should show `host: aarch64-apple-darwin`. Use a native terminal and Python rather than Rosetta. The runner rejects macOS x86/Rosetta execution, so the result cannot accidentally become another x86 measurement.

The default run takes a few minutes, including first-time compilation. It first runs all six correctness tests in debug and release, builds a release binary and emits assembly, then measures **480 cases × nine blocks**. Each case targets about 10 ms, with calibration and fixture checks outside its measurement. A quick installation check is available:

```sh
python3 benchmarks/fused-merge/run.py --quick
```

Quick mode runs all cases for one block at 1 ms per case. Use the default run for meaningful comparisons.

For the primary comparison, use the default compiler target. If you also want a host-tuned result, run it separately:

```sh
python3 benchmarks/fused-merge/run.py --native
```

This adds `-C target-cpu=native`; it does not reuse the default binary or mix its results. Existing `RUSTFLAGS` are recorded. For an ordinary default run, leave custom compiler flags, wrappers, Cargo target overrides and global Cargo target configuration unset. The runner rejects `CARGO_BUILD_TARGET` overrides because it must execute the binary natively.

Use AC power, disable Low Power Mode, let the laptop settle, and avoid other benchmarks or builds during the run. The runner records macOS CPU/model, memory, core counts, available power settings, compiler and flags. It does not pin threads or force performance-core selection; macOS scheduling, thermal state and core selection remain part of the measurement. Record any relevant differences when comparing machines.

## Results to return

The runner prints a representative table and two paths:

- `results/<timestamp>-<architecture>-<mode>/`: `SUMMARY.md`, full CSVs, host/compiler metadata, build/test log, exact source snapshot, measured binary and emitted assembly.
- The matching **`.tar.gz` archive**, with a `.sha256` file. This is the file to share for incorporation into the full histogram report.

Build intermediates are excluded from the archive. Existing result directories and archives are never overwritten. Use `--output /path/to/new-directory` to choose a location. A failed build/run leaves its log and a `status: failed` metadata record. Python validates the exact case matrix, all block identities and timing normalization before producing a complete result.

## What is measured

Both u32 and u64; 2, 8 or 64 inputs; 2, 8, 3072 or 21504 configured buckets; clustered or fully occupied counts; merge-only and merge plus one native five-quantile report. Every method starts with the same retained inputs and returns an owned result. Construction, allocation and destruction of that result are timed. Input preparation and the independent u128 bucket-count oracle are excluded. A report requests p0, p50, p90, p99 and p100 together.

Fused aggregation checks configuration before cloning the first input. For each subsequent input it stores wrapped sums and reduces overflow flags, returning an error after that source if any bucket overflowed. A failing private result is discarded; inputs are unchanged. The tests cover exact MAX results and overflow at every position across small arrays and vector boundaries. Only successful merges are timed; overflow latency is not measured.

All methods are shuffled within each fresh subprocess block. Compare medians from the same run. Min/max values show within-run variation, not cross-host confidence intervals. Tiny cases are especially sensitive to code layout and overhead. Fewer passes and SIMD instructions alone do not guarantee lower elapsed time.

## x86 reference and ARM question

The previous generic x86-64 build (Rust 1.97.1, i5-13500H) measured these medians for eight clustered gp10/max30 inputs:

| Counters | Clone-first in-place | Scalar owned | Fused owned |
|---|---:|---:|---:|
| u32 | 30.46 µs | 45.28 µs | 19.51 µs |
| u64 | 80.92 µs | 55.33 µs | 77.65 µs |

Both fused loops vectorized in that binary. Its generic x86 target needed additional compares and shuffles for unsigned u64 overflow checks. AArch64 has direct unsigned 64-bit vector comparisons, so the u64 ordering may change. The emitted `.s` files let us inspect what the compiler actually generated on your laptop. These are hypotheses to test; the summary generator does not assume either strategy wins.

The timed merge implementations are unchanged from that experiment. This portable harness has a different crate name and adds configurable duration outside the timed operation, so binary layout and tiny-case timings can differ even on the same CPU. All baselines are rerun together.

## Reproduce an archive or run checks alone

From an unpacked result, the sources are self-contained:

```sh
python3 <result-dir>/source-at-run/benchmark/run.py --output /tmp/fused-replay
```

For checks in this branch:

```sh
python3 -m unittest discover -s benchmarks/fused-merge -p test_runner.py
cargo test --manifest-path benchmarks/fused-merge/Cargo.toml --locked
cargo test --release --manifest-path benchmarks/fused-merge/Cargo.toml --locked
```

The results archive captures exact compiler input and emitted assembly, but no cross-architecture timing is simulated. This runner has been exercised on Linux; the native Apple Silicon run supplies the ARM evidence.
