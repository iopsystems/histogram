//! Compare the memory requirement of `Histogram`, `SparseHistogram`, and
//! `CumulativeROHistogram` for histogram columns in typical Rezolus parquet
//! recordings.
//!
//! Usage:
//! ```text
//! cargo run --release --example rezolus_memory -- <file.parquet> [<file2.parquet> ...]
//! ```
//!
//! Each parquet file produced by Rezolus stores histograms as
//! `List<UInt64>` columns whose field-level metadata carries
//! `metric_type = "histogram"`, `grouping_power`, and `max_value_power`.
//! Each list element for a given row is the dense bucket vector for that
//! sample. This example:
//!
//! 1. Enumerates every histogram column.
//! 2. Reconstructs each sample as a `Histogram` via `Histogram::from_buckets`.
//! 3. Converts to `SparseHistogram` and `CumulativeROHistogram`.
//! 4. Computes the in-memory footprint (including heap-allocated bucket
//!    storage) of each representation and prints per-column and overall
//!    distribution statistics.

use std::env;
use std::fs::File;
use std::mem;
use std::path::{Path, PathBuf};

use arrow::array::{Array, ListArray, UInt64Array};
use arrow::datatypes::DataType;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use histogram::{CumulativeROHistogram, Histogram, SparseHistogram};

/// Memory footprint, in bytes, of a standard [`Histogram`] including its
/// heap-allocated dense bucket array.
fn histogram_bytes(h: &Histogram) -> usize {
    mem::size_of::<Histogram>() + mem::size_of_val(h.as_slice())
}

/// Memory footprint, in bytes, of a [`SparseHistogram`] including the heap
/// allocations backing its `index` and `count` vectors.
fn sparse_bytes(h: &SparseHistogram) -> usize {
    mem::size_of::<SparseHistogram>()
        + mem::size_of_val(h.index())
        + mem::size_of_val(h.count())
}

/// Memory footprint, in bytes, of a [`CumulativeROHistogram`] including the
/// heap allocations backing its `index` and `count` vectors.
fn cumulative_bytes(h: &CumulativeROHistogram) -> usize {
    mem::size_of::<CumulativeROHistogram>()
        + mem::size_of_val(h.index())
        + mem::size_of_val(h.count())
}

#[derive(Default, Clone)]
struct Stats {
    samples: usize,
    total_buckets: usize,
    nonzero_buckets_sum: u64,
    std_bytes_sum: u64,
    sparse_bytes_sum: u64,
    cumulative_bytes_sum: u64,
    std_bytes_min: usize,
    std_bytes_max: usize,
    sparse_bytes_min: usize,
    sparse_bytes_max: usize,
    cumulative_bytes_min: usize,
    cumulative_bytes_max: usize,
    nonzero_buckets_min: usize,
    nonzero_buckets_max: usize,
    grouping_power: u8,
    max_value_power: u8,
}

impl Stats {
    fn new(grouping_power: u8, max_value_power: u8, total_buckets: usize) -> Self {
        Self {
            total_buckets,
            grouping_power,
            max_value_power,
            std_bytes_min: usize::MAX,
            sparse_bytes_min: usize::MAX,
            cumulative_bytes_min: usize::MAX,
            nonzero_buckets_min: usize::MAX,
            ..Default::default()
        }
    }

    fn record(&mut self, nnz: usize, std_b: usize, sparse_b: usize, cumulative_b: usize) {
        self.samples += 1;
        self.nonzero_buckets_sum += nnz as u64;
        self.std_bytes_sum += std_b as u64;
        self.sparse_bytes_sum += sparse_b as u64;
        self.cumulative_bytes_sum += cumulative_b as u64;
        self.nonzero_buckets_min = self.nonzero_buckets_min.min(nnz);
        self.nonzero_buckets_max = self.nonzero_buckets_max.max(nnz);
        self.std_bytes_min = self.std_bytes_min.min(std_b);
        self.std_bytes_max = self.std_bytes_max.max(std_b);
        self.sparse_bytes_min = self.sparse_bytes_min.min(sparse_b);
        self.sparse_bytes_max = self.sparse_bytes_max.max(sparse_b);
        self.cumulative_bytes_min = self.cumulative_bytes_min.min(cumulative_b);
        self.cumulative_bytes_max = self.cumulative_bytes_max.max(cumulative_b);
    }

    fn merge(&mut self, other: &Stats) {
        if other.samples == 0 {
            return;
        }
        if self.samples == 0 {
            // Clone minimums (and the other fields) from `other` so that the
            // `Default`-initialized zero-valued minimums don't dominate.
            *self = other.clone();
            return;
        }
        self.total_buckets = self.total_buckets.max(other.total_buckets);
        self.samples += other.samples;
        self.nonzero_buckets_sum += other.nonzero_buckets_sum;
        self.std_bytes_sum += other.std_bytes_sum;
        self.sparse_bytes_sum += other.sparse_bytes_sum;
        self.cumulative_bytes_sum += other.cumulative_bytes_sum;
        self.nonzero_buckets_min = self.nonzero_buckets_min.min(other.nonzero_buckets_min);
        self.nonzero_buckets_max = self.nonzero_buckets_max.max(other.nonzero_buckets_max);
        self.std_bytes_min = self.std_bytes_min.min(other.std_bytes_min);
        self.std_bytes_max = self.std_bytes_max.max(other.std_bytes_max);
        self.sparse_bytes_min = self.sparse_bytes_min.min(other.sparse_bytes_min);
        self.sparse_bytes_max = self.sparse_bytes_max.max(other.sparse_bytes_max);
        self.cumulative_bytes_min = self.cumulative_bytes_min.min(other.cumulative_bytes_min);
        self.cumulative_bytes_max = self.cumulative_bytes_max.max(other.cumulative_bytes_max);
    }

    fn mean_std(&self) -> f64 {
        self.std_bytes_sum as f64 / self.samples.max(1) as f64
    }
    fn mean_sparse(&self) -> f64 {
        self.sparse_bytes_sum as f64 / self.samples.max(1) as f64
    }
    fn mean_cumulative(&self) -> f64 {
        self.cumulative_bytes_sum as f64 / self.samples.max(1) as f64
    }
    fn mean_nnz(&self) -> f64 {
        self.nonzero_buckets_sum as f64 / self.samples.max(1) as f64
    }
}

fn format_bytes(b: f64) -> String {
    if b < 1024.0 {
        format!("{b:>8.0}  B")
    } else if b < 1024.0 * 1024.0 {
        format!("{:>8.2} KiB", b / 1024.0)
    } else {
        format!("{:>8.2} MiB", b / (1024.0 * 1024.0))
    }
}

fn process_file(path: &Path, overall: &mut Stats) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== {} ===", path.display());

    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let schema = builder.schema().clone();

    // Find histogram columns and capture their (grouping_power, max_value_power)
    // parameters. Skip columns that are not List<UInt64> or that are missing
    // the histogram parameters.
    let mut columns: Vec<(usize, String, u8, u8)> = Vec::new();
    for (idx, field) in schema.fields().iter().enumerate() {
        let meta = field.metadata();
        if meta.get("metric_type").map(String::as_str) != Some("histogram") {
            continue;
        }
        let (Some(gp), Some(mvp)) = (meta.get("grouping_power"), meta.get("max_value_power"))
        else {
            continue;
        };
        let (Ok(gp), Ok(mvp)) = (gp.parse::<u8>(), mvp.parse::<u8>()) else {
            continue;
        };
        match field.data_type() {
            DataType::List(inner) if inner.data_type() == &DataType::UInt64 => {
                columns.push((idx, field.name().clone(), gp, mvp));
            }
            _ => continue,
        }
    }

    if columns.is_empty() {
        println!("(no histogram columns)");
        return Ok(());
    }

    let mut per_column: Vec<Stats> = columns
        .iter()
        .map(|(_, _, gp, mvp)| {
            let cfg = histogram::Config::new(*gp, *mvp).expect("valid config");
            Stats::new(*gp, *mvp, cfg.total_buckets())
        })
        .collect();

    let reader = builder.build()?;
    for batch in reader {
        let batch = batch?;
        for (slot, (col_idx, _name, gp, mvp)) in columns.iter().enumerate() {
            let col = batch.column(*col_idx);
            let list = col
                .as_any()
                .downcast_ref::<ListArray>()
                .expect("histogram column is List");
            for row in 0..list.len() {
                if list.is_null(row) {
                    continue;
                }
                let values = list.value(row);
                let buckets_arr = values
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .expect("histogram bucket values are UInt64");
                let buckets: Vec<u64> = buckets_arr.iter().flatten().collect();

                let Ok(h) = Histogram::from_buckets(*gp, *mvp, buckets) else {
                    continue;
                };
                let sparse = SparseHistogram::from(&h);
                let cumulative = CumulativeROHistogram::from(&h);
                let nnz = sparse.index().len();

                per_column[slot].record(
                    nnz,
                    histogram_bytes(&h),
                    sparse_bytes(&sparse),
                    cumulative_bytes(&cumulative),
                );
            }
        }
    }

    // Per-column summary
    println!(
        "{:<48} {:>9} {:>4} {:>4} {:>8} {:>10} {:>10} {:>10} {:>7} {:>7}",
        "column",
        "samples",
        "gp",
        "mvp",
        "buckets",
        "std (avg)",
        "sparse avg",
        "cumul avg",
        "sp/std",
        "cu/std",
    );
    println!("{}", "-".repeat(128));
    for (stats, (_, name, _, _)) in per_column.iter().zip(columns.iter()) {
        if stats.samples == 0 {
            continue;
        }
        let std_mean = stats.mean_std();
        let sp_mean = stats.mean_sparse();
        let cu_mean = stats.mean_cumulative();
        let sp_ratio = sp_mean / std_mean;
        let cu_ratio = cu_mean / std_mean;
        println!(
            "{:<48} {:>9} {:>4} {:>4} {:>8} {:>14} {:>14} {:>14} {:>6.2}x {:>6.2}x",
            truncate(name, 48),
            stats.samples,
            stats.grouping_power,
            stats.max_value_power,
            stats.total_buckets,
            format_bytes(std_mean),
            format_bytes(sp_mean),
            format_bytes(cu_mean),
            sp_ratio,
            cu_ratio,
        );
    }

    // File-level aggregate
    let mut file_total = Stats::default();
    for s in &per_column {
        file_total.merge(s);
    }

    if file_total.samples > 0 {
        println!();
        println!(
            "File totals: {} samples across {} histogram columns",
            file_total.samples,
            per_column.iter().filter(|s| s.samples > 0).count()
        );
        println!(
            "  avg non-zero buckets per sample: {:.1}  (range {}..{})",
            file_total.mean_nnz(),
            file_total.nonzero_buckets_min,
            file_total.nonzero_buckets_max,
        );
        println!(
            "  total memory (Histogram):         {}   (avg {}/sample, range {}..{})",
            format_bytes(file_total.std_bytes_sum as f64),
            format_bytes(file_total.mean_std()),
            format_bytes(file_total.std_bytes_min as f64),
            format_bytes(file_total.std_bytes_max as f64),
        );
        println!(
            "  total memory (SparseHistogram):   {}   (avg {}/sample, range {}..{})",
            format_bytes(file_total.sparse_bytes_sum as f64),
            format_bytes(file_total.mean_sparse()),
            format_bytes(file_total.sparse_bytes_min as f64),
            format_bytes(file_total.sparse_bytes_max as f64),
        );
        println!(
            "  total memory (CumulativeRO):      {}   (avg {}/sample, range {}..{})",
            format_bytes(file_total.cumulative_bytes_sum as f64),
            format_bytes(file_total.mean_cumulative()),
            format_bytes(file_total.cumulative_bytes_min as f64),
            format_bytes(file_total.cumulative_bytes_max as f64),
        );
        let savings_sparse =
            1.0 - file_total.sparse_bytes_sum as f64 / file_total.std_bytes_sum as f64;
        let savings_cumulative =
            1.0 - file_total.cumulative_bytes_sum as f64 / file_total.std_bytes_sum as f64;
        println!(
            "  sparse saves {:.1}% vs Histogram; cumulative saves {:.1}%",
            savings_sparse * 100.0,
            savings_cumulative * 100.0,
        );
    }

    overall.merge(&file_total);
    Ok(())
}

fn truncate(s: &str, n: usize) -> &str {
    if s.len() <= n { s } else { &s[..n] }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let files: Vec<PathBuf> = env::args_os().skip(1).map(PathBuf::from).collect();
    if files.is_empty() {
        eprintln!("usage: rezolus_memory <file.parquet> [<file2.parquet> ...]");
        std::process::exit(2);
    }

    let mut overall = Stats::default();
    for path in &files {
        process_file(path, &mut overall)?;
    }

    if files.len() > 1 && overall.samples > 0 {
        println!("\n=== overall across {} files ===", files.len());
        println!(
            "  {} histogram samples, avg non-zero buckets per sample: {:.1}",
            overall.samples,
            overall.mean_nnz(),
        );
        println!(
            "  Histogram total:       {}   (avg {}/sample)",
            format_bytes(overall.std_bytes_sum as f64),
            format_bytes(overall.mean_std()),
        );
        println!(
            "  SparseHistogram total: {}   (avg {}/sample)  -> {:.2}x of Histogram",
            format_bytes(overall.sparse_bytes_sum as f64),
            format_bytes(overall.mean_sparse()),
            overall.sparse_bytes_sum as f64 / overall.std_bytes_sum as f64,
        );
        println!(
            "  CumulativeRO total:    {}   (avg {}/sample)  -> {:.2}x of Histogram",
            format_bytes(overall.cumulative_bytes_sum as f64),
            format_bytes(overall.mean_cumulative()),
            overall.cumulative_bytes_sum as f64 / overall.std_bytes_sum as f64,
        );
    }

    Ok(())
}
