#!/usr/bin/env python3
"""Build, test and measure fused aggregation on native macOS or Linux."""
import argparse
import csv
import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parent / "library" if (HERE.parent / "library" / "Cargo.toml").exists() else HERE.parents[1]
PIN = "2562b8e8cfd328a4a0ef4e7e6143841535f14b0e"
KEYS = ["width", "gp", "max_power", "windows", "shape", "report", "method"]
METHODS = ["fused", "owned", "pairwise_clone", "pairwise_zero", "owned_pairwise"]
EXPECTED = {(str(w), str(g), str(m), str(n), shape, report, method)
            for w in [32, 64] for g, m in [(0, 1), (0, 7), (7, 30), (10, 30)]
            for n in [2, 8, 64] for shape in ["clustered", "full"]
            for report in ["false", "true"] for method in METHODS}


def capture(command):
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT,
                                       timeout=15).strip()
    except (OSError, subprocess.SubprocessError) as error:
        return "unavailable: " + str(error)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_rows(rows, expected=EXPECTED):
    keys = [tuple(row[k] for k in KEYS) for row in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Missing, duplicate or unexpected benchmark cases")
    for row in rows:
        iterations = int(row["iterations"])
        elapsed = int(row["elapsed_ns"])
        ns = float(row["ns_per_set"])
        if (iterations <= 0 or elapsed <= 0 or not math.isfinite(ns) or ns <= 0
                or abs(ns - elapsed / iterations) > 0.000501):
            raise ValueError("Invalid timing or inconsistent normalization")


def summarize(rows, blocks):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in KEYS), []).append(row)
    result = []
    for key, samples in sorted(groups.items()):
        if sorted(int(r["block"]) for r in samples) != list(range(blocks)):
            raise ValueError("Missing or duplicate timing blocks")
        values = [float(r["ns_per_set"]) for r in samples]
        result.append(dict(zip(KEYS, key), median_ns=statistics.median(values),
                           min_ns=min(values), max_ns=max(values)))
    return result


def write_csv(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(summary, metadata):
    lookup = {tuple(r[k] for k in KEYS): r["median_ns"] / 1000 for r in summary}
    lines = ["# Fused merge results", "",
             f"Host: {metadata['machine']} / {metadata['platform']}",
             f"Blocks: {metadata['blocks']}; target: {metadata['target_ms']} ms per case.",
             f"Library matches comparison pin: {metadata['library_matches_pin']}.", "",
             "Eight clustered inputs, gp10/max30; microseconds per complete batch.", "",
             "| Counters | Operation | Fused | Scalar owned | Clone-first | Zero-first | Allocating pairwise |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for width in ["32", "64"]:
        for reporting in ["false", "true"]:
            times = [lookup[(width, "10", "30", "8", "clustered", reporting, m)]
                     for m in METHODS]
            operation = "merge + five quantiles" if reporting == "true" else "merge"
            lines.append("| " + " | ".join(["u" + width, operation,
                                           *[f"{v:.2f}" for v in times]]) + " |")
    lines += ["", "All 480 cases are in summary.csv; raw block samples are in timing.csv.",
              "Output construction and destruction are timed; input preparation is excluded.",
              "All methods return the same exact counts; only successful merges are timed.",
              "Fresh subprocess blocks randomize case order. Medians/min/max describe this host/run, not cross-host confidence.",
              "No explicit CPU affinity is applied; scheduling, core selection and thermal state may affect results.",
              "Assembly comes from the same cargo rustc invocation that produced the measured binary.",
              "Quick mode is a smoke test, not evidence for selecting an implementation.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New result directory; defaults to a timestamped results/ directory")
    parser.add_argument("--blocks", type=int, default=9)
    parser.add_argument("--target-ms", type=float, default=10)
    parser.add_argument("--quick", action="store_true", help="One block, 1 ms per case; smoke test only")
    parser.add_argument("--native", action="store_true", help="Additionally compile with -C target-cpu=native")
    parser.add_argument("--offline", action="store_true", help="Use only cached Cargo dependencies")
    args = parser.parse_args()
    if args.quick:
        args.blocks, args.target_ms = 1, 1
    if args.blocks < 1 or not math.isfinite(args.target_ms) or args.target_ms <= 0:
        parser.error("blocks and target-ms must be positive and finite")
    if os.environ.get("CARGO_BUILD_TARGET"):
        parser.error("Unset CARGO_BUILD_TARGET: this runner builds and executes on the native host")
    env = os.environ.copy()
    if args.native:
        if env.get("CARGO_ENCODED_RUSTFLAGS"):
            parser.error("Unset CARGO_ENCODED_RUSTFLAGS before using --native")
        env["RUSTFLAGS"] = (env.get("RUSTFLAGS", "") + " -C target-cpu=native").strip()
    compiler = subprocess.check_output(["rustc", "-Vv"], text=True)
    machine = platform.machine()
    host = {}
    if platform.system() == "Darwin":
        for key in ["machdep.cpu.brand_string", "hw.model", "hw.memsize", "hw.ncpu",
                    "hw.perflevel0.physicalcpu", "hw.perflevel1.physicalcpu",
                    "hw.optional.arm64", "sysctl.proc_translated"]:
            host[key] = capture(["sysctl", "-n", key])
        host["power_settings"] = capture(["pmset", "-g", "custom"])
        if (
                machine not in ["arm64", "aarch64"] or host["sysctl.proc_translated"] == "1"
                or "host: aarch64-apple-darwin" not in compiler):
            parser.error("Use a native Apple Silicon terminal, Python and aarch64-apple-darwin Rust; Rosetta would measure x86")
    else:
        host["lscpu"] = capture(["lscpu"])
    if hasattr(os, "sched_getaffinity"):
        host["allowed_cpus"] = sorted(os.sched_getaffinity(0))
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    mode = "native" if args.native else "default"
    out = (args.output or HERE / "results" / f"{stamp}-{machine}-{mode}").resolve()
    archive = Path(str(out) + ".tar.gz")
    if archive.exists():
        parser.error("Archive already exists; use a new output path")
    out.mkdir(parents=True, exist_ok=False)
    env["CARGO_TARGET_DIR"] = str(out / "target")
    metadata = dict(status="preparing", started_utc=stamp, machine=machine,
                    platform=platform.platform(), host=host, compiler=compiler,
                    cargo=capture(["cargo", "-V"]), comparison_library_pin=PIN,
                    branch_head=capture(["git", "-C", str(REPO), "rev-parse", "HEAD"]),
                    tracked_changes=capture(["git", "-C", str(REPO), "status", "--short", "--untracked-files=no"]),
                    blocks=args.blocks, target_ms=args.target_ms, quick=args.quick,
                    environment={k: env.get(k) for k in ["RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS",
                                 "CARGO_BUILD_TARGET", "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER"]})
    def save():
        (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    save()
    try:
        source = out / "source-at-run"
        harness, library = source / "benchmark", source / "library"
        harness.mkdir(parents=True)
        library.mkdir()
        for name in ["src", "tests", "benches", "examples"]:
            if (REPO / name).exists():
                shutil.copytree(REPO / name, library / name)
        for name in ["Cargo.toml", "README.md", "LICENSE-APACHE", "LICENSE-MIT", "build.rs"]:
            if (REPO / name).exists():
                shutil.copy2(REPO / name, library / name)
        for name in ["src", "Cargo.toml", "Cargo.lock", "run.py", "test_runner.py", "README.md", "library-pin.json"]:
            path = HERE / name
            if path.is_dir():
                shutil.copytree(path, harness / name)
            else:
                shutil.copy2(path, harness / name)
        manifest = harness / "Cargo.toml"
        original = manifest.read_text()
        if 'path = "../.."' not in original and 'path = "../library"' not in original:
            raise ValueError("Unexpected dependency path in benchmark manifest")
        (harness / "Cargo.toml.original").write_text(original)
        manifest.write_text(original.replace('path = "../.."', 'path = "../library"'))
        pinned = json.loads((HERE / "library-pin.json").read_text())
        actual = {str(p.relative_to(library)): sha(p)
                  for p in [library / "Cargo.toml", *sorted((library / "src").rglob("*"))]
                  if p.is_file()}
        metadata["library_matches_pin"] = actual == pinned["sha256"] and pinned["commit"] == PIN
        metadata["source_sha256"] = {str(p.relative_to(source)): sha(p)
                                     for p in source.rglob("*") if p.is_file()}
        save()
        with (out / "build.log").open("w") as log:
            commands = [["cargo", "fetch", "--locked", *( ["--offline"] if args.offline else [])],
                        ["cargo", "test", "--locked", "--offline"],
                        ["cargo", "test", "--release", "--locked", "--offline"],
                        ["cargo", "rustc", "--release", "--locked", "--offline", "--", "--emit=asm,link"]]
            for command in commands:
                print(" ".join(command), flush=True)
                log.write("$ " + " ".join(command) + "\n")
                log.flush()
                subprocess.run(command, cwd=harness, env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
        (out / "bin").mkdir()
        binary = out / "bin/probe"
        shutil.copy2(out / "target/release/histogram-fused-merge-study", binary)
        (out / "assembly").mkdir()
        assembly = list((out / "target/release/deps").glob("histogram_fused_merge_study-*.s"))
        if not assembly:
            raise RuntimeError("Compiler did not emit the requested assembly")
        for path in assembly:
            shutil.copy2(path, out / "assembly" / path.name)
        metadata.update(status="running", binary_sha256=sha(binary))
        save()
        rows = []
        for block in range(args.blocks):
            raw = subprocess.check_output([str(binary), str(5231 + block * 917),
                                           str(args.target_ms)], text=True)
            (out / f"block-{block}.csv").write_text(raw)
            cells = list(csv.DictReader(io.StringIO(raw)))
            validate_rows(cells)
            rows.extend(dict(row, block=block) for row in cells)
            print(f"Block {block + 1}/{args.blocks} complete", flush=True)
        summary = summarize(rows, args.blocks)
        write_csv(out / "timing.csv", rows)
        write_csv(out / "summary.csv", summary)
        text = report(summary, metadata)
        (out / "SUMMARY.md").write_text(text)
        metadata.update(status="complete", files_sha256={
            str(p.relative_to(out)): sha(p) for p in out.rglob("*")
            if p.is_file() and p.relative_to(out).parts[0] != "target" and p.name != "metadata.json"})
        save()
        with tarfile.open(archive, "w:gz") as bundle:
            for path in sorted(out.rglob("*")):
                if path.is_file() and path.relative_to(out).parts[0] != "target":
                    bundle.add(path, arcname=str(Path(out.name) / path.relative_to(out)), recursive=False)
        archive.with_name(archive.name + ".sha256").write_text(sha(archive) + "  " + archive.name + "\n")
        print("\n" + text)
        print(f"Results: {out}\nShareable archive: {archive}")
    except Exception as error:
        metadata.update(status="failed", error=str(error))
        save()
        print(f"Failed; details in {out / 'build.log'} and metadata.json", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
