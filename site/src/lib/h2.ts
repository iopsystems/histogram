// Math helpers ported from the Observable notebook @iopsystems/h2-histogram.
// All bucket counts that may exceed 2^53 are computed in BigInt.

export type SummaryRow = {
  p: number;
  relativeError: string;
  buckets: string;
  size8: string;
  size16: string;
  size32: string;
  size64: string;
};

export type BucketGroup = {
  width: bigint;
  lower: bigint;
  upper: bigint;
  buckets: bigint;
};

const KiB = 1024;
const MiB = 1024 * 1024;

function formatSize(bytes: number): string {
  return bytes > MiB
    ? (bytes / MiB).toFixed(1) + ' MiB'
    : (bytes / KiB).toFixed(1) + ' KiB';
}

export function summarize(nVal = 64, pVals: number[] = defaultPVals()): SummaryRow[] {
  return pVals.map((k) => {
    const nbucket = (1n << BigInt(k)) * BigInt(nVal - k + 1);
    const base = Number(nbucket);
    return {
      p: k,
      relativeError: (2 ** -k * 100).toPrecision(3) + '%',
      buckets: nbucket.toString(),
      size8: formatSize(base),
      size16: formatSize(base * 2),
      size32: formatSize(base * 4),
      size64: formatSize(base * 8),
    };
  });
}

export function defaultPVals(): number[] {
  return [...Array(15).keys()].slice(2);
}

export function totalBuckets(n: number, p: number): bigint {
  return (1n << BigInt(p)) * BigInt(n - p + 1);
}

export function truncatedBuckets(l: number, p: number): bigint {
  return l > p + 1
    ? BigInt(l - p + 1) * (1n << BigInt(p))
    : 1n << BigInt(l);
}

export function reductionPct(truncated: bigint, total: bigint): number {
  return (Number(truncated) / Number(total)) * 100;
}

export function maxValue(n: number): bigint {
  return (1n << BigInt(n)) - 1n;
}

export function relativeError(p: number): number {
  return 1 / 2 ** p;
}

export function computeBuckets(n: number, p: number): BucketGroup[] {
  const groups: BucketGroup[] = [];
  let lower = 0n;
  for (let w = 0; w + p < n; w++) {
    const width = 1n << BigInt(w);
    const upper = 1n << BigInt(p + w + 1);
    const groupSize = (upper - lower) / width;
    groups.push({ width, lower, upper, buckets: groupSize });
    lower = upper;
  }
  return groups;
}

export function histogramSizeKiB(total: bigint, counterBits: 32 | 64): bigint {
  const bytesPerCounter = BigInt(counterBits / 8);
  return (total * bytesPerCounter) / 1024n;
}
