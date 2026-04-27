// Math helpers ported from the Observable notebook @iopsystems/h2-histogram.
// All bucket counts that may exceed 2^53 are computed in BigInt.

export type BucketGroup = {
  width: bigint;
  lower: bigint;
  upper: bigint;
  buckets: bigint;
};

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
