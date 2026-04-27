<script lang="ts">
  import katex from 'katex';
  import {
    computeBuckets,
    histogramSizeKiB,
    maxValue,
    reductionPct,
    relativeError,
    totalBuckets,
    truncatedBuckets,
  } from '../lib/h2';

  let n = $state(64);
  let p = $state(10);
  let l = $state(10);

  $effect(() => {
    if (l > n - 1) l = n - 1;
  });

  const N = $derived(maxValue(n));
  const e = $derived(relativeError(p));
  const total = $derived(totalBuckets(n, p));
  const truncated = $derived(truncatedBuckets(l, p));
  const reduction = $derived(reductionPct(truncated, total));
  const groups = $derived(computeBuckets(n, p));
  const size64 = $derived(histogramSizeKiB(total, 64));
  const size32 = $derived(histogramSizeKiB(total, 32));

  function tex(src: string): string {
    return katex.renderToString(src, { throwOnError: false });
  }
</script>

<section class="calc">
  <div class="row">
    <label>
      <span class="sym">{@html tex('n')}</span>
      <input type="range" min="8" max="64" step="1" bind:value={n} />
      <span class="val">{n}</span>
    </label>
    <p class="readout">
      For {@html tex(`n=${n}`)}, {@html tex('N')} = {N.toString()}
    </p>
  </div>

  <div class="row">
    <label>
      <span class="sym">{@html tex('p')}</span>
      <input type="range" min="2" max="22" step="1" bind:value={p} />
      <span class="val">{p}</span>
    </label>
    <p class="readout">
      For {@html tex(`p=${p}`)}, {@html tex(`e = ${e.toPrecision(3)}`)}
    </p>
  </div>

  <h3>Bucket summary</h3>
  <ul class="readouts">
    <li>
      Total number of buckets:
      {@html tex('(n - p + 1) \\times 2^p')}
      = <strong>{total.toString()}</strong>
    </li>
    <li>
      Histogram size:
      <strong>{size64.toString()} KiB</strong> (64-bit counters)
      or <strong>{size32.toString()} KiB</strong> (32-bit counters)
    </li>
  </ul>

  <h3>Bucket layout</h3>
  <table>
    <thead>
      <tr>
        <th>width</th>
        <th>lower</th>
        <th>upper</th>
        <th>buckets</th>
      </tr>
    </thead>
    <tbody>
      {#each groups as g}
        <tr>
          <td>{g.width.toString()}</td>
          <td>{g.lower.toString()}</td>
          <td>{g.upper.toString()}</td>
          <td>{g.buckets.toString()}</td>
        </tr>
      {/each}
    </tbody>
  </table>

  <h3>Truncated histogram</h3>
  <div class="row">
    <label>
      <span class="sym">{@html tex('l')}</span>
      <input type="range" min="0" max={n - 1} step="1" bind:value={l} />
      <span class="val">{l}</span>
    </label>
  </div>

  <ul class="readouts">
    <li>
      Number of buckets (original histogram): <strong>{total.toString()}</strong>
    </li>
    <li>
      Number of buckets (truncated histogram):
      <strong>{(total - truncated).toString()}</strong>
    </li>
    <li>
      Buckets saved: <strong>{truncated.toString()}</strong>
      ({reduction.toFixed(1)}%)
    </li>
  </ul>
</section>

<style>
  .calc {
    border: 1px solid var(--rule, #e3e3e3);
    border-radius: 6px;
    padding: 1.25rem 1.5rem;
    margin: 1.5em 0;
    background: var(--code-bg, #f5f5f5);
  }

  .row {
    margin: 0.75em 0 1em;
  }

  label {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    flex-wrap: wrap;
  }

  .sym {
    min-width: 1.5em;
    display: inline-block;
  }

  input[type='range'] {
    flex: 1 1 200px;
    max-width: 360px;
  }

  .val {
    font-variant-numeric: tabular-nums;
    font-weight: 600;
    min-width: 2.5em;
    text-align: right;
  }

  .readout {
    margin: 0.4em 0 0 2.25em;
    color: var(--muted, #555);
    font-size: 0.95rem;
  }

  .readouts {
    margin: 0.5em 0 1em;
    padding-left: 1.4em;
  }

  .readouts li {
    margin: 0.25em 0;
  }

  h3 {
    margin-top: 1.6em;
    margin-bottom: 0.4em;
    font-size: 1rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--muted, #555);
  }

  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
    margin: 0.5em 0 1em;
  }

  th, td {
    border: 1px solid var(--rule, #e3e3e3);
    padding: 0.3em 0.6em;
    text-align: right;
    font-variant-numeric: tabular-nums;
  }

  th {
    background: var(--table-header, #f0f0f0);
    text-align: center;
  }
</style>
