<script>
  // A series of violin plots: one violin per step (rollout_id), so the shape of
  // the advantage distribution at each step is legible side-by-side. Used for
  // the full run (every step, read left→right as "distribution over time").
  //
  // Each violin is built as a histogram of horizontal ("rotated") bars: the
  // value axis runs vertically and every bucket is a bar whose length encodes
  // how much mass falls in that value range, mirrored around the centre so the
  // stack of bars reads as a violin. Bars are individually hoverable (SVG
  // <title>) so you can inspect a bucket's range and its share of samples.
  //
  // The list endpoint only carries per-step quantiles (min/p10/p25/p50/p75/p90/
  // max), so we reconstruct a piecewise-uniform density from them and integrate
  // it over each bucket to get that bucket's mass. Bar lengths are normalised
  // across all buckets of all violins so widths are comparable between steps.

  import { brushZoom, fractionsToDomain } from "../lib/brushZoom.js";
  import ChartZoomButtons from "./ChartZoomButtons.svelte";
  import TimeAxis from "./TimeAxis.svelte";

  let {
    steps = [],
    labels = null,
    // `[min, max]` rollout ids; only steps inside are drawn.
    xDomain = null,
    // Called with `[min, max]` rollout ids when the user drags a window or
    // presses the zoom buttons.
    onChangeDomainX = null,
    // rollout id <-> epoch seconds; when both are given a wall-clock axis is
    // drawn under the plot.
    xToTime = null,
    timeToX = null,
  } = $props();

  const W = 640;
  const H = 210;
  const PAD = 8;
  const N_BUCKETS = 26;

  function num(v, fallback = 0) {
    const n = Number(v);
    return Number.isFinite(n) ? n : fallback;
  }

  function fmt(v) {
    return Number.isFinite(v) ? v.toFixed(3) : "—";
  }

  // ~`count` evenly-rounded tick values spanning [lo, hi] for the value axis.
  function niceTicks(lo, hi, count = 5) {
    if (!(hi > lo)) return [lo];
    const rawStep = (hi - lo) / (count - 1);
    const mag = Math.pow(10, Math.floor(Math.log10(rawStep)));
    const norm = rawStep / mag;
    const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
    const ticks = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-6; v += step) {
      ticks.push(v);
    }
    return ticks;
  }

  // A smooth closed outline through `pts` ([x, y] loop) via a Catmull-Rom spline
  // converted to cubic béziers. This turns the discrete bucket half-widths into a
  // continuous, rounded violin silhouette instead of a stack of hard-edged bars.
  function smoothClosedPath(pts) {
    const n = pts.length;
    if (n < 3) return "";
    const c = (v) => v.toFixed(2);
    let d = `M ${c(pts[0][0])} ${c(pts[0][1])} `;
    for (let i = 0; i < n; i++) {
      const p0 = pts[(i - 1 + n) % n];
      const p1 = pts[i];
      const p2 = pts[(i + 1) % n];
      const p3 = pts[(i + 2) % n];
      const c1x = p1[0] + (p2[0] - p0[0]) / 6;
      const c1y = p1[1] + (p2[1] - p0[1]) / 6;
      const c2x = p2[0] - (p3[0] - p1[0]) / 6;
      const c2y = p2[1] - (p3[1] - p1[1]) / 6;
      d += `C ${c(c1x)} ${c(c1y)}, ${c(c2x)} ${c(c2y)}, ${c(p2[0])} ${c(p2[1])} `;
    }
    return d + "Z";
  }

  // Inter-quantile segments and the fraction of samples each holds.
  const SEGMENTS = [
    ["min", "p10", 0.1],
    ["p10", "p25", 0.15],
    ["p25", "p50", 0.25],
    ["p50", "p75", 0.25],
    ["p75", "p90", 0.15],
    ["p90", "max", 0.1],
  ];

  let hasDomain = $derived(
    Array.isArray(xDomain) &&
      Number.isFinite(xDomain[0]) &&
      Number.isFinite(xDomain[1]) &&
      xDomain[1] > xDomain[0],
  );
  let zoomable = $derived(typeof onChangeDomainX === "function");

  let allPts = $derived(
    (steps || [])
      .filter((s) => s && s.stats)
      .map((s) => {
        const st = s.stats;
        const q = st.quantiles || {};
        return {
          x: num(s.rollout_id),
          min: num(st.min),
          max: num(st.max),
          mean: num(st.mean),
          p10: num(q.p10, num(st.min)),
          p25: num(q.p25, num(st.min)),
          p50: num(q.p50, num(st.mean)),
          p75: num(q.p75, num(st.max)),
          p90: num(q.p90, num(st.max)),
        };
      })
      .sort((a, b) => a.x - b.x),
  );

  let pts = $derived(
    hasDomain ? allPts.filter((p) => p.x >= xDomain[0] && p.x <= xDomain[1]) : allPts,
  );

  // Inverse of the brush mapping: rollout id -> horizontal fraction of the
  // plot, with the same linear extrapolation past the outermost columns.
  function columnFraction(x) {
    const xs = pts.map((p) => p.x);
    const n = xs.length;
    if (n === 0) {
      const [lo, hi] = hasDomain ? xDomain : [0, 1];
      return (x - lo) / (hi - lo || 1);
    }
    const spacing = n > 1 ? (xs[n - 1] - xs[0]) / (n - 1) : 1;
    let idx;
    if (x <= xs[0]) idx = (x - xs[0]) / spacing;
    else if (x >= xs[n - 1]) idx = n - 1 + (x - xs[n - 1]) / spacing;
    else {
      let i = 0;
      while (i < n - 2 && xs[i + 1] < x) i++;
      idx = i + (x - xs[i]) / (xs[i + 1] - xs[i] || 1);
    }
    return (idx + 0.5) / n;
  }

  let timeAxis = $derived.by(() => {
    if (typeof xToTime !== "function" || typeof timeToX !== "function") return null;
    if (hasDomain) return { start: xToTime(xDomain[0]), end: xToTime(xDomain[1]) };
    const n = pts.length;
    if (!n) return null;
    const spacing = n > 1 ? (pts[n - 1].x - pts[0].x) / (n - 1) : 1;
    return {
      start: xToTime(pts[0].x - spacing / 2),
      end: xToTime(pts[n - 1].x + spacing / 2),
    };
  });

  // Violins sit in equal-width columns, so a brush selection is a span of
  // column indices; map it back to rollout ids, extrapolating past the ends
  // with the mean step spacing so zooming out past the data keeps working.
  function handleBrush([f0, f1]) {
    if (!zoomable) return;
    // Expand the current domain even when it contains just one sparse rollout.
    if (hasDomain && f1 - f0 > 1) {
      onChangeDomainX(fractionsToDomain([f0, f1], xDomain));
      return;
    }
    const xs = pts.map((p) => p.x);
    const n = xs.length;
    let toX;
    if (n === 0) {
      const [lo, hi] = hasDomain ? xDomain : [0, 1];
      toX = (f) => lo + f * (hi - lo);
    } else {
      const spacing = n > 1 ? (xs[n - 1] - xs[0]) / (n - 1) : 1;
      toX = (f) => {
        const idx = f * n - 0.5;
        if (idx <= 0) return xs[0] + idx * spacing;
        if (idx >= n - 1) return xs[n - 1] + (idx - (n - 1)) * spacing;
        const i = Math.floor(idx);
        return xs[i] + (idx - i) * (xs[i + 1] - xs[i]);
      };
    }
    onChangeDomainX([toX(f0), toX(f1)]);
  }

  let model = $derived.by(() => {
    if (!pts.length) return null;

    const yLo = Math.min(...pts.map((p) => p.min));
    const yHi = Math.max(...pts.map((p) => p.max));
    const ySpan = yHi - yLo || 1;
    const bw = ySpan / N_BUCKETS;
    const tiny = ySpan * 1e-6;
    const sy = (v) => H - PAD - ((v - yLo) / ySpan) * (H - 2 * PAD);

    // Reconstructed piecewise density → mass falling in bucket [a, b).
    const bucketMass = (segs, a, b) => {
      let m = 0;
      for (const s of segs) {
        const len = s.hi - s.lo;
        if (len <= tiny) {
          const mid = (s.lo + s.hi) / 2;
          if (mid >= a && mid < b) m += s.mass;
        } else {
          const ov = Math.min(s.hi, b) - Math.max(s.lo, a);
          if (ov > 0) m += s.mass * (ov / len);
        }
      }
      return m;
    };

    let maxMass = 0;
    const perStep = pts.map((p) => {
      const segs = SEGMENTS.map(([lo, hi, mass]) => ({
        lo: p[lo],
        hi: p[hi],
        mass,
      }));
      const buckets = Array.from({ length: N_BUCKETS }, (_, j) => {
        const lo = yLo + j * bw;
        const hi = j === N_BUCKETS - 1 ? yHi + tiny : yLo + (j + 1) * bw;
        const mass = bucketMass(segs, lo, hi);
        if (mass > maxMass) maxMass = mass;
        return { lo, hi: yLo + (j + 1) * bw, mass };
      });
      return { x: p.x, p50: p.p50, buckets };
    });

    const colW = W / pts.length;
    const halfMax = (colW / 2) * 0.86;
    const hw = (mass) => (mass / (maxMass || 1)) * halfMax;

    const violins = perStep.map((s, i) => {
      const cx = i * colW + colW / 2;
      // Half-width at each bucket centre; the outline tapers to zero width at the
      // value extremes so the smoothed silhouette closes into a rounded teardrop.
      const centres = s.buckets.map((b) => ({
        y: sy((b.lo + b.hi) / 2),
        hw: hw(b.mass),
      }));
      const outline = [
        [cx, sy(yHi)], // top centre
        ...[...centres].reverse().map((cP) => [cx + cP.hw, cP.y]), // right edge, high→low
        [cx, sy(yLo)], // bottom centre
        ...centres.map((cP) => [cx - cP.hw, cP.y]), // left edge, low→high
      ];
      // Median tick spans the bucket that contains p50.
      const medBucket = s.buckets.find((b, bi) => s.p50 >= b.lo && (s.p50 < b.hi || bi === s.buckets.length - 1));
      const medHW = hw(medBucket ? medBucket.mass : 0);
      return {
        path: smoothClosedPath(outline),
        cx: cx.toFixed(2),
        medY: sy(s.p50).toFixed(2),
        medX1: (cx - Math.max(medHW, 3)).toFixed(2),
        medX2: (cx + Math.max(medHW, 3)).toFixed(2),
        x: s.x,
        p50: s.p50,
      };
    });

    return {
      violins,
      yLo,
      yHi,
      ticks: niceTicks(yLo, yHi, 5).map((val) => ({ val, y: sy(val).toFixed(2) })),
      zeroY: yLo <= 0 && yHi >= 0 ? sy(0) : null,
      firstX: pts[0].x,
      lastX: pts[pts.length - 1].x,
      latestMedian: perStep[perStep.length - 1].p50,
      // Per-violin labels get crowded past ~12 steps; fall back to endpoints.
      showEachLabel: pts.length <= 12,
    };
  });

  function labelFor(v, i) {
    if (labels && labels[i] != null) return labels[i];
    return `step ${v.x}`;
  }

  // Cursor readout: map the pointer's vertical position back to an advantage
  // value so hovering anywhere in the plot shows the score at that height.
  let hover = $state(null);

  function onPlotMove(e) {
    if (!model) return;
    const rect = e.currentTarget.getBoundingClientRect();
    if (!rect.height || !rect.width) return;
    const top = Math.max(0, Math.min(rect.height, e.clientY - rect.top));
    const left = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
    const pv = (top / rect.height) * H; // px → viewBox units
    const frac = (H - PAD - pv) / (H - 2 * PAD);
    const value = model.yLo + frac * (model.yHi - model.yLo);
    hover = {
      top,
      left,
      value: Math.max(model.yLo, Math.min(model.yHi, value)),
    };
  }

  function onPlotLeave() {
    hover = null;
  }
</script>

{#if model}
  <div class="flex flex-wrap items-center gap-[12px] mb-[8px] text-[11px] text-(--muted)">
    <span class="chart-legend-item"><span class="vsw fill"></span>sample density</span>
    <span class="chart-legend-item"><span class="vsw median"></span>median</span>
    {#if zoomable}
      <span class="ml-auto">
        <ChartZoomButtons onChangeDomainX={handleBrush} canZoomIn={pts.length > 1} />
      </span>
    {/if}
  </div>
  <div class="flex items-stretch">
    <div class="flex flex-[0_0_56px] w-[56px] h-[210px]">
      <span class="[writing-mode:vertical-rl] [transform:rotate(180deg)] self-center w-[14px] text-center text-[10px] tracking-[0.04em] uppercase text-(--muted)">advantage</span>
      <div class="relative flex-1 h-full">
        {#each model.ticks as t (t.val)}
          <span class="absolute right-[4px] [transform:translateY(-50%)] text-[10px] text-(--muted) [font-variant-numeric:tabular-nums] whitespace-nowrap" style:top={t.y + "px"}>{fmt(t.val)}</span>
        {/each}
      </div>
    </div>
    <div
      class="relative flex-1 min-w-0 h-[210px] overflow-hidden rounded-[4px]"
      role="presentation"
      onpointermove={onPlotMove}
      onpointerleave={onPlotLeave}
      use:brushZoom={{ onChangeDomainX: handleBrush, enabled: zoomable }}
    >
      <svg class="w-full h-[210px] block bg-[#0a0e14] rounded-[4px]" viewBox="0 0 {W} {H}" preserveAspectRatio="none" aria-hidden="true">
        {#each model.ticks as t (t.val)}
          <line class="stroke-[#fff] [stroke-opacity:0.08] [stroke-width:0.5]" x1="0" x2={W} y1={t.y} y2={t.y} />
        {/each}
        {#if model.zeroY != null}
          <line class="stroke-[#fff] [stroke-opacity:0.35] [stroke-width:0.75] [stroke-dasharray:4_4]" x1="0" x2={W} y1={model.zeroY} y2={model.zeroY} />
        {/if}
        {#each model.violins as v (v.x)}
          <path class="fill-(--accent) [fill-opacity:0.35] stroke-(--accent) [stroke-width:1] [stroke-opacity:0.9] [vector-effect:non-scaling-stroke] [transition:fill-opacity_0.08s_ease]" d={v.path} />
          <line class="median" x1={v.medX1} x2={v.medX2} y1={v.medY} y2={v.medY} />
        {/each}
      </svg>
      {#if hover}
        <div class="absolute left-0 w-full h-0 [border-top:1px_dashed_rgba(255,255,255,0.5)] pointer-events-none" style:top={hover.top + "px"}></div>
        <div
          class="absolute [transform:translate(10px,-50%)] p-[1px_6px] rounded-[4px] bg-[rgba(10,14,20,0.92)] [border:1px_solid_var(--accent)] [color:var(--text-bright,#fff)] text-[10px] [font-variant-numeric:tabular-nums] whitespace-nowrap pointer-events-none z-[2]"
          style:top={hover.top + "px"}
          style:left={hover.left + "px"}
        >
          {fmt(hover.value)}
        </div>
      {/if}
    </div>
  </div>
  <div class="pl-[56px]">
    {#if timeAxis}
      <TimeAxis
        start={timeAxis.start}
        end={timeAxis.end}
        fractionAt={(t) => columnFraction(timeToX(t))}
      />
    {/if}
    <div class="fan-meta">
      <span>min {fmt(model.yLo)}</span>
      <span>latest median {fmt(model.latestMedian)}</span>
      <span>max {fmt(model.yHi)}</span>
    </div>
    {#if model.showEachLabel}
      <div class="grid mt-[2px] text-[10px] text-(--muted)" style:grid-template-columns={`repeat(${model.violins.length}, 1fr)`}>
        {#each model.violins as v, i (v.x)}
          <span class="text-center overflow-hidden text-ellipsis whitespace-nowrap p-[0_2px]" title={labelFor(v, i)}>{labelFor(v, i)}</span>
        {/each}
      </div>
    {:else}
      <div class="fan-axis">
        <span>step {model.firstX}</span>
        <span class="fan-axis-label">training step</span>
        <span>step {model.lastX}</span>
      </div>
    {/if}
  </div>
{:else if hasDomain && allPts.length}
  <div
    class="relative h-[210px] rounded-[4px] bg-[#0a0e14] flex items-center justify-center gap-[10px] text-(--muted) text-[12px]"
    use:brushZoom={{ onChangeDomainX: handleBrush, enabled: zoomable }}
  >
    No data in this range.
    {#if zoomable}
      <ChartZoomButtons onChangeDomainX={handleBrush} canZoomIn={false} />
    {/if}
  </div>
{:else}
  <div class="plot-empty">Advantage distribution needs ≥1 step of data.</div>
{/if}
