<script>
  // Spread-over-time: is the advantage distribution widening or collapsing as
  // training proceeds? Each step's distribution is reduced to scalar spread
  // measures and plotted vs step, so "increasing?" is a single glance at the
  // slope rather than something you eyeball off overlapping bands.
  //
  //   • std   — standard deviation of the step's advantages (stats.std)
  //   • IQR   — p75 − p25, the robust middle-50% width
  //
  // A least-squares trend line is fitted to std and its net change over the run
  // is reported with a direction arrow.

  import { brushZoom } from "../lib/brushZoom.js";
  import ChartZoomButtons from "./ChartZoomButtons.svelte";
  import TimeAxis from "./TimeAxis.svelte";

  let {
    steps = [],
    // `[min, max]` rollout ids; defaults to the data extent.
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
  const H = 200;
  const PAD = 6;

  function num(v, fallback = 0) {
    const n = Number(v);
    return Number.isFinite(n) ? n : fallback;
  }

  function fmt(v) {
    return Number.isFinite(v) ? v.toFixed(3) : "—";
  }

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
          std: num(st.std),
          iqr: Math.max(num(q.p75, num(st.max)) - num(q.p25, num(st.min)), 0),
        };
      })
      .sort((a, b) => a.x - b.x),
  );

  let model = $derived.by(() => {
    const pts = hasDomain
      ? allPts.filter((p) => p.x >= xDomain[0] && p.x <= xDomain[1])
      : allPts;
    if (pts.length < 2 && !(hasDomain && allPts.length >= 2)) return null;

    const xs = pts.map((p) => p.x);
    const xMin = hasDomain ? xDomain[0] : Math.min(...xs);
    const xMax = hasDomain ? xDomain[1] : Math.max(...xs);
    const xSpan = xMax - xMin || 1;
    const yHi = (pts.length ? Math.max(...pts.map((p) => Math.max(p.std, p.iqr))) : 0) || 1;
    const ySpan = yHi || 1;

    const sx = (x) => PAD + ((x - xMin) / xSpan) * (W - 2 * PAD);
    const sy = (v) => H - PAD - (v / ySpan) * (H - 2 * PAD);

    const line = (key) =>
      pts
        .map((p, i) => `${i ? "L" : "M"} ${sx(p.x).toFixed(1)} ${sy(p[key]).toFixed(1)}`)
        .join(" ");

    // Least-squares fit of std vs step index for the trend line + net change.
    const n = pts.length;
    const mx = pts.reduce((a, p) => a + p.x, 0) / n;
    const my = pts.reduce((a, p) => a + p.std, 0) / n;
    let sxy = 0;
    let sxx = 0;
    for (const p of pts) {
      sxy += (p.x - mx) * (p.std - my);
      sxx += (p.x - mx) ** 2;
    }
    const slope = sxx ? sxy / sxx : 0;
    const intercept = my - slope * mx;
    const fit = (x) => intercept + slope * x;
    const firstX = pts.length ? Math.min(...xs) : xMin;
    const lastX = pts.length ? Math.max(...xs) : xMax;
    const fitStart = fit(firstX);
    const fitEnd = fit(lastX);
    const delta = fitEnd - fitStart;
    const pct = fitStart > 1e-9 ? (delta / fitStart) * 100 : null;
    const dir = delta > ySpan * 0.02 ? "up" : delta < -ySpan * 0.02 ? "down" : "flat";

    return {
      std: line("std"),
      iqr: line("iqr"),
      trend:
        pts.length >= 2
          ? `M ${sx(firstX).toFixed(1)} ${sy(fitStart).toFixed(1)} L ${sx(lastX).toFixed(1)} ${sy(fitEnd).toFixed(1)}`
          : "",
      dots: pts.map((p) => ({ cx: sx(p.x).toFixed(1), cy: sy(p.std).toFixed(1) })),
      yHi,
      xMin,
      xMax,
      firstX,
      lastX,
      count: pts.length,
      latestStd: pts.length ? pts[pts.length - 1].std : NaN,
      delta,
      pct,
      dir,
    };
  });

  function handleBrush([f0, f1]) {
    if (!zoomable || !model) return;
    const { xMin, xMax } = model;
    const plotW = W - 2 * PAD;
    const toX = (f) => xMin + ((f * W - PAD) / plotW) * (xMax - xMin);
    onChangeDomainX([toX(f0), toX(f1)]);
  }

  let timeAxis = $derived(
    model && typeof xToTime === "function" && typeof timeToX === "function"
      ? { start: xToTime(model.xMin), end: xToTime(model.xMax) }
      : null,
  );

  function timeFraction(t) {
    const { xMin, xMax } = model;
    return (PAD + ((timeToX(t) - xMin) / (xMax - xMin || 1)) * (W - 2 * PAD)) / W;
  }

  function fmtStep(x) {
    return Number.isInteger(x) ? String(x) : x.toFixed(1);
  }

  const ARROW = { up: "↑", down: "↓", flat: "→" };
</script>

{#if model}
  <div class="flex flex-wrap items-center justify-between gap-[8px] mb-[8px] text-[11px] text-(--muted)">
    <span class="inline-flex flex-wrap gap-[12px]">
      <span class="chart-legend-item"><span class="sw std"></span>std</span>
      <span class="chart-legend-item"><span class="sw iqr"></span>IQR (p25–p75)</span>
      <span class="chart-legend-item"><span class="sw trend"></span>trend</span>
    </span>
    <span class="inline-flex items-center gap-[10px]">
      <span class="spread-trend trend-{model.dir}">
        std {ARROW[model.dir]}
        {model.delta >= 0 ? "+" : ""}{fmt(model.delta)}
        {#if model.pct != null}
          ({model.pct >= 0 ? "+" : ""}{model.pct.toFixed(0)}%)
        {/if}
        over run
      </span>
      {#if zoomable}
        <ChartZoomButtons onChangeDomainX={handleBrush} canZoomIn={model.count > 1} />
      {/if}
    </span>
  </div>
  <div
    class="relative overflow-hidden rounded-[4px]"
    use:brushZoom={{ onChangeDomainX: handleBrush, enabled: zoomable }}
  >
    <svg class="w-full h-[200px] block" viewBox="0 0 {W} {H}" preserveAspectRatio="none" aria-hidden="true">
      <path d={model.iqr} fill="none" stroke="var(--muted)" stroke-width="1.25" stroke-opacity="0.7" />
      {#if model.trend}
        <path
          d={model.trend}
          fill="none"
          stroke="var(--accent)"
          stroke-width="1"
          stroke-opacity="0.55"
          stroke-dasharray="5 4"
        />
      {/if}
      <path d={model.std} fill="none" stroke="var(--accent)" stroke-width="1.75" />
      {#each model.dots as d}
        <circle cx={d.cx} cy={d.cy} r="2" fill="var(--accent)" />
      {/each}
    </svg>
    {#if !model.count}
      <div class="absolute inset-0 flex items-center justify-center text-(--muted) text-[12px] pointer-events-none">
        No data in this range.
      </div>
    {/if}
  </div>
  {#if timeAxis}
    <TimeAxis start={timeAxis.start} end={timeAxis.end} fractionAt={timeFraction} />
  {/if}
  <div class="fan-meta">
    <span>0</span>
    <span>latest std {fmt(model.latestStd)}</span>
    <span>max {fmt(model.yHi)}</span>
  </div>
  <div class="fan-axis">
    <span>step {fmtStep(model.xMin)}</span>
    <span class="fan-axis-label">training step</span>
    <span>step {fmtStep(model.xMax)}</span>
  </div>
{:else}
  <div class="plot-empty">Advantage distribution needs ≥2 steps of data.</div>
{/if}
