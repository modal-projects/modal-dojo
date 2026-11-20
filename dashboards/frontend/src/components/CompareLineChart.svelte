<script>
  // One metric panel with a line per run, for the Metrics compare page. Unlike
  // LineChart, series don't share x positions: each run logs on its own steps,
  // so every series keeps its own points and hover snaps each run to its
  // nearest step.
  import { brushZoom, fractionsToDomain } from "../lib/brushZoom.js";
  import { nearestIndex } from "../lib/metricSeries.js";
  import { trailingMean } from "../lib/smoothing.js";
  import { niceTicks } from "../lib/ticks.js";
  import ChartZoomButtons from "./ChartZoomButtons.svelte";

  let {
    title = "",
    // `[{ id, label, color, points: [{ x, y }] }]`, points sorted by x.
    series = [],
    height = 160,
    smoothingWindow = 1,
    xLabel = "Step",
    formatY = (value) => String(value),
    xDomain = null,
    onChangeDomainX = null,
  } = $props();

  let chartEl = $state(null);
  let hoverX = $state(null);

  let drawn = $derived(
    series
      .map((s) => ({
        ...s,
        points: smoothingWindow > 1 ? trailingMean(s.points, ["y"], smoothingWindow) : s.points,
      }))
      .filter((s) => s.points.length),
  );

  let hasDomain = $derived(
    Array.isArray(xDomain) && Number.isFinite(xDomain[0]) && Number.isFinite(xDomain[1]) && xDomain[1] > xDomain[0],
  );
  let dataExtent = $derived.by(() => {
    let lo = Infinity;
    let hi = -Infinity;
    for (const s of drawn) {
      lo = Math.min(lo, s.points[0].x);
      hi = Math.max(hi, s.points.at(-1).x);
    }
    return Number.isFinite(lo) ? [lo, hi] : [0, 1];
  });
  let xMin = $derived(hasDomain ? xDomain[0] : dataExtent[0]);
  let xMax = $derived(hasDomain ? xDomain[1] : dataExtent[1]);
  let xSpan = $derived(xMax - xMin || 1);

  let yExtent = $derived.by(() => {
    let lo = Infinity;
    let hi = -Infinity;
    for (const s of drawn) {
      for (const p of s.points) {
        if (p.x < xMin || p.x > xMax) continue;
        if (p.y < lo) lo = p.y;
        if (p.y > hi) hi = p.y;
      }
    }
    if (!Number.isFinite(lo)) return [0, 1];
    const pad = (hi - lo || Math.abs(lo) || 1) * 0.08;
    return [lo - pad, hi + pad];
  });
  let ySpan = $derived(yExtent[1] - yExtent[0] || 1);
  let yTicks = $derived(niceTicks(yExtent[0], yExtent[1], 4));
  let xTicks = $derived(xMax > xMin ? niceTicks(xMin, xMax, 5) : []);

  const xPct = (x) => (xMax > xMin ? ((x - xMin) / xSpan) * 100 : 50);
  const yPct = (y) => 100 - ((y - yExtent[0]) / ySpan) * 100;

  function pathFor(points) {
    // Keep one neighbour on each side of the window so lines run off the edge.
    let first = 0;
    let last = points.length - 1;
    if (hasDomain) {
      while (first < last && points[first + 1].x < xMin) first++;
      while (last > first && points[last - 1].x > xMax) last--;
    }
    let d = "";
    for (let i = first; i <= last; i++) {
      d += `${i === first ? "M" : " L"} ${xPct(points[i].x).toFixed(3)} ${yPct(points[i].y).toFixed(3)}`;
    }
    return d;
  }
  let paths = $derived(drawn.map((s) => ({ ...s, d: pathFor(s.points) })));
  let singles = $derived(drawn.filter((s) => s.points.length === 1));

  let hovered = $derived.by(() => {
    if (hoverX == null) return [];
    return drawn
      .map((s) => {
        // A run that stopped logging before (or started after) the cursor has
        // no value there; allow one logging interval of slack at either end.
        const first = s.points[0].x;
        const last = s.points.at(-1).x;
        const slack = s.points.length > 1 ? (last - first) / (s.points.length - 1) : 0;
        if (hoverX < first - slack || hoverX > last + slack) return null;
        const point = s.points[nearestIndex(s.points, hoverX)];
        return point.x >= xMin && point.x <= xMax ? { ...s, point } : null;
      })
      .filter(Boolean)
      .sort((a, b) => b.point.y - a.point.y);
  });
  let hoverPct = $derived(hoverX == null ? null : xPct(hoverX));

  let zoomable = $derived(typeof onChangeDomainX === "function");
  function handleBrush(fractions) {
    if (zoomable) onChangeDomainX(fractionsToDomain(fractions, [xMin, xMax]));
  }

  function onPointerMove(event) {
    if (!chartEl) return;
    const rect = chartEl.getBoundingClientRect();
    const ratio = rect.width ? Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)) : 0;
    hoverX = xMin + ratio * xSpan;
  }
</script>

<div class="min-w-0">
  <div class="flex items-center gap-[8px] mb-[6px]">
    <div class="text-(--text-bright) text-[12px] font-[600] mr-auto min-w-0 truncate" {title}>{title}</div>
    {#if zoomable}
      <ChartZoomButtons onChangeDomainX={handleBrush} canZoomIn={drawn.length > 0} />
    {/if}
  </div>

  {#if drawn.length}
    <div class="line-chart-axes">
      <div class="line-chart-yaxis" aria-hidden="true">
        {#each yTicks as tick (tick)}
          <span style:top={`${yPct(tick)}%`}>{formatY(tick)}</span>
        {/each}
      </div>
      <!-- The tooltip sits outside the clipped plot so it can overhang the panel. -->
      <div class="relative">
      <div
        class="relative overflow-hidden bg-(--color-c-gray-08,#1c1c1c) rounded-[6px] cursor-crosshair"
        bind:this={chartEl}
        style:height={`${height}px`}
        role="img"
        aria-label={`${title} for ${drawn.length} run${drawn.length === 1 ? "" : "s"}`}
        onpointermove={onPointerMove}
        onpointerleave={() => (hoverX = null)}
        use:brushZoom={{ onChangeDomainX: handleBrush, enabled: zoomable }}
      >
        <svg class="block w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
          {#each yTicks as tick (tick)}
            <line x1="0" x2="100" y1={yPct(tick)} y2={yPct(tick)} class="stroke-[rgba(255,255,255,0.07)] [stroke-width:1]" vector-effect="non-scaling-stroke" />
          {/each}
          {#each paths as s (s.id)}
            <path d={s.d} fill="none" stroke={s.color} stroke-width="1.5" vector-effect="non-scaling-stroke" />
          {/each}
          {#if hoverPct != null}
            <line x1={hoverPct} x2={hoverPct} y1="0" y2="100" class="stroke-[rgba(255,255,255,0.22)] [stroke-width:1]" vector-effect="non-scaling-stroke" />
          {/if}
        </svg>

        {#each singles as s (s.id)}
          <span class="point-dot" style:left={`${xPct(s.points[0].x)}%`} style:top={`${yPct(s.points[0].y)}%`} style:background={s.color}></span>
        {/each}

        {#each hovered as h (h.id)}
          <span
            class="point-dot z-[2]!"
            style:left={`${xPct(h.point.x)}%`}
            style:top={`${yPct(h.point.y)}%`}
            style:background={h.color}
          ></span>
        {/each}
      </div>
        {#if hovered.length}
          <div class="chart-tooltip z-[30]!" class:reverse={hoverPct > 50} style:left={`${hoverPct}%`}>
            <div class="text-[rgba(255,255,255,0.5)] text-[11px]">{xLabel}</div>
            {#each hovered as h (h.id)}
              <div class="flex items-center gap-[6px] [font-variant-numeric:tabular-nums]">
                <span class="inline-block w-[8px] h-[8px] rounded-[2px] shrink-0" style:background={h.color}></span>
                <span class="text-[rgba(255,255,255,0.72)] max-w-[200px] truncate">{h.label}</span>
                <span class="text-[rgba(255,255,255,0.45)] text-[11px]">@{h.point.x}</span>
                <span class="[color:white] font-[600] ml-auto pl-[8px]">{formatY(h.point.y)}</span>
              </div>
            {/each}
            {#if smoothingWindow > 1}
              <div class="text-[rgba(255,255,255,0.5)] text-[10px]">smoothed · last {smoothingWindow}</div>
            {/if}
          </div>
        {/if}
      </div>
      <div class="line-chart-xaxis" aria-hidden="true">
        {#each xTicks as tick (tick)}
          {@const f = (tick - xMin) / xSpan}
          <span class:anchor-start={f < 0.08} class:anchor-end={f > 0.92} style:left={`${f * 100}%`}>{tick}</span>
        {/each}
      </div>
    </div>
  {:else}
    <div class="text-(--muted) text-[12px] leading-[16px]">No data.</div>
  {/if}
</div>
