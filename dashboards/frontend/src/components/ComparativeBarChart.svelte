<script>
  // Generic grouped ("comparative") bar chart: one cluster per category, one bar
  // per series within the cluster, so several series can be compared side-by-side
  // at each category. Hovering a bar (or its cluster) surfaces a tooltip with the
  // category and every series' value; each bar also carries a native `title` as a
  // no-JS / screen-reader fallback.
  //
  // Signed values are supported: bars grow up from a shared zero baseline for
  // positive values and down for negative ones.
  //
  // Props
  //   categories        string[]                       — one cluster per entry (x labels)
  //   series            { name, values:number[], color? }[]
  //   height            number (px plot height)
  //   format            (value) => string              — tooltip / label formatter
  //   axisLabel         string                         — caption under the x-axis
  //   showLegend        boolean (auto-off for 1 series)
  //   showCategoryLabels boolean                       — per-cluster x labels
  //   showValueLabels   boolean                        — direct labels above bars
  //   yMax              number | null                  — fix the positive top of the scale
  //   emptyText         string
  //   zoomable          boolean                        — drag over the plot (or the +/−
  //                                                      buttons) narrows the visible span
  //                                                      of categories

  import ZoomOutIcon from "lucide-svelte/icons/zoom-out";

  import { brushZoom } from "../lib/brushZoom.js";
  import ChartZoomButtons from "./ChartZoomButtons.svelte";

  let {
    categories = [],
    series = [],
    height = 150,
    format = (v) => `${v}`,
    axisLabel = "",
    showLegend = true,
    showCategoryLabels = true,
    showValueLabels = false,
    yMax = null,
    emptyText = "No data to compare.",
    zoomable = true,
  } = $props();

  // Inclusive `[lo, hi]` category indices on screen; `null` shows everything.
  let zoomRange = $state(null);
  let categoryCount = $derived((categories || []).length);
  $effect(() => {
    categoryCount;
    zoomRange = null;
  });

  let visibleRange = $derived.by(() => {
    const n = categoryCount;
    if (!n) return [0, -1];
    if (!zoomRange) return [0, n - 1];
    const lo = Math.max(0, Math.min(n - 1, zoomRange[0]));
    const hi = Math.max(lo, Math.min(n - 1, zoomRange[1]));
    return [lo, hi];
  });
  let isZoomed = $derived(
    categoryCount > 0 && (visibleRange[0] > 0 || visibleRange[1] < categoryCount - 1),
  );

  function setZoom(lo, hi) {
    const n = categoryCount;
    if (!n) return;
    lo = Math.max(0, Math.min(n - 1, Math.round(lo)));
    hi = Math.max(lo, Math.min(n - 1, Math.round(hi)));
    zoomRange = lo === 0 && hi === n - 1 ? null : [lo, hi];
  }

  // The plot lays visible clusters out in equal columns, so brush fractions
  // are column offsets from the current window.
  function handleBrush([f0, f1]) {
    const [lo, hi] = visibleRange;
    const count = hi - lo + 1;
    if (count <= 0) return;
    // Column `i` spans `[i, i + 1)`; a column counts as selected when more
    // than half of it is covered.
    let newLo = Math.round(lo + f0 * count);
    let newHi = Math.round(lo + f1 * count) - 1;
    if (f1 - f0 > 1) {
      // Expand on both sides so clamping at an edge cannot undo the step.
      newLo = Math.min(newLo, lo - 1);
      newHi = Math.max(newHi, hi + 1);
    } else if (newLo === lo && newHi === hi && f1 - f0 < 1 && count > 1) {
      if (f0 > 1 - f1) newLo = lo + 1;
      else newHi = hi - 1;
    }
    setZoom(newLo, newHi);
  }

  function resetZoom() {
    zoomRange = null;
  }

  let canZoom = $derived(zoomable && categoryCount > 1);

  // Validated dark-surface categorical ramp (skill reference palette), in fixed
  // order — never cycled cosmetically; the order itself is the CVD-safety choice.
  const DEFAULT_PALETTE = [
    "#3987e5", // blue
    "#199e70", // aqua
    "#c98500", // yellow
    "#008300", // green
    "#9085e9", // violet
    "#e66767", // red
    "#d55181", // magenta
    "#d95926", // orange
  ];

  const colorFor = (s, si) => s.color || DEFAULT_PALETTE[si % DEFAULT_PALETTE.length];

  function finite(v) {
    const n = Number(v);
    return Number.isFinite(n) ? n : 0;
  }

  let model = $derived.by(() => {
    const cats = categories || [];
    const ser = (series || []).filter((s) => s && Array.isArray(s.values));
    if (!cats.length || !ser.length) return null;

    const [lo, hi] = visibleRange;

    // Loop (not Math.min(...arr)): value arrays can be large enough that the
    // spread would blow the engine's max-argument limit.
    let dataMax = 0;
    let dataMin = 0;
    for (const s of ser) {
      for (let ci = lo; ci <= hi; ci++) {
        const v = finite(s.values[ci]);
        if (v > dataMax) dataMax = v;
        if (v < dataMin) dataMin = v;
      }
    }
    const posTop = yMax != null && Number.isFinite(yMax) ? Math.max(yMax, dataMax) : dataMax;
    const span = posTop - dataMin || 1;
    const zeroPct = (posTop / span) * 100; // distance from top down to the zero line

    const clusters = cats.slice(lo, hi + 1).map((label, offset) => {
      const ci = lo + offset;
      const bars = ser.map((s, si) => {
        const v = finite(s.values[ci]);
        const magPct = (Math.abs(v) / span) * 100;
        return {
          si,
          name: s.name ?? `series ${si + 1}`,
          color: colorFor(s, si),
          value: v,
          positive: v >= 0,
          // Keep a sliver visible for tiny-but-nonzero values.
          heightPct: v === 0 ? 0 : Math.max(magPct, 0.8),
          topPct: v >= 0 ? zeroPct - Math.max(magPct, v === 0 ? 0 : 0.8) : zeroPct,
        };
      });
      return { label: label == null ? `${ci}` : String(label), ci, bars };
    });

    return {
      clusters,
      total: cats.length,
      series: ser.map((s, si) => ({ name: s.name ?? `series ${si + 1}`, color: colorFor(s, si) })),
      zeroPct,
      hasNegative: dataMin < 0,
      multi: ser.length > 1,
    };
  });

  let hover = $state(null); // { ci, si }
  let ptr = $state({ x: 0, y: 0, w: 0, h: 0 });

  function onPlotMove(e) {
    const r = e.currentTarget.getBoundingClientRect();
    ptr = { x: e.clientX - r.left, y: e.clientY - r.top, w: r.width, h: r.height };
  }

  function enterBar(ci, si) {
    hover = { ci, si };
  }

  function clearHover() {
    hover = null;
  }

  let tip = $derived.by(() => {
    if (!hover || !model) return null;
    const cluster = model.clusters.find((c) => c.ci === hover.ci);
    if (!cluster) return null;
    const TIP_W = 168;
    const left = Math.max(0, Math.min(ptr.x + 12, Math.max(0, ptr.w - TIP_W)));
    const top = Math.max(0, Math.min(ptr.y, ptr.h));
    return {
      left,
      top,
      label: cluster.label,
      rows: cluster.bars.map((b) => ({
        name: b.name,
        color: b.color,
        value: format(b.value),
        active: b.si === hover.si,
      })),
    };
  });

  const barTitle = (clusterLabel, bar) =>
    `${clusterLabel} · ${bar.name}: ${format(bar.value)}`;
</script>

{#if model}
  {#if (showLegend && model.multi) || isZoomed || canZoom}
    <div class="flex flex-wrap items-center gap-[16px] mb-[8px] text-[11px] text-(--muted)">
      {#if showLegend && model.multi}
        {#each model.series as s (s.name)}
          <span class="inline-flex items-center gap-[6px]">
            <span class="cbc-swatch" style:background={s.color}></span>
            {s.name}
          </span>
        {/each}
      {/if}
      {#if isZoomed}
        <button
          type="button"
          class="cbc-zoom-reset"
          onclick={resetZoom}
          title="Show all categories"
        >
          <ZoomOutIcon size={12} />
          {model.clusters[0]?.label} – {model.clusters[model.clusters.length - 1]?.label}
          <span class="text-(--text)">· reset</span>
        </button>
      {/if}
      {#if canZoom}
        <span class="ml-auto">
          <ChartZoomButtons
            onChangeDomainX={handleBrush}
            canZoomIn={visibleRange[1] > visibleRange[0]}
            canZoomOut={isZoomed}
          />
        </span>
      {/if}
    </div>
  {/if}

  <div
    class="cbc-plot overflow-hidden"
    class:has-negative={model.hasNegative}
    style:height={`${height}px`}
    role="img"
    aria-label={`Comparative bar chart across ${model.clusters.length} of ${model.total} categories`}
    onpointermove={onPlotMove}
    onpointerleave={clearHover}
    use:brushZoom={{ onChangeDomainX: handleBrush, enabled: zoomable && model.total > 1 }}
  >
    <div class="cbc-baseline" style:top={`${model.zeroPct}%`}></div>

    <div class="absolute inset-0 flex items-stretch gap-[3px]">
      {#each model.clusters as cluster (cluster.ci)}
        <div class="cbc-cluster" class:dim={hover && hover.ci !== cluster.ci}>
          {#each cluster.bars as bar (bar.si)}
            <div class="relative flex-[1_1_0] min-w-0">
              <div
                class="cbc-bar"
                class:negative={!bar.positive}
                class:cbar-active={hover && hover.ci === cluster.ci && hover.si === bar.si}
                style:background={bar.color}
                style:top={`${bar.topPct}%`}
                style:height={`${bar.heightPct}%`}
                title={barTitle(cluster.label, bar)}
                role="presentation"
                onpointerenter={() => enterBar(cluster.ci, bar.si)}
              >
                {#if showValueLabels && bar.value !== 0}
                  <span class="absolute top-[-14px] left-[50%] [transform:translateX(-50%)] text-[9px] text-(--muted) [font-variant-numeric:tabular-nums] whitespace-nowrap pointer-events-none">{format(bar.value)}</span>
                {/if}
              </div>
            </div>
          {/each}
        </div>
      {/each}
    </div>

    {#if tip}
      <div class="absolute z-[5] [transform:translateY(-50%)] min-w-[120px] max-w-[168px] p-[6px_8px] rounded-[6px] bg-[rgba(10,14,20,0.94)] [border:1px_solid_var(--border-strong,#464646)] [box-shadow:0_4px_14px_rgba(0,0,0,0.4)] pointer-events-none text-[11px]" style:left={`${tip.left}px`} style:top={`${tip.top}px`}>
        <div class="[color:var(--text-bright,#fff)] font-[600] mb-[4px] [font-variant-numeric:tabular-nums]">{tip.label}</div>
        {#each tip.rows as row (row.name)}
          <div class="cbc-tip-row" class:cbar-active={row.active}>
            <span class="cbc-swatch" style:background={row.color}></span>
            <span class="flex-[1_1_auto] overflow-hidden text-ellipsis whitespace-nowrap">{row.name}</span>
            <span class="[font-variant-numeric:tabular-nums]">{row.value}</span>
          </div>
        {/each}
      </div>
    {/if}
  </div>

  {#if showCategoryLabels}
    <div class="flex gap-[3px] mt-[4px]">
      {#each model.clusters as cluster (cluster.ci)}
        <span class="flex-[1_1_0] min-w-0 text-center text-[10px] text-(--muted) overflow-hidden text-ellipsis whitespace-nowrap p-[0_1px]" title={cluster.label}>{cluster.label}</span>
      {/each}
    </div>
  {/if}
  {#if axisLabel}
    <div class="mt-[2px] text-center text-[10px] uppercase tracking-[0.04em] text-(--muted)">{axisLabel}</div>
  {/if}
{:else}
  <div class="plot-empty">{emptyText}</div>
{/if}
