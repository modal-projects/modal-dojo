<script>
  import { brushZoom, fractionsToDomain } from "../lib/brushZoom.js";
  import { trailingMean } from "../lib/smoothing.js";
  import { niceTicks } from "../lib/ticks.js";
  import ChartZoomButtons from "./ChartZoomButtons.svelte";
  import TimeAxis from "./TimeAxis.svelte";

  let {
    title = "",
    data = [],
    height = 140,
    color = "var(--accent)",
    label = "mean",
    // `{ key, label, color, dash }` secondary series read from `row[key]`.
    lines = [],
    smoothable = false,
    smoothingWindow = 5,
    // W&B-style panel: y range fit to the data, tick labels + gridlines.
    axes = false,
    ariaLabel = title || "Line chart",
    formatX = (row) => String(row?.x ?? ""),
    formatY = (value) => String(value),
    // `[min, max]` in x units; defaults to the data extent.
    xDomain = null,
    // Called with `[min, max]` in x units when the user drags a window or
    // presses the zoom buttons.
    onChangeDomainX = null,
    // x units <-> epoch seconds; when both are given a wall-clock axis is
    // drawn under the plot.
    xToTime = null,
    timeToX = null,
  } = $props();

  let chartEl = $state(null);
  let hoveredIndex = $state(null);
  let smoothing = $state(false);
  let hiddenKeys = $state(new Set());
  let pendingEvent = null;
  let frame = null;

  function num(value) {
    const n = Number(value);
    return value == null || value === "" || !Number.isFinite(n) ? null : n;
  }

  let extraLines = $derived(
    (Array.isArray(lines) ? lines : []).filter((line) => line && typeof line.key === "string"),
  );
  let extraKeys = $derived(extraLines.map((line) => line.key));
  let seriesKeys = $derived(["y", ...extraKeys]);
  let visibleExtraLines = $derived(extraLines.filter((line) => !hiddenKeys.has(line.key)));
  let visibleKeys = $derived([
    ...(hiddenKeys.has("y") ? [] : ["y"]),
    ...visibleExtraLines.map((line) => line.key),
  ]);

  let rawRows = $derived.by(() => {
    const extras = extraKeys;
    return (Array.isArray(data) ? data : [])
      .map((row, index) => {
        const out = { ...row, index, x: Number(row?.x), y: num(row?.y) };
        for (const key of extras) out[key] = num(row?.[key]);
        return out;
      })
      .filter((row) => Number.isFinite(row.x) && row.y != null);
  });
  let smoothingOn = $derived(smoothable && smoothing);
  let allRows = $derived(
    smoothingOn ? trailingMean(rawRows, seriesKeys, smoothingWindow) : rawRows,
  );

  let hasDomain = $derived(
    Array.isArray(xDomain) &&
      Number.isFinite(xDomain[0]) &&
      Number.isFinite(xDomain[1]) &&
      xDomain[1] > xDomain[0],
  );
  let xMin = $derived(
    hasDomain ? xDomain[0] : allRows.length ? Math.min(...allRows.map((row) => row.x)) : 0,
  );
  let xMax = $derived(
    hasDomain ? xDomain[1] : allRows.length ? Math.max(...allRows.map((row) => row.x)) : 1,
  );
  let xSpan = $derived(xMax - xMin || 1);

  // Rows inside the visible window drive the y scale and hover; neighbours
  // just outside it are kept so the line runs off the edge instead of
  // stopping short.
  let rows = $derived(
    hasDomain ? allRows.filter((row) => row.x >= xMin && row.x <= xMax) : allRows,
  );
  let drawnRows = $derived.by(() => {
    if (!hasDomain) return allRows;
    const first = allRows.findIndex((row) => row.x >= xMin);
    if (first === -1) return allRows.slice(-1);
    let last = allRows.length - 1;
    while (last > 0 && allRows[last].x > xMax) last--;
    return allRows.slice(Math.max(0, first - 1), Math.min(allRows.length, last + 2));
  });

  let yValues = $derived(
    rows.flatMap((row) => visibleKeys.map((key) => row[key]).filter((v) => v != null)),
  );
  let yExtent = $derived.by(() => {
    if (!yValues.length) return [0, 1];
    const lo = Math.min(...yValues);
    const hi = Math.max(...yValues);
    if (!axes && hi > lo) return [Math.min(0, lo), Math.max(0, hi)];
    const pad = (hi - lo || Math.abs(lo) || 1) * 0.08;
    return [lo - pad, hi + pad];
  });
  let yMin = $derived(yExtent[0]);
  let yMax = $derived(yExtent[1]);
  let ySpan = $derived(yMax - yMin || 1);
  let singlePoint = $derived(rows.length === 1 && !hasDomain);
  let yTicks = $derived(axes ? niceTicks(yMin, yMax, 4) : []);
  let xTicks = $derived(axes && !singlePoint ? niceTicks(xMin, xMax, 5) : []);

  function yPercent(value) {
    return axes ? 100 - ((value - yMin) / ySpan) * 100 : 100 - ((value - yMin) / ySpan) * 96 - 2;
  }
  function point(row, key = "y") {
    const x = singlePoint ? 2 : ((row.x - xMin) / xSpan) * 100;
    return { x, y: yPercent(row[key]) };
  }
  let singlePointMarkers = $derived(
    singlePoint
      ? [{ key: "y", color }, ...visibleExtraLines]
          .filter((line) => !hiddenKeys.has(line.key) && rows[0][line.key] != null)
          .map((line) => ({ ...line, ...point(rows[0], line.key) }))
      : [],
  );

  let zoomable = $derived(typeof onChangeDomainX === "function");
  let timeAxis = $derived(
    typeof xToTime === "function" && typeof timeToX === "function" && !singlePoint
      ? { start: xToTime(xMin), end: xToTime(xMax) }
      : null,
  );

  function handleBrush(fractions) {
    if (!zoomable) return;
    onChangeDomainX(fractionsToDomain(fractions, [xMin, xMax]));
  }

  function pathFor(key) {
    let d = "";
    let pen = false;
    for (const row of drawnRows) {
      if (row[key] == null) {
        pen = false;
        continue;
      }
      const p = point(row, key);
      d += `${pen ? " L" : " M"} ${p.x.toFixed(3)} ${p.y.toFixed(3)}`;
      pen = true;
    }
    return d.trim();
  }
  let path = $derived(hiddenKeys.has("y") ? "" : pathFor("y"));
  let extraPaths = $derived(visibleExtraLines.map((line) => ({ ...line, d: pathFor(line.key) })));

  let hoveredRow = $derived(
    hoveredIndex == null ? null : rows[Math.max(0, Math.min(rows.length - 1, hoveredIndex))],
  );
  let hoveredPoint = $derived(hoveredRow ? point(hoveredRow) : null);
  let reverseTooltip = $derived(hoveredPoint ? hoveredPoint.x > (axes ? 55 : 72) : false);
  let hoveredExtras = $derived(
    hoveredRow
      ? visibleExtraLines
          .filter((line) => hoveredRow[line.key] != null)
          .map((line) => ({ ...line, value: hoveredRow[line.key] }))
      : [],
  );
  let showLegend = $derived(extraLines.length > 0);

  function toggleKey(key) {
    const next = new Set(hiddenKeys);
    if (next.has(key)) next.delete(key);
    else next.add(key);
    hiddenKeys = next;
  }

  function updateHoverFromPointer(event) {
    if (!chartEl || !rows.length) return;
    const rect = chartEl.getBoundingClientRect();
    const x = Math.max(0, Math.min(rect.width, event.clientX - rect.left));
    const ratio = rect.width ? x / rect.width : 0;
    const targetX = xMin + ratio * xSpan;
    let nearestIndex = 0;
    let nearestDistance = Infinity;
    rows.forEach((row, index) => {
      const distance = Math.abs(row.x - targetX);
      if (distance < nearestDistance) {
        nearestDistance = distance;
        nearestIndex = index;
      }
    });
    hoveredIndex = nearestIndex;
  }

  function onPointerMove(event) {
    pendingEvent = event;
    if (frame != null) return;
    frame = requestAnimationFrame(() => {
      frame = null;
      if (pendingEvent) updateHoverFromPointer(pendingEvent);
    });
  }

  function onPointerLeave() {
    hoveredIndex = null;
    pendingEvent = null;
  }

  $effect(() => {
    return () => {
      if (frame != null) cancelAnimationFrame(frame);
    };
  });
</script>

<div class="min-w-0">
  {#if title || showLegend || smoothable || zoomable}
    <div class="flex flex-wrap items-center gap-x-[12px] gap-y-[4px] mb-[6px]">
      {#if title}
        <div class="text-(--text-bright) text-[12px] font-[600] mr-auto min-w-0 truncate" {title}>{title}</div>
      {/if}
      {#if showLegend}
        <div class="flex flex-wrap items-center gap-[10px] text-[11px] text-(--muted)" aria-label="Series">
          {#each [{ key: "y", label, color }, ...extraLines] as line (line.key)}
            <button
              type="button"
              class="inline-flex items-center gap-[5px] bg-transparent border-0 p-0 text-inherit cursor-pointer hover:text-(--text-bright)"
              class:opacity-40={hiddenKeys.has(line.key)}
              aria-pressed={!hiddenKeys.has(line.key)}
              title={`${hiddenKeys.has(line.key) ? "Show" : "Hide"} ${line.label}`}
              onclick={() => toggleKey(line.key)}
            >
              <span
                class="inline-block w-[12px] h-0 border-t-2"
                class:border-dashed={Boolean(line.dash)}
                style:border-color={line.color}
              ></span>
              {line.label}
            </button>
          {/each}
        </div>
      {/if}
      {#if smoothable}
        <label
          class="inline-flex items-center gap-[5px] text-[11px] text-(--muted) cursor-pointer select-none"
          title={`Trailing mean over the last ${smoothingWindow} steps`}
        >
          <input type="checkbox" bind:checked={smoothing} />
          <span>Smoothing</span>
        </label>
      {/if}
      {#if zoomable}
        <ChartZoomButtons
          onChangeDomainX={handleBrush}
          canZoomIn={rows.length > 1 || hasDomain}
        />
      {/if}
    </div>
  {/if}

  {#if rows.length || (hasDomain && allRows.length)}
    <div class:line-chart-axes={axes}>
    {#if axes}
      <div class="line-chart-yaxis" aria-hidden="true">
        {#each yTicks as tick (tick)}
          <span style:top={`${yPercent(tick)}%`}>{formatY(tick)}</span>
        {/each}
      </div>
    {/if}
    <div
      class="relative overflow-hidden bg-(--color-c-gray-08,#1c1c1c) rounded-[6px] cursor-crosshair"
      bind:this={chartEl}
      style:height={`${height}px`}
      role="img"
      aria-label={ariaLabel}
      onpointermove={onPointerMove}
      onpointerleave={onPointerLeave}
      use:brushZoom={{ onChangeDomainX: handleBrush, enabled: zoomable }}
    >
      <svg class="block w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
        {#each yTicks as tick (tick)}
          <line
            x1="0"
            x2="100"
            y1={yPercent(tick)}
            y2={yPercent(tick)}
            class="stroke-[rgba(255,255,255,0.07)] [stroke-width:1]"
            vector-effect="non-scaling-stroke"
          />
        {/each}
        {#each extraPaths as line (line.key)}
          <path
            d={line.d}
            fill="none"
            stroke={line.color}
            stroke-width="1"
            stroke-dasharray={line.dash || null}
            vector-effect="non-scaling-stroke"
          />
        {/each}
        <path d={path} fill="none" stroke={color} stroke-width="1.5" vector-effect="non-scaling-stroke" />
        {#if hoveredPoint}
          <line
            x1={hoveredPoint.x}
            x2={hoveredPoint.x}
            y1="0"
            y2="100"
            class="stroke-[rgba(255,255,255,0.22)] [stroke-width:1]"
            vector-effect="non-scaling-stroke"
          />
        {/if}
      </svg>

      {#if !rows.length}
        <div class="absolute inset-0 flex items-center justify-center text-(--muted) text-[12px] pointer-events-none">
          No data in this range.
        </div>
      {/if}

      {#each singlePointMarkers as p (p.key)}
        <span
          class="point-dot"
          style:width={p.key === "y" ? null : "5px"}
          style:height={p.key === "y" ? null : "5px"}
          style:left={`${p.x}%`}
          style:top={`${p.y}%`}
          style:background={p.color}
        ></span>
      {/each}

      {#if hoveredPoint && !hiddenKeys.has("y")}
        <span
          class="point-dot z-[2]! w-[8px]! h-[8px]!"
          style:left={`${hoveredPoint.x}%`}
          style:top={`${hoveredPoint.y}%`}
          style:background={color}
        ></span>
      {/if}

      {#if hoveredRow && hoveredPoint}
        <div
          class="chart-tooltip"
          class:reverse={reverseTooltip}
          style:left={`${hoveredPoint.x}%`}
        >
          <div class="text-[rgba(255,255,255,0.72)]">{formatX(hoveredRow)}</div>
          {#if showLegend && !hiddenKeys.has("y")}
            <div class="flex items-center gap-[5px] [font-variant-numeric:tabular-nums]">
              <span class="inline-block w-[8px] h-[8px] rounded-[2px]" style:background={color}></span>
              <span class="text-[rgba(255,255,255,0.72)]">{label}</span>
              <span class="[color:white] font-[600]">{formatY(hoveredRow.y, hoveredRow)}</span>
            </div>
          {:else if !showLegend}
            <div class="[color:white] font-[600] [font-variant-numeric:tabular-nums]">{formatY(hoveredRow.y, hoveredRow)}</div>
          {/if}
          {#each hoveredExtras as line (line.key)}
            <div class="flex items-center gap-[5px] [font-variant-numeric:tabular-nums]">
              <span class="inline-block w-[8px] h-[8px] rounded-[2px]" style:background={line.color}></span>
              <span class="text-[rgba(255,255,255,0.72)]">{line.label}</span>
              <span class="[color:white]">{formatY(line.value, hoveredRow)}</span>
            </div>
          {/each}
          {#if smoothingOn}
            <div class="text-[rgba(255,255,255,0.5)] text-[10px]">smoothed · last {smoothingWindow}</div>
          {/if}
        </div>
      {/if}
    </div>
    {#if axes}
      <div class="line-chart-xaxis" aria-hidden="true">
        {#each xTicks as tick (tick)}
          {@const f = (tick - xMin) / xSpan}
          <span
            class:anchor-start={f < 0.08}
            class:anchor-end={f > 0.92}
            style:left={`${f * 100}%`}>{tick}</span>
        {/each}
      </div>
    {/if}
    </div>
    {#if timeAxis}
      <TimeAxis
        start={timeAxis.start}
        end={timeAxis.end}
        fractionAt={(t) => (timeToX(t) - xMin) / xSpan}
      />
    {/if}
  {:else}
    <div class="text-(--muted) text-[12px] leading-[16px]">No data.</div>
  {/if}
</div>
