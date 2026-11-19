<script>
  import { onDestroy } from "svelte";
  import { Download, ZoomOut } from "lucide-svelte";
  import { brushZoom } from "../lib/brushZoom.js";
  import {
    CATEGORIES,
    categoryLabelFor,
    APPROXIMATE_LANE_NOTE,
    breakLabelLayout,
    colorFor,
    fmtSecs,
    groupTooltipChildren,
    labelFor,
    PHASE_COLORS,
    TRAIN_OUTLINE_COLOR,
    nestedHitTargetsForRow,
    isRenderedTimingSpan,
    runTimeline,
    isApproximateSpan,
    shouldShowOpenRolloutAction,
  } from "../lib/timing.js";
  import { fmtDate } from "../lib/format.js";
  import ChartZoomButtons from "./ChartZoomButtons.svelte";
  import TimeAxis from "./TimeAxis.svelte";

  let {
    timings = null,
    timelineKey = "",
    downloadName = "substep_timing.json",
    onOpenRollout = null,
    rolloutIds = [],
    runOrigin = null,
    asyncOverride = null,
    showOpenRollout = true,
    trainingType = null,
    attemptMarkers = [],
    // Wall-clock window `{ start, end }` (epoch seconds) to show; null shows
    // the whole timeline. With `onChangeTimeRange` the window is controlled
    // by the parent and every brush / zoom-button change is reported back
    // through it; without it the timeline keeps its own.
    timeRange = null,
    onChangeTimeRange = null,
  } = $props();

  const MAX_ZOOM = 128;

  const ROW_HEIGHT_PX = 15;
  const DETAIL_ROW_HEIGHT_PX = 22;
  const ROW_GAP_PX = 4;
  const HEADER_PX = 0;
  const GROUP_GAP_PX = 12;
  const STEP_GAP_PX = 8;
  const BAR_GAP_PX = 1;
  const ATTEMPT_STRIP_PX = 14;
  let showDetails = $state(false);
  let localRange = $state(null);
  let controlled = $derived(typeof onChangeTimeRange === "function");
  let activeRange = $derived(controlled ? timeRange : localRange);
  let viewportWidth = $state(0);
  // Evaluate once to derive unmeasured gaps, then again for pixel-aware rendering.
  // Parent-bounded minimum widths leave both passes with the same overall span.
  let baseTimeline = $derived(runTimeline(timings, asyncOverride));
  // The slice of the compressed axis `[0, span]` that fills the viewport. A
  // window narrower than the zoom cap is widened around its centre so the
  // track always fills the viewport.
  let window_ = $derived.by(() => {
    const span = baseTimeline.span;
    if (!span || baseTimeline.runStart == null || !activeRange) return [0, span];
    const clamp = (o) => Math.min(span, Math.max(0, o));
    let o0 = clamp(baseTimeline.mapOffset(Number(activeRange.start) - baseTimeline.runStart));
    let o1 = clamp(baseTimeline.mapOffset(Number(activeRange.end) - baseTimeline.runStart));
    if (!(o1 > o0)) return [o0, o1];
    const minWidth = span / MAX_ZOOM;
    if (o1 - o0 < minWidth) {
      o0 = clamp((o0 + o1) / 2 - minWidth / 2);
      o1 = clamp(o0 + minWidth);
      o0 = clamp(o1 - minWidth);
    }
    return [o0, o1];
  });
  let outOfRange = $derived(window_[1] - window_[0] <= 0);
  let zoom = $derived(outOfRange ? 1 : baseTimeline.span / (window_[1] - window_[0]));
  let trackShift = $derived(outOfRange ? 0 : (window_[0] / baseTimeline.span) * zoom * 100);
  let zoomed = $derived(!outOfRange && zoom > 1.001);
  let pixelsPerSecond = $derived(
    viewportWidth > 0 ? (viewportWidth * zoom) / baseTimeline.span : 0,
  );
  let timeline = $derived(
    runTimeline(timings, asyncOverride, pixelsPerSecond),
  );
  let timingStale = $derived(timings?.metadata?.timing_stale === true);
  let intervalOrigin = $derived(runOrigin ?? timeline.runStart);
  let rowHeight = $derived(showDetails ? DETAIL_ROW_HEIGHT_PX : ROW_HEIGHT_PX);
  let groups = $derived(
    timeline.groups.map((group) => ({
      ...group,
      height: HEADER_PX + group.rows.length * (rowHeight + ROW_GAP_PX),
    })),
  );
  $effect(() => {
    timelineKey;
    pinned = false;
    tip = null;
    laneTip = null;
  });
  $effect(() => {
    const spanKeys = new Set(
      timeline.groups.flatMap((group) =>
        group.rows.flatMap((row) => row.sortedSpans.map((span) => span.key)),
      ),
    );
    if (pinned && (!tip?.bar?.key || !spanKeys.has(tip.bar.key))) {
      clearPin();
    }
  });
  let visibleGroups = $derived(groups);
  let trackHeight = $derived(
    visibleGroups.reduce((total, group) => total + group.height + GROUP_GAP_PX, 0),
  );

  const pct = (seconds) => (seconds / timeline.span) * 100;

  let breaks = $derived(timeline.breaks || []);
  let breakLabelLayouts = $derived.by(() => {
    const widths = breaks.map(
      (gap) => `${fmtSecs(gap.hidden)} unmeasured`.length * 5.4,
    );
    return breakLabelLayout(
      breaks,
      widths,
      timeline.span,
      pixelsPerSecond,
    );
  });

  let markers = $derived(
    (attemptMarkers || [])
      .map((marker) => ({
        label: marker.label,
        offset: timeline.mapOffset(Number(marker.at) - timeline.runStart),
      }))
      .filter(
        (marker) =>
          Number.isFinite(marker.offset) &&
          marker.offset > 0 &&
          marker.offset < timeline.span,
      ),
  );

  let nestedHitTargets = $derived.by(() => {
    const targets = new WeakMap();
    for (const group of groups) {
      for (const row of group.rows) {
        targets.set(row, nestedHitTargetsForRow(row, pixelsPerSecond));
      }
    }
    return targets;
  });

  function changeRange(next) {
    if (controlled) onChangeTimeRange(next);
    else localRange = next;
  }

  // Wall-clock bounds of the visible window; ticks are placed through the
  // gap compression so they line up with the bars above them.
  let axisRange = $derived.by(() => {
    if (outOfRange || !baseTimeline.span || baseTimeline.runStart == null) return null;
    const [w0, w1] = window_;
    return {
      start: baseTimeline.runStart + baseTimeline.unmapOffset(w0),
      end: baseTimeline.runStart + baseTimeline.unmapOffset(w1),
    };
  });

  function axisFraction(t) {
    const [w0, w1] = window_;
    return (baseTimeline.mapOffset(t - baseTimeline.runStart) - w0) / (w1 - w0);
  }

  // Brush output arrives as fractions of the viewport; walk them back through
  // the visible window and the gap compression to wall-clock seconds.
  function handleBrush([f0, f1]) {
    const span = baseTimeline.span;
    if (!span || baseTimeline.runStart == null || outOfRange) return;
    const [w0, w1] = window_;
    const width = w1 - w0;
    let r0 = Math.max(0, w0 + f0 * width);
    let r1 = Math.min(span, w0 + f1 * width);
    if (!(r1 > r0)) return;
    const minWidth = span / MAX_ZOOM;
    if (r1 - r0 < minWidth) {
      const mid = (r0 + r1) / 2;
      r0 = Math.max(0, mid - minWidth / 2);
      r1 = Math.min(span, r0 + minWidth);
    }
    if (r0 <= 0 && r1 >= span) {
      changeRange(null);
      return;
    }
    changeRange({
      start: baseTimeline.runStart + baseTimeline.unmapOffset(r0),
      end: baseTimeline.runStart + baseTimeline.unmapOffset(r1),
      live: false,
    });
  }

  function downloadJson() {
    const blob = new Blob([JSON.stringify(timings || {}, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = downloadName;
    a.click();
    URL.revokeObjectURL(url);
  }

  let tip = $state(null);
  let laneTip = $state(null);
  let groupedTipChildren = $derived(
    tip?.bar.children
      ? groupTooltipChildren(tip.bar.children, tip.bar.aggregateStats, trainingType)
      : [],
  );
  let pinned = $state(false);
  let hideTimer = null;
  onDestroy(clearHideTimer);

  const isActive = (bar) => tip && tip.bar.key === bar.key;

  function clearHideTimer() {
    if (hideTimer !== null) {
      window.clearTimeout(hideTimer);
      hideTimer = null;
    }
  }

  function scheduleHide(target = "bar") {
    clearHideTimer();
    if (!pinned) {
      hideTimer = window.setTimeout(() => {
        if (target === "lane") laneTip = null;
        else tip = null;
      }, 180);
    }
  }

  function showTip(e, bar, role = null) {
    clearHideTimer();
    if (pinned) return;
    laneTip = null;
    tip = { x: e.clientX, y: e.clientY, bar, role };
  }

  function moveTip(e) {
    clearHideTimer();
    if (pinned || !tip) return;
    tip = { ...tip, x: e.clientX, y: e.clientY };
  }

  function hideTip() {
    scheduleHide();
  }

  function showLaneTip(e, row) {
    clearHideTimer();
    if (pinned || !roleDescriptions[row.role]) return;
    tip = null;
    laneTip = { x: e.clientX, y: e.clientY, role: row.role };
  }

  function moveLaneTip(e) {
    clearHideTimer();
    if (pinned || !laneTip) return;
    laneTip = { ...laneTip, x: e.clientX, y: e.clientY };
  }

  function hideLaneTip() {
    scheduleHide("lane");
  }

  function pinTip(e, bar, role = null) {
    e.stopPropagation();
    if (pinned && isActive(bar)) {
      pinned = false;
      tip = null;
      return;
    }
    pinned = true;
    tip = { x: e.clientX, y: e.clientY, bar, role };
  }

  function clearPin() {
    if (!pinned) return;
    pinned = false;
    tip = null;
  }

  function tipTitle(bar) {
    const name = labelFor(bar.name, bar.rolloutId, trainingType);
    const title = bar.ordinal ? `${name} ${bar.ordinal}` : name;
    return isApproximateSpan(bar) ? `${title}*` : title;
  }

  function tipRole(role) {
    return role ? role[0].toUpperCase() + role.slice(1) : null;
  }

  const roleDescriptions = {
    driver: "Runs the training loop.",
    rollout: "Inference engines sampling from the current policy.",
    actor: "Policy being trained.",
    critic: "Value model.",
  };

  function nestedChild(bar, name) {
    for (const child of bar.children || []) {
      if (child.name === name) return child;
      const nested = nestedChild(child, name);
      if (nested) return nested;
    }
    return null;
  }

  function generationStats(bar) {
    if (bar.aggregateStats?.sample_generation) {
      return bar.aggregateStats.sample_generation;
    }
    return bar.name === "generate_samples"
      ? nestedChild(bar, "sample_generation")
      : null;
  }

  // Per-sample phases run many invocations at once, so their busy time is the sum
  // over concurrent samples and routinely exceeds the wall span they occupy.
  function concurrencyStat(bar) {
    if (!bar || bar.count <= 1) return null;
    const busy = Number(bar.total);
    const wall = Number(bar.duration);
    if (!Number.isFinite(busy) || !Number.isFinite(wall)) return null;
    if (busy <= wall * 1.05) return null;
    return `${fmtSecs(busy)} busy across ${bar.count} concurrent runs · ${fmtSecs(wall)} wall clock`;
  }

  // Sample generation runs many instrumented invocations at once; report the
  // wall span occupied by those invocations rather than the busy sum.
  function samplingStat(bar) {
    const stats = generationStats(bar);
    if (!stats || !(stats.count > 1)) return null;
    const wall = Number(stats.end) - Number(stats.start);
    if (!Number.isFinite(wall) || wall <= 0) return null;
    return `${stats.count} generation invocations · ${fmtSecs(wall)}`;
  }

  function clockSpan(bar) {
    const start = Number(bar?.clockStart);
    const end = Number(bar?.clockEnd);
    if (!Number.isFinite(start) || !Number.isFinite(end)) return null;
    return `${fmtDate(start)} → ${fmtDate(end)}`;
  }

  function rolloutIdForBar(bar) {
    const id = Number(bar.rolloutId);
    return Number.isInteger(id) && id >= 0 ? id : null;
  }

  function hasVisibleChildren(bar) {
    return bar.children?.some(isRenderedTimingSpan) ?? false;
  }

  function isExpandedParent(bar) {
    return showDetails && bar.depth === 0 && hasVisibleChildren(bar);
  }

  function displaySpans(row) {
    return row.sortedSpans
      .filter(
        (bar) =>
          isRenderedTimingSpan(bar) &&
          (showDetails || bar.depth === 0),
      )
  }

  function visualInset(row, bar) {
    return row.insetKeys.has(bar.key) ? BAR_GAP_PX : 0;
  }

  function childrenByRole(bar) {
    const groups = new Map();
    for (const child of visibleChildren(bar)) {
      const role = child.role || "unknown";
      const group = groups.get(role) || [];
      group.push(child);
      groups.set(role, group);
    }
    return [...groups.entries()];
  }

  function visibleChildren(bar) {
    return showDetails
      ? (bar.children || []).filter(isRenderedTimingSpan)
      : [];
  }

  function shellLeft(row, bar) {
    if (bar.insideStart == null) {
      return `calc(${pct(bar.renderOffset ?? bar.offset)}% + ${visualInset(row, bar)}px)`;
    }
    const insideStart = bar.insideRenderStart ?? bar.insideStart;
    const insideEnd = bar.insideRenderEnd ?? bar.insideEnd;
    const startValue = bar.renderStart ?? bar.start;
    const duration = Math.max(insideEnd - insideStart, 0);
    const start = duration
      ? Math.min(1, Math.max(0, (startValue - insideStart) / duration))
      : 0;
    return `${start * 100}%`;
  }

  function shellWidth(bar) {
    if (bar.insideStart == null) {
      return `${Math.max(pct(bar.renderDuration ?? bar.duration), 0.01)}%`;
    }
    const insideStart = bar.insideRenderStart ?? bar.insideStart;
    const insideEnd = bar.insideRenderEnd ?? bar.insideEnd;
    const startValue = bar.renderStart ?? bar.start;
    const endValue = bar.renderEnd ?? bar.end;
    const duration = Math.max(insideEnd - insideStart, 0);
    const end = duration
      ? Math.min(1, Math.max(0, (endValue - insideStart) / duration))
      : 0;
    const start = duration
      ? Math.min(1, Math.max(0, (startValue - insideStart) / duration))
      : 0;
    return `${Math.max(0, end - start) * 100}%`;
  }

</script>

<svelte:window onclick={clearPin} />

<div class="run-timeline">
  {#if timingStale}
    <div class="timing-stale">Timing data may be out of date.</div>
  {/if}
  {#if !groups.length}
    <div class="empty">No substep timing recorded for these rollouts yet.</div>
  {:else}
    <div class="toolbar">
      <div class="legend">
        {#each timeline.categories as key (key)}
          <span class="legend-item">
            <span
              class="swatch"
              class:idle-swatch={key === "idle"}
              style:background={key === "idle" ? "transparent" : CATEGORIES[key].color}
            ></span>
            {categoryLabelFor(key, trainingType)}
          </span>
        {/each}
      </div>
      <div class="controls">
        <ChartZoomButtons
          onChangeDomainX={handleBrush}
          canZoomIn={!outOfRange && zoom < MAX_ZOOM}
          canZoomOut={zoomed}
        />
        {#if !controlled && zoomed}
          <button
            class="dl-btn"
            onclick={() => changeRange(null)}
            title="Show the whole timeline"
          >
            <ZoomOut size={13} />
            {zoom >= 10 ? Math.round(zoom) : zoom.toFixed(1).replace(/\.0$/, "")}× · reset
          </button>
        {/if}
        <button class="dl-btn" onclick={downloadJson} title="Download timing as JSON">
          <Download size={13} />
          Download JSON
        </button>
        <button
          class="dl-btn detail-btn"
          onclick={() => {
            showDetails = !showDetails;
          }}
          aria-pressed={showDetails}
          title={showDetails ? "Hide detailed view" : "Show detailed view"}
        >
          {showDetails ? "Hide detailed view" : "Show detailed view"}
        </button>
      </div>
    </div>

    <div class="chart">
      <div
        class="gutter"
        style:padding-top={`${ATTEMPT_STRIP_PX + rowHeight + STEP_GAP_PX}px`}
      >
        {#each visibleGroups as group (group.key)}
          <div style:margin-bottom={`${GROUP_GAP_PX}px`}>
            {#each group.rows as row, index (index)}
              <div
                class="gutter-row"
                class:lane={true}
                style:height={`${rowHeight}px`}
                style:margin-bottom={`${ROW_GAP_PX}px`}
                role="button"
                tabindex="0"
                onmouseenter={(e) => showLaneTip(e, row)}
                onmousemove={moveLaneTip}
                onmouseleave={hideLaneTip}
              >
                {row.label}{#if row.unaligned}<span class="lane-note-mark">*</span>{/if}
              </div>
            {/each}
          </div>
        {/each}
      </div>

      <div
        class="viewport"
        bind:clientWidth={viewportWidth}
        use:brushZoom={{ onChangeDomainX: handleBrush, enabled: !outOfRange }}
      >
        {#if outOfRange}
          <div class="empty">No substep timing in the selected range.</div>
        {/if}
        <div
          class="track"
          class:hidden-track={outOfRange}
          style:width={`${zoom * 100}%`}
          style:margin-left={`-${trackShift}%`}
        >
          <div class="steps" style:height={`${rowHeight}px`}>
            {#each timeline.steps as step (step.id)}
              <div
                class="step"
                style:left={`${pct(step.renderOffset ?? step.offset)}%`}
                style:width={`${Math.max(pct(step.renderDuration ?? step.duration), 0.05)}%`}
                title={`Step ${step.number}: ${fmtSecs(step.duration)} wall clock, ${fmtSecs(step.work)} work, ${fmtSecs(step.idle)} measured idle`}
              >
                <span class="step-text"
                  >Step {step.number} · {fmtSecs(step.duration)}</span
                >
              </div>
            {/each}
          </div>

          <div class="groups" style:height={`${trackHeight}px`}>
            {#each visibleGroups as group (group.key)}
              <div
                class="group"
                style:height={`${group.height}px`}
                style:margin-bottom={`${GROUP_GAP_PX}px`}
              >
                {#each group.rows as row, index (index)}
                  <div
                    class="row"
                    style:top={`${HEADER_PX + index * (rowHeight + ROW_GAP_PX)}px`}
                    style:height={`${rowHeight}px`}
                  >
                    {#snippet renderBar(bar, row, roleLine = false)}
                      <div
                        class="bar-shell"
                        class:nested-shell={bar.depth > 0}
                        style:left={shellLeft(row, bar)}
                        style:width={shellWidth(bar)}
                      >
                        <button
                          class="bar"
                          data-bar-key={bar.key}
                          class:idle={bar.kind === "idle"}
                          class:detail-idle={showDetails && bar.kind === "idle"}
                          class:sampled={bar.kind === "sampled"}
                          class:nested-bar={showDetails && bar.depth > 0}
                          class:outlined={isExpandedParent(bar)}
                          class:train-parent={
                            isExpandedParent(bar) &&
                            bar.name === "train_models"
                          }
                          class:expanded-parent={isExpandedParent(bar)}
                          class:active={pinned && isActive(bar)}
                          aria-label={`${labelFor(bar.name, bar.rolloutId, trainingType)} ${fmtSecs(bar.duration)}`}
                          style:left="0"
                          style:width="100%"
                          style:--bar-color={
                            isExpandedParent(bar) &&
                            bar.name === "train_models"
                              ? TRAIN_OUTLINE_COLOR
                              : colorFor(bar.name)
                          }
                          style:background={
                            bar.kind === "work" && !isExpandedParent(bar)
                              ? colorFor(bar.name)
                              : undefined
                          }
                          style:z-index={
                            bar.depth > 0
                              ? nestedHitTargets.get(row)?.get(bar.key)?.zIndex
                              : undefined
                          }
                          style:--hit-left={
                            bar.depth > 0
                              ? nestedHitTargets.get(row)?.get(bar.key)?.left
                              : undefined
                          }
                          style:--hit-right={
                            bar.depth > 0
                              ? nestedHitTargets.get(row)?.get(bar.key)?.right
                              : undefined
                          }
                          style:border-color={
                            isExpandedParent(bar)
                              ? bar.name === "train_models"
                                ? TRAIN_OUTLINE_COLOR
                                : colorFor(bar.name)
                              : undefined
                          }
                          onmouseenter={(e) => showTip(e, bar, roleLine ? bar.role : null)}
                          onmousemove={moveTip}
                          onmouseleave={hideTip}
                          onclick={(e) => pinTip(e, bar, roleLine ? bar.role : null)}
                        >
                        </button>
                        {#if bar.depth > 0}
                          <button
                            type="button"
                            class="bar-hit-target"
                            data-bar-key={bar.key}
                            aria-label={`${labelFor(bar.name, bar.rolloutId, trainingType)} ${fmtSecs(bar.duration)}`}
                            style:z-index={nestedHitTargets.get(row)?.get(bar.key)?.zIndex}
                            style:--hit-left={nestedHitTargets.get(row)?.get(bar.key)?.left}
                            style:--hit-right={nestedHitTargets.get(row)?.get(bar.key)?.right}
                            onmouseenter={(e) => showTip(e, bar, roleLine ? bar.role : null)}
                            onmousemove={moveTip}
                            onmouseleave={hideTip}
                            onclick={(e) => pinTip(e, bar, roleLine ? bar.role : null)}
                          ></button>
                        {/if}
                        {#if visibleChildren(bar).length}
                          {@const roleGroups = childrenByRole(bar)}
                          <div
                            class="bar-children"
                            class:multi-role={roleGroups.length > 1}
                          >
                            {#if roleGroups.length === 1}
                              {#each roleGroups[0][1] as child (child.key)}
                                {@render renderBar(child, row)}
                              {/each}
                            {:else}
                              {#each roleGroups as [role, children], roleIndex (role)}
                                <div
                                  class="role-line"
                                  style:top={`${(roleIndex * 100) / roleGroups.length}%`}
                                  style:height={`${100 / roleGroups.length}%`}
                                  data-role={role}
                                >
                                  {#each children as child (child.key)}
                                    {@render renderBar(child, row, true)}
                                  {/each}
                                </div>
                              {/each}
                            {/if}
                          </div>
                        {/if}
                      </div>
                    {/snippet}
                    {#each displaySpans(row).filter((bar) => bar.depth === 0) as bar (bar.key)}
                      {@render renderBar(bar, row)}
                    {/each}
                  </div>
                {/each}
              </div>
            {/each}
          </div>

          {#each breaks as gap, index}
            {@const labelLayout = breakLabelLayouts[index]}
            <div
              class="timeline-break"
              style:left={`${pct(gap.offset)}%`}
              style:width={`${Math.max(pct(gap.duration), 0.4)}%`}
              title={`${fmtSecs(gap.hidden)} not instrumented, drawn compressed`}
              aria-label={`${fmtSecs(gap.hidden)} not instrumented, drawn compressed`}
            >
              <span
                class="timeline-break-label"
                class:hidden={labelLayout?.hidden}
                class:right-aligned={labelLayout?.rightAligned}
                >{fmtSecs(gap.hidden)} unmeasured</span
              >
            </div>
          {/each}

          {#each markers as marker}
            <div class="attempt-marker" style:left={`${pct(marker.offset)}%`}>
              <span class="attempt-marker-label">{marker.label}</span>
            </div>
          {/each}
        </div>
        {#if axisRange}
          <TimeAxis start={axisRange.start} end={axisRange.end} fractionAt={axisFraction} />
        {/if}
      </div>
    </div>

    {#if timeline.unaligned}
      <div class="lane-note">
        <span class="lane-note-mark">*</span>{APPROXIMATE_LANE_NOTE}
      </div>
    {/if}

  {/if}
</div>

{#if tip}
  <div
    class="tg-tip"
    class:pinned
    role="tooltip"
    style:left={`${tip.x}px`}
    style:top={`${tip.y}px`}
    onmouseenter={clearHideTimer}
    onmouseleave={scheduleHide}
  >
    <div class="tg-tip-main">
      <span class="tg-tip-time">
        {tip.bar.rolloutId == null ? "" : `Step ${tip.bar.rolloutId + 1} · `}{fmtSecs(tip.bar.duration)}
      </span>
    </div>
    <span class="tg-tip-name">
      {#if tipRole(tip.role)}<span class="tg-tip-role">{tipRole(tip.role)} · </span>{/if}{tipTitle(tip.bar)}
    </span>
    {#if clockSpan(tip.bar)}
      <span class="tg-tip-when">{clockSpan(tip.bar)}</span>
    {/if}
    {#if concurrencyStat(tip.bar)}
      <span class="tg-tip-stat">{concurrencyStat(tip.bar)}</span>
    {/if}
    {#if samplingStat(tip.bar)}
      <span class="tg-tip-stat">{samplingStat(tip.bar)}</span>
    {/if}
    {#if generationStats(tip.bar)}
      <span class="tg-tip-stat">
        Average sample generation time: {fmtSecs(generationStats(tip.bar).average)}
      </span>
      <span class="tg-tip-stat">
        Longest sample generation time: {fmtSecs(
          generationStats(tip.bar).longest ?? generationStats(tip.bar).duration,
        )}
      </span>
    {/if}
    {#if groupedTipChildren.length}
      <div class="tg-tip-children">
        {#each groupedTipChildren as child (`${child.role || ""}:${child.name}`)}
          <span class="tg-tip-child">
            <span class="tg-tip-child-line">
              {child.label}
              {#if child.count > 1}
                {" "}×{child.count}
              {/if}
              <span class="tg-tip-child-duration">
                {" "}· {fmtSecs(child.duration)}{child.concurrent
                  ? ` busy · ${fmtSecs(child.wall)} wall`
                  : ""}</span
              >
            </span>
            {#if showDetails && child.count === 1}
              <span class="tg-tip-when">
                {#if child.representative.clockStart != null && child.representative.clockEnd != null}
                  {fmtDate(child.representative.clockStart)} → {fmtDate(child.representative.clockEnd)}
                {:else}
                  {fmtSecs(child.representative.start - intervalOrigin)} → {fmtSecs(child.representative.end - intervalOrigin)}
                {/if}
              </span>
            {/if}
          </span>
        {/each}
      </div>
    {/if}
    {#if shouldShowOpenRolloutAction({ showOpenRollout, onOpenRollout, bar: tip.bar, rolloutIds, rolloutId: rolloutIdForBar(tip.bar) })}
      <button
        class="tg-tip-action"
        onclick={(e) => {
          e.stopPropagation();
          onOpenRollout(rolloutIdForBar(tip.bar));
        }}
      >
        Open in Rollouts →
      </button>
    {/if}
  </div>
{/if}

{#if laneTip}
  <div
    class="tg-tip"
    role="tooltip"
    style:left={`${laneTip.x}px`}
    style:top={`${laneTip.y}px`}
    onmouseenter={clearHideTimer}
    onmouseleave={hideLaneTip}
  >
    <span class="tg-tip-stat">{roleDescriptions[laneTip.role]}</span>
  </div>
{/if}

<style>
  .run-timeline {
    display: flex;
    flex-direction: column;
    gap: 12px;
    font-family: var(--font-sans);
  }

  .empty {
    color: var(--color-c-gray-45, #6e6e6e);
    font-size: 0.85rem;
    padding: 0.5rem 0;
  }

  .toolbar {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: flex-start;
    gap: 12px;
  }

  .lane-note {
    margin-top: 6px;
    font-size: 10px;
    line-height: 1.4;
    color: var(--tg-text-muted, #8a8f98);
  }

  .lane-note-mark {
    color: var(--tg-text-muted, #8a8f98);
    padding-right: 2px;
  }

  .legend {
    flex: 1 1 100%;
    min-width: 0;
    display: flex;
    flex-wrap: wrap;
    gap: 3px 8px;
    font-size: 9px;
    color: var(--muted);
  }

  .legend-item {
    display: inline-flex;
    align-items: center;
    gap: 3px;
    flex-shrink: 0;
  }

  .swatch {
    width: 7px;
    height: 7px;
    border-radius: 2px;
    flex-shrink: 0;
  }

  .idle-swatch {
    height: 2px;
    border-radius: 0;
    border-top: 2px solid var(--color-c-gray-30, #6a6a6a);
  }

  .controls {
    margin-left: auto;
    display: flex;
    align-items: center;
    gap: 8px;
    flex-shrink: 0;
  }

  .dl-btn {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: none;
    border: 1px solid var(--border, #2f2f2f);
    border-radius: 4px;
    color: var(--muted);
    font-size: 11px;
    padding: 3px 8px;
    cursor: pointer;
    font-family: inherit;
  }

  .dl-btn:hover {
    color: var(--text);
    border-color: var(--border-strong, #4a4a4a);
  }

  .chart {
    display: flex;
    align-items: flex-start;
    gap: 10px;
  }

  .gutter {
    flex-shrink: 0;
    width: 108px;
  }

  .gutter-row {
    font-size: 11px;
    line-height: 20px;
    color: var(--muted);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    font-family: var(--font-sans);
  }

  .gutter-row.lane {
    color: var(--text);
    font-weight: 600;
  }

  .viewport {
    position: relative;
    flex: 1;
    min-width: 0;
    overflow: hidden;
    padding-bottom: 10px;
    overscroll-behavior-x: contain;
  }

  .hidden-track {
    visibility: hidden;
  }

  .track {
    position: relative;
    min-width: 100%;
    /* Reserve a strip for attempt-boundary labels so they never sit on top of
       the step headers; the gutter reserves the same strip to stay aligned. */
    padding-top: 14px;
  }

  .steps {
    position: relative;
    margin-bottom: 8px;
  }

  .step {
    position: absolute;
    top: 0;
    bottom: 0;
    border-left: 1px solid var(--border-strong, #4a4a4a);
    background: var(--color-c-gray-05, #171717);
    overflow: visible;
  }

  .step-text {
    display: block;
    padding: 0 6px;
    font-size: 10px;
    line-height: 20px;
    color: var(--muted);
    white-space: nowrap;
    font-variant-numeric: tabular-nums;
  }

  .groups {
    position: relative;
  }

  .timeline-break {
    position: absolute;
    top: 40%;
    bottom: 40%;
    background: var(--color-c-gray-15, #303030);
    opacity: 0.55;
    border-left: 1px solid var(--color-c-gray-20, #424242);
    border-right: 1px solid var(--color-c-gray-20, #424242);
    pointer-events: none;
  }

  .timeline-break-label {
    position: absolute;
    bottom: 0;
    left: 2px;
    font-size: 9px;
    line-height: 12px;
    color: var(--muted);
    white-space: nowrap;
    font-variant-numeric: tabular-nums;
  }

  .timeline-break-label.right-aligned {
    right: 2px;
    left: auto;
  }

  .timeline-break-label.hidden {
    display: none;
  }

  .attempt-marker {
    position: absolute;
    top: 0;
    bottom: 0;
    border-left: 1px dashed var(--color-c-yellow-60, #b8912f);
    pointer-events: none;
  }

  .attempt-marker-label {
    position: absolute;
    top: 0;
    left: 3px;
    font-size: 9px;
    line-height: 12px;
    color: var(--color-c-yellow-60, #b8912f);
    white-space: nowrap;
  }

  .group {
    position: relative;
    border-top: 1px solid var(--border, #2f2f2f);
  }

  .row {
    position: absolute;
    left: 0;
    right: 0;
    pointer-events: none;
  }

  .bar {
    position: absolute;
    top: 0;
    height: 100%;
    display: flex;
    align-items: center;
    min-width: 1px;
    padding: 0;
    border: none;
    border-radius: 1px;
    box-sizing: border-box;
    outline: 1px solid var(--panel, #1a1a1a);
    overflow: hidden;
    cursor: pointer;
    pointer-events: auto;
    background: transparent;
    font-family: inherit;
  }

  .bar-shell {
    position: absolute;
    top: 0;
    height: 100%;
    pointer-events: none;
  }

  .bar-shell > .bar {
    pointer-events: auto;
  }

  .bar-children {
    position: absolute;
    inset: 2px;
    z-index: 3;
    overflow: hidden;
    pointer-events: none;
  }

  .bar-children > .bar-shell {
    pointer-events: auto;
  }

  .role-line {
    position: absolute;
    left: 0;
    right: 0;
    pointer-events: none;
  }

  .role-line > .bar-shell {
    pointer-events: auto;
  }

  .bar-children.multi-role .bar.nested-bar {
    top: 1px;
    height: calc(100% - 2px);
  }

  .bar-children.multi-role .bar-hit-target {
    top: 1px;
    bottom: 1px;
  }

  .bar.outlined {
    background: transparent;
    border: 1px solid var(--bar-color);
    outline: none;
  }

  .bar.train-parent {
    border-width: 2px;
  }

  .bar.expanded-parent {
    z-index: 2;
    pointer-events: auto;
    background: color-mix(in srgb, var(--bar-color) 18%, transparent) !important;
  }

  .bar.nested-bar {
    min-width: 2px;
    top: 4px;
    height: calc(100% - 8px);
    z-index: 3;
    outline: none;
    border-radius: 0;
    overflow: visible;
    pointer-events: none;
  }

  .bar-hit-target {
    position: absolute;
    top: 4px;
    bottom: 4px;
    left: calc(-1 * var(--hit-left, 0px));
    right: calc(-1 * var(--hit-right, 0px));
    padding: 0;
    border: none;
    background: transparent;
    pointer-events: auto;
    cursor: pointer;
  }

  .bar.idle {
    background: linear-gradient(
      var(--color-c-gray-30, #6a6a6a),
      var(--color-c-gray-30, #6a6a6a)
    );
    background-size: 100% 2px;
    background-position: center;
    background-repeat: no-repeat;
  }

  .bar.detail-idle {
    opacity: 0.35;
    z-index: 1;
  }

  .bar.sampled {
    background: transparent;
    border: none;
    background: linear-gradient(var(--bar-color), var(--bar-color));
    background-size: 100% 2px;
    background-position: center;
    background-repeat: no-repeat;
  }

  .bar.active {
    outline: 2px solid var(--color-c-green-80, #6ac355);
    outline-offset: -1px;
  }

  .tg-tip {
    position: fixed;
    z-index: 1000;
    transform: translate(-50%, calc(-100% - 10px));
    pointer-events: auto;
    display: flex;
    flex-direction: column;
    gap: 1px;
    padding: 6px 9px;
    border-radius: 6px;
    background: var(--color-c-gray-02, #0d0d0d);
    border: 1px solid var(--border, #3a3a3a);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);
    font-size: 11px;
    white-space: nowrap;
    font-family: var(--font-sans);
  }

  .tg-tip::after {
    content: "";
    position: absolute;
    left: 0;
    right: 0;
    bottom: -12px;
    height: 12px;
    pointer-events: none;
  }

  .tg-tip.pinned {
    border-color: var(--accent, #60a5fa);
    pointer-events: auto;
  }

  .tg-tip-main {
    color: var(--muted);
    font-family: var(--font-mono);
    font-size: 11px;
    font-variant-numeric: tabular-nums;
  }

  .tg-tip-name {
    color: var(--color-c-gray-100);
    font-weight: 600;
  }

  .tg-tip-stat {
    display: block;
    color: var(--muted);
    font-size: 9px;
    line-height: 12px;
  }

  .timing-stale {
    color: var(--muted);
    font-size: 12px;
  }

  .tg-tip-when {
    color: var(--muted);
    font-family: var(--font-mono);
    font-size: 9px;
    font-variant-numeric: tabular-nums;
  }

  .tg-tip-children {
    display: flex;
    flex-direction: column;
    gap: 2px;
    margin-top: 3px;
    padding-top: 3px;
    border-top: 1px solid var(--border, #3a3a3a);
  }

  .tg-tip-child {
    display: flex;
    flex-direction: column;
    gap: 1px;
  }

  .tg-tip-child-line {
    color: var(--muted);
    font-weight: 600;
    font-family: inherit;
    font-size: inherit;
    font-variant-numeric: tabular-nums;
  }

  .tg-tip-child-duration {
    font-family: var(--font-mono);
    font-weight: 400;
  }

  .tg-tip-action {
    align-self: flex-start;
    margin-top: 4px;
    padding: 2px 0;
    border: none;
    background: none;
    color: var(--accent, #60a5fa);
    font-family: inherit;
    cursor: pointer;
  }

  .tg-tip-action:hover {
    text-decoration: underline;
  }

  .detail-btn {
    background: var(--color-c-gray-12, #262626);
    border-color: var(--color-c-gray-25, #555);
    color: var(--text-bright, #fff);
  }

</style>
