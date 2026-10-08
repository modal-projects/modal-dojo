<script>
  let { trace = [] } = $props();

  let spans = $derived(Array.isArray(trace) ? trace : []);

  // ── Derived data ──────────────────────────────────────────────────────
  let domainMax = $derived.by(() => {
    let max = 0;
    for (const s of spans) {
      const start = Number(s.start) || 0;
      const end = s.end == null ? start : Number(s.end) || start;
      if (end > max) max = end;
    }
    return max || 1;
  });

  // Unique span names in first-seen order.
  let spanNames = $derived([...new Set(spans.map((s) => s.name || ""))]);

  // Deterministic hash-based color per span name, using the design system dataviz palette.
  const PALETTE = [
    "var(--color-c-dataviz-primary-1)",
    "var(--color-c-dataviz-primary-2)",
    "var(--color-c-dataviz-primary-3)",
    "var(--color-c-dataviz-primary-4)",
    "var(--color-c-dataviz-primary-5)",
    "var(--color-c-dataviz-primary-6)",
    "var(--color-c-dataviz-primary-7)",
    "var(--color-c-dataviz-primary-8)",
  ];

  // Resolved hex colors for canvas (CSS vars can't be used in canvas).
  const PALETTE_RESOLVED = [
    "#adeaab",
    "#d9866b",
    "#ffc1f7",
    "#4aa19d",
    "#decb6c",
    "#4fbe5f",
    "#648fe0",
    "#8d324c",
  ];

  let colorMap = $derived.by(() => {
    const map = new Map();
    spanNames.forEach((n, i) => map.set(n, i % PALETTE.length));
    return map;
  });

  function colorFor(name) {
    return PALETTE[colorMap.get(name || "") ?? 0];
  }
  function colorForCanvas(name) {
    return PALETTE_RESOLVED[colorMap.get(name || "") ?? 0];
  }

  // Build lane layout: parent spans at depth 0, children indented.
  // Also compute parent→children mapping for nesting.
  let laneItems = $derived.by(() => {
    const parentNames = new Set(spans.map((s) => s.parent).filter(Boolean));
    const items = [];
    const byParent = new Map();

    for (const s of spans) {
      if (!byParent.has(s.parent)) byParent.set(s.parent, []);
      byParent.get(s.parent).push(s);
    }

    // Top-level spans (no parent, or parent not in this trace).
    const roots = spans.filter(
      (s) => !s.parent || !spans.some((p) => p.name === s.parent),
    );
    const seen = new Set();
    for (const s of roots) {
      items.push({ ...s, depth: 0 });
      seen.add(s);
    }

    // Children: walk by parent name.
    for (const s of spans) {
      if (!seen.has(s) && s.parent) {
        items.push({ ...s, depth: 1 });
        seen.add(s);
      }
    }

    // Anything remaining.
    for (const s of spans) {
      if (!seen.has(s)) {
        items.push({ ...s, depth: 0 });
      }
    }

    return items;
  });

  // ── Canvas rendering ──────────────────────────────────────────────────
  let canvasEl = $state(null);
  let wrapEl = $state(null);

  // Zoom/pan state.
  let viewStart = $state(0);
  let viewEnd = $state(null);

  let effectiveViewEnd = $derived(viewEnd ?? domainMax);

  const ROW_H = 22;
  const LANE_PAD = 2;
  const BAR_H = 14;
  const BAR_R = 3;
  const LABEL_W = 130;
  const DUR_W = 70;
  const POINT_R = 5;
  const HEADER_H = 24;

  let canvasW = $state(600);
  let canvasH = $derived(HEADER_H + laneItems.length * ROW_H + 8);
  let trackW = $derived(Math.max(canvasW - LABEL_W - DUR_W, 80));

  // Observe container width for responsive canvas.
  let resizeObs = $state(null);

  $effect(() => {
    if (!wrapEl) return;
    const obs = new ResizeObserver((entries) => {
      const w = entries[0]?.contentRect?.width;
      if (w && w > 0) canvasW = Math.floor(w);
    });
    obs.observe(wrapEl);
    resizeObs = obs;
    return () => obs.disconnect();
  });

  // DPR-aware canvas sizing.
  $effect(() => {
    if (!canvasEl) return;
    const dpr = window.devicePixelRatio || 1;
    canvasEl.width = canvasW * dpr;
    canvasEl.height = canvasH * dpr;
    canvasEl.style.width = canvasW + "px";
    canvasEl.style.height = canvasH + "px";
    const ctx = canvasEl.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    drawTimeline(ctx);
  });

  function timeToX(t) {
    const range = effectiveViewEnd - viewStart;
    if (range <= 0) return LABEL_W;
    return LABEL_W + ((t - viewStart) / range) * trackW;
  }

  function xToTime(x) {
    const range = effectiveViewEnd - viewStart;
    return viewStart + ((x - LABEL_W) / trackW) * range;
  }

  function fmtDur(s) {
    if (s == null) return "";
    if (s < 0.001) return `${(s * 1e6).toFixed(0)}µs`;
    if (s < 1) return `${(s * 1000).toFixed(1)}ms`;
    return `${s.toFixed(2)}s`;
  }

  function fmtTime(t) {
    if (t < 1) return `${(t * 1000).toFixed(0)}ms`;
    return `${t.toFixed(2)}s`;
  }

  function drawTimeline(ctx) {
    const W = canvasW;
    const H = canvasH;

    ctx.clearRect(0, 0, W, H);

    // Background.
    ctx.fillStyle = "#1c1c1c";
    ctx.fillRect(0, 0, W, H);

    // Time axis ticks.
    const range = effectiveViewEnd - viewStart;
    const tickCount = Math.max(2, Math.min(8, Math.floor(trackW / 80)));
    ctx.fillStyle = "#5d5d5d";
    ctx.font = "10px 'Inter Variable', sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";

    for (let i = 0; i <= tickCount; i++) {
      const t = viewStart + (range * i) / tickCount;
      const x = timeToX(t);
      // Tick line.
      ctx.strokeStyle = "#2f2f2f";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, HEADER_H);
      ctx.lineTo(x, H);
      ctx.stroke();
      // Label.
      ctx.fillStyle = "#5d5d5d";
      ctx.fillText(fmtTime(t), x, 6);
    }

    // Draw cursor line if set.
    if (cursorTime != null) {
      const cx = timeToX(cursorTime);
      if (cx >= LABEL_W && cx <= LABEL_W + trackW) {
        ctx.strokeStyle = "#7fee64";
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(cx, HEADER_H);
        ctx.lineTo(cx, H);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle = "#7fee64";
        ctx.font = "9px 'Inter Variable', sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "bottom";
        ctx.fillText(fmtTime(cursorTime), cx, HEADER_H - 1);
      }
    }

    // Draw each lane.
    for (let i = 0; i < laneItems.length; i++) {
      const item = laneItems[i];
      const y = HEADER_H + i * ROW_H + LANE_PAD;
      const start = Number(item.start) || 0;
      const end = item.end == null ? start : Number(item.end) || start;
      const color = colorForCanvas(item.name);
      const indent = item.depth * 10;

      // Row label.
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, y, LABEL_W, BAR_H);
      ctx.clip();
      ctx.fillStyle = item.depth > 0 ? "#8b8b8b" : "#d1d1d1";
      ctx.font =
        item.depth > 0
          ? "10px 'Inter Variable', sans-serif"
          : "11px 'Inter Variable', sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillText(item.name || "—", 6 + indent, y + BAR_H / 2);
      ctx.restore();

      // Track background.
      ctx.fillStyle = "#272727";
      roundRect(ctx, LABEL_W, y, trackW, BAR_H, 2);
      ctx.fill();

      if (item.end == null) {
        // Instant event: dot.
        const cx = timeToX(start);
        if (cx >= LABEL_W - POINT_R && cx <= LABEL_W + trackW + POINT_R) {
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(cx, y + BAR_H / 2, POINT_R, 0, Math.PI * 2);
          ctx.fill();

          // Diamond outline for better visibility.
          ctx.strokeStyle = "#1c1c1c";
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.arc(cx, y + BAR_H / 2, POINT_R, 0, Math.PI * 2);
          ctx.stroke();
        }
      } else {
        // Duration span: bar.
        const x1 = Math.max(timeToX(start), LABEL_W);
        const x2 = Math.min(timeToX(end), LABEL_W + trackW);
        const barW = Math.max(x2 - x1, 2);
        if (x2 > LABEL_W && x1 < LABEL_W + trackW) {
          ctx.fillStyle = color;
          roundRect(ctx, x1, y, barW, BAR_H, BAR_R);
          ctx.fill();

          // Span name inside bar if it fits.
          if (barW > 50) {
            ctx.save();
            ctx.beginPath();
            ctx.rect(x1, y, barW, BAR_H);
            ctx.clip();
            ctx.fillStyle = "#1c1c1c";
            ctx.font = "10px 'Inter Variable', sans-serif";
            ctx.textAlign = "left";
            ctx.textBaseline = "middle";
            ctx.fillText(item.name || "", x1 + 4, y + BAR_H / 2);
            ctx.restore();
          }
        }
      }

      // Duration label on right.
      ctx.fillStyle = "#8b8b8b";
      ctx.font = "10px 'Inter Variable', sans-serif";
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      const durText =
        item.end == null
          ? `@${fmtTime(start)}`
          : fmtDur(end - start);
      ctx.fillText(durText, canvasW - 4, y + BAR_H / 2);
    }
  }

  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
  }

  // ── Interaction: zoom, pan, hover, click ──────────────────────────────
  let hoveredItem = $state(null);
  let tooltipX = $state(0);
  let tooltipY = $state(0);
  let cursorTime = $state(null);
  let isPanning = $state(false);
  let panStartX = $state(0);
  let panStartViewStart = $state(0);
  let panStartViewEnd = $state(0);

  function hitTest(mx, my) {
    for (let i = 0; i < laneItems.length; i++) {
      const item = laneItems[i];
      const y = HEADER_H + i * ROW_H + LANE_PAD;
      if (my < y || my > y + BAR_H) continue;

      const start = Number(item.start) || 0;
      const end = item.end == null ? start : Number(item.end) || start;

      if (item.end == null) {
        const cx = timeToX(start);
        if (Math.abs(mx - cx) <= POINT_R + 2) return { item, index: i };
      } else {
        const x1 = Math.max(timeToX(start), LABEL_W);
        const x2 = Math.min(timeToX(end), LABEL_W + trackW);
        if (mx >= x1 && mx <= x2) return { item, index: i };
      }
    }
    return null;
  }

  function onMouseMove(e) {
    if (isPanning) {
      const dx = e.clientX - panStartX;
      const timePerPx =
        (panStartViewEnd - panStartViewStart) / trackW;
      const dt = -dx * timePerPx;
      const newStart = Math.max(0, panStartViewStart + dt);
      const range = panStartViewEnd - panStartViewStart;
      viewStart = newStart;
      viewEnd = newStart + range;
      return;
    }

    const rect = canvasEl?.getBoundingClientRect();
    if (!rect) return;
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const hit = hitTest(mx, my);
    hoveredItem = hit?.item ?? null;
    tooltipX = e.clientX;
    tooltipY = e.clientY;
  }

  function onMouseDown(e) {
    if (e.button !== 0) return;
    const rect = canvasEl?.getBoundingClientRect();
    if (!rect) return;
    const mx = e.clientX - rect.left;
    if (mx < LABEL_W || mx > LABEL_W + trackW) return;
    isPanning = true;
    panStartX = e.clientX;
    panStartViewStart = viewStart;
    panStartViewEnd = effectiveViewEnd;
    e.preventDefault();
  }

  function onMouseUp(e) {
    if (isPanning) {
      const dx = Math.abs(e.clientX - panStartX);
      if (dx < 3) {
        // Click (not drag) — set cursor.
        const rect = canvasEl?.getBoundingClientRect();
        if (rect) {
          const mx = e.clientX - rect.left;
          cursorTime = xToTime(mx);
        }
      }
      isPanning = false;
    }
  }

  function onMouseLeave() {
    hoveredItem = null;
    isPanning = false;
  }

  function resetZoom() {
    viewStart = 0;
    viewEnd = null;
    cursorTime = null;
  }

  function nudgeZoom(factor) {
    const range = effectiveViewEnd - viewStart;
    const mid = viewStart + range / 2;
    const newRange = Math.min(
      domainMax * 1.1,
      Math.max(range * factor, 0.001),
    );
    const newStart = Math.max(0, mid - newRange / 2);
    viewStart = newStart;
    viewEnd = newStart + newRange;
  }

  // Tooltip content builder.
  function tooltipContent(item) {
    if (!item) return "";
    const lines = [];
    lines.push(item.name || "span");
    const start = Number(item.start) || 0;
    if (item.end == null) {
      lines.push(`instant @ ${fmtTime(start)}`);
    } else {
      const end = Number(item.end) || start;
      lines.push(
        `${fmtTime(start)} → ${fmtTime(end)}  (${fmtDur(end - start)})`,
      );
    }
    if (item.parent) lines.push(`parent: ${item.parent}`);
    const attrs = item.attributes || {};
    for (const [k, v] of Object.entries(attrs)) {
      lines.push(`${k}: ${v}`);
    }
    return lines.join("\n");
  }
</script>

{#if spans.length}
  <div class="bg-[var(--color-c-gray-08,#1c1c1c)] rounded-[6px] p-0 overflow-hidden">
    <!-- Legend -->
    <div class="flex flex-wrap gap-[10px] items-center p-[8px_10px_4px] [border-bottom:1px_solid_var(--color-c-gray-10,#2f2f2f)]">
      {#each spanNames as name, i (name)}
        <span class="inline-flex items-center gap-[5px] text-[11px] [color:var(--text,#d1d1d1)]">
          <span class="w-[12px] h-[8px] rounded-[2px] inline-block [flex-shrink:0]" style:background={colorFor(name)}></span>
          {name || "—"}
        </span>
      {/each}
      <span class="ml-auto text-[10px] [color:var(--muted,#a3a3a3)] [font-variant-numeric:tabular-nums]">{spans.length} spans · {fmtDur(domainMax)}</span>
      <button class="text-[10px] [color:var(--text,#d1d1d1)] [background:none] [border:1px_solid_var(--color-c-gray-15,#3b3b3b)] rounded-[4px] min-h-[28px] min-w-[28px] p-[1px_6px] cursor-pointer [font-family:inherit] hover:[background:var(--color-c-gray-10,#2f2f2f)]" onclick={() => nudgeZoom(1 / 1.15)} aria-label="Zoom in" title="Zoom in">+</button>
      <button class="text-[10px] [color:var(--text,#d1d1d1)] [background:none] [border:1px_solid_var(--color-c-gray-15,#3b3b3b)] rounded-[4px] min-h-[28px] min-w-[28px] p-[1px_6px] cursor-pointer [font-family:inherit] hover:[background:var(--color-c-gray-10,#2f2f2f)]" onclick={() => nudgeZoom(1.15)} aria-label="Zoom out" title="Zoom out">−</button>
      {#if viewEnd != null}
        <button class="text-[10px] [color:var(--accent,#7fee64)] [background:none] [border:1px_solid_var(--color-c-gray-15,#3b3b3b)] rounded-[4px] min-h-[28px] p-[1px_6px] cursor-pointer [font-family:inherit] hover:[background:var(--color-c-gray-10,#2f2f2f)]" onclick={resetZoom}>reset</button>
      {/if}
    </div>

    <!-- Canvas timeline -->
    <div class="relative cursor-grab select-none active:cursor-grabbing" bind:this={wrapEl}>
      <canvas
        class="block w-full"
        bind:this={canvasEl}
        onmousemove={onMouseMove}
        onmousedown={onMouseDown}
        onmouseup={onMouseUp}
        onmouseleave={onMouseLeave}
      ></canvas>
    </div>

    <!-- Tooltip -->
    {#if hoveredItem}
      <div
        class="tl-tooltip"
        class:visible={!!hoveredItem}
        style:left={`${tooltipX + 12}px`}
        style:top={`${tooltipY - 8}px`}
      >
        {tooltipContent(hoveredItem)}
      </div>
    {/if}

    <!-- Footer hint -->
    <div class="p-[4px_10px_6px] text-[10px] [color:var(--muted-strong,#747474)]">
      drag to pan · +/− to zoom · click to set cursor
    </div>
  </div>
{:else}
  <div class="text-[12px] [color:var(--muted,#a3a3a3)] p-[4px_0]">No trace recorded for this sample.</div>
{/if}
