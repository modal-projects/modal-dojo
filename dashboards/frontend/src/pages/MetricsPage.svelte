<script>
  // Metrics compare page: pick several training runs and overlay their
  // wandb/trackio scalars W&B-style. Panels are the union of every selected
  // run's metric keys; each panel draws one line per run that logged it.
  import { Eye, EyeOff, Plus, X } from "lucide-svelte";
  import AddRunsModal from "../components/AddRunsModal.svelte";
  import ChartSkeleton from "../components/ChartSkeleton.svelte";
  import CompareLineChart from "../components/CompareLineChart.svelte";
  import GroupSection from "../components/GroupSection.svelte";
  import { fetchRunMetrics } from "../lib/api.js";
  import { formatMetricValue, groupMetricKeys, metricAxisLabel, runColor } from "../lib/metricSeries.js";
  import { toggleInSet } from "../lib/set.js";

  let {
    runIds = [],
    // The training list's rows and filters, shared with the Add-run modal.
    runs = [],
    filterProps = {},
    search: runSearch = $bindable(""),
    matchingRunCount = 0,
    hasMoreRuns = false,
    runsLoading = false,
    getStatus,
    onLoadMore = () => {},
    isRunning = () => false,
    onChangeRunIds = () => {},
    onOpenRun = () => {},
  } = $props();

  const POLL_MS = 5000;

  // runId → { payload, error }
  let results = $state({});
  let search = $state("");
  let collapsed = $state(new Set());
  let hiddenRuns = $state(new Set());
  let stepDomains = $state({});
  let smoothing = $state(false);
  let addOpen = $state(false);

  let runKey = $derived(runIds.join("\n"));
  let anyLive = $derived(
    runIds.some((id) => isRunning(id) || results[id]?.payload?.stale),
  );

  async function loadOne(id, signal) {
    try {
      const payload = await fetchRunMetrics(id, { signal });
      if (signal.aborted) return;
      results = { ...results, [id]: { payload, error: "" } };
    } catch (err) {
      if (signal.aborted) return;
      const error = err instanceof Error ? err.message : String(err);
      results = { ...results, [id]: { payload: results[id]?.payload ?? null, error } };
    }
  }

  $effect(() => {
    const ids = runKey ? runKey.split("\n") : [];
    const poll = anyLive;
    if (!ids.length) return;
    const controller = new AbortController();
    const loadAll = () => Promise.all(ids.map((id) => loadOne(id, controller.signal)));
    loadAll();
    const interval = poll ? setInterval(loadAll, POLL_MS) : null;
    return () => {
      controller.abort();
      if (interval) clearInterval(interval);
    };
  });

  let runMeta = $derived(
    runIds.map((id, index) => ({ id, label: id, color: runColor(index) })),
  );
  let visibleRuns = $derived(runMeta.filter((run) => !hiddenRuns.has(run.id)));
  let loadingCount = $derived(runIds.filter((id) => !results[id]).length);

  // Union of keys across visible runs; the first run that names a key's
  // step axis decides it (runs of one framework agree).
  let union = $derived.by(() => {
    const stepKeys = {};
    const keys = new Set();
    for (const run of visibleRuns) {
      const payload = results[run.id]?.payload;
      if (!payload) continue;
      for (const key of Object.keys(payload.series ?? {})) {
        keys.add(key);
        if (!(key in stepKeys) && payload.step_keys?.[key]) stepKeys[key] = payload.step_keys[key];
      }
    }
    return { keys: [...keys], stepKeys };
  });
  let groups = $derived(groupMetricKeys(union.keys, search));

  function seriesFor(key) {
    const out = [];
    for (const run of visibleRuns) {
      const raw = results[run.id]?.payload?.series?.[key];
      if (!raw?.length) continue;
      const points = raw
        .map(([x, y]) => ({ x: Number(x), y: Number(y) }))
        .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y))
        .sort((a, b) => a.x - b.x);
      out.push({ ...run, points });
    }
    return out;
  }

  function onDomainChange(axis, domain) {
    stepDomains = {
      ...stepDomains,
      [axis]: domain && domain[1] > domain[0] ? [Math.floor(domain[0]), Math.ceil(domain[1])] : null,
    };
  }

  function removeRun(id) {
    hiddenRuns = new Set([...hiddenRuns].filter((runId) => runId !== id));
    onChangeRunIds(runIds.filter((runId) => runId !== id));
  }

  function addRuns(ids) {
    const fresh = ids.filter((id) => !runIds.includes(id));
    if (fresh.length) onChangeRunIds([...runIds, ...fresh]);
  }
</script>

<section class="flex flex-col gap-[16px] p-[0_24px_24px] max-[900px]:p-[0_16px_24px] min-w-0">
  <div class="flex flex-wrap items-center gap-[8px]">
    {#each runMeta as run (run.id)}
      <span class="run-chip" class:run-chip-hidden={hiddenRuns.has(run.id)}>
        <span class="inline-block w-[10px] h-[10px] rounded-[3px] shrink-0" style:background={run.color}></span>
        <a
          class="truncate text-inherit no-underline hover:text-(--text-bright) hover:underline"
          href={`/training/${encodeURIComponent(run.id)}`}
          title={run.id}
          onclick={(event) => {
            if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey) return;
            event.preventDefault();
            onOpenRun(run.id);
          }}>{run.label}</a
        >
        {#if results[run.id]?.error}
          <span class="text-(--red) text-[11px]" title={results[run.id].error}>error</span>
        {/if}
        <button
          type="button"
          title={hiddenRuns.has(run.id) ? "Show run" : "Hide run"}
          aria-label={`${hiddenRuns.has(run.id) ? "Show" : "Hide"} ${run.id}`}
          onclick={() => (hiddenRuns = toggleInSet(hiddenRuns, run.id))}
        >
          {#if hiddenRuns.has(run.id)}<EyeOff size={12} />{:else}<Eye size={12} />{/if}
        </button>
        <button type="button" title="Remove run" aria-label={`Remove ${run.id}`} onclick={() => removeRun(run.id)}>
          <X size={12} />
        </button>
      </span>
    {/each}

    <button type="button" class="log-button inline-flex items-center gap-[4px]" onclick={() => (addOpen = true)}>
      <Plus size={12} strokeWidth={2.1} />
      <span>Add runs</span>
    </button>
  </div>

  <AddRunsModal
    open={addOpen}
    {filterProps}
    bind:search={runSearch}
    {runs}
    {matchingRunCount}
    {hasMoreRuns}
    loading={runsLoading}
    {getStatus}
    existingRunIds={runIds}
    {onLoadMore}
    onAdd={addRuns}
    onclose={() => (addOpen = false)}
  />

  {#if !runIds.length}
    <div class="detail-empty">
      <div class="text-(--text-bright) mb-[6px]">No runs selected.</div>
      <div class="max-w-[64ch] leading-[1.5]">
        Use <strong>Add runs</strong>, or tick runs on the Training runs page and press <strong>Compare metrics</strong>.
      </div>
    </div>
  {:else if loadingCount === runIds.length}
    <div class="metric-grid" aria-busy="true" aria-label="Loading metrics">
      {#each [0, 1, 2, 3, 4, 5] as i (i)}
        <div class="metric-panel"><ChartSkeleton variant="line" height={160} showTitle /></div>
      {/each}
    </div>
  {:else}
    <div class="flex flex-wrap items-center gap-[10px]">
      <input
        class="flex-1 min-w-[200px] bg-(--color-c-gray-08,#1c1c1c) text-(--text) [border:1px_solid_var(--border,#3a3a3a)] rounded-[5px] p-[6px_10px] text-[12px] [font-family:inherit] focus:outline-none focus:[border-color:color-mix(in_srgb,var(--accent)_55%,transparent)]"
        type="search"
        placeholder="Search panels (e.g. loss, rollout/)"
        bind:value={search}
        aria-label="Search panels by metric name"
      />
      <!-- Same trailing window LineChart's Smoothing checkbox uses by default. -->
      <label class="inline-flex items-center gap-[5px] text-[12px] text-(--muted) cursor-pointer select-none" title="Trailing mean over the last 5 steps">
        <input type="checkbox" bind:checked={smoothing} />
        <span>Smoothing</span>
      </label>
      {#if Object.values(stepDomains).some(Boolean)}
        <button class="log-button" type="button" onclick={() => (stepDomains = {})}>Reset zoom</button>
      {/if}
      {#if loadingCount}
        <span class="text-(--muted) text-[12px]" role="status">Loading {loadingCount} more…</span>
      {/if}
    </div>

    {#if !union.keys.length}
      <div class="detail-empty">No metrics reported yet for the {visibleRuns.length === runIds.length ? "" : "visible "}selected runs.</div>
    {:else if !groups.length}
      <div class="detail-empty">No metrics match “{search}”.</div>
    {/if}

    {#each groups as group (group.name)}
      <GroupSection
        title={group.name}
        subtitle={`${group.keys.length} ${group.keys.length === 1 ? "panel" : "panels"}`}
        expanded={!collapsed.has(group.name)}
        onToggle={() => (collapsed = toggleInSet(collapsed, group.name))}
      >
        <div class="metric-grid p-[0_12px_12px]">
          {#each group.keys as key (key)}
            {@const axis = union.stepKeys[key] ?? ""}
            <div class="metric-panel">
              <CompareLineChart
                title={key}
                series={seriesFor(key)}
                smoothingWindow={smoothing ? 5 : 1}
                xLabel={metricAxisLabel(union.stepKeys[key])}
                formatY={formatMetricValue}
                xDomain={stepDomains[axis] ?? null}
                onChangeDomainX={(domain) => onDomainChange(axis, domain)}
              />
            </div>
          {/each}
        </div>
      </GroupSection>
    {/each}
  {/if}
</section>
