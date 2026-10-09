<script>
  // Metrics tab: the scalars mirrored from wandb/trackio, laid out like a
  // W&B workspace — one collapsible section per key prefix (`train/`,
  // `rollout/`, ...), a grid of small panels, a search box, and a shared
  // step window per axis so training and rollout counters stay distinct.
  import ChartSkeleton from "./ChartSkeleton.svelte";
  import GroupSection from "./GroupSection.svelte";
  import LineChart from "./LineChart.svelte";
  import { fetchRunMetrics } from "../lib/api.js";
  import { formatMetricValue, groupMetricKeys, metricAxisLabel } from "../lib/metricSeries.js";

  let { runId, isRunning = false } = $props();

  const POLL_MS = 5000;

  let payload = $state(null);
  let error = $state("");
  let search = $state("");
  let collapsed = $state(new Set());
  let stepDomains = $state({}); // axis → [lo, hi] while zoomed

  async function load(id, signal) {
    try {
      const next = await fetchRunMetrics(id, { signal });
      if (signal.aborted) return;
      payload = next;
      error = "";
    } catch (err) {
      if (!signal.aborted) error = err instanceof Error ? err.message : String(err);
    }
  }

  $effect(() => {
    runId;
    payload = null;
    error = "";
    collapsed = new Set();
    stepDomains = {};
  });
  // Poll while the run is live or the dashboard still has points buffered.
  // (`stale` is a derived boolean so a fresh payload only reruns this when it flips.)
  let stale = $derived(payload?.stale ?? false);
  $effect(() => {
    const id = runId;
    const poll = isRunning || stale;
    if (!id) return;
    const controller = new AbortController();
    load(id, controller.signal);
    const interval = poll ? setInterval(() => load(id, controller.signal), POLL_MS) : null;
    return () => {
      controller.abort();
      if (interval) clearInterval(interval);
    };
  });

  let series = $derived(payload?.series ?? {});
  let stepKeys = $derived(payload?.step_keys ?? {});
  let allKeys = $derived(Object.keys(series));
  let groups = $derived(groupMetricKeys(allKeys, search));
  let rows = $derived.by(() => {
    const out = new Map();
    for (const key of allKeys) out.set(key, series[key].map(([x, y]) => ({ x, y })));
    return out;
  });

  function toggleGroup(name) {
    const next = new Set(collapsed);
    if (!next.delete(name)) next.add(name);
    collapsed = next;
  }

  function onDomainChange(axis, domain) {
    stepDomains = { ...stepDomains, [axis]: domain && domain[1] > domain[0] ? [Math.floor(domain[0]), Math.ceil(domain[1])] : null };
  }
</script>

<div class="flex flex-col gap-[16px] min-w-0">
  {#if !payload && !error}
    <div class="metric-grid" aria-busy="true" aria-label="Loading metrics">
      {#each [0, 1, 2, 3, 4, 5] as i (i)}
        <div class="metric-panel"><ChartSkeleton variant="line" height={150} showTitle /></div>
      {/each}
    </div>
  {:else if !payload}
    <div class="detail-empty" role="alert">Couldn't load metrics: {error}</div>
  {:else if !allKeys.length}
    <div class="detail-empty">
      <div class="text-(--text-bright) mb-[6px]">No metrics reported yet.</div>
      <div class="max-w-[64ch] leading-[1.5]">
        Scalars the framework logs through <code>wandb.log</code> show up here for every metric provider.
      </div>
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
      {#if Object.values(stepDomains).some(Boolean)}
        <button class="log-button" type="button" onclick={() => stepDomains = {}}>Reset zoom</button>
      {/if}
      {#if error}
        <span class="text-[#fbbf24] text-[12px]" role="status">Metrics may be out of date: {error}</span>
      {/if}
    </div>

    {#if !groups.length}
      <div class="detail-empty">No metrics match “{search}”.</div>
    {/if}

    {#each groups as group (group.name)}
      <GroupSection
        title={group.name}
        subtitle={`${group.keys.length} ${group.keys.length === 1 ? "panel" : "panels"}`}
        expanded={!collapsed.has(group.name)}
        onToggle={() => toggleGroup(group.name)}
      >
        <div class="metric-grid p-[0_12px_12px]">
          {#each group.keys as key (key)}
            <div class="metric-panel">
              <LineChart
                title={key}
                data={rows.get(key)}
                height={150}
                label="value"
                axes
                ariaLabel={`${key} over ${metricAxisLabel(stepKeys[key]).toLowerCase()}s`}
                formatX={(row) => `${metricAxisLabel(stepKeys[key])} ${row?.x ?? ""}`}
                formatY={formatMetricValue}
                xDomain={stepDomains[stepKeys[key] ?? ""] ?? null}
                onChangeDomainX={(domain) => onDomainChange(stepKeys[key] ?? "", domain)}
              />
            </div>
          {/each}
        </div>
      </GroupSection>
    {/each}
  {/if}
</div>
