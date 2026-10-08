<script>
  import { onMount } from "svelte";
  import { ChevronDown, ChevronRight, ExternalLink, Play } from "lucide-svelte";
  import { fetchCatalogue, fetchSweep, submitSweep } from "../lib/api.js";
  import {
    CONTEXT_STOPS,
    SORTS,
    baseModel,
    contextRangeLabel,
    formatDate,
    formatDuration,
    formatReward,
    formatRewardChange,
    formatTokens,
    formatUsd,
    inContextRange,
    SWEEP_DATASETS,
    SWEEP_FLAGS,
    gridFromFlags,
    rewardDigits,
    sortLabel,
    sortRows,
    sweepRunCount,
  } from "../lib/catalogue.js";

  let { onOpenRun } = $props();

  const REFRESH_MS = 15000;
  const POLL_MS = 2000;
  const LAST_STOP = CONTEXT_STOPS.length - 1;

  let catalogue = $state(null);
  let error = $state(null);
  let loading = $state(true);

  // ── Catalogue controls ────────────────────────────────────────────────
  let hiddenModels = $state(new Set());
  let contextLow = $state(0);
  let contextHigh = $state(LAST_STOP);
  let sortKey = $state(null);
  let sortDirection = $state("asc");
  let expanded = $state(new Set());

  let rows = $derived(catalogue?.rows ?? []);
  let models = $derived([...new Set(rows.map(baseModel))]);
  let visibleRows = $derived.by(() => {
    const kept = rows.filter(
      (row) =>
        !hiddenModels.has(baseModel(row)) &&
        inContextRange(row.context_length, contextLow, contextHigh),
    );
    return sortKey ? sortRows(kept, sortKey, sortDirection) : kept;
  });
  let runCount = $derived(rows.filter((row) => row.run).length);

  function toggleModel(model) {
    const next = new Set(hiddenModels);
    if (next.has(model)) next.delete(model);
    else next.add(model);
    hiddenModels = next;
  }

  function toggleSort(key) {
    if (sortKey === key) {
      sortDirection = sortDirection === "asc" ? "desc" : "asc";
      return;
    }
    sortKey = key;
    sortDirection = SORTS.find((sort) => sort.key === key).direction;
  }

  function toggleExpanded(id) {
    const next = new Set(expanded);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    expanded = next;
  }

  function rowKey(row) {
    return row.run ? row.run.id : `pending:${row.name}`;
  }

  async function load() {
    try {
      catalogue = await fetchCatalogue();
      error = null;
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    } finally {
      loading = false;
    }
  }

  onMount(() => {
    void load();
    const timer = window.setInterval(() => void load(), REFRESH_MS);
    return () => window.clearInterval(timer);
  });

  // ── Sweep launcher ────────────────────────────────────────────────────
  let launcherOpen = $state(false);
  let selectedEntries = $state(new Set());
  let steps = $state(3);
  let sweepName = $state("");
  let flags = $state([]);
  let datasetLabel = $state(SWEEP_DATASETS[0].label);
  let submitting = $state(false);
  let submitError = $state(null);
  // Operations submitted this page load, newest first, each polled until done.
  let operations = $state([]);

  let entries = $derived(catalogue?.entries ?? []);
  let gridParse = $derived.by(() => {
    try {
      return { grid: gridFromFlags(flags), error: null };
    } catch (err) {
      return { grid: {}, error: err instanceof Error ? err.message : String(err) };
    }
  });
  let plannedRuns = $derived(sweepRunCount([...selectedEntries], gridParse.grid));
  let canSubmit = $derived(
    selectedEntries.size > 0 && !gridParse.error && !submitting,
  );

  let unusedFlags = $derived(SWEEP_FLAGS.filter((f) => !flags.some((row) => row.path === f.path)));

  function addFlag() {
    const next = unusedFlags[0];
    if (next) flags = [...flags, { path: next.path, values: "" }];
  }

  function removeFlag(index) {
    flags = flags.filter((_, i) => i !== index);
  }

  function flagPlaceholder(path) {
    return SWEEP_FLAGS.find((f) => f.path === path)?.placeholder ?? "";
  }

  function toggleEntry(name) {
    const next = new Set(selectedEntries);
    if (next.has(name)) next.delete(name);
    else next.add(name);
    selectedEntries = next;
  }

  function openLauncherFor(name) {
    launcherOpen = true;
    if (!selectedEntries.has(name)) toggleEntry(name);
  }

  async function submit() {
    if (!canSubmit) return;
    submitting = true;
    submitError = null;
    try {
      const op = await submitSweep({
        entries: [...selectedEntries],
        steps: Number(steps),
        name: sweepName.trim() || null,
        grid: gridParse.grid,
        dataset: SWEEP_DATASETS.find((d) => d.label === datasetLabel).spec,
      });
      operations = [{ ...op, entries: [...selectedEntries] }, ...operations];
      poll(op.operation_id);
    } catch (err) {
      submitError = err instanceof Error ? err.message : String(err);
    } finally {
      submitting = false;
    }
  }

  function updateOperation(id, patch) {
    operations = operations.map((op) =>
      op.operation_id === id ? { ...op, ...patch } : op,
    );
  }

  async function poll(id) {
    try {
      const op = await fetchSweep(id);
      updateOperation(id, op);
      if (op.status === "pending") {
        window.setTimeout(() => poll(id), POLL_MS);
      } else {
        void load();
      }
    } catch (err) {
      updateOperation(id, {
        status: "failed",
        error: { type: "poll", message: err instanceof Error ? err.message : String(err) },
      });
    }
  }

  const pillClass = {
    pending: "text-(--muted) border-(--border)",
    succeeded: "text-[#7fee64] border-[#7fee6466]",
    failed: "text-[#f87171] border-[#f8717166]",
  };
</script>

<div class="flex flex-col gap-6 p-6 max-w-[1400px] w-full">
  <section class="flex flex-wrap items-start justify-between gap-4">
    <div class="max-w-[720px]">
      <h2 class="text-lg font-semibold text-(--text-bright)">Recipe catalogue</h2>
    </div>
    <button
      class="btn"
      type="button"
      onclick={() => (launcherOpen = !launcherOpen)}
    >
      <Play size={14} class="icon" />
      {launcherOpen ? "Hide sweep launcher" : "Kick off sweep"}
    </button>
  </section>

  {#if launcherOpen}
    <section class="card flex flex-col gap-4">
      <div class="flex flex-wrap items-baseline justify-between gap-2">
        <h3 class="text-sm font-semibold text-(--text-bright)">New sweep</h3>
        <span class="text-xs text-(--muted)">
          {plannedRuns} run{plannedRuns === 1 ? "" : "s"} will launch
        </span>
      </div>

      <div class="flex flex-col gap-1">
        <span class="text-xs uppercase tracking-wide text-(--muted)">Recipes</span>
        <div class="flex flex-wrap gap-2">
          {#each entries as entry (entry.name)}
            <button
              type="button"
              class={[
                "chip",
                selectedEntries.has(entry.name) && "chip-active",
              ]}
              onclick={() => toggleEntry(entry.name)}
            >
              {entry.name}
            </button>
          {/each}
        </div>
      </div>

      <div class="grid gap-4 md:grid-cols-[120px_1fr]">
        <label class="flex flex-col gap-1 text-xs text-(--muted)">
          Steps
          <input class="field" type="number" min="1" max="100" bind:value={steps} />
        </label>
        <label class="flex flex-col gap-1 text-xs text-(--muted)">
          Name (optional)
          <input class="field" type="text" placeholder="lr sweep" bind:value={sweepName} />
        </label>
      </div>

      <div class="flex flex-col gap-2 text-xs text-(--muted)">
        <span>Sweep flags</span>
        {#each flags as flag, i (flag.path)}
          <div class="flex flex-wrap items-center gap-2">
            <select class="field font-mono" bind:value={flag.path}>
              {#each SWEEP_FLAGS.filter((f) => f.path === flag.path || !flags.some((row) => row.path === f.path)) as option (option.path)}
                <option value={option.path}>{option.path}</option>
              {/each}
            </select>
            <input
              class="field min-w-[200px] flex-1 font-mono"
              type="text"
              placeholder={flagPlaceholder(flag.path)}
              aria-label="Values for {flag.path}"
              bind:value={flag.values}
            />
            <button class="link" type="button" aria-label="Remove {flag.path}" onclick={() => removeFlag(i)}>
              ✕
            </button>
          </div>
        {/each}
        <div>
          <button class="chip" type="button" disabled={!unusedFlags.length} onclick={addFlag}>
            + Add flag
          </button>
        </div>
        {#if gridParse.error}
          <span class="text-[#f87171]">{gridParse.error}</span>
        {/if}
      </div>

      <label class="flex w-fit flex-col gap-1 text-xs text-(--muted)">
        Dataset
        <select class="field" bind:value={datasetLabel}>
          {#each SWEEP_DATASETS as dataset (dataset.label)}
            <option value={dataset.label}>{dataset.label}</option>
          {/each}
        </select>
      </label>

      <div class="flex flex-wrap items-center gap-3">
        <button class="btn" type="button" disabled={!canSubmit} onclick={submit}>
          <Play size={14} class="icon" />
          {submitting ? "Submitting…" : `Launch ${plannedRuns} run${plannedRuns === 1 ? "" : "s"}`}
        </button>
        {#if submitError}
          <span class="text-xs text-[#f87171]">{submitError}</span>
        {/if}
      </div>
    </section>
  {/if}

  {#if operations.length}
    <section class="flex flex-col gap-2">
      {#each operations as op (op.operation_id)}
        <article class="card flex flex-col gap-2 text-sm">
          <div class="flex flex-wrap items-center gap-2">
            <span class={["pill", pillClass[op.status]]}>{op.status}</span>
            <span class="font-mono text-xs text-(--muted)">{op.group_id}</span>
            <span class="text-xs text-(--muted)">· {op.entries.join(", ")}</span>
            <span class="ml-auto font-mono text-[11px] text-(--muted)">{op.operation_id}</span>
          </div>
          {#if op.status === "pending"}
            <p class="text-xs text-(--muted)">Launching training apps…</p>
          {:else if op.status === "failed"}
            <p class="text-xs text-[#f87171]">{op.error?.type}: {op.error?.message}</p>
          {:else if op.result}
            <ul class="flex flex-col gap-1 text-xs">
              {#each op.result.launched as run (run.training_run_id)}
                <li class="flex flex-wrap items-center gap-2">
                  <button
                    type="button"
                    class="link font-mono"
                    onclick={() => onOpenRun?.(run.training_run_id)}
                  >
                    {run.training_run_id}
                  </button>
                  <span class="text-(--muted)">{run.entry}</span>
                  {#each Object.entries(run.overrides) as [key, value] (key)}
                    <span class="tag">{key.split(".").at(-1)}={JSON.stringify(value)}</span>
                  {/each}
                </li>
              {/each}
              {#each op.result.failures as failure, index (index)}
                <li class="text-[#f87171]">
                  {failure.entry}
                  {Object.keys(failure.overrides).length ? JSON.stringify(failure.overrides) : ""}
                  — {failure.error}
                </li>
              {/each}
            </ul>
          {/if}
        </article>
      {/each}
    </section>
  {/if}

  {#if error}
    <p class="text-sm text-[#f87171]">{error}</p>
  {/if}

  {#if !loading && rows.length}
    <section class="flex flex-wrap items-center gap-x-6 gap-y-3 text-xs">
      <div class="flex flex-wrap items-center gap-2">
        <span class="text-(--muted)">Models</span>
        {#each models as model (model)}
          <button
            type="button"
            class={["chip", !hiddenModels.has(model) && "chip-active"]}
            onclick={() => toggleModel(model)}
          >
            {model}
          </button>
        {/each}
      </div>
      <div class="flex items-center gap-2">
        <span class="text-(--muted)">Context</span>
        <div
          class="dual-range"
          style:--lo="{(contextLow / LAST_STOP) * 100}%"
          style:--hi="{(contextHigh / LAST_STOP) * 100}%"
        >
          <input
            type="range"
            min="0"
            max={LAST_STOP}
            aria-label="Minimum context length"
            style:z-index={contextLow === LAST_STOP ? 2 : 1}
            bind:value={contextLow}
            oninput={() => { if (contextLow > contextHigh) contextLow = contextHigh; }}
          />
          <input
            type="range"
            min="0"
            max={LAST_STOP}
            aria-label="Maximum context length"
            bind:value={contextHigh}
            oninput={() => { if (contextHigh < contextLow) contextHigh = contextLow; }}
          />
        </div>
        <span class="min-w-[90px] text-(--text)">{contextRangeLabel(contextLow, contextHigh)}</span>
      </div>
      <div class="flex items-center gap-2">
        <span class="text-(--muted)">Sort</span>
        <div class="btn-group">
          {#each SORTS as sort (sort.key)}
            <button
              type="button"
              class={["btn", sortKey === sort.key && "btn-active"]}
              onclick={() => toggleSort(sort.key)}
            >
              {sortLabel(sort.key)}
              {#if sortKey === sort.key}{sortDirection === "asc" ? "↑" : "↓"}{/if}
            </button>
          {/each}
        </div>
      </div>
      <span class="ml-auto text-(--muted)">
        {visibleRows.length} of {rows.length} rows · {runCount} with runs
      </span>
    </section>

    <div class="overflow-x-auto">
      <table class="minimal-table w-full text-sm">
        <thead>
          <tr>
            <th></th>
            <th>Recipe</th>
            <th>Hardware</th>
            <th>Context</th>
            <th class="text-right">Cost / step</th>
            <th class="text-right">Time / step</th>
            <th class="text-right">Step 1 reward</th>
            <th>Run</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {#each visibleRows as row (rowKey(row))}
            {@const run = row.run}
            {@const key = rowKey(row)}
            {@const digits = run ? rewardDigits(run.steps.map((s) => s.raw_reward)) : 3}
            <tr>
              <td class="w-6">
                {#if run?.steps.length}
                  <button
                    type="button"
                    class="link"
                    aria-label="Toggle step breakdown"
                    onclick={() => toggleExpanded(key)}
                  >
                    {#if expanded.has(key)}<ChevronDown size={14} />{:else}<ChevronRight size={14} />{/if}
                  </button>
                {/if}
              </td>
              <td>
                <div class="flex flex-col">
                  <a class="link font-medium" href={row.recipe_href} target="_blank" rel="noreferrer">
                    {row.recipe}
                  </a>
                  <a class="text-xs text-(--muted) hover:underline" href={row.model_href} target="_blank" rel="noreferrer">
                    {row.model} · {row.framework}
                  </a>
                </div>
              </td>
              <td class="whitespace-nowrap">
                {#if run}
                  {run.gpus}× {run.gpu_type}
                  <span class="text-xs text-(--muted)">
                    · {run.nodes} node{run.nodes === 1 ? "" : "s"}
                    {#if run.gpu_hour_usd !== null}· {formatUsd(run.gpu_hour_usd)}/GPU·h{/if}
                  </span>
                {:else}
                  <span class="text-(--muted)">—</span>
                {/if}
              </td>
              <td class="whitespace-nowrap">
                {#if row.context_length !== null}
                  {formatTokens(row.context_length)}
                {:else}
                  <span class="text-(--muted)">—</span>
                {/if}
              </td>
              <td class="text-right tabular-nums">{run ? formatUsd(run.cost_per_step_usd) : "—"}</td>
              <td class="text-right tabular-nums">{run ? formatDuration(run.time_per_step_s) : "—"}</td>
              <td class="text-right tabular-nums">{run ? formatReward(run.initial_reward, digits) : "—"}</td>
              <td class="whitespace-nowrap text-xs">
                {#if run}
                  <button type="button" class="link font-mono" onclick={() => onOpenRun?.(run.id)}>
                    {run.id}
                  </button>
                  <span class="text-(--muted)">
                    · {run.status} · {formatDate(run.created_at)}
                    · {run.steps_completed}/{catalogue.steps} steps
                  </span>
                {:else}
                  <span class="text-(--muted)">Not run yet</span>
                {/if}
              </td>
              <td class="text-right">
                <button type="button" class="link text-xs" onclick={() => openLauncherFor(row.name)}>
                  {run ? "Re-run" : "Run"}
                </button>
              </td>
            </tr>
            {#if run && expanded.has(key)}
              <tr class="bg-(--panel)">
                <td></td>
                <td colspan="8">
                  <table class="text-xs tabular-nums">
                    <thead>
                      <tr>
                        <th class="pr-6 text-left">Step</th>
                        <th class="pr-6 text-right">Time</th>
                        <th class="pr-6 text-right">Cost</th>
                        <th class="pr-6 text-right">Total</th>
                        <th class="pr-6 text-right">Raw reward</th>
                        <th class="text-right">Δ</th>
                      </tr>
                    </thead>
                    <tbody>
                      {#each run.steps as step, index (step.step)}
                        <tr>
                          <td class="pr-6">{step.step}</td>
                          <td class="pr-6 text-right">{formatDuration(step.seconds)}</td>
                          <td class="pr-6 text-right">{formatUsd(step.cost_usd)}</td>
                          <td class="pr-6 text-right">{formatUsd(step.total_usd)}</td>
                          <td class="pr-6 text-right">{formatReward(step.raw_reward, digits)}</td>
                          <td class="text-right text-(--muted)">
                            {index ? (formatRewardChange(run.steps[index - 1].raw_reward, step.raw_reward, digits) ?? "—") : ""}
                          </td>
                        </tr>
                      {/each}
                    </tbody>
                  </table>
                </td>
              </tr>
            {/if}
          {/each}
        </tbody>
      </table>
    </div>
  {:else if loading}
    <p class="text-sm text-(--muted)">Loading catalogue…</p>
  {/if}
</div>

<style>
  .card {
    border-radius: 6px;
    background: rgba(255, 255, 255, 0.03);
    padding: 14px 16px;
  }
  .chip {
    border: 1px solid var(--border, #2f2f2f);
    border-radius: 999px;
    padding: 2px 10px;
    font-size: 12px;
    color: var(--muted);
    background: transparent;
    cursor: pointer;
  }
  .chip:hover {
    color: var(--text-bright);
  }
  .chip-active {
    color: var(--text-bright);
    background: rgba(127, 238, 100, 0.1);
    border-color: rgba(127, 238, 100, 0.4);
  }
  .btn-active {
    color: var(--text-bright);
    background: rgba(127, 238, 100, 0.1);
  }
  .field {
    border: 1px solid var(--border, #2f2f2f);
    border-radius: 8px;
    background: var(--panel);
    color: var(--text);
    padding: 6px 10px;
    font-size: 13px;
  }
  .link {
    background: none;
    border: 0;
    padding: 0;
    color: var(--text);
    cursor: pointer;
    text-decoration: none;
  }
  .link:hover {
    text-decoration: underline;
    color: var(--text-bright);
  }
  .pill {
    border: 1px solid;
    border-radius: 999px;
    padding: 1px 8px;
    font-size: 11px;
    text-transform: capitalize;
  }
  .tag {
    border: 1px solid var(--border, #2f2f2f);
    border-radius: 999px;
    padding: 1px 7px;
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text);
  }
  .dual-range {
    position: relative;
    width: 160px;
    height: 16px;
  }
  .dual-range::before {
    content: "";
    position: absolute;
    top: 50%;
    left: 0;
    right: 0;
    height: 4px;
    transform: translateY(-50%);
    border-radius: 999px;
    background: linear-gradient(
      to right,
      var(--border, #2f2f2f) var(--lo),
      var(--accent) var(--lo) var(--hi),
      var(--border, #2f2f2f) var(--hi)
    );
  }
  .dual-range input {
    position: absolute;
    inset: 0;
    width: 100%;
    margin: 0;
    background: none;
    pointer-events: none;
    appearance: none;
  }
  .dual-range input::-webkit-slider-runnable-track {
    background: none;
  }
  .dual-range input::-moz-range-track {
    background: none;
  }
  .dual-range input::-webkit-slider-thumb {
    appearance: none;
    pointer-events: auto;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: var(--text-bright);
    border: 2px solid var(--accent);
    cursor: pointer;
  }
  .dual-range input::-moz-range-thumb {
    pointer-events: auto;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: var(--text-bright);
    border: 2px solid var(--accent);
    cursor: pointer;
  }
</style>
