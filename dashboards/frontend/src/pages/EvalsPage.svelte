<script>
  import {
    Check,
    ChevronDown,
    ChevronRight,
    ExternalLink,
    Filter,
    PanelRightClose,
    Search,
  } from "lucide-svelte";
  import ConversationView from "../components/ConversationView.svelte";
  import Drawer from "../components/Drawer.svelte";
  import FilterBulkActions from "../components/FilterBulkActions.svelte";
  import GroupSection from "../components/GroupSection.svelte";
  import MinimalTableSkeleton from "../components/MinimalTableSkeleton.svelte";
  import ResizableTable from "../components/ResizableTable.svelte";
  import StatusPill from "../components/StatusPill.svelte";
  import TimeAgo from "../components/TimeAgo.svelte";
  import { toggleInSet } from "../lib/set.js";

  let {
    allEvals,
    evalCompletedTotal,
    evalPendingTotal,
    evalFailedTotal,
    loading,
    error,
    evalConfigGroups,
    fetchEvalDetail,
    getEvalDisplay,
    evalConfigMeta,
  } = $props();

  let search = $state("");
  let activeStatusFilters = $state(new Set(["completed", "pending", "failed"]));
  let statusMenuOpen = $state(false);
  let activeDatasetFilters = $state(new Set());
  let datasetMenuOpen = $state(false);
  let expandedConfigIds = $state(new Set());
  let expandedInitialized = $state(false);
  let seenDatasets = new Set();
  const evalColumns = [
    { key: "eval_id", label: "Eval ID", width: 220, minWidth: 140 },
    { key: "dataset", label: "Dataset", width: 180, minWidth: 120 },
    { key: "model", label: "Base model", width: 210, minWidth: 140 },
    { key: "status", label: "Status", width: 130, minWidth: 96 },
    { key: "score", label: "Average score", width: 130, minWidth: 110 },
    { key: "examples", label: "Examples", width: 100, minWidth: 86 },
    { key: "created", label: "Created", width: 116, minWidth: 96 },
  ];
  const evalSkeletonColumns = evalColumns.map((column) => column.label);

  function safeText(value) {
    if (value && typeof value === "object" && "value" in value) return value.value;
    return value != null ? String(value) : "";
  }

  function includesText(value, query) {
    return safeText(value).toLowerCase().includes(query);
  }

  function nonPlaceholderText(value) {
    const text = safeText(value).trim();
    if (!text || text === "—") return "";
    return text;
  }

  function evalConfigIdFallbackMeta(evalConfigId) {
    const raw = safeText(evalConfigId).trim();
    if (!raw) return { dataset: "", evalFn: "" };
    const parts = raw.split(".");
    if (parts.length < 3) return { dataset: "", evalFn: "" };
    const [prefix, dataset, evalFn] = parts;
    if (prefix !== "EvalConfig") return { dataset: "", evalFn: "" };
    return { dataset: dataset || "", evalFn: evalFn || "" };
  }

  function groupSubtitle(group) {
    const fallback = evalConfigIdFallbackMeta(group.evalConfigId);
    const parts = [];
    const dataset = nonPlaceholderText(group.meta.dataset) || fallback.dataset;
    const fnName =
      nonPlaceholderText(group.meta.evalFn || group.meta.judge) || fallback.evalFn;
    if (dataset) parts.push(dataset);
    if (fnName) parts.push(fnName);
    return parts.join(" • ");
  }

  function groupDataset(group) {
    const fallback = evalConfigIdFallbackMeta(group.evalConfigId);
    return nonPlaceholderText(group.meta.dataset) || fallback.dataset || "Unknown";
  }

  function evalBaseModel(run) {
    return nonPlaceholderText(run.eval.model_name) || "—";
  }

  function evalId(run) {
    return safeText(run.eval.eval_id).trim() || "—";
  }

  function toggleGroup(evalConfigId) {
    expandedConfigIds = toggleInSet(expandedConfigIds, evalConfigId);
  }

  $effect(() => {
    if (expandedInitialized || !evalConfigGroups.length) return;
    expandedConfigIds = new Set(evalConfigGroups.map((group) => group.evalConfigId));
    expandedInitialized = true;
  });

  let datasetOptions = $derived.by(() =>
    [...new Set(evalConfigGroups.map((group) => groupDataset(group)))].sort((a, b) =>
      a.localeCompare(b),
    ),
  );

  let datasetCounts = $derived.by(
    () =>
      evalConfigGroups.reduce((acc, group) => {
        const dataset = groupDataset(group);
        acc[dataset] = (acc[dataset] || 0) + group.runs.length;
        return acc;
      }, {}),
  );

  let allStatusFiltersActive = $derived(activeStatusFilters.size === 3);
  let allDatasetFiltersActive = $derived.by(() =>
    datasetOptions.length > 0 && datasetOptions.every((dataset) => activeDatasetFilters.has(dataset)),
  );

  $effect(() => {
    const optionsSet = new Set(datasetOptions);
    const next = new Set(
      [...activeDatasetFilters].filter((dataset) => optionsSet.has(dataset)),
    );
    let changed = false;

    if (next.size !== activeDatasetFilters.size) {
      changed = true;
    }

    for (const dataset of datasetOptions) {
      if (!seenDatasets.has(dataset)) {
        seenDatasets.add(dataset);
        next.add(dataset);
        changed = true;
      }
    }

    for (const dataset of [...seenDatasets]) {
      if (!optionsSet.has(dataset)) {
        seenDatasets.delete(dataset);
      }
    }

    if (changed) activeDatasetFilters = next;
  });

  let filteredGroups = $derived.by(() => {
    const query = search.trim().toLowerCase();
    return evalConfigGroups
      .map((group) => {
        let runs = group.runs;
        if (!activeDatasetFilters.has(groupDataset(group))) {
          return null;
        }
        if (activeStatusFilters.size === 0) {
          return null;
        }
        runs = runs.filter((run) => {
          const bucket =
            run.status === "Completed"
              ? "completed"
              : run.status === "Pending"
                ? "pending"
                : "failed";
          return activeStatusFilters.has(bucket);
        });
        if (!runs.length) {
          return null;
        }
        if (query) {
          const groupMatches =
            includesText(group.evalConfigId, query) ||
            includesText(groupDataset(group), query) ||
            includesText(group.meta.model, query) ||
            includesText(group.meta.judge, query) ||
            includesText(group.meta.evalFn, query);
          if (!groupMatches) {
            runs = runs.filter((run) => {
              return (
                includesText(run.eval.eval_id, query) ||
                includesText(run.eval.model_name, query)
              );
            });
          }
        }
        if (!runs.length) return null;
        return {
          ...group,
          visibleRuns: runs,
        };
      })
      .filter(Boolean);
  });

  function toggleStatusFilter(status) {
    activeStatusFilters = toggleInSet(activeStatusFilters, status);
  }

  function selectAllStatusFilters() {
    activeStatusFilters = new Set(["completed", "pending", "failed"]);
  }

  function clearStatusFilters() {
    activeStatusFilters = new Set();
  }

  function toggleDatasetFilter(dataset) {
    activeDatasetFilters = toggleInSet(activeDatasetFilters, dataset);
  }

  function selectAllDatasetFilters() {
    activeDatasetFilters = new Set(datasetOptions);
  }

  function clearDatasetFilters() {
    activeDatasetFilters = new Set();
  }

  let selectedEval = $state(null);
  let selectedEvalDetail = $state(null);
  let loadingDetail = $state(false);
  let exampleSearch = $state("");

  let expandedExamples = $state(new Set());
  let expandedThinking = $state(new Set());

  function scoreColor(score) {
    if (score >= 0.8) return "var(--color-c-green-80)";
    if (score > 0) return "var(--color-c-orange-80)";
    return "var(--color-c-red-80)";
  }

  function examplePromptText(row) {
    return row.prompt || row.metadata?.prompt || row.metadata?.question || row.metadata?.instruction || row.metadata?.input || "";
  }

  async function openEvalDrawer(run, group) {
    selectedEval = { run, group };
    selectedEvalDetail = null;
    loadingDetail = true;
    exampleSearch = "";

    expandedExamples = new Set();
    expandedThinking = new Set();
    const evalId = run.eval.eval_id;
    if (evalId && fetchEvalDetail) {
      try {
        const detail = await fetchEvalDetail(evalId);
        if (selectedEval?.run?.eval?.eval_id === evalId) {
          selectedEvalDetail = detail;
        }
      } catch {
        // detail fetch failed — drawer still shows summary info
      }
    }
    loadingDetail = false;
  }

  function closeEvalDrawer() {
    selectedEval = null;
    selectedEvalDetail = null;
  }

  function toggleExample(index) {
    expandedExamples = toggleInSet(expandedExamples, index);
  }

  function toggleThinking(index) {
    expandedThinking = toggleInSet(expandedThinking, index);
  }

  let drawerRows = $derived.by(() => {
    const rows = selectedEvalDetail?.rows || [];
    const query = exampleSearch.trim().toLowerCase();
    return rows
      .map((row, index) => ({ ...row, _index: index }))
      .filter((row) => {
        if (query) {
          const text = examplePromptText(row).toLowerCase();
          if (!text.includes(query) && !(row.response || "").toLowerCase().includes(query)) return false;
        }
        return true;
      });
  });

  const HISTOGRAM_BINS = 10;

  let scoreHistogram = $derived.by(() => {
    const rows = selectedEvalDetail?.rows || [];
    if (!rows.length) return null;
    const bins = Array.from({ length: HISTOGRAM_BINS }, (_, i) => ({
      min: i / HISTOGRAM_BINS,
      max: (i + 1) / HISTOGRAM_BINS,
      count: 0,
    }));
    for (const row of rows) {
      // Clamp into [0, BINS-1]: scores can fall outside [0,1] (e.g. a reward-style
      // metric), and a negative score would index bins[-n] → undefined → crash.
      const idx = Math.max(
        0,
        Math.min(Math.floor(row.score * HISTOGRAM_BINS), HISTOGRAM_BINS - 1),
      );
      bins[idx].count++;
    }
    const maxCount = Math.max(...bins.map((b) => b.count));
    return { bins, maxCount, total: rows.length };
  });

  let drawerMeta = $derived.by(() => {
    if (!selectedEval) return null;
    const { run, group } = selectedEval;
    const ev = run.eval;
    const meta = evalConfigMeta(group.config, ev);
    const display = getEvalDisplay(ev);
    return {
      evalId: ev.eval_id || "",
      status: display.bucket,
      pillStatus: display.pill,
      statusLabel: display.label,
      model: evalBaseModel(run),
      config: nonPlaceholderText(meta.dataset) || "—",
      grading: nonPlaceholderText(meta.evalFn || meta.judge) || "—",
      avgScore: run.avgScore,
      totalRows: run.totalRows,
      createdAt: run.createdAt,
      modalAppUrl: ev.modal_app_url || null,
    };
  });
</script>

<svelte:window
  onclick={() => {
    statusMenuOpen = false;
    datasetMenuOpen = false;
  }}
/>

<section class="summary-sticky grid [grid-template-columns:repeat(3,minmax(0,1fr))] gap-[14px] p-[0_24px] mb-[24px] max-[1080px]:[grid-template-columns:repeat(2,minmax(0,1fr))]">
  <article class="summary-card">
    <span class="summary-label">Total runs</span>
    <strong>{allEvals.length}</strong>
  </article>
  <article class="summary-card">
    <span class="summary-label">Completed runs</span>
    <strong>{evalCompletedTotal}</strong>
  </article>
  <article class="summary-card">
    <span class="summary-label">Pending runs</span>
    <strong>{evalPendingTotal}</strong>
  </article>
</section>

<section class="[border:0] [background:transparent] p-[0_24px_16px] max-[900px]:pb-[24px]">
  <div class="mb-[24px] flex items-center gap-[0.4rem] max-[900px]:flex-col max-[900px]:[align-items:stretch]">
    <label class="inline-flex items-center gap-[0.42rem] [border:1px_solid_var(--border)] rounded-[7px] bg-(--panel) min-w-[220px] w-[min(320px,100%)] p-[0.24rem_0.55rem] max-[900px]:w-full" aria-label="Search eval runs">
      <span class="search-icon"><Search size={13} /></span>
      <input
        type="search"
        class="[border:0] [outline:0] [background:transparent] text-(--text) w-full min-w-0 [font:inherit] text-[0.78rem] placeholder:text-(--muted)"
        placeholder="Search"
        bind:value={search}
        autocomplete="off"
        spellcheck="false"
      />
    </label>
    <div class="evals-menu-wrap">
      <button
        class="evals-status-filter"
        class:evals-open={statusMenuOpen}
        onclick={(event) => {
          event.stopPropagation();
          statusMenuOpen = !statusMenuOpen;
          datasetMenuOpen = false;
        }}
      >
        <Filter size={12} />
        <span>Status</span>
        <ChevronDown
          size={12}
          style={`transform: ${statusMenuOpen ? "rotate(180deg)" : "rotate(0deg)"};`}
        />
      </button>
      {#if statusMenuOpen}
        <div class="status-menu">
          <FilterBulkActions
            allSelected={allStatusFiltersActive}
            noneSelected={activeStatusFilters.size === 0}
            onSelectAll={selectAllStatusFilters}
            onDeselectAll={clearStatusFilters}
          />
          <button
            class="status-item"
            onclick={(event) => {
              event.stopPropagation();
              toggleStatusFilter("completed");
            }}
          >
            <span class="checkmark" class:checked={activeStatusFilters.has("completed")}>
              {#if activeStatusFilters.has("completed")}
                <Check size={11} />
              {/if}
            </span>
            <span class="item-label">Completed</span>
            <span class="status-count">{evalCompletedTotal}</span>
          </button>
          <button
            class="status-item"
            onclick={(event) => {
              event.stopPropagation();
              toggleStatusFilter("pending");
            }}
          >
            <span class="checkmark" class:checked={activeStatusFilters.has("pending")}>
              {#if activeStatusFilters.has("pending")}
                <Check size={11} />
              {/if}
            </span>
            <span class="item-label">Pending</span>
            <span class="status-count">{evalPendingTotal}</span>
          </button>
          <button
            class="status-item"
            onclick={(event) => {
              event.stopPropagation();
              toggleStatusFilter("failed");
            }}
          >
            <span class="checkmark" class:checked={activeStatusFilters.has("failed")}>
              {#if activeStatusFilters.has("failed")}
                <Check size={11} />
              {/if}
            </span>
            <span class="item-label">Failed</span>
            <span class="status-count">{evalFailedTotal}</span>
          </button>
        </div>
      {/if}
    </div>
    <div class="evals-menu-wrap">
      <button
        class="evals-status-filter"
        class:evals-open={datasetMenuOpen}
        onclick={(event) => {
          event.stopPropagation();
          datasetMenuOpen = !datasetMenuOpen;
          statusMenuOpen = false;
        }}
      >
        <Filter size={12} />
        <span>Dataset</span>
        <ChevronDown
          size={12}
          style={`transform: ${datasetMenuOpen ? "rotate(180deg)" : "rotate(0deg)"};`}
        />
      </button>
      {#if datasetMenuOpen}
        <div class="status-menu w-[min(320px,calc(100vw_-_2rem))]!">
          <FilterBulkActions
            allSelected={allDatasetFiltersActive}
            noneSelected={activeDatasetFilters.size === 0}
            onSelectAll={selectAllDatasetFilters}
            onDeselectAll={clearDatasetFilters}
          />
          {#each datasetOptions as dataset (dataset)}
            <button
              class="status-item"
              onclick={(event) => {
                event.stopPropagation();
                toggleDatasetFilter(dataset);
              }}
            >
              <span class="checkmark" class:checked={activeDatasetFilters.has(dataset)}>
                {#if activeDatasetFilters.has(dataset)}
                  <Check size={11} />
                {/if}
              </span>
              <span class="dataset-item-label">{dataset}</span>
              <span class="status-count">{datasetCounts[dataset] || 0}</span>
            </button>
          {/each}
        </div>
      {/if}
    </div>
  </div>

  <div class="p-0">
    {#if loading}
      <div class="table-wrap freeze-header">
        <MinimalTableSkeleton
          class="evals-runs-table"
          columns={evalSkeletonColumns}
          rows={6}
        />
      </div>
    {:else if error}
      <div class="page-empty">Failed to load: {error}</div>
    {:else if !allEvals.length}
      <div class="page-empty">No eval results yet.</div>
    {:else}
      <div class="flex flex-col gap-[24px] p-0">
        {#if !filteredGroups.length}
          <div class="page-empty">No evals match the current filters.</div>
        {/if}
        {#each filteredGroups as group (group.evalConfigId)}
          <GroupSection
            title={group.evalConfigId || group.meta.dataset}
            subtitle={groupSubtitle(group)}
            expanded={expandedConfigIds.has(group.evalConfigId)}
            onToggle={() => toggleGroup(group.evalConfigId)}
          >
            {#snippet meta()}
              {#if nonPlaceholderText(group.meta.model)}
                <span class="group-meta-pill">{group.meta.model}</span>
              {/if}
              {#if nonPlaceholderText(group.meta.split)}
                <span class="group-meta-pill">split: {group.meta.split}</span>
              {/if}
              <span class="group-meta-pill">total evals: {group.totalEvals}</span>
              <span class="group-meta-pill">avg: {group.avgAccuracy.toFixed(4)}</span>
              {#if group.latestCreatedAt}
                <span class="group-meta-pill [font-variant-numeric:tabular-nums]">
                  <TimeAgo timestamp={group.latestCreatedAt} showJustNow falsyRepresentation="—" />
                </span>
              {/if}
            {/snippet}

            <div class="table-wrap freeze-header" style="--frozen-table-offset: 360px;">
              <ResizableTable class="evals-runs-table" columns={evalColumns} stickyFirstColumn>
                <tbody>
                  {#each group.visibleRuns as run, runIndex (run.eval.eval_id || `${group.evalConfigId}-${run.eval.created_at || 0}-${runIndex}`)}
                    {@const id = evalId(run)}
                    {@const baseModel = evalBaseModel(run)}
                    {@const dataset = groupDataset(group)}
                    <tr
                      class="eval-row-clickable"
                      class:row-selected={selectedEval?.run?.eval?.eval_id === run.eval.eval_id}
                      onclick={() => openEvalDrawer(run, group)}
                    >
                      <td class="evals-mono evals-name-cell" title={id}>
                        <span class="truncate-text">{id}</span>
                      </td>
                      <td class="evals-dataset-cell" title={dataset}>
                        <span class="truncate-text">{dataset}</span>
                      </td>
                      <td class="base-model-cell" title={baseModel}>
                        <span class="truncate-text">{baseModel}</span>
                      </td>
                      <td>
                        <StatusPill status={run.pillStatus} label={run.statusLabel} />
                      </td>
                      <td class="text-(--text-bright) font-[600]">
                        {run.status === "Failed" ? "—" : run.avgScore.toFixed(4)}
                      </td>
                      <td>{run.totalRows ? run.totalRows : "—"}</td>
                      <td class="created-cell">
                        <TimeAgo timestamp={run.createdAt} showJustNow falsyRepresentation="—" />
                      </td>
                    </tr>
                  {/each}
                </tbody>
              </ResizableTable>
            </div>
          </GroupSection>
        {/each}
      </div>
    {/if}
  </div>
</section>

{#if selectedEval && drawerMeta}
  <Drawer open={!!selectedEval} onclose={closeEvalDrawer} width="min(720px, 100vw)">
    <div class="h-full flex flex-col">
      <div class="p-[24px_24px_16px] flex justify-between [align-items:flex-start] gap-[12px]">
        <div class="min-w-0">
          <span class="text-(--muted) text-[14px] leading-[20px]">Eval</span>
          <div class="flex items-center gap-[8px] mt-[4px]">
            <h2 class="text-(--text-bright) [font-family:var(--font-mono)] text-[20px] font-normal leading-[32px] overflow-hidden text-ellipsis whitespace-nowrap">{drawerMeta.evalId}</h2>
            <StatusPill status={drawerMeta.pillStatus} label={drawerMeta.statusLabel} />
          </div>
        </div>
        <div class="flex items-center gap-[16px] [flex-shrink:0]">
          {#if drawerMeta.modalAppUrl}
            <a
              class="inline-flex items-center gap-[4px] [border:1px_solid_var(--color-c-gray-20)] rounded-[4px] p-[2px_6px] text-(--color-c-gray-80) text-[12px] font-medium leading-[16px] [text-decoration:none] whitespace-nowrap ghost-hover"
              href={drawerMeta.modalAppUrl}
              target="_blank"
              rel="noopener noreferrer"
            >
              <span>Open in Modal</span>
              <ExternalLink size={12} />
            </a>
          {/if}
          <button class="[border:0] [background:transparent] text-(--muted) cursor-pointer inline-flex items-center p-0 hover:text-(--text-bright)" onclick={closeEvalDrawer} aria-label="Close drawer">
            <PanelRightClose size={20} />
          </button>
        </div>
      </div>

      <section class="p-[0_24px_16px]">
        <div class="drawer-meta-row">
          <span class="drawer-meta-key">Model</span>
          <span class="drawer-meta-value">{drawerMeta.model}</span>
        </div>
        <div class="drawer-meta-row">
          <span class="drawer-meta-key">Config</span>
          <span class="drawer-meta-value evals-mono">{drawerMeta.config}</span>
        </div>
        <div class="drawer-meta-row">
          <span class="drawer-meta-key">Grading</span>
          <span class="drawer-meta-value">{drawerMeta.grading}</span>
        </div>
        <div class="drawer-meta-row">
          <span class="drawer-meta-key">Avg score</span>
          <span class="drawer-meta-value" style:color="var(--color-c-green-100)">
            {drawerMeta.avgScore.toFixed(3)}
          </span>
        </div>
        <div class="drawer-meta-row">
          <span class="drawer-meta-key">Created</span>
          <span class="drawer-meta-value">
            <TimeAgo timestamp={drawerMeta.createdAt} showJustNow falsyRepresentation="—" />
          </span>
        </div>
      </section>

      <div class="drawer-divider"></div>

      {#if scoreHistogram}
        <section class="p-[16px_24px] flex flex-col gap-[8px]">
          <span class="text-(--text-bright) text-[14px] font-medium leading-[20px]">Score distribution</span>
          <div class="flex [align-items:flex-end] gap-[2px] h-[64px] p-[0_1px]">
            {#each scoreHistogram.bins as bin, i (i)}
              <div class="flex-1 h-full flex [align-items:flex-end]" title="{bin.min.toFixed(1)}–{bin.max.toFixed(1)}: {bin.count}">
                <div
                  class="w-full min-h-[2px] rounded-[2px_2px_0_0] bg-(--color-c-gray-30)"
                  style:height="{scoreHistogram.maxCount > 0 ? (bin.count / scoreHistogram.maxCount) * 100 : 0}%"
                ></div>
              </div>
            {/each}
          </div>
          <div class="flex justify-between text-(--muted) text-[11px] leading-[16px] [font-variant-numeric:tabular-nums]">
            <span>0</span>
            <span>0.5</span>
            <span>1.0</span>
          </div>
        </section>
      {/if}

      <div class="drawer-divider"></div>

      <section class="p-[24px] flex flex-col gap-[16px] flex-1 min-h-0">
        <div class="pt-[4px]">
          <div class="flex items-center gap-[8px]">
            <span class="text-(--text-bright) text-[14px] font-medium leading-[20px]">Examples</span>
            {#if drawerMeta.totalRows}
              <span class="bg-(--color-surface-secondary) rounded-[4px] p-[4px_6px] text-(--muted) text-[12px] leading-[12px]">{drawerMeta.totalRows} examples</span>
            {/if}
          </div>
        </div>

        <div class="flex items-center justify-between gap-[12px] max-[900px]:flex-col max-[900px]:[align-items:stretch]">
          <label class="inline-flex items-center gap-[8px] [border:1px_solid_var(--color-c-gray-10)] rounded-[6px] [background:transparent] w-[260px] h-[32px] p-[6px_8px] max-[900px]:w-full" aria-label="Search prompts">
            <span class="inline-flex text-(--muted-strong) [flex-shrink:0]"><Search size={16} /></span>
            <input
              type="search"
              class="[border:0] [outline:0] [background:transparent] text-(--text) w-full min-w-0 [font:inherit] text-[14px] placeholder:text-(--muted-strong)"
              placeholder="Search prompts"
              bind:value={exampleSearch}
              autocomplete="off"
              spellcheck="false"
            />
          </label>
        </div>

        {#if loadingDetail}
          <div class="p-[24px] text-(--muted) text-center text-[14px]">Loading examples...</div>
        {:else if !selectedEvalDetail?.rows?.length}
          <div class="examples-empty">No example data available for this eval.</div>
        {:else if !drawerRows.length}
          <div class="examples-empty">No examples match the current filter.</div>
        {:else}
          <div class="flex flex-col gap-[16px]">
            {#each drawerRows as row (row._index)}
              {@const promptText = examplePromptText(row)}
              <div class="bg-(--color-c-gray-5) [border:1px_solid_var(--color-c-gray-10)] rounded-[6px] overflow-hidden">
                <button class="w-full [border:0] [background:transparent] text-inherit flex items-center gap-0 p-0 cursor-pointer text-left [font:inherit] hover:[background:rgba(255,255,255,0.02)]" onclick={() => toggleExample(row._index)}>
                  <span class="flex items-center justify-center w-[40px] p-[8px_12px] text-(--muted) shrink-0 self-stretch">
                    {#if expandedExamples.has(row._index)}
                      <ChevronDown size={16} />
                    {:else}
                      <ChevronRight size={16} />
                    {/if}
                  </span>
                  <span class="text-(--muted) text-[12px] leading-[16px] [flex-shrink:0] w-[20px]">{row._index}</span>
                  {#if promptText}
                    <span class="example-prompt">{promptText}</span>
                  {:else}
                    <span class="example-prompt text-(--muted)! italic">
                      {row.response ? row.response.slice(0, 80) : `Example ${row._index}`}
                    </span>
                  {/if}
                  <span class="text-[12px] font-medium leading-[16px] p-[8px_16px] shrink-0 [font-variant-numeric:tabular-nums]" style:color={scoreColor(row.score)}>
                    {row.score.toFixed(2)}
                  </span>
                </button>
                {#if expandedExamples.has(row._index)}
                  <div class="[border-top:1px_solid_var(--color-c-gray-10)] p-[12px_16px_12px_40px] flex flex-col gap-[12px]">
                    {#if promptText}
                      <div class="example-section">
                        <span class="example-section-label">Prompt</span>
                        <pre class="example-section-text">{promptText}</pre>
                      </div>
                    {/if}
                    {#if row.metadata?.trajectory_messages?.length}
                      <div class="example-section">
                        <span class="example-section-label">Trajectory</span>
                        <ConversationView
                          messages={row.metadata.trajectory_messages}
                          response={row.response || ""}
                        />
                      </div>
                    {:else if row.parsed_response}
                      {#if row.parsed_response.thinking}
                        <div class="example-section">
                          <button class="inline-flex items-center gap-[4px] self-start [background:none] [border:1px_solid_var(--color-c-gray-10)] rounded-[4px] text-(--muted) text-[11px] p-[2px_8px] cursor-pointer [font:inherit] hover:text-(--text)" onclick={() => toggleThinking(row._index)}>
                            {#if expandedThinking.has(row._index)}
                              <ChevronDown size={12} />
                            {:else}
                              <ChevronRight size={12} />
                            {/if}
                            <span>Thinking</span>
                          </button>
                          {#if expandedThinking.has(row._index)}
                            <pre class="example-section-text text-(--muted)! [border:1px_solid_var(--color-c-gray-10,#2f2f2f)] [border-left:3px_solid_var(--color-c-orange-80,#f0a040)] bg-[rgba(240,160,64,0.06)]!">{row.parsed_response.thinking}</pre>
                          {/if}
                        </div>
                      {/if}
                      {#if row.parsed_response.content}
                        <div class="example-section">
                          <span class="example-section-label">Answer</span>
                          <pre class="example-section-text [border:1px_solid_var(--color-c-gray-10,#2f2f2f)] [border-left:3px_solid_var(--green,var(--accent))] bg-[rgba(255,255,255,0.02)]!">{row.parsed_response.content}</pre>
                        </div>
                      {/if}
                      {#if row.parsed_response.tool_calls?.length}
                        <div class="example-section">
                          <span class="example-section-label">Tool calls</span>
                          <div class="flex flex-col gap-[10px]">
                            {#each row.parsed_response.tool_calls as call, i (i)}
                              {@const result = call.response ?? call.result ?? call.output}
                              <div class="flex flex-col gap-[4px] [border:1px_solid_var(--color-c-gray-10,#2f2f2f)] [border-left:3px_solid_var(--accent)] rounded-[4px] p-[8px] [background:rgba(124,156,255,0.05)]">
                                <div class="[font-family:var(--font-mono)] text-[12px] [font-weight:600] text-(--text-bright)">{call.name || `tool ${i + 1}`}</div>
                                <pre class="example-section-text bg-[rgba(0,0,0,0.25)]">{JSON.stringify(call.arguments ?? {}, null, 2)}</pre>
                                {#if result != null}
                                  <span class="example-section-label">Response</span>
                                  <pre class="example-section-text bg-[rgba(0,0,0,0.25)] [border-left:2px_solid_var(--accent)]">{typeof result === "string" ? result : JSON.stringify(result, null, 2)}</pre>
                                {/if}
                              </div>
                            {/each}
                          </div>
                        </div>
                      {/if}
                    {:else if row.response}
                      <div class="example-section">
                        <span class="example-section-label">Response</span>
                        <pre class="example-section-text">{row.response}</pre>
                      </div>
                    {/if}
                    {#if row.metadata?.reference}
                      <div class="example-section">
                        <span class="example-section-label">Reference</span>
                        <pre class="example-section-text">{row.metadata.reference}</pre>
                      </div>
                    {/if}
                    {#if row.metadata?._metadata_type === "audio" || row.metadata?.audio}
                      <div class="example-section">
                        <span class="example-section-label">Audio</span>
                        <!-- TODO(ben/joy): metadata.audio is passed straight to the
                          browser <audio> element, so it only renders what the browser
                          natively decodes from a data-URI (wav/mp3/ogg/flac/aac/webm);
                          anything else shows a silent/broken player. Gate-check media
                          here: validate the data-URI MIME against a renderable set and
                          fall back to a download link when unsupported (and pick the
                          element by modality once we also show image/video). Upstream
                          fix is to normalize to a canonical container at the dataset
                          boundary (see MultimodalDataset.modality). -->
                        <audio
                          class="w-full h-[36px] rounded-[6px] [filter:saturate(0.9)]"
                          controls
                          preload="none"
                          src={row.metadata.audio}
                        ></audio>
                      </div>
                    {/if}
                    {#if row.metadata?._metadata_type === "image" || row.metadata?.image}
                      <div class="example-section">
                        <span class="example-section-label">Image</span>
                        <img
                          class="block w-full max-w-[480px] h-auto rounded-[6px] [border:1px_solid_var(--border)]"
                          src={row.metadata.image}
                          alt="eval input"
                          loading="lazy"
                        />
                      </div>
                    {/if}
                    {#each Object.entries(row.metadata?.metrics ?? {}) as [name, value]}
                      <div class="example-section">
                        <span class="example-section-label">{name.toUpperCase()}</span>
                        <span class="example-section-score">
                          {typeof value === "number" ? value.toFixed(3) : value}
                        </span>
                      </div>
                    {/each}
                    <div class="example-section">
                      <span class="example-section-label">Score</span>
                      <span class="example-section-score" style:color={scoreColor(row.score)}>
                        {row.score.toFixed(4)}
                      </span>
                    </div>
                    {#if row.metadata && Object.keys(row.metadata).length}
                      {@const extraMeta = Object.fromEntries(
                        Object.entries(row.metadata).filter(
                          ([k]) =>
                            ![
                              "_metadata_type",
                              "audio",
                              "image",
                              "reference",
                              "metrics",
                              "hyp",
                            ].includes(k),
                        ),
                      )}
                      {#if Object.keys(extraMeta).length}
                        <div class="example-section">
                          <span class="example-section-label">Metadata</span>
                          <pre class="example-section-text">{JSON.stringify(extraMeta, null, 2)}</pre>
                        </div>
                      {/if}
                    {/if}
                  </div>
                {/if}
              </div>
            {/each}
          </div>
        {/if}
      </section>
    </div>
  </Drawer>
{/if}
