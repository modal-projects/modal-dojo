<script>
  import { ExternalLink, Maximize2, PanelRightClose } from "lucide-svelte";
  import Drawer from "../components/Drawer.svelte";
  import FilterBar from "../components/FilterBar.svelte";
  import FrameworkStageProgress from "../components/FrameworkStageProgress.svelte";
  import GroupSection from "../components/GroupSection.svelte";
  import MinimalTableSkeleton from "../components/MinimalTableSkeleton.svelte";
  import ResizableTable from "../components/ResizableTable.svelte";
  import RunSummary from "../components/RunSummary.svelte";
  import StatusPill from "../components/StatusPill.svelte";
  import TimeAgo from "../components/TimeAgo.svelte";
  import { fetchRun } from "../lib/api.js";
  import { formatTagValue, getGroupTags } from "../lib/format.js";
  import { normalizeMetricLinks } from "../lib/metricLinks.js";
  import { toggleInSet } from "../lib/set.js";

  let {
    totalRuns,
    matchingRunCount,
    hasMoreRuns,
    onLoadMore = () => {},
    completedTotal,
    runningTotal,
    stoppedTotal,
    failedTotal,
    recipes,
    recipeCounts,
    activeRecipes,

    statuses,
    statusCounts,
    activeStatuses,
    groups,
    groupCounts,
    activeGroups,
    trainingTypes,
    trainingTypeCounts,
    activeTrainingTypes,
    filteredRuns,
    runGroups,
    groupBy = $bindable(),
    loading,
    error,
    modelName,
    getStatus,
    showFrameworkStatus,
    fmtDuration,
    search = $bindable(),
    drawerRunId = null,
    onOpenDetail = () => {},
    onCloseDrawer = () => {},
    onToggleRecipe,
    onSelectAllRecipes,
    onClearRecipes,
    onToggleStatus,
    onSelectAllStatuses,
    onClearStatuses,
    onToggleGroup,
    onSelectAllGroups,
    onClearGroups,
    onToggleTrainingType,
    onSelectAllTrainingTypes,
    onClearTrainingTypes,
  } = $props();

  // The drawer is now driven by the parent: it holds the run-summary while the
  // full rollouts/logs detail lives on its own page. Clicking a run (or the
  // drawer's Expand button) navigates to that page; the page's Collapse button
  // brings the summary back as this drawer.
  // The list payload carries only the fields the table renders, so the drawer
  // fetches the run's full record (config and all) when it opens.
  let drawerDetail = $state(null);
  // Set only when the server says the run is gone. The list holds one page, so
  // a run being absent from it means nothing — the drawer renders from the
  // detail fetch for runs deep-linked or scrolled past.
  let drawerRunMissing = $state(false);

  $effect(() => {
    const runId = drawerRunId;
    drawerDetail = null;
    drawerRunMissing = false;
    if (!runId) return;
    const controller = new AbortController();
    fetchRun(runId, { signal: controller.signal })
      .then((detail) => {
        if (detail) drawerDetail = detail;
        else drawerRunMissing = true;
      })
      .catch(() => {});
    return () => controller.abort();
  });

  let selectedRun = $derived.by(() => {
    const listRun = filteredRuns.find((run) => run.run_id === drawerRunId) || null;
    const detail = drawerDetail?.run_id === drawerRunId ? drawerDetail : null;
    if (!listRun) return detail;
    if (!detail) return listRun;
    return { ...listRun, ...detail };
  });

  const drawerWidth = "min(420px, calc(100vw - 24px))";
  const columns = [
    { key: "name", label: "Name", width: 240, minWidth: 140 },
    { key: "status", label: "Status", width: 116, minWidth: 96 },
    { key: "stage", label: "Stage", width: 190, minWidth: 130 },
    { key: "model", label: "Model", width: 210, minWidth: 120 },
    { key: "dataset", label: "Dataset", width: 180, minWidth: 120 },
    { key: "recipe", label: "Recipe", width: 116, minWidth: 88 },
    { key: "training_type", label: "Training type", width: 116, minWidth: 88 },
    { key: "group", label: "Group", width: 280, minWidth: 220 },
    { key: "tags", label: "Tags", width: 520, minWidth: 360 },
    { key: "created", label: "Created", width: 150, minWidth: 130 },
    { key: "updated", label: "Last updated", width: 170, minWidth: 150 },
    { key: "actions", label: "", ariaLabel: "Actions", width: 236, minWidth: 180 },
  ];

  function trainingRunDetailPath(runId) {
    return `/training/${encodeURIComponent(runId)}`;
  }

  function openRunInNewTab(runId) {
    if (typeof window === "undefined") return;
    const url = new URL(trainingRunDetailPath(runId), window.location.href);
    window.open(url.href, "_blank", "noopener,noreferrer");
  }

  function isPlainLeftClick(event) {
    return event.button === 0 && !event.metaKey && !event.ctrlKey && !event.altKey && !event.shiftKey;
  }

  function selectRun(runId, event) {
    if (event?.shiftKey) {
      event.preventDefault();
      openRunInNewTab(runId);
      return;
    }

    if (event && !isPlainLeftClick(event)) return;
    event?.preventDefault();
    onOpenDetail(runId);
  }

  function closeDrawer() {
    onCloseDrawer();
  }

  function frameworkProgress(run) {
    const progress = run?.framework_progress;
    if (!progress || typeof progress !== "object") return null;
    const current = Number(progress.current);
    const total = Number(progress.total);
    if (!Number.isFinite(current) || !Number.isFinite(total) || total <= 0) {
      return null;
    }
    return {
      current: Math.max(0, Math.min(current, total)),
      total,
      unit: progress.unit || "step",
    };
  }

  function progressLabel(progress) {
    if (!progress) return "";
    const unit = String(progress.unit || "step");
    const label = unit.charAt(0).toUpperCase() + unit.slice(1);
    return `${label} ${progress.current} / ${progress.total}`;
  }

  function resumeBadge(run) {
    const state = run?.resume_state;
    if (!state) return "";
    const parts = [];
    if (state.attempt_count > 1) parts.push(`attempt ${state.attempt_count}`);
    if (state.resumed_from_checkpoint) {
      parts.push(
        state.resume_from_iteration != null
          ? `resumed @ ${state.resume_from_iteration}`
          : "resumed",
      );
    }
    return parts.join(" · ");
  }

  function metricLinksForRun(run) {
    return normalizeMetricLinks(
      run?.metric_links?.length ? run.metric_links : run?.wandb_links,
    );
  }

  $effect(() => {
    if (drawerRunId && drawerRunMissing) {
      onCloseDrawer();
    }
  });

  // Inverted vs. evals' expanded set so groups default to expanded, including
  // ones that appear later via auto-refresh.
  let collapsedGroupKeys = $state(new Set());

  function toggleGroupSection(key) {
    collapsedGroupKeys = toggleInSet(collapsedGroupKeys, key);
  }

  $effect(() => {
    void groupBy;
    collapsedGroupKeys = new Set();
  });

  let list = $state(null);

  // A new query restarts paging at the first page. Staying scrolled to the
  // bottom of the old, longer list would immediately ask for page after page
  // again, so send the reader back to the top of the new results.
  $effect(() => {
    void search;
    const root = list;
    if (!root) return;
    for (const wrap of root.querySelectorAll(".table-wrap")) wrap.scrollTop = 0;
  });

  // A run row is ~50 DOM nodes, so a full history is seconds of layout on a
  // phone and a re-render of every row on each refresh. The server hands out
  // one page at a time and scrolling a table near its end asks for the next
  // one. Rows scroll inside `.table-wrap`, and a scroll event is the only
  // signal that means "the reader reached the end of *this* scrollport": a
  // sentinel + IntersectionObserver either never re-fires once it is already
  // visible (a short group's table doesn't scroll at all) or pages the whole
  // history unprompted, which is what this PR exists to stop. Grouped tables
  // may not scroll at all, so the footer's button is what keeps every run
  // reachable; scrolling is only the shortcut.
  function growOnScroll(event) {
    if (!hasMoreRuns) return;
    const el = event.currentTarget;
    if (el.scrollTop + el.clientHeight >= el.scrollHeight - 600) {
      onLoadMore();
    }
  }
</script>

<section class="summary-sticky grid grid-cols-5 gap-[14px] p-[0_24px] mb-[24px] max-[900px]:grid-cols-2">
  <article class="summary-card">
    <span class="summary-label">Total runs</span>
    <strong>{totalRuns}</strong>
  </article>
  <article class="summary-card">
    <span class="summary-label">Completed runs</span>
    <strong>{completedTotal}</strong>
  </article>
  <article class="summary-card">
    <span class="summary-label">Pending runs</span>
    <strong>{runningTotal}</strong>
  </article>
  <article class="summary-card">
    <span class="summary-label">Stopped runs</span>
    <strong>{stoppedTotal}</strong>
  </article>
  <article class="summary-card">
    <span class="summary-label">Failed runs</span>
    <strong>{failedTotal}</strong>
  </article>
</section>

<section class="[border:0] [background:transparent] flex flex-col gap-[24px] p-[0_24px_16px] max-[900px]:p-[0_16px_24px] min-w-0">
  <div class="m-0">
    <FilterBar
      {recipes}
      {recipeCounts}
      {activeRecipes}
      allRecipesActive={activeRecipes.size === recipes.length}
      {statuses}
      {statusCounts}
      {activeStatuses}
      allStatusesActive={activeStatuses.size === statuses.length}
      {groups}
      {groupCounts}
      {activeGroups}
      allGroupsActive={activeGroups.size === groups.length}
      {trainingTypes}
      {trainingTypeCounts}
      {activeTrainingTypes}
      allTrainingTypesActive={activeTrainingTypes.size === trainingTypes.length}
      bind:search
      bind:groupBy
      onToggleRecipe={onToggleRecipe}
      onSelectAllRecipes={onSelectAllRecipes}
      onClearRecipes={onClearRecipes}
      onToggleStatus={onToggleStatus}
      onSelectAllStatuses={onSelectAllStatuses}
      onClearStatuses={onClearStatuses}
      onToggleGroup={onToggleGroup}
      onSelectAllGroups={onSelectAllGroups}
      onClearGroups={onClearGroups}
      onToggleTrainingType={onToggleTrainingType}
      onSelectAllTrainingTypes={onSelectAllTrainingTypes}
      onClearTrainingTypes={onClearTrainingTypes}
    />
  </div>

  <div class="p-0" bind:this={list}>
    {#if loading}
      <div class="table-wrap freeze-header">
        <MinimalTableSkeleton
          class="training-runs-table"
          columns={["Name", "Status", "Stage", "Model", "Dataset", "Recipe", "Training type", "Group", "Tags", "Created", "Last updated", ""]}
          rows={8}
        />
      </div>
    {:else if error}
      <div class="page-empty">Failed to load: {error}</div>
    {:else if !totalRuns}
      <div class="page-empty">No training runs found yet.</div>
    {:else if !filteredRuns.length}
      <div class="page-empty">No runs match the current filters.</div>
    {:else}
      {#snippet runsTable(runs, frozenOffset)}
        <div
          class="table-wrap freeze-header"
          style={frozenOffset ? `--frozen-table-offset: ${frozenOffset};` : ""}
          onscroll={growOnScroll}
        >
          <ResizableTable class="training-runs-table" {columns} stickyFirstColumn>
            <tbody>
              {#each runs as run, runIndex (`${run.run_id || "run"}-${run.created_at || 0}-${runIndex}`)}
                {@const runName = run.run_id || "—"}
                {@const status = getStatus(run)}
                {@const stageLabel = run.display_stage}
                {@const progress = frameworkProgress(run)}
                {@const groupTags = getGroupTags(run)}
                <tr class="run-row" class:row-selected={drawerRunId === run.run_id}>
                  <td class="min-w-0 row-open-cell">
                    <a
                      href={trainingRunDetailPath(run.run_id)}
                      class="cell-open-button"
                      title={runName}
                      aria-label={`Open training run ${runName}`}
                      onclick={(event) => selectRun(run.run_id, event)}
                    >
                      <div class="block text-(--text-bright) [font-family:var(--font-mono)] [font-weight:400] text-[14px] leading-[20px] overflow-hidden text-ellipsis whitespace-nowrap">{runName}</div>
                    </a>
                  </td>
                  <td class="row-open-cell">
                    <a href={trainingRunDetailPath(run.run_id)} class="cell-open-button" onclick={(event) => selectRun(run.run_id, event)}>
                      <span class="flex flex-col items-start gap-[4px] min-w-0 max-w-full">
                        <StatusPill status={status} />
                        {#if resumeBadge(run)}
                          <span class="[border:1px_solid_color-mix(in_srgb,var(--yellow,#fbbf24)_42%,transparent)] rounded-[999px] bg-[color-mix(in_srgb,var(--yellow,#fbbf24)_10%,transparent)] text-(--yellow,#fbbf24) text-[11px] leading-[14px] p-[1px_6px] whitespace-nowrap">{resumeBadge(run)}</span>
                        {/if}
                      </span>
                    </a>
                  </td>
                  <td class="min-w-0 row-open-cell">
                    <a
                      href={trainingRunDetailPath(run.run_id)}
                      class="cell-open-button whitespace-normal! leading-[16px]!"
                      onclick={(event) => selectRun(run.run_id, event)}
                    >
                      {#if showFrameworkStatus(run) && stageLabel}
                        <FrameworkStageProgress
                          progress={progress}
                          progressLabel={progressLabel(progress)}
                          stageLabel={stageLabel}
                          compact
                          active={status.toLowerCase() === "pending"}
                        />
                      {:else}
                        <span class="text-(--muted)">—</span>
                      {/if}
                    </a>
                  </td>
                  <td class="min-w-0 row-open-cell" title={modelName(run)}>
                    <a href={trainingRunDetailPath(run.run_id)} class="cell-open-button" onclick={(event) => selectRun(run.run_id, event)}>
                      {modelName(run)}
                    </a>
                  </td>
                  <td class="min-w-0 row-open-cell" title={run.dataset || "—"}>
                    <a href={trainingRunDetailPath(run.run_id)} class="cell-open-button" onclick={(event) => selectRun(run.run_id, event)}>
                      {run.dataset || "—"}
                    </a>
                  </td>
                  <td class="row-open-cell">
                    <a href={trainingRunDetailPath(run.run_id)} class="cell-open-button" onclick={(event) => selectRun(run.run_id, event)}>
                      {run.recipe || "—"}
                    </a>
                  </td>
                  <td class="row-open-cell">
                    <a href={trainingRunDetailPath(run.run_id)} class="cell-open-button" onclick={(event) => selectRun(run.run_id, event)}>
                      {(run.training_type ?? "rl").toUpperCase()}
                    </a>
                  </td>
                  <td class="group-cell row-open-cell" title={groupTags?.group_id || run.group_id || ""}>
                    <a href={trainingRunDetailPath(run.run_id)} class="cell-open-button" onclick={(event) => selectRun(run.run_id, event)}>
                      {#if groupTags?.group_id || run.group_id}
                        <span class="inline-block max-w-full whitespace-normal [overflow-wrap:anywhere] align-bottom p-[2px_8px] rounded-[999px] text-[0.72rem] [font-variant-numeric:tabular-nums] text-(--muted) [border:1px_solid_var(--border,#2f2f2f)] bg-[color-mix(in_srgb,var(--panel-alt)_70%,transparent)]">{groupTags?.group_id || run.group_id}</span>
                      {:else}
                        <span class="group-empty">—</span>
                      {/if}
                    </a>
                  </td>
                  <td class="h-auto align-top row-open-cell">
                    <a
                      href={trainingRunDetailPath(run.run_id)}
                      class="cell-open-button overflow-visible! text-clip! whitespace-normal!"
                      onclick={(event) => selectRun(run.run_id, event)}
                    >
                      {#if groupTags?.tags.length}
                        <span class="flex flex-wrap gap-[4px] min-w-0 max-w-full">
                          {#each groupTags.tags as tag (tag.key)}
                            <span class="inline-flex items-baseline max-w-full min-w-0 overflow-hidden p-[2px_7px] [border:1px_solid_var(--border,#2f2f2f)] rounded-[999px] bg-[color-mix(in_srgb,var(--panel-alt)_70%,transparent)] text-(--text) [font-family:var(--font-mono)] text-[11px] leading-[14px]" title={`${tag.key}=${formatTagValue(tag.value)}`}>
                              <span class="min-w-0 overflow-hidden text-ellipsis whitespace-nowrap text-(--muted)">{tag.key}</span><span>=</span><span class="min-w-0 overflow-hidden text-ellipsis whitespace-nowrap">{formatTagValue(tag.value)}</span>
                            </span>
                          {/each}
                        </span>
                      {:else}
                        <span class="group-empty">—</span>
                      {/if}
                    </a>
                  </td>
                  <td class="whitespace-nowrap row-open-cell">
                    <a href={trainingRunDetailPath(run.run_id)} class="cell-open-button" onclick={(event) => selectRun(run.run_id, event)}>
                      <TimeAgo timestamp={run.created_at} showJustNow falsyRepresentation="—" />
                    </a>
                  </td>
                  <td class="whitespace-nowrap row-open-cell">
                    <a href={trainingRunDetailPath(run.run_id)} class="cell-open-button" onclick={(event) => selectRun(run.run_id, event)}>
                      <TimeAgo timestamp={run.updated_at} showJustNow falsyRepresentation="—" />
                    </a>
                  </td>
                  <td class="min-w-0 overflow-visible">
                    <div class="flex items-center flex-wrap gap-[6px]">
                      <button
                        class="inline-flex items-center gap-[6px] whitespace-nowrap [border:1px_solid_var(--color-c-gray-10,#2f2f2f)] rounded-[6px] p-[4px_8px] [font:inherit] text-[12px] font-medium leading-[16px] text-(--muted) bg-transparent cursor-pointer ghost-hover"
                        title="Open expanded view"
                        aria-label={`Open expanded view for training run ${run.run_id}`}
                        onclick={(event) => {
                          event.stopPropagation();
                          selectRun(run.run_id, event);
                        }}
                      >
                        <Maximize2 size={12} strokeWidth={2.1} />
                        <span class="expand-button-label">Expand</span>
                      </button>
                      {#if run.modal_app_url}
                        <a
                          class="training-open-modal-link"
                          href={run.modal_app_url}
                          target="_blank"
                          rel="noopener noreferrer"
                          onclick={(event) => event.stopPropagation()}
                        >
                          <span class="open-modal-link-label">Open in Modal</span>
                          <ExternalLink class="training-open-modal-link-icon" size={12} strokeWidth={2.1} />
                        </a>
                      {:else}
                        <span class="training-open-modal-link open-modal-link-disabled">
                          <span class="open-modal-link-label">Open in Modal</span>
                          <ExternalLink class="training-open-modal-link-icon" size={12} strokeWidth={2.1} />
                        </span>
                      {/if}
                      {#each metricLinksForRun(run) as link (link.url)}
                        <a
                          class="training-open-modal-link training-open-metric-link"
                          href={link.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          onclick={(event) => event.stopPropagation()}
                        >
                          <span class="open-modal-link-label">{link.label}</span>
                          <ExternalLink class="training-open-modal-link-icon" size={12} strokeWidth={2.1} />
                        </a>
                      {/each}
                    </div>
                  </td>
                </tr>
              {/each}
            </tbody>
          </ResizableTable>
        </div>
      {/snippet}

      {#if groupBy === "none"}
        {@render runsTable(filteredRuns)}
      {:else}
        <div class="flex flex-col gap-[24px] p-0">
          {#each runGroups as group (group.key)}
            <GroupSection
              title={group.key}
              subtitle={`${group.runs.length} run${group.runs.length === 1 ? "" : "s"}`}
              expanded={!collapsedGroupKeys.has(group.key)}
              onToggle={() => toggleGroupSection(group.key)}
            >
              {#snippet meta()}
                {#if group.latestCreatedAt}
                  <span class="group-meta-pill [font-variant-numeric:tabular-nums]">
                    <TimeAgo timestamp={group.latestCreatedAt} showJustNow falsyRepresentation="—" />
                  </span>
                {/if}
              {/snippet}
              {@render runsTable(group.runs, "360px")}
            </GroupSection>
          {/each}
        </div>
      {/if}

      <div class="page-empty flex flex-col items-center gap-[10px]">
        <span>Showing {filteredRuns.length} of {matchingRunCount} runs</span>
        {#if hasMoreRuns}
          <button type="button" class="load-more-button" onclick={onLoadMore}>
            Load more
          </button>
        {/if}
      </div>
    {/if}
  </div>
</section>

{#if selectedRun}
  <Drawer open={!!selectedRun} onclose={closeDrawer} width={drawerWidth}>
    <div
      class="w-full h-full max-h-[100vh]"
      aria-label={`Training run ${selectedRun.run_id}`}
    >
      <div class="drawer-panel-header">
        <div class="min-w-0 overflow-hidden">
          <div class="drawer-panel-eyebrow">Training run</div>
          <h2 class="text-(--text-bright) text-[16px] font-medium [font-family:var(--font-mono)] leading-[24px] whitespace-nowrap overflow-hidden text-ellipsis" title={selectedRun.run_id}>{selectedRun.run_id}</h2>
        </div>
        <div class="flex items-center gap-[8px] shrink-0">
          <button
            class="inline-flex items-center gap-[6px] [border:1px_solid_var(--color-c-gray-10,#2f2f2f)] rounded-[6px] p-[4px_8px] [font:inherit] text-[12px] font-medium leading-[16px] text-(--muted) bg-transparent cursor-pointer ghost-hover"
            onclick={(event) => selectRun(selectedRun.run_id, event)}
            title="Expand to full view"
          >
            <Maximize2 size={12} />
            <span>Expand</span>
          </button>
          {#if selectedRun.modal_app_url}
            <a
              class="inline-flex items-center gap-[6px] [border:1px_solid_var(--color-c-gray-10,#2f2f2f)] rounded-[6px] p-[4px_8px] no-underline text-(--muted) text-[12px] font-medium leading-[16px] ghost-hover"
              href={selectedRun.modal_app_url}
              target="_blank"
              rel="noopener noreferrer"
            >
              <span>View in app</span>
              <ExternalLink size={12} />
            </a>
          {/if}
          <button class="drawer-panel-close ghost-hover" onclick={closeDrawer} aria-label="Close run drawer">
            <PanelRightClose size={16} />
          </button>
        </div>
      </div>

      <div class="p-[4px_20px_16px]">
        <RunSummary
          run={selectedRun}
          {getStatus}
          {showFrameworkStatus}
          {modelName}
          {fmtDuration}
        />
      </div>
    </div>
  </Drawer>
{/if}
