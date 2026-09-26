<script>
  import { onDestroy } from "svelte";
  import { Search } from "lucide-svelte";
  import StatusPill from "./StatusPill.svelte";
  import FrameworkStageProgress from "./FrameworkStageProgress.svelte";
  import TimeAgo from "./TimeAgo.svelte";
  import { formatTagValue, getGroupTags } from "../lib/format.js";
  import { normalizeMetricLinks } from "../lib/metricLinks.js";

  // The run-summary block shared by the list drawer and the detail page's
  // Summary tab, so both render identical metadata: status, stage, model,
  // dataset, recipe, timing, the full Slime parameter dump, and the tuned
  // recipe fields.
  let { run, getStatus, showFrameworkStatus, modelName, fmtDuration } = $props();

  let recipe = $derived.by(() => run?.config?.recipe || run?.config?.preset || {});
  let recipeEntries = $derived.by(() =>
    Object.entries(recipe).filter(
      ([, value]) => value !== undefined && value !== null && String(value) !== "",
    ),
  );
  let recipeParams = $derived.by(() =>
    recipeEntries.map(([key, value]) => {
      let fullValue;
      if (typeof value === "number") {
        fullValue = Number.isInteger(value) ? String(value) : value.toExponential(1);
      } else if (typeof value === "object") {
        fullValue = JSON.stringify(value);
      } else {
        fullValue = String(value);
      }
      return {
        key,
        label: key.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase()),
        fullValue,
        displayValue: fullValue.length <= 72 ? fullValue : `${fullValue.slice(0, 71)}…`,
      };
    }),
  );
  let recipeJson = $derived.by(() =>
    recipeEntries.length ? JSON.stringify(Object.fromEntries(recipeEntries), null, 2) : "",
  );
  let paramFilter = $state("");
  let copiedRecipeJson = $state(false);
  let copyFailed = $state(false);
  let copyResetTimer = null;

  $effect(() => {
    run?.run_id;
    paramFilter = "";
    copiedRecipeJson = false;
    copyFailed = false;
    if (copyResetTimer) {
      clearTimeout(copyResetTimer);
      copyResetTimer = null;
    }
  });
  let filteredRecipeParams = $derived.by(() => {
    const q = paramFilter.trim().toLowerCase();
    if (!q) return recipeParams;
    return recipeParams.filter(
      (row) =>
        row.key.toLowerCase().includes(q) ||
        row.label.toLowerCase().includes(q) ||
        row.fullValue.toLowerCase().includes(q),
    );
  });
  let showParamFilter = $derived(recipeParams.length > 8 || Boolean(paramFilter.trim()));
  let recipeSectionTitle = $derived(
    String(run?.framework || "").toLowerCase() === "slime" ? "Slime parameters" : "Training recipe",
  );
  let modalAppUrl = $derived.by(() =>
    run?.modal_app_url ||
    (run?.modal_app_id ? `https://modal.com/id/${run.modal_app_id}` : ""),
  );
  let groupTags = $derived(getGroupTags(run));
  let metricLinks = $derived(
    normalizeMetricLinks(
      run?.metric_links?.length ? run.metric_links : run?.wandb_links,
    ),
  );
  let attemptMetadata = $derived.by(() => {
    const state = run?.resume_state;
    if (!state) return null;
    return {
      attemptCount: Number(state.attempt_count) || 0,
      lastAttemptStartedAt: Number(state.last_attempt_started_at) || 0,
      lastAttemptStatus: String(state.last_attempt_status || ""),
      resumedFromCheckpoint: state.resumed_from_checkpoint === true,
      resumeCheckpointPath: String(state.resume_checkpoint_path || ""),
      resumeCheckpointName: String(state.resume_checkpoint_name || ""),
      resumeFromIteration:
        state.resume_from_iteration == null ? null : Number(state.resume_from_iteration),
    };
  });

  function frameworkProgress() {
    const p = run?.framework_progress;
    if (!p || typeof p !== "object") return null;
    const current = Number(p.current);
    const total = Number(p.total);
    if (!Number.isFinite(current) || !Number.isFinite(total) || total <= 0) return null;
    return {
      current: Math.max(0, Math.min(current, total)),
      total,
      unit: p.unit || "step",
    };
  }

  function progressLabel(progress) {
    if (!progress) return "";
    const unit = String(progress.unit || "step");
    const label = unit.charAt(0).toUpperCase() + unit.slice(1);
    return `${label} ${progress.current} / ${progress.total}`;
  }

  function runDuration() {
    if (!run) return "—";
    if (typeof run.duration_seconds === "number" && run.duration_seconds >= 0) {
      return fmtDuration(0, run.duration_seconds);
    }
    if (run.started_at) return fmtDuration(run.started_at, run.ended_at);
    return "—";
  }

  async function copyRecipeJson() {
    if (!recipeJson) return;
    try {
      await navigator.clipboard.writeText(recipeJson);
      copiedRecipeJson = true;
      copyFailed = false;
    } catch {
      copiedRecipeJson = false;
      copyFailed = true;
    }
    if (copyResetTimer) clearTimeout(copyResetTimer);
    copyResetTimer = setTimeout(() => {
      copiedRecipeJson = false;
      copyFailed = false;
      copyResetTimer = null;
    }, 1200);
  }

  onDestroy(() => {
    if (copyResetTimer) clearTimeout(copyResetTimer);
  });
</script>

{#if run}
  <div class="run-summary">
    <section class="summary-section">
      <div class="kv">
        <span class="kv-key">Status</span>
        <StatusPill status={getStatus(run)} />
      </div>
      {#if showFrameworkStatus(run) && run.display_stage}
        {@const progress = frameworkProgress()}
        <div class="kv">
          <span class="kv-key">Stage</span>
          <FrameworkStageProgress
            {progress}
            progressLabel={progressLabel(progress)}
            stageLabel={run.display_stage}
            active={getStatus(run).toLowerCase() === "pending"}
          />
        </div>
      {/if}
      <div class="kv">
        <span class="kv-key">Model</span>
        <span class="kv-value">{modelName(run)}</span>
      </div>
      <div class="kv">
        <span class="kv-key">Dataset</span>
        <span class="kv-value">{run.dataset || "—"}</span>
      </div>
      <div class="kv">
        <span class="kv-key">Recipe</span>
        <span class="kv-value">{run.recipe || "—"}</span>
      </div>
      <div class="kv">
        <span class="kv-key">Training type</span>
        <span class="kv-value">{(run.training_type ?? "rl").toUpperCase()}</span>
      </div>
      {#if modalAppUrl}
        <div class="kv">
          <span class="kv-key">Modal app</span>
          <a
            class="text-(--accent) [overflow-wrap:anywhere] [text-decoration:none] hover:[text-decoration:underline] kv-value-mono"
            href={modalAppUrl}
            target="_blank"
            rel="noopener noreferrer"
            title={run.modal_app_id || modalAppUrl}
          >
            {run.modal_app_id || modalAppUrl}
          </a>
        </div>
      {/if}
      <div class="kv">
        <span class="kv-key">Duration</span>
        <span class="kv-value">{runDuration()}</span>
      </div>
      <div class="kv">
        <span class="kv-key">Started</span>
        <span class="kv-value">
          <TimeAgo timestamp={run.started_at || run.created_at} showJustNow />
        </span>
      </div>
      <div class="kv">
        <span class="kv-key">Last updated</span>
        <span class="kv-value">
          <TimeAgo timestamp={run.updated_at} showJustNow falsyRepresentation="—" />
        </span>
      </div>
    </section>

    {#if attemptMetadata}
      <section class="summary-section">
        <h3 class="summary-section-title">Retry / Resume</h3>
        {#if attemptMetadata.attemptCount}
          <div class="kv">
            <span class="kv-key">Attempts</span>
            <span class="kv-value">{attemptMetadata.attemptCount}</span>
          </div>
        {/if}
        {#if attemptMetadata.lastAttemptStartedAt}
          <div class="kv">
            <span class="kv-key">Latest attempt</span>
            <span class="kv-value">
              <TimeAgo timestamp={attemptMetadata.lastAttemptStartedAt} showJustNow />
            </span>
          </div>
        {/if}
        {#if attemptMetadata.lastAttemptStatus}
          <div class="kv">
            <span class="kv-key">Attempt status</span>
            <span class="kv-value">{attemptMetadata.lastAttemptStatus}</span>
          </div>
        {/if}
        <div class="kv">
          <span class="kv-key">Resumed</span>
          <span class="kv-value">{attemptMetadata.resumedFromCheckpoint ? "yes" : "no"}</span>
        </div>
        {#if attemptMetadata.resumeCheckpointPath}
          <div class="kv">
            <span class="kv-key">Checkpoint</span>
            <span class="kv-value kv-value-mono" title={attemptMetadata.resumeCheckpointPath}>
              {attemptMetadata.resumeCheckpointName || attemptMetadata.resumeCheckpointPath}
            </span>
          </div>
        {/if}
        {#if attemptMetadata.resumeFromIteration !== null}
          <div class="kv">
            <span class="kv-key">Resume step</span>
            <span class="kv-value">{attemptMetadata.resumeFromIteration}</span>
          </div>
        {/if}
      </section>
    {/if}

    {#if metricLinks.length}
      <section class="summary-section">
        <h3 class="summary-section-title">Metric</h3>
        <div class="flex flex-wrap gap-[6px]">
          {#each metricLinks as link (link.url)}
            <a
              class="[border:1px_solid_var(--color-c-gray-10,#2f2f2f)] rounded-[999px] text-(--accent) text-[12px] leading-[16px] p-[2px_8px] [text-decoration:none] hover:[text-decoration:underline] [border-color:color-mix(in_srgb,var(--yellow,#fbbf24)_45%,transparent)] text-(--yellow,#fbbf24)!"
              href={link.url}
              target="_blank"
              rel="noopener noreferrer"
              title={link.run_id || link.url}
            >
              {link.label}
            </a>
          {/each}
        </div>
      </section>
    {/if}

    {#if groupTags}
      <section class="summary-section">
        <h3 class="summary-section-title">Group</h3>
        <div class="kv">
          <span class="kv-key">Group ID</span>
          <span class="kv-value kv-value-mono">{groupTags.group_id || "—"}</span>
        </div>
        {#if groupTags.axes.length}
          <div class="kv">
            <span class="kv-key">Customized params</span>
            <div class="flex flex-wrap gap-[6px] min-w-0">
              {#each groupTags.axes as axis (axis)}
                <span class="[border:1px_solid_var(--color-c-gray-10,#2f2f2f)] rounded-[999px] text-(--text) bg-[color-mix(in_srgb,var(--panel-alt)_74%,black)] p-[2px_8px] kv-value-mono">{axis}</span>
              {/each}
            </div>
          </div>
        {/if}
        {#if groupTags.tags.length}
          <div class="kv items-start!">
            <span class="kv-key">This run differs by</span>
            <div class="grid gap-[6px] min-w-0">
              {#each groupTags.tags as tag (tag.key)}
                <div class="grid grid-cols-[minmax(0,1fr)_max-content] gap-[8px] items-baseline min-w-0">
                  <span class="text-(--muted) [overflow-wrap:anywhere] kv-value-mono">{tag.key}</span>
                  <span class="text-(--text) text-[12px] leading-[16px] [overflow-wrap:anywhere]">{formatTagValue(tag.value)}</span>
                </div>
              {/each}
            </div>
          </div>
        {/if}
      </section>
    {/if}

    <section class="summary-section">
      <div class="recipe-params-header">
        <h3 class="summary-section-title recipe-params-title">{recipeSectionTitle}</h3>
        {#if recipeParams.length}
          <span class="recipe-params-count">{recipeParams.length}</span>
          {#if recipeJson}
            <button
              type="button"
              class="recipe-params-copy"
              onclick={copyRecipeJson}
              title="Copy recipe JSON"
              aria-label="Copy recipe JSON"
            >
              {#if copiedRecipeJson}
                Copied
              {:else if copyFailed}
                Copy failed
              {:else}
                Copy JSON
              {/if}
            </button>
          {/if}
        {/if}
      </div>
      {#if recipeParams.length}
        {#if showParamFilter}
          <label
            class="recipe-params-search"
            aria-label="Filter recipe parameters"
          >
            <span class="search-icon">
              <Search size={13} />
            </span>
            <input
              type="search"
              class="search-input"
              placeholder="Filter parameters"
              bind:value={paramFilter}
            />
          </label>
        {/if}
        {#if filteredRecipeParams.length}
          <div class="recipe-params-list">
            {#each filteredRecipeParams as row (row.key)}
              <div class="kv">
                <span class="kv-key" title={row.key}>{row.label}</span>
                <span class="kv-value kv-value-mono" title={row.fullValue}>{row.displayValue}</span>
              </div>
            {/each}
          </div>
        {:else}
          <div class="text-(--muted) text-[12px] leading-[16px]">No parameters match "{paramFilter.trim()}".</div>
        {/if}
      {:else}
        <div class="text-(--muted) text-[12px] leading-[16px]">No recipe values found for this run.</div>
      {/if}
    </section>
  </div>
{/if}
