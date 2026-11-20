<script>
  // Add-run picker for the Metrics compare page: the training list's own
  // search bar and filters over the same server-filtered runs, with a
  // checkbox per run. Runs already being compared show as checked.
  import { X } from "lucide-svelte";
  import FilterBar from "./FilterBar.svelte";
  import StatusPill from "./StatusPill.svelte";
  import TimeAgo from "./TimeAgo.svelte";
  import { toggleInSet } from "../lib/set.js";

  let {
    open = false,
    // Props for FilterBar, shared with the Training runs page.
    filterProps = {},
    search = $bindable(""),
    runs = [],
    matchingRunCount = 0,
    hasMoreRuns = false,
    loading = false,
    getStatus = (run) => run.display_status || "pending",
    existingRunIds = [],
    onLoadMore = () => {},
    onAdd = () => {},
    onclose = () => {},
  } = $props();

  let picked = $state(new Set());
  let existing = $derived(new Set(existingRunIds));

  $effect(() => {
    if (!open) return;
    picked = new Set();
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = previous;
    };
  });

  function add() {
    onAdd([...picked]);
    onclose();
  }

  function growOnScroll(event) {
    const el = event.currentTarget;
    if (hasMoreRuns && el.scrollTop + el.clientHeight >= el.scrollHeight - 300) onLoadMore();
  }
</script>

<svelte:window
  onkeydown={(event) => {
    // FilterBar's own Escape closes its menus first; a second press closes us.
    if (open && event.key === "Escape" && !event.target.closest?.(".filter-row")) onclose();
  }}
/>

{#if open}
  <div class="fixed inset-0 z-[40] flex items-start justify-center p-[6vh_16px_16px] [background:rgba(0,0,0,0.5)]">
    <button type="button" class="absolute inset-0 [border:0] [background:transparent] cursor-default" onclick={onclose} aria-label="Close"></button>
    <div
      class="add-runs-modal relative z-[1] flex flex-col w-[min(880px,100%)] max-h-[88vh] bg-(--bg-depth) [border:1px_solid_var(--border-strong)] rounded-[8px] [box-shadow:0_24px_64px_rgba(0,0,0,0.5)]"
      role="dialog"
      aria-modal="true"
      aria-label="Add runs to compare"
    >
      <div class="flex items-center justify-between gap-[12px] p-[14px_16px_10px]">
        <h2 class="text-(--text-bright) text-[16px] font-medium">Add runs to compare</h2>
        <button type="button" class="drawer-panel-close ghost-hover" onclick={onclose} aria-label="Close">
          <X size={16} />
        </button>
      </div>

      <div class="p-[0_16px_12px]">
        <FilterBar {...filterProps} bind:search showGroupBy={false} />
      </div>

      <div class="flex-1 min-h-[160px] overflow-y-auto [border-top:1px_solid_var(--border)] [border-bottom:1px_solid_var(--border)]" onscroll={growOnScroll}>
        {#if loading && !runs.length}
          <div class="page-empty">Loading runs…</div>
        {:else if !runs.length}
          <div class="page-empty">No runs match the current filters.</div>
        {:else}
          <ul class="list-none m-0 p-[4px]">
            {#each runs as run (run.run_id)}
              {@const already = existing.has(run.run_id)}
              <li>
                <label class="add-run-row" class:opacity-50={already}>
                  <input
                    type="checkbox"
                    class="w-[14px] h-[14px] m-0 shrink-0 [accent-color:var(--accent)]"
                    checked={already || picked.has(run.run_id)}
                    disabled={already}
                    onchange={() => (picked = toggleInSet(picked, run.run_id))}
                  />
                  <span class="[font-family:var(--font-mono)] text-(--text-bright) truncate min-w-0 flex-[1_1_40%]" title={run.run_id}>{run.run_id}</span>
                  <span class="shrink-0"><StatusPill status={getStatus(run)} /></span>
                  <span class="text-(--muted) truncate min-w-0 flex-[1_1_25%] max-[640px]:hidden" title={run.model}>{run.model || "—"}</span>
                  <span class="text-(--muted) truncate min-w-0 flex-[1_1_15%] max-[640px]:hidden" title={run.dataset}>{run.dataset || "—"}</span>
                  <span class="text-(--muted) whitespace-nowrap shrink-0 w-[118px] text-right max-[640px]:hidden">
                    <TimeAgo timestamp={run.created_at} showJustNow falsyRepresentation="—" />
                  </span>
                </label>
              </li>
            {/each}
          </ul>
          <div class="flex items-center justify-center gap-[10px] p-[8px] text-(--muted) text-[12px]">
            <span>Showing {runs.length} of {matchingRunCount} runs</span>
            {#if hasMoreRuns}
              <button type="button" class="load-more-button" onclick={onLoadMore}>Load more</button>
            {/if}
          </div>
        {/if}
      </div>

      <div class="flex items-center justify-end gap-[8px] p-[12px_16px]">
        <button type="button" class="log-button" onclick={onclose}>Cancel</button>
        <button type="button" class="compare-bar-primary disabled:opacity-50 disabled:cursor-default" disabled={!picked.size} onclick={add}>
          Add {picked.size || ""} run{picked.size === 1 ? "" : "s"}
        </button>
      </div>
    </div>
  </div>
{/if}
