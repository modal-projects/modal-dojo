<script>
  import { onMount } from "svelte";
  import { CircleStop } from "lucide-svelte";
  import { fetchMutationsAllowed, stopRun } from "../lib/api.js";

  let { run, onStopped, compact = false } = $props();

  let stopping = $state(false);
  let stopError = $state("");
  // Open dashboards refuse /stop; hide the button there rather than
  // presenting an action that always fails.
  let canStop = $state(true);
  const label = $derived(stopping ? "Stopping…" : "Stop run");

  onMount(async () => {
    canStop = await fetchMutationsAllowed();
  });

  async function confirmStop() {
    if (stopping || !run?.run_id) return;
    if (!window.confirm(`Stop training run ${run.run_id}? This stops its Modal app.`))
      return;
    stopping = true;
    stopError = "";
    try {
      const updated = await stopRun(run.run_id);
      onStopped?.(updated);
    } catch (err) {
      stopError = String(err?.message || err);
    } finally {
      stopping = false;
    }
  }
</script>

{#if canStop && run?.status === "running"}
  {#if stopError}
    <span class="stop-run-error" title={stopError}>Stop failed</span>
  {/if}
  <button
    class={compact
      ? "drawer-panel-close ghost-hover"
      : "inline-flex items-center gap-[6px] [border:1px_solid_var(--border,#2f2f2f)] rounded-[6px] [background:none] text-(--muted) cursor-pointer [font:inherit] text-[12px] font-medium leading-[16px] min-h-[32px] p-[4px_8px] hover:text-(--text-bright) hover:[border-color:#f87171]"}
    onclick={confirmStop}
    disabled={stopping}
    aria-label={label}
    title={label}
  >
    <span class="stop-icon">
      <CircleStop size={compact ? 16 : 12} strokeWidth={2.1} />
    </span>
    {#if !compact}
      <span class="max-[520px]:hidden">{label}</span>
    {/if}
  </button>
{/if}

<style>
  .stop-icon {
    display: inline-flex;
    color: #f87171;
  }
</style>
