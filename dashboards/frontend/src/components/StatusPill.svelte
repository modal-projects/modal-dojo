<script>
  import { CheckCircle2, CircleX, Clock, Loader2, OctagonX, MinusCircle } from "lucide-svelte";

  let { status, iconOnly = false, label = null } = $props();

  const STATUS_MAP = {
    completed: "Completed",
    ready: "Ready",
    pending: "Pending",
    running: "Pending",
    queued: "Queued",
    stopped: "Stopped",
    cancelled: "Cancelled",
    failed: "Failed",
    inactive: "Inactive",
  };

  let normalizedStatus = $derived.by(() => {
    const s = String(status || "").toLowerCase();
    if (s === "running") return "pending";
    return s in STATUS_MAP ? s : "pending";
  });

  let statusLabel = $derived(label ?? (STATUS_MAP[normalizedStatus] ?? "Pending"));
</script>

<div
  class="status-pill"
  class:icon-only={iconOnly}
  class:status-completed={normalizedStatus === "completed"}
  class:status-ready={normalizedStatus === "ready"}
  class:status-running={normalizedStatus === "running"}
  class:status-pending={normalizedStatus === "pending"}
  class:status-queued={normalizedStatus === "queued"}
  class:status-stopped={normalizedStatus === "stopped"}
  class:status-cancelled={normalizedStatus === "cancelled"}
  class:status-failed={normalizedStatus === "failed"}
  class:status-inactive={normalizedStatus === "inactive"}
  aria-label={statusLabel}
>
  {#if normalizedStatus === "completed" || normalizedStatus === "ready"}
    <CheckCircle2 size={14} />
  {:else if normalizedStatus === "stopped" || normalizedStatus === "cancelled"}
    <OctagonX size={14} />
  {:else if normalizedStatus === "failed" || normalizedStatus === "inactive"}
    <CircleX size={14} />
  {:else if normalizedStatus === "queued"}
    <Clock size={14} />
  {:else}
    <span class="inline-flex items-center justify-center w-[16px] h-[16px] flex-[0_0_16px] [animation:status-pill-spin_1s_linear_infinite]">
      <Loader2 size={16} class="block" />
    </span>
  {/if}
  {#if !iconOnly}
    <span class="font-medium [font-variant-numeric:tabular-nums]">{statusLabel}</span>
  {/if}
</div>
