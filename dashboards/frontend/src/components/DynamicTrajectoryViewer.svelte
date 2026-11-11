<script>
  import { untrack } from "svelte";
  import DefaultTrajectoryViewer from "./TrajectoryViewer.svelte";
  import ComponentFrame from "./ComponentFrame.svelte";
  import { MANIFEST_POLL_MS } from "../lib/componentFrame.js";

  /**
   * Renders the run-scoped trajectory viewer when one is attached to `run`,
   * otherwise the built-in `TrajectoryViewer`. The run-scoped viewer runs in
   * a sandboxed frame (see `ComponentFrame`).
   */
  let {
    sample = null,
    samples = [],
    trajectory = [],
    rewardEvents = [],
    rollout = null,
    run = null,
    position = null,
  } = $props();

  let frameSrc = $state("");
  let mode = $state("fallback"); // "fallback" | "loading" | "custom"
  let loadError = $state("");
  let loadedDigest = null;
  // Digests the server rejected deterministically (bad source / compile
  // error); polling skips these, but retries anything transient.
  const brokenDigests = new Set();
  let loading = false;
  let generation = 0;

  // Primitive derivations of `run`, so the load effect only reruns when the
  // id or attached components actually change, not on every run poll.
  const activeRunId = $derived(run?.training_run_id || run?.run_id || "");
  const componentKey = $derived(JSON.stringify(run?.metadata?.dashboard_components || ""));

  // Re-send props (which remounts the component) when the selection changes,
  // but not when only `run` was re-polled.
  const propsKey = $derived([sample, samples, trajectory, rewardEvents, rollout, position]);
  const componentProps = $derived.by(() => {
    propsKey;
    return untrack(() => ({ sample, samples, trajectory, rewardEvents, rollout, run, position }));
  });

  function fallback(message = "") {
    generation += 1;
    loadError = message;
    mode = "fallback";
    frameSrc = "";
    loadedDigest = null;
  }

  async function fetchManifest(base) {
    const response = await fetch(base, { credentials: "same-origin" });
    if (response.status === 404) return null;
    if (!response.ok) throw new Error(`component lookup failed (${response.status})`);
    const manifest = await response.json();
    const digest = manifest?.sha256;
    if (typeof digest !== "string" || !digest) return null;
    return { digest, frameUrl: manifest.frame_url };
  }

  // `force` tears down whatever is mounted; otherwise a manifest whose digest
  // matches the mounted component is a no-op so polling never flickers.
  async function load(force = true) {
    const id = activeRunId;
    if (force) {
      fallback();
      if (!id) return;
      mode = "loading";
    } else if (!id || loading) {
      return;
    }
    loading = true;
    const before = generation;
    const base = `/api/runs/${encodeURIComponent(id)}/dashboard-components/trajectory_viewer`;
    try {
      let found;
      try {
        found = await fetchManifest(base);
      } catch (error) {
        if (before !== generation || !force) return;
        fallback(error?.message || String(error));
        return;
      }
      if (before !== generation) return;
      if (!found) {
        if (force || loadedDigest) fallback();
        return;
      }
      if (!force && (found.digest === loadedDigest || brokenDigests.has(found.digest))) return;
      loadError = "";
      mode = "loading";
      loadedDigest = found.digest;
      frameSrc =
        found.frameUrl ||
        `${base}/${encodeURIComponent(found.digest)}/frame.html?bridge=modal-dojo-v1`;
    } finally {
      loading = false;
    }
  }

  function onstatus(status, detail) {
    if (status === "rendered") {
      mode = "custom";
    } else if (status === "error") {
      if (detail?.deterministic && loadedDigest) brokenDigests.add(loadedDigest);
      fallback(detail?.message || "run-scoped viewer failed");
    }
  }

  $effect(() => {
    activeRunId;
    componentKey;
    untrack(() => load(true));
  });
  $effect(() => {
    const timer = setInterval(() => load(false), MANIFEST_POLL_MS);
    return () => clearInterval(timer);
  });
</script>

{#if mode !== "custom"}
  <div class="rollout-sample-label">conversation</div>
  <DefaultTrajectoryViewer
    {sample}
    {samples}
    {trajectory}
    {rewardEvents}
    {rollout}
    {run}
  />
  {#if mode === "loading"}
    <div class="viewer-note">Loading run-scoped trajectory viewer…</div>
  {:else if loadError}
    <div class="viewer-note viewer-error" title={loadError}>Run-scoped viewer unavailable; showing the default viewer.</div>
  {/if}
{/if}
{#if mode === "custom"}
  <div class="rollout-sample-label">trajectory viewer</div>
{/if}
{#if frameSrc}
  <ComponentFrame
    src={frameSrc}
    {componentProps}
    {propsKey}
    title="Run-scoped trajectory viewer"
    {onstatus}
  />
{/if}

<style>
  .viewer-note { margin: -8px 0 10px; color: var(--muted, #8b8b8b); font-size: 10px; }
  .viewer-error { color: var(--red, #f87171); }
</style>
