<script>
  import { untrack } from "svelte";
  import DefaultTrajectoryViewer from "./TrajectoryViewer.svelte";

  /**
   * Renders the run-scoped trajectory viewer when one is attached to `run`,
   * otherwise the built-in `TrajectoryViewer`.
   *
   * Run-scoped components are untrusted code. They run inside a sandboxed
   * <iframe> served by `/dashboard-components/trajectory_viewer/<sha256>/frame.html`
   * (opaque origin, CSP without network access) and only receive props via
   * postMessage, so they can never act with the dashboard's authority.
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

  const MARK = "modalDojoDashboardComponent";
  const READY_TIMEOUT_MS = 20000;
  // Components can be attached after a run finishes, when the page has
  // stopped polling `run`; re-check the manifest so they still show up.
  const MANIFEST_POLL_MS = 30000;

  let frame = $state(null);
  let frameSrc = $state("");
  let frameHeight = $state(0);
  let mode = $state("fallback"); // "fallback" | "loading" | "custom"
  let loadError = $state("");
  let frameReady = false;
  let generation = 0;
  let readyTimer = null;
  let loadedDigest = null;
  // Digests the server rejected deterministically (bad source / compile
  // error); polling skips these, but retries anything transient.
  const brokenDigests = new Set();
  let loading = false;

  // Primitive derivations of `run`, so the load effect only reruns when the
  // id or attached components actually change, not on every run poll.
  const activeRunId = $derived(run?.training_run_id || run?.run_id || "");
  const componentKey = $derived(JSON.stringify(run?.metadata?.dashboard_components || ""));

  function reset() {
    generation += 1;
    frameReady = false;
    clearTimeout(readyTimer);
    readyTimer = null;
    frameSrc = "";
    frameHeight = 0;
    loadedDigest = null;
  }

  function fail(message) {
    loadError = message;
    mode = "fallback";
    reset();
  }

  function postProps() {
    if (!frameReady || !frame?.contentWindow) return;
    const props = $state.snapshot({ sample, samples, trajectory, rewardEvents, rollout, run, position });
    // The frame has an opaque origin, so "*" is the only valid target.
    frame.contentWindow.postMessage({ [MARK]: true, type: "props", props }, "*");
  }

  async function fetchManifest(base) {
    const response = await fetch(base, { credentials: "same-origin" });
    if (response.status === 404) return null;
    if (!response.ok) throw new Error(`component lookup failed (${response.status})`);
    const manifest = await response.json();
    const digest = manifest?.sha256;
    return typeof digest === "string" && digest ? digest : null;
  }

  // `force` tears down whatever is mounted; otherwise a manifest whose digest
  // matches the mounted component is a no-op so polling never flickers.
  async function load(force = true) {
    const id = activeRunId;
    if (force) {
      reset();
      loadError = "";
      if (!id) {
        mode = "fallback";
        return;
      }
      mode = "loading";
    } else if (!id || loading) {
      return;
    }
    loading = true;
    const before = generation;
    const base = `/api/runs/${encodeURIComponent(id)}/dashboard-components/trajectory_viewer`;
    try {
      let digest;
      try {
        digest = await fetchManifest(base);
      } catch (error) {
        if (before !== generation || !force) return;
        fail(error?.message || String(error));
        return;
      }
      if (before !== generation) return;
      if (!force) {
        if (digest === loadedDigest || brokenDigests.has(digest)) return;
        reset();
        loadError = "";
        mode = "loading";
      }
      const current = generation;
      if (!digest) {
        mode = "fallback";
        return;
      }
      const src = `${base}/${encodeURIComponent(digest)}/frame.html?bridge=modal-dojo-v1`;
      try {
        // Compile errors surface here as a readable status instead of a blank frame.
        const frameResponse = await fetch(src, { credentials: "same-origin" });
        if (current !== generation) return;
        if (!frameResponse.ok) {
          let detail = `component request failed (${frameResponse.status})`;
          try {
            detail = (await frameResponse.json())?.detail || detail;
          } catch {
            // Non-JSON error bodies keep the generic status message.
          }
          if ([409, 413, 422].includes(frameResponse.status)) brokenDigests.add(digest);
          throw new Error(detail);
        }
        frameSrc = src;
        loadedDigest = digest;
        readyTimer = setTimeout(() => {
          if (current !== generation || frameReady) return;
          fail("run-scoped viewer did not start");
        }, READY_TIMEOUT_MS);
      } catch (error) {
        if (current !== generation) return;
        fail(error?.message || String(error));
      }
    } finally {
      loading = false;
    }
  }

  function onMessage(event) {
    if (!frame || event.source !== frame.contentWindow) return;
    const data = event.data;
    if (!data || data[MARK] !== true) return;
    if (data.type === "ready") {
      frameReady = true;
      clearTimeout(readyTimer);
      readyTimer = null;
      postProps();
    } else if (data.type === "rendered") {
      mode = "custom";
    } else if (data.type === "resize") {
      const height = Number(data.height);
      if (Number.isFinite(height) && height >= 0) frameHeight = Math.ceil(height);
    } else if (data.type === "error") {
      fail(String(data.message || "run-scoped viewer failed"));
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

  $effect(() => {
    // Track the identity of the selection props so a new selection (or a
    // freshly fetched rollout) is re-sent; deep reads inside `postProps` are
    // untracked so `run` polling does not remount the component.
    sample;
    samples;
    trajectory;
    rewardEvents;
    rollout;
    position;
    untrack(postProps);
  });

  $effect(() => {
    window.addEventListener("message", onMessage);
    return () => {
      window.removeEventListener("message", onMessage);
      reset();
    };
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
  <iframe
    bind:this={frame}
    class="custom-viewer-frame"
    class:visible={mode === "custom"}
    src={frameSrc}
    sandbox="allow-scripts"
    referrerpolicy="no-referrer"
    title="Run-scoped trajectory viewer"
    style:height={mode === "custom" ? `${frameHeight}px` : "0px"}
  ></iframe>
{/if}

<style>
  .custom-viewer-frame { display: block; width: 100%; height: 0; border: 0; overflow: hidden; color-scheme: dark; }
  /* Keep the full width while hidden: the frame measures itself as soon as it
     loads, and a zero-width frame reflows its content into a far taller box
     than the one the viewer actually renders at. */
  .custom-viewer-frame:not(.visible) { height: 0; visibility: hidden; }
  .viewer-note { margin: -8px 0 10px; color: var(--muted, #8b8b8b); font-size: 10px; }
  .viewer-error { color: var(--red, #f87171); }
</style>
