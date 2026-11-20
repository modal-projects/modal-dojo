<script>
  import { untrack } from "svelte";
  import ComponentFrame from "./ComponentFrame.svelte";
  import { MANIFEST_POLL_MS, componentsUrl, manifestsKey } from "../lib/componentFrame.js";

  /**
   * Every run-scoped component of one panel type (`summary_panel`,
   * `rollout_panel`) attached to `run`, in attachment order, each in its own
   * sandboxed frame. Renders nothing when none are attached.
   *
   * `componentProps` goes to every panel; it is re-sent only when `propsKey`
   * changes (see `ComponentFrame`).
   */
  let { run = null, type, componentProps = {}, propsKey = null } = $props();

  let manifests = $state([]);
  let errors = $state({});
  let loading = false;

  const activeRunId = $derived(run?.training_run_id || run?.run_id || "");
  // Attachments live in run metadata, so a change there triggers an immediate
  // reload; the interval covers attachments made after the page stopped
  // polling the run.
  const attachmentKey = $derived(JSON.stringify(run?.metadata?.dashboard_components || ""));

  async function load(id) {
    if (!id) {
      manifests = [];
      return;
    }
    if (loading) return;
    loading = true;
    try {
      const response = await fetch(componentsUrl(id, type), { credentials: "same-origin" });
      if (!response.ok || id !== activeRunId) return;
      const next = await response.json();
      if (!Array.isArray(next)) return;
      // Keep the current array (and so the mounted frames) unless an
      // attachment actually changed.
      if (manifestsKey(next) !== manifestsKey(manifests)) {
        manifests = next;
        errors = {};
      } else if (Object.values(errors).some((error) => !error.deterministic)) {
        // Retry panels that failed transiently (network, 5xx); a source that
        // does not compile fails the same way until it is replaced.
        errors = Object.fromEntries(
          Object.entries(errors).filter(([, error]) => error.deterministic),
        );
      }
    } catch {
      // A failed poll keeps what is mounted; the next one retries.
    } finally {
      loading = false;
    }
  }

  function onstatus(name, status, detail) {
    if (status === "error") {
      errors = {
        ...errors,
        [name]: {
          message: detail?.message || "component failed",
          deterministic: Boolean(detail?.deterministic),
        },
      };
    } else if (status === "rendered" && errors[name]) {
      const { [name]: _cleared, ...rest } = errors;
      errors = rest;
    }
  }

  $effect(() => {
    const id = activeRunId;
    attachmentKey;
    untrack(() => load(id));
  });
  $effect(() => {
    const timer = setInterval(() => load(activeRunId), MANIFEST_POLL_MS);
    return () => clearInterval(timer);
  });
</script>

{#each manifests as manifest (`${manifest.name}:${manifest.sha256}`)}
  <section class="dashboard-panel" aria-label={manifest.name}>
    <div class="rollout-sample-label" title="Run-scoped component">{manifest.name}</div>
    {#if errors[manifest.name]}
      <div class="panel-error" title={errors[manifest.name].message}>
        {manifest.name} failed: {errors[manifest.name].message}
      </div>
    {:else}
      <ComponentFrame
        src={manifest.frame_url}
        {componentProps}
        {propsKey}
        title={`Run-scoped ${type.replace("_", " ")}: ${manifest.name}`}
        onstatus={(status, detail) => onstatus(manifest.name, status, detail)}
      />
    {/if}
  </section>
{/each}

<style>
  .dashboard-panel { margin: 0 0 12px; }
  .panel-error { color: var(--red, #f87171); font-size: 11px; overflow-wrap: anywhere; }
</style>
