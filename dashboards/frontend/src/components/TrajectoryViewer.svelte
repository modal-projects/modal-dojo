<script>
  /**
   * Modal Dojo's built-in trajectory viewer.
   *
   * This is the slot that `modal-dojo setup --trajectory-viewer` replaces
   * at build time, and the fallback when a run has no run-scoped viewer. It
   * renders the conversation exactly as the rollout page always has (tool
   * calls, thinking, eval reports) via
   * `ConversationView`; replacements receive the same props.
   *
   * Props supplied by TrainingRunDetailPage:
   *   sample       - selected normalized sample object
   *   samples      - all samples in the selected rollout/prompt group
   *   trajectory   - sample.metadata.trajectory_messages[]
   *   rewardEvents - sample.reward_events[] (if emitted)
   *   rollout      - full expanded TrainingRolloutResult
   *   run          - current TrainingRun summary
   */
  import ConversationView from "./ConversationView.svelte";

  let {
    sample = null,
    samples = [],
    trajectory = [],
    rewardEvents = [],
    rollout = null,
    run = null,
  } = $props();

  // `trajectory` is the page's copy of metadata.trajectory_messages; fall back
  // to the sample itself so the viewer still works when mounted with only a
  // sample (e.g. from a run-scoped bundle that omits the prop).
  let messages = $derived(
    Array.isArray(trajectory) && trajectory.length
      ? trajectory
      : sample?.metadata?.trajectory_messages ?? null,
  );
</script>

<ConversationView
  {messages}
  response={sample?.response || ""}
  thinking={sample?.thinking || ""}
  evalReport={sample?.metadata?.eval_report}
/>
