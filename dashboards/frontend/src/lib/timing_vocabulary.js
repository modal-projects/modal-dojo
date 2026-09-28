const slot = (name) => `var(--color-c-dataviz-${name})`;

export const TRAIN_OUTLINE_COLOR = slot("train-outline");

export const CATEGORIES = {
  train: {
    label: "Train",
    color: slot("primary-1"),
    owner: "train_models",
    phases: [
      "train_models",
      "compute_log_probs",
      "forward_backward",
      "optimizer_step",
      "trainer_finalize",
      "train_step_finalize",
    ],
  },
  generate: {
    label: "Rollout",
    color: slot("primary-3"),
    owner: "generate_rollouts",
    phases: [
      "generate_rollouts",
      "generate_samples",
      "sample_generation",
      "reward",
      "reward_batch",
      "reward_post_process",
    ],
  },
  transfer: {
    label: "Weight sync",
    color: slot("primary-2"),
    phases: [
      "weight_sync",
      "initial_weight_sync",
      "offload_train",
      "offload_rollout",
    ],
  },
  checkpoint: {
    label: "Checkpoint",
    color: slot("primary-5"),
    phases: ["checkpoint_save"],
  },
  eval: {
    label: "Eval",
    color: slot("primary-4"),
    phases: ["evaluate_rollouts", "evaluate_rollouts_end"],
  },
  idle: {
    label: "Idle",
    color: "var(--color-c-gray-30)",
    phases: ["wait_for_rollout", "wait_for_next_rollout"],
  },
};

const PHASE_CATEGORY = Object.fromEntries(
  Object.entries(CATEGORIES).flatMap(([category, { phases }]) =>
    phases.map((phase) => [phase, category]),
  ),
);

export const PHASE_COLORS = {
  compute_log_probs: slot("train-large"),
  forward_backward: slot("train-alt-a"),
  optimizer_step: slot("train-alt-b"),
  trainer_finalize: slot("train-alt-c"),
  train_step_finalize: slot("train-alt-d"),
};

export const TIMING_LABELS = {
  evaluate_rollouts: "Eval (before training)",
  evaluate_rollouts_end: "Eval (after training)",
  generate_rollouts: "Rollout generation",
  offload_rollout: "Offload rollout engines",
  compute_log_probs: "Calculate log probs",
  train_models: "Train",
  checkpoint_save: "Save checkpoint",
  offload_train: "Offload trainer",
  weight_sync: "Weight sync",
  initial_weight_sync: "Initial weight sync",
  wait_for_rollout: "Waiting for this rollout",
  wait_for_next_rollout: "Waiting for the next rollout",
  generate_samples: "Rollout generation",
  sample_generation: "Sample generation",
  reward: "Reward",
  reward_batch: "Reward (whole batch)",
  reward_post_process: "Reward post process",
  forward_backward: "Forward/backward",
  optimizer_step: "Optimizer step",
  // Labels stay role-neutral: the tooltip prefixes the role that recorded the
  // phase, and the actor and the critic record the same phase names.
  trainer_finalize: "Trainer cleanup",
  train_step_finalize: "Train-step cleanup & metrics",
};

const SFT_TIMING_LABELS = {
  generate_rollouts: "Preparing batch",
  generate_samples: "Preparing batch",
};

const SFT_CATEGORY_LABELS = {
  generate: "Preparing batch",
  transfer: "Offload",
};

export const IDLE_PHASES = new Set([
  "wait_for_rollout",
  "wait_for_next_rollout",
]);

export const SAMPLED = new Set(["reward", "reward_batch", "sample_generation"]);
export const HIDDEN_PHASES = new Set([
  "reward",
  "reward_batch",
  "reward_post_process",
  "sample_generation",
]);
export const TOOLTIP_HIDDEN_PHASES = new Set([
  "reward",
  "reward_batch",
  "sample_generation",
]);

export const NESTS_IN = {
  generate_samples: ["generate_rollouts"],
  compute_log_probs: ["train_models"],
  forward_backward: ["train_models"],
  optimizer_step: ["train_models"],
  trainer_finalize: ["train_models"],
  train_step_finalize: ["train_models"],
  reward: ["generate_samples"],
  reward_batch: ["generate_samples"],
  reward_post_process: ["generate_samples"],
  sample_generation: ["generate_samples"],
};

export const GROUPS = [
  {
    key: "timeline",
    label: "Timeline",
    hint: "measured phases on the shared wall clock",
  },
];

export const NEGLIGIBLE_WORK_S = 0.0005;
export const CROSS_LANE_CONTAINMENT_TOLERANCE_S = 0.01;
export function labelFor(name, rolloutId = null, trainingType = null) {
  if (trainingType === "sft" && SFT_TIMING_LABELS[name]) {
    return SFT_TIMING_LABELS[name];
  }
  if (
    (name === "wait_for_rollout" || name === "wait_for_next_rollout") &&
    rolloutId != null
  ) {
    const step = Number(rolloutId) + (name === "wait_for_next_rollout" ? 2 : 1);
    if (Number.isInteger(step)) {
      return `Waiting for rollout generation (step ${step})`;
    }
  }
  return TIMING_LABELS[name] || name.replace(/_/g, " ");
}

export function categoryLabelFor(key, trainingType = null) {
  if (trainingType === "sft" && key in SFT_CATEGORY_LABELS) {
    return SFT_CATEGORY_LABELS[key];
  }
  return CATEGORIES[key].label;
}

export function isLegacyTiming(timings) {
  return timings?.metadata?.legacy_derived === true;
}

export function rolloutIdForTimingKey(id) {
  if (id === "") return null;
  const parsedId = Number(id);
  return Number.isInteger(parsedId) && parsedId >= 0 ? parsedId : null;
}

export function shouldShowTimingSection(timings) {
  return (
    timings?.metadata?.timing_stale === true ||
    Object.entries(timings || {}).some(([id, value]) => {
      if (id === "metadata") return false;
      if (rolloutIdForTimingKey(id) !== null) return true;
      return (
        value &&
        typeof value === "object" &&
        value.roles &&
        typeof value.roles === "object"
      );
    })
  );
}

export function shouldShowOpenRolloutAction({
  showOpenRollout,
  onOpenRollout,
  bar,
  rolloutIds,
  rolloutId,
}) {
  return Boolean(
    showOpenRollout &&
    typeof onOpenRollout === "function" &&
    bar &&
    ["generate_samples", "generate_rollouts"].includes(bar.name) &&
    bar.kind !== "idle" &&
    rolloutId != null &&
    rolloutIds.includes(rolloutId),
  );
}

export function categoryOf(name) {
  return PHASE_CATEGORY[name] || "idle";
}

export function colorFor(name) {
  return PHASE_COLORS[name] || CATEGORIES[categoryOf(name)].color;
}

export function fmtSecs(s, unit = null) {
  if (s == null) return "—";
  const n = Number(s);
  if (!Number.isFinite(n)) return "—";
  const trim = (x) => x.toFixed(3).replace(/\.?0+$/, "");
  if (unit === "ms") return `${trim(n * 1000)}ms`;
  if (unit === "s") return `${trim(n)}s`;
  if (n > 0 && n < 0.01) return `${trim(n * 1000)}ms`;
  if (n >= 60) {
    const m = Math.floor(n / 60);
    return `${m}m ${trim(n - m * 60)}s`;
  }
  return `${trim(n)}s`;
}
