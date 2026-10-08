// Sorting, filtering and formatting for the Autoconfig catalogue, whose rows
// come from GET /api/autoconfig/catalogue (modal_dojo/common/catalogue.py).

// Each sort opens in the direction a reader most likely wants first:
// cheapest and fastest, or highest reward.
export const SORTS = [
  { key: "cost", direction: "asc" },
  { key: "time", direction: "asc" },
  { key: "initial", direction: "desc" },
];

export function sortLabel(key) {
  switch (key) {
    case "cost":
      return "Cost / step";
    case "time":
      return "Time / step";
    case "initial":
      return "Step 1 reward";
    default:
      return key;
  }
}

export function sortValue(row, key) {
  const run = row.run;
  if (!run) return null;
  switch (key) {
    case "cost":
      return run.cost_per_step_usd;
    case "time":
      return run.time_per_step_s;
    case "initial":
      return run.initial_reward;
    default:
      return null;
  }
}

/** The base model's name without its Hugging Face organization. */
export function baseModel(row) {
  return row.model.slice(row.model.lastIndexOf("/") + 1);
}

// Context length slider stops. The end stops are open-ended: the first means
// no minimum and the last no maximum, so the full range filters nothing.
export const CONTEXT_STOPS = [
  4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576,
];

/** Whether a recipe's context length falls between two slider stops. */
export function inContextRange(context, low, high) {
  const last = CONTEXT_STOPS.length - 1;
  if (low === 0 && high === last) return true;
  if (context === null) return false;
  return (
    (low === 0 || context >= CONTEXT_STOPS[low]) &&
    (high === last || context <= CONTEXT_STOPS[high])
  );
}

export function contextRangeLabel(low, high) {
  const last = CONTEXT_STOPS.length - 1;
  const lowLabel = formatTokens(CONTEXT_STOPS[low]);
  const highLabel = formatTokens(CONTEXT_STOPS[high]);
  if (low === 0 && high === last) return "Any length";
  if (low === 0) return `Up to ${highLabel}`;
  if (high === last) return `${lowLabel} and up`;
  return low === high ? lowLabel : `${lowLabel}–${highLabel}`;
}

/** Order two sort values; a missing value sorts last in either direction. */
export function compareValues(left, right, direction) {
  if (left === null || right === null) {
    if (left === right) return 0;
    return left === null ? 1 : -1;
  }
  return direction === "asc" ? left - right : right - left;
}

/** Sort rows by one key, keeping catalogue order between equal values. */
export function sortRows(rows, key, direction) {
  return rows
    .map((row, index) => ({ row, index }))
    .sort(
      (left, right) =>
        compareValues(sortValue(left.row, key), sortValue(right.row, key), direction) ||
        left.index - right.index,
    )
    .map(({ row }) => row);
}

const usd = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" });
const date = new Intl.DateTimeFormat("en-US", {
  month: "short",
  day: "numeric",
  year: "numeric",
  timeZone: "UTC",
});
const MINUS = "−";

export function formatUsd(value) {
  return value === null || value === undefined ? "—" : usd.format(value);
}

export function formatDuration(seconds) {
  if (seconds === null || seconds === undefined) return "—";
  const total = Math.round(seconds);
  if (total < 60) return `${total}s`;
  if (total < 3600) {
    return `${Math.floor(total / 60)}m ${String(total % 60).padStart(2, "0")}s`;
  }
  const minutes = Math.floor((total % 3600) / 60);
  return `${Math.floor(total / 3600)}h ${String(minutes).padStart(2, "0")}m`;
}

/** Token counts in binary units, the way context windows are quoted: 32K, 1M. */
export function formatTokens(tokens) {
  const mebi = 1024 * 1024;
  if (tokens >= mebi) {
    const millions = tokens / mebi;
    return `${Number.isInteger(millions) ? millions : millions.toFixed(1)}M`;
  }
  return tokens >= 1024 ? `${Math.round(tokens / 1024)}K` : String(tokens);
}

export function formatDate(unixSeconds) {
  return date.format(new Date(unixSeconds * 1000));
}

/**
 * Decimal places for a run's raw rewards: three when they stay within ±1, as
 * binary pass/fail rewards do, one for larger scales.
 */
export function rewardDigits(rewards) {
  return rewards.every((reward) => reward === null || Math.abs(reward) <= 1) ? 3 : 1;
}

export function formatReward(reward, digits) {
  return reward === null || reward === undefined ? "—" : reward.toFixed(digits);
}

/** Signed change between two rewards, or null when either is missing. */
export function formatRewardChange(from, to, digits) {
  if (from === null || to === null) return null;
  const change = to - from;
  const magnitude = Math.abs(change).toFixed(digits);
  if (Number(magnitude) === 0) return `±${magnitude}`;
  return `${change > 0 ? "+" : MINUS}${magnitude}`;
}

// ── Sweep form ────────────────────────────────────────────────────────────

/**
 * Parse the sweep grid textarea: one `path = v1, v2` line per axis, values as
 * JSON where they parse (numbers, booleans, quoted strings) else as text.
 * Throws with the offending line on malformed input.
 */
// Recipe fields every catalogue recipe accepts that are worth sweeping.
export const SWEEP_FLAGS = [
  { path: "recipe.lr", placeholder: "1e-6, 5e-6" },
  { path: "recipe.gpu_type", placeholder: "H200, B300" },
  { path: "recipe.actor_num_nodes", placeholder: "1, 2" },
  { path: "recipe.actor_num_gpus_per_node", placeholder: "4, 8" },
  { path: "recipe.tensor_model_parallel_size", placeholder: "1, 2" },
  { path: "recipe.context_parallel_size", placeholder: "1, 2" },
  { path: "recipe.max_tokens_per_gpu", placeholder: "8192, 16384" },
  { path: "recipe.weight_decay", placeholder: "0, 0.1" },
  { path: "recipe.global_batch_size", placeholder: "64, 128" },
  { path: "recipe.rollout_batch_size", placeholder: "16, 32" },
  { path: "recipe.n_samples_per_prompt", placeholder: "4, 8" },
  { path: "recipe.rollout_temperature", placeholder: "0.8, 1.0" },
  { path: "recipe.rollout_top_p", placeholder: "0.9, 1.0" },
  { path: "recipe.rollout_max_response_len", placeholder: "4096, 8192" },
  { path: "recipe.kl_loss_coef", placeholder: "0, 0.01" },
  { path: "recipe.entropy_coef", placeholder: "0, 0.001" },
  { path: "recipe.eps_clip", placeholder: "0.2, 0.28" },
  { path: "recipe.eps_clip_high", placeholder: "0.28, 0.32" },
];

// Training datasets the launcher offers; the sweep API takes the spec as-is.
export const SWEEP_DATASETS = [
  {
    label: "SWE-smith",
    spec: {
      hf_repo: "SWE-bench/SWE-smith",
      hf_split: "train",
      input_column: "problem_statement",
      output_column: "patch",
    },
  },
];

/** Build a sweep grid from `{ path, values }` rows; values are comma-separated. */
export function gridFromFlags(flags) {
  const grid = {};
  for (const { path, values } of flags) {
    if (!path) continue;
    const parsed = values
      .split(",")
      .map((v) => v.trim())
      .filter(Boolean)
      .map(parseGridValue);
    if (!parsed.length) throw new Error(`no values for "${path}"`);
    grid[path] = parsed;
  }
  return grid;
}

function parseGridValue(text) {
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

/** Build base overrides from `{ path, values }` rows; each takes one value. */
export function overridesFromFlags(flags) {
  const overrides = {};
  for (const [path, values] of Object.entries(gridFromFlags(flags))) {
    if (values.length !== 1) throw new Error(`one value expected for "${path}"`);
    overrides[path] = values[0];
  }
  return overrides;
}

export function sweepRunCount(entries, grid) {
  const variants = Object.values(grid).reduce((n, values) => n * values.length, 1);
  return entries.length * variants;
}
