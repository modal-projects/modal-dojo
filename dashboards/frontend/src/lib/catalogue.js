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
export function parseGrid(text) {
  const grid = {};
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    const eq = line.indexOf("=");
    if (eq < 1) throw new Error(`expected "path = value, value" but got "${line}"`);
    const path = line.slice(0, eq).trim();
    const values = line
      .slice(eq + 1)
      .split(",")
      .map((v) => v.trim())
      .filter(Boolean)
      .map(parseGridValue);
    if (!values.length) throw new Error(`no values for "${path}"`);
    grid[path] = values;
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

export function sweepRunCount(entries, grid) {
  const variants = Object.values(grid).reduce((n, values) => n * values.length, 1);
  return entries.length * variants;
}
