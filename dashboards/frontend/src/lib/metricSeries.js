// Helpers for the Metrics tab: W&B-style grouping of metric keys by prefix.

// Keys without a `/` prefix land in the same panel group W&B uses for them.
export const UNGROUPED = "Charts";

export function metricAxisLabel(stepKey) {
  return ({ "train/step": "Training step", "rollout/step": "Rollout step", "eval/step": "Evaluation step" })[stepKey]
    ?? stepKey ?? "Logging step";
}

// `[{ name, keys }]`, filtered by a case-insensitive substring search,
// ungrouped keys first, then groups and keys alphabetically.
export function groupMetricKeys(keys, search = "") {
  const needle = search.trim().toLowerCase();
  const groups = new Map();
  for (const key of keys) {
    if (!key.toLowerCase().includes(needle)) continue;
    const slash = key.indexOf("/");
    const name = slash > 0 ? key.slice(0, slash) : UNGROUPED;
    if (!groups.has(name)) groups.set(name, []);
    groups.get(name).push(key);
  }
  return [...groups]
    .map(([name, members]) => ({ name, keys: members.sort() }))
    .sort((a, b) => (b.name === UNGROUPED) - (a.name === UNGROUPED) || a.name.localeCompare(b.name));
}

// Compact axis/tooltip formatting: integers stay integers, tiny or huge
// magnitudes switch to exponent form, everything else gets four significant digits.
export function formatMetricValue(value) {
  if (!Number.isFinite(value)) return "—";
  if (Number.isInteger(value) && Math.abs(value) < 1e9) return String(value);
  const magnitude = Math.abs(value);
  if (magnitude !== 0 && (magnitude < 1e-3 || magnitude >= 1e6)) return value.toExponential(2);
  return String(Number(value.toPrecision(4)));
}

// One color per compared run, assigned by selection order (W&B-style).
export const RUN_COLORS = [
  "#4ade80",
  "#60a5fa",
  "#f472b6",
  "#fbbf24",
  "#a78bfa",
  "#22d3ee",
  "#fb7185",
  "#a3e635",
  "#f97316",
  "#94a3b8",
];

export function runColor(index) {
  return RUN_COLORS[index % RUN_COLORS.length];
}

// Index of the point in x-sorted `points` nearest to `x`.
export function nearestIndex(points, x) {
  let lo = 0;
  let hi = points.length - 1;
  if (hi < 0) return -1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (points[mid].x < x) lo = mid + 1;
    else hi = mid;
  }
  if (lo > 0 && Math.abs(points[lo - 1].x - x) <= Math.abs(points[lo].x - x)) return lo - 1;
  return lo;
}
