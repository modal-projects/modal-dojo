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
