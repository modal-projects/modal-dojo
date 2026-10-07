// Shape of src/generated/catalogue.json (written by scripts/generate_catalogue.py)
// plus the sorting and formatting the /catalogue page shares with its script.

export type SortKey = 'cost' | 'time' | 'initial';
export type SortDirection = 'asc' | 'desc';

export interface CatalogueStep {
  step: number;
  seconds: number | null;
  cost_usd: number | null;
  // Running GPU cost through this step, summed before rounding.
  total_usd: number | null;
  raw_reward: number | null;
}

export interface CatalogueRun {
  id: string;
  // The run's page on the dojo dashboard of the environment that ran it.
  dashboard_url: string | null;
  created_at: number;
  status: string;
  dataset: string;
  gpu_type: string;
  gpus: number;
  nodes: number;
  gpu_hour_usd: number | null;
  steps: CatalogueStep[];
  steps_completed: number;
  time_per_step_s: number | null;
  cost_per_step_usd: number | null;
  initial_reward: number | null;
}

export interface CatalogueRow {
  name: string;
  model: string;
  model_href: string;
  framework: string;
  recipe: string;
  recipe_href: string;
  // Longest sequence, prompt plus response, the recipe rolls out.
  context_length: number | null;
  // "recipe" when the recipe sets it, "model" for the model's own window.
  context_source: 'recipe' | 'model' | null;
  run: CatalogueRun | null;
}

export interface Catalogue {
  benchmark: string;
  steps: number;
  priced_at: string | null;
  // True when generated with `--example`: placeholder runs, not real ones.
  example?: boolean;
  rows: CatalogueRow[];
}

// Each sort opens in the direction a reader most likely wants first:
// cheapest and fastest, or highest reward.
export const SORTS: readonly { key: SortKey; direction: SortDirection }[] = [
  { key: 'cost', direction: 'asc' },
  { key: 'time', direction: 'asc' },
  { key: 'initial', direction: 'desc' },
];

export function sortLabel(key: SortKey): string {
  switch (key) {
    case 'cost':
      return 'Cost / step';
    case 'time':
      return 'Time / step';
    case 'initial':
      return 'Step 1 reward';
  }
}

export function sortValue(row: CatalogueRow, key: SortKey): number | null {
  const run = row.run;
  if (!run) return null;
  switch (key) {
    case 'cost':
      return run.cost_per_step_usd;
    case 'time':
      return run.time_per_step_s;
    case 'initial':
      return run.initial_reward;
  }
}

/** The base model's name without its Hugging Face organization. */
export function baseModel(row: CatalogueRow): string {
  return row.model.slice(row.model.lastIndexOf('/') + 1);
}

// Context length slider stops. The end stops are open-ended: the first means
// no minimum and the last no maximum, so the full range filters nothing.
export const CONTEXT_STOPS: readonly number[] = [
  4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576,
];

/** Whether a recipe's context length falls between two slider stops. */
export function inContextRange(context: number | null, low: number, high: number): boolean {
  const last = CONTEXT_STOPS.length - 1;
  if (low === 0 && high === last) return true;
  if (context === null) return false;
  return (
    (low === 0 || context >= CONTEXT_STOPS[low]) &&
    (high === last || context <= CONTEXT_STOPS[high])
  );
}

export function contextRangeLabel(low: number, high: number): string {
  const last = CONTEXT_STOPS.length - 1;
  const lowLabel = formatTokens(CONTEXT_STOPS[low]);
  const highLabel = formatTokens(CONTEXT_STOPS[high]);
  if (low === 0 && high === last) return 'Any length';
  if (low === 0) return `Up to ${highLabel}`;
  if (high === last) return `${lowLabel} and up`;
  return low === high ? lowLabel : `${lowLabel}–${highLabel}`;
}

/** Order two sort values; a missing value sorts last in either direction. */
export function compareValues(
  left: number | null,
  right: number | null,
  direction: SortDirection,
): number {
  if (left === null || right === null) {
    if (left === right) return 0;
    return left === null ? 1 : -1;
  }
  return direction === 'asc' ? left - right : right - left;
}

/** Sort rows by one key, keeping catalogue order between equal values. */
export function sortRows(
  rows: readonly CatalogueRow[],
  key: SortKey,
  direction: SortDirection,
): CatalogueRow[] {
  return rows
    .map((row, index) => ({ row, index }))
    .sort(
      (left, right) =>
        compareValues(sortValue(left.row, key), sortValue(right.row, key), direction) ||
        left.index - right.index,
    )
    .map(({ row }) => row);
}

const usd = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' });
const date = new Intl.DateTimeFormat('en-US', {
  month: 'short',
  day: 'numeric',
  year: 'numeric',
  timeZone: 'UTC',
});
const MINUS = '−';

export function formatUsd(value: number | null): string {
  return value === null ? '—' : usd.format(value);
}

export function formatDuration(seconds: number | null): string {
  if (seconds === null) return '—';
  const total = Math.round(seconds);
  if (total < 60) return `${total}s`;
  if (total < 3600) {
    return `${Math.floor(total / 60)}m ${String(total % 60).padStart(2, '0')}s`;
  }
  const minutes = Math.floor((total % 3600) / 60);
  return `${Math.floor(total / 3600)}h ${String(minutes).padStart(2, '0')}m`;
}

/** Token counts in binary units, the way context windows are quoted: 32K, 1M. */
export function formatTokens(tokens: number): string {
  const mebi = 1024 * 1024;
  if (tokens >= mebi) {
    const millions = tokens / mebi;
    return `${Number.isInteger(millions) ? millions : millions.toFixed(1)}M`;
  }
  return tokens >= 1024 ? `${Math.round(tokens / 1024)}K` : String(tokens);
}

export function formatDate(unixSeconds: number): string {
  return date.format(new Date(unixSeconds * 1000));
}

/**
 * Decimal places for a run's raw rewards: three when they stay within ±1, as
 * binary pass/fail rewards like SWE-bench's do, one for larger scales.
 */
export function rewardDigits(rewards: (number | null)[]): number {
  return rewards.every((reward) => reward === null || Math.abs(reward) <= 1) ? 3 : 1;
}

export function formatReward(reward: number | null, digits: number): string {
  return reward === null ? '—' : reward.toFixed(digits);
}

/** Signed change between two rewards, or null when either is missing. */
export function formatRewardChange(
  from: number | null,
  to: number | null,
  digits: number,
): string | null {
  if (from === null || to === null) return null;
  const change = to - from;
  const magnitude = Math.abs(change).toFixed(digits);
  if (Number(magnitude) === 0) return `±${magnitude}`;
  return `${change > 0 ? '+' : MINUS}${magnitude}`;
}
