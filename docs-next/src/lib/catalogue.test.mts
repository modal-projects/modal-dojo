import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import {
  baseModel,
  CONTEXT_STOPS,
  contextRangeLabel,
  formatDuration,
  formatReward,
  formatRewardChange,
  formatTokens,
  formatUsd,
  inContextRange,
  rewardDigits,
  sortRows,
  type Catalogue,
  type CatalogueRow,
  type CatalogueRun,
} from './catalogue.ts';

function row(name: string, run: Partial<CatalogueRun> | null): CatalogueRow {
  return {
    name,
    model: `org/${name}`,
    model_href: `https://huggingface.co/org/${name}`,
    framework: 'slime',
    recipe: `${name}_Recipe`,
    recipe_href: `/reference/${name.toLowerCase()}_recipe`,
    context_length: null,
    run:
      run === null
        ? null
        : {
            id: `${name}-run`,
            dashboard_url: null,
            created_at: 0,
            status: 'completed',
            dataset: '',
            gpu_type: 'H200',
            gpus: 8,
            nodes: 1,
            gpu_hour_usd: 4.54,
            steps: [],
            steps_completed: 0,
            time_per_step_s: null,
            cost_per_step_usd: null,
            initial_reward: null,
            ...run,
          },
  };
}

const names = (rows: CatalogueRow[]) => rows.map((entry) => entry.name);

test('sorting puts recipes without a value last in both directions', () => {
  const rows = [
    row('pending', null),
    row('cheap', { cost_per_step_usd: 5 }),
    row('unpriced', {}),
    row('pricey', { cost_per_step_usd: 50 }),
  ];
  assert.deepEqual(names(sortRows(rows, 'cost', 'asc')), ['cheap', 'pricey', 'pending', 'unpriced']);
  assert.deepEqual(names(sortRows(rows, 'cost', 'desc')), ['pricey', 'cheap', 'pending', 'unpriced']);
});

test('sorting keeps catalogue order between equal values', () => {
  const rows = [
    row('first', { initial_reward: 0.4 }),
    row('second', { initial_reward: 0.4 }),
    row('best', { initial_reward: 0.5 }),
  ];
  assert.deepEqual(names(sortRows(rows, 'initial', 'desc')), ['best', 'first', 'second']);
  assert.deepEqual(names(sortRows(rows, 'initial', 'asc')), ['first', 'second', 'best']);
});

test('durations read as seconds, minutes, or hours', () => {
  assert.equal(formatDuration(null), '—');
  assert.equal(formatDuration(45.4), '45s');
  assert.equal(formatDuration(472.2), '7m 52s');
  assert.equal(formatDuration(3725), '1h 02m');
});

test('costs read as dollars and cents', () => {
  assert.equal(formatUsd(33.16), '$33.16');
  assert.equal(formatUsd(1234.5), '$1,234.50');
  assert.equal(formatUsd(null), '—');
});

test('pass/fail rewards keep three decimals and larger scales keep one', () => {
  assert.equal(rewardDigits([0.34, null, 1, -0.5]), 3);
  assert.equal(rewardDigits([84.86, 96.22]), 1);
  assert.equal(formatReward(0.3421, 3), '0.342');
  assert.equal(formatReward(84.8554, 1), '84.9');
  assert.equal(formatReward(null, 3), '—');
});

test('reward changes are signed', () => {
  assert.equal(formatRewardChange(0.34, 0.375, 3), '+0.035');
  assert.equal(formatRewardChange(0.375, 0.34, 3), '−0.035');
  assert.equal(formatRewardChange(84.8554, 96.2159, 1), '+11.4');
  assert.equal(formatRewardChange(0.5, 0.5, 3), '±0.000');
  assert.equal(formatRewardChange(0.5, null, 3), null);
});

test('base models drop the Hugging Face organization', () => {
  assert.equal(baseModel({ ...row('x', null), model: 'zai-org/GLM-5.3-Flash' }), 'GLM-5.3-Flash');
  assert.equal(baseModel({ ...row('x', null), model: 'local-model' }), 'local-model');
});

test('context lengths read in binary units', () => {
  assert.equal(formatTokens(32768), '32K');
  assert.equal(formatTokens(40960), '40K');
  assert.equal(formatTokens(1048576), '1M');
  assert.equal(formatTokens(1572864), '1.5M');
  assert.equal(formatTokens(512), '512');
});

test('the context range is open-ended at its end stops', () => {
  const last = CONTEXT_STOPS.length - 1;
  const at32k = CONTEXT_STOPS.indexOf(32768);
  const at128k = CONTEXT_STOPS.indexOf(131072);
  // The full range keeps every recipe, even one with no known context length.
  assert.ok(inContextRange(null, 0, last));
  assert.ok(inContextRange(2048, 0, last));
  // Any narrower range needs a known length inside it.
  assert.ok(!inContextRange(null, at32k, last));
  assert.ok(inContextRange(32768, at32k, at128k));
  assert.ok(inContextRange(131072, at32k, at128k));
  assert.ok(!inContextRange(16384, at32k, at128k));
  assert.ok(!inContextRange(262144, at32k, at128k));
  // The end stops mean no minimum and no maximum.
  assert.ok(inContextRange(2048, 0, at32k));
  assert.ok(inContextRange(2097152, at32k, last));
});

test('context range labels name the open ends', () => {
  const last = CONTEXT_STOPS.length - 1;
  const at32k = CONTEXT_STOPS.indexOf(32768);
  const at128k = CONTEXT_STOPS.indexOf(131072);
  assert.equal(contextRangeLabel(0, last), 'Any length');
  assert.equal(contextRangeLabel(0, at32k), 'Up to 32K');
  assert.equal(contextRangeLabel(at32k, last), '32K and up');
  assert.equal(contextRangeLabel(at32k, at128k), '32K–128K');
  assert.equal(contextRangeLabel(at32k, at32k), '32K');
});

test('generated catalogue has a row for every recipe', () => {
  const catalogue = JSON.parse(
    readFileSync(new URL('../generated/catalogue.json', import.meta.url), 'utf8'),
  ) as Catalogue;
  assert.ok(catalogue.rows.length > 0);
  for (const entry of catalogue.rows) {
    assert.match(entry.recipe_href, /^\/reference\/[a-z0-9_]+_recipe$/);
    if (entry.run) {
      assert.ok(entry.run.gpus > 0, `${entry.name} run has no GPUs`);
      assert.equal(entry.run.steps_completed, entry.run.steps.length);
    }
  }
});
