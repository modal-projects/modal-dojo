import assert from "node:assert/strict";
import test from "node:test";

import { UNGROUPED, formatMetricValue, groupMetricKeys } from "./metricSeries.js";
import { niceTicks } from "./ticks.js";

const KEYS = ["train/loss", "rollout/reward", "perf/tokens_per_s", "lr", "eval/acc", "train/grad_norm"];

test("groupMetricKeys groups by prefix, ungrouped first, then alphabetically", () => {
  assert.deepEqual(groupMetricKeys(KEYS), [
    { name: UNGROUPED, keys: ["lr"] },
    { name: "eval", keys: ["eval/acc"] },
    { name: "perf", keys: ["perf/tokens_per_s"] },
    { name: "rollout", keys: ["rollout/reward"] },
    { name: "train", keys: ["train/grad_norm", "train/loss"] },
  ]);
});

test("groupMetricKeys search is a case-insensitive substring match", () => {
  assert.deepEqual(groupMetricKeys(KEYS, " LOSS "), [{ name: "train", keys: ["train/loss"] }]);
  assert.deepEqual(groupMetricKeys(KEYS, "nothing"), []);
});

test("niceTicks picks round values inside the range", () => {
  assert.deepEqual(niceTicks(0, 39, 5), [0, 10, 20, 30]);
  assert.deepEqual(niceTicks(1.53, 2.07, 4), [1.6, 1.8, 2]);
  assert.deepEqual(niceTicks(-0.012, 0.031, 4), [0, 0.02]);
  assert.deepEqual(niceTicks(-0.012, 0.031, 8), [-0.01, 0, 0.01, 0.02, 0.03]);
  assert.deepEqual(niceTicks(5, 5, 4), []);
});

test("formatMetricValue is compact across magnitudes", () => {
  assert.equal(formatMetricValue(3), "3");
  assert.equal(formatMetricValue(0.123456), "0.1235");
  assert.equal(formatMetricValue(0.00001234), "1.23e-5");
  assert.equal(formatMetricValue(12345678.9), "1.23e+7");
  assert.equal(formatMetricValue(0), "0");
  assert.equal(formatMetricValue(NaN), "—");
  assert.equal(formatMetricValue("x"), "—");
});
