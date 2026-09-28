import assert from "node:assert/strict";
import test from "node:test";

import { clipIdleSpans, nestedHitTargetsForRow, runTimeline } from "./timing.js";
import { retainedRollouts, retainedTimings } from "./resume.js";

const resumedRun = {
  attempt_count: 2,
  last_attempt_started_at: 1000,
  resumed_from_checkpoint: true,
  resume_from_iteration: 19,
};

test("resume keeps checkpoint history and fresh rollouts, not discarded steps", () => {
  const rows = [
    { rollout_id: 19, created_at: 900 },
    { rollout_id: 20, created_at: 1000 },
    { rollout_id: 21, created_at: 1100 },
    { rollout_id: 22, created_at: 950 },
    { rollout_id: 33, created_at: 990 },
    { rollout_id: 34 },
  ];
  assert.deepEqual(retainedRollouts(rows, resumedRun), rows.slice(0, 3));
  assert.deepEqual(retainedRollouts(rows, null), rows);
  assert.deepEqual(retainedRollouts(rows, { ...resumedRun, attempt_count: 1 }), rows);
  assert.deepEqual(retainedRollouts(rows, { ...resumedRun, last_attempt_started_at: null }), rows);
  assert.deepEqual(retainedRollouts(rows, { ...resumedRun, resumed_from_checkpoint: false }), rows.slice(1, 3));
  assert.deepEqual(retainedRollouts(rows, { ...resumedRun, resume_from_iteration: null }), rows.slice(1, 3));
});

test("a checkpoint at zero retains its completed first step", () => {
  const rows = [{ rollout_id: 0, created_at: 900 }, { rollout_id: 1, created_at: 950 }];
  assert.deepEqual(retainedRollouts(rows, { ...resumedRun, resume_from_iteration: 0 }), [rows[0]]);
});

test("resume filters timing lanes before aligning repeated step IDs", () => {
  const old = { lane_start_unix_s: 900, phases: {} };
  const current = { lane_start_unix_s: 1000, phases: {} };
  const timings = {
    metadata: { timing_stale: false },
    startup: { roles: { driver: old } },
    19: { roles: { actor: old } },
    20: { roles: { driver: current, actor: old, rollout: current } },
    33: { roles: { driver: old, actor: old } },
    34: { roles: { actor: { phases: {} } } },
  };
  assert.deepEqual(retainedTimings(timings, resumedRun), {
    metadata: timings.metadata,
    startup: timings.startup,
    19: timings[19],
    20: { roles: { driver: current, rollout: current } },
  });
  assert.equal(timings[20].roles.actor, old);
  assert.deepEqual(retainedTimings(timings, null), timings);
});

test("nested hit targets cover drawn bars without entering siblings", () => {
  const bars = [
    {
      key: "floored-first",
      depth: 1,
      renderStart: 100,
      renderEnd: 100.003,
      renderDuration: 0.003,
      insideRenderStart: 100,
      insideRenderEnd: 100.02,
    },
    {
      key: "floored-second",
      depth: 1,
      renderStart: 100.01,
      renderEnd: 100.013,
      renderDuration: 0.003,
      insideRenderStart: 100,
      insideRenderEnd: 100.02,
    },
  ];
  const targets = nestedHitTargetsForRow({ sortedSpans: bars }, 1000);

  for (const bar of bars) {
    const target = targets.get(bar.key);
    const left = bar.renderStart - Number.parseFloat(target.left) / 1000;
    const right = bar.renderEnd + Number.parseFloat(target.right) / 1000;
    assert.ok(left <= (bar.renderStart + bar.renderEnd) / 2);
    assert.ok(right >= (bar.renderStart + bar.renderEnd) / 2);
    for (const sibling of bars) {
      if (sibling === bar) continue;
      assert.ok(right <= sibling.renderStart || left >= sibling.renderEnd);
    }
  }

  const overlappingBars = [
    {
      key: "wide",
      depth: 1,
      renderStart: 0,
      renderEnd: 100,
      renderDuration: 100,
      insideRenderStart: 0,
      insideRenderEnd: 100,
    },
    {
      key: "narrow-middle",
      depth: 1,
      renderStart: 60,
      renderEnd: 62,
      renderDuration: 2,
      insideRenderStart: 0,
      insideRenderEnd: 100,
    },
    {
      key: "narrow-late",
      depth: 1,
      renderStart: 70,
      renderEnd: 72,
      renderDuration: 2,
      insideRenderStart: 0,
      insideRenderEnd: 100,
    },
  ];
  const overlappingTargets = nestedHitTargetsForRow(
    { sortedSpans: overlappingBars },
    1,
  );
  for (const bar of overlappingBars) {
    const target = overlappingTargets.get(bar.key);
    const left = bar.renderStart - Number.parseFloat(target.left);
    const right = bar.renderEnd + Number.parseFloat(target.right);
    const center = (bar.renderStart + bar.renderEnd) / 2;
    assert.ok(left <= center && right >= center);
  }
  assert.equal(
    Number.parseFloat(overlappingTargets.get("narrow-late").left),
    0,
  );

  const idlePieces = clipIdleSpans(
    [
      {
        kind: "idle",
        role: "driver",
        start: 0,
        end: 30,
        total: 30,
        duration: 30,
        clockStart: 1000,
        clockEnd: 1030,
      },
      {
        kind: "work",
        role: "driver",
        start: 5,
        end: 10,
        renderStart: 5,
        renderEnd: 10,
      },
    ],
    false,
  );
  assert.deepEqual(
    idlePieces
      .filter(({ kind }) => kind === "idle")
      .map(({ start, end, clockStart, clockEnd }) => ({
        start,
        end,
        clockStart,
        clockEnd,
      })),
    [
      { start: 0, end: 5, clockStart: 1000, clockEnd: 1005 },
      { start: 10, end: 30, clockStart: 1010, clockEnd: 1030 },
    ],
  );
});

test("empty packed rollout groups do not produce labelled rows", () => {
  const phase = (start, end) => ({
    count: 1,
    busy_duration_s: end - start,
    first_start_s: start,
    last_end_s: end,
    invocations: [[start, end]],
  });
  const timeline = runTimeline(
    {
      0: {
        roles: {
          driver: {
            lane_start_unix_s: 100,
            phases: { train_models: phase(0, 4) },
          },
          rollout: {
            lane_start_unix_s: 100,
            phases: {
              reward: phase(0, 2),
              generate_samples: phase(1, 3),
            },
          },
        },
      },
    },
    true,
  );
  const rolloutRows = timeline.groups[0].rows.filter(
    (row) => row.role === "rollout",
  );
  assert.equal(rolloutRows.length, 1);
  assert.equal(rolloutRows[0].key, "rollout-0");
  assert.ok(rolloutRows[0].spans.some((span) => span.name === "generate_samples"));
});

test("unmapOffset inverts the collapsed-gap compression", () => {
  const phase = (start, end) => ({
    count: 1,
    busy_duration_s: end - start,
    first_start_s: start,
    last_end_s: end,
    invocations: [[start, end]],
  });
  const timeline = runTimeline({
    0: {
      roles: {
        driver: { lane_start_unix_s: 1000, phases: { train_models: phase(0, 10) } },
      },
    },
    1: {
      roles: {
        driver: { lane_start_unix_s: 1000, phases: { train_models: phase(600, 610) } },
      },
    },
  });
  assert.ok(timeline.breaks.length > 0);
  assert.ok(timeline.span < 610);
  for (const seconds of [0, 5, 10, 300, 599, 600, 605, 610]) {
    const mapped = timeline.mapOffset(seconds);
    assert.ok(Math.abs(timeline.unmapOffset(mapped) - seconds) < 1e-6, `${seconds}`);
  }
  assert.equal(timeline.unmapOffset(0), 0);
  assert.ok(Math.abs(timeline.unmapOffset(timeline.span) - 610) < 1e-6);
});
