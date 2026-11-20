import assert from "node:assert/strict";
import test from "node:test";

import {
  FRAME_MARK,
  componentsUrl,
  frameHeight,
  frameMessage,
  isDeterministicFailure,
  manifestsKey,
  preflightFrame,
  propsMessage,
} from "./componentFrame.js";

const frameWindow = {};

test("frameMessage accepts only protocol messages from the frame", () => {
  const ready = { [FRAME_MARK]: true, type: "ready" };
  assert.deepEqual(frameMessage({ source: frameWindow, data: ready }, frameWindow), ready);
  // Another window, a missing frame, unmarked data and unknown types are ignored.
  assert.equal(frameMessage({ source: {}, data: ready }, frameWindow), null);
  assert.equal(frameMessage({ source: frameWindow, data: ready }, null), null);
  assert.equal(frameMessage({ source: frameWindow, data: { type: "ready" } }, frameWindow), null);
  assert.equal(
    frameMessage({ source: frameWindow, data: { [FRAME_MARK]: true, type: "navigate" } }, frameWindow),
    null,
  );
  assert.equal(frameMessage({ source: frameWindow, data: "ready" }, frameWindow), null);
});

test("frameHeight rounds up and rejects unusable heights", () => {
  assert.equal(frameHeight({ height: 120.2 }), 121);
  assert.equal(frameHeight({ height: 0 }), 0);
  assert.equal(frameHeight({ height: -1 }), null);
  assert.equal(frameHeight({ height: "tall" }), null);
  assert.equal(frameHeight({}), null);
});

test("propsMessage carries the protocol mark", () => {
  assert.deepEqual(propsMessage({ a: 1 }), { [FRAME_MARK]: true, type: "props", props: { a: 1 } });
});

test("only source, size and compile failures are deterministic", () => {
  for (const status of [409, 413, 422]) assert.equal(isDeterministicFailure(status), true);
  for (const status of [404, 500, 502, 503]) assert.equal(isDeterministicFailure(status), false);
});

test("preflightFrame reports server detail and whether to retry", async () => {
  const ok = await preflightFrame("/frame", async () => ({ ok: true, status: 200 }));
  assert.deepEqual(ok, { ok: true });

  const broken = await preflightFrame("/frame", async () => ({
    ok: false,
    status: 422,
    json: async () => ({ detail: "Dashboard component failed to compile: x" }),
  }));
  assert.deepEqual(broken, {
    ok: false,
    message: "Dashboard component failed to compile: x",
    deterministic: true,
  });

  const flaky = await preflightFrame("/frame", async () => ({
    ok: false,
    status: 502,
    json: async () => {
      throw new Error("not json");
    },
  }));
  assert.deepEqual(flaky, { ok: false, message: "component request failed (502)", deterministic: false });

  const offline = await preflightFrame("/frame", async () => {
    throw new Error("network down");
  });
  assert.deepEqual(offline, { ok: false, message: "network down", deterministic: false });
});

test("componentsUrl encodes the run id and slot type", () => {
  assert.equal(componentsUrl("run/1", "summary_panel"), "/api/runs/run%2F1/dashboard-components?type=summary_panel");
  assert.equal(componentsUrl("r"), "/api/runs/r/dashboard-components");
});

test("manifestsKey changes only when an attachment changes", () => {
  const a = { name: "a", sha256: "1", frame_url: "/f1", size: 3 };
  const b = { name: "b", sha256: "2", frame_url: "/f2" };
  assert.equal(manifestsKey([a, b]), manifestsKey([{ ...a, size: 99 }, b]));
  assert.notEqual(manifestsKey([a, b]), manifestsKey([b, a]));
  assert.notEqual(manifestsKey([a]), manifestsKey([{ ...a, sha256: "3" }]));
  assert.notEqual(manifestsKey([a]), manifestsKey([{ ...a, frame_url: "/f1?v=2" }]));
  assert.equal(manifestsKey(null), "");
});
