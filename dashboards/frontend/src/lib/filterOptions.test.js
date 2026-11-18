import assert from "node:assert/strict";
import test from "node:test";

import { newTagOption, tagOptions } from "./filterOptions.js";

test("tag options flatten counts, sort pairs and retain arbitrary selections", () => {
  const options = tagOptions({ owner: { Bob: 2, Alice: 1 }, backend: { alpha: 3 } }, { owner: ["Bob", "Zed"] });
  assert.deepEqual(options.map(({ label, count, selected }) => ({ label, count, selected })), [
    { label: "backend:alpha", count: 3, selected: false },
    { label: "owner:Alice", count: 1, selected: false },
    { label: "owner:Bob", count: 2, selected: true },
    { label: "owner:Zed", count: 0, selected: true },
  ]);
  assert.deepEqual(tagOptions({}, {}), []);
});

test("arbitrary tag entry splits only the first colon and matches exact case", () => {
  const options = tagOptions({ owner: { Bob: 2 } }, {});
  assert.equal(newTagOption("owner:Bob", options), null);
  assert.deepEqual(newTagOption("owner:bob", options), {
    id: '["owner","bob"]', key: "owner", value: "bob", label: "owner:bob",
  });
  assert.equal(newTagOption("custom:a:b,c = d", []).value, "a:b,c = d");
  for (const query of ["", "owner", ":Bob", "owner:"]) assert.equal(newTagOption(query, []), null);
});

test("tag option identities distinguish colons and prototype-like keys", () => {
  const options = tagOptions(JSON.parse('{"a:b":{"c":1},"a":{"b:c":2},"__proto__":{"x":3}}'), {});
  assert.equal(new Set(options.map(({ id }) => id)).size, 3);
  assert.equal(options.find(({ key }) => key === "__proto__").selected, false);
});
