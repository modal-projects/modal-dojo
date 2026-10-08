const SERVER = "/api";


async function getErrorFromResponse(res) {
  let detail = res.statusText;
  try {
    const body = await res.json();
    if (body?.detail) detail = String(body.detail);
  } catch (error) {
    if (!(error instanceof SyntaxError)) throw error;
  }
  return detail ? `HTTP ${res.status}: ${detail}` : `HTTP ${res.status}`;
}

// `Response.json()` on a non-JSON body throws a parser error that says nothing
// about the request — on WebKit it's the famously opaque "The string did not
// match the expected pattern" — so a proxy login page, an edge error page or a
// truncated response all surface as the same riddle. Parse the text ourselves
// and say what came back instead.
async function readJson(res, what) {
  const body = await res.text();
  try {
    return JSON.parse(body);
  } catch {
    const contentType = res.headers.get("content-type") || "unknown type";
    throw new Error(
      `${what}: expected JSON, got ${contentType} (HTTP ${res.status}, ${body.length} bytes)`,
    );
  }
}

// The run list is paged and filtered server-side: `facets` maps a facet name
// ("status", "recipe", "group") to the selected buckets, and a facet left out
// means every bucket. Totals and per-bucket counts come from `fetchRunCounts`,
// which sees every run rather than the page the client holds.
function runQueryParams({ limit, offset, query, facets }) {
  const params = new URLSearchParams();
  if (limit != null) params.set("limit", String(limit));
  if (offset) params.set("offset", String(offset));
  if (query) params.set("q", query);
  for (const [facet, values] of Object.entries(facets || {})) {
    for (const value of values) params.append(facet, value);
  }
  const search = params.toString();
  return search ? `?${search}` : "";
}

export async function fetchRuns({ signal, limit, offset, query, facets } = {}) {
  const search = runQueryParams({ limit, offset, query, facets });
  const response = await fetch(`${SERVER}/runs${search}`, { signal });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const runs = await readJson(response, "runs");
  return Array.isArray(runs) ? runs : [];
}

// Per-bucket counts cover every run; `matching` counts the given query, so the
// page knows how many rows are still reachable by paging.
export async function fetchRunCounts({ signal, query, facets } = {}) {
  const search = runQueryParams({ query, facets });
  const res = await fetch(`${SERVER}/runs/counts${search}`, { signal });
  if (!res.ok) throw new Error(await getErrorFromResponse(res));
  return await readJson(res, "run counts");
}

export async function fetchRun(trainingRunId, { signal } = {}) {
  const res = await fetch(
    `${SERVER}/runs/${encodeURIComponent(trainingRunId)}`,
    { signal },
  );
  if (res.status === 404) return null;
  if (!res.ok) {
    throw new Error(await getErrorFromResponse(res));
  }
  return await readJson(res, "run detail");
}

export async function fetchEvals({ signal } = {}) {
  const res = await fetch(`${SERVER}/evals`, { signal });
  if (!res.ok) {
    throw new Error(await getErrorFromResponse(res));
  }
  const evals = await readJson(res, "evals");
  const seen = new Set();
  return evals.filter((e) => {
    if (seen.has(e.eval_id)) return false;
    seen.add(e.eval_id);
    return true;
  });
}

export async function fetchTrainResult(trainingRunId) {
  const res = await fetch(
    `${SERVER}/train-results/${encodeURIComponent(trainingRunId)}`
  );
  if (!res.ok) return null;
  return await res.json();
}

export async function fetchEvalDetail(evalId) {
  const res = await fetch(
    `${SERVER}/evals/${encodeURIComponent(evalId)}`
  );
  if (!res.ok) return null;
  return await res.json();
}

export async function fetchRunRollouts(trainingRunId, { signal } = {}) {
  const res = await fetch(
    `${SERVER}/runs/${encodeURIComponent(trainingRunId)}/rollouts`,
    { signal },
  );
  if (!res.ok) return [];
  const data = await res.json();
  if (!Array.isArray(data)) return [];
  return data
    .filter((item) => item && typeof item === "object")
    .map((item) => ({
      rollout_id: Number(item.rollout_id) || 0,
      created_at: Number(item.created_at) || 0,
      total: Number(item.total) || 0,
      episode_count:
        item.episode_count == null ? null : Number(item.episode_count) || 0,
      mean: typeof item.mean === "number" ? item.mean : Number(item.mean) || 0,
      reward_stats:
        item.reward_stats && typeof item.reward_stats === "object" ? item.reward_stats : null,
      rollout_time: Number.isFinite(Number(item.rollout_time))
        ? Number(item.rollout_time)
        : null,
      error_summary: item.error_summary || null,
      tag_stats: item.tag_stats || {},
    }))
    .sort((a, b) => a.rollout_id - b.rollout_id);
}

export async function fetchRollout(trainingRunId, rolloutId) {
  const res = await fetch(
    `${SERVER}/runs/${encodeURIComponent(trainingRunId)}/rollouts/${encodeURIComponent(rolloutId)}`,
  );
  if (!res.ok) return null;
  return await res.json();
}

// Historical Modal logs for a run, served from the durable storage.
//
// Returns the newest `maxLines` lines within the (since, until] window, oldest
// first, plus a `nextUntil` cursor for paging further back through history.
// When omitted, `maxLines` defaults to 100 on the server.
// `since` defaults to the run's start (then creation time or the Unix epoch);
// `until` defaults to the run's end/completion time, or now when unavailable.
export async function fetchRunLogs(
  trainingRunId,
  { since, until, maxLines, search, signal } = {},
) {
  const params = new URLSearchParams();
  if (since != null && since !== "") params.set("since", String(since));
  if (until != null && until !== "") params.set("until", String(until));
  if (maxLines != null) params.set("max_lines", String(maxLines));
  if (search) params.set("search", search);
  const qs = params.toString();
  const res = await fetch(
    `${SERVER}/runs/${encodeURIComponent(trainingRunId)}/logs` + (qs ? `?${qs}` : ""),
    { signal },
  );
  if (!res.ok) {
    throw new Error(await getErrorFromResponse(res));
  }
  const data = await res.json();
  return {
    logs: Array.isArray(data.logs) ? data.logs : [],
    hasMore: !!data.has_more,
    nextUntil: data.next_until ?? null,
  };
}

// Per-step advantage distribution summaries (one row per training step, each
// carrying that step's overall stats + quantiles). Drives the fan chart that
// shows how the advantage distribution shifts/spreads over training.
export async function fetchRunAdvantages(trainingRunId, { signal } = {}) {
  const res = await fetch(
    `${SERVER}/runs/${encodeURIComponent(trainingRunId)}/advantages`,
    { signal },
  );
  if (!res.ok) return [];
  const data = await res.json();
  if (!Array.isArray(data)) return [];
  return data
    .filter((item) => item && typeof item === "object" && item.stats)
    .map((item) => ({
      rollout_id: Number(item.rollout_id) || 0,
      created_at: Number(item.created_at) || 0,
      num_samples: Number(item.num_samples) || 0,
      num_groups: Number(item.num_groups) || 0,
      stats: item.stats,
    }))
    .sort((a, b) => a.rollout_id - b.rollout_id);
}

export async function fetchRunTimings(trainingRunId, { signal } = {}) {
  const res = await fetch(
    `${SERVER}/runs/${encodeURIComponent(trainingRunId)}/timings`,
    { signal },
  );
  if (!res.ok) throw new Error(await getErrorFromResponse(res));
  return await res.json();
}

// Scalar metrics mirrored from wandb/trackio: `{ series: { key: [[step, value], ...] }, stale }`.
export async function fetchRunMetrics(trainingRunId, { signal } = {}) {
  const res = await fetch(`${SERVER}/runs/${encodeURIComponent(trainingRunId)}/metrics`, { signal });
  if (!res.ok) throw new Error(await getErrorFromResponse(res));
  return await res.json();
}

// One step's full per-group advantage distribution (for drill-in).
export async function fetchRunAdvantageStep(trainingRunId, rolloutId) {
  const res = await fetch(
    `${SERVER}/runs/${encodeURIComponent(trainingRunId)}/advantages/${encodeURIComponent(rolloutId)}`,
  );
  if (!res.ok) return null;
  return await res.json();
}

// ── Autoconfig ────────────────────────────────────────────────────────────

export async function fetchCatalogue({ signal } = {}) {
  const res = await fetch(`${SERVER}/autoconfig/catalogue`, { signal });
  if (!res.ok) throw new Error(await getErrorFromResponse(res));
  return readJson(res, "catalogue");
}

// Kicks off a sweep; the server answers 202 with an operation to poll via
// `fetchSweep` until its status leaves "pending".
export async function submitSweep(request) {
  const res = await fetch(`${SERVER}/autoconfig/sweeps`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  if (!res.ok) throw new Error(await getErrorFromResponse(res));
  return readJson(res, "sweep");
}

export async function fetchSweep(operationId, { signal } = {}) {
  const res = await fetch(
    `${SERVER}/autoconfig/sweeps/${encodeURIComponent(operationId)}`,
    { signal },
  );
  if (!res.ok) throw new Error(await getErrorFromResponse(res));
  return readJson(res, "sweep");
}
