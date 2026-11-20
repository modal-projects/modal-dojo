// Shared protocol for run-scoped dashboard components, which run in sandboxed
// <iframe>s served by `/api/runs/<id>/dashboard-components/<type>/<sha256>/frame.html`
// (see `_COMPONENT_FRAME_TEMPLATE` in modal_dojo/_dashboard.py). The parent
// sends `props`; the frame reports `ready`, `rendered`, `resize` and `error`.

export const FRAME_MARK = "modalDojoDashboardComponent";

// How long a frame may take to report `ready` before it counts as failed.
export const FRAME_READY_TIMEOUT_MS = 20000;

// Components can be attached after a run finishes, when the page has stopped
// polling the run; re-check attachments on this interval so they still show up.
export const MANIFEST_POLL_MS = 30000;

// Frame responses that fail the same way on every retry: a source that does not
// match its manifest, is too large, or does not compile. Callers stop retrying
// these digests; anything else (5xx, network) is retried on the next poll.
const DETERMINISTIC_FAILURES = new Set([409, 413, 422]);

export function isDeterministicFailure(status) {
  return DETERMINISTIC_FAILURES.has(status);
}

/**
 * Return the message a frame posted, or null when the event did not come from
 * `frameWindow` or is not part of the component protocol.
 */
export function frameMessage(event, frameWindow) {
  if (!frameWindow || event?.source !== frameWindow) return null;
  const data = event.data;
  if (!data || typeof data !== "object" || data[FRAME_MARK] !== true) return null;
  if (!["ready", "rendered", "resize", "error"].includes(data.type)) return null;
  return data;
}

/** A frame's reported content height in whole pixels, or null if unusable. */
export function frameHeight(message) {
  const height = Number(message?.height);
  return Number.isFinite(height) && height >= 0 ? Math.ceil(height) : null;
}

export function propsMessage(props) {
  return { [FRAME_MARK]: true, type: "props", props };
}

/**
 * Fetch a frame once before mounting it, so a compile error shows up as a
 * readable message instead of a blank frame. Resolves to `{ ok: true }` or
 * `{ ok: false, message, deterministic }`.
 */
export async function preflightFrame(src, fetchImpl = fetch) {
  let response;
  try {
    response = await fetchImpl(src, { credentials: "same-origin" });
  } catch (error) {
    return { ok: false, message: error?.message || String(error), deterministic: false };
  }
  if (response.ok) return { ok: true };
  let message = `component request failed (${response.status})`;
  try {
    message = (await response.json())?.detail || message;
  } catch {
    // Non-JSON error bodies keep the generic status message.
  }
  return { ok: false, message, deterministic: isDeterministicFailure(response.status) };
}

/** The attachment list URL for a run, optionally filtered to one slot type. */
export function componentsUrl(runId, type) {
  const base = `/api/runs/${encodeURIComponent(runId)}/dashboard-components`;
  return type ? `${base}?type=${encodeURIComponent(type)}` : base;
}

/**
 * Stable key for a list of manifests: changes only when an attachment is
 * added, removed, renamed or replaced, so polling never remounts frames.
 */
export function manifestsKey(manifests) {
  return (manifests || [])
    .map((manifest) => `${manifest?.name}:${manifest?.sha256}:${manifest?.frame_url}`)
    .join("|");
}
