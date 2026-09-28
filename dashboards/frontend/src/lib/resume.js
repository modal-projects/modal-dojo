import { toEpochSeconds } from "./format.js";
import { rolloutIdForTimingKey } from "./timing_vocabulary.js";

function keepStep(rolloutId, createdAt, resume) {
  const startedAt = toEpochSeconds(resume?.last_attempt_started_at);
  if (!(resume?.attempt_count > 1) || !(startedAt > 0)) return true;
  const checkpoint = resume.resumed_from_checkpoint && resume.resume_from_iteration != null
    ? Number(resume.resume_from_iteration)
    : -1;
  return (rolloutId != null && Number(rolloutId) <= checkpoint)
    || (toEpochSeconds(createdAt) ?? 0) >= startedAt;
}

export function retainedRollouts(rows, resume) {
  return rows.filter((row) => keepStep(row.rollout_id, row.created_at, resume));
}

export function retainedTimings(timings, resume) {
  return Object.fromEntries(Object.entries(timings).flatMap(([key, value]) => {
    const rolloutId = rolloutIdForTimingKey(key);
    if (rolloutId == null || keepStep(rolloutId, null, resume)) return [[key, value]];
    const roles = Object.fromEntries(Object.entries(value?.roles || {}).filter(
      ([, lane]) => keepStep(rolloutId, lane?.lane_start_unix_s, resume),
    ));
    return Object.keys(roles).length ? [[key, { ...value, roles }]] : [];
  }));
}
