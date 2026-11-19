import {
  CROSS_LANE_CONTAINMENT_TOLERANCE_S,
  CATEGORIES,
  IDLE_PHASES,
  NESTS_IN,
  NEGLIGIBLE_WORK_S,
  SAMPLED,
  TOOLTIP_HIDDEN_PHASES,
  categoryOf,
  labelFor,
  rolloutIdForTimingKey,
} from "./timing_vocabulary.js";

export function collect(timings) {
  const spans = [];
  for (const [id, lanes] of Object.entries(timings || {})) {
    const rolloutId = rolloutIdForTimingKey(id);
    for (const [role, lane] of Object.entries(lanes?.roles || {})) {
      if (lane?.lane_start_unix_s == null) continue;
      const laneStart = Number(lane?.lane_start_unix_s);
      if (!Number.isFinite(laneStart)) continue;
      for (const [name, phase] of Object.entries(lane?.phases || {})) {
        const count = Number(phase?.count) || 0;
        const total = Number(phase?.busy_duration_s) || 0;
        if (!count || total < NEGLIGIBLE_WORK_S) continue;
        const where = {
          rolloutId,
          role,
          group: role === "rollout" ? "generation" : "step",
          name,
          category: categoryOf(name),
        };
        const invocations = Array.isArray(phase?.invocations)
          ? phase.invocations
          : [];
        const runs =
          !SAMPLED.has(name) && invocations.length === count
            ? invocations
            : [];
        if (runs.length) {
          for (const [from, to] of runs) {
            const start = laneStart + (Number(from) || 0);
            const end = laneStart + (Number(to) || 0);
            spans.push({
              ...where,
              laneKey: `${id}:${role}`,
              kind: IDLE_PHASES.has(name) ? "idle" : "work",
              start,
              end,
              clockStart: start,
              clockEnd: end,
              count: 1,
              total: end - start,
              longest: end - start,
            });
          }
          continue;
        }
        spans.push({
          ...where,
          laneKey: `${id}:${role}`,
          kind: IDLE_PHASES.has(name)
            ? "idle"
            : SAMPLED.has(name) || count > 1
              ? "sampled"
              : "work",
          start: laneStart + (Number(phase?.first_start_s) || 0),
          end: laneStart + (Number(phase?.last_end_s) || 0),
          clockStart: laneStart + (Number(phase?.first_start_s) || 0),
          clockEnd: laneStart + (Number(phase?.last_end_s) || 0),
          count,
          total,
          longest: Number(phase?.longest_invocation_s) || 0,
        });
      }
    }
  }
  return anchorLanes(spans);
}

export function anchorLanes(spans) {
  const lanes = new Map();
  const drivers = new Map();
  for (const span of spans) {
    const key = `${span.rolloutId}:${span.role}`;
    const bucket = lanes.get(key) || [];
    bucket.push(span);
    lanes.set(key, bucket);
    if (span.role === "driver" && span.rolloutId != null) {
      const phases = drivers.get(span.rolloutId) || new Map();
      const candidates = phases.get(span.name) || [];
      candidates.push(span);
      phases.set(span.name, candidates);
      drivers.set(span.rolloutId, phases);
    }
  }

  for (const lane of lanes.values()) {
    const first = lane[0];
    if (first.role === "driver") continue;
    if (first.rolloutId == null) {
      for (const span of lane) span.unaligned = true;
      continue;
    }
    // A phase the frontend vocabulary doesn't know yet (an older run's name, or
    // one added framework-side first) has no category. It travels with its lane
    // either way, so it must not by itself cost the whole lane its anchor.
    const categories = new Set(
      lane
        .map((span) => span.category)
        .filter((category) => CATEGORIES[category]?.owner),
    );
    if (categories.size !== 1) {
      for (const span of lane) span.unaligned = true;
      continue;
    }
    const category = CATEGORIES[[...categories][0]];
    const candidates = drivers.get(first.rolloutId)?.get(category.owner);
    if (!candidates?.length) {
      for (const span of lane) span.unaligned = true;
      continue;
    }

    let laneStart = Infinity;
    let laneEnd = -Infinity;
    for (const span of lane) {
      laneStart = Math.min(laneStart, span.start);
      laneEnd = Math.max(laneEnd, span.end);
    }
    if (candidates.length === 1) {
      const parent = candidates[0];
      const laneDuration = laneEnd - laneStart;
      const parentDuration = parent.end - parent.start;
      if (parentDuration < laneDuration - CROSS_LANE_CONTAINMENT_TOLERANCE_S) {
        for (const span of lane) span.unaligned = true;
        continue;
      }
      let offset = 0;
      if (laneEnd > parent.end) offset = parent.end - laneEnd;
      if (laneStart + offset < parent.start) offset = parent.start - laneStart;

      for (const span of lane) {
        if (offset) {
          span.start += offset;
          span.end += offset;
          span.clockShifted = true;
          span.clockOffset = offset;
        }
      }
      continue;
    }

    const laneDuration = laneEnd - laneStart;
    let best = null;
    for (const candidate of candidates) {
      if (
        candidate.end - candidate.start <
        laneDuration - CROSS_LANE_CONTAINMENT_TOLERANCE_S
      ) {
        continue;
      }
      const minOffset =
        candidate.start -
        CROSS_LANE_CONTAINMENT_TOLERANCE_S -
        laneStart;
      const maxOffset =
        candidate.end +
        CROSS_LANE_CONTAINMENT_TOLERANCE_S -
        laneEnd;
      if (minOffset > maxOffset) continue;
      const offset = Math.max(minOffset, Math.min(0, maxOffset));
      if (
        !best ||
        Math.abs(offset) < Math.abs(best.offset) ||
        (Math.abs(offset) === Math.abs(best.offset) &&
          candidate.start < best.candidate.start)
      ) {
        best = { candidate, offset };
      }
    }
    if (!best) {
      for (const span of lane) span.unaligned = true;
      continue;
    }
    const { offset } = best;

    for (const span of lane) {
      if (offset) {
        span.start += offset;
        span.end += offset;
        span.clockShifted = true;
        span.clockOffset = offset;
      }
    }
  }
  return spans;
}

export function timingIsAsync(timings) {
  return isAsyncSpans(collect(timings));
}

export function isAsyncSpans(spans) {
  let sync = false;
  let hasRollout = false;
  for (const span of spans) {
    if (span.role === "rollout") {
      hasRollout = true;
    } else if (span.role === "driver" && span.name === "generate_rollouts") {
      sync = true;
    }
  }
  return !sync && hasRollout;
}

export const APPROXIMATE_LANE_NOTE =
  "Recorded on another node, so bar positions are approximate."

export function isApproximateSpan(span) {
  return Boolean(span?.unaligned || span?.clockShifted);
}

export function clockAlignmentDisclosure(span) {
  return isApproximateSpan(span) ? "*" : null;
}

export function groupTooltipChildren(
  children,
  aggregateStats = {},
  trainingType = null,
) {
  const groups = new Map();
  const allChildren = [...(children || []), ...Object.values(aggregateStats)];
  const rolesByName = new Map();
  const workerRoles = new Set();
  for (const child of allChildren) {
    if (child.mergedGeneration || TOOLTIP_HIDDEN_PHASES.has(child.name)) {
      continue;
    }
    const roles = rolesByName.get(child.name) || new Set();
    roles.add(child.role || "");
    rolesByName.set(child.name, roles);
    if (child.role === "actor" || child.role === "critic") {
      workerRoles.add(child.role);
    }
  }
  const allWorkerRolesPresent =
    workerRoles.has("actor") && workerRoles.has("critic");
  for (const child of allChildren) {
    if (child.mergedGeneration || TOOLTIP_HIDDEN_PHASES.has(child.name)) {
      continue;
    }
    const count = child.count || 1;
    const key = `${child.role || ""}:${child.name}`;
    const group = groups.get(key);
    if (group) {
      group.duration += child.duration;
      group.count += count;
      group.start = Math.min(group.start, child.start);
      group.end = Math.max(group.end, child.end);
      continue;
    }
    const roleIsAmbiguous = (rolesByName.get(child.name)?.size || 0) > 1;
    const roleLabel =
      (roleIsAmbiguous || allWorkerRolesPresent) &&
      child.role &&
      child.role !== "driver"
        ? ` (${child.role[0].toUpperCase()}${child.role.slice(1)})`
        : "";
    groups.set(key, {
      name: child.name,
      label: `${labelFor(child.name, child.rolloutId, trainingType)}${roleLabel}`,
      duration: child.duration,
      count,
      start: child.start,
      end: child.end,
      representative: child,
      ...(roleIsAmbiguous || allWorkerRolesPresent
        ? { role: child.role }
        : {}),
    });
  }
  // Per-sample phases run concurrently, so a group's summed duration is busy time
  // and can exceed the wall span it occupies.
  for (const group of groups.values()) {
    group.wall = Math.max(group.end - group.start, 0);
    group.concurrent = group.count > 1 && group.duration > group.wall * 1.05;
  }
  return [...groups.values()];
}

export function stepsOf(spans) {
  const byRollout = new Map();
  for (const span of spans) {
    if (span.role !== "driver" || span.rolloutId == null) continue;
    const step = byRollout.get(span.rolloutId) ?? {
      id: span.rolloutId,
      number: span.rolloutId + 1,
      start: span.start,
      end: span.end,
      work: 0,
      idle: 0,
    };
    step.start = Math.min(step.start, span.start);
    step.end = Math.max(step.end, span.end);
    if (span.kind === "idle") step.idle += span.total;
    else step.work += span.total;
    byRollout.set(span.rolloutId, step);
  }
  return [...byRollout.values()].sort((a, b) => a.start - b.start);
}

export function nest(spans) {
  const ordered = [...spans].sort((a, b) => a.start - b.start || b.end - a.end);
  const byRolloutAndName = new Map();
  for (const [index, span] of ordered.entries()) {
    span.orderIndex = index;
    span.depth = 0;
    span.children = [];
    const key = `${span.rolloutId}:${span.name}`;
    const bucket = byRolloutAndName.get(key) || [];
    bucket.push(span);
    byRolloutAndName.set(key, bucket);
  }
  for (const span of ordered) {
    let parent = null;
    for (const parentName of NESTS_IN[span.name] || []) {
      for (const other of byRolloutAndName.get(
        `${span.rolloutId}:${parentName}`,
      ) || []) {
        if (
          other !== span &&
          (other.group === span.group ||
            (span.name === "generate_samples" &&
              other.name === "generate_rollouts")) &&
          (other.laneKey === span.laneKey
            ? other.start <= span.start && span.end <= other.end
            : other.start <= span.start + CROSS_LANE_CONTAINMENT_TOLERANCE_S &&
              span.end <= other.end + CROSS_LANE_CONTAINMENT_TOLERANCE_S) &&
          (!parent ||
            other.depth > parent.depth ||
            (other.depth === parent.depth &&
              other.orderIndex < parent.orderIndex))
        ) {
          parent = other;
        }
      }
    }
    span.depth = parent ? parent.depth + 1 : 0;
    span.parent = parent ?? null;
    if (parent) parent.contains = true;
    if (parent) parent.children.push(span);
  }
  for (const span of ordered) {
    const occurrences = new Map();
    for (const child of [...span.children].sort((a, b) => a.start - b.start)) {
      if (!["forward_backward", "optimizer_step"].includes(child.name)) continue;
      const occurrence = (occurrences.get(child.name) || 0) + 1;
      occurrences.set(child.name, occurrence);
      child.ordinal = occurrence;
    }
    const duration = Math.max(span.end - span.start, 0);
    const aggregates = new Map();
    for (const child of span.children) {
      if (!SAMPLED.has(child.name)) {
        continue;
      }
      const childDuration = Math.max(child.end - child.start, 0);
      const current = aggregates.get(child.name) ?? {
        name: child.name,
        kind: child.kind,
        clockShifted: child.clockShifted,
        clockOffset: child.clockOffset,
        category: child.category,
        role: child.role,
        rolloutId: child.rolloutId,
        duration: 0,
        total: 0,
        count: 0,
        longest: 0,
        start: child.start,
        end: child.end,
      };
      current.duration += childDuration;
      current.total += child.total ?? childDuration;
      current.count += child.count || 1;
      current.longest = Math.max(current.longest, child.longest || childDuration);
      current.start = Math.min(current.start, child.start);
      current.end = Math.max(current.end, child.end);
      aggregates.set(child.name, current);
    }
    const children = span.children.filter(
      (child) => !SAMPLED.has(child.name),
    );
    for (const child of aggregates.values()) {
      children.push({
        ...child,
        share: duration > 0 ? child.duration / duration : 0,
        average: child.count ? child.total / child.count : 0,
      });
    }
    span.children = children;
    delete span.orderIndex;
  }
  return ordered;
}
