<script>
  import { onMount, tick, untrack } from "svelte";
  import { ArrowLeft, ChevronLeft, ChevronRight, Download, ExternalLink, Minimize2, X } from "lucide-svelte";
  import Tabs from "../components/Tabs.svelte";
  import RunSummary from "../components/RunSummary.svelte";
  import RunTimeline from "../components/RunTimeline.svelte";
  import StatusPill from "../components/StatusPill.svelte";
  import TimeAgo from "../components/TimeAgo.svelte";
  import InferenceStats from "../components/InferenceStats.svelte";
  import SampleTimeline from "../components/SampleTimeline.svelte";
  import DynamicTrajectoryViewer from "../components/DynamicTrajectoryViewer.svelte";
  import AdvantageViolins from "../components/AdvantageViolins.svelte";
  import AdvantageSpreadChart from "../components/AdvantageSpreadChart.svelte";
  import ComparativeBarChart from "../components/ComparativeBarChart.svelte";
  import ChartSkeleton from "../components/ChartSkeleton.svelte";
  import LineChart from "../components/LineChart.svelte";
  import ResizableTable from "../components/ResizableTable.svelte";
  import ChartControls from "../components/ChartControls.svelte";
  import MetricsRangeDropdown from "../components/MetricsRangeDropdown.svelte";
  import ZoomOutButton from "../components/ZoomOutButton.svelte";
  import RunMetricsPanel from "../components/RunMetricsPanel.svelte";
  import { toEpochSeconds } from "../lib/format.js";
  import {
    getTimeRangeParams,
    resolveTimeRange,
    rolloutDomainToTimeRange,
    rolloutTimeKnots,
    rolloutToTime,
    timeToRollout,
  } from "../lib/timeRange.js";
  import {
    fetchRun,
    fetchRunRollouts,
    fetchRollout,
    fetchRunTimings,
    fetchRunAdvantages,
    fetchRunAdvantageStep,
    fetchRunLogs,
    fetchRunMetrics,
  } from "../lib/api.js";
  import { formatMetricValue } from "../lib/metricSeries.js";
  import { groupByRollout, rolloutIndex, rolloutScores } from "../lib/rolloutGrouping.js";
  import { PERCENTILE_LINES, percentileRowFields } from "../lib/percentileLines.js";
  import {
    MAX_TERMINAL_TIMING_FAILURES,
    TERMINAL_TIMING_SETTLE_WINDOW_MS,
    terminalTimingReadSettled,
    timingReadFingerprint,
    updateTimingFailureState,
  } from "../lib/timing_retry.js";
  import {
    rolloutIdForTimingKey,
    shouldShowTimingSection,
    timingIsAsync,
    timingRunStart,
  } from "../lib/timing.js";

  // Number of historical log lines requested per page.
  const HIST_PAGE = 500;
  // Maximum number of historical log lines retained in the browser.
  const HIST_BUFFER_MAX = 2000;

  /** @typedef {"summary" | "metrics" | "rollouts" | "logs"} TabId */
  const DETAIL_TABS = new Set(["summary", "metrics", "rollouts", "logs"]);
  const DEFAULT_TAB = "summary";

  function parseTabFromUrl() {
    if (typeof window === "undefined") return DEFAULT_TAB;
    const raw = new URLSearchParams(window.location.search).get("tab");
    return DETAIL_TABS.has(raw) ? /** @type {TabId} */ (raw) : DEFAULT_TAB;
  }

  function urlForTab(tab) {
    const url = new URL(window.location.href);
    url.searchParams.set("tab", DETAIL_TABS.has(tab) ? tab : DEFAULT_TAB);
    return `${url.pathname}${url.search}${url.hash}`;
  }

  function locationKey() {
    return `${window.location.pathname}${window.location.search}${window.location.hash}`;
  }

  let {
    runId,
    initialRun = null,
    modelName,
    getStatus,
    getFrameworkStatus,
    showFrameworkStatus,
    fmtDuration,
    onBack,
    // "Collapse" drops the full detail page back to the list as a summary drawer.
    onCollapse,
    // When rendered inside the expanded run drawer the surrounding UI already
    // shows the header/title/summary, so we hide them and render only the
    // unique rollouts + logs content.
    embedded = false,
  } = $props();

  let run = $state(null);
  let runLoading = $state(false);
  let runError = $state("");
  // A run id that isn't in the metadata volume won't appear later, so the
  // 5s poll stops instead of refetching the same 404 for as long as the page
  // stays open.
  let runMissing = $state(false);

  async function loadRun(id, parentSignal) {
    if (parentSignal.aborted) return;
    const controller = new AbortController();
    let timedOut = false;
    const abortRequest = () => controller.abort();
    parentSignal.addEventListener("abort", abortRequest, { once: true });
    const timeout = window.setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, 30000);

    runLoading = true;
    try {
      const nextRun = await fetchRun(id, { signal: controller.signal });
      if (parentSignal.aborted) return;
      if (nextRun === null) {
        run = null;
        runMissing = true;
        runError = `Training run "${id}" was not found.`;
        return;
      }
      run = nextRun;
      runError = "";
    } catch (err) {
      if (parentSignal.aborted) return;
      if (!run) {
        runError = timedOut
          ? "Run request timed out after 30 seconds."
          : String(err?.message || err);
      }
    } finally {
      window.clearTimeout(timeout);
      parentSignal.removeEventListener("abort", abortRequest);
      if (!parentSignal.aborted) runLoading = false;
    }
  }

  // Status as a primitive so effects depending on it don't re-run every time
  // the auto-refresh hands us a new `run` object with the same status (which
  // would otherwise tear down and rebuild the log stream, flashing the tail).
  let runStatus = $derived(String(run?.status || "").toLowerCase());
  let metricUrl = $derived.by(() => {
    const directUrl =
      run?.train_result?.metric_url ||
      run?.config_summary?.metric_url ||
      run?.train_result?.wandb_url ||
      run?.config_summary?.wandb_url ||
      "";
    if (directUrl) return directUrl;

    const project =
      run?.config_summary?.metric_project || run?.config_summary?.wandb_project || "";
    return project ? `https://wandb.ai/home?search=${encodeURIComponent(project)}` : "";
  });
  // Latest-attempt external metrics link: attempt links sort ascending, so
  // the highest attempt wins; a bare metric_url is the fallback.
  let metricLink = $derived.by(() => {
    const raw = run?.metric_links?.length ? run.metric_links : run?.wandb_links;
    const links = raw?.length ? raw : metricUrl ? [{ url: metricUrl }] : [];
    return links.length
      ? links.reduce((best, link) =>
          (link?.attempt ?? 0) > (best?.attempt ?? 0) ? link : best,
        )
      : null;
  });

  $effect(() => {
    const id = runId;
    run = initialRun?.run_id === id ? initialRun : null;
    runError = "";
    runMissing = false;
    if (!id) {
      runLoading = false;
      return;
    }

    const controller = new AbortController();
    void loadRun(id, controller.signal);
    const interval = window.setInterval(() => {
      if (runMissing || runLoading || (runStatus && runStatus !== "running"))
        return;
      void loadRun(id, controller.signal);
    }, 5000);

    return () => {
      controller.abort();
      window.clearInterval(interval);
    };
  });

  // Active tab: "summary" | "rollouts" | "logs". One-way sync with the URL:
  // init/popstate/runId read URL → activeTab; selectTab writes pushState.
  let activeTab = $state(/** @type {TabId} */ (DEFAULT_TAB));
  let isSftRun = $derived(run?.training_type === "sft");

  function selectTab(tab) {
    const next = DETAIL_TABS.has(tab) ? /** @type {TabId} */ (tab) : DEFAULT_TAB;
    activeTab = next;
    if (embedded || typeof window === "undefined") return;
    const target = urlForTab(next);
    if (target !== locationKey()) {
      history.pushState({}, "", target);
    }
  }

  onMount(() => {
    if (embedded) {
      activeTab = DEFAULT_TAB;
    } else {
      const tab = parseTabFromUrl();
      activeTab = tab;
      const target = urlForTab(tab);
      if (target !== locationKey()) {
        history.replaceState({}, "", target);
      }
    }

    const onPopState = () => {
      if (embedded) return;
      activeTab = parseTabFromUrl();
    };
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  });

  $effect(() => {
    runId;
    if (embedded || typeof window === "undefined") return;
    activeTab = parseTabFromUrl();
  });

  $effect(() => {
    if (!isSftRun || activeTab !== "rollouts") return;
    activeTab = DEFAULT_TAB;
    if (!embedded) history.replaceState({}, "", urlForTab(DEFAULT_TAB));
  });

  // With an external metrics link the Metrics tab is a link, not a tab.
  $effect(() => {
    if (!metricLink || activeTab !== "metrics") return;
    activeTab = DEFAULT_TAB;
    if (!embedded) history.replaceState({}, "", urlForTab(DEFAULT_TAB));
  });

  function formatMean(value) {
    if (typeof value !== "number" || !Number.isFinite(value)) return "—";
    return value.toFixed(3);
  }

  function resumeBadge(run) {
    const state = run?.resume_state;
    if (!state) return "";
    const parts = [];
    if (state.attempt_count > 1) parts.push(`attempt ${state.attempt_count}`);
    if (state.resumed_from_checkpoint) {
      parts.push(
        state.resume_from_iteration != null
          ? `resumed @ ${state.resume_from_iteration}`
          : "resumed",
      );
    }
    return parts.join(" · ");
  }

  // Only the latest attempt's checkpoint is recorded, so earlier boundaries are
  // labelled with their attempt number alone.
  let attemptMarkers = $derived.by(() => {
    const state = run?.resume_state;
    if (!state || !(state.attempt_count > 1)) return [];
    const latest = Number(state.last_attempt_started_at);
    const starts = (Array.isArray(state.attempt_starts) ? state.attempt_starts : [])
      .map(Number)
      .filter(Number.isFinite);
    if (Number.isFinite(latest) && !starts.includes(latest)) starts.push(latest);
    starts.sort((a, b) => a - b);
    const first = starts.length > state.attempt_count ? 0 : state.attempt_count - starts.length;
    return starts
      .map((at, index) => ({ at, attempt: first + index + 1 }))
      .filter((marker) => marker.attempt > 1)
      .map(({ at, attempt }) => {
        const rawStep =
          at === latest && state.resumed_from_checkpoint
            ? state.resume_from_iteration
            : null;
        const step = rawStep != null ? Number(rawStep) : null;
        const completedStep =
          step != null && Number.isFinite(step) ? step + 1 : null;
        return {
          at,
          label: `attempt ${attempt}${completedStep != null ? ` · resumed after step ${completedStep}` : ""}`,
        };
      });
  });

  // ── Rollouts (auto-refresh while run is running) ─────────────────────
  let rolloutSummaries = $state([]);
  let rolloutsLoading = $state(false);
  let rolloutsError = $state("");
  let expandedRolloutId = $state(null);
  let expandedRollout = $state(null);
  let expandedRolloutLoading = $state(false);
  const rolloutColumns = [
    { key: "step", label: "Step", width: 72, minWidth: 56 },
    { key: "mean", label: "Mean reward", width: 118, minWidth: 96 },
    { key: "rollouts", label: "Rollouts", width: 80, minWidth: 64 },
    { key: "when", label: "When", width: 88, minWidth: 64 },
  ];

  // Per-step advantage distribution summaries (one row per step, each with the
  // step's overall stats + quantiles) — drives the advantage fan chart.
  let advantageSteps = $state([]);
  let hasAdvantages = $derived(advantageSteps.length > 0);

  let lossPoints = $state(null);
  let lossError = $state("");
  let lossStale = $state(false);

  const BUCKET_COUNT = 12;
  let activeBucket = $state(null); // histogram bucket index, or null
  let activeSamplePos = $state(0); // position within the active bucket's list

  let sampleDist = $derived.by(() => {
    const samples = expandedRollout?.samples || [];
    const rollouts = groupByRollout(samples);
    if (!rollouts.length) return null;
    const scores = rolloutScores(samples, rollouts);
    let lo = Infinity;
    let hi = -Infinity;
    for (const v of scores) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    const count = lo === hi ? 1 : BUCKET_COUNT;
    const span = hi - lo || 1;
    const buckets = Array.from({ length: count }, () => []);
    rollouts.forEach((positions, r) => {
      let b = count === 1 ? 0 : Math.floor(((scores[r] - lo) / span) * count);
      b = Math.max(0, Math.min(count - 1, b));
      buckets[b].push({ positions, score: scores[r] });
    });
    const maxCount = Math.max(...buckets.map((b) => b.length), 1);
    return {
      lo,
      hi,
      count,
      span,
      buckets,
      maxCount,
      total: rollouts.length,
      sampleCount: samples.length,
    };
  });

  let distSummary = $derived(!sampleDist ? "" : plural(sampleDist.total, "rollout"));

  function plural(n, unit) {
    return `${n} ${unit}${n === 1 ? "" : "s"}`;
  }

  function bucketLabel(bucket, b) {
    return `${plural(bucket.length, "rollout")} · reward ${bucketRange(b)}`;
  }

  function bucketRange(b) {
    const d = sampleDist;
    if (!d) return "";
    if (d.count === 1) return formatMean(d.lo);
    const step = d.span / d.count;
    return `${formatMean(d.lo + b * step)}–${formatMean(d.lo + (b + 1) * step)}`;
  }

  function openBucket(b) {
    const d = sampleDist;
    if (!d || !d.buckets[b]?.length) return;
    activeBucket = b;
    activeSamplePos = 0;
  }

  function closeBucket() {
    activeBucket = null;
    activeSamplePos = 0;
  }

  function stepSample(delta) {
    const d = sampleDist;
    if (!d || activeBucket == null) return;
    const list = d.buckets[activeBucket] || [];
    if (!list.length) return;
    activeSamplePos = Math.max(0, Math.min(list.length - 1, activeSamplePos + delta));
  }

  // A prompt group shares one screenshot: bytes on the first sample as `image`, the
  // rest carry only `image_ref`.
  let rolloutImages = $derived.by(() => {
    const byRef = {};
    for (const s of expandedRollout?.samples ?? []) {
      const meta = s?.metadata;
      if (meta?.image_ref && meta.image) byRef[meta.image_ref] = meta.image;
    }
    return byRef;
  });

  function sampleImage(sample) {
    const meta = sample?.metadata;
    if (!meta) return null;
    return meta.image ?? (meta.image_ref ? rolloutImages[meta.image_ref] : null) ?? null;
  }

  // The rollout currently shown in the viewer (or null when no bucket is open).
  let activeSample = $derived.by(() => {
    const d = sampleDist;
    if (!d || activeBucket == null) return null;
    const list = d.buckets[activeBucket] || [];
    const entry = list[activeSamplePos];
    if (!entry) return null;
    const sample = expandedRollout.samples[entry.positions[0]];
    return {
      sample,
      samples: entry.positions.map((p) => expandedRollout.samples[p]),
      score: entry.score,
      image: sampleImage(sample),
      pos: activeSamplePos,
      count: list.length,
    };
  });

  function onSampleKeydown(e) {
    if (activeBucket == null) return;
    const tag = (e.target?.tagName || "").toLowerCase();
    if (tag === "input" || tag === "textarea") return;
    if (e.key === "ArrowLeft") {
      e.preventDefault();
      stepSample(-1);
    } else if (e.key === "ArrowRight") {
      e.preventDefault();
      stepSample(1);
    }
  }

  // `imageHandling`: "ignore" | "refs_only" | "resolve".
  function sampleToPayload(s, imageHandling = "ignore") {
    let metadata = s.metadata || null;
    if (imageHandling === "resolve" && metadata?.image_ref && !metadata.image) {
      // Only add bytes if the lookup resolved — the carrier sample may not be loaded.
      const resolved = sampleImage(s);
      if (resolved) metadata = { ...metadata, image: resolved };
    } else if (imageHandling === "refs_only" && metadata?.image && metadata.image_ref) {
      // Bytes travel once in the payload's `images` map; keep only the ref here.
      const { image, ...rest } = metadata;
      metadata = rest;
    }
    return {
      score: s.score,
      rollout_index: rolloutIndex(s),
      sample_index: s.sample_index ?? null,
      group_index: s.group_index ?? null,
      prompt: s.prompt || null,
      response: s.response || null,
      thinking: s.thinking || null,
      raw_response: s.raw_response || null,
      raw_prompt: s.raw_prompt || null,
      trace: s.trace || null,
      metadata,
    };
  }

  function downloadSampleTrajectory() {
    if (!activeSample) return;
    const turns = activeSample.samples;
    const payload =
      turns.length === 1
        ? sampleToPayload(turns[0], "resolve")
        : {
            mean: activeSample.score,
            turns: turns.length,
            samples: turns.map((s) => sampleToPayload(s, "resolve")),
          };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    const rollout = expandedRolloutId ?? 0;
    a.download = `trajectory_r${rollout}_rollout${rolloutIndex(activeSample.sample) ?? activeSample.pos}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function downloadAllTrajectories() {
    if (!expandedRollout?.samples?.length) return;
    const rollout = expandedRolloutId ?? 0;
    const samples = expandedRollout.samples;
    const groups = groupByRollout(samples);
    const scores = rolloutScores(samples, groups);
    const payload = {
      training_run_id: runId,
      rollout_id: rollout,
      total: samples.length,
      rollouts: groups.length,
      n_samples_per_prompt: expandedRollout.n_samples_per_prompt ?? null,
      mean: scores.reduce((a, v) => a + v, 0) / scores.length,
      // Shared images, keyed by the `metadata.image_ref` each sample carries.
      images: rolloutImages,
      samples: samples.map((s) => sampleToPayload(s, "refs_only")),
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `rollout_${runId}_r${rollout}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  async function loadRollouts(signal) {
    if (!runId) return;
    try {
      const wasEmpty = untrack(() => rolloutSummaries.length === 0);
      const rows = await fetchRunRollouts(runId, { signal });
      if (signal?.aborted) return;
      rolloutSummaries = rows;
      rolloutsError = "";

      // Reveal the first rollout
      if (wasEmpty && rolloutSummaries.length > 0 && expandedRolloutId === null) {
        toggleRolloutDetail(rolloutSummaries[0].rollout_id);
      }
    } catch (err) {
      if (signal?.aborted) return;
      // Keep the rollouts we already have on a transient poll failure — only
      // surface the error when there's nothing to show, so the charts/table
      // don't flip to an error message (and back) every flaky 5s poll.
      if (!rolloutSummaries.length) rolloutsError = String(err?.message || err);
    } finally {
      rolloutsLoading = false;
    }
  }

  let timingInFlight = null;
  let timingError = $state("");
  let timingFailures = $state(0);
  let terminalTimingFailures = $state(0);
  let timingStaleFailures = $state(0);
  let timingLoaded = $state(false);
  let terminalTimingStartedAt = $state(null);
  let terminalTimingFingerprint = $state(null);
  let terminalTimingIdenticalReads = $state(0);
  const MAX_TERMINAL_TIMING_STALE_READS = 3;

  async function loadTimings(parentSignal) {
    if (
      !runId ||
      (timingInFlight?.runId === runId && !timingInFlight.signal.aborted)
    )
      return;
    const controller = new AbortController();
    const abortRequest = () => controller.abort();
    parentSignal.addEventListener("abort", abortRequest, { once: true });
    const timeout = window.setTimeout(() => controller.abort(), 10000);
    timingInFlight = { runId, signal: controller.signal };
    const status = String(untrack(() => run?.status) || "").toLowerCase();
    const terminal = status && status !== "running";
    try {
      const timings = await fetchRunTimings(runId, {
        signal: controller.signal,
      });
      if (controller.signal.aborted) return;
      const serialized = timingReadFingerprint(timings);
      if (serialized !== runTimingsSerialized) {
        runTimingsSerialized = serialized;
        runTimings = timings;
      }
      const timingStale = Boolean(timings?.metadata?.timing_stale);
      if (!terminal) {
        terminalTimingStartedAt = null;
        terminalTimingFingerprint = null;
        terminalTimingIdenticalReads = 0;
        timingLoaded = false;
      } else if (terminalTimingStartedAt == null) {
        terminalTimingStartedAt = Date.now();
      }
      if (terminal && !timingStale) {
        const settled = terminalTimingReadSettled({
          previousFingerprint: terminalTimingFingerprint,
          fingerprint: serialized,
          identicalReads: terminalTimingIdenticalReads,
          startedAt: terminalTimingStartedAt,
        });
        terminalTimingFingerprint = serialized;
        terminalTimingIdenticalReads = settled.identicalReads;
        timingLoaded = settled.settled;
      } else if (
        terminal &&
        terminalTimingStartedAt != null &&
        Date.now() - terminalTimingStartedAt >= TERMINAL_TIMING_SETTLE_WINDOW_MS
      ) {
        timingLoaded = true;
      }
      const nextFailures = updateTimingFailureState(
        {
          runningFailures: timingFailures,
          terminalFailures: terminalTimingFailures,
          staleFailures: timingStaleFailures,
        },
        { terminal, success: true, stale: timingStale },
      );
      timingFailures = nextFailures.runningFailures;
      terminalTimingFailures = nextFailures.terminalFailures;
      timingStaleFailures = nextFailures.staleFailures;
      if (!timingStale) timingError = "";
    } catch {
      if (parentSignal.aborted) return;
      const nextFailures = updateTimingFailureState(
        {
          runningFailures: timingFailures,
          terminalFailures: terminalTimingFailures,
          staleFailures: timingStaleFailures,
        },
        { terminal },
      );
      timingFailures = nextFailures.runningFailures;
      terminalTimingFailures = nextFailures.terminalFailures;
      if (terminal) {
        if (terminalTimingFailures >= MAX_TERMINAL_TIMING_FAILURES) {
          timingError = "Timing unavailable right now.";
        }
      } else if (timingFailures >= 2) {
        timingError = "Timing unavailable right now.";
      }
    } finally {
      window.clearTimeout(timeout);
      parentSignal.removeEventListener("abort", abortRequest);
      if (timingInFlight?.signal === controller.signal) timingInFlight = null;
    }
  }

  async function loadAdvantages(signal) {
    if (!runId) return;
    try {
      const rows = await fetchRunAdvantages(runId, { signal });
      if (signal?.aborted) return;
      advantageSteps = rows;
      return true;
    } catch {
      // Advantage data is optional (only present once the slime hook has
      // reported a step) — keep whatever we have on a transient failure.
    }
  }

  async function loadLoss(signal) {
    if (!runId) return;
    try {
      const payload = await fetchRunMetrics(runId, { signal });
      if (signal?.aborted) return;
      lossPoints = (payload.series?.["train/loss"] ?? []).map(([x, y, t]) => ({ x, y, t }));
      lossStale = payload.stale ?? false;
      lossError = "";
      return true;
    } catch (err) {
      if (!signal?.aborted) lossError = String(err?.message || err);
    }
  }

  // Reset rollout state when the run changes (separate from the fetch effect
  // so flipping between the summary/rollouts tabs doesn't clear what's loaded).
  $effect(() => {
    runId;
    rolloutSummaries = [];
    rolloutsError = "";
    runTimings = {};
    runTimingsSerialized = "{}";
    timingFailures = 0;
    terminalTimingFailures = 0;
    timingStaleFailures = 0;
    timingLoaded = false;
    terminalTimingStartedAt = null;
    terminalTimingFingerprint = null;
    terminalTimingIdenticalReads = 0;
    timingError = "";
    expandedRolloutId = null;
    expandedRollout = null;
    advantageSteps = [];
    lossPoints = null;
    lossError = "";
    lossStale = false;
    closeBucket();
  });

  $effect(() => {
    const id = runId;
    if (!id || runMissing || activeTab !== "summary") return;

    const load = isSftRun ? loadLoss : loadAdvantages;
    const controller = new AbortController();
    let finalLoadDone = false;
    let inFlight = false;
    const poll = async () => {
      const status = String(run?.status || "").toLowerCase();
      const final = status && status !== "running" && !(isSftRun && lossStale);
      if (inFlight || (final && finalLoadDone)) return;
      inFlight = true;
      const ok = await load(controller.signal);
      inFlight = false;
      if (final && ok) finalLoadDone = true;
    };
    void untrack(poll);
    const interval = window.setInterval(poll, 5000);

    return () => {
      controller.abort();
      window.clearInterval(interval);
    };
  });

  // Lazy load: only fetch rollout summaries while a tab that needs them is
  // active (the chart on Summary, the table on Rollouts).
  $effect(() => {
    const id = runId;
    const tab = activeTab;
    if (!id || runMissing || (tab !== "summary" && tab !== "rollouts")) return;

    const controller = new AbortController();
    if (!isSftRun) {
      rolloutsLoading = true;
      void loadRollouts(controller.signal);
    }
    void loadTimings(controller.signal);

    // Poll while the run is active so new rollouts stream in.
    const interval = window.setInterval(() => {
      const status = String(run?.status || "").toLowerCase();
      if (
        status &&
        status !== "running" &&
        (timingLoaded ||
          terminalTimingFailures >= MAX_TERMINAL_TIMING_FAILURES ||
          timingStaleFailures >= MAX_TERMINAL_TIMING_STALE_READS)
      )
        return;
      if (!isSftRun && (!status || status === "running")) {
        void loadRollouts(controller.signal);
      }
      void loadTimings(controller.signal);
    }, 5000);

    return () => {
      controller.abort();
      window.clearInterval(interval);
    };
  });

  let runTimings = $state({});
  let runTimingsSerialized = $state("{}");
  let showTimingSection = $derived(shouldShowTimingSection(runTimings));
  let timelineAsync = $derived(timingIsAsync(runTimings));
  let timelineRunOrigin = $derived(timingRunStart(runTimings));
  let stepTimingIds = $derived(
    Object.keys(runTimings).filter((id) => rolloutIdForTimingKey(id) !== null),
  );

  async function toggleRolloutDetail(rolloutId) {
    if (!runId) return;
    if (expandedRolloutId === rolloutId) {
      expandedRolloutId = null;
      expandedRollout = null;
      closeBucket();
      return;
    }
    expandedRolloutId = rolloutId;
    expandedRollout = null;
    closeBucket();
    expandedRolloutLoading = true;
    try {
      const detail = await fetchRollout(runId, rolloutId);
      if (expandedRolloutId === rolloutId) {
        expandedRollout = detail;
        const d = sampleDist;
        const first = d ? d.buckets.findIndex((b) => b.length > 0) : -1;
        if (first >= 0) openBucket(first);
      }
    } finally {
      if (expandedRolloutId === rolloutId) {
        expandedRolloutLoading = false;
      }
    }
  }

  // ── Live Modal log stream (SSE, pure pass-through) ───────────────────
  const LOG_BUFFER_MAX = 2000;
  let logLines = $state([]); // [{task_id, line, ts}]
  let logState = $state("idle"); // idle | streaming | paused | done | error | reconnecting
  let logError = $state("");
  let logDropped = $state(0); // server-side rate-capped lines (cumulative since reconnect)
  let logTailEl = $state(null);

  // User controls
  let logPaused = $state(false);
  let logSearch = $state("");
  let logSearchInput = $state(""); // debounced into logSearch
  let logRateCap = $state(0); // 0 = no cap
  let logFollow = $state(true); // auto-scroll to bottom

  // The backend emits one SSE event per log line. Mutating `logLines` (and
  // auto-scrolling) on every message means hundreds of synchronous reactive
  // updates per second under a chatty run, which freezes the tab. Instead we
  // buffer incoming lines and flush them into `logLines` once per animation
  // frame — coalescing a burst into a single render + scroll.
  let pendingLogLines = []; // plain array, not reactive
  let logFlushHandle = null; // requestAnimationFrame handle
  let logSeq = 0; // monotonic id for stable keying

  function flushPendingLogs() {
    logFlushHandle = null;
    if (!pendingLogLines.length) return;
    const next = logLines.length
      ? logLines.concat(pendingLogLines)
      : pendingLogLines;
    pendingLogLines = [];
    logLines = next.length > LOG_BUFFER_MAX ? next.slice(-LOG_BUFFER_MAX) : next;
    if (logFollow && logTailEl) {
      // Scroll after Svelte has flushed the new rows to the DOM.
      queueMicrotask(() => {
        if (logTailEl) logTailEl.scrollTop = logTailEl.scrollHeight;
      });
    }
  }

  function scheduleLogFlush() {
    if (logFlushHandle != null) return;
    logFlushHandle = requestAnimationFrame(flushPendingLogs);
  }

  function resetLogBuffer() {
    if (logFlushHandle != null) {
      cancelAnimationFrame(logFlushHandle);
      logFlushHandle = null;
    }
    pendingLogLines = [];
  }

  // Debounce search input → URL
  $effect(() => {
    const value = logSearchInput;
    const handle = window.setTimeout(() => {
      logSearch = value.trim();
    }, 350);
    return () => window.clearTimeout(handle);
  });

  $effect(() => {
    const id = runId;
    const tab = activeTab;
    // Depend on the primitive status (not `run`) so a same-status refresh
    // doesn't tear down and rebuild the stream.
    const status = runStatus;
    // Re-create the EventSource whenever any of the connection params change.
    const search = logSearch;
    const rate = logRateCap;
    const paused = logPaused;

    resetLogBuffer();
    logLines = [];
    logState = "idle";
    logError = "";
    logDropped = 0;
    // Lazy load: only open the log stream while the Logs tab is active.
    if (tab !== "logs" || !id || !status || status !== "running" || paused) {
      if (tab === "logs" && paused) logState = "paused";
      return;
    }

    const params = new URLSearchParams();
    if (search) params.set("search", search);
    if (rate > 0) params.set("max_lines_per_sec", String(rate));
    const qs = params.toString();
    const url =
      `/api/runs/${encodeURIComponent(id)}/logs/stream` +
      (qs ? `?${qs}` : "");

    const es = new EventSource(url);
    logState = "streaming";

    es.onopen = () => {
      logState = "streaming";
      logError = "";
    };

    es.onmessage = (evt) => {
      if (logState !== "streaming") {
        logState = "streaming";
        logError = "";
      }
      try {
        const payload = JSON.parse(evt.data);
        const line = String(payload.line || "");
        if (!line) return;
        const parts = line.split(/\r?\n/);
        const task_id = payload.task_id || "";
        const ts = payload.ts || Date.now();
        for (const p of parts) {
          if (!p.length) continue;
          pendingLogLines.push({ id: logSeq++, task_id, line: p, ts });
        }
        if (pendingLogLines.length > LOG_BUFFER_MAX) {
          pendingLogLines = pendingLogLines.slice(-LOG_BUFFER_MAX);
        }
        scheduleLogFlush();
      } catch {
        // ignore malformed payloads
      }
    };

    es.addEventListener("done", () => {
      logState = "done";
      es.close();
    });

    es.addEventListener("reconnect", (evt) => {
      try {
        const { reason } = JSON.parse(evt.data || "{}");
        logError = String(reason || "");
      } catch {
        logError = "";
      }
      logState = "reconnecting";
    });

    es.addEventListener("dropped", (evt) => {
      try {
        const { dropped } = JSON.parse(evt.data || "{}");
        logDropped += Number(dropped) || 0;
      } catch {}
    });

    es.addEventListener("error", (evt) => {
      try {
        const { error } = JSON.parse(evt.data || "{}");
        logError = String(error || "");
      } catch {
        logError = "";
      }
      logState = "error";
      es.close();
    });

    es.onerror = () => {
      if (logState === "streaming") logState = "reconnecting";
    };

    return () => {
      resetLogBuffer();
      try {
        es.close();
      } catch {}
    };
  });

  function toggleLogPaused() {
    logPaused = !logPaused;
  }

  function clearLogs() {
    resetLogBuffer();
    logLines = [];
    logDropped = 0;
  }

  // ── Historical logs ─────
  let isRunning = $derived(runStatus === "running");

  let histLines = $state([]), histNewerWindows = $state([]);
  let histLoading = $state(false),
    histLoadingOlder = $state(false),
    histLoadingNewer = $state(false);
  let histError = $state(""), histHasMore = $state(false);
  let histNextUntil = $state(null), histTailEl = $state(null);
  let histSeq = 0, histLastScrollTop = 0;
  let histController = null, histRestoringScroll = false;

  let histRangeInput = $state({ since: "", until: "" });
  let histRange = $state({ since: "", until: "" });

  function epochToLocalInput(epoch) {
    if (!epoch) return "";
    const d = new Date(Number(epoch) * 1000);
    const p = (x) => String(x).padStart(2, "0");
    return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
  }
  function localInputToEpoch(str) {
    if (!str || !str.trim()) return null;
    const ms = new Date(str.trim().replace(" ", "T")).getTime();
    return Number.isNaN(ms) ? null : Math.floor(ms / 1000);
  }

  function histRangeFromText(sinceText, untilText) {
    const since = localInputToEpoch(sinceText);
    const until = localInputToEpoch(untilText);
    return {
      since: since != null ? String(since) : "",
      // The picker has minute precision, so include the entire final minute.
      until: until != null ? String(until + 59.999999999) : "",
    };
  }

  $effect(() => {
    const { since, until } = histRangeInput;
    const handle = window.setTimeout(() => {
      histRange = histRangeFromText(since, until);
    }, 350);
    return () => window.clearTimeout(handle);
  });

  // Seed the pickers with the run's lifetime the first time we view a finished
  // run.
  let histPrefilledFor = null;
  $effect(() => {
    const id = runId;
    const running = isRunning;
    if (!id || running || histPrefilledFor === id) return;

    const startedAt = run?.started_at || run?.created_at || 0;
    const endedAt = run?.ended_at || run?.completed_at || 0;
    if (!startedAt && !endedAt) return;

    histPrefilledFor = id;
    const sinceText = epochToLocalInput(startedAt);
    const untilText = epochToLocalInput(endedAt);
    const range = histRangeFromText(sinceText, untilText);
    histRangeInput = { since: sinceText, until: untilText };
    histRange = range;
  });

  // Expand server entries into per-line rows (a single ClickHouse entry can
  // carry an embedded newline), matching how the live stream splits lines.
  function pushHistRows(target, entries) {
    for (const entry of entries) {
      const task_id = entry.task_id || "";
      const ts = entry.ts || 0;
      const ts_ns = entry.ts_ns || 0;
      for (const part of String(entry.line ?? "").split(/\r?\n/)) {
        if (!part.length) continue;
        target.push({ id: histSeq++, task_id, line: part, ts, ts_ns });
      }
    }
  }

  function resetHist() {
    histLines = [];
    histError = "";
    histHasMore = false;
    histNewerWindows = [];
    histNextUntil = null;
    histLoading = false;
    histLoadingOlder = false;
    histLoadingNewer = false;
    histLastScrollTop = 0;
    histRestoringScroll = false;
  }

  function histRowTime(row) {
    if (!row) return 0;
    if (row.ts_ns) return Number(row.ts_ns) / 1_000_000_000;
    return Number(row.ts) || 0;
  }

  function histCursorBefore(row) {
    if (!row) return null;
    if (row.ts_ns) return (Number(row.ts_ns) - 1) / 1_000_000_000;
    const ts = Number(row.ts) || 0;
    return ts > 0 ? ts - 0.000000001 : null;
  }

  // Capture a visible row and its exact viewport position. Restoring this
  // after replacing the bounded window avoids any perceptible jump.
  function captureHistAnchor() {
    const el = histTailEl;
    if (!el) return null;
    const containerTop = el.getBoundingClientRect().top;
    const rows = el.querySelectorAll("[data-hist-id]");
    for (const node of rows) {
      const rect = node.getBoundingClientRect();
      if (rect.bottom >= containerTop) {
        return {
          id: node.dataset.histId,
          top: rect.top,
          scrollTop: el.scrollTop,
        };
      }
    }
    return null;
  }

  async function restoreHistAnchor(anchor) {
    if (!histTailEl || !anchor) return;
    histRestoringScroll = true;
    await tick();
    if (!histTailEl) {
      histRestoringScroll = false;
      return;
    }
    const node = histTailEl.querySelector(`[data-hist-id="${anchor.id}"]`);
    if (node) {
      histTailEl.scrollTop += node.getBoundingClientRect().top - anchor.top;
    } else {
      histTailEl.scrollTop = anchor.scrollTop;
    }
    histLastScrollTop = histTailEl.scrollTop;
    requestAnimationFrame(() => {
      if (histTailEl) histLastScrollTop = histTailEl.scrollTop;
      histRestoringScroll = false;
    });
  }

  function rememberTrimmedNewer(kept, dropped) {
    if (!kept.length || !dropped.length) return;
    const since = histRowTime(kept[kept.length - 1]);
    const until = histRowTime(dropped[dropped.length - 1]);
    if (since > 0 && until >= since) {
      histNewerWindows = [...histNewerWindows, { since, until }];
    }
  }

  function markTrimmedOlder(kept, dropped) {
    if (!kept.length || !dropped.length) return;
    histHasMore = true;
    histNextUntil = histCursorBefore(kept[0]);
  }

  async function loadHistInitial(id, { search, since, until }, signal) {
    histLoading = true;
    histError = "";
    try {
      const data = await fetchRunLogs(id, {
        maxLines: HIST_PAGE,
        search,
        since,
        until,
        signal,
      });
      if (signal?.aborted) return;
      histNewerWindows = [];
      const rows = [];
      pushHistRows(rows, data.logs);
      const droppedOlder =
        rows.length > HIST_BUFFER_MAX
          ? rows.slice(0, rows.length - HIST_BUFFER_MAX)
          : [];
      histLines =
        rows.length > HIST_BUFFER_MAX
          ? rows.slice(-HIST_BUFFER_MAX)
          : rows;
      if (droppedOlder.length) {
        markTrimmedOlder(histLines, droppedOlder);
      } else {
        histHasMore = data.hasMore;
        histNextUntil = data.nextUntil;
      }
      // Land on the newest line, like the live tail does.
      queueMicrotask(() => {
        if (histTailEl) {
          histRestoringScroll = true;
          histTailEl.scrollTop = histTailEl.scrollHeight;
          histLastScrollTop = histTailEl.scrollTop;
          requestAnimationFrame(() => {
            histRestoringScroll = false;
          });
        }
      });
    } catch (err) {
      if (signal?.aborted) return;
      histError = String(err?.message || err);
    } finally {
      if (!signal?.aborted) histLoading = false;
    }
  }

  async function loadHistPage(direction, { since, until }) {
    const loadingOlder = direction === "older";
    if (loadingOlder) {
      histLoadingOlder = true;
    } else {
      histLoadingNewer = true;
    }
    histError = "";
    const anchor = captureHistAnchor();
    const signal = histController?.signal;
    try {
      const data = await fetchRunLogs(runId, {
        since,
        until,
        maxLines: HIST_PAGE,
        search: logSearch,
        signal,
      });
      if (signal?.aborted) return;
      const rows = [];
      pushHistRows(rows, data.logs);
      if (loadingOlder && !rows.length) {
        histHasMore = false;
        histNextUntil = null;
        return;
      }

      const adjacent =
        rows.length <= HIST_PAGE
          ? rows
          : loadingOlder
            ? rows.slice(-HIST_PAGE)
            : rows.slice(0, HIST_PAGE);
      const merged = loadingOlder
        ? adjacent.concat(histLines)
        : histLines.concat(adjacent);

      if (loadingOlder) {
        const dropped = merged.slice(HIST_BUFFER_MAX);
        histLines = merged.slice(0, HIST_BUFFER_MAX);
        rememberTrimmedNewer(histLines, dropped);
        if (rows.length > HIST_PAGE) {
          histHasMore = true;
          histNextUntil = histCursorBefore(adjacent[0]);
        } else {
          histHasMore = data.hasMore;
          histNextUntil = data.nextUntil;
        }
      } else {
        const dropped = merged.slice(
          0,
          Math.max(0, merged.length - HIST_BUFFER_MAX),
        );
        histLines = merged.slice(-HIST_BUFFER_MAX);
        markTrimmedOlder(histLines, dropped);
        histNewerWindows = histNewerWindows.slice(0, -1);
      }

      await restoreHistAnchor(anchor);
    } catch (err) {
      if (signal?.aborted) return;
      histError = String(err?.message || err);
    } finally {
      if (!signal?.aborted) {
        if (loadingOlder) {
          histLoadingOlder = false;
        } else {
          histLoadingNewer = false;
        }
      }
    }
  }

  function loadHistOlder() {
    if (
      !histHasMore ||
      histNextUntil == null ||
      histLoadingOlder ||
      histLoadingNewer ||
      histLoading
    ) {
      return;
    }
    return loadHistPage("older", {
      since: histRange.since,
      until: histNextUntil,
    });
  }

  function loadHistNewer() {
    if (
      !histNewerWindows.length ||
      histLoadingNewer ||
      histLoadingOlder ||
      histLoading
    ) {
      return;
    }
    const range = histNewerWindows[histNewerWindows.length - 1];
    return loadHistPage("newer", range);
  }

  // Load only in the direction the user moved. Scroll restoration after a
  // page swap is ignored, preventing older/newer fetches from ping-ponging.
  function onHistScroll() {
    const el = histTailEl;
    if (!el || histRestoringScroll) return;
    const top = el.scrollTop;
    const delta = top - histLastScrollTop;
    histLastScrollTop = top;
    if (delta < 0 && top <= 40) {
      void loadHistOlder();
      return;
    }
    if (delta > 0) {
      const distanceFromBottom = el.scrollHeight - top - el.clientHeight;
      if (distanceFromBottom <= 40) void loadHistNewer();
    }
  }

  function jumpHistToLatest() {
    if (!runId || histLoading || histLoadingOlder || histLoadingNewer) return;
    void loadHistInitial(
      runId,
      {
        search: logSearch,
        since: histRange.since,
        until: histRange.until,
      },
      histController?.signal,
    );
  }

  function resetHistRange() {
    const startedAt = run?.started_at || run?.created_at || 0;
    const endedAt = run?.ended_at || run?.completed_at || 0;
    histRangeInput = {
      since: epochToLocalInput(startedAt),
      until: epochToLocalInput(endedAt),
    };
  }

  // Load the newest page when the Logs tab opens on a finished run, and reload
  // when the debounced search/since/until change.
  $effect(() => {
    const id = runId;
    const tab = activeTab;
    const status = runStatus;
    const running = isRunning;
    const search = logSearch;
    const { since, until } = histRange;

    resetHist();
    if (tab !== "logs" || !id || !status || running) return;

    const controller = new AbortController();
    histController = controller;
    void loadHistInitial(id, { search, since, until }, controller.signal);
    return () => {
      controller.abort();
      if (histController === controller) histController = null;
    };
  });

  // ── Shared chart time range ─────
  // One `{ start, end, live }` window (epoch seconds) drives every
  // rollout-indexed chart on the summary tab, after the Modal dashboard's
  // metrics controls. The charts plot against rollout ids, so the window is
  // translated to an id range through each rollout's `created_at`.
  let chartRangeSelection = $state(null); // null → entire run
  let clockNow = $state(Date.now() / 1000);
  $effect(() => {
    const interval = window.setInterval(() => {
      clockNow = Date.now() / 1000;
    }, 5000);
    return () => window.clearInterval(interval);
  });
  $effect(() => {
    runId;
    chartRangeSelection = null;
  });

  let rolloutKnots = $derived(rolloutTimeKnots(rolloutSummaries));
  let lossTimestamps = $derived(
    (lossPoints ?? []).map((p) => p.t).filter((t) => Number.isFinite(t) && t > 0),
  );
  let chartRunStart = $derived.by(() => {
    const candidates = [
      toEpochSeconds(run?.started_at || run?.created_at),
      rolloutKnots[0]?.t,
      timelineRunOrigin,
      lossTimestamps.length ? Math.min(...lossTimestamps) : null,
    ].filter((t) => Number.isFinite(t) && t > 0);
    return candidates.length ? Math.min(...candidates) : clockNow;
  });
  // A finished run's clock stops when it did, so "Past 1 hour" reads as the
  // last hour of the run rather than an empty window.
  let chartNow = $derived.by(() => {
    if (isRunning) return clockNow;
    const ended = toEpochSeconds(run?.ended_at || run?.completed_at) ?? 0;
    const last = rolloutKnots[rolloutKnots.length - 1]?.t ?? 0;
    const lastLoss = lossTimestamps.length ? Math.max(...lossTimestamps) : 0;
    const stopped = Math.max(ended, last, lastLoss);
    return stopped > chartRunStart ? stopped : clockNow;
  });
  let chartRange = $derived(
    resolveTimeRange(chartRangeSelection, { runStart: chartRunStart, now: chartNow }),
  );
  let chartRangeParams = $derived(getTimeRangeParams(chartRange));
  let chartMaxDuration = $derived(Math.max(1, chartNow - chartRunStart));

  function setChartRange(next) {
    if (!next) {
      chartRangeSelection = null;
      return;
    }
    const start = Math.max(chartRunStart, next.start);
    const end = Math.min(chartNow, next.end);
    if (!(end > start)) return;
    const live = Boolean(next.live) && end >= chartNow;
    // A paused full-run window stays explicit so it freezes instead of
    // tailing like "entire run" does.
    if (live && start <= chartRunStart) {
      chartRangeSelection = null;
      return;
    }
    chartRangeSelection = { start, end, live };
  }

  // Rollout-id window the charts show; null while the whole run is visible.
  let chartXDomain = $derived.by(() => {
    if (chartRange.entireRun) return null;
    const lo = timeToRollout(rolloutKnots, chartRange.start);
    const hi = timeToRollout(rolloutKnots, chartRange.end);
    return hi > lo ? [lo, hi] : null;
  });

  function onChartDomainChange(domain) {
    const next = rolloutDomainToTimeRange(rolloutKnots, domain, {
      runStart: chartRunStart,
      now: chartNow,
    });
    if (next !== undefined) setChartRange(next);
  }

  function inChartDomain(x) {
    return !chartXDomain || (x >= chartXDomain[0] && x <= chartXDomain[1]);
  }

  // Rollout id <-> wall clock for the charts' time axes; needs two timed
  // rollouts to interpolate between.
  let chartXToTime = $derived(
    rolloutKnots.length >= 2 ? (x) => rolloutToTime(rolloutKnots, x) : null,
  );
  let chartTimeToX = $derived(
    rolloutKnots.length >= 2 ? (t) => timeToRollout(rolloutKnots, t) : null,
  );

  function _seriesStats(getY) {
    const rows = rolloutSummaries.filter((r) => inChartDomain(Number(r.rollout_id) || 0));
    if (!rows.length) return null;
    const values = rows.map(getY);
    return {
      min: Math.min(...values),
      max: Math.max(...values),
      latest: values[values.length - 1],
    };
  }

  let chartStats = $derived(_seriesStats((r) => Number(r.mean) || 0));

  let lossInRange = $derived(
    (lossPoints ?? []).filter(
      (p) =>
        chartRange.entireRun ||
        p.t == null ||
        (p.t >= chartRange.start && p.t <= chartRange.end),
    ),
  );
  let lossStats = $derived.by(() => {
    if (!lossInRange.length) return null;
    const values = lossInRange.map((p) => p.y);
    return { min: Math.min(...values), max: Math.max(...values), latest: values[values.length - 1] };
  });
  let rewardChartData = $derived(
    rolloutSummaries.map((r) => ({
      x: Number(r.rollout_id) || 0,
      y: Number(r.mean) || 0,
      rollout_id: Number(r.rollout_id) || 0,
      ...percentileRowFields(r.reward_stats),
    })),
  );

  // Custom reward-function tags: any numeric sample.metadata key a reward fn
  // sets (e.g. sample.metadata["step_K"] = k) gets aggregated per rollout on
  // the backend (TrainingRolloutResult.tag_stats) and charted here the same
  // way as the reward line above — one chart per discovered tag name.
  let customTagNames = $derived(
    Array.from(
      new Set(rolloutSummaries.flatMap((r) => Object.keys(r.tag_stats || {}))),
    ).sort(),
  );

  function tagChartData(tag) {
    return rolloutSummaries
      .filter((r) => r.tag_stats?.[tag])
      .map((r) => ({
        x: Number(r.rollout_id) || 0,
        y: Number(r.tag_stats[tag].mean) || 0,
        rollout_id: Number(r.rollout_id) || 0,
        ...percentileRowFields(r.tag_stats[tag]),
      }));
  }

  function tagChartStats(tag) {
    const values = rolloutSummaries
      .filter((r) => r.tag_stats?.[tag] && inChartDomain(Number(r.rollout_id) || 0))
      .map((r) => Number(r.tag_stats[tag].mean) || 0);
    if (!values.length) return null;
    return { min: Math.min(...values), max: Math.max(...values), latest: values[values.length - 1] };
  }

  // Score-distribution comparison: the first rollout (step 0) vs the most
  // recent one, to see how sample scores shifted over training.
  let firstRolloutId = $derived(
    rolloutSummaries.length
      ? Math.min(...rolloutSummaries.map((r) => Number(r.rollout_id) || 0))
      : null,
  );
  let lastRolloutId = $derived(
    rolloutSummaries.length
      ? Math.max(...rolloutSummaries.map((r) => Number(r.rollout_id) || 0))
      : null,
  );
  let scoreDist = $state(null);

  // Grouped-bar histogram of two value sets (first vs latest), sharing a common
  // [lo, hi] range so the bars line up. Drives both the score and advantage
  // before/after comparison charts.
  function buildDist(firstValues, lastValues, firstId, lastId) {
    if (!firstValues.length && !lastValues.length) return null;
    // Loop instead of Math.min(...arr): advantage arrays can exceed the
    // engine's max argument count and make the spread throw a RangeError.
    let lo = Infinity;
    let hi = -Infinity;
    for (const v of firstValues) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    for (const v of lastValues) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    const n = lo === hi ? 1 : 12;
    const span = hi - lo || 1;
    const bins = Array.from({ length: n }, (_, i) => ({
      lo: lo + (i / n) * span,
      hi: lo + ((i + 1) / n) * span,
      first: 0,
      last: 0,
    }));
    const idx = (v) => Math.max(0, Math.min(n - 1, Math.floor(((v - lo) / span) * n)));
    for (const v of firstValues) bins[idx(v)].first += 1;
    for (const v of lastValues) bins[idx(v)].last += 1;
    const max = Math.max(1, ...bins.map((b) => Math.max(b.first, b.last)));
    return { bins, max, lo, hi, firstId, lastId };
  }

  function buildScoreDist(firstSamples, lastSamples, firstId, lastId) {
    return buildDist(rolloutScores(firstSamples), rolloutScores(lastSamples), firstId, lastId);
  }

  // Adapt a buildDist result into ComparativeBarChart inputs: one category per
  // bin (value range, shown only in the tooltip — the endpoint axis is drawn
  // separately), and a series per rollout endpoint. The latest series is dropped
  // when the first and latest rollout are the same one.
  function distCategories(dist) {
    return dist.bins.map((b) => `${formatMean(b.lo)}–${formatMean(b.hi)}`);
  }

  function distSeries(dist) {
    const series = [
      {
        name: `rollout ${dist.firstId}`,
        color: "var(--color-c-gray-40)",
        values: dist.bins.map((b) => b.first),
      },
    ];
    if (dist.firstId !== dist.lastId) {
      series.push({
        name: `latest (rollout ${dist.lastId})`,
        color: "var(--accent)",
        values: dist.bins.map((b) => b.last),
      });
    }
    return series;
  }

  // Fetch the two rollouts' samples only when the endpoints change (a new step
  // lands), not on every 5s poll — the payloads are large.
  $effect(() => {
    if (activeTab !== "summary") return;
    const id = runId;
    const fId = firstRolloutId;
    const lId = lastRolloutId;
    if (!id || fId == null || lId == null) {
      scoreDist = null;
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const first = await fetchRollout(id, fId);
        const last = fId === lId ? first : await fetchRollout(id, lId);
        if (cancelled) return;
        scoreDist = buildScoreDist(first?.samples || [], last?.samples || [], fId, lId);
      } catch {
        if (!cancelled) scoreDist = null;
      }
    })();
    return () => {
      cancelled = true;
    };
  });

  // Advantage-distribution comparison: raw advantages of the first vs latest
  // step, bucketed the same way as the score comparison. The merged-step
  // payloads are large, so only refetch when the endpoint ids change.
  let advantageDist = $state(null);

  function mergedAdvantages(merged) {
    return (merged?.groups || []).flatMap((g) =>
      (g?.advantages || []).map((v) => Number(v) || 0),
    );
  }

  $effect(() => {
    if (activeTab !== "summary") return;
    const id = runId;
    const fId = firstRolloutId;
    const lId = lastRolloutId;
    if (!id || fId == null || lId == null) {
      advantageDist = null;
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const first = await fetchRunAdvantageStep(id, fId);
        const last = fId === lId ? first : await fetchRunAdvantageStep(id, lId);
        if (cancelled) return;
        advantageDist = buildDist(
          mergedAdvantages(first),
          mergedAdvantages(last),
          fId,
          lId,
        );
      } catch {
        if (!cancelled) advantageDist = null;
      }
    })();
    return () => {
      cancelled = true;
    };
  });
</script>

<svelte:window onkeydown={onSampleKeydown} />

<section class="detail" class:embedded>
  {#if !embedded}
    <header class="flex flex-wrap items-center gap-x-[10px] gap-y-[8px] p-[0_24px] mb-[16px] max-[900px]:p-[0_16px]">
      <button type="button" class="inline-flex items-center gap-[6px] [background:none] [border:0] text-(--muted) cursor-pointer text-[13px] leading-[16px] min-h-[32px] p-[4px_8px] rounded-[6px] hover:text-(--text) hover:bg-(--color-c-gray-10,#2f2f2f) max-[900px]:basis-full" onclick={onBack}>
        <ArrowLeft size={14} strokeWidth={2.1} />
        <span>Back to runs</span>
      </button>
      {#if onCollapse}
        <button type="button" class="inline-flex items-center gap-[6px] [border:1px_solid_var(--border,#2f2f2f)] rounded-[6px] [background:none] text-(--muted) cursor-pointer [font:inherit] text-[12px] font-medium leading-[16px] min-h-[32px] p-[4px_8px] hover:text-(--text-bright) hover:border-(--border-strong,#4a4a4a)" onclick={onCollapse} title="Collapse to drawer">
          <Minimize2 size={12} strokeWidth={2.1} />
          <span>Collapse</span>
        </button>
      {/if}
      {#if run?.modal_app_url}
        <a
          class="header-link inline-flex items-center gap-[6px] min-h-[32px] leading-[16px]"
          href={run.modal_app_url}
          target="_blank"
          rel="noopener noreferrer"
        >
          <span>Open in Modal</span>
          <ExternalLink size={12} strokeWidth={2.1} />
        </a>
      {/if}
    </header>
  {/if}

  {#if !run}
    <div class="detail-empty px-[24px]">
      {#if runError}
        Failed to load run: {runError}
      {:else}
        Loading run {runId}…
      {/if}
    </div>
  {:else}
    {#if !embedded}
    <div class="flex items-center gap-[16px] p-[0_24px] mb-[16px] max-[900px]:p-[0_16px] min-w-0">
      <h1 class="text-[22px] font-[600] text-(--text-bright) m-0 overflow-hidden text-ellipsis whitespace-nowrap min-w-0" title={run.run_id}>{run.run_id}</h1>
      <StatusPill status={getStatus(run)} />
      {#if resumeBadge(run)}
        <span class="[border:1px_solid_color-mix(in_srgb,var(--yellow,#fbbf24)_42%,transparent)] rounded-[999px] bg-[color-mix(in_srgb,var(--yellow,#fbbf24)_10%,transparent)] text-(--yellow,#fbbf24) text-[12px] leading-[16px] p-[2px_8px] whitespace-nowrap">{resumeBadge(run)}</span>
      {/if}
    </div>
    {/if}

    <Tabs
      active={activeTab}
      onSelect={selectTab}
      tabs={[
        { value: "summary", label: "Summary" },
        { value: "metrics", label: "Metrics", href: metricLink?.url },
        ...(isSftRun
          ? []
          : [{ value: "rollouts", label: "Rollouts", count: rolloutSummaries.length || undefined }]),
        { value: "logs", label: "Logs" },
      ]}
    />

    {#if activeTab === "summary"}
      <div class="summary-tab">
        <div class="summary-tab-main">
          {#if run?.error_message}
            <div class="mb-[20px]">
              <div class="text-(--red,#f87171) text-[12px] font-[600] tracking-[0.02em] mb-[6px] uppercase">Error</div>
              <pre class="[border:1px_solid_color-mix(in_srgb,var(--red,#f87171)_45%,transparent)] rounded-[8px] bg-[color-mix(in_srgb,var(--red,#f87171)_12%,transparent)] text-(--red,#f87171) [font-family:var(--font-mono)] text-[12px] leading-[17px] m-0 max-h-[320px] overflow-auto p-[12px_14px] whitespace-pre-wrap [word-break:break-word]">{run.error_message}</pre>
            </div>
          {/if}
          {#if showTimingSection || (isSftRun ? lossPoints?.length : rolloutSummaries.length)}
            <div class="chart-range-bar">
              <div class="chart-range-dropdown">
                <MetricsRangeDropdown
                  live={chartRangeParams.live}
                  start={chartRangeParams.start}
                  end={chartRangeParams.end}
                  duration={chartRangeParams.duration}
                  entireRun={chartRange.entireRun}
                  now={chartNow}
                  liveLabel={isRunning ? "now" : "end of run"}
                  onupdate={setChartRange}
                />
              </div>
              <ChartControls
                timeRange={chartRange}
                setTimeRange={setChartRange}
                minStart={chartRunStart}
                now={chartNow}
              />
              <ZoomOutButton
                timeRange={chartRange}
                setTimeRange={setChartRange}
                maxDuration={chartMaxDuration}
                now={chartNow}
              />
            </div>
          {/if}
          {#if timingError || showTimingSection}
            <div class="rollout-chart">
              {#if timingError}
                <div class="text-(--text-muted) text-[12px] mb-[8px]">
                  {timingError}
                </div>
              {/if}
              {#if showTimingSection}
              <div class="rollout-chart-title">
                Substep Timing{#if stepTimingIds.length}{" "}({stepTimingIds.length}
                  {stepTimingIds.length === 1 ? "step" : "steps"}){/if}
              </div>
              <RunTimeline
                timings={runTimings}
                asyncOverride={timelineAsync}
                runOrigin={timelineRunOrigin}
                timelineKey={runId}
                downloadName={`substep_timing_${runId}.json`}
                rolloutIds={rolloutSummaries.map((r) => r.rollout_id)}
                {attemptMarkers}
                timeRange={chartRange.entireRun ? null : chartRange}
                onChangeTimeRange={setChartRange}
                showOpenRollout={!isSftRun}
                trainingType={run?.training_type}
                onOpenRollout={(id) => {
                  selectTab("rollouts");
                  if (expandedRolloutId !== id) void toggleRolloutDetail(id);
                }}
              />
              {/if}
            </div>
          {/if}
          {#if !isSftRun}
          {#if rolloutsLoading && !rolloutSummaries.length}
            <div class="rollout-chart">
              <ChartSkeleton variant="line" height={140} showTitle />
            </div>
            <div class="rollout-chart">
              <div class="rollout-chart-title">Score distribution</div>
              <ChartSkeleton variant="bars" height={120} />
            </div>
            <div class="chart-grid">
              <div class="rollout-chart">
                <div class="rollout-chart-title">Advantage spread over time</div>
                <ChartSkeleton variant="line" height={200} />
              </div>
              <div class="rollout-chart">
                <div class="rollout-chart-title">Advantage distribution over time</div>
                <ChartSkeleton variant="violins" height={210} />
              </div>
            </div>
          {:else if rolloutsError}
            <div class="detail-empty">Failed to load rollouts: {rolloutsError}</div>
          {:else if !rolloutSummaries.length}
            <div class="detail-empty">No rollouts recorded yet.</div>
          {:else}
            <div class="rollout-chart">
              <div class="chart-scroll">
                <LineChart
                  title="Reward"
                  data={rewardChartData}
                  lines={PERCENTILE_LINES}
                  smoothable
                  formatX={(row) => `rollout ${row.rollout_id}`}
                  formatY={(value) => formatMean(value)}
                  ariaLabel="Reward chart"
                  xDomain={chartXDomain}
                  onChangeDomainX={onChartDomainChange}
                  xToTime={chartXToTime}
                  timeToX={chartTimeToX}
                />
              </div>
              {#if chartStats}
                <div class="flex flex-wrap gap-[16px] mt-[6px] text-[11px] text-(--muted) [font-variant-numeric:tabular-nums]">
                  <span>min {formatMean(chartStats.min)}</span>
                  <span>latest {formatMean(chartStats.latest)}</span>
                  <span>max {formatMean(chartStats.max)}</span>
                </div>
              {/if}
            </div>

            <!-- Score distribution: second graph, above the advantage graphs. -->
            <div class="rollout-chart">
              <div class="rollout-chart-title">Score distribution</div>
              <div class="chart-scroll">
                {#if scoreDist}
                  <ComparativeBarChart
                    categories={distCategories(scoreDist)}
                    series={distSeries(scoreDist)}
                    height={120}
                    showCategoryLabels={false}
                    zoomable={false}
                    format={(v) => `${v}`}
                  />
                  <div class="dist-axis">
                    <span>{formatMean(scoreDist.lo)}</span>
                    <span class="dist-axis-label">reward</span>
                    <span>{formatMean(scoreDist.hi)}</span>
                  </div>
                {:else}
                  <ChartSkeleton variant="bars" height={120} />
                {/if}
              </div>
            </div>

            {#if hasAdvantages}
              <div class="chart-grid">
                <div class="rollout-chart">
                  <div class="rollout-chart-title">Advantage spread over time</div>
                  <AdvantageSpreadChart
                    steps={advantageSteps}
                    xDomain={chartXDomain}
                    onChangeDomainX={onChartDomainChange}
                    xToTime={chartXToTime}
                    timeToX={chartTimeToX}
                  />
                </div>
                <div class="rollout-chart">
                  <div class="rollout-chart-title">Advantage distribution over time</div>
                  <AdvantageViolins
                    steps={advantageSteps}
                    xDomain={chartXDomain}
                    onChangeDomainX={onChartDomainChange}
                    xToTime={chartXToTime}
                    timeToX={chartTimeToX}
                  />
                </div>
                <div class="rollout-chart">
                  <div class="rollout-chart-title">Advantage distribution: rollout {firstRolloutId} vs latest</div>
                  {#if advantageDist}
                    <ComparativeBarChart
                      categories={distCategories(advantageDist)}
                      series={distSeries(advantageDist)}
                      height={120}
                      showCategoryLabels={false}
                      zoomable={false}
                      format={(v) => `${v}`}
                    />
                    <div class="dist-axis">
                      <span>{formatMean(advantageDist.lo)}</span>
                      <span class="dist-axis-label">advantage</span>
                      <span>{formatMean(advantageDist.hi)}</span>
                    </div>
                  {:else}
                    <ChartSkeleton variant="bars" height={120} />
                  {/if}
                </div>
              </div>
            {/if}

            {#if customTagNames.length}
              <div class="chart-grid">
                {#each customTagNames as tag (tag)}
                  <div class="rollout-chart">
                    <LineChart
                      title={tag}
                      data={tagChartData(tag)}
                      lines={PERCENTILE_LINES}
                      smoothable
                      formatX={(row) => `rollout ${row.rollout_id}`}
                      formatY={(value) => formatMean(value)}
                      ariaLabel={`${tag} chart`}
                      xDomain={chartXDomain}
                      onChangeDomainX={onChartDomainChange}
                      xToTime={chartXToTime}
                      timeToX={chartTimeToX}
                    />
                    {#if tagChartStats(tag)}
                      <div class="flex gap-[16px] mt-[6px] text-[11px] text-(--muted) [font-variant-numeric:tabular-nums]">
                        <span>min {formatMean(tagChartStats(tag).min)}</span>
                        <span>latest {formatMean(tagChartStats(tag).latest)}</span>
                        <span>max {formatMean(tagChartStats(tag).max)}</span>
                      </div>
                    {/if}
                  </div>
                {/each}
              </div>
            {/if}
          {/if}
          {:else if lossPoints === null && !lossError}
            <div class="rollout-chart">
              <ChartSkeleton variant="line" height={140} showTitle />
            </div>
          {:else if lossPoints === null}
            <div class="detail-empty">Failed to load loss: {lossError}</div>
          {:else if !lossPoints.length}
            <div class="detail-empty">No loss reported yet.</div>
          {:else}
            <div class="rollout-chart">
              <div class="chart-scroll">
                <LineChart
                  title="Loss"
                  data={lossInRange}
                  label="loss"
                  formatX={(row) => `step ${row.x}`}
                  formatY={formatMetricValue}
                  ariaLabel="Loss chart"
                />
              </div>
              {#if lossStats}
                <div class="flex flex-wrap gap-[16px] mt-[6px] text-[11px] text-(--muted) [font-variant-numeric:tabular-nums]">
                  <span>min {formatMetricValue(lossStats.min)}</span>
                  <span>latest {formatMetricValue(lossStats.latest)}</span>
                  <span>max {formatMetricValue(lossStats.max)}</span>
                </div>
              {/if}
            </div>
          {/if}
        </div>
        <aside class="summary-tab-side">
          <RunSummary
            {run}
            {getStatus}
            {showFrameworkStatus}
            {getFrameworkStatus}
            {modelName}
            {fmtDuration}
          />
        </aside>
      </div>
    {:else if activeTab === "rollouts"}
      <div class="tab-panel">
      {#if rolloutsLoading && !rolloutSummaries.length}
        <div class="detail-empty">Loading rollouts…</div>
      {:else if rolloutsError}
        <div class="detail-empty">Failed to load rollouts: {rolloutsError}</div>
      {:else if !rolloutSummaries.length}
        <div class="detail-empty">No rollouts recorded yet.</div>
      {:else}
        <div class="table-wrap">
        <ResizableTable class="rollout-table" columns={rolloutColumns}>
          <tbody>
            {#each rolloutSummaries as r (r.rollout_id)}
              <tr
                class:expanded={expandedRolloutId === r.rollout_id}
                class:rollout-error={r.error_summary?.verdict === "all_infra_failure"}
                class:rollout-warn={r.error_summary?.verdict === "partial_infra_failure"}
                onclick={() => toggleRolloutDetail(r.rollout_id)}
              >
                <td>#{r.rollout_id}</td>
                <td class="text-(--text-bright)">
                  {formatMean(r.mean)}
                  {#if r.error_summary?.verdict === "all_infra_failure"}
                    <span class="inline-block text-[10px] font-medium p-[1px_6px] rounded-[3px] ml-[6px] align-middle bg-[rgba(239,68,68,0.15)] text-[#ef4444] [border:1px_solid_rgba(239,68,68,0.25)]" title="All samples hit infrastructure errors">infra failure</span>
                  {:else if r.error_summary?.verdict === "partial_infra_failure"}
                    <span class="inline-block text-[10px] font-medium p-[1px_6px] rounded-[3px] ml-[6px] align-middle bg-[rgba(251,191,36,0.15)] text-[#fbbf24] [border:1px_solid_rgba(251,191,36,0.25)]" title="Some samples hit infrastructure errors">partial failure</span>
                  {/if}
                </td>
                <td>{r.episode_count ?? "—"}</td>
                <td>
                  <TimeAgo timestamp={r.created_at} showJustNow falsyRepresentation="—" />
                </td>
              </tr>
              {#if expandedRolloutId === r.rollout_id}
                <tr>
                  <td class="p-[12px_10px] bg-(--color-c-gray-08,#1c1c1c) cursor-default" colspan={rolloutColumns.length}>
                    {#if expandedRolloutLoading}
                      <div class="detail-empty">Loading rollouts…</div>
                    {:else if !expandedRollout || !sampleDist}
                      <div class="detail-empty">No rollouts recorded.</div>
                    {:else}
                      {#if runTimings[r.rollout_id]}
                        <div class="rollout-chart">
                          <div class="rollout-chart-title">Substep timing</div>
                          <RunTimeline
                            timings={{ [r.rollout_id]: runTimings[r.rollout_id] }}
                            asyncOverride={timelineAsync}
                            runOrigin={timelineRunOrigin}
                            showOpenRollout={false}
                            timelineKey={`${runId}:${r.rollout_id}`}
                            downloadName={`substep_timing_${runId}_rollout_${r.rollout_id}.json`}
                            rolloutIds={[r.rollout_id]}
                            onOpenRollout={(id) => {
                              selectTab("rollouts");
                              if (expandedRolloutId !== id) void toggleRolloutDetail(id);
                            }}
                          />
                        </div>
                      {/if}
                      {#if expandedRollout.metrics && Object.keys(expandedRollout.metrics).length}
                        {@const m = expandedRollout.metrics}
                        {@const remoteErr = Number(m["agent/exit_status/remoteerror_sample_count"]) || 0}
                        {@const responseMissing = Number(m["agent/response_missing_sample_count"]) || 0}
                        {@const infraInvalid = Number(m["agent/invalid_infra_sample_count"]) || 0}
                        {@const limitsExceeded = Number(m["agent/limits_exceeded_sample_count"]) || 0}
                        {@const totalSamples = Number(m["agent/valid_sample_count"]) || sampleDist.sampleCount || 0}
                        {@const hasErrors = remoteErr > 0 || responseMissing > 0 || infraInvalid > 0}
                        {#if hasErrors}
                          <div class="rollout-diagnostics" class:diag-critical={remoteErr >= totalSamples}>
                            <div class="diag-title">
                              {#if remoteErr >= totalSamples}
                                All {totalSamples} samples hit infrastructure errors
                              {:else}
                                {remoteErr + infraInvalid} / {totalSamples} samples hit infrastructure errors
                              {/if}
                            </div>
                            <div class="flex flex-wrap gap-[6px] mb-[4px]">
                              {#if remoteErr}
                                <span class="diag-tag">RemoteError: {remoteErr}</span>
                              {/if}
                              {#if responseMissing}
                                <span class="diag-tag">Response missing: {responseMissing}</span>
                              {/if}
                              {#if infraInvalid}
                                <span class="diag-tag">Infra invalid: {infraInvalid}</span>
                              {/if}
                              {#if limitsExceeded}
                                <span class="diag-tag">Limits exceeded: {limitsExceeded}</span>
                              {/if}
                            </div>
                            {#if remoteErr >= totalSamples}
                              <div class="text-[11px] text-(--muted,#a3a3a3) mt-[6px]">
                                Check the Modal app logs for sandbox or image build errors.
                              </div>
                            {/if}
                          </div>
                        {/if}
                      {/if}
                      <div class="mb-[16px]">
                        <div class="flex justify-end mb-[6px]">
                          <button
                            type="button"
                            class="inline-flex items-center gap-[5px] [background:none] [border:1px_solid_var(--border,#2f2f2f)] rounded-[4px] text-(--muted) text-[11px] p-[3px_8px] cursor-pointer hover:text-(--text) hover:border-(--border-strong,#4a4a4a)"
                            onclick={downloadAllTrajectories}
                            title="Download all samples as JSON"
                          >
                            <Download size={13} />
                            Download all ({sampleDist.sampleCount} samples)
                          </button>
                        </div>
                        <div class="chart-scroll">
                          <div
                            class="flex items-end gap-[2px] h-[120px] pt-[14px] min-w-[280px] [border-bottom:1px_solid_var(--border,#2f2f2f)]"
                            role="group"
                            aria-label="Reward distribution"
                          >
                            {#each sampleDist.buckets as bucket, b (b)}
                              <button
                                type="button"
                                class="dist-bar"
                                class:detail-active={activeBucket === b}
                                class:is-empty={!bucket.length}
                                style:height={`${(bucket.length / sampleDist.maxCount) * 100}%`}
                                disabled={!bucket.length}
                                title={bucketLabel(bucket, b)}
                                onclick={() => openBucket(b)}
                              >
                                <span class="absolute top-[-14px] left-0 right-0 text-center text-[10px] text-(--muted) [font-variant-numeric:tabular-nums]">{bucket.length || ""}</span>
                              </button>
                            {/each}
                          </div>
                          <div class="dist-axis">
                            <span>{formatMean(sampleDist.lo)}</span>
                            <span class="dist-axis-label">reward · {distSummary}</span>
                            <span>{formatMean(sampleDist.hi)}</span>
                          </div>
                        </div>
                      </div>

                      {#if activeSample}
                        <div class="sample-viewer">
                          <div class="sample-viewer-header">
                            <div class="sample-viewer-nav">
                              <button
                                class="sample-nav-btn"
                                onclick={() => stepSample(-1)}
                                disabled={activeSample.pos === 0}
                                aria-label="Previous rollout"
                              >
                                <ChevronLeft size={14} />
                              </button>
                              <span class="text-[12px] text-(--text-bright) [font-variant-numeric:tabular-nums]">
                                Rollout {activeSample.pos + 1} / {activeSample.count}
                              </span>
                              <button
                                class="sample-nav-btn"
                                onclick={() => stepSample(1)}
                                disabled={activeSample.pos === activeSample.count - 1}
                                aria-label="Next rollout"
                              >
                                <ChevronRight size={14} />
                              </button>
                              <span class="sample-viewer-hint">← / → to navigate</span>
                            </div>
                            <div class="sample-viewer-actions">
                              <span class="text-(--text-bright) [font-variant-numeric:tabular-nums]">
                                reward {formatMean(activeSample.score)}{activeSample.samples.length > 1
                                  ? ` · first of ${activeSample.samples.length} turns`
                                  : ""}
                              </span>
                              <button
                                class="sample-nav-btn"
                                onclick={downloadSampleTrajectory}
                                aria-label="Download trajectory JSON"
                                title="Download trajectory"
                              >
                                <Download size={14} />
                              </button>
                              <button
                                class="sample-nav-btn"
                                onclick={closeBucket}
                                aria-label="Close rollout viewer"
                              >
                                <X size={14} />
                              </button>
                            </div>
                          </div>
                          {#if activeSample.sample.metadata?.inference}
                            <div class="rollout-sample-label">inference</div>
                            <InferenceStats inference={activeSample.sample.metadata.inference} />
                          {/if}
                          {#if activeSample.sample.metadata?._metadata_type === "audio" || activeSample.sample.metadata?.audio}
                            <div class="rollout-sample-label">audio</div>
                            <audio
                              class="block w-full max-w-[400px] m-[4px_0_8px] rounded-[4px]"
                              controls
                              preload="none"
                              src={activeSample.sample.metadata.audio}
                            ></audio>
                          {/if}
                          {#if activeSample.image}
                            <div class="rollout-sample-label">image</div>
                            <img
                              class="block w-full max-w-[400px] h-auto m-[4px_0_8px] rounded-[4px] [border:1px_solid_var(--border)]"
                              src={activeSample.image}
                              alt="rollout input"
                              loading="lazy"
                            />
                          {/if}
                          {#if activeSample.sample.prompt}
                            <div class="rollout-sample-label">prompt</div>
                            <pre class="rollout-sample-text">{activeSample.sample.prompt}</pre>
                          {/if}
                          <DynamicTrajectoryViewer
                            sample={activeSample.sample}
                            samples={activeSample.samples}
                            trajectory={activeSample.sample.metadata?.trajectory_messages || []}
                            rewardEvents={activeSample.sample.reward_events || []}
                            rollout={expandedRollout}
                            run={run}
                            position={`${activeBucket}/${activeSample.pos}`}
                          />
                          {#if activeSample.sample.metadata?.reference}
                            <div class="rollout-sample-label">reference</div>
                            <pre class="rollout-sample-text">{activeSample.sample.metadata.reference}</pre>
                          {/if}
                          {#each Object.entries(activeSample.sample.metadata?.metrics ?? {}) as [name, value]}
                            <div class="rollout-sample-label">{name}</div>
                            <span class="rollout-sample-metric">
                              {typeof value === "number" ? value.toFixed(3) : value}
                            </span>
                          {/each}
                          {#if activeSample.sample.metadata?.exit_status}
                            <div class="rollout-sample-label">exit status</div>
                            <span class="rollout-sample-metric p-[2px_8px] rounded-[3px] text-[11px]! font-medium" class:exit-ok={activeSample.sample.metadata.exit_status === "ok"} class:exit-err={activeSample.sample.metadata.exit_status !== "ok"}>
                              {activeSample.sample.metadata.exit_status}
                            </span>
                          {/if}
                          {#if activeSample.sample.metadata?.eval_detail}
                            <div class="rollout-sample-label">failure reason</div>
                            <pre class="rollout-sample-text">{activeSample.sample.metadata.eval_detail}</pre>
                          {/if}
                          <!-- Catch-all: any other tag a custom reward/rollout function set on
                               sample.metadata (e.g. sample.metadata["guessing"] = {...}) that
                               isn't one of the known keys rendered explicitly above. -->
                          {#each Object.entries(activeSample.sample.metadata ?? {}).filter(
                            ([key]) =>
                              ![
                                "inference",
                                "_metadata_type",
                                "audio",
                                "image",
                                "image_ref",
                                "trajectory_messages",
                                "eval_report",
                                "reference",
                                "metrics",
                                "exit_status",
                                "eval_detail",
                                "response_length",
                                "prompt_length",
                                "rollout_id",
                                "rollout_idx",
                              ].includes(key),
                          ) as [name, value] (name)}
                            <div class="rollout-sample-label">{name}</div>
                            {#if value !== null && typeof value === "object"}
                              <pre class="rollout-sample-text">{JSON.stringify(value, null, 2)}</pre>
                            {:else}
                              <span class="rollout-sample-metric">{String(value)}</span>
                            {/if}
                          {/each}
                          {#if activeSample.sample.trace?.length}
                            <div class="rollout-sample-label">trajectory timeline</div>
                            <div class="chart-scroll">
                              <SampleTimeline trace={activeSample.sample.trace} />
                            </div>
                          {/if}
                        </div>
                      {:else}
                        <div class="text-[12px] text-(--muted) p-[4px_0]">Click a bar to inspect its rollouts.</div>
                      {/if}
                    {/if}
                  </td>
                </tr>
              {/if}
            {/each}
          </tbody>
        </ResizableTable>
        </div>
      {/if}
      </div>
    {:else if activeTab === "metrics"}
      <div class="tab-panel">
        <RunMetricsPanel {runId} {isRunning} />
      </div>
    {:else if activeTab === "logs"}
      <div class="tab-panel">
      {#if isRunning}
      <div class="flex justify-end mb-[8px]">
        <span class="inline-flex items-center gap-[6px] text-[11px] text-(--muted) uppercase tracking-[0.04em]">
          {#if logState === "streaming"}
            <span class="dot bg-[#4ade80]! shadow-[0_0_0_2px_rgba(74,222,128,0.18)]"></span> live
          {:else if logState === "paused"}
            <span class="dot dot-dim"></span> paused
          {:else if logState === "reconnecting"}
            <span class="dot bg-[#fbbf24]!"></span> reconnecting…{#if logError} <span class="log-reconnect-reason">({logError})</span>{/if}
          {:else if logState === "done"}
            <span class="dot dot-dim"></span> finished
          {:else if logState === "error"}
            <span class="dot bg-[#f87171]!"></span> error
          {:else if String(run?.status || "").toLowerCase() !== "running"}
            <span class="dot dot-dim"></span> run not active
          {:else}
            <span class="dot dot-dim"></span> idle
          {/if}
        </span>
      </div>

      <div class="flex flex-wrap items-center gap-[8px] mb-[8px]">
        <button
          class="log-button"
          onclick={toggleLogPaused}
          disabled={String(run?.status || "").toLowerCase() !== "running"}
        >
          {logPaused ? "Resume" : "Pause"}
        </button>
        <button class="log-button" onclick={clearLogs} disabled={!logLines.length}>
          Clear
        </button>
        <input
          class="[flex:1] min-w-[160px] bg-(--color-c-gray-08,#1c1c1c) text-(--text) [border:1px_solid_var(--border,#3a3a3a)] rounded-[4px] p-[4px_8px] text-[12px] [font-family:inherit]"
          type="search"
          placeholder="filter substring…"
          bind:value={logSearchInput}
          aria-label="Filter log lines"
        />
        <label class="inline-flex items-center gap-[6px] text-(--muted) text-[11px]">
          <span>Rate cap</span>
          <select class="bg-(--color-c-gray-08,#1c1c1c) text-(--text) [border:1px_solid_var(--border,#3a3a3a)] rounded-[4px] p-[2px_6px] text-[12px]" bind:value={logRateCap} aria-label="Lines per second cap">
            <option value={0}>off</option>
            <option value={10}>10/s</option>
            <option value={50}>50/s</option>
            <option value={200}>200/s</option>
            <option value={1000}>1000/s</option>
          </select>
        </label>
        <label class="inline-flex items-center gap-[6px] text-(--muted) text-[11px]">
          <input type="checkbox" bind:checked={logFollow} />
          <span>Follow tail</span>
        </label>
      </div>

      {#if logState === "error" && logError}
        <div class="detail-empty">Log stream error: {logError}</div>
      {/if}

      {#if !logLines.length}
        <div class="detail-empty">
          {#if String(run?.status || "").toLowerCase() !== "running"}
            Logs only stream while the run is active.
          {:else if logPaused}
            Stream paused.
          {:else if logSearch}
            Waiting for log output matching "{logSearch}"…
          {:else}
            Waiting for log output…
          {/if}
        </div>
      {:else}
        <div class="bg-(--color-c-gray-08,#0e0e0e) rounded-[6px] p-[8px_12px] max-h-[420px] overflow-y-auto overflow-x-auto [font-family:ui-monospace,SFMono-Regular,Menlo,monospace] text-[12px] leading-[1.45] text-(--text)" bind:this={logTailEl}>
          {#each logLines as entry (entry.id)}
            <div class="flex gap-[10px] whitespace-pre">
              <span class="shrink-0 text-(--muted) text-[10px] min-w-[64px] overflow-hidden text-ellipsis">{entry.task_id || ""}</span>
              <span class="flex-1 whitespace-pre-wrap break-all">{entry.line}</span>
            </div>
          {/each}
        </div>
        <div class="mt-[6px] text-[11px] text-(--muted) [font-variant-numeric:tabular-nums] flex gap-[6px]">
          <span>
            Showing last {logLines.length} line{logLines.length === 1 ? "" : "s"} (cap {LOG_BUFFER_MAX})
          </span>
          {#if logDropped > 0}
            <span class="text-[#fbbf24]">
              · {logDropped} dropped by rate cap
            </span>
          {/if}
        </div>
      {/if}
      {:else}
        <!-- Finished run: page through the durable copy on demand. -->
        <div class="flex flex-col gap-[12px] mb-[12px] p-[12px_14px] rounded-[8px] bg-(--color-c-gray-08,#161616) [border:1px_solid_var(--border,#2f2f2f)]">
          <div class="flex items-center gap-[12px]">
            <input
              class="flex-1 min-w-0 bg-(--color-c-gray-10,#1c1c1c) text-(--text) [border:1px_solid_var(--border,#3a3a3a)] rounded-[5px] p-[6px_10px] text-[12px] [font-family:inherit] focus:outline-none focus:[border-color:color-mix(in_srgb,var(--accent)_55%,transparent)]"
              type="search"
              placeholder="filter substring…"
              bind:value={logSearchInput}
              aria-label="Filter log lines"
            />
            <span class="inline-flex items-center gap-[6px] text-[11px] text-(--muted) uppercase tracking-[0.04em] shrink-0">
              <span class="dot dot-dim"></span> stored logs
            </span>
          </div>
          <div class="flex flex-wrap items-center gap-[10px]">
            <span class="text-(--muted) text-[11px] uppercase tracking-[0.04em]">Time range</span>
            <input
              class="w-[160px] bg-(--color-c-gray-10,#1c1c1c) text-(--text) [border:1px_solid_var(--border,#3a3a3a)] rounded-[5px] p-[5px_8px] text-[12px] [font-family:inherit] [font-variant-numeric:tabular-nums] focus:outline-none focus:[border-color:color-mix(in_srgb,var(--accent)_55%,transparent)]"
              type="text"
              placeholder="YYYY-MM-DD HH:MM"
              bind:value={histRangeInput.since}
              aria-label="Show logs since"
            />
            <span class="text-(--muted-strong) text-[13px]">→</span>
            <input
              class="w-[160px] bg-(--color-c-gray-10,#1c1c1c) text-(--text) [border:1px_solid_var(--border,#3a3a3a)] rounded-[5px] p-[5px_8px] text-[12px] [font-family:inherit] [font-variant-numeric:tabular-nums] focus:outline-none focus:[border-color:color-mix(in_srgb,var(--accent)_55%,transparent)]"
              type="text"
              placeholder="YYYY-MM-DD HH:MM"
              bind:value={histRangeInput.until}
              aria-label="Show logs until"
            />
            <button
              class="log-button text-[11px] px-[10px] py-[4px]"
              onclick={resetHistRange}
              title="Reset to the run's time range"
            >
              Reset
            </button>
          </div>
        </div>

        {#if histError}
          <div class="detail-empty">Failed to load logs: {histError}</div>
        {/if}

        {#if histLoading && !histLines.length}
          <div class="detail-empty">Loading logs…</div>
        {:else if !histLines.length}
          <div class="detail-empty">
            {#if logSearch}
              No log lines matching "{logSearch}".
            {:else}
              No logs recorded for this run.
            {/if}
          </div>
        {:else}
          <div class="bg-(--color-c-gray-08,#0e0e0e) rounded-[6px] p-[8px_12px] max-h-[420px] overflow-y-auto overflow-x-auto [overflow-anchor:none] [font-family:ui-monospace,SFMono-Regular,Menlo,monospace] text-[12px] leading-[1.45] text-(--text)" bind:this={histTailEl} onscroll={onHistScroll}>
            {#if histHasMore}
              <div class="text-center text-[10px] text-(--muted) pb-[6px]">
                {histLoadingOlder ? "Loading older lines…" : "Scroll up for older lines"}
              </div>
            {/if}
            {#each histLines as entry (entry.id)}
              <div class="flex gap-[10px] whitespace-pre" data-hist-id={entry.id}>
                <span class="shrink-0 text-(--muted) text-[10px] min-w-[64px] overflow-hidden text-ellipsis">{entry.task_id || ""}</span>
                <span class="flex-1 whitespace-pre-wrap break-all">{entry.line}</span>
              </div>
            {/each}
            {#if histNewerWindows.length}
              <div class="text-center text-[10px] text-(--muted) pt-[6px]">
                {histLoadingNewer ? "Loading newer lines…" : "Scroll down for newer lines"}
              </div>
            {/if}
          </div>
          <div class="mt-[6px] text-[11px] text-(--muted) [font-variant-numeric:tabular-nums] flex items-center gap-[8px]">
            <span>Showing {histLines.length} line{histLines.length === 1 ? "" : "s"}</span>
            {#if histNewerWindows.length}
              <span class="h-[12px] w-px bg-(--border)"></span>
              <button
                type="button"
                class="text-(--accent) underline-offset-2 hover:underline bg-transparent border-0 p-0 cursor-pointer text-[11px]"
                onclick={jumpHistToLatest}
              >
                Jump to latest
              </button>
            {/if}
          </div>
        {/if}
      {/if}
      </div>
    {/if}
  {/if}
</section>
