<script>
  import { onMount } from "svelte";
  import { Book, CheckCircle2, Zap } from "lucide-svelte";
  import "./app.css";
  import Sidebar from "./components/Sidebar.svelte";
  import DashboardHeader from "./components/DashboardHeader.svelte";
  import TrainingPage from "./pages/TrainingPage.svelte";
  import TrainingRunDetailPage from "./pages/TrainingRunDetailPage.svelte";
  import EvalsPage from "./pages/EvalsPage.svelte";
  import {
    fetchRuns,
    fetchRunCounts,
    fetchEvals,
    fetchEvalDetail,
  } from "./lib/api.js";
  import logoSvg from "./lib/logo.svg";
  import { fmtDuration } from "./lib/format.js";
  import { createSidebarCollapsedState } from "./lib/sidebarCollapsed.svelte.js";
  const DOCS_URL = "https://gym.modal.dev";

  // The server filters, sorts and pages the run list, so `runs` only holds the
  // rows the page asked for and every total comes from `runCounts`.
  const RUNS_PAGE_SIZE = 100;
  let runs = $state([]);
  let loadedRunCount = $state(RUNS_PAGE_SIZE);
  let runCounts = $state({
    total: 0,
    matching: 0,
    status: {},
    recipe: {},
    group: {},
    training_type: {},
  });
  let allEvals = $state([]);
  let loading = $state(true);
  let loadingEvals = $state(false);
  let error = $state(null);
  let search = $state("");
  let activeRecipes = $state(new Set());
  let activeStatuses = $state(new Set());
  let activeGroups = $state(new Set());
  let activeTrainingTypes = $state(new Set());
  let trainingGroupBy = $state("none");
  // Facet values we've seen across loads. New ones are
  // auto-enabled in the filters once; the user's selections are never reset by
  // a refresh.
  let seenRecipes = new Set();
  let seenStatuses = new Set();
  let seenGroups = new Set();
  let seenTrainingTypes = new Set();
  let activePage = $state("training");
  const sidebar = createSidebarCollapsedState();
  let activeTrainingRunId = $state(null);
  // When set (and no full detail page is open), the training list shows a
  // summary drawer for this run — set by "Collapse" on the detail page.
  let drawerRunId = $state(null);
  // True while any data fetch is in flight (manual or the 5s auto-refresh) —
  // drives the spinning refresh button. Distinct from `loading`, which only
  // gates the cold-start skeleton.
  let refreshing = $state(false);
  let runsRequestId = 0;
  let hasLoadedRuns = false;
  let initialRunsLoadStarted = false;
  let evalsRequestId = 0;
  let hasLoadedEvals = $state(false);

  const pageMeta = {
    training: { title: "Training runs" },
    evals: { title: "Evals" },
  };

  const pagePaths = {
    training: "/training",
    evals: "/evals",
  };

  function pageFromPath(pathname) {
    if (pathname === "/" || pathname.startsWith("/training")) return "training";
    if (pathname.startsWith("/evals")) return "evals";
    return "training";
  }

  function runIdFromPath(pathname) {
    if (!pathname.startsWith("/training/")) return null;
    const tail = pathname.slice("/training/".length).split("/")[0];
    return tail ? decodeURIComponent(tail) : null;
  }

  const navItems = [
    { key: "training", label: "Training runs", Icon: Zap, path: pagePaths.training },
    { key: "evals", label: "Evals", Icon: CheckCircle2, path: pagePaths.evals },
  ];

  if (typeof window !== "undefined") {
    activePage = pageFromPath(window.location.pathname);
    activeTrainingRunId = runIdFromPath(window.location.pathname);
  }

  onMount(() => {
    const syncPageWithPath = () => {
      activePage = pageFromPath(window.location.pathname);
      activeTrainingRunId = runIdFromPath(window.location.pathname);
    };

    if (window.location.pathname === "/") {
      window.history.replaceState({}, "", pagePaths.training);
    } else {
      syncPageWithPath();
    }

    window.addEventListener("popstate", syncPageWithPath);

    // Cmd/Ctrl+B toggles the sidebar, except while typing in a field.
    const onKeyDown = (event) => {
      if (event.key.toLowerCase() !== "b" || !(event.metaKey || event.ctrlKey)) return;
      if (event.altKey || event.shiftKey || event.defaultPrevented) return;
      const target = event.target;
      if (
        target instanceof HTMLElement &&
        (target.isContentEditable ||
          ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName))
      ) {
        return;
      }
      event.preventDefault();
      sidebar.toggle();
    };
    window.addEventListener("keydown", onKeyDown);

    // Auto-refresh the active page's data every 5s so running training runs,
    // their status/stage and rollouts stay live. Current data stays on screen
    // (no skeleton) and only the refresh button spins while fetching. A run
    // detail page refreshes its own run, so skip the full list there.
    // A hidden tab polls nothing — mobile browsers discard tabs that keep
    // working off-screen — and catches up once on the way back.
    let refresh = null;

    const shouldRefresh = () =>
      !(activePage === "training" && activeTrainingRunId);

    const startPolling = () => {
      if (refresh !== null) return;
      refresh = window.setInterval(() => {
        if (shouldRefresh()) void load();
      }, 5000);
    };

    const stopPolling = () => {
      if (refresh === null) return;
      window.clearInterval(refresh);
      refresh = null;
    };

    const onVisibilityChange = () => {
      if (document.hidden) {
        stopPolling();
        return;
      }
      startPolling();
      if (shouldRefresh()) void load();
    };

    if (!document.hidden) startPolling();
    document.addEventListener("visibilitychange", onVisibilityChange);

    return () => {
      window.removeEventListener("popstate", syncPageWithPath);
      window.removeEventListener("keydown", onKeyDown);
      document.removeEventListener("visibilitychange", onVisibilityChange);
      stopPolling();
    };
  });

  // Mirrors the server's facet buckets (`run_facet_values`), which is what the
  // filter chips and counts are keyed by.
  const NO_GROUP = "(no group)";
  const UNTAGGED_RECIPE = "(untagged)";

  function getGroup(run) {
    return safeText(run.group_id) || NO_GROUP;
  }

  function getStatus(run) {
    return safeText(run.display_status) || "pending";
  }

  function getTrainingRunStatus(run) {
    return safeText(run.status).toLowerCase();
  }

  function getFrameworkStatus(run) {
    return safeText(run.framework_status);
  }

  function showFrameworkStatus(run) {
    if (getTrainingRunStatus(run) === "running") return true;
    return !!run.framework_status;
  }

  function modelName(run) {
    return run.model || "—";
  }

  function safeText(value) {
    if (value && typeof value === "object" && "value" in value) return value.value;
    return value != null ? String(value) : "";
  }

  function getErrorMessage(value) {
    if (value instanceof Error) return value.message;
    if (typeof value === "string") return value;
    return "unknown error";
  }

  function fetchWithTimeout(fn, timeoutMs, label) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
    return fn({ signal: controller.signal })
      .then((value) => {
        clearTimeout(timeoutId);
        return value;
      })
      .catch((err) => {
        clearTimeout(timeoutId);
        if (err.name === "AbortError")
          throw new Error(`${label} request timed out after ${timeoutMs}ms`);
        throw err;
      });
  }

  // A chip group with every value selected is the same query as no filter at
  // all, and leaving it out of the request keeps the URL (and the response
  // cache key) stable while the user is only toggling within one group.
  function requestFacets() {
    const universes = {
      status: statuses,
      recipe: recipes,
      group: groups,
      training_type: trainingTypes,
    };
    const selections = {
      status: activeStatuses,
      recipe: activeRecipes,
      group: activeGroups,
      training_type: activeTrainingTypes,
    };
    const facets = {};
    for (const [name, selected] of Object.entries(selections)) {
      const universe = universes[name];
      if (!universe.length) continue;
      if (universe.every((value) => selected.has(value))) continue;
      facets[name] = [...selected];
    }
    return facets;
  }

  // Nothing selected in a chip group matches nothing, and an empty repeated
  // query param reads as "unfiltered" on the server — so answer it here.
  let matchesNothing = $derived.by(() =>
    Object.values(requestFacets()).some((values) => !values.length),
  );

  function adoptFacetValues(counts) {
    // Auto-enable newly-seen facet values without resetting the
    // user's current filter selection on every refresh.
    const seen = {
      recipe: [seenRecipes, activeRecipes],
      status: [seenStatuses, activeStatuses],
      group: [seenGroups, activeGroups],
      training_type: [seenTrainingTypes, activeTrainingTypes],
    };
    for (const [facet, [seenValues, active]] of Object.entries(seen)) {
      const next = new Set(active);
      let changed = false;
      for (const value of Object.keys(counts[facet] || {})) {
        if (seenValues.has(value)) continue;
        seenValues.add(value);
        next.add(value);
        changed = true;
      }
      if (!changed) continue;
      if (facet === "recipe") activeRecipes = next;
      else if (facet === "status") activeStatuses = next;
      else if (facet === "group") activeGroups = next;
      else activeTrainingTypes = next;
    }
  }

  // The counts endpoint is the only source of whole-history totals, so losing
  // it (a preview build talking to an older backend, a slow volume read) must
  // not take the rows down with it: count the loaded window instead. Exact
  // once everything is paged in, an undercount before that.
  function countsFromPage(page) {
    const buckets = { status: {}, recipe: {}, group: {}, training_type: {} };
    for (const run of page) {
      const values = {
        status: getStatus(run),
        recipe:
          safeText(run.recipe) || safeText(run.framework) || UNTAGGED_RECIPE,
        group: getGroup(run),
        training_type: run.config_summary?.loss_type === "sft_loss" ? "sft" : "rl",
      };
      for (const [name, value] of Object.entries(values)) {
        buckets[name][value] = (buckets[name][value] || 0) + 1;
      }
    }
    return { total: page.length, matching: page.length, ...buckets };
  }

  async function loadRuns() {
    const requestId = ++runsRequestId;
    const isStale = () => requestId !== runsRequestId;

    // Skeleton only until the first response settles. Once we've completed a
    // load attempt (success or failure), refreshes keep the current rows on
    // screen — the spinning refresh button is the only "loading" affordance.
    if (!hasLoadedRuns) loading = true;
    error = null;

    const query = search.trim();
    const facets = requestFacets();
    // Refetch the whole loaded window rather than only the newest page: it is
    // the rows that are actually on screen, so the payload stays proportional
    // to what the user has scrolled through, and everything visible stays live.
    const limit = Math.max(loadedRunCount, RUNS_PAGE_SIZE);

    try {
      const [pageResult, countsResult] = await Promise.allSettled([
        matchesNothing
          ? Promise.resolve([])
          : fetchWithTimeout(
              (options) => fetchRuns({ ...options, limit, query, facets }),
              30000,
              "runs",
            ),
        fetchWithTimeout(
          (options) => fetchRunCounts({ ...options, query, facets }),
          15000,
          "run counts",
        ),
      ]);
      if (isStale()) return;
      if (pageResult.status === "rejected") throw pageResult.reason;
      const page = pageResult.value;
      const counts =
        countsResult.status === "fulfilled"
          ? countsResult.value
          : countsFromPage(page);
      runs = page;
      runCounts = counts;
      adoptFacetValues(counts);
    } catch (e) {
      if (isStale()) return;
      // Keep the data we already have on a transient refresh failure — only
      // surface the error (and clear) when there's nothing to show yet.
      // Otherwise the page flickers to "Loading…"/empty on every flaky poll.
      if (!runs.length) {
        error = getErrorMessage(e);
        activeRecipes = new Set();
        activeStatuses = new Set();
        activeGroups = new Set();
        activeTrainingTypes = new Set();
      }
    } finally {
      // Always retire the cold-start skeleton once any attempt settles — even a
      // stale one. A slow request superseded by the 5s auto-refresh must not
      // leave `loading` pinned true forever.
      hasLoadedRuns = true;
      loading = false;
    }
  }

  async function loadEvals() {
    const requestId = ++evalsRequestId;
    const isStale = () => requestId !== evalsRequestId;

    if (!allEvals.length) loadingEvals = true;
    try {
      const evals = await fetchWithTimeout(fetchEvals, 15000, "evals");
      if (isStale()) return;
      allEvals = evals;
      hasLoadedEvals = true;
    } catch (reason) {
      if (isStale()) return;
      if (!allEvals.length) allEvals = [];
      console.warn(getErrorMessage(reason));
    }
    if (!isStale()) loadingEvals = false;
  }

  let reloadQueued = false;

  // `queue`: this load asks for data the current one won't return (a new query
  // or facet selection), so it waits its turn instead of being dropped. Polls
  // pass it up — a request slower than the 5s interval would otherwise queue a
  // successor on every tick and refresh without pause.
  async function load({ queue = false } = {}) {
    // One refresh at a time. The runs payload can take longer to arrive than
    // the 5s interval, and overlapping fetches stack whole copies of it in
    // memory for a response that gets thrown away as stale anyway.
    if (refreshing) {
      if (!queue) return;
      reloadQueued = true;
      // The in-flight request carries the previous query, so retire it: its
      // rows would otherwise land as the current ones until the queued load
      // answers.
      runsRequestId++;
      return;
    }
    refreshing = true;
    try {
      do {
        reloadQueued = false;
        const tasks = [loadRuns()];
        if (activePage === "evals") {
          tasks.push(loadEvals());
        }
        await Promise.all(tasks);
      } while (reloadQueued);
    } finally {
      refreshing = false;
    }
  }

  $effect(() => {
    if (
      !activeTrainingRunId &&
      !hasLoadedRuns &&
      !initialRunsLoadStarted
    ) {
      initialRunsLoadStarted = true;
      void loadRuns();
    } else if (activeTrainingRunId && !hasLoadedRuns) {
      loading = false;
    }
    if (activePage === "evals" && !hasLoadedEvals) {
      void loadEvals();
    }
  });

  let recipeCounts = $derived(runCounts.recipe || {});
  let statusCounts = $derived(runCounts.status || {});
  let groupCounts = $derived(runCounts.group || {});
  let trainingTypeCounts = $derived(runCounts.training_type || {});

  let recipes = $derived(Object.keys(recipeCounts).sort());
  let statuses = $derived(Object.keys(statusCounts).sort());
  // Real group ids first (alphabetical), with "(no group)" pinned last so the
  // sweep groups are what you see at the top of the filter.
  let groups = $derived(
    Object.keys(groupCounts).sort((a, b) => {
      if (a === NO_GROUP) return 1;
      if (b === NO_GROUP) return -1;
      return a.localeCompare(b);
    }),
  );
  let trainingTypes = $derived(Object.keys(trainingTypeCounts).sort());

  // Runs arrive already filtered and sorted newest-first by the server.
  let filteredRuns = $derived(runs);
  let matchingRunCount = $derived(matchesNothing ? 0 : runCounts.matching || 0);
  let hasMoreRuns = $derived(runs.length < matchingRunCount);

  function loadMoreRuns() {
    if (!hasMoreRuns) return;
    const nextCount = runs.length + RUNS_PAGE_SIZE;
    // Scrolling fires this on every event, so while the fetch for this page is
    // in flight or queued it isn't asked for again. Once nothing is in flight a
    // page that failed can be retried.
    if (loadedRunCount >= nextCount && (refreshing || reloadQueued)) return;
    loadedRunCount = nextCount;
    // Queued, so asking for the next page during a refresh grows the window
    // once that refresh lands instead of being dropped.
    void load({ queue: true });
  }

  // Changing the query means a different result set, so paging restarts at the
  // first page. Debounced: typing in the search box shouldn't be one request
  // per keystroke.
  let runQueryKey = $derived(
    JSON.stringify({ q: search.trim(), facets: requestFacets() }),
  );
  let lastRunQueryKey = "";
  $effect(() => {
    const key = runQueryKey;
    if (!hasLoadedRuns || key === lastRunQueryKey) {
      lastRunQueryKey = key;
      return;
    }
    lastRunQueryKey = key;
    const timer = window.setTimeout(() => {
      loadedRunCount = RUNS_PAGE_SIZE;
      void load({ queue: true });
    }, 250);
    return () => window.clearTimeout(timer);
  });

  const trainingGroupKeyFns = {
    group: getGroup,
    dataset: (run) => safeText(run.dataset) || "(no dataset)",
    model: modelName,
  };

  const trainingGroupKey = (run, groupBy) => trainingGroupKeyFns[groupBy]?.(run) ?? "";

  // Buckets inherit filteredRuns' recency sort: groups come out ordered by
  // newest member and runs stay sorted within each group.
  let trainingRunGroups = $derived.by(() => {
    if (trainingGroupBy === "none") return [];
    const buckets = Map.groupBy(filteredRuns, (run) => trainingGroupKey(run, trainingGroupBy));
    return [...buckets].map(([key, bucketRuns]) => ({
      key,
      runs: bucketRuns,
      latestCreatedAt: bucketRuns[0]?.created_at || null,
    }));
  });

  let totalRuns = $derived(runCounts.total || 0);
  let completedTotal = $derived(statusCounts.completed || 0);
  let cancelledTotal = $derived(statusCounts.cancelled || 0);
  let stoppedTotal = $derived(statusCounts.stopped || 0);
  let failedTotal = $derived(statusCounts.failed || 0);
  let runningTotal = $derived(
    totalRuns - completedTotal - cancelledTotal - stoppedTotal - failedTotal,
  );

  function evalAccuracy(ev) {
    if (typeof ev.mean === "number") return ev.mean;
    const rows = ev.rows || [];
    if (!rows.length) return 0;
    return rows.reduce((sum, row) => sum + (row.score || 0), 0) / rows.length;
  }

  function evalCreatedAt(ev) {
    const raw = ev?.created_at;
    if (raw && typeof raw === "object" && "value" in raw) {
      return evalCreatedAt({ created_at: raw.value });
    }
    if (typeof raw === "number") return Number.isFinite(raw) ? raw : 0;
    const text = safeText(raw).trim();
    if (!text) return 0;
    const numeric = Number(text);
    if (Number.isFinite(numeric)) return numeric;
    const epochMs = Date.parse(text);
    if (Number.isFinite(epochMs)) return Math.floor(epochMs / 1000);
    return 0;
  }

  // Maps a raw eval status onto a coarse filter/count bucket
  // ("Completed" | "Pending" | "Failed"), the StatusPill color/icon variant,
  // and a human label for the four eval phases.
  function getEvalDisplay(ev) {
    const rawStatus = safeText(ev.status).toLowerCase();
    if (rawStatus === "deploying_model" || rawStatus === "deploying") {
      return { bucket: "Pending", pill: "running", label: "Deploying model" };
    }
    if (
      rawStatus === "running_eval" ||
      rawStatus === "running" ||
      rawStatus === "pending" ||
      rawStatus === "queued" ||
      rawStatus === "initializing"
    ) {
      return { bucket: "Pending", pill: "running", label: "Running eval" };
    }
    if (
      rawStatus === "completed" ||
      rawStatus === "success" ||
      rawStatus === "succeeded"
    ) {
      return { bucket: "Completed", pill: "completed", label: "Success" };
    }
    if (rawStatus === "failed" || rawStatus === "error") {
      return { bucket: "Failed", pill: "failed", label: "Failed" };
    }
    const total = ev.total ?? (Array.isArray(ev.rows) ? ev.rows.length : 0);
    if (total > 0) {
      return { bucket: "Completed", pill: "completed", label: "Success" };
    }
    return { bucket: "Pending", pill: "running", label: "Pending" };
  }

  function getEvalStatus(ev) {
    return getEvalDisplay(ev).bucket;
  }

  function normalizeConfigValue(value) {
    if (value && typeof value === "object" && "value" in value) {
      return normalizeConfigValue(value.value);
    }
    if (Array.isArray(value)) {
      return value.map((item) => normalizeConfigValue(item));
    }
    if (value && typeof value === "object") {
      return Object.keys(value)
        .sort()
        .reduce((acc, key) => {
          acc[key] = normalizeConfigValue(value[key]);
          return acc;
        }, {});
    }
    return value ?? null;
  }

  function evalConfigKey(ev) {
    return JSON.stringify(normalizeConfigValue(ev.config || {}));
  }

  function evalConfigMeta(config, ev = null) {
    const evalConfig = ev?.eval_config || {};
    const sourceConfig = ev?.config || {};
    const dataset =
      safeText(config?.dataset?.name) ||
      safeText(config?.dataset?.hf_repo) ||
      safeText(config?.dataset?.prompt_data) ||
      safeText(config?.dataset_name) ||
      safeText(sourceConfig?.dataset?.name) ||
      safeText(sourceConfig?.dataset?.hf_repo) ||
      safeText(sourceConfig?.dataset?.prompt_data) ||
      safeText(sourceConfig?.dataset_name) ||
      safeText(evalConfig?.dataset_name) ||
      safeText(ev?.dataset_name) ||
      "—";
    const model = safeText(ev?.model_name) || "—";
    const split =
      safeText(config?.dataset?.split) ||
      safeText(sourceConfig?.dataset?.split) ||
      safeText(evalConfig?.dataset?.split);
    const judge =
      safeText(config?.judge?.model_name) ||
      safeText(config?.judge_model_name) ||
      safeText(sourceConfig?.judge?.model_name) ||
      safeText(sourceConfig?.judge_model_name) ||
      safeText(evalConfig?.judge?.model_name) ||
      safeText(evalConfig?.judge_model_name) ||
      "";
    const evalFn =
      safeText(config?.eval_fn_name) ||
      safeText(config?.grader_name) ||
      safeText(sourceConfig?.eval_fn_name) ||
      safeText(sourceConfig?.grader_name) ||
      safeText(evalConfig?.eval_fn_name) ||
      safeText(evalConfig?.grader_name) ||
      safeText(ev?.eval_fn_name) ||
      "";
    return { dataset, model, split, judge, evalFn };
  }

  let sortedEvals = $derived(
    [...allEvals].sort((a, b) => evalCreatedAt(b) - evalCreatedAt(a)),
  );

  let evalConfigGroups = $derived.by(() => {
    const groups = new Map();
    for (const ev of sortedEvals) {
      const key = safeText(ev.eval_config_id).trim() || evalConfigKey(ev);
      if (!groups.has(key)) {
        groups.set(key, {
          key,
          evalConfigId: key,
          config: ev.config || {},
          runs: [],
          latestCreatedAt: 0,
        });
      }
      const group = groups.get(key);
      const createdAt = evalCreatedAt(ev);
      if (
        (!group.config || Object.keys(group.config).length === 0) &&
        ev.config &&
        Object.keys(ev.config).length > 0
      ) {
        group.config = ev.config;
      }
      const avgScore = evalAccuracy(ev);
      const totalRows = ev.total ?? (ev.rows || []).length;
      const display = getEvalDisplay(ev);
      group.runs.push({
        eval: ev,
        avgScore,
        totalRows,
        status: display.bucket,
        pillStatus: display.pill,
        statusLabel: display.label,
        createdAt,
      });
      group.latestCreatedAt = Math.max(group.latestCreatedAt, createdAt);
    }

    return [...groups.values()]
      .map((group) => {
        const sortedRuns = [...group.runs].sort(
          (a, b) =>
            b.createdAt - a.createdAt ||
            b.avgScore - a.avgScore,
        );
        const totalEvals = sortedRuns.length;
        const totalExamples = sortedRuns.reduce(
          (sum, run) => sum + run.totalRows,
          0,
        );
        const weightedScoreTotal = sortedRuns.reduce(
          (sum, run) => sum + run.avgScore * run.totalRows,
          0,
        );
        const bestScore = sortedRuns[0]?.avgScore ?? 0;
        const avgAccuracy =
          totalExamples > 0
            ? weightedScoreTotal / totalExamples
            : totalEvals > 0
              ? sortedRuns.reduce((sum, run) => sum + run.avgScore, 0) / totalEvals
              : 0;
        const completedCount = sortedRuns.filter(
          (run) => run.status === "Completed",
        ).length;
        const pendingCount = sortedRuns.filter(
          (run) => run.status === "Pending",
        ).length;
        const failedCount = sortedRuns.filter((run) => run.status === "Failed").length;
        return {
          ...group,
          meta: evalConfigMeta(group.config, sortedRuns[0]?.eval),
          bestScore,
          totalEvals,
          avgAccuracy,
          completedCount,
          pendingCount,
          failedCount,
          runs: sortedRuns,
        };
      })
      .sort(
        (a, b) =>
          (b.latestCreatedAt || 0) - (a.latestCreatedAt || 0) ||
          b.runs.length - a.runs.length,
      );
  });

  let evalCompletedTotal = $derived(
    allEvals.filter((ev) => getEvalStatus(ev) === "Completed").length,
  );
  let evalPendingTotal = $derived(
    allEvals.filter((ev) => getEvalStatus(ev) === "Pending").length,
  );
  let evalFailedTotal = $derived(
    allEvals.filter((ev) => getEvalStatus(ev) === "Failed").length,
  );
  let activeTrainingRun = $derived(
    runs.find((run) => run.run_id === activeTrainingRunId) || null,
  );

  let statusText = $derived.by(() => {
    if (activePage === "training" && activeTrainingRunId) return "run details";
    if (activePage === "training" && loading) return "loading...";
    if (activePage === "evals" && loadingEvals) return "loading...";
    if (error) return "error";
    if (activePage === "evals")
      return `${allEvals.length} eval${allEvals.length === 1 ? "" : "s"}`;
    if (!totalRuns) return "0 runs";
    return `${matchingRunCount} of ${totalRuns} runs`;
  });

  function toggleRecipe(recipe) {
    const next = new Set(activeRecipes);
    if (next.has(recipe)) next.delete(recipe);
    else next.add(recipe);
    activeRecipes = next;
  }

  function selectAllRecipes() {
    activeRecipes = new Set(recipes);
  }

  function clearRecipes() {
    activeRecipes = new Set();
  }

  function toggleStatus(status) {
    const next = new Set(activeStatuses);
    if (next.has(status)) next.delete(status);
    else next.add(status);
    activeStatuses = next;
  }

  function selectAllStatuses() {
    activeStatuses = new Set(statuses);
  }

  function clearStatuses() {
    activeStatuses = new Set();
  }

  function toggleGroup(group) {
    const next = new Set(activeGroups);
    if (next.has(group)) next.delete(group);
    else next.add(group);
    activeGroups = next;
  }

  function selectAllGroups() {
    activeGroups = new Set(groups);
  }

  function clearGroups() {
    activeGroups = new Set();
  }

  function toggleTrainingType(value) {
    const next = new Set(activeTrainingTypes);
    if (next.has(value)) next.delete(value);
    else next.add(value);
    activeTrainingTypes = next;
  }

  function selectAllTrainingTypes() {
    activeTrainingTypes = new Set(trainingTypes);
  }

  function clearTrainingTypes() {
    activeTrainingTypes = new Set();
  }

  function setActivePage(page) {
    activePage = page;
    activeTrainingRunId = null;
    drawerRunId = null;
    if (typeof window === "undefined") return;
    const targetPath = pagePaths[page] || pagePaths.training;
    if (window.location.pathname !== targetPath) {
      window.history.pushState({}, "", targetPath);
    }
  }

  function backToTrainingList() {
    activeTrainingRunId = null;
    drawerRunId = null;
    if (typeof window === "undefined") return;
    if (window.location.pathname !== pagePaths.training) {
      window.history.pushState({}, "", pagePaths.training);
    }
  }

  // Opening a run shows the full detail page (a real route, not a drawer).
  function openTrainingRunDetail(runId) {
    drawerRunId = null;
    activeTrainingRunId = runId;
    if (typeof window === "undefined") return;
    const target = `${pagePaths.training}/${encodeURIComponent(runId)}`;
    if (window.location.pathname !== target) {
      window.history.pushState({}, "", target);
    }
  }

  // "Collapse" on the detail page drops back to the list and reopens the run
  // as a summary drawer.
  function collapseTrainingRunToDrawer() {
    drawerRunId = activeTrainingRunId;
    activeTrainingRunId = null;
    if (typeof window === "undefined") return;
    if (window.location.pathname !== pagePaths.training) {
      window.history.pushState({}, "", pagePaths.training);
    }
  }

  function closeTrainingDrawer() {
    drawerRunId = null;
  }
</script>

<div class="h-[100dvh] grid grid-rows-[auto_1fr] bg-(--bg) overflow-x-hidden">
  <header class="[border-bottom:1px_solid_var(--color-c-surface-highlight-gray-opaque,#272727)] bg-(--bg-depth) flex items-center justify-between gap-[1rem] min-h-[53px] p-[0_1rem] max-[900px]:min-h-[53px] max-[900px]:p-[0_0.75rem]">
    <div class="inline-flex items-center gap-[0.55rem] flex-[0_0_auto] min-w-0">
      <img src={logoSvg} alt="Modal" class="h-[17.5px] w-auto flex-[0_0_auto]" />
      <span class="inline-flex items-center gap-[0.18rem] [font-family:var(--font-display)] [font-feature-settings:'ss01'_on] text-[17.6px] leading-[1] [padding-block:0.08rem] font-[600] tracking-[-0.02em] [transform:translateY(1px)] whitespace-nowrap max-[360px]:text-[15px]">
        <span class="text-[#ddffdc]">Modal</span>
        <span class="text-(--green)">Training Gym</span>
      </span>
    </div>
    <a
      class="[border:0] rounded-[10px] text-(--text) [background:transparent] [text-decoration:none] text-[14px] font-medium p-[8px] inline-flex items-center gap-[8px] flex-[0_0_auto] hover:text-(--text-bright) hover:[background:color-mix(in_srgb,white_4%,transparent)] max-[520px]:hidden"
      href={DOCS_URL}
      target="_blank"
      rel="noopener noreferrer"
    >
      <Book size={14} strokeWidth={2.1} />
      <span>Docs</span>
    </a>
  </header>

  <div
    class={[
      "grid min-h-0 h-full bg-(--bg) transition-[grid-template-columns] duration-100 ease-out max-[900px]:grid-cols-[1fr] max-[900px]:grid-rows-[auto_minmax(0,1fr)]",
      sidebar.collapsed
        ? "grid-cols-[56px_minmax(0,1fr)]"
        : "grid-cols-[232px_minmax(0,1fr)]",
    ]}
  >
    <Sidebar
      {navItems}
      {activePage}
      onNavigate={setActivePage}
      collapsed={sidebar.collapsed}
      onToggleCollapsed={() => sidebar.toggle()}
    />

    <main class="min-w-0 min-h-0 h-full flex flex-col overflow-y-auto">
      <DashboardHeader
        title={pageMeta[activePage].title}
        {statusText}
        {refreshing}
        onRefresh={() => load()}
      />

    {#if activePage === "training" && activeTrainingRunId}
      <TrainingRunDetailPage
        runId={activeTrainingRunId}
        initialRun={activeTrainingRun}
        {modelName}
        {getStatus}
        {getFrameworkStatus}
        {showFrameworkStatus}
        {fmtDuration}
        onBack={backToTrainingList}
        onCollapse={collapseTrainingRunToDrawer}
      />
    {:else if activePage === "training"}
      <TrainingPage
        {totalRuns}
        {matchingRunCount}
        {hasMoreRuns}
        onLoadMore={loadMoreRuns}
        {completedTotal}
        {runningTotal}
        {stoppedTotal}
        {failedTotal}
        {recipes}
        {recipeCounts}
        {activeRecipes}
        {statuses}
        {statusCounts}
        {activeStatuses}
        {groups}
        {groupCounts}
        {activeGroups}
        {trainingTypes}
        {trainingTypeCounts}
        {activeTrainingTypes}
        {filteredRuns}
        runGroups={trainingRunGroups}
        bind:groupBy={trainingGroupBy}
        {loading}
        {error}
        {modelName}
        {getStatus}
        {showFrameworkStatus}
        {fmtDuration}
        bind:search
        {drawerRunId}
        onOpenDetail={openTrainingRunDetail}
        onCloseDrawer={closeTrainingDrawer}
        onToggleRecipe={toggleRecipe}
        onSelectAllRecipes={selectAllRecipes}
        onClearRecipes={clearRecipes}
        onToggleStatus={toggleStatus}
        onSelectAllStatuses={selectAllStatuses}
        onClearStatuses={clearStatuses}
        onToggleGroup={toggleGroup}
        onSelectAllGroups={selectAllGroups}
        onClearGroups={clearGroups}
        onToggleTrainingType={toggleTrainingType}
        onSelectAllTrainingTypes={selectAllTrainingTypes}
        onClearTrainingTypes={clearTrainingTypes}
      />
    {:else if activePage === "evals"}
      <EvalsPage
        {allEvals}
        {evalCompletedTotal}
        {evalPendingTotal}
        {evalFailedTotal}
        loading={loadingEvals}
        {error}
        {evalConfigGroups}
        {fetchEvalDetail}
        {getEvalDisplay}
        {evalConfigMeta}
      />
    {/if}
    </main>
  </div>
</div>
