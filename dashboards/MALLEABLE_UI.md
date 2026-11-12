# Malleable dashboard

The goal: people and agents can change what the dashboard shows for a run
without a frontend PR or a redeploy. A run's producer ships the panels that read
its data, a team rearranges a page, and someone writes a one-off view in the
browser. This follows Ink & Switch's
[Malleable software](https://www.inkandswitch.com/essay/malleable-software/)
(a gentle slope from tweaking to building) and the `Sample.metadata` convention
in this repo: producers write data, the UI renders the shapes it knows, and
unknown shapes degrade instead of erroring.

This document continues the closed #395. That PR ran browser-compiled Svelte
inside the dashboard's own origin; this one builds on the sandboxed run-scoped
components from #544 instead. See [Security](#security).

## The slope

| Rung | What you do | Surface | Status |
| --- | --- | --- | --- |
| 1 | Attach a component to a run's slot | `TrainingRun.add_dashboard_component` | **shipped** (`trajectory_viewer`, `summary_panel`, `rollout_panel`) |
| 2 | Rearrange or hide a page's sections | layout documents | planned (phase C) |
| 3 | Write or AI-generate a component in the browser | editor with live sandboxed preview | planned (phase C) |
| 4 | Fork a built-in section and change it | built-ins written against the shared host kit | planned (phases B and D) |
| 5 | Promote a component to a built-in | a PR into `dashboards/frontend/src` | always available |

## Components and slots (shipped)

A component is one Svelte file. `TrainingRun.add_dashboard_component` uploads
it content-addressed (`components/<type>/<sha256>/…` on the
`modal-dojo-dashboard-overlay` Volume) and records its manifest in the run's
`metadata.dashboard_components`. The run record is the authoritative list of
what a run shows; `remove_dashboard_component` detaches one.

Each slot has a fixed props contract (`COMPONENT_PROPS` in
`modal_dojo/common/dashboard_components.py`, documented in
[`README.md`](README.md#slots)). The dashboard fetches data with its existing
typed API calls and posts it to the component; components never fetch.

- `GET /api/runs/{id}/dashboard-components[?type=]`: a run's attachments in
  attachment order, each with its `frame_url`.
- `GET /api/runs/{id}/dashboard-components/{type}`: the latest attachment, for
  single-instance slots (`trajectory_viewer`).
- `GET …/{type}/{sha256}/frame.html?v=`: the compiled component in its host
  page. `v` fingerprints the host template, the bundler, and the frontend's
  locked dependencies, so the browser's immutable cache never serves a frame
  built by an older dashboard.

Frontend pieces: `ComponentFrame.svelte` runs one frame (preflight, ready
timeout, resize, errors), `DashboardPanels.svelte` renders every attachment of
a panel slot, and `DynamicTrajectoryViewer.svelte` keeps the single-instance
viewer with its built-in fallback. The protocol lives in
`src/lib/componentFrame.js`.

## Planned

**Phase B: host kit and bridge.** `component_bundle.mjs` resolves
`$host/components`, `$host/format` and `$host/icons` to a curated kit (charts,
conversation view, formatters, icons) and rejects other imports. The frame
inlines a host stylesheet with the theme variables. Each frame still bundles
its own Svelte runtime; the kit has to fit the size limit below. Frames may post `select` (a
rollout or sample), `hover` and `navigate` (to a tab); the parent validates
these and ignores everything else.

**Phase C: layouts and the editor.** A layout document lists a page's sections
(built-in ids or component digests) with order, `hidden` and per-placement
`config`. Layouts are data only and never carry code. Scopes resolve
`builtin < deployment < run`, plus an optional per-browser override; the
dashboard has one shared password and no per-user identity, so #395's `user`
and `org` scopes have nothing to key on. `?safe=1` ignores every non-built-in
layout and component. `GET /api/ui/schema` describes both document types so an
agent can write them without a browser. The editor shows the source, a live
preview in a sandboxed frame (compiled through the same no-secrets function),
and the props the page would post; saving uploads the source exactly as
`add_dashboard_component` does. On a dashboard without a password the editor is
read-only by default.

**Phase D: built-ins as views.** The run page's sections move into
`src/views/builtin/` one tab at a time (Logs, Metrics, Summary, Rollouts), each
move pixel-identical. Built-ins import through the same `$host/*` aliases, so
"fork this section" compiles the built-in's own source into a sandboxed
component. This waits for the open PRs that touch `TrainingRunDetailPage.svelte`
to land or close.

## Decisions

- **Scopes are `builtin`, `deployment` and `run`.** No named users: the
  dashboard has one shared password, so per-user scopes would need real auth.
- **Without a dashboard password, the editor is read-only by default.**
- **A compiled component must stay under 400 KB**
  (`MAX_COMPILED_COMPONENT_BYTES` in `modal_dojo/common/dashboard_components.py`).
  Compiling a larger bundle fails like any other compile error. A minimal
  component, Svelte runtime included, compiles to about 128 KB today.

#395's `/api/docs` substrate and lenses are deferred: a frame cannot fetch, and
the parent's typed API calls cover every slot so far.

## Security

**Only code in the frontend bundle is trusted.** Everything else (a run's
components today; layouts' referenced components, editor drafts and forked
built-ins later) is untrusted and goes through one path:

1. **Stored content-addressed.** The run's manifest names a `sha256`; the
   dashboard reads the source from the overlay Volume and refuses it (409) if
   the bytes don't match.
2. **Compiled in isolation.** `compile_dashboard_component` runs the Svelte
   compiler and esbuild in a Modal function with no secrets and no Volumes, a
   90-second timeout and bounded resources. The output must stay under
   `MAX_COMPILED_COMPONENT_BYTES`.
3. **Run in an opaque-origin frame.** The frame response's CSP is `sandbox
   allow-scripts` (no same-origin, top navigation, popups or forms),
   `default-src 'none'`, `connect-src 'none'`, nonce-only scripts and
   `frame-ancestors 'self'`. The frame cannot read the dashboard's cookies or
   storage, call its API, or send data anywhere.
4. **Talks only through validated messages.** The parent accepts messages only
   from that frame's window and only of the known types (`ready`, `rendered`,
   `resize`, `error`; later `select`, `hover`, `navigate`).

What a component can do: read the props the page chose to post and draw
inside its own frame. The remaining risk is misleading content inside that
frame. Run-scoped panels are labeled with their attachment name, and phase C
adds a "reset to built-in" control on every non-built-in section.

Layout documents never carry code, so writing a layout can rearrange a page but
cannot run anything. Anyone who can write to the metadata or overlay Volumes can
attach components; that is the same group who can already deploy the
dashboard.
