<script>
  import { untrack } from "svelte";
  import {
    FRAME_READY_TIMEOUT_MS,
    frameHeight,
    frameMessage,
    preflightFrame,
    propsMessage,
  } from "../lib/componentFrame.js";

  /**
   * Runs one compiled run-scoped component in a sandboxed <iframe>.
   *
   * Run-scoped components are untrusted code. The frame is served from an
   * opaque origin with a CSP that blocks network access, and only receives
   * props over postMessage, so it can never act with the dashboard's
   * authority.
   *
   * `componentProps` is what the frame receives. It is re-sent (which
   * remounts the component) only when `propsKey` changes, so callers can keep
   * background polls such as the run refresh from resetting the component.
   * `onstatus(status, detail)` reports "loading", "rendered" and
   * "error" (`detail` is `{ message, deterministic }` for errors).
   */
  let {
    src = "",
    componentProps = {},
    propsKey = null,
    title = "Run-scoped dashboard component",
    onstatus = () => {},
  } = $props();

  let frame = $state(null);
  let frameSrc = $state("");
  let height = $state(0);
  let rendered = $state(false);
  let frameReady = false;
  let readyTimer = null;
  let generation = 0;

  function reset() {
    generation += 1;
    frameReady = false;
    clearTimeout(readyTimer);
    readyTimer = null;
    frameSrc = "";
    height = 0;
    rendered = false;
  }

  function fail(message, deterministic = false) {
    reset();
    onstatus("error", { message, deterministic });
  }

  function postProps() {
    if (!frameReady || !frame?.contentWindow) return;
    // The frame has an opaque origin, so "*" is the only valid target.
    frame.contentWindow.postMessage(propsMessage($state.snapshot(componentProps)), "*");
  }

  async function load(url) {
    reset();
    if (!url) return;
    const current = generation;
    onstatus("loading", null);
    const result = await preflightFrame(url);
    if (current !== generation) return;
    if (!result.ok) {
      fail(result.message, result.deterministic);
      return;
    }
    frameSrc = url;
    readyTimer = setTimeout(() => {
      if (current !== generation || frameReady) return;
      fail("component did not start");
    }, FRAME_READY_TIMEOUT_MS);
  }

  function onMessage(event) {
    const message = frameMessage(event, frame?.contentWindow);
    if (!message) return;
    if (message.type === "ready") {
      frameReady = true;
      clearTimeout(readyTimer);
      readyTimer = null;
      postProps();
    } else if (message.type === "rendered") {
      rendered = true;
      onstatus("rendered", null);
    } else if (message.type === "resize") {
      const next = frameHeight(message);
      if (next !== null) height = next;
    } else if (message.type === "error") {
      fail(String(message.message || "component failed"));
    }
  }

  $effect(() => {
    const url = src;
    untrack(() => load(url));
  });

  $effect(() => {
    propsKey;
    untrack(postProps);
  });

  $effect(() => {
    window.addEventListener("message", onMessage);
    return () => {
      window.removeEventListener("message", onMessage);
      reset();
    };
  });
</script>

{#if frameSrc}
  <iframe
    bind:this={frame}
    class="component-frame"
    class:visible={rendered}
    src={frameSrc}
    sandbox="allow-scripts"
    referrerpolicy="no-referrer"
    {title}
    style:height={rendered ? `${height}px` : "0px"}
  ></iframe>
{/if}

<style>
  .component-frame { display: block; width: 100%; height: 0; border: 0; overflow: hidden; color-scheme: dark; }
  /* Keep the full width while hidden: the frame measures itself as soon as it
     loads, and a zero-width frame reflows its content into a far taller box
     than the one the component actually renders at. */
  .component-frame:not(.visible) { height: 0; visibility: hidden; }
</style>
