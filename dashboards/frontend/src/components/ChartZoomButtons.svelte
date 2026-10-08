<script>
  // Emit brush-compatible fractions that halve or double the visible window.
  import ZoomInIcon from "lucide-svelte/icons/zoom-in";
  import ZoomOutIcon from "lucide-svelte/icons/zoom-out";

  let { onChangeDomainX, canZoomIn = true, canZoomOut = true } = $props();
</script>

<!-- Capture before the plot's native brush listener runs. -->
<div
  class="btn-group chart-zoom-buttons"
  role="group"
  aria-label="Zoom"
  onpointerdowncapture={(event) => event.stopPropagation()}
>
  <button
    class="btn"
    type="button"
    disabled={!canZoomIn}
    onclick={() => onChangeDomainX([0.25, 0.75])}
    title="Zoom in"
    aria-label="Zoom in"
  >
    <ZoomInIcon size={12} />
  </button>
  <button
    class="btn"
    type="button"
    disabled={!canZoomOut}
    onclick={() => onChangeDomainX([-0.5, 1.5])}
    title="Zoom out"
    aria-label="Zoom out"
  >
    <ZoomOutIcon size={12} />
  </button>
</div>
