<script>
  import { onDestroy } from "svelte";
  import { ArrowDown, ArrowUp, ArrowUpDown } from "lucide-svelte";
  import MinimalTable from "./MinimalTable.svelte";

  let {
    columns = [],
    stickyFirstColumn = false,
    sort = null,
    onSort = null,
    class: classOverride = "",
    style: styleOverride = "",
    ...restProps
  } = $props();

  let columnWidths = $state({});
  let resizeState = $state(null);
  let tableWidth = $derived(
    columns.reduce((total, column) => total + columnWidth(column), 0),
  );

  function columnWidth(column) {
    return columnWidths[column.key] ?? column.width;
  }

  function startColumnResize(event, column) {
    event.preventDefault();
    event.stopPropagation();

    resizeState = {
      key: column.key,
      minWidth: column.minWidth,
      startX: event.clientX,
      startWidth: columnWidth(column),
      previousUserSelect: document.body.style.userSelect,
      previousCursor: document.body.style.cursor,
    };

    document.body.style.userSelect = "none";
    document.body.style.cursor = "col-resize";
    window.addEventListener("pointermove", resizeColumn);
    window.addEventListener("pointerup", stopColumnResize, { once: true });
    window.addEventListener("pointercancel", stopColumnResize, { once: true });
  }

  function resizeColumn(event) {
    if (!resizeState) return;
    const nextWidth = Math.max(
      resizeState.minWidth,
      Math.round(resizeState.startWidth + event.clientX - resizeState.startX),
    );
    columnWidths = { ...columnWidths, [resizeState.key]: nextWidth };
  }

  function stopColumnResize() {
    if (!resizeState) return;
    document.body.style.userSelect = resizeState.previousUserSelect;
    document.body.style.cursor = resizeState.previousCursor;
    resizeState = null;
    window.removeEventListener("pointermove", resizeColumn);
    window.removeEventListener("pointerup", stopColumnResize);
    window.removeEventListener("pointercancel", stopColumnResize);
  }

  function stopFrozenColumnHorizontalScroll(event) {
    if (!stickyFirstColumn || Math.abs(event.deltaX) < 4 || Math.abs(event.deltaY) > 1) return;
    const firstColumnCell = event.target?.closest?.("th:first-child, td:first-child");
    if (!firstColumnCell) return;
    event.preventDefault();
    event.stopPropagation();
  }

  onDestroy(() => {
    stopColumnResize();
  });
</script>

<MinimalTable
  {...restProps}
  class={`resizable-table ${stickyFirstColumn ? "sticky-first-column" : ""} ${classOverride}`.trim()}
  onwheel={stopFrozenColumnHorizontalScroll}
  style={`--resizable-grid-width: ${tableWidth}px; ${styleOverride}`.trim()}
>
  <colgroup>
    {#each columns as column (column.key)}
      <col style={`width: ${columnWidth(column)}px;`} />
    {/each}
  </colgroup>
  <thead>
    <tr>
      {#each columns as column (column.key)}
        <th class:resizing={resizeState?.key === column.key}>
          {#if column.sortable && onSort}
            <button
              type="button"
              class="column-sort"
              class:sort-active={sort?.key === column.key}
              aria-label={`Sort by ${column.ariaLabel || column.label || "column"}`}
              onclick={() => onSort(column.key)}
            >
              <span class="column-label">{column.label}</span>
              {#if sort?.key === column.key && sort.dir === "asc"}
                <ArrowUp size={11} strokeWidth={2.2} />
              {:else if sort?.key === column.key}
                <ArrowDown size={11} strokeWidth={2.2} />
              {:else}
                <ArrowUpDown size={11} strokeWidth={2.2} class="sort-idle" />
              {/if}
            </button>
          {:else}
            <span class="column-label">{column.label}</span>
          {/if}
          <button
            type="button"
            class="column-resize-handle"
            aria-label={`Resize ${column.ariaLabel || column.label || "column"} column`}
            onpointerdown={(event) => startColumnResize(event, column)}
          ></button>
        </th>
      {/each}
    </tr>
  </thead>
  <!-- svelte-ignore slot_element_deprecated -->
  <slot />
</MinimalTable>
