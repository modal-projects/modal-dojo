<script>
  import { Check, ChevronDown, Search, X } from "lucide-svelte";
  import { innerWidth } from "svelte/reactivity/window";
  import FilterDropdown from "./FilterDropdown.svelte";
  import { tagOptions } from "../lib/filterOptions.js";

  let {
    recipes, recipeCounts, activeRecipes, allRecipesActive,
    statuses, statusCounts, activeStatuses, allStatusesActive,
    groups, groupCounts, activeGroups, allGroupsActive,
    tagCounts, activeTags,
    trainingTypes, trainingTypeCounts, activeTrainingTypes, allTrainingTypesActive,
    search = $bindable(), groupBy = $bindable(),
    onToggleRecipe, onSelectAllRecipes, onClearRecipes,
    onToggleStatus, onSelectAllStatuses, onClearStatuses,
    onToggleGroup, onSelectAllGroups, onClearGroups,
    onAddTagValue, onRemoveTagValue, onClearTags,
    onToggleTrainingType, onSelectAllTrainingTypes, onClearTrainingTypes,
    // Pickers that list runs flat (the Metrics page's Add-run modal) hide it.
    showGroupBy = true,
  } = $props();

  let root = $state();
  let searchInput = $state();
  let openMenu = $state(null);
  let view = $state(null);
  let forceShow = $state(false);
  let maxVisibleChips = $derived(innerWidth.current >= 640 ? 3 : 2);

  function facet(id, label, values, counts, selected, allSelected, toggle, clear, reset, format = (value) => value) {
    return {
      id, label, active: !allSelected, clear, reset,
      options: values.map((value) => ({ id: value, value, label: format(value), count: counts[value] || 0, selected: selected.has(value) })),
      toggle: (item) => toggle(item.value),
      only: (item) => { clear(); toggle(item.value); },
    };
  }

  let categories = $derived([
    facet("status", "Status", statuses, statusCounts, activeStatuses, allStatusesActive, onToggleStatus, onClearStatuses, onSelectAllStatuses, (value) => value.charAt(0).toUpperCase() + value.slice(1)),
    facet("recipe", "Recipe", recipes, recipeCounts, activeRecipes, allRecipesActive, onToggleRecipe, onClearRecipes, onSelectAllRecipes),
    facet("group", "Group", groups, groupCounts, activeGroups, allGroupsActive, onToggleGroup, onClearGroups, onSelectAllGroups),
    facet("training_type", "Type", trainingTypes, trainingTypeCounts, activeTrainingTypes, allTrainingTypesActive, onToggleTrainingType, onClearTrainingTypes, onSelectAllTrainingTypes, (value) => value.toUpperCase()),
    {
      id: "tags", label: "Tag", active: Object.values(activeTags).some((values) => values.length),
      options: tagOptions(tagCounts, activeTags), clear: onClearTags, reset: onClearTags,
      toggle: (item) => item.selected ? onRemoveTagValue(item.key, item.value) : onAddTagValue(item.key, item.value),
      only: (item) => { onClearTags(); onAddTagValue(item.key, item.value); },
      add: (item) => onAddTagValue(item.key, item.value),
    },
  ]);

  let showSearchMenu = $derived(openMenu === "search" && (forceShow || !search.trim()));

  function openCategory(id) {
    openMenu = id;
    view = id;
  }

  function handleFocusOut(event) {
    if (!event.currentTarget.contains(event.relatedTarget)) openMenu = null;
  }

  function handleKeydown(event) {
    if (event.key === "Escape") {
      openMenu = null;
      searchInput?.blur();
    }
    if (event.key !== "/" || event.ctrlKey || event.metaKey || event.altKey || event.defaultPrevented) return;
    if (event.target instanceof Element && event.target.closest("input, textarea, select, [contenteditable]:not([contenteditable='false'])")) return;
    event.preventDefault();
    searchInput?.focus();
    view = null;
    forceShow = true;
    openMenu = "search";
  }

  const groupByOptions = [
    { value: "none", label: "None" },
    { value: "group", label: "Group" },
    { value: "dataset", label: "Dataset" },
    { value: "model", label: "Model" },
  ];
  let groupByLabel = $derived(groupByOptions.find((option) => option.value === groupBy)?.label ?? "None");
</script>

<svelte:window onkeydown={handleKeydown} onpointerdown={(event) => { if (!root?.contains(event.target)) openMenu = null; }} />

<nav bind:this={root} class="filter-row">
  <div class="search-filter-wrap" onfocusout={handleFocusOut}>
    <div class="search-filter-control">
      <span class="search-icon"><Search size={16} /></span>
      <input
        class="search-input"
        type="search"
        placeholder="Search or filter"
        aria-label="Search or filter training runs"
        bind:value={search}
        bind:this={searchInput}
        autocomplete="off"
        spellcheck="false"
        onfocus={() => { openMenu = "search"; view = null; forceShow = false; }}
        oninput={() => { forceShow = false; view = null; openMenu = "search"; }}
      />
      {#if !search.trim() && openMenu === null}<kbd>/</kbd>{/if}
      <button
        class="icon-control"
        aria-label="Toggle filters"
        aria-expanded={showSearchMenu}
        onclick={() => {
          if (showSearchMenu) { openMenu = null; searchInput?.blur(); }
          else { searchInput?.focus(); forceShow = true; view = null; openMenu = "search"; }
        }}
      ><ChevronDown size={16} /></button>
    </div>
    {#if showSearchMenu}
      <div class="search-filter-menu">
        <FilterDropdown {categories} {view} onViewChange={(id) => { if (id === null) searchInput?.focus(); view = id; }} />
      </div>
    {/if}
  </div>

  {#each categories.filter((category) => category.active) as category (category.id)}
    {@const selected = category.options.filter((item) => item.selected)}
    {@const hiddenCount = Math.max(0, selected.length - maxVisibleChips)}
    <div class="pill-wrap" onfocusout={handleFocusOut}>
      <div class="category-pill">
        <button class="pill-label" aria-label={`Edit ${category.label} filters`} onclick={() => openCategory(category.id)}>{category.label}:</button>
        <div class="pill-values">
          {#each selected.slice(0, maxVisibleChips) as item (item.id)}
            <div class="chip-slot">
              <span class="filter-chip">
                <button class="chip-label" title={item.label} onclick={() => openCategory(category.id)}>{item.label}</button>
                <button class="icon-control" aria-label={`Remove ${category.label} ${item.label}`} onclick={() => category.toggle(item)}><X size={12} /></button>
              </span>
            </div>
          {:else}
            <button class="chip-label" onclick={() => openCategory(category.id)}>None</button>
          {/each}
          {#if hiddenCount > 0}
            <button class="pill-overflow" aria-label={`Show ${hiddenCount} more ${category.label} filters`} onclick={() => openCategory(category.id)}>
              <span>+{hiddenCount}</span>
              {#if hiddenCount >= 2}
                <svg width="8" height="20" viewBox="0 0 8 20" aria-hidden="true"><path d="M4 0C6.20914 0 8 1.79086 8 4V16C8 18.2091 6.20914 20 4 20H0C2.20914 20 4 18.2091 4 16V4C4 1.79086 2.20914 0 0 0H4Z" /></svg>
              {/if}
            </button>
          {/if}
        </div>
        <button class="icon-control" aria-label={`Dismiss ${category.label} filter`} onclick={() => { category.reset(); openMenu = null; }}><X size={16} /></button>
      </div>
      {#if openMenu === category.id}
        <div class="search-filter-menu pill-menu">
          <FilterDropdown {categories} view={category.id} showBack={false} onViewChange={(id) => (view = id)} />
        </div>
      {/if}
    </div>
  {/each}

  {#if showGroupBy}
  <div class="filterbar-menu-wrap ml-auto">
    <button
      class="group-by-button ghost-hover"
      class:filterbar-open={openMenu === "groupBy"}
      aria-haspopup="listbox"
      aria-expanded={openMenu === "groupBy"}
      onclick={(event) => {
        event.stopPropagation();
        openMenu = openMenu === "groupBy" ? null : "groupBy";
      }}
    >
      <span>Group by:</span>
      <span class="group-by-value">{groupByLabel}</span>
      <span class="chevron" class:rotated={openMenu === "groupBy"}><ChevronDown size={12} /></span>
    </button>
    {#if openMenu === "groupBy"}
      <div class="menu menu-right group-by-menu" role="listbox">
        {#each groupByOptions as option (option.value)}
          <button class="menu-item" role="option" aria-selected={groupBy === option.value} onclick={() => { groupBy = option.value; openMenu = null; }}>
            <span class="w-[14px] h-[14px] flex justify-center items-center">{#if groupBy === option.value}<Check size={11} />{/if}</span>
            <span class="item-label">{option.label}</span>
          </button>
        {/each}
      </div>
    {/if}
  </div>
  {/if}
</nav>

<style>
  .filter-row { display: flex; align-items: center; flex-wrap: wrap; gap: 8px; font-size: 14px; line-height: 20px; }
  .search-filter-wrap { position: relative; width: 360px; max-width: 100%; flex-shrink: 0; }
  .search-filter-control { height: 32px; display: flex; align-items: center; gap: 8px; padding: 0 8px; border: 1px solid var(--border); border-radius: 6px; background: transparent; }
  .search-filter-control:focus-within { border-color: var(--border-strong); }
  .search-filter-control kbd { color: var(--muted); background: var(--panel-alt); padding: 0 6px; border-radius: 4px; font-size: 12px; line-height: 16px; }
  .search-filter-menu { position: absolute; top: calc(100% + 4px); left: 0; width: 100%; z-index: 20; }
  .pill-wrap { position: relative; max-width: 100%; }
  .category-pill { display: flex; align-items: center; gap: 4px; height: 32px; max-width: 100%; padding: 0 8px; border: 1px solid var(--border); border-radius: 6px; background: var(--panel-alt); }
  .category-pill:hover { border-color: var(--border-strong); background: var(--color-c-surface-highlight-gray-opaque); }
  .category-pill .icon-control { color: var(--muted-strong); }
  .category-pill:hover > .icon-control, .filter-chip:hover .icon-control { color: var(--muted); }
  button { border: 0; background: transparent; color: var(--text); cursor: pointer; }
  .pill-label { font-weight: 500; white-space: nowrap; }
  .icon-control { display: inline-flex; align-items: center; flex-shrink: 0; color: var(--muted); }
  .icon-control:hover { color: var(--text-bright); }
  .pill-values { display: flex; align-items: center; flex-wrap: nowrap; gap: 4px; min-width: 0; }
  .chip-slot { display: flex; align-items: center; max-width: 140px; min-width: 0; }
  .filter-chip { display: inline-flex; align-items: center; gap: 4px; max-width: 100%; min-width: 0; height: 20px; padding: 0 6px; border: 0; border-radius: 4px; background: var(--surface); color: var(--color-c-gray-70); font-size: 12px; line-height: 16px; }
  .filter-chip:hover { background: var(--surface-raised); }
  .filter-chip .chip-label { color: inherit; }
  .chip-label { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; min-width: 0; }
  .pill-overflow { display: flex; align-items: center; flex-shrink: 0; padding: 0; color: var(--color-c-gray-70); }
  .pill-overflow span { background: var(--surface); border-radius: 4px; padding: 4px 6px; font-size: 12px; line-height: 12px; font-variant-numeric: tabular-nums; }
  .pill-overflow svg { margin-left: -2px; fill: var(--surface); }
  .pill-menu { width: 320px; max-width: 90vw; }
  @media (min-width: 640px) and (max-width: 1023px) { .pill-wrap { width: 100%; } }
  @media (min-width: 840px) { .chip-slot { max-width: 200px; } }
</style>
