<script>
  import { Check, ChevronDown, Filter, Search } from "lucide-svelte";
  import FilterBulkActions from "./FilterBulkActions.svelte";

  let {
    recipes,
    recipeCounts,
    activeRecipes,
    allRecipesActive,
    statuses,
    statusCounts,
    activeStatuses,
    allStatusesActive,
    groups,
    groupCounts,
    activeGroups,
    allGroupsActive,
    trainingTypes,
    trainingTypeCounts,
    activeTrainingTypes,
    allTrainingTypesActive,
    search = $bindable(),
    groupBy = $bindable(),
    onToggleRecipe,
    onSelectAllRecipes,
    onClearRecipes,
    onToggleStatus,
    onSelectAllStatuses,
    onClearStatuses,
    onToggleGroup,
    onSelectAllGroups,
    onClearGroups,
    onToggleTrainingType,
    onSelectAllTrainingTypes,
    onClearTrainingTypes,
  } = $props();

  let openMenu = $state(null);

  function toggleMenu(menu) {
    openMenu = openMenu === menu ? null : menu;
  }

  const groupByOptions = [
    { value: "none", label: "None" },
    { value: "group", label: "Group" },
    { value: "dataset", label: "Dataset" },
    { value: "model", label: "Model" },
  ];

  let groupByLabel = $derived(
    groupByOptions.find((option) => option.value === groupBy)?.label ?? "None",
  );
</script>

<svelte:window onclick={() => (openMenu = null)} />

<nav class="p-0 flex items-center gap-[0.5rem] relative flex-wrap max-[900px]:[align-items:stretch]">
  <label class="inline-flex items-center gap-[8px] [border:1px_solid_var(--color-c-gray-10,#2f2f2f)] rounded-[6px] [background:transparent] w-[260px] p-[6px_8px] max-[900px]:w-full" aria-label="Search training runs by name">
    <span class="search-icon">
      <Search size={13} />
    </span>
    <input
      type="search"
      class="search-input"
      placeholder="Search"
      bind:value={search}
      autocomplete="off"
      spellcheck="false"
    />
  </label>

  <div class="filterbar-menu-wrap">
    <button
      class="filter-button ghost-hover"
      class:filterbar-open={openMenu === "status"}
      onclick={(event) => {
        event.stopPropagation();
        toggleMenu("status");
      }}
    >
      <span class="button-icon">
        <Filter size={12} />
      </span>
      <span>Status</span>
      <span class="chevron" class:rotated={openMenu === "status"}>
        <ChevronDown size={12} />
      </span>
    </button>
    {#if openMenu === "status"}
      <div class="menu">
        <FilterBulkActions
          allSelected={allStatusesActive}
          noneSelected={activeStatuses.size === 0}
          onSelectAll={onSelectAllStatuses}
          onDeselectAll={onClearStatuses}
        />
        {#each statuses as st (st)}
          <button
            class="menu-item"
            onclick={(event) => {
              event.stopPropagation();
              onToggleStatus(st);
            }}
          >
            <span class="checkmark" class:checked={activeStatuses.has(st)}>
              {#if activeStatuses.has(st)}
                <Check size={11} />
              {/if}
            </span>
            <span class="item-label capitalize">{st}</span>
            <span class="item-count">{statusCounts[st] || 0}</span>
          </button>
        {/each}
      </div>
    {/if}
  </div>

  <div class="filterbar-menu-wrap">
    <button
      class="filter-button ghost-hover"
      class:filterbar-open={openMenu === "recipes"}
      onclick={(event) => {
        event.stopPropagation();
        toggleMenu("recipes");
      }}
    >
      <span class="button-icon">
        <Filter size={12} />
      </span>
      <span>Recipe</span>
      <span class="chevron" class:rotated={openMenu === "recipes"}>
        <ChevronDown size={12} />
      </span>
    </button>
    {#if openMenu === "recipes"}
      <div class="menu">
        <FilterBulkActions
          allSelected={allRecipesActive}
          noneSelected={activeRecipes.size === 0}
          onSelectAll={onSelectAllRecipes}
          onDeselectAll={onClearRecipes}
        />
        {#each recipes as recipe (recipe)}
          <button
            class="menu-item"
            onclick={(event) => {
              event.stopPropagation();
              onToggleRecipe(recipe);
            }}
          >
            <span class="checkmark" class:checked={activeRecipes.has(recipe)}>
              {#if activeRecipes.has(recipe)}
                <Check size={11} />
              {/if}
            </span>
            <span class="item-label">{recipe}</span>
            <span class="item-count">{recipeCounts[recipe] || 0}</span>
          </button>
        {/each}
      </div>
    {/if}
  </div>

  <div class="filterbar-menu-wrap">
    <button
      class="filter-button ghost-hover"
      class:filterbar-open={openMenu === "groups"}
      onclick={(event) => {
        event.stopPropagation();
        toggleMenu("groups");
      }}
    >
      <span class="button-icon">
        <Filter size={12} />
      </span>
      <span>Group</span>
      <span class="chevron" class:rotated={openMenu === "groups"}>
        <ChevronDown size={12} />
      </span>
    </button>
    {#if openMenu === "groups"}
      <div class="menu">
        <FilterBulkActions
          allSelected={allGroupsActive}
          noneSelected={activeGroups.size === 0}
          onSelectAll={onSelectAllGroups}
          onDeselectAll={onClearGroups}
        />
        {#each groups as group (group)}
          <button
            class="menu-item"
            onclick={(event) => {
              event.stopPropagation();
              onToggleGroup(group);
            }}
          >
            <span class="checkmark" class:checked={activeGroups.has(group)}>
              {#if activeGroups.has(group)}
                <Check size={11} />
              {/if}
            </span>
            <span class="item-label">{group}</span>
            <span class="item-count">{groupCounts[group] || 0}</span>
          </button>
        {/each}
      </div>
    {/if}
  </div>

  <div class="filterbar-menu-wrap">
    <button
      class="filter-button ghost-hover"
      class:filterbar-open={openMenu === "training_type"}
      onclick={(event) => {
        event.stopPropagation();
        toggleMenu("training_type");
      }}
    >
      <span class="button-icon">
        <Filter size={12} />
      </span>
      <span>Training type</span>
      <span class="chevron" class:rotated={openMenu === "training_type"}>
        <ChevronDown size={12} />
      </span>
    </button>
    {#if openMenu === "training_type"}
      <div class="menu">
        <FilterBulkActions
          allSelected={allTrainingTypesActive}
          noneSelected={activeTrainingTypes.size === 0}
          onSelectAll={onSelectAllTrainingTypes}
          onDeselectAll={onClearTrainingTypes}
        />
        {#each trainingTypes as value (value)}
          <button
            class="menu-item"
            onclick={(event) => {
              event.stopPropagation();
              onToggleTrainingType(value);
            }}
          >
            <span class="checkmark" class:checked={activeTrainingTypes.has(value)}>
              {#if activeTrainingTypes.has(value)}
                <Check size={11} />
              {/if}
            </span>
            <span class="item-label uppercase">{value}</span>
            <span class="item-count">{trainingTypeCounts[value] || 0}</span>
          </button>
        {/each}
      </div>
    {/if}
  </div>

  <div class="filterbar-menu-wrap ml-auto">
    <button
      class="group-by-button ghost-hover"
      class:filterbar-open={openMenu === "groupBy"}
      aria-haspopup="listbox"
      aria-expanded={openMenu === "groupBy"}
      onclick={(event) => {
        event.stopPropagation();
        toggleMenu("groupBy");
      }}
    >
      <span>Group by:</span>
      <span class="group-by-value">{groupByLabel}</span>
      <span class="chevron" class:rotated={openMenu === "groupBy"}>
        <ChevronDown size={12} />
      </span>
    </button>
    {#if openMenu === "groupBy"}
      <div class="menu menu-right group-by-menu" role="listbox">
        {#each groupByOptions as option (option.value)}
          <button
            class="menu-item"
            role="option"
            aria-selected={groupBy === option.value}
            onclick={() => {
              groupBy = option.value;
              openMenu = null;
            }}
          >
            <span class="w-[14px] h-[14px] flex justify-center items-center">
              {#if groupBy === option.value}
                <Check size={11} />
              {/if}
            </span>
            <span class="item-label">{option.label}</span>
          </button>
        {/each}
      </div>
    {/if}
  </div>
</nav>
