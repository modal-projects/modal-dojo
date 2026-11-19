<script>
  import { Check, ChevronLeft, ChevronRight, Plus } from "lucide-svelte";
  import { newTagOption } from "../lib/filterOptions.js";

  let { categories, view, onViewChange, showBack = true } = $props();
  let query = $state("");
  let searchInput = $state();
  let category = $derived(categories.find((item) => item.id === view));
  let filtered = $derived(category?.options.filter((item) => item.label.toLowerCase().includes(query.toLowerCase())) || []);
  let candidate = $derived(category?.id === "tags" ? newTagOption(query, category.options) : null);

  $effect(() => {
    view;
    query = "";
  });
  $effect(() => {
    if (view && searchInput) searchInput.focus();
  });
</script>

<div
  class="filter-dropdown"
  role="menu"
  tabindex={-1}
  aria-label={category ? `${category.label} filters` : "Filter categories"}
  onmousedown={(event) => { if (!(event.target instanceof HTMLInputElement)) event.preventDefault(); }}
>
  {#if category}
    <div class="submenu-header">
      {#if showBack}
        <button class="icon-control" aria-label="Back to filters" onclick={() => onViewChange(null)}><ChevronLeft size={16} /></button>
      {/if}
      <input
        class="search-input"
        type="search"
        placeholder="Search"
        aria-label={`Search ${category.label} options`}
        bind:value={query}
        bind:this={searchInput}
        autocomplete="off"
        spellcheck="false"
        onkeydown={(event) => {
          if (event.key === "Enter" && candidate && !event.isComposing) {
            event.preventDefault();
            category.add(candidate);
            query = "";
          }
        }}
      />
      <button class="clear-control" onclick={category.clear}>Clear</button>
    </div>
    <div class="option-list">
      {#each filtered as item (item.id)}
        <div class="option-row">
          <button class="option-toggle" role="menuitemcheckbox" aria-checked={item.selected} aria-label={item.label} onclick={() => category.toggle(item)}>
            <span class="checkmark" class:checked={item.selected}>{#if item.selected}<Check size={12} />{/if}</span>
            <span class="item-label" title={item.label}>{item.label}</span>
            <span class="item-count">{item.count}</span>
          </button>
          {#if !item.selected || category.options.filter((option) => option.selected).length > 1}
            <button class="only-control" aria-label={`Only ${item.label}`} onclick={() => category.only(item)}>Only</button>
          {/if}
        </div>
      {/each}
      {#if candidate}
        <button class="add-option" role="menuitem" onclick={() => { category.add(candidate); query = ""; }}>
          <Plus size={16} /><span class="item-label">Add {candidate.label}</span>
        </button>
      {:else if filtered.length === 0}
        <p class="empty-options">No matching options</p>
      {/if}
    </div>
  {:else}
    {#each categories as item (item.id)}
      <button class="category-option" role="menuitem" onclick={() => onViewChange(item.id)}>
        <span>{item.label}</span><ChevronRight size={16} />
      </button>
    {/each}
  {/if}
</div>

<style>
  .filter-dropdown { display: flex; flex-direction: column; max-height: 320px; border: 1px solid var(--border); border-radius: 6px; background: var(--panel); padding: 4px 0; box-shadow: 0 12px 26px color-mix(in srgb, black 55%, transparent); font-size: 14px; line-height: 20px; }
  button { color: var(--text); border: 0; background: transparent; cursor: pointer; }
  .category-option, .add-option { display: flex; align-items: center; justify-content: space-between; gap: 8px; padding: 8px; margin: 0 4px; width: calc(100% - 8px); border-radius: 4px; text-align: left; }
  .category-option:hover, .add-option:hover, .option-row:hover { background: var(--panel-alt); }
  .submenu-header { display: flex; align-items: center; flex-shrink: 0; gap: 8px; padding: 6px 16px; border-bottom: 1px solid var(--border); }
  .icon-control { display: inline-flex; color: var(--muted); }
  .clear-control, .only-control { font-size: 12px; line-height: 16px; }
  .option-list { display: flex; flex-direction: column; gap: 4px; min-height: 0; overflow-y: auto; padding: 4px 8px; }
  .option-row { display: flex; align-items: center; gap: 8px; border-radius: 4px; padding: 8px; }
  .option-toggle { display: flex; align-items: center; gap: 8px; flex: 1; min-width: 0; text-align: left; }
  .option-toggle .item-label { flex: 1; }
  .item-count { font-size: 12px; line-height: 16px; }
  .checkmark { flex-shrink: 0; width: 18px; height: 18px; }
  .only-control { opacity: 0; }
  .option-row:hover .only-control, .option-row:focus-within .only-control { opacity: 1; }
  .empty-options { color: var(--muted); padding: 8px; }
  .add-option { justify-content: flex-start; }
</style>
