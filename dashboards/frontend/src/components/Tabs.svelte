<script>
  // Underlined tab bar, ported from Modal's UnderlinedTabList / UnderlinedTab:
  // a bottom-bordered nav where the active tab gets a green underline + bright
  // text. `tabs` is [{ value, label, count? }].
  let { tabs = [], active, onSelect } = $props();
</script>

<div class="flex shrink-0 p-[0_24px] [border-bottom:1px_solid_var(--color-c-gray-10,#2f2f2f)] gap-8" role="tablist">
  {#each tabs as tab (tab.value)}
    <button
      class="tab"
      class:tabs-active={active === tab.value}
      role="tab"
      aria-selected={active === tab.value}
      onclick={() => onSelect(tab.value)}
    >
      <span>{tab.label}</span>
      {#if tab.count != null}
        <span class="tab-count">{tab.count}</span>
      {/if}
    </button>
  {/each}
</div>
