<script>
  // Underlined tab bar, ported from Modal's UnderlinedTabList / UnderlinedTab:
  // a bottom-bordered nav where the active tab gets a green underline + bright
  // text. `tabs` is [{ value, label, count?, href? }] — an entry with `href`
  // renders as an external link instead of a selectable tab.
  import { ExternalLink } from "lucide-svelte";

  let { tabs = [], active, onSelect } = $props();
</script>

<div class="flex shrink-0 p-[0_24px] [border-bottom:1px_solid_var(--color-c-gray-10,#2f2f2f)] gap-8" role="tablist">
  {#each tabs as tab (tab.value)}
    {#if tab.href}
      <a class="tab" href={tab.href} target="_blank" rel="noopener noreferrer">
        <span>{tab.label}</span>
        <ExternalLink size={12} strokeWidth={2.1} />
      </a>
    {:else}
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
    {/if}
  {/each}
</div>
