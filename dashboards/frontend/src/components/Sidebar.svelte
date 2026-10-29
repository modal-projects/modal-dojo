<script>
  import { LogOut, PanelLeftClose, PanelLeftOpen } from "lucide-svelte";
  import { fetchPasswordAuth } from "../lib/api.js";

  let {
    navItems,
    activePage,
    onNavigate,
    collapsed = false,
    onToggleCollapsed,
  } = $props();

  const navId = $props.id();
  // Matches the modifier `App.svelte` binds for the toggle hotkey.
  const isMac =
    typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform);
  const hotkeyLabel = isMac ? "⌘B" : "Ctrl+B";

  // Only a password-protected dashboard has a session to end.
  let passwordProtected = $state(false);
  $effect(() => {
    fetchPasswordAuth()
      .then((value) => (passwordProtected = value))
      .catch(() => {});
  });
</script>

<aside
  class={[
    "flex flex-col min-h-0 [border-right:1px_solid_rgba(255,255,255,0.1)] bg-(--bg-depth) transition-[padding] duration-100 ease-out max-[900px]:[border-right:0] max-[900px]:[border-bottom:1px_solid_var(--border)] max-[900px]:p-[0_8px]",
    collapsed ? "p-[4px_8px_0]" : "p-[4px_20px_0]",
  ]}
  aria-label="Primary"
>
  <nav
    id={navId}
    class="flex flex-col grow gap-[4px] pt-[12px] max-[900px]:grow-0 max-[900px]:flex-row max-[900px]:gap-[4px] max-[900px]:pt-[6px] max-[900px]:pb-[6px] max-[900px]:overflow-x-auto max-[900px]:[-webkit-overflow-scrolling:touch]"
    aria-label="Sections"
  >
    {#each navItems as item (item.key)}
      <a
        href={item.path}
        class="nav-item overflow-hidden whitespace-nowrap max-[900px]:flex-[1_1_0] max-[900px]:justify-center max-[900px]:p-[10px_8px] max-[900px]:min-h-[44px] max-[900px]:text-[13px]"
        class:sidebar-active={activePage === item.key}
        aria-current={activePage === item.key ? "page" : undefined}
        aria-label={item.label}
        title={collapsed ? item.label : undefined}
        onclick={(event) => {
          event.preventDefault();
          onNavigate(item.key);
        }}
      >
        <span class="inline-flex items-center flex-[0_0_auto] opacity-[0.6]">
          <item.Icon size={14} strokeWidth={2.1} />
        </span>
        <span
          class={[
            "transition-opacity duration-150 ease-out max-[900px]:opacity-100",
            collapsed ? "opacity-0" : "opacity-100",
          ]}
        >
          {item.label}
        </span>
      </a>
    {/each}
  </nav>

  <div
    class={[
      "[border-top:1px_solid_rgba(255,255,255,0.1)] py-[8px] max-[900px]:hidden",
      collapsed ? "mx-[-8px] px-[8px]" : "mx-[-20px] px-[20px]",
    ]}
  >
    {#if passwordProtected}
      <form method="post" action="/logout">
        <button
          type="submit"
          class="nav-item w-full overflow-hidden whitespace-nowrap"
          aria-label="Log out"
          title={collapsed ? "Log out" : undefined}
        >
          <span class="inline-flex items-center flex-[0_0_auto] opacity-[0.6]">
            <LogOut size={14} strokeWidth={2.1} />
          </span>
          <span
            class={[
              "transition-opacity duration-150 ease-out",
              collapsed ? "opacity-0" : "opacity-100",
            ]}
          >
            Log out
          </span>
        </button>
      </form>
    {/if}
    <button
      type="button"
      class="nav-item w-full"
      aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
      aria-controls={navId}
      aria-expanded={!collapsed}
      title="{collapsed ? 'Expand' : 'Collapse'} sidebar {hotkeyLabel}"
      onclick={onToggleCollapsed}
    >
      <span class="inline-flex items-center flex-[0_0_auto] opacity-[0.6]">
        {#if collapsed}
          <PanelLeftOpen size={14} strokeWidth={2.1} />
        {:else}
          <PanelLeftClose size={14} strokeWidth={2.1} />
        {/if}
      </span>
    </button>
  </div>
</aside>
