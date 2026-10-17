const DEFAULT_STORAGE_KEY = "sidebarCollapsed";

// Web Storage can be present but throw (storage blocked by browser policy),
// so a failed read is treated as no stored preference.
function readStored(key) {
  try {
    const raw = localStorage.getItem(key);
    if (raw === "true" || raw === "false") return raw === "true";
  } catch {
    // fall through
  }
  return undefined;
}

function writeStored(key, value) {
  try {
    localStorage.setItem(key, String(value));
  } catch {
    // in-memory state still applies for this page load
  }
}

// Reactive collapsed flag for a sidebar, persisted so the layout comes back
// the way it was left across reloads.
export function createSidebarCollapsedState(initial = false, key = DEFAULT_STORAGE_KEY) {
  let collapsed = $state(readStored(key) ?? initial);

  return {
    get collapsed() {
      return collapsed;
    },
    set collapsed(value) {
      collapsed = value;
      writeStored(key, value);
    },
    toggle() {
      this.collapsed = !collapsed;
    },
  };
}
