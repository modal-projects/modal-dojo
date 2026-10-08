export const DOCS_NAV = [
  { section: 'Guides', prefix: '/guides' },
  { section: 'Tutorials', prefix: '/tutorials' },
  { section: 'Reference', prefix: '/reference' },
] as const;

export const REFERENCE_SURFACES = [
  { label: 'SDK', href: '/reference/sdk', prefix: '/reference' },
  { label: 'CLI', href: '/reference/cli', prefix: '/reference/cli' },
] as const;

export type DocsSection = 'Home' | (typeof DOCS_NAV)[number]['section'];
export type ReferenceSurface = (typeof REFERENCE_SURFACES)[number];

const PREFIXES_LONGEST_FIRST = [...DOCS_NAV].sort(
  (a, b) => b.prefix.length - a.prefix.length,
);

const SURFACES_LONGEST_FIRST = [...REFERENCE_SURFACES].sort(
  (a, b) => b.prefix.length - a.prefix.length,
);

function normalizedPath(url: string): string {
  const pathname = url.split(/[?#]/)[0] ?? url;
  return pathname.length > 1 && pathname.endsWith('/') ? pathname.slice(0, -1) : pathname;
}

function matchesPrefix(path: string, prefix: string): boolean {
  return path === prefix || path.startsWith(`${prefix}/`);
}

export function sectionForUrl(url: string): DocsSection | undefined {
  const path = normalizedPath(url);
  if (path === '/') return 'Home';
  for (const { section, prefix } of PREFIXES_LONGEST_FIRST) {
    if (matchesPrefix(path, prefix)) return section;
  }
  return undefined;
}

export function referenceSurfaceForPath(path: string): ReferenceSurface | undefined {
  const normalized = normalizedPath(path);
  return SURFACES_LONGEST_FIRST.find((surface) =>
    matchesPrefix(normalized, surface.prefix),
  );
}

export function isIdentifierTitlePath(url: string): boolean {
  const path = normalizedPath(url);
  const surface = referenceSurfaceForPath(path);
  return Boolean(surface && path !== surface.href && path !== surface.prefix);
}

export type SidebarLinkEntry = {
  type: 'link';
  label: string;
  href: string;
  isCurrent?: boolean;
};

export type SidebarGroupEntry = {
  type: 'group';
  label: string;
  entries: readonly SidebarTreeEntry[];
};

export type SidebarTreeEntry = SidebarLinkEntry | SidebarGroupEntry;

export type ReferenceSidebarSection = {
  label: string;
  href: string;
  isCurrent: boolean;
  items: SidebarLinkEntry[];
};

function samePath(left: string, right: string): boolean {
  return normalizedPath(left) === normalizedPath(right);
}

export function flattenSidebarLinks(
  entries: readonly SidebarTreeEntry[],
): SidebarLinkEntry[] {
  const links: SidebarLinkEntry[] = [];
  for (const entry of entries) {
    if (entry.type === 'link') {
      links.push(entry);
    } else {
      links.push(...flattenSidebarLinks(entry.entries));
    }
  }
  return links;
}

export function compareSdkSidebarLabels(left: string, right: string): number {
  const leftFunction = /^[a-z]/.test(left) ? 1 : 0;
  const rightFunction = /^[a-z]/.test(right) ? 1 : 0;
  if (leftFunction !== rightFunction) return leftFunction - rightFunction;
  return left.localeCompare(right, 'en', { sensitivity: 'base' });
}

export function referenceSidebarSections(
  surface: ReferenceSurface,
  groupEntries: readonly SidebarTreeEntry[],
  currentPath: string,
): ReferenceSidebarSection[] {
  const items = flattenSidebarLinks(groupEntries).filter(
    (entry) => !samePath(entry.href, surface.href),
  );
  if (surface.label === 'SDK') {
    items.sort((left, right) => compareSdkSidebarLabels(left.label, right.label));
  }
  return [
    {
      label: surface.label,
      href: surface.href,
      isCurrent: samePath(currentPath, surface.href),
      items,
    },
  ];
}

export function flattenDocId(entry: string): string {
  const withoutExt = entry.replace(/\.[^./]+$/, '');
  const withoutIndex = withoutExt.replace(/\/index$/, '');
  const parts = withoutIndex.split('/').filter(Boolean);
  const idPath = `/${parts.join('/')}`;
  const nested = REFERENCE_SURFACES.some(
    (surface) =>
      surface.prefix.split('/').filter(Boolean).length >= 2 &&
      matchesPrefix(idPath, surface.prefix),
  );
  if (parts.length <= 2 || nested) {
    return withoutIndex;
  }
  return `${parts[0]}/${parts[parts.length - 1]}`;
}
