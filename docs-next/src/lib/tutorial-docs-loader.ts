import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { docsLoader } from '@astrojs/starlight/loaders';
import type { Loader, LoaderContext } from 'astro/loaders';
import { flattenDocId } from './docs-sections';
import { discoverTutorialEntries } from './tutorial-slugs';

const TUTORIAL_ENTRY_PREFIX = 'tutorials/';
const dependencyPattern =
  /^[A-Za-z0-9_.-]+(?: @ (?:git\+)?https:\/\/[A-Za-z0-9._/-]+(?:@[A-Za-z0-9._-]+)?)?$/;

function generateDocsId({
  entry,
  data,
}: {
  entry: string;
  data: Record<string, unknown>;
}): string {
  if (typeof data.slug === 'string' && data.slug.length > 0) {
    return data.slug;
  }
  return flattenDocId(entry);
}

export function parseTutorialMetadata(source: string, tutorialPath: string) {
  const isMarkdown = tutorialPath.endsWith('.md');
  const prefix = isMarkdown ? '' : '# ';
  const lines = source.split(/\r?\n/);
  if (lines[0] !== `${prefix}---`) {
    throw new Error(`${tutorialPath} must start with tutorial frontmatter`);
  }
  const frontmatterEnd = lines.indexOf(`${prefix}---`, 1);
  if (frontmatterEnd === -1) {
    throw new Error(`${tutorialPath} has unterminated tutorial frontmatter`);
  }

  const fieldPattern = new RegExp(`^${prefix}([a-z_]+):\\s*(.*)$`);
  const fields = new Map<string, string>();
  for (const line of lines.slice(1, frontmatterEnd)) {
    const match = line.match(fieldPattern);
    if (!match) {
      throw new Error(`${tutorialPath} has invalid frontmatter line: ${line}`);
    }
    const [, name, value] = match;
    if (name !== 'order' && name !== 'deps' && name !== 'github') {
      throw new Error(`${tutorialPath} has unsupported frontmatter field: ${name}`);
    }
    if (fields.has(name)) {
      throw new Error(`${tutorialPath} has duplicate frontmatter field: ${name}`);
    }
    fields.set(name, value);
  }

  const orderText = fields.get('order');
  if (!orderText || !/^\d+$/.test(orderText)) {
    throw new Error(`${tutorialPath} frontmatter requires a non-negative integer order`);
  }
  const order = Number(orderText);
  if (!Number.isSafeInteger(order)) {
    throw new Error(`${tutorialPath} frontmatter order exceeds the safe integer range`);
  }

  const github = fields.get('github');
  if (github !== undefined && !github.startsWith('https://')) {
    throw new Error(`${tutorialPath} frontmatter github must be an https URL`);
  }

  const deps = (fields.get('deps') ?? '')
    .split(',')
    .map((dependency) => dependency.trim())
    .filter(Boolean);
  if (new Set(deps).size !== deps.length) {
    throw new Error(`${tutorialPath} frontmatter deps must be unique`);
  }
  const invalidDeps = deps.filter((dependency) => !dependencyPattern.test(dependency));
  if (invalidDeps.length > 0) {
    throw new Error(`${tutorialPath} has invalid frontmatter deps: ${invalidDeps.join(', ')}`);
  }
  if (deps.length > 0 && (github !== undefined || isMarkdown)) {
    throw new Error(`${tutorialPath} cannot set deps when github is overridden or in Markdown`);
  }

  const contentLines = lines.slice(frontmatterEnd + 1);
  const titlePrefix = `${prefix}# `;
  const titleLine = contentLines.find((line) => line.startsWith(titlePrefix));
  if (!titleLine) {
    throw new Error(`${tutorialPath} is missing an H1 heading`);
  }

  return {
    order,
    title: titleLine.slice(titlePrefix.length).trim(),
    deps,
    github,
    isMarkdown,
    content: contentLines.join('\n'),
  };
}

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');
const tutorialsDirectory = path.join(repoRoot, 'tutorials');
const pyprojectPath = path.join(repoRoot, 'pyproject.toml');

export interface Tutorial {
  path: string;
  slug: string;
  runTarget: string;
  sourcePath: string;
  order: number;
  title: string;
  body: string;
  runCommand?: string;
  githubUrl?: string;
  deps: string[];
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function firstHeading(source: string): string | undefined {
  let body = source;
  if (body.startsWith('---')) {
    const frontmatterEnd = body.indexOf('\n---', 3);
    if (frontmatterEnd !== -1) {
      const afterFence = body.indexOf('\n', frontmatterEnd + 4);
      body = afterFence === -1 ? '' : body.slice(afterFence + 1);
    }
  }
  for (const line of body.split('\n')) {
    if (line.startsWith('# ')) {
      return line.slice(2).trim();
    }
  }
  return undefined;
}

function markdownFromComments(lines: string[]): string {
  return lines
    .map((line) => (line === '#' ? '' : line.slice(2)))
    .join('\n')
    .trim();
}

function renderBody(source: string): string {
  const lines = source.split('\n');
  const blocks: Array<{ kind: 'markdown' | 'code'; content: string }> = [];
  let index = 0;

  while (index < lines.length) {
    const isMarkdown = lines[index] === '#' || lines[index].startsWith('# ');
    const start = index;
    while (
      index < lines.length &&
      (lines[index] === '#' || lines[index].startsWith('# ')) === isMarkdown
    ) {
      index += 1;
    }
    const blockLines = lines.slice(start, index);
    const content = isMarkdown
      ? markdownFromComments(blockLines)
      : blockLines.join('\n').trim();
    if (content) {
      blocks.push({ kind: isMarkdown ? 'markdown' : 'code', content });
    }
  }

  return blocks
    .map((block) => {
      if (block.kind === 'code') {
        return `\`\`\`python\n${block.content}\n\`\`\``;
      }
      return block.content;
    })
    .filter(Boolean)
    .join('\n\n');
}

function resolveContentFile(filePath: string, root: URL): string {
  return path.isAbsolute(filePath) ? filePath : path.join(fileURLToPath(root), filePath);
}

async function derivedMarkdownData(
  id: string,
  data: Record<string, unknown>,
  filePath: string | undefined,
  root: URL,
): Promise<Record<string, unknown>> {
  if (typeof data.order !== 'number' || !Number.isInteger(data.order)) {
    throw new Error(`${id} frontmatter requires an integer order`);
  }
  if (!filePath) {
    throw new Error(`${id} is missing a file path`);
  }
  const title = firstHeading(await readFile(resolveContentFile(filePath, root), 'utf8'));
  if (!title) {
    throw new Error(`${id} is missing an H1 heading`);
  }
  return {
    ...data,
    title,
    sidebar: {
      ...(isRecord(data.sidebar) ? data.sidebar : {}),
      order: data.order,
    },
  };
}

function formatRunCommand(runTarget: string, extras: string[]): string {
  if (extras.length === 0) {
    return `uv run ${runTarget}`;
  }
  return `uv run ${extras.map((pkg) => `--with '${pkg}'`).join(' ')} ${runTarget}`;
}

async function readTutorial(
  tutorialPath: string,
  slug: string,
  runTarget: string,
  sourcePath: string,
): Promise<Tutorial> {
  const source = await readFile(tutorialPath, 'utf8');
  const { order, title, deps, github, isMarkdown, content } = parseTutorialMetadata(
    source,
    tutorialPath,
  );
  return {
    path: tutorialPath,
    slug,
    runTarget,
    sourcePath,
    order,
    title,
    body: isMarkdown ? content.trim() : renderBody(content),
    runCommand:
      github === undefined && !isMarkdown ? formatRunCommand(runTarget, deps) : undefined,
    githubUrl: github,
    deps,
  };
}

export async function loadTutorials(): Promise<Tutorial[]> {
  const tutorials = await Promise.all(
    (await discoverTutorialEntries(tutorialsDirectory)).map((entry) =>
      readTutorial(entry.path, entry.slug, entry.runTarget, entry.sourcePath),
    ),
  );
  tutorials.sort((left, right) => left.order - right.order || left.slug.localeCompare(right.slug));
  const expectedOrders = tutorials.map((_, index) => index);
  const actualOrders = tutorials.map((tutorial) => tutorial.order);
  if (actualOrders.some((order, index) => order !== expectedOrders[index])) {
    throw new Error(`Tutorial orders must be contiguous from 0: ${actualOrders.join(', ')}`);
  }
  return tutorials;
}

async function storeEntry(
  context: LoaderContext,
  id: string,
  data: Record<string, unknown>,
  body: string,
  filePath?: string,
): Promise<void> {
  const parsedData = await context.parseData({ id, data, filePath });
  context.store.set({
    id,
    data: parsedData,
    body,
    filePath,
    digest: context.generateDigest(JSON.stringify({ body, data })),
    rendered: await context.renderMarkdown(body),
  });
}

export function tutorialDocsLoader(): Loader {
  const starlightDocsLoader = docsLoader({ generateId: generateDocsId });
  let tutorialEntryIds = new Set<string>();

  async function syncTutorialEntries(context: LoaderContext): Promise<void> {
    const tutorials = await loadTutorials();
    const nextEntryIds = new Set<string>();
    for (const tutorial of tutorials) {
      const id = `${TUTORIAL_ENTRY_PREFIX}${tutorial.slug}`;
      nextEntryIds.add(id);
      await storeEntry(
        context,
        id,
        {
          title: tutorial.title,
          order: tutorial.order,
          sidebar: { order: tutorial.order },
          ...(tutorial.runCommand !== undefined
            ? { runCommand: tutorial.runCommand }
            : {}),
          ...(tutorial.githubUrl !== undefined
            ? { githubUrl: tutorial.githubUrl }
            : {}),
          sourcePath: tutorial.sourcePath,
        },
        tutorial.body,
        path.relative(fileURLToPath(context.config.root), tutorial.path),
      );
    }
    for (const id of tutorialEntryIds) {
      if (!nextEntryIds.has(id)) {
        context.store.delete(id);
      }
    }
    tutorialEntryIds = nextEntryIds;
  }

  return {
    name: 'modal-dojo-tutorial-docs-loader',
    async load(context) {
      const parseData = context.parseData.bind(context);
      context.parseData = async (args) => {
        if (!args.id.startsWith(TUTORIAL_ENTRY_PREFIX)) {
          args.data = (await derivedMarkdownData(
            args.id,
            args.data,
            args.filePath,
            context.config.root,
          )) as typeof args.data;
        }
        return parseData(args);
      };
      await starlightDocsLoader.load(context);
      await syncTutorialEntries(context);

      const { watcher } = context;
      if (!watcher) {
        return;
      }
      watcher.add([tutorialsDirectory, pyprojectPath]);

      const isTutorialSource = (changedPath: string) => {
        const resolvedPath = path.resolve(changedPath);
        const relativePath = path.relative(tutorialsDirectory, resolvedPath);
        return (
          relativePath !== '' &&
          !relativePath.startsWith('..') &&
          !path.isAbsolute(relativePath) &&
          ['.py', '.md'].includes(path.extname(resolvedPath))
        );
      };
      let pendingReload = Promise.resolve();
      const reloadTutorials = (changedPath: string) => {
        const resolvedPath = path.resolve(changedPath);
        if (resolvedPath !== pyprojectPath && !isTutorialSource(resolvedPath)) {
          return;
        }
        pendingReload = pendingReload
          .catch(() => undefined)
          .then(async () => {
            await syncTutorialEntries(context);
            context.logger.info(`Reloaded tutorials after ${path.basename(resolvedPath)} changed`);
          });
        return pendingReload;
      };
      watcher.on('add', reloadTutorials);
      watcher.on('change', reloadTutorials);
      watcher.on('unlink', reloadTutorials);
    },
  };
}
