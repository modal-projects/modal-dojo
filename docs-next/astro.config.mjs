// @ts-check
import { readdirSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import rehypeKatex from 'rehype-katex';
import remarkMath from 'remark-math';
import { rehypeTableWrapper } from './rehype-table-wrapper.mjs';
import { flattenDocId } from './src/lib/docs-sections.ts';
import { parseTutorialMetadata } from './src/lib/tutorial-docs-loader.ts';
import { discoverTutorialEntries } from './src/lib/tutorial-slugs.ts';
import referenceSidebar from './src/generated/reference-sidebar.json';

const configDirectory = path.dirname(fileURLToPath(import.meta.url));
const guideSectionOrder = ['start', 'tools', 'migration'];

/** @param {string} section */
function guideSectionRank(section) {
  const index = guideSectionOrder.indexOf(section);
  return index === -1 ? guideSectionOrder.length : index;
}

function remarkStripPageTitle() {
  return (/** @type {{ children: Array<{ type: string, depth?: number }> }} */ tree) => {
    const index = tree.children.findIndex(
      (node) => node.type === 'heading' && node.depth === 1,
    );
    if (index !== -1) {
      tree.children.splice(index, 1);
    }
  };
}

async function loadTutorialPages() {
  const tutorialsDirectory = path.resolve(configDirectory, '../tutorials');
  const tutorials = (await discoverTutorialEntries(tutorialsDirectory)).map((entry) => {
    const source = readFileSync(entry.path, 'utf8');
    const { order } = parseTutorialMetadata(source, entry.path);
    return { order, slug: entry.slug };
  });
  tutorials.sort((left, right) => left.order - right.order || left.slug.localeCompare(right.slug));
  if (!tutorials[0]) {
    throw new Error('No tutorial pages found');
  }
  return tutorials;
}

function firstGuidePath() {
  const guidesDirectory = path.resolve(configDirectory, 'src/content/docs/guides');
  const guides = readdirSync(guidesDirectory, { recursive: true, encoding: 'utf8' })
    .filter((fileName) => fileName.endsWith('.md'))
    .map((fileName) => {
      const guidePath = path.join(guidesDirectory, fileName);
      const source = readFileSync(guidePath, 'utf8');
      const frontmatterEnd = source.indexOf('\n---', 4);
      const frontmatter = frontmatterEnd === -1 ? '' : source.slice(4, frontmatterEnd);
      const orderMatch = frontmatter.match(/^order:\s*(\d+)\s*$/m);
      if (!orderMatch) {
        throw new Error(`${guidePath} is missing order`);
      }
      const relative = fileName.replaceAll(path.sep, '/');
      const slug = flattenDocId(`guides/${relative}`);
      const section = relative.split('/')[0] ?? '';
      return { order: Number(orderMatch[1]), section, slug };
    })
    .sort(
      (left, right) =>
        guideSectionRank(left.section) - guideSectionRank(right.section) ||
        left.section.localeCompare(right.section) ||
        left.order - right.order ||
        left.slug.localeCompare(right.slug)
    );
  if (!guides[0]) {
    throw new Error('No guide pages found');
  }
  return `/${guides[0].slug}`;
}

function nestedDocRedirects() {
  const docsRoot = path.resolve(configDirectory, 'src/content/docs');
  /** @type {Record<string, string>} */
  const redirects = {};
  for (const fileName of readdirSync(docsRoot, { recursive: true, encoding: 'utf8' })) {
    if (typeof fileName !== 'string') continue;
    if (!fileName.endsWith('.md') && !fileName.endsWith('.mdx')) continue;
    const entry = fileName.replaceAll(path.sep, '/');
    const flat = flattenDocId(entry);
    const nested = entry.replace(/\.[^./]+$/, '').replace(/\/index$/, '');
    if (nested === flat || nested === 'index') continue;
    redirects[`/${nested}`] = `/${flat}`;
  }
  return redirects;
}

const tutorialPages = await loadTutorialPages();
const firstTutorial = `/tutorials/${tutorialPages[0].slug}`;
const firstGuide = firstGuidePath();

export default defineConfig({
  trailingSlash: 'never',
  redirects: {
    ...nestedDocRedirects(),
    '/guides/tools/agent-driven-training': '/guides/agent',
    '/guides/agent-driven-training': '/guides/agent',
    '/reference': '/reference/sdk',
    '/reference/core/modelconfig': '/reference/modelconfig',
    '/reference/core/hfmodelconfiguration': '/reference/hfmodelconfiguration',
    '/reference/core/modelarchitecture': '/reference/modelarchitecture',
    '/reference/core/datasetconfig': '/reference/datasetconfig',
    '/reference/core/huggingfacedataset': '/reference/huggingfacedataset',
    '/reference/core/harbordataset': '/reference/harbordataset',
    '/reference/core/trainresult': '/reference/trainingrun',
    '/reference/trainresult': '/reference/trainingrun',
    '/reference/core/metricconfig': '/reference/metricconfig',
    '/reference/core/trackioconfig': '/reference/trackioconfig',
    '/reference/core/wandbconfig': '/reference/wandbconfig',
    '/reference/core/modalraycluster': '/reference/modalraycluster',
    '/reference/training/qwen3_5_4b_miles_recipe': '/reference/qwen3_5_4b_miles_recipe',
    '/reference/training/moonlight_16b_a3b_recipe': '/reference/moonlight_16b_a3b_recipe',
    '/reference/training/gemma4_26b_a4b_recipe': '/reference/gemma4_26b_a4b_recipe',
    '/reference/training/inkling_small_recipe': '/reference/inkling_small_recipe',
    '/reference/training/inkling_small_lora_recipe': '/reference/inkling_small_lora_recipe',
    '/reference/training/qwen3_6_35b_recipe': '/reference/qwen3_6_35b_recipe',
    '/reference/training/qwen3_6_27b_recipe': '/reference/qwen3_6_27b_recipe',
    '/reference/training/qwen3_8_27b_recipe': '/reference/qwen3_8_27b_recipe',
    '/reference/training/qwen3_5_0_8b_recipe': '/reference/qwen3_5_0_8b_recipe',
    '/reference/training/qwen3_5_2b_recipe': '/reference/qwen3_5_2b_recipe',
    '/reference/training/qwen3_5_4b_recipe': '/reference/qwen3_5_4b_recipe',
    '/reference/training/qwen3_5_9b_recipe': '/reference/qwen3_5_9b_recipe',
    '/reference/models/parsedresponse': '/reference/parsedresponse',
    '/reference/models/parse_qwen3_response':
      '/reference/modelconfig#parse_response',
    '/reference/parse_qwen3_response':
      '/reference/modelconfig#parse_response',
    '/guides': firstGuide,
    '/support': '/',
    '/tutorials': firstTutorial,
    '/tutorials/multinode/002_glm_4_7': '/tutorials/multinode',
    '/tutorials/rl': firstTutorial,
    '/tutorials/rl/000_rl_basics': '/tutorials/rl_basics',
    '/tutorials/rl/001_sandboxes': '/tutorials/paint_flowers',
    '/tutorials/sandboxes': '/tutorials/paint_flowers',
    '/tutorials/rl/003_on_policy_distillation': '/tutorials/on_policy_distillation',
    '/tutorials/rl/005_dapo': '/tutorials/dapo',
    '/tutorials/rl/006_audio_asr': '/tutorials/audio_asr',
    '/tutorials/rl/007_param_sweep': '/tutorials/param_sweep',
    '/tutorials/rl/008_computer_use': '/tutorials/computer_use',
    '/tutorials/rl/009_cross_tokenizer_distillation':
      '/tutorials/cross_tok_distill',
    '/tutorials/cross_tokenizer_distillation': '/tutorials/cross_tok_distill',
    '/tutorials/tools/000_observability_dashboard':
      '/guides/dashboard',
    '/guides/observability-dashboard': '/guides/dashboard',
    '/guides/tools/observability-dashboard': '/guides/dashboard',
    '/guides/wandb-integration': '/guides/metric',
    '/guides/tools/wandb-integration': '/guides/metric',
    '/tutorials/tools/001_wandb_integration': '/guides/metric',
    '/guides/trackio-integration': '/guides/metric',
    '/guides/tools/trackio-integration': '/guides/metric',
    '/guides/metrics-integration': '/guides/metric',
    '/guides/tools/metrics-integration': '/guides/metric',
  },
  markdown: {
    remarkPlugins: [remarkMath, remarkStripPageTitle],
    rehypePlugins: [rehypeKatex, rehypeTableWrapper],
  },
  site: 'https://gym.modal.dev',
  integrations: [
    starlight({
      title: 'Modal Dojo',
      favicon: '/modal-logo.svg',
      description:
        'Open-source Python SDK for GRPO and RL post-training of LLMs on Modal GPU clusters — tutorials, API reference, and runnable examples.',
      tagline:
        'GRPO, PPO, custom reward and generate functions — runnable on Modal in minutes.',
      head: [
        {
          tag: 'meta',
          attrs: { property: 'og:image', content: 'https://gym.modal.dev/og-image.png' },
        },
        {
          tag: 'meta',
          attrs: { property: 'og:image:width', content: '2400' },
        },
        {
          tag: 'meta',
          attrs: { property: 'og:image:height', content: '1260' },
        },
        {
          tag: 'meta',
          attrs: { name: 'twitter:card', content: 'summary_large_image' },
        },
        {
          tag: 'meta',
          attrs: { name: 'twitter:image', content: 'https://gym.modal.dev/og-image.png' },
        },
        {
          tag: 'meta',
          attrs: { name: 'twitter:site', content: '@modal_labs' },
        },
        {
          tag: 'script',
          attrs: { type: 'application/ld+json' },
          content: JSON.stringify({
            '@context': 'https://schema.org',
            '@type': 'WebSite',
            name: 'Modal Dojo',
            url: 'https://gym.modal.dev',
            description:
              'Open-source Python SDK for GRPO and RL post-training of LLMs on Modal GPU clusters.',
            publisher: {
              '@type': 'Organization',
              name: 'Modal Labs',
              url: 'https://modal.com',
            },
          }),
        },
      ],
      social: [
        {
          icon: 'github',
          label: 'GitHub',
          href: 'https://github.com/modal-projects/modal-dojo',
        },
      ],
      // Code blocks mirror the main Modal docs site (modal.com/docs): shiki's
      // dark-plus theme, Fira Mono 14px/20px, and a frameless container with a
      // translucent white surface over the page background.
      expressiveCode: {
        themes: ['dark-plus'],
        useStarlightUiThemeColors: false,
        defaultProps: { frame: 'none' },
        // The code surface is translucent white, which Expressive Code's
        // contrast pass reads as a light background and compensates for by
        // darkening every token color. Off, so dark-plus renders as-is.
        minSyntaxHighlightingColorContrast: 0,
        styleOverrides: {
          borderColor: '#2f2f2f',
          borderRadius: '0.25rem',
          borderWidth: '1px',
          codeBackground: 'rgba(255, 255, 255, 0.06)',
          codeForeground: '#d4d4d4',
          codeFontFamily: 'var(--sl-font-mono)',
          codeFontSize: '0.875rem',
          codeLineHeight: '1.4286',
          codePaddingBlock: '0.875rem',
          codePaddingInline: '0.875rem',
          codeSelectionBackground: '#264f78',
          scrollbarThumbColor: '#ffffff17',
          scrollbarThumbHoverColor: '#ffffff40',
          gutterBorderColor: '#2f2f2f',
          gutterForeground: '#747474',
          gutterHighlightForeground: '#e8e8e8',
          frames: {
            frameBoxShadowCssValue: 'none',
            editorBackground: 'rgba(255, 255, 255, 0.06)',
            terminalBackground: 'rgba(255, 255, 255, 0.06)',
            inlineButtonBackground: '#222222',
            inlineButtonBackgroundIdleOpacity: '1',
            inlineButtonBackgroundHoverOrFocusOpacity: '1',
            inlineButtonBackgroundActiveOpacity: '1',
            inlineButtonBorder: '#464646',
            inlineButtonBorderOpacity: '1',
            inlineButtonForeground: '#d1d1d1',
            tooltipSuccessBackground: '#222222',
            tooltipSuccessForeground: '#d1d1d1',
          },
        },
      },
      customCss: ['./src/styles/custom.css'],
      components: {
        Header: './src/components/Header.astro',
        MarkdownContent: './src/components/MarkdownContent.astro',
        Search: './src/components/Search.astro',
        Sidebar: './src/components/Sidebar.astro',
        PageSidebar: './src/components/PageSidebar.astro',
        PageTitle: './src/components/PageTitle.astro',
        TableOfContents: './src/components/TableOfContents.astro',
      },
      sidebar: [
        {
          label: 'Guides',
          items: [
            {
              label: 'Start here',
              autogenerate: { directory: 'guides/start' },
            },
            {
              label: 'Tools',
              autogenerate: { directory: 'guides/tools' },
            },
          ],
        },
        {
          label: 'Tutorials',
          items: [
            {
              label: 'Featured',
              items: tutorialPages.map(({ slug }) => ({ slug: `tutorials/${slug}` })),
            },
          ],
        },
        {
          label: 'SDK',
          items: referenceSidebar.sdk,
        },
        {
          label: 'CLI',
          items: referenceSidebar.cli,
        },
      ],
      lastUpdated: false,
      pagination: false,
      disable404Route: true,
    }),
  ],
});
