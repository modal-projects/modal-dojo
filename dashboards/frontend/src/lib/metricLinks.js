const PROVIDER_LABELS = { wandb: "W&B", trackio: "Trackio" };

export function metricProviderLabel(provider) {
  return PROVIDER_LABELS[String(provider || "").toLowerCase()] || "Metric";
}

export function metricLinkProviderLabel(link, fallbackProvider) {
  const provider = metricProviderLabel(link?.provider || fallbackProvider);
  const label = String(link?.label || "").trim();
  if (!label || label === "Metric") return provider;
  if (label.startsWith("Metric")) return provider + label.slice("Metric".length);
  return label;
}

export function metricLinkLabel(label) {
  const text = String(label || "").trim();
  if (!text || text === "Open in W&B" || text === "W&B") return "Metric";
  return text.replace(/\bW&B\b/g, "Metric").replace(/\bwandb\b/gi, "Metric");
}

export function normalizeMetricLinks(links) {
  return Array.isArray(links)
    ? links.map((link) => ({ ...link, label: metricLinkLabel(link?.label) }))
    : [];
}
