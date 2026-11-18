export function tagOptions(counts, activeTags) {
  const options = new Map();
  function add(key, value, count = 0) {
    const id = JSON.stringify([key, value]);
    options.set(id, {
      id, key, value, label: `${key}:${value}`, count,
      selected: Object.hasOwn(activeTags, key) && activeTags[key].includes(value),
    });
  }
  for (const [key, values] of Object.entries(counts)) {
    for (const [value, count] of Object.entries(values)) add(key, value, count);
  }
  for (const [key, values] of Object.entries(activeTags)) {
    for (const value of values) {
      if (!options.has(JSON.stringify([key, value]))) add(key, value);
    }
  }
  return [...options.values()].sort((a, b) => a.label.localeCompare(b.label));
}

export function newTagOption(query, options) {
  const colon = query.indexOf(":");
  if (colon <= 0 || colon === query.length - 1) return null;
  const key = query.slice(0, colon);
  const value = query.slice(colon + 1);
  if (options.some((option) => option.key === key && option.value === value)) return null;
  return { id: JSON.stringify([key, value]), key, value, label: query };
}
