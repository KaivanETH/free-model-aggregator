// The catalog: every provider of the directory, with the models its own catalog route lists.
//
// Each provider's listing is read in its own wire and auth scheme, one request at a time and in the
// directory's order (no parallel probe, so the env sees the same sequence on every pass):
//
//   openai     GET <base>/v1/models
//   anthropic  GET <base>/v1/models      (+ anthropic-version)
//   gemini     GET <base>/v1beta/models
//
// The listing is the truth for the `free` flag (the brief's quality table is the cross-check and the
// fallback when a listing cannot be read). The routable models carry the table's rating and
// capabilities; a model the table does not rate is routable only as a last resort.

import { authHeaders, catalogBase, isRecord, joinBase, type Brief, type Json, type ProviderEntry, type RatingRow } from './brief.ts';

export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: ProviderEntry['wire'];
  models: CatalogModel[];
}

export interface Catalog {
  providers: CatalogProvider[];
}

/** One model the router may send traffic to. */
export interface ModelInfo {
  id: string;
  provider: string;
  free: boolean;
  rating: number;
  rated: boolean;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
}

interface Listed {
  id: string;
  free: boolean | null;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

const str = (v: unknown): string => (typeof v === 'string' ? v : '');
const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const bool = (v: unknown): boolean | null => (typeof v === 'boolean' ? v : null);

/** A listing's price fields: true when every price is zero, null when it names none. */
function zeroPriced(v: Json): boolean | null {
  const pricing = isRecord(v.pricing) ? v.pricing : null;
  if (pricing === null) return null;
  const values = Object.keys(pricing)
    .sort()
    .flatMap((k) => {
      const x = pricing[k];
      if (typeof x === 'number') return [x];
      if (typeof x === 'string' && x.trim() !== '') return [Number(x)];
      return [];
    });
  if (values.length === 0) return null;
  return values.every((n) => Number.isFinite(n) && n === 0);
}

function listedOf(wire: ProviderEntry['wire'], v: unknown): Listed | null {
  if (!isRecord(v)) return null;
  const id = wire === 'gemini' ? (str(v.name).replace(/^models\//, '') || str(v.id)) : str(v.id);
  if (id === '') return null;
  let free = bool(v.free) ?? zeroPriced(v);
  if (free === null && id.endsWith(':free')) free = true;
  return {
    id,
    free,
    context_length: num(v.context_length) ?? num(v.context_window) ?? num(v.inputTokenLimit) ?? num(v.max_input_tokens),
    tools: bool(v.tools),
    vision: bool(v.vision),
    json: bool(v.json),
  };
}

export interface Probe {
  provider: ProviderEntry;
  ok: boolean;
  listed: Listed[];
  error: string | null;
}

/** Read one provider's catalog route in its own wire. */
export async function probe(envUrl: string, provider: ProviderEntry): Promise<Probe> {
  const base = joinBase(envUrl, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  try {
    const res = await fetch(url, { headers });
    const text = await res.text();
    if (!res.ok) return { provider, ok: false, listed: [], error: `GET ${url} answered HTTP ${res.status}` };
    const body: unknown = JSON.parse(text);
    const list = isRecord(body) ? (provider.wire === 'gemini' ? body.models : body.data) : null;
    if (!Array.isArray(list)) return { provider, ok: false, listed: [], error: `GET ${url} answered no model list` };
    const seen = new Set<string>();
    const listed: Listed[] = [];
    for (const raw of list) {
      const m = listedOf(provider.wire, raw);
      if (m === null || seen.has(m.id)) continue;
      seen.add(m.id);
      listed.push(m);
    }
    return { provider, ok: true, listed, error: null };
  } catch (err) {
    return { provider, ok: false, listed: [], error: `GET ${url} failed: ${(err as Error).message}` };
  }
}

const byId = <T extends { id: string }>(a: T, b: T): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

function rowOf(ratings: readonly RatingRow[], provider: string, model: string): RatingRow | null {
  return ratings.find((r) => r.provider === provider && r.model === model) ?? null;
}

export interface Built {
  catalog: Catalog;
  models: ModelInfo[];
}

/**
 * The catalog document and the routable models. A provider whose listing could not be read is reported
 * from the quality table (its free models), so the directory still shows every provider.
 */
export function build(brief: Brief, probes: readonly Probe[]): Built {
  const providers: CatalogProvider[] = [];
  const models: ModelInfo[] = [];
  for (const entry of brief.providers) {
    const p = probes.find((x) => x.provider.id === entry.id);
    const rows = brief.ratings.filter((r) => r.provider === entry.id);
    let listed: Listed[];
    if (p !== undefined && p.ok) listed = p.listed;
    else listed = rows.map((r) => ({ id: r.model, free: r.free, context_length: r.context_length, tools: r.tools, vision: r.vision, json: r.json }));
    const catalogModels: CatalogModel[] = [];
    for (const m of [...listed].sort(byId)) {
      const row = rowOf(brief.ratings, entry.id, m.id);
      // The listing's flag is the truth; with none, the table's row (it lists free models) decides.
      const free = m.free ?? (row !== null ? row.free : false);
      const context = m.context_length ?? row?.context_length ?? null;
      catalogModels.push({ id: m.id, free, context_length: context });
      if (!free) continue;
      models.push({
        id: m.id,
        provider: entry.id,
        free,
        rating: row?.rating_milli ?? 0,
        rated: row !== null,
        context_length: row?.context_length ?? context,
        tools: row?.tools ?? m.tools ?? false,
        vision: row?.vision ?? m.vision ?? false,
        json: row?.json ?? m.json ?? false,
      });
    }
    providers.push({ id: entry.id, base_url: catalogBase(entry.base), key_env: entry.key_env, wire: entry.wire, models: catalogModels });
  }
  return { catalog: { providers }, models };
}

/** The free model ids the catalog lists, deduplicated, in catalog order: what GET /v1/models answers. */
export function freeIds(catalog: Catalog): { id: string; provider: string }[] {
  const seen = new Set<string>();
  const out: { id: string; provider: string }[] = [];
  for (const p of catalog.providers) {
    for (const m of p.models) {
      if (!m.free || seen.has(m.id)) continue;
      seen.add(m.id);
      out.push({ id: m.id, provider: p.id });
    }
  }
  return out;
}
