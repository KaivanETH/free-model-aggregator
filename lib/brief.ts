// The brief (the file at $MAKEFUN_PROMPT_FILE): three JSON documents in prose, each found by its `schema`.
//
//   free-model-aggregator-01/providers/v1  the provider directory (id, wire, auth, key, base path)
//   free-model-aggregator-01/ratings/v1    the quality table (rating, context, tools/vision/json, free)
//   free-model-aggregator-01/workload/v1   the manifest of the requests the client will send, in order
//
// Pure parsing: no clock, no randomness, no network.

import { readFileSync } from 'node:fs';

export type Json = Record<string, unknown>;
export const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);

export type Wire = 'openai' | 'anthropic' | 'gemini';
export type ClientApi = 'openai' | 'anthropic';
export type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

export const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
export const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
export const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

export interface ProviderEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  /** The base path (`/p/<id>`) or an absolute base URL: joined to the env's origin only to dial. */
  base: string;
  key_env: string;
  key: string;
}

export interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

export interface ManifestRow {
  id: string;
  api: ClientApi;
  stream: boolean;
  weight: number;
  requires: Requirement;
}

export interface Brief {
  providers: ProviderEntry[];
  ratings: RatingRow[];
  best: Map<string, number>;
  manifest: ManifestRow[];
  error: string | null;
}

const str = (v: unknown): string => (typeof v === 'string' ? v : '');
const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);

/** Every JSON object in `text` that carries a string `schema`, in the order they appear. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; ) {
    const end = objectEnd(text, start);
    if (end !== -1) {
      try {
        const value: unknown = JSON.parse(text.slice(start, end + 1));
        if (isRecord(value) && typeof value.schema === 'string') {
          out.push(value);
          start = text.indexOf('{', end + 1);
          continue;
        }
      } catch {
        // not a document on its own: look further
      }
    }
    start = text.indexOf('{', start + 1);
  }
  return out;
}

/** The index of the `}` that closes the object opening at `start`, or -1 (string contents skipped). */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

function providerOf(v: unknown): ProviderEntry | null {
  if (!isRecord(v)) return null;
  const id = str(v.id);
  const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (id === '' || wire === null) return null;
  const auth: AuthScheme = v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none' ? v.auth : 'bearer';
  const base = [v.path, v.base_path, v.base_url, v.base].map(str).find((s) => s !== '') ?? `/p/${id}`;
  return { id, name: str(v.name) || id, wire, auth, base, key_env: str(v.key_env), key: str(v.key) };
}

function ratingOf(v: unknown): RatingRow | null {
  if (!isRecord(v)) return null;
  const model = str(v.model);
  const provider = str(v.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    rating_milli: num(v.rating_milli) ?? 0,
    context_length: num(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    // The table lists free models; a row that says nothing about `free` is one of them.
    free: v.free !== false,
  };
}

export function requirementOf(v: unknown): Requirement {
  const r = isRecord(v) ? v : {};
  const out: Requirement = {};
  if (r.tools === true) out.tools = true;
  if (r.vision === true) out.vision = true;
  if (r.json === true) out.json = true;
  const ctx = num(r.min_context);
  if (ctx !== null) out.min_context = ctx;
  return out;
}

function manifestRowOf(v: unknown): ManifestRow | null {
  if (!isRecord(v)) return null;
  const id = str(v.id);
  if (id === '') return null;
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: num(v.weight) ?? 1,
    requires: requirementOf(v.requires),
  };
}

/** Parse the brief's text; tolerant of anything around the three documents. */
export function parseBrief(text: string): Brief {
  const docs = documentsIn(text);
  const dir = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers));
  const rat = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings));
  const wl = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest));
  const providers = dir === undefined ? [] : (dir.providers as unknown[]).flatMap((v) => providerOf(v) ?? []);
  const ratings = rat === undefined ? [] : (rat.ratings as unknown[]).flatMap((v) => ratingOf(v) ?? []);
  const best = new Map<string, number>();
  if (rat !== undefined && Array.isArray(rat.best)) {
    for (const b of rat.best) if (isRecord(b) && str(b.id) !== '' && num(b.rating_milli) !== null) best.set(str(b.id), num(b.rating_milli) as number);
  }
  const manifest = wl === undefined ? [] : (wl.manifest as unknown[]).flatMap((v) => manifestRowOf(v) ?? []);
  return { providers, ratings, best, manifest, error: dir === undefined ? `the brief holds no ${PROVIDERS_SCHEMA} document` : null };
}

export function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: [], best: new Map(), manifest: [], error: null };
  if (path === '') return { ...empty, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  return parseBrief(text);
}

/** The base path joined to the env's origin; an absolute base URL is used as it stands. */
export function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The path a catalog reports for a provider: `/p/<id>` from an absolute base URL, the path as it stands otherwise. */
export function catalogBase(base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) {
    try {
      return new URL(b).pathname.replace(/\/+$/, '') || '/';
    } catch {
      return b;
    }
  }
  return b.replace(/\/+$/, '');
}

/** The headers that carry a provider's key under its auth scheme (`none` sends none). */
export function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (auth === 'none' || key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  return { 'x-goog-api-key': key };
}
