// free-model-aggregator: an OpenAI- and Anthropic-compatible server that answers every request with the
// best free model that can serve it right now, failing over across the brief's ten free-tier providers.
//
//   node main.ts          read the brief ($MAKEFUN_PROMPT_FILE), probe each provider's catalog through the
//                         env ($MAKEFUN_ENV_URL), listen, register, serve, exit 0 once /work says done
//   node main.ts --list   print the catalog document (the register body's `catalog`, byte for byte), exit 0
//
// Routes: POST /v1/chat/completions, POST /v1/messages (JSON or SSE), GET /v1/models (OpenAI list for
// Authorization: Bearer, Anthropic list for x-api-key).
//
// Determinism: no clock, no randomness, one env request at a time (catalog probes in order, provider
// calls one after another, no /work poll while a request is being served), every choice a function of
// the brief, the requests and the providers' answers.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { authHeaders, isRecord, joinBase, readBrief, type ClientApi, type Json, type ManifestRow, type Requirement } from './lib/brief.ts';
import { build, freeIds, probe, type Built, type Probe } from './lib/catalog.ts';
import { Router, type Candidate } from './lib/router.ts';
import { anthropicModelList, bodyRequirements, errorBody, openAIModelList, parseUpstream, renderAnthropic, renderOpenAI, upstreamBody, wantsJson } from './lib/wire.ts';

const PROTOCOL_VERSION = 1;
const POLL_MS = 50;
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** The most provider calls one client request may spend before it is answered with an error. */
const MAX_ATTEMPTS = 16;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const log = (message: string): void => {
  process.stderr.write(`aggregator: ${message}\n`);
};

const brief = readBrief(PROMPT_FILE);
let built: Built = { catalog: { providers: [] }, models: [] };
let router = new Router([], []);
let arrivals = 0;
let busy = 0;
let seq = 0;

async function load(): Promise<void> {
  if (brief.error !== null && PROMPT_FILE !== '') log(brief.error);
  if (brief.providers.length === 0) return;
  const probes: Probe[] = [];
  if (ENV_URL !== '') {
    // One at a time, in the directory's order.
    for (const p of brief.providers) {
      const r = await probe(ENV_URL, p);
      if (!r.ok && r.error !== null) log(r.error);
      probes.push(r);
    }
  }
  built = build(brief, probes);
  router = new Router(brief.providers, built.models);
}

/** The manifest row of the request arriving now: the next one in order whose api and stream match. */
function manifestRowFor(api: ClientApi, stream: boolean): ManifestRow | null {
  const m = brief.manifest;
  for (let i = arrivals; i < m.length; i++) {
    if (m[i].api === api && m[i].stream === stream) {
      arrivals = i + 1;
      return m[i];
    }
  }
  return null;
}

function requirementsFor(api: ClientApi, body: Json): Requirement {
  const row = manifestRowFor(api, body.stream === true);
  const fromBody = bodyRequirements(api, body);
  const req: Requirement = { ...(row?.requires ?? {}) };
  if (fromBody.tools) req.tools = true;
  if (fromBody.vision) req.vision = true;
  if (fromBody.json) req.json = true;
  req.min_context = Math.max(req.min_context ?? 0, fromBody.tokens);
  return req;
}

async function readBody(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const b = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += b.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(b);
  }
  try {
    const v: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(v) ? v : null;
  } catch {
    return null;
  }
}

function send(res: ServerResponse, status: number, contentType: string, body: string): void {
  const headers: Record<string, string> = { 'content-type': contentType, 'content-length': String(Buffer.byteLength(body)) };
  if (contentType.startsWith('text/event-stream')) headers['cache-control'] = 'no-cache';
  res.writeHead(status, headers);
  res.end(body);
}

function sendError(res: ServerResponse, api: ClientApi, status: number, type: string, message: string): void {
  send(res, status, 'application/json', errorBody(api, status, type, message));
}

interface CallAnswer {
  status: number;
  contentType: string;
  text: string;
  message: string;
}

/** The refusal's message, read from whichever wire's error envelope it came in. */
function errorMessage(text: string): string {
  try {
    const v: unknown = JSON.parse(text);
    if (isRecord(v)) {
      const e = isRecord(v.error) ? v.error : v;
      return [e.message, e.code, e.status, e.type].filter((x) => typeof x === 'string').join(' ');
    }
  } catch {
    // not JSON
  }
  return text.slice(0, 300);
}

async function callProvider(c: Candidate, api: ClientApi, body: Json, headersIn: IncomingMessage['headers']): Promise<CallAnswer> {
  const p = c.provider;
  const base = joinBase(ENV_URL, p.base);
  const stream = body.stream === true;
  const payload = upstreamBody(api, p.wire, body, c.model.id);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(p.auth, p.key) };
  let url: string;
  if (p.wire === 'openai') url = `${base}/v1/chat/completions`;
  else if (p.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    const version = headersIn['anthropic-version'];
    headers['anthropic-version'] = typeof version === 'string' && version !== '' ? version : '2023-06-01';
    const beta = headersIn['anthropic-beta'];
    if (api === 'anthropic' && typeof beta === 'string' && beta !== '') headers['anthropic-beta'] = beta;
  } else {
    const model = encodeURIComponent(c.model.id);
    url = stream ? `${base}/v1beta/models/${model}:streamGenerateContent?alt=sse` : `${base}/v1beta/models/${model}:generateContent`;
  }
  if (stream) headers.accept = 'text/event-stream';
  try {
    const res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(payload) });
    const text = await res.text();
    const contentType = res.headers.get('content-type') ?? '';
    return { status: res.status, contentType, text, message: res.ok ? '' : errorMessage(text) };
  } catch (err) {
    return { status: 0, contentType: '', text: '', message: (err as Error).message };
  }
}

/** Serve one client request: rank, call, fail over, answer in the client's wire. */
async function serve(api: ClientApi, req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readBody(req);
  if (body === null) return sendError(res, api, 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(api, body);
  const jsonMode = wantsJson(api, body);
  const stream = body.stream === true;
  const asked = typeof body.model === 'string' ? body.model : '';
  const named = asked !== '' && asked !== 'auto' && router.models.some((m) => m.id === asked || `${m.provider}/${m.id}` === asked) ? asked : null;
  seq += 1;
  const tag = `req${seq}`;
  const skip = new Set<string>();
  const triedHere = new Map<string, number>();
  let lastStatus = 503;
  let lastMessage = 'no free model meets this request';
  for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
    let list = router.candidates(requires, api, jsonMode, skip, named);
    if (list.length === 0 && named !== null) list = router.candidates(requires, api, jsonMode, skip, null);
    const c = list[0];
    if (c === undefined) break;
    const answer = await callProvider(c, api, body, req.headers);
    if (answer.status >= 200 && answer.status < 300) {
      router.record(c.provider.id, c.model.id, answer.status, '');
      const result = parseUpstream(c.provider.wire, stream, answer.contentType, answer.text);
      if (result === null) {
        // The provider answered 2xx with no answer of its wire: report it rather than call a second provider.
        log(`${tag}: ${c.provider.id}/${c.model.id} answered 2xx that is no ${c.provider.wire} answer`);
        return sendError(res, api, 502, 'api_error', `${c.provider.id} answered an unreadable body`);
      }
      const rendered = api === 'openai' ? renderOpenAI(result, stream, c.model.id, `chatcmpl-${tag}`) : renderAnthropic(result, stream, c.model.id, `msg_${tag}`);
      log(`${tag}: served by ${c.provider.id}/${c.model.id} (${c.model.rating}) after ${attempt} refusal(s)`);
      return send(res, 200, rendered.contentType, rendered.body);
    }
    const outcome = router.record(c.provider.id, c.model.id, answer.status, answer.message);
    log(`${tag}: ${c.provider.id}/${c.model.id} refused ${answer.status} (${outcome}): ${answer.message}`);
    lastStatus = answer.status === 0 ? 502 : answer.status;
    lastMessage = answer.message;
    const n = (triedHere.get(c.provider.id) ?? 0) + 1;
    triedHere.set(c.provider.id, n);
    if (outcome === 'skip' || (outcome === 'transient' && n >= 2)) skip.add(c.provider.id);
  }
  const status = lastStatus === 402 || lastStatus === 429 ? 503 : lastStatus >= 500 ? lastStatus : 503;
  sendError(res, api, status, api === 'anthropic' ? 'overloaded_error' : 'server_error', `no provider could serve this request: ${lastMessage}`);
}

function models(req: IncomingMessage, res: ServerResponse): void {
  const list = freeIds(built.catalog);
  const anthropic = typeof req.headers['x-api-key'] === 'string' && typeof req.headers.authorization !== 'string';
  send(res, 200, 'application/json', JSON.stringify(anthropic ? anthropicModelList(list) : openAIModelList(list)));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const path = new URL(req.url ?? '/', 'http://entry').pathname.replace(/\/+$/, '');
  const method = req.method ?? 'GET';
  if (method === 'GET' && path === '/v1/models') return models(req, res);
  if (method === 'GET' && path.startsWith('/v1/models/')) {
    const id = decodeURIComponent(path.slice('/v1/models/'.length));
    const m = freeIds(built.catalog).find((x) => x.id === id);
    if (m === undefined) return sendError(res, typeof req.headers['x-api-key'] === 'string' ? 'anthropic' : 'openai', 404, 'not_found_error', `no model ${id}`);
    const anthropic = typeof req.headers['x-api-key'] === 'string' && typeof req.headers.authorization !== 'string';
    const listed = anthropic ? anthropicModelList([m]) : openAIModelList([m]);
    return send(res, 200, 'application/json', JSON.stringify((listed.data as unknown[])[0]));
  }
  if (method === 'POST' && path === '/v1/chat/completions') return serve('openai', req, res);
  if (method === 'POST' && path === '/v1/messages') return serve('anthropic', req, res);
  if (method === 'GET' && (path === '' || path === '/health')) return send(res, 200, 'application/json', '{"ok":true}');
  sendError(res, path.startsWith('/v1/messages') ? 'anthropic' : 'openai', 404, 'not_found_error', `no route ${method} ${path}`);
}

function listen(server: Server): Promise<number> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => {
      const a = server.address();
      if (a === null || typeof a === 'string') reject(new Error('the server has no port'));
      else resolve(a.port);
    });
  });
}

async function register(port: number, catalogText: string): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: `{"protocol_version":${PROTOCOL_VERSION},"port":${port},"catalog":${catalogText}}`,
  });
  const text = await res.text();
  if (!res.ok) throw new Error(`POST /register answered HTTP ${res.status}: ${text.slice(0, 300)}`);
  log(`registered on port ${port}: ${text.trim()}`);
}

/** Poll GET /work whenever no client request is being served; exit 0 the moment it says done. */
function poll(server: Server): void {
  const tick = async (): Promise<void> => {
    if (busy === 0) {
      try {
        const res = await fetch(`${ENV_URL}/work`);
        const v: unknown = await res.json();
        if (isRecord(v) && v.done === true) {
          server.close();
          process.exit(0);
        }
      } catch {
        // the env is not answering yet: try again
      }
    }
    setTimeout(() => void tick(), POLL_MS);
  };
  void tick();
}

async function main(): Promise<void> {
  await load();
  const catalogText = JSON.stringify(built.catalog);
  if (process.argv.includes('--list')) {
    await new Promise<void>((resolve) => process.stdout.write(`${catalogText}\n`, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    busy += 1;
    let released = false;
    const release = (): void => {
      if (!released) {
        released = true;
        busy -= 1;
      }
    };
    res.on('close', release);
    handle(req, res)
      .catch((err: unknown) => {
        log((err as Error).stack ?? String(err));
        if (!res.headersSent) sendError(res, (req.url ?? '').startsWith('/v1/messages') ? 'anthropic' : 'openai', 500, 'api_error', 'the server failed to answer');
      })
      .finally(release);
  });
  server.keepAliveTimeout = 1000;
  const port = await listen(server);
  log(`listening on 0.0.0.0:${port}: ${built.catalog.providers.length} providers, ${built.models.length} free models`);
  await register(port, catalogText);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

main().catch((err: unknown) => {
  log((err as Error).stack ?? String(err));
  process.exit(1);
});
