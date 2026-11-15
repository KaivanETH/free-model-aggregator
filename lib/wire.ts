// The wires: client request -> provider request, provider answer -> one normalized Result, Result -> client answer.
//
// Client surfaces: OpenAI Chat Completions (POST /v1/chat/completions) and Anthropic Messages
// (POST /v1/messages), each JSON or SSE. Provider wires: openai, anthropic and gemini
// (generateContent / streamGenerateContent?alt=sse).
//
// A request is passed through as faithfully as its target wire can carry it: the system prompt, the
// user text (text parts kept apart, images kept as the same bytes or the same URL), the tool names and
// schemas, the stop sequences, the output cap and JSON mode. Same-wire traffic is forwarded untouched but
// for the model id (and OpenAI's stream_options.include_usage, so the usage arrives).
//
// Everything here is a pure function of its arguments: no clock and no randomness.

import { isRecord, type ClientApi, type Json, type Wire } from './brief.ts';

export type Stop = 'stop' | 'length' | 'tool_calls' | 'content_filter';

export interface ToolCall {
  id: string;
  name: string;
  arguments: string;
}

export interface Result {
  id: string | null;
  created: number | null;
  model: string | null;
  text: string;
  /** The text as the provider streamed it (one piece for a JSON answer). */
  pieces: string[];
  toolCalls: ToolCall[];
  stop: Stop;
  usage: { input: number; output: number };
}

const FIXED_CREATED = 1767225600;

const str = (v: unknown): string => (typeof v === 'string' ? v : '');
const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const int0 = (v: unknown): number => num(v) ?? 0;

function stringList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
}

/** The text of a string, or of a block/part array's text items joined by newlines (how the wires read it). */
export function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const b of content) if (isRecord(b) && b.type === 'text' && typeof b.text === 'string') parts.push(b.text);
  return parts.join('\n');
}

interface DataUrl {
  mime: string;
  data: string;
}

/** A base64 data URL's media type and payload; null for any other URL. */
function dataUrl(url: string): DataUrl | null {
  const m = /^data:([^;,]*)((?:;[^;,]*)*);base64,(.*)$/s.exec(url);
  if (m === null) return null;
  return { mime: m[1] !== '' ? m[1] : 'application/octet-stream', data: m[3] };
}

function guessMime(url: string): string {
  const u = url.toLowerCase().split('?')[0];
  if (u.endsWith('.png')) return 'image/png';
  if (u.endsWith('.jpg') || u.endsWith('.jpeg')) return 'image/jpeg';
  if (u.endsWith('.gif')) return 'image/gif';
  if (u.endsWith('.webp')) return 'image/webp';
  return 'image/jpeg';
}

function imageUrlOf(part: Json): string {
  if (typeof part.image_url === 'string') return part.image_url;
  if (isRecord(part.image_url)) return str(part.image_url.url);
  return '';
}

function parseArgs(v: unknown): Json {
  if (isRecord(v)) return v;
  if (typeof v === 'string' && v.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(v);
      if (isRecord(parsed)) return parsed;
    } catch {
      // not JSON: an empty input
    }
  }
  return {};
}

function jsonModeOf(body: Json): 'json_object' | 'json_schema' | null {
  const f = isRecord(body.response_format) ? body.response_format.type : undefined;
  return f === 'json_object' || f === 'json_schema' ? f : null;
}

/** Whether a client request asks for JSON mode on the OpenAI surface (only it can carry it). */
export function wantsJson(api: ClientApi, body: Json): boolean {
  return api === 'openai' && jsonModeOf(body) !== null;
}

function systemTextOfOpenAI(messages: Json[]): string {
  const system = messages.filter((m) => m.role === 'system').map((m) => blocksText(m.content)).filter((t) => t !== '');
  if (system.length > 0) return system.join('\n');
  return messages.filter((m) => m.role === 'developer').map((m) => blocksText(m.content)).filter((t) => t !== '').join('\n');
}

const messagesOf = (body: Json): Json[] => (Array.isArray(body.messages) ? body.messages.filter(isRecord) : []);

// ---------------------------------------------------------------- client OpenAI -> provider requests

/** OpenAI -> OpenAI: the body untouched but for the model (and include_usage on a stream). */
export function openAIPassThrough(body: Json, model: string): Json {
  const out: Json = { ...body, model };
  if (body.stream === true) out.stream_options = { ...(isRecord(body.stream_options) ? body.stream_options : {}), include_usage: true };
  return out;
}

/** OpenAI Chat Completions request -> Anthropic Messages request. */
export function anthropicFromOpenAI(body: Json, model: string): Json {
  const messages = messagesOf(body);
  const out: Json = { model };
  const msgs: Json[] = [];
  for (const m of messages) {
    if (m.role === 'system' || m.role === 'developer') continue;
    if (m.role === 'tool') {
      const block = { type: 'tool_result', tool_use_id: str(m.tool_call_id), content: typeof m.content === 'string' ? m.content : blocksText(m.content) };
      const last = msgs[msgs.length - 1];
      if (last !== undefined && last.role === 'user' && Array.isArray(last.content) && last.content.every((b) => isRecord(b) && b.type === 'tool_result')) {
        (last.content as Json[]).push(block);
      } else msgs.push({ role: 'user', content: [block] });
      continue;
    }
    if (m.role === 'assistant') {
      const blocks: Json[] = [];
      const text = blocksText(m.content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(m.tool_calls) ? m.tool_calls.filter(isRecord) : [];
      for (const c of calls) {
        const fn = isRecord(c.function) ? c.function : {};
        blocks.push({ type: 'tool_use', id: str(c.id), name: str(fn.name), input: parseArgs(fn.arguments) });
      }
      if (blocks.length > 0) msgs.push({ role: 'assistant', content: blocks });
      continue;
    }
    // user
    if (typeof m.content === 'string') {
      msgs.push({ role: 'user', content: m.content });
      continue;
    }
    const blocks: Json[] = [];
    for (const p of Array.isArray(m.content) ? m.content.filter(isRecord) : []) {
      if (p.type === 'text' && typeof p.text === 'string') blocks.push({ type: 'text', text: p.text });
      else if (p.type === 'image_url') {
        const url = imageUrlOf(p);
        const d = dataUrl(url);
        if (d !== null) blocks.push({ type: 'image', source: { type: 'base64', media_type: d.mime, data: d.data } });
        else if (url !== '') blocks.push({ type: 'image', source: { type: 'url', url } });
      }
    }
    msgs.push({ role: 'user', content: blocks });
  }
  const system = systemTextOfOpenAI(messages);
  if (system !== '') out.system = system;
  out.messages = msgs;
  const maxTokens = num(body.max_tokens) ?? num(body.max_completion_tokens);
  if (maxTokens !== null) out.max_tokens = maxTokens;
  const stops = stringList(body.stop);
  if (stops.length > 0) out.stop_sequences = stops;
  if (num(body.temperature) !== null) out.temperature = Math.min(1, Math.max(0, num(body.temperature) as number));
  if (num(body.top_p) !== null) out.top_p = body.top_p;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.filter(isRecord).flatMap((t) => {
      const fn = isRecord(t.function) ? t.function : null;
      if (fn === null || typeof fn.name !== 'string') return [];
      const tool: Json = { name: fn.name, input_schema: isRecord(fn.parameters) ? fn.parameters : { type: 'object', properties: {} } };
      if (typeof fn.description === 'string') tool.description = fn.description;
      return [tool];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  let tc: Json | null = null;
  if (choice === 'auto') tc = { type: 'auto' };
  else if (choice === 'required') tc = { type: 'any' };
  else if (choice === 'none') tc = { type: 'none' };
  else if (isRecord(choice) && isRecord(choice.function) && typeof choice.function.name === 'string') tc = { type: 'tool', name: choice.function.name };
  if (tc !== null && out.tools !== undefined) {
    if (body.parallel_tool_calls === false && tc.type !== 'none') tc.disable_parallel_tool_use = true;
    out.tool_choice = tc;
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** Gemini's schema dialect: JSON Schema without the keywords it refuses. */
function geminiSchema(v: unknown): unknown {
  if (Array.isArray(v)) return v.map(geminiSchema);
  if (!isRecord(v)) return v;
  const out: Json = {};
  for (const k of Object.keys(v)) {
    if (k === 'additionalProperties' || k === '$schema' || k === '$id' || k === 'strict') continue;
    out[k] = geminiSchema(v[k]);
  }
  return out;
}

/** OpenAI Chat Completions request -> Gemini generateContent body (the model goes in the path). */
export function geminiFromOpenAI(body: Json): Json {
  const messages = messagesOf(body);
  const out: Json = {};
  const contents: Json[] = [];
  const toolNames = new Map<string, string>();
  for (const m of messages) {
    if (m.role === 'system' || m.role === 'developer') continue;
    if (m.role === 'assistant') {
      const parts: Json[] = [];
      const text = blocksText(m.content);
      if (text !== '') parts.push({ text });
      for (const c of Array.isArray(m.tool_calls) ? m.tool_calls.filter(isRecord) : []) {
        const fn = isRecord(c.function) ? c.function : {};
        toolNames.set(str(c.id), str(fn.name));
        parts.push({ functionCall: { name: str(fn.name), args: parseArgs(fn.arguments) } });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (m.role === 'tool') {
      const raw = typeof m.content === 'string' ? m.content : blocksText(m.content);
      let response: Json;
      try {
        const parsed: unknown = JSON.parse(raw);
        response = isRecord(parsed) ? parsed : { content: parsed };
      } catch {
        response = { content: raw };
      }
      const part = { functionResponse: { name: toolNames.get(str(m.tool_call_id)) ?? str(m.name), response } };
      const last = contents[contents.length - 1];
      if (last !== undefined && last.role === 'user' && Array.isArray(last.parts) && last.parts.every((p) => isRecord(p) && isRecord(p.functionResponse))) {
        (last.parts as Json[]).push(part);
      } else contents.push({ role: 'user', parts: [part] });
      continue;
    }
    const parts: Json[] = [];
    if (typeof m.content === 'string') parts.push({ text: m.content });
    else {
      for (const p of Array.isArray(m.content) ? m.content.filter(isRecord) : []) {
        if (p.type === 'text' && typeof p.text === 'string') parts.push({ text: p.text });
        else if (p.type === 'image_url') {
          const url = imageUrlOf(p);
          const d = dataUrl(url);
          if (d !== null) parts.push({ inlineData: { mimeType: d.mime, data: d.data } });
          else if (url !== '') parts.push({ fileData: { mimeType: guessMime(url), fileUri: url } });
        }
      }
    }
    if (parts.length === 0) parts.push({ text: '' });
    contents.push({ role: 'user', parts });
  }
  const system = systemTextOfOpenAI(messages);
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  out.contents = contents;
  if (Array.isArray(body.tools)) {
    const decls = body.tools.filter(isRecord).flatMap((t) => {
      const fn = isRecord(t.function) ? t.function : null;
      if (fn === null || typeof fn.name !== 'string') return [];
      const d: Json = { name: fn.name };
      if (typeof fn.description === 'string') d.description = fn.description;
      if (isRecord(fn.parameters)) d.parameters = geminiSchema(fn.parameters);
      return [d];
    });
    if (decls.length > 0) out.tools = [{ functionDeclarations: decls }];
    const choice = body.tool_choice;
    let mode: string | null = null;
    let allowed: string[] | null = null;
    if (choice === 'auto') mode = 'AUTO';
    else if (choice === 'required') mode = 'ANY';
    else if (choice === 'none') mode = 'NONE';
    else if (isRecord(choice) && isRecord(choice.function) && typeof choice.function.name === 'string') {
      mode = 'ANY';
      allowed = [choice.function.name];
    }
    if (mode !== null && decls.length > 0) out.toolConfig = { functionCallingConfig: allowed === null ? { mode } : { mode, allowedFunctionNames: allowed } };
  }
  const config: Json = {};
  const maxTokens = num(body.max_tokens) ?? num(body.max_completion_tokens);
  if (maxTokens !== null) config.maxOutputTokens = maxTokens;
  if (num(body.temperature) !== null) config.temperature = body.temperature;
  if (num(body.top_p) !== null) config.topP = body.top_p;
  const stops = stringList(body.stop);
  if (stops.length > 0) config.stopSequences = stops;
  const mode = jsonModeOf(body);
  if (mode !== null) {
    config.responseMimeType = 'application/json';
    const rf = body.response_format as Json;
    if (mode === 'json_schema' && isRecord(rf.json_schema) && isRecord(rf.json_schema.schema)) config.responseJsonSchema = rf.json_schema.schema;
  }
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

// ---------------------------------------------------------------- client Anthropic -> provider requests

/** Anthropic -> Anthropic: the body untouched but for the model. */
export function anthropicPassThrough(body: Json, model: string): Json {
  return { ...body, model };
}

/** Anthropic Messages request -> OpenAI Chat Completions request. */
export function openAIFromAnthropic(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = blocksText(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  for (const m of messagesOf(body)) {
    const content = m.content;
    if (m.role === 'assistant') {
      if (typeof content === 'string') {
        messages.push({ role: 'assistant', content });
        continue;
      }
      const blocks = Array.isArray(content) ? content.filter(isRecord) : [];
      const calls = blocks
        .filter((b) => b.type === 'tool_use')
        .map((b) => ({ id: str(b.id), type: 'function', function: { name: str(b.name), arguments: JSON.stringify(isRecord(b.input) ? b.input : {}) } }));
      const text = blocksText(blocks);
      const msg: Json = { role: 'assistant', content: text !== '' ? text : null };
      if (calls.length > 0) msg.tool_calls = calls;
      messages.push(msg);
      continue;
    }
    if (typeof content === 'string') {
      messages.push({ role: 'user', content });
      continue;
    }
    const parts: Json[] = [];
    for (const b of Array.isArray(content) ? content.filter(isRecord) : []) {
      if (b.type === 'tool_result') {
        const c = b.content;
        const text = typeof c === 'string' ? c : Array.isArray(c) ? blocksText(c) || JSON.stringify(c) : c === undefined ? '' : JSON.stringify(c);
        messages.push({ role: 'tool', tool_call_id: str(b.tool_use_id), content: text });
      } else if (b.type === 'text' && typeof b.text === 'string') parts.push({ type: 'text', text: b.text });
      else if (b.type === 'image' && isRecord(b.source)) {
        const s = b.source;
        if (s.type === 'base64' && typeof s.data === 'string') parts.push({ type: 'image_url', image_url: { url: `data:${str(s.media_type) || 'image/png'};base64,${s.data}` } });
        else if (s.type === 'url' && typeof s.url === 'string') parts.push({ type: 'image_url', image_url: { url: s.url } });
      }
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }
  const out: Json = { model, messages };
  if (num(body.max_tokens) !== null) out.max_tokens = body.max_tokens;
  if (num(body.temperature) !== null) out.temperature = body.temperature;
  if (num(body.top_p) !== null) out.top_p = body.top_p;
  const stops = stringList(body.stop_sequences);
  if (stops.length > 0) out.stop = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.filter(isRecord).flatMap((t) => {
      if (typeof t.name !== 'string') return [];
      const fn: Json = { name: t.name, parameters: isRecord(t.input_schema) ? t.input_schema : { type: 'object', properties: {} } };
      if (typeof t.description === 'string') fn.description = t.description;
      return [{ type: 'function', function: fn }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice) && out.tools !== undefined) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'none') out.tool_choice = 'none';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
    if (choice.disable_parallel_tool_use === true) out.parallel_tool_calls = false;
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** The provider request for one client request: the body, and for gemini the model and stream in the path. */
export function upstreamBody(api: ClientApi, wire: Wire, body: Json, model: string): Json {
  if (api === 'openai') {
    if (wire === 'openai') return openAIPassThrough(body, model);
    if (wire === 'anthropic') return anthropicFromOpenAI(body, model);
    return geminiFromOpenAI(body);
  }
  if (wire === 'anthropic') return anthropicPassThrough(body, model);
  const oa = openAIFromAnthropic(body, model);
  if (wire === 'openai') return oa;
  return geminiFromOpenAI(oa);
}

// ---------------------------------------------------------------- provider answers -> Result

interface Frame {
  event: string | null;
  data: string;
}

export function sseFrames(text: string): Frame[] {
  const frames: Frame[] = [];
  for (const raw of text.replace(/\r\n/g, '\n').split('\n\n')) {
    if (raw.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of raw.split('\n')) {
      if (line.startsWith('event:')) event = line.slice(6).trim();
      else if (line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function jsonOf(text: string): Json | null {
  try {
    const v: unknown = JSON.parse(text);
    return isRecord(v) ? v : null;
  } catch {
    return null;
  }
}

function openAIStop(v: unknown): Stop {
  if (v === 'length') return 'length';
  if (v === 'tool_calls' || v === 'function_call') return 'tool_calls';
  if (v === 'content_filter') return 'content_filter';
  return 'stop';
}

function anthropicStop(v: unknown): Stop {
  if (v === 'max_tokens' || v === 'model_context_window_exceeded') return 'length';
  if (v === 'tool_use') return 'tool_calls';
  if (v === 'refusal') return 'content_filter';
  return 'stop';
}

function geminiStop(v: unknown, hasCalls: boolean): Stop {
  if (v === 'MAX_TOKENS') return 'length';
  if (typeof v === 'string' && ['SAFETY', 'RECITATION', 'BLOCKLIST', 'PROHIBITED_CONTENT', 'SPII', 'IMAGE_SAFETY', 'LANGUAGE'].includes(v)) return 'content_filter';
  return hasCalls ? 'tool_calls' : 'stop';
}

function contentText(c: unknown): string {
  if (typeof c === 'string') return c;
  if (!Array.isArray(c)) return '';
  return c.flatMap((b) => (isRecord(b) && typeof b.text === 'string' ? [b.text] : [])).join('');
}

function fromOpenAIJson(v: Json): Result | null {
  const choice = Array.isArray(v.choices) && isRecord(v.choices[0]) ? v.choices[0] : null;
  if (choice === null) return null;
  const msg = isRecord(choice.message) ? choice.message : {};
  const text = contentText(msg.content);
  const toolCalls = (Array.isArray(msg.tool_calls) ? msg.tool_calls.filter(isRecord) : []).map((c, i) => {
    const fn = isRecord(c.function) ? c.function : {};
    return { id: str(c.id) || `call_${i}`, name: str(fn.name), arguments: typeof fn.arguments === 'string' ? fn.arguments : JSON.stringify(fn.arguments ?? {}) };
  });
  const usage = isRecord(v.usage) ? v.usage : {};
  return {
    id: str(v.id) || null,
    created: num(v.created),
    model: str(v.model) || null,
    text,
    pieces: [text],
    toolCalls,
    stop: openAIStop(choice.finish_reason),
    usage: { input: int0(usage.prompt_tokens), output: int0(usage.completion_tokens) },
  };
}

function fromOpenAIStream(frames: Frame[]): Result | null {
  let id: string | null = null;
  let created: number | null = null;
  let model: string | null = null;
  const pieces: string[] = [];
  const calls = new Map<number, ToolCall>();
  let finish: unknown = null;
  let usage = { input: 0, output: 0 };
  let any = false;
  for (const f of frames) {
    const payload = f.data.trim();
    if (payload === '[DONE]') continue;
    const chunk = jsonOf(payload);
    if (chunk === null) continue;
    if (isRecord(chunk.error)) return null;
    any = true;
    id ??= str(chunk.id) || null;
    created ??= num(chunk.created);
    model ??= str(chunk.model) || null;
    if (isRecord(chunk.usage)) usage = { input: int0(chunk.usage.prompt_tokens), output: int0(chunk.usage.completion_tokens) };
    const choice = Array.isArray(chunk.choices) && isRecord(chunk.choices[0]) ? chunk.choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string' && delta.content !== '') pieces.push(delta.content);
    for (const c of Array.isArray(delta.tool_calls) ? delta.tool_calls.filter(isRecord) : []) {
      const index = num(c.index) ?? calls.size;
      const fn = isRecord(c.function) ? c.function : {};
      const cur = calls.get(index) ?? { id: '', name: '', arguments: '' };
      if (str(c.id) !== '') cur.id = str(c.id);
      if (str(fn.name) !== '') cur.name = str(fn.name);
      if (typeof fn.arguments === 'string') cur.arguments += fn.arguments;
      calls.set(index, cur);
    }
    if (typeof choice.finish_reason === 'string') finish = choice.finish_reason;
  }
  if (!any) return null;
  const toolCalls = [...calls.keys()].sort((a, b) => a - b).map((k, i) => {
    const c = calls.get(k) as ToolCall;
    return { ...c, id: c.id || `call_${i}` };
  });
  return { id, created, model, text: pieces.join(''), pieces, toolCalls, stop: openAIStop(finish), usage };
}

function fromAnthropicJson(v: Json): Result | null {
  if (v.type !== 'message' || !Array.isArray(v.content)) return null;
  const blocks = v.content.filter(isRecord);
  const text = blocks.flatMap((b) => (b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('');
  const toolCalls = blocks
    .filter((b) => b.type === 'tool_use')
    .map((b, i) => ({ id: str(b.id) || `toolu_${i}`, name: str(b.name), arguments: JSON.stringify(isRecord(b.input) ? b.input : {}) }));
  const usage = isRecord(v.usage) ? v.usage : {};
  return {
    id: str(v.id) || null,
    created: null,
    model: str(v.model) || null,
    text,
    pieces: [text],
    toolCalls,
    stop: anthropicStop(v.stop_reason),
    usage: { input: int0(usage.input_tokens), output: int0(usage.output_tokens) },
  };
}

function fromAnthropicStream(frames: Frame[]): Result | null {
  let id: string | null = null;
  let model: string | null = null;
  const pieces: string[] = [];
  const blocks = new Map<number, { kind: string; call: ToolCall }>();
  let stop: unknown = null;
  let input = 0;
  let output = 0;
  let started = false;
  for (const f of frames) {
    const d = jsonOf(f.data);
    if (d === null) continue;
    const type = f.event ?? str(d.type);
    if (type === 'error') return null;
    if (type === 'message_start' && isRecord(d.message)) {
      started = true;
      id = str(d.message.id) || null;
      model = str(d.message.model) || null;
      if (isRecord(d.message.usage)) {
        input = int0(d.message.usage.input_tokens);
        output = int0(d.message.usage.output_tokens);
      }
    } else if (type === 'content_block_start' && isRecord(d.content_block)) {
      const cb = d.content_block;
      const index = num(d.index) ?? blocks.size;
      blocks.set(index, { kind: str(cb.type), call: { id: str(cb.id), name: str(cb.name), arguments: '' } });
      if (cb.type === 'text' && typeof cb.text === 'string' && cb.text !== '') pieces.push(cb.text);
    } else if (type === 'content_block_delta' && isRecord(d.delta)) {
      const b = blocks.get(num(d.index) ?? -1);
      if (d.delta.type === 'text_delta' && typeof d.delta.text === 'string') pieces.push(d.delta.text);
      else if (d.delta.type === 'input_json_delta' && b !== undefined && typeof d.delta.partial_json === 'string') b.call.arguments += d.delta.partial_json;
    } else if (type === 'message_delta') {
      if (isRecord(d.delta) && typeof d.delta.stop_reason === 'string') stop = d.delta.stop_reason;
      if (isRecord(d.usage)) {
        if (num(d.usage.output_tokens) !== null) output = int0(d.usage.output_tokens);
        if (num(d.usage.input_tokens) !== null) input = int0(d.usage.input_tokens);
      }
    }
  }
  if (!started) return null;
  const toolCalls = [...blocks.keys()]
    .sort((a, b) => a - b)
    .flatMap((k) => {
      const b = blocks.get(k) as { kind: string; call: ToolCall };
      return b.kind === 'tool_use' ? [{ ...b.call, id: b.call.id || `toolu_${k}`, arguments: b.call.arguments || '{}' }] : [];
    });
  return { id, created: null, model, text: pieces.join(''), pieces, toolCalls, stop: anthropicStop(stop), usage: { input, output } };
}

function fromGemini(responses: Json[]): Result | null {
  if (responses.length === 0) return null;
  const pieces: string[] = [];
  const toolCalls: ToolCall[] = [];
  let finish: unknown = null;
  let usage = { input: 0, output: 0 };
  let model: string | null = null;
  let any = false;
  for (const r of responses) {
    if (isRecord(r.error)) return null;
    const cand = Array.isArray(r.candidates) && isRecord(r.candidates[0]) ? r.candidates[0] : null;
    if (isRecord(r.usageMetadata)) usage = { input: int0(r.usageMetadata.promptTokenCount), output: int0(r.usageMetadata.candidatesTokenCount) };
    if (typeof r.modelVersion === 'string') model = r.modelVersion;
    if (cand === null) continue;
    any = true;
    const parts = isRecord(cand.content) && Array.isArray(cand.content.parts) ? cand.content.parts.filter(isRecord) : [];
    for (const p of parts) {
      if (p.thought === true) continue;
      if (typeof p.text === 'string' && p.text !== '') pieces.push(p.text);
      if (isRecord(p.functionCall)) {
        toolCalls.push({ id: `call_${toolCalls.length}`, name: str(p.functionCall.name), arguments: JSON.stringify(isRecord(p.functionCall.args) ? p.functionCall.args : {}) });
      }
    }
    if (typeof cand.finishReason === 'string') finish = cand.finishReason;
  }
  if (!any) return null;
  return { id: null, created: null, model, text: pieces.join(''), pieces, toolCalls, stop: geminiStop(finish, toolCalls.length > 0), usage };
}

/** One provider's 2xx answer as a Result; null when it is not an answer of its wire. */
export function parseUpstream(wire: Wire, stream: boolean, contentType: string, text: string): Result | null {
  const isSse = stream || contentType.includes('text/event-stream');
  if (isSse && !(contentType.includes('application/json') && text.trimStart().startsWith('{'))) {
    const frames = sseFrames(text);
    if (wire === 'openai') return fromOpenAIStream(frames);
    if (wire === 'anthropic') return fromAnthropicStream(frames);
    return fromGemini(frames.flatMap((f) => jsonOf(f.data) ?? []));
  }
  const trimmed = text.trim();
  if (wire === 'gemini' && trimmed.startsWith('[')) {
    try {
      const arr: unknown = JSON.parse(trimmed);
      if (Array.isArray(arr)) return fromGemini(arr.filter(isRecord));
    } catch {
      return null;
    }
  }
  const v = jsonOf(trimmed);
  if (v === null) return null;
  if (wire === 'openai') return fromOpenAIJson(v);
  if (wire === 'anthropic') return fromAnthropicJson(v);
  return fromGemini([v]);
}

// ---------------------------------------------------------------- Result -> client answer

export interface Rendered {
  contentType: string;
  body: string;
}

const sse = (event: string | null, data: unknown): string => `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;

function anthropicStopOf(stop: Stop): string {
  if (stop === 'length') return 'max_tokens';
  if (stop === 'tool_calls') return 'tool_use';
  if (stop === 'content_filter') return 'refusal';
  return 'end_turn';
}

export function renderOpenAI(r: Result, stream: boolean, model: string, fallbackId: string): Rendered {
  const id = r.id !== null && r.id.startsWith('chatcmpl') ? r.id : fallbackId;
  const created = r.created ?? FIXED_CREATED;
  const usage = { prompt_tokens: r.usage.input, completion_tokens: r.usage.output, total_tokens: r.usage.input + r.usage.output };
  const calls = r.toolCalls.map((c) => ({ id: c.id, type: 'function', function: { name: c.name, arguments: c.arguments } }));
  if (!stream) {
    const message: Json = { role: 'assistant', content: r.text === '' && calls.length > 0 ? null : r.text, refusal: null };
    if (calls.length > 0) message.tool_calls = calls;
    return {
      contentType: 'application/json',
      body: JSON.stringify({ id, object: 'chat.completion', created, model, choices: [{ index: 0, message, logprobs: null, finish_reason: r.stop }], usage }),
    };
  }
  const chunk = (delta: Json, finish: Stop | null): Json => ({ id, object: 'chat.completion.chunk', created, model, choices: [{ index: 0, delta, logprobs: null, finish_reason: finish }] });
  const out: string[] = [sse(null, chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of r.pieces) if (piece !== '') out.push(sse(null, chunk({ content: piece }, null)));
  calls.forEach((c, index) => {
    out.push(sse(null, chunk({ tool_calls: [{ index, id: c.id, type: 'function', function: { name: c.function.name, arguments: '' } }] }, null)));
    if (c.function.arguments !== '') out.push(sse(null, chunk({ tool_calls: [{ index, function: { arguments: c.function.arguments } }] }, null)));
  });
  out.push(sse(null, chunk({}, r.stop)));
  out.push(sse(null, { id, object: 'chat.completion.chunk', created, model, choices: [], usage }));
  out.push('data: [DONE]\n\n');
  return { contentType: 'text/event-stream; charset=utf-8', body: out.join('') };
}

export function renderAnthropic(r: Result, stream: boolean, model: string, fallbackId: string): Rendered {
  const id = r.id !== null && r.id.startsWith('msg_') ? r.id : fallbackId;
  const stopReason = anthropicStopOf(r.stop);
  const toolBlocks = r.toolCalls.map((c) => ({ type: 'tool_use', id: c.id, name: c.name, input: parseArgs(c.arguments) }));
  if (!stream) {
    const content: Json[] = [];
    if (r.text !== '' || toolBlocks.length === 0) content.push({ type: 'text', text: r.text });
    content.push(...toolBlocks);
    return {
      contentType: 'application/json',
      body: JSON.stringify({
        id,
        type: 'message',
        role: 'assistant',
        model,
        content,
        stop_reason: stopReason,
        stop_sequence: null,
        usage: { input_tokens: r.usage.input, output_tokens: r.usage.output },
      }),
    };
  }
  const out: string[] = [
    sse('message_start', {
      type: 'message_start',
      message: { id, type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: r.usage.input, output_tokens: 0 } },
    }),
  ];
  let index = 0;
  if (r.text !== '' || toolBlocks.length === 0) {
    out.push(sse('content_block_start', { type: 'content_block_start', index, content_block: { type: 'text', text: '' } }));
    for (const piece of r.pieces) if (piece !== '') out.push(sse('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'text_delta', text: piece } }));
    out.push(sse('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  for (const b of toolBlocks) {
    out.push(sse('content_block_start', { type: 'content_block_start', index, content_block: { type: 'tool_use', id: b.id, name: b.name, input: {} } }));
    out.push(sse('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: JSON.stringify(b.input) } }));
    out.push(sse('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  out.push(
    sse('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { input_tokens: r.usage.input, output_tokens: r.usage.output } }),
    sse('message_stop', { type: 'message_stop' }),
  );
  return { contentType: 'text/event-stream; charset=utf-8', body: out.join('') };
}

export function errorBody(api: ClientApi, status: number, type: string, message: string): string {
  if (api === 'anthropic') return JSON.stringify({ type: 'error', error: { type, message } });
  return JSON.stringify({ error: { message, type, param: null, code: status === 429 ? 'rate_limit_exceeded' : type } });
}

export function openAIModelList(models: readonly { id: string; provider: string }[]): Json {
  return { object: 'list', data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider })) };
}

export function anthropicModelList(models: readonly { id: string; provider: string }[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.id, created_at: '2026-01-01T00:00:00Z' })),
    has_more: false,
    first_id: models.length > 0 ? models[0].id : null,
    last_id: models.length > 0 ? models[models.length - 1].id : null,
  };
}

/** What a request body itself demands of a model (unioned with the manifest's row). */
export function bodyRequirements(api: ClientApi, body: Json): { tools: boolean; vision: boolean; json: boolean; tokens: number } {
  const messages = messagesOf(body);
  let vision = false;
  for (const m of messages) {
    if (!Array.isArray(m.content)) continue;
    for (const b of m.content) if (isRecord(b) && (b.type === 'image' || b.type === 'image_url')) vision = true;
  }
  const tools = Array.isArray(body.tools) && body.tools.length > 0;
  // A rough context need: ~3 characters a token over the whole body, plus the output cap.
  const chars = JSON.stringify(messages).length + JSON.stringify(body.system ?? '').length + JSON.stringify(body.tools ?? []).length;
  const cap = num(body.max_tokens) ?? num(body.max_completion_tokens) ?? 0;
  return { tools, vision, json: wantsJson(api, body), tokens: Math.ceil(chars / 3) + cap };
}
