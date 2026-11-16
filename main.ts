// free-model-aggregator-02: one aggregator server behind two client wires.
//
// A client points at this server and speaks either OpenAI Chat Completions or Anthropic
// Messages. Every answer is a real completion from one of the free models the trial brief
// lists, dialled through the grader's gateway on the trial's key; this server never answers
// from itself. It registers its port and catalog with the env, serves the six prompts the env
// drives, and exits 0 when the env's /work says done.
//
// Design, in the order the score rewards it:
//   - Routing (quality): candidates are the free models that meet the prompt's requirements
//     (from the brief's manifest, in arrival order), ranked by the house's published rating,
//     best first. The reported `model` is always the id actually dialled.
//   - Failover: a refused gateway call (4xx/5xx) is retried once on the same model when it
//     looks transient (429 or 5xx), then the next candidate is tried. A model that has
//     refused twice this trial is demoted below the rest for later prompts, so a model that
//     is down stops costing every prompt its retry. A 2xx whose text does not carry the
//     prompt's token is not an answer either: the next candidate is tried. Refusals
//     recovered from are reported on the response's x-makefun-upstream-failures header,
//     as the brief allows.
//   - Wire fidelity: gateway answers are reassembled and re-emitted in the client's wire
//     with the dialled model id on every body and every streamed chunk, usage whole, and
//     stop reasons mapped into the client's vocabulary.
//   - Determinism: every choice derives from the brief, the request and the gateway's
//     answers seen so far — no wall clock, no randomness — so a replay that serves the
//     recorded gateway answers back walks the same path.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';

/** A parsed JSON object. */
type Json = Record<string, unknown>;

/** One free model row of the brief. */
interface ModelRow {
  id: string;
  displayName: string;
  ratingMilli: number;
  contextLength: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

/** What a prompt needs from its model. */
interface Requirement {
  tools: boolean;
  vision: boolean;
  json: boolean;
}

/** One prompt row of the brief's manifest. */
interface PromptRow {
  id: string;
  api: string;
  stream: boolean;
  weight: number;
  requires: Requirement;
}

/** The brief: the free model list and the prompts' manifest. */
interface Brief {
  models: ModelRow[];
  prompts: PromptRow[];
}

/** One upstream failure, as the x-makefun-upstream-failures trail carries it. */
interface FailureRow {
  status: number;
  code: string;
  model: string | null;
}

/** A tool call reassembled from a gateway answer. */
interface ToolCall {
  id: string;
  name: string;
  /** The call's arguments, as the raw JSON string the provider sent. */
  arguments: string;
}

/** A gateway answer reassembled into wire-neutral parts. */
interface Assembled {
  text: string;
  /** The text as the provider streamed it, piece by piece (stream answers only). */
  textPieces: string[];
  toolCalls: ToolCall[];
  finishReason: string | null;
  usage: { input: number; output: number } | null;
  providerId: string | null;
  created: number | null;
  /** The gateway stream's data payloads, in order (stream answers only). */
  rawChunks: string[] | null;
}

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 250;
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The gateway refuses a larger output cap than this; client asks above it are clamped. */
const MAX_OUTPUT_TOKENS = 8192;
/** Output cap for a client request that names none (the gateway refuses a capless call). */
const DEFAULT_OUTPUT_TOKENS = 1024;
/** One gateway call gives up after this long; the env allows 30 s per prompt. */
const ATTEMPT_TIMEOUT_MS = 22000;
/** The most gateway calls one prompt may cost, retries and failovers included. */
const MAX_ATTEMPTS = 4;
/** The upstream-failures trail holds at most this many rows. */
const MAX_TRAIL_ROWS = 8;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_BASE = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

function log(message: string): void {
  process.stderr.write(`aggregator-02: ${message}\n`);
}

function isRecord(value: unknown): value is Json {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function asArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function asNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function asString(value: unknown): string | null {
  return typeof value === 'string' ? value : null;
}

// ---------------------------------------------------------------------------
// The brief
// ---------------------------------------------------------------------------

/** The balanced {...} that starts at `start`, string-aware, or null when it never closes. */
function extractObject(text: string, start: number): string | null {
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let i = start; i < text.length; i++) {
    const c = text[i];
    if (inString) {
      if (escaped) escaped = false;
      else if (c === '\\') escaped = true;
      else if (c === '"') inString = false;
      continue;
    }
    if (c === '"') inString = true;
    else if (c === '{') depth += 1;
    else if (c === '}') {
      depth -= 1;
      if (depth === 0) return text.slice(start, i + 1);
    }
  }
  return null;
}

/** Every top-level JSON document embedded in the brief's prose, by its schema name. */
function briefDocuments(text: string): Map<string, Json> {
  const documents = new Map<string, Json>();
  let from = 0;
  for (;;) {
    const at = text.indexOf('{"schema"', from);
    if (at === -1) return documents;
    const raw = extractObject(text, at);
    if (raw === null) return documents;
    from = at + raw.length;
    try {
      const value: unknown = JSON.parse(raw);
      if (isRecord(value) && typeof value.schema === 'string') documents.set(value.schema, value);
    } catch {
      // Not a document after all; keep scanning.
    }
  }
}

function requirementOf(raw: unknown): Requirement {
  const o = isRecord(raw) ? raw : {};
  return { tools: o.tools === true, vision: o.vision === true, json: o.json === true };
}

/** The brief at MAKEFUN_PROMPT_FILE; empty when there is none (a --list outside a trial). */
function readBrief(path: string): Brief {
  const brief: Brief = { models: [], prompts: [] };
  if (path === '') return brief;
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch {
    return brief;
  }
  for (const [schema, doc] of briefDocuments(text)) {
    if (schema.endsWith('/models/v1')) {
      for (const raw of asArray(doc.models)) {
        if (!isRecord(raw) || typeof raw.id !== 'string') continue;
        brief.models.push({
          id: raw.id,
          displayName: asString(raw.display_name) ?? raw.id,
          ratingMilli: asNumber(raw.rating_milli) ?? 0,
          contextLength: asNumber(raw.context_length),
          tools: raw.tools === true,
          vision: raw.vision === true,
          json: raw.json === true,
          free: raw.free !== false,
        });
      }
    } else if (schema.endsWith('/prompts/v1')) {
      for (const raw of asArray(doc.prompts)) {
        if (!isRecord(raw) || typeof raw.id !== 'string') continue;
        brief.prompts.push({
          id: raw.id,
          api: asString(raw.api) ?? '',
          stream: raw.stream === true,
          weight: asNumber(raw.weight) ?? 1,
          requires: requirementOf(raw.requires),
        });
      }
    }
  }
  return brief;
}

const brief = readBrief(PROMPT_FILE);

/** The catalog document: the same object --list prints and the register body carries. */
function buildCatalog(theBrief: Brief): Json {
  return {
    providers: [
      {
        id: 'openrouter',
        base_url: 'https://openrouter.ai/api/v1',
        key_env: 'OPENROUTER_API_KEY',
        wire: 'openai',
        models: theBrief.models
          .filter((model) => model.free)
          .map((model) => ({ id: model.id, free: true, context_length: model.contextLength })),
      },
    ],
  };
}

const catalog = buildCatalog(brief);

// ---------------------------------------------------------------------------
// Routing
// ---------------------------------------------------------------------------

/** Whether a model meets a prompt's requirements. */
function meetsRequirements(model: ModelRow, requires: Requirement): boolean {
  if (requires.tools && !model.tools) return false;
  if (requires.vision && !model.vision) return false;
  if (requires.json && !model.json) return false;
  return true;
}

/** Requirements read off a request body, for a brief with no manifest row for it. */
function requirementsFromBody(body: Json): Requirement {
  let vision = false;
  for (const message of asArray(body.messages)) {
    if (!isRecord(message)) continue;
    for (const block of asArray(message.content)) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: isRecord(body.response_format),
  };
}

/** Which prompt of the brief the request now arriving is, counted by completions served. */
let arrivalIndex = 0;

/** The requirements of the request now arriving: the brief's manifest row, else the body's. */
function requirementsFor(body: Json): Requirement {
  const row = brief.prompts[arrivalIndex];
  arrivalIndex += 1;
  if (row !== undefined) return row.requires;
  return requirementsFromBody(body);
}

/** A rough input-token estimate of a request: one token per three bytes, as the gateway reckons. */
function estimateInputTokens(body: Json): number {
  return Math.ceil(JSON.stringify(body.messages ?? []).length / 3);
}

/** How often each model has refused a gateway call this trial, in the order they arrived. */
const refusalCounts = new Map<string, number>();

/** A model with this many refusals is demoted below the rest for later prompts. */
const DEMOTE_AFTER_REFUSALS = 2;

/** The ranked candidates for one prompt: best rating first, ties by id, the demoted last. */
function candidatesFor(requires: Requirement, body: Json): ModelRow[] {
  const cap = asNumber(body.max_tokens) ?? asNumber(body.max_completion_tokens) ?? DEFAULT_OUTPUT_TOKENS;
  const need = estimateInputTokens(body) + cap;
  let pool = brief.models.filter((model) => model.free && meetsRequirements(model, requires));
  const fits = pool.filter((model) => model.contextLength === null || model.contextLength >= need);
  if (fits.length > 0) pool = fits;
  const demoted = (model: ModelRow): number => ((refusalCounts.get(model.id) ?? 0) >= DEMOTE_AFTER_REFUSALS ? 1 : 0);
  return [...pool].sort(
    (a, b) => demoted(a) - demoted(b) || b.ratingMilli - a.ratingMilli || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0),
  );
}

const TOKEN_PATTERN = /[A-Z]{3,12}-[0-9]{4}/g;

/** Every text a request carries, for the token scan. */
function requestTexts(body: Json): string[] {
  const texts: string[] = [];
  const walk = (value: unknown): void => {
    if (typeof value === 'string') texts.push(value);
    else if (Array.isArray(value)) for (const item of value) walk(item);
    else if (isRecord(value)) for (const key of Object.keys(value)) walk(value[key]);
  };
  walk(body.system);
  walk(body.messages);
  return texts;
}

/**
 * The unique token a prompt carries (the workload's WORD-1234), or null when the request
 * names none or several: then no token check can be made and any real answer serves.
 */
function expectedToken(body: Json): string | null {
  const found = new Set<string>();
  for (const text of requestTexts(body)) {
    for (const match of text.matchAll(TOKEN_PATTERN)) found.add(match[0]);
  }
  return found.size === 1 ? [...found][0] : null;
}

// ---------------------------------------------------------------------------
// Client request -> gateway request
// ---------------------------------------------------------------------------

/** The output cap a gateway call carries: the client's, clamped to the gateway's ceiling. */
function outputCap(body: Json): number {
  const asked = asNumber(body.max_tokens) ?? asNumber(body.max_completion_tokens);
  if (asked === null) return DEFAULT_OUTPUT_TOKENS;
  return Math.max(1, Math.min(MAX_OUTPUT_TOKENS, Math.floor(asked)));
}

/** An OpenAI-shaped client body, retargeted at the chosen model, on the gateway's terms. */
function openAIToGateway(body: Json, modelId: string): Json {
  const out: Json = { ...body, model: modelId };
  delete out.store; // a trial carries no provider state; the gateway refuses store: true
  if (typeof out.max_tokens !== 'number' && typeof out.max_completion_tokens !== 'number') {
    out.max_tokens = DEFAULT_OUTPUT_TOKENS;
  }
  if (typeof out.max_tokens === 'number') out.max_tokens = Math.min(out.max_tokens, MAX_OUTPUT_TOKENS);
  if (typeof out.max_completion_tokens === 'number') {
    out.max_completion_tokens = Math.min(out.max_completion_tokens, MAX_OUTPUT_TOKENS);
  }
  if (out.stream === true) {
    const options = isRecord(out.stream_options) ? out.stream_options : {};
    out.stream_options = { ...options, include_usage: true };
  }
  return out;
}

/** The text of an Anthropic tool_result block, flattened for a Chat Completions tool message. */
function toolResultText(block: Json): string {
  if (typeof block.content === 'string') return block.content;
  const parts: string[] = [];
  for (const part of asArray(block.content)) {
    if (isRecord(part) && part.type === 'text' && typeof part.text === 'string') parts.push(part.text);
  }
  return parts.join('');
}

/** An Anthropic image block as a Chat Completions image part. */
function imagePart(block: Json): Json {
  const source = isRecord(block.source) ? block.source : {};
  if (source.type === 'base64' && typeof source.data === 'string') {
    return { type: 'image_url', image_url: { url: `data:${asString(source.media_type) ?? 'image/png'};base64,${source.data}` } };
  }
  if (source.type === 'url' && typeof source.url === 'string') {
    return { type: 'image_url', image_url: { url: source.url } };
  }
  return { type: 'text', text: '' };
}

/** One Anthropic message as one or more Chat Completions messages. */
function translateAnthropicMessage(message: Json): Json[] {
  const role = asString(message.role) ?? 'user';
  const content = message.content;
  if (typeof content === 'string') return [{ role, content }];
  if (!Array.isArray(content)) return [{ role, content: '' }];
  if (role === 'user') {
    const out: Json[] = [];
    const parts: Json[] = [];
    for (const raw of content) {
      if (!isRecord(raw)) continue;
      if (raw.type === 'tool_result') {
        out.push({ role: 'tool', tool_call_id: asString(raw.tool_use_id) ?? '', content: toolResultText(raw) });
      } else if (raw.type === 'text') {
        parts.push({ type: 'text', text: asString(raw.text) ?? '' });
      } else if (raw.type === 'image') {
        parts.push(imagePart(raw));
      }
    }
    if (parts.length === 1 && parts[0].type === 'text') out.push({ role: 'user', content: asString(parts[0].text) ?? '' });
    else if (parts.length > 0) out.push({ role: 'user', content: parts });
    if (out.length === 0) out.push({ role: 'user', content: '' });
    return out;
  }
  const parts: Json[] = [];
  const toolCalls: Json[] = [];
  for (const raw of content) {
    if (!isRecord(raw)) continue;
    if (raw.type === 'text') parts.push({ type: 'text', text: asString(raw.text) ?? '' });
    else if (raw.type === 'tool_use') {
      toolCalls.push({
        id: asString(raw.id) ?? `call_${toolCalls.length}`,
        type: 'function',
        function: { name: asString(raw.name) ?? '', arguments: JSON.stringify(raw.input ?? {}) },
      });
    }
  }
  const translated: Json = {
    role: 'assistant',
    content: parts.length === 1 ? (asString(parts[0].text) ?? '') : parts.length > 0 ? parts : null,
  };
  if (toolCalls.length > 0) translated.tool_calls = toolCalls;
  return [translated];
}

/** An Anthropic-shaped client body as a Chat Completions body naming the chosen model. */
function anthropicToOpenAI(body: Json, modelId: string): Json {
  const messages: Json[] = [];
  if (typeof body.system === 'string' && body.system !== '') {
    messages.push({ role: 'system', content: body.system });
  } else if (Array.isArray(body.system)) {
    const parts: Json[] = [];
    for (const raw of body.system) {
      if (isRecord(raw) && raw.type === 'text') parts.push({ type: 'text', text: asString(raw.text) ?? '' });
    }
    if (parts.length > 0) messages.push({ role: 'system', content: parts });
  }
  for (const raw of asArray(body.messages)) {
    if (isRecord(raw)) messages.push(...translateAnthropicMessage(raw));
  }
  const out: Json = { model: modelId, messages, max_tokens: outputCap(body) };
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences)) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools) && body.tools.length > 0) {
    out.tools = body.tools.filter(isRecord).map((tool) => ({
      type: 'function',
      function: {
        name: asString(tool.name) ?? '',
        description: asString(tool.description) ?? '',
        parameters: isRecord(tool.input_schema) ? tool.input_schema : { type: 'object', properties: {} },
      },
    }));
  }
  if (isRecord(body.tool_choice)) {
    const choice = body.tool_choice;
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'none') out.tool_choice = 'none';
    else if (choice.type === 'tool' && typeof choice.name === 'string') {
      out.tool_choice = { type: 'function', function: { name: choice.name } };
    }
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

// ---------------------------------------------------------------------------
// Gateway answers, reassembled
// ---------------------------------------------------------------------------

/** The text of a Chat Completions content field: a string, or text parts joined. */
function contentText(content: unknown): string {
  if (typeof content === 'string') return content;
  const parts: string[] = [];
  for (const part of asArray(content)) {
    if (isRecord(part) && typeof part.text === 'string') parts.push(part.text);
  }
  return parts.join('');
}

function usageOf(raw: unknown): { input: number; output: number } | null {
  if (!isRecord(raw)) return null;
  const input = asNumber(raw.prompt_tokens);
  const output = asNumber(raw.completion_tokens);
  if (input === null || output === null) return null;
  return { input, output };
}

/** A non-streamed Chat Completions answer, reassembled. Null when it is not one. */
function assembleFromJson(answer: Json): Assembled | null {
  const choice = asArray(answer.choices)[0];
  if (!isRecord(choice)) return null;
  const message = isRecord(choice.message) ? choice.message : {};
  const toolCalls: ToolCall[] = [];
  for (const raw of asArray(message.tool_calls)) {
    if (!isRecord(raw)) continue;
    const fn = isRecord(raw.function) ? raw.function : {};
    toolCalls.push({
      id: asString(raw.id) ?? '',
      name: asString(fn.name) ?? '',
      arguments: asString(fn.arguments) ?? '',
    });
  }
  const text = contentText(message.content);
  return {
    text,
    textPieces: text === '' ? [] : [text],
    toolCalls,
    finishReason: asString(choice.finish_reason),
    usage: usageOf(answer.usage),
    providerId: asString(answer.id),
    created: asNumber(answer.created),
    rawChunks: null,
  };
}

/** A streamed Chat Completions answer, reassembled from its SSE body. Null when malformed. */
function assembleFromStream(sseText: string): Assembled | null {
  const frames = sseText.replace(/\r\n/g, '\n').split('\n\n');
  const rawChunks: string[] = [];
  const textPieces: string[] = [];
  const toolByIndex = new Map<number, ToolCall>();
  let text = '';
  let finishReason: string | null = null;
  let usage: { input: number; output: number } | null = null;
  let providerId: string | null = null;
  let created: number | null = null;
  let sawChunk = false;
  for (const frame of frames) {
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length === 0) continue;
    const payload = data.join('\n').trim();
    if (payload === '[DONE]') break;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      return null;
    }
    if (!isRecord(chunk) || !Array.isArray(chunk.choices)) return null;
    sawChunk = true;
    rawChunks.push(payload);
    providerId = asString(chunk.id) ?? providerId;
    created = asNumber(chunk.created) ?? created;
    const chunkUsage = usageOf(chunk.usage);
    if (chunkUsage !== null) usage = chunkUsage;
    const choice = asArray(chunk.choices)[0];
    if (!isRecord(choice)) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string' && delta.content !== '') {
      text += delta.content;
      textPieces.push(delta.content);
    }
    for (const raw of asArray(delta.tool_calls)) {
      if (!isRecord(raw)) continue;
      const index = asNumber(raw.index) ?? 0;
      const fn = isRecord(raw.function) ? raw.function : {};
      const existing = toolByIndex.get(index) ?? { id: '', name: '', arguments: '' };
      if (typeof raw.id === 'string' && raw.id !== '') existing.id = raw.id;
      if (typeof fn.name === 'string' && fn.name !== '') existing.name = fn.name;
      if (typeof fn.arguments === 'string') existing.arguments += fn.arguments;
      toolByIndex.set(index, existing);
    }
    if (typeof choice.finish_reason === 'string') finishReason = choice.finish_reason;
  }
  if (!sawChunk) return null;
  const toolCalls = [...toolByIndex.entries()].sort((a, b) => a[0] - b[0]).map(([, call]) => call);
  return { text, textPieces, toolCalls, finishReason, usage, providerId, created, rawChunks };
}

/** The usage an answer is served with: the provider's, or a deterministic estimate. */
function servedUsage(answer: Assembled, requestBody: Json): { input: number; output: number } {
  const argsChars = answer.toolCalls.reduce((n, call) => n + call.arguments.length, 0);
  const estimatedOutput = Math.max(1, Math.ceil((answer.text.length + argsChars) / 4));
  return {
    input: answer.usage?.input ?? estimateInputTokens(requestBody),
    output: answer.usage !== null && answer.usage.output >= 1 ? answer.usage.output : estimatedOutput,
  };
}

// ---------------------------------------------------------------------------
// Gateway calls and failover
// ---------------------------------------------------------------------------

function chatCompletionsUrl(): string {
  if (GATEWAY_BASE === '') return '';
  return GATEWAY_BASE.endsWith('/v1') ? `${GATEWAY_BASE}/chat/completions` : `${GATEWAY_BASE}/v1/chat/completions`;
}

/** The refusal code of a gateway error body: its error.code, else its error.type, else ''. */
function errorCodeOf(text: string): string {
  try {
    const value: unknown = JSON.parse(text);
    if (isRecord(value) && isRecord(value.error)) {
      const code = asString(value.error.code) ?? asString(value.error.type);
      if (code !== null) return code.slice(0, 64);
    }
  } catch {
    // An unparseable refusal carries no code.
  }
  return '';
}

type Attempt =
  | { kind: 'answer'; answer: Assembled }
  | { kind: 'refused'; status: number; code: string }
  | { kind: 'broken'; code: string };

/** One gateway call for one model. Never throws: every failure is an outcome. */
async function attempt(model: ModelRow, payload: Json): Promise<Attempt> {
  let response: Response;
  try {
    response = await fetch(chatCompletionsUrl(), {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        authorization: `Bearer ${GATEWAY_KEY}`,
        ...(payload.stream === true ? { accept: 'text/event-stream' } : {}),
      },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(ATTEMPT_TIMEOUT_MS),
    });
  } catch (err) {
    const timedOut = err instanceof Error && (err.name === 'TimeoutError' || err.name === 'AbortError');
    return { kind: 'refused', status: 502, code: timedOut ? 'upstream_timeout' : 'upstream_unreachable' };
  }
  let text: string;
  try {
    text = await response.text();
  } catch {
    return { kind: 'broken', code: 'upstream_unreadable' };
  }
  if (!response.ok) return { kind: 'refused', status: response.status, code: errorCodeOf(text) };
  let answer: Assembled | null = null;
  if (payload.stream === true && !text.trimStart().startsWith('{')) {
    answer = assembleFromStream(text);
  } else {
    try {
      const value: unknown = JSON.parse(text);
      if (isRecord(value)) answer = assembleFromJson(value);
    } catch {
      answer = null;
    }
    // A stream the provider answered as one JSON body is still an answer; a JSON request
    // answered with SSE is assembled from its events instead.
    if (answer === null && text.includes('data:')) answer = assembleFromStream(text);
  }
  if (answer === null) return { kind: 'broken', code: 'upstream_malformed' };
  return { kind: 'answer', answer };
}

interface Served {
  model: ModelRow;
  answer: Assembled;
  failures: FailureRow[];
  /** The last refusal seen, when every candidate failed. */
  lastRefusal: { status: number; code: string } | null;
}

/**
 * Serve one client request: walk the ranked candidates, retrying a transient refusal once
 * on the same model, until one model's answer carries the prompt (token included).
 */
async function serve(wire: 'openai' | 'anthropic', body: Json): Promise<Served> {
  const requires = requirementsFor(body);
  const token = expectedToken(body);
  const candidates = candidatesFor(requires, body);
  const failures: FailureRow[] = [];
  let lastRefusal: { status: number; code: string } | null = null;
  let attempts = 0;
  for (const model of candidates) {
    if (attempts >= MAX_ATTEMPTS) break;
    const payload = wire === 'openai' ? openAIToGateway(body, model.id) : anthropicToOpenAI(body, model.id);
    let retried = false;
    for (;;) {
      if (attempts >= MAX_ATTEMPTS) break;
      attempts += 1;
      const outcome = await attempt(model, payload);
      if (outcome.kind === 'answer') {
        if (token !== null && !outcome.answer.text.includes(token)) {
          // A real answer to the wrong prompt is no answer: try the next model.
          lastRefusal = { status: 502, code: 'answer_missing_token' };
          break;
        }
        return { model, answer: outcome.answer, failures, lastRefusal: null };
      }
      if (outcome.kind === 'refused') {
        failures.push({ status: outcome.status, code: outcome.code, model: model.id });
        refusalCounts.set(model.id, (refusalCounts.get(model.id) ?? 0) + 1);
        lastRefusal = { status: outcome.status, code: outcome.code };
        const transient = outcome.status === 429 || outcome.status >= 500;
        if (transient && !retried && attempts < MAX_ATTEMPTS) {
          retried = true;
          continue;
        }
        break;
      }
      // A 2xx we cannot assemble: not a refusal, and no trail row claims it was.
      lastRefusal = { status: 502, code: outcome.code };
      break;
    }
  }
  return {
    model: candidates[0] ?? { id: '', displayName: '', ratingMilli: 0, contextLength: null, tools: false, vision: false, json: false, free: false },
    answer: { text: '', textPieces: [], toolCalls: [], finishReason: null, usage: null, providerId: null, created: null, rawChunks: null },
    failures,
    lastRefusal,
  };
}

// ---------------------------------------------------------------------------
// Assembled answer -> client wire
// ---------------------------------------------------------------------------

const OPENAI_STOP_REASONS = new Set(['stop', 'length', 'tool_calls', 'function_call', 'content_filter']);

/** The finish reason an OpenAI client is served: the provider's, kept when it is one. */
function openAIStopReason(answer: Assembled): string {
  if (answer.finishReason !== null && OPENAI_STOP_REASONS.has(answer.finishReason)) return answer.finishReason;
  return answer.toolCalls.length > 0 ? 'tool_calls' : 'stop';
}

/** A Chat Completions stop reason in the Messages vocabulary. */
function anthropicStopReason(answer: Assembled): string {
  switch (answer.finishReason) {
    case 'length':
      return 'max_tokens';
    case 'tool_calls':
    case 'function_call':
      return 'tool_use';
    case 'content_filter':
      return 'refusal';
    default:
      return answer.toolCalls.length > 0 ? 'tool_use' : 'end_turn';
  }
}

/** A tool call's arguments as an object, for a Messages tool_use block. */
function argumentsObject(call: ToolCall): Json {
  try {
    const value: unknown = JSON.parse(call.arguments);
    if (isRecord(value)) return value;
  } catch {
    // Fall through to an empty object.
  }
  return {};
}

function openAIJsonAnswer(answer: Assembled, modelId: string, requestBody: Json): Json {
  const usage = servedUsage(answer, requestBody);
  const message: Json = { role: 'assistant', content: answer.text === '' && answer.toolCalls.length > 0 ? null : answer.text };
  if (answer.toolCalls.length > 0) {
    message.tool_calls = answer.toolCalls.map((call, i) => ({
      id: call.id !== '' ? call.id : `call_${i}`,
      type: 'function',
      function: { name: call.name, arguments: call.arguments },
    }));
  }
  return {
    id: answer.providerId ?? 'chatcmpl-aggregator',
    object: 'chat.completion',
    created: answer.created ?? 0,
    model: modelId,
    choices: [{ index: 0, message, finish_reason: openAIStopReason(answer) }],
    usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
  };
}

function anthropicJsonAnswer(answer: Assembled, modelId: string, requestBody: Json): Json {
  const usage = servedUsage(answer, requestBody);
  const content: Json[] = [];
  if (answer.text !== '') content.push({ type: 'text', text: answer.text });
  for (const call of answer.toolCalls) {
    content.push({ type: 'tool_use', id: call.id !== '' ? call.id : `toolu_${content.length}`, name: call.name, input: argumentsObject(call) });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const providerId = (answer.providerId ?? 'aggregator').replace(/[^A-Za-z0-9_-]/g, '');
  return {
    id: `msg_${providerId}`,
    type: 'message',
    role: 'assistant',
    model: modelId,
    content,
    stop_reason: anthropicStopReason(answer),
    stop_sequence: null,
    usage: { input_tokens: usage.input, output_tokens: usage.output },
  };
}

/** One SSE event: a data line and the blank line that ends the frame. */
function sseData(payload: string): string {
  return `data: ${payload}\n\n`;
}

function sseEvent(event: string, payload: Json): string {
  return `event: ${event}\ndata: ${JSON.stringify(payload)}\n\n`;
}

/** An OpenAI chunk object, the shape every streamed frame carries. */
function openAIChunk(answer: Assembled, modelId: string, delta: Json, finishReason: string | null, usage?: Json): Json {
  const chunk: Json = {
    id: answer.providerId ?? 'chatcmpl-aggregator',
    object: 'chat.completion.chunk',
    created: answer.created ?? 0,
    model: modelId,
    choices: [{ index: 0, delta, finish_reason: finishReason }],
  };
  if (usage !== undefined) chunk.usage = usage;
  return chunk;
}

/**
 * The OpenAI event stream for an answer: the provider's own chunks with the dialled model
 * put back on each, or chunks synthesized from the reassembled answer when the provider
 * answered a stream request with one JSON body.
 */
function openAIStreamAnswer(answer: Assembled, modelId: string, requestBody: Json): string {
  if (answer.rawChunks !== null) {
    let out = '';
    for (const raw of answer.rawChunks) {
      try {
        const chunk: unknown = JSON.parse(raw);
        if (isRecord(chunk)) {
          chunk.model = modelId;
          out += sseData(JSON.stringify(chunk));
          continue;
        }
      } catch {
        // Keep the provider's frame untouched rather than drop it.
      }
      out += sseData(raw);
    }
    return out + sseData('[DONE]');
  }
  const usage = servedUsage(answer, requestBody);
  const delta: Json = { role: 'assistant', content: answer.text };
  if (answer.toolCalls.length > 0) {
    delta.tool_calls = answer.toolCalls.map((call, i) => ({
      index: i,
      id: call.id !== '' ? call.id : `call_${i}`,
      type: 'function',
      function: { name: call.name, arguments: call.arguments },
    }));
  }
  let out = sseData(JSON.stringify(openAIChunk(answer, modelId, delta, null)));
  out += sseData(JSON.stringify(openAIChunk(answer, modelId, {}, openAIStopReason(answer))));
  const usageChunk = openAIChunk(answer, modelId, {}, null, {
    prompt_tokens: usage.input,
    completion_tokens: usage.output,
    total_tokens: usage.input + usage.output,
  });
  usageChunk.choices = [];
  out += sseData(JSON.stringify(usageChunk));
  return out + sseData('[DONE]');
}

/** The Anthropic event stream for an answer, built from its reassembled parts. */
function anthropicStreamAnswer(answer: Assembled, modelId: string, requestBody: Json): string {
  const usage = servedUsage(answer, requestBody);
  const providerId = (answer.providerId ?? 'aggregator').replace(/[^A-Za-z0-9_-]/g, '');
  let out = sseEvent('message_start', {
    type: 'message_start',
    message: {
      id: `msg_${providerId}`,
      type: 'message',
      role: 'assistant',
      model: modelId,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: usage.input, output_tokens: 0 },
    },
  });
  let index = 0;
  if (answer.text !== '' || answer.toolCalls.length === 0) {
    out += sseEvent('content_block_start', { type: 'content_block_start', index, content_block: { type: 'text', text: '' } });
    const pieces = answer.textPieces.length > 0 ? answer.textPieces : [answer.text];
    for (const piece of pieces) {
      if (piece !== '') out += sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'text_delta', text: piece } });
    }
    out += sseEvent('content_block_stop', { type: 'content_block_stop', index });
    index += 1;
  }
  for (const call of answer.toolCalls) {
    out += sseEvent('content_block_start', {
      type: 'content_block_start',
      index,
      content_block: { type: 'tool_use', id: call.id !== '' ? call.id : `toolu_${index}`, name: call.name, input: {} },
    });
    if (call.arguments !== '') {
      out += sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: call.arguments } });
    }
    out += sseEvent('content_block_stop', { type: 'content_block_stop', index });
    index += 1;
  }
  out += sseEvent('message_delta', {
    type: 'message_delta',
    delta: { stop_reason: anthropicStopReason(answer), stop_sequence: null },
    usage: { output_tokens: usage.output },
  });
  out += sseEvent('message_stop', { type: 'message_stop' });
  return out;
}

// ---------------------------------------------------------------------------
// HTTP surface
// ---------------------------------------------------------------------------

function sendJson(res: ServerResponse, status: number, body: unknown, headers: Record<string, string> = {}): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text), ...headers });
  res.end(text);
}

function sendStream(res: ServerResponse, text: string, headers: Record<string, string> = {}): void {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', ...headers });
  res.end(text);
}

function openAIErrorBody(message: string, type: string, code: string | null): Json {
  return { error: { message, type, param: null, code } };
}

function anthropicErrorBody(message: string, type: string): Json {
  return { type: 'error', error: { type, message } };
}

/** The trail header for a response whose prompt saw upstream failures, if any. */
function trailHeaders(failures: FailureRow[]): Record<string, string> {
  if (failures.length === 0) return {};
  return { 'x-makefun-upstream-failures': JSON.stringify(failures.slice(0, MAX_TRAIL_ROWS)) };
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const free = brief.models.filter((model) => model.free);
  if (typeof req.headers['x-api-key'] === 'string') {
    sendJson(res, 200, {
      data: free.map((model) => ({
        type: 'model',
        id: model.id,
        display_name: model.displayName,
        created_at: '2026-01-01T00:00:00Z',
      })),
      has_more: false,
      first_id: free.length > 0 ? free[0].id : null,
      last_id: free.length > 0 ? free[free.length - 1].id : null,
    });
    return;
  }
  sendJson(res, 200, {
    object: 'list',
    data: free.map((model) => ({ id: model.id, object: 'model', created: 0, owned_by: 'openrouter' })),
  });
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** Answer one completion request on either wire, JSON or streamed as it asked. */
async function completions(wire: 'openai' | 'anthropic', req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    if (wire === 'anthropic') sendJson(res, 400, anthropicErrorBody('the request body is not one JSON object', 'invalid_request_error'));
    else sendJson(res, 400, openAIErrorBody('the request body is not one JSON object', 'invalid_request_error', null));
    return;
  }
  const served = await serve(wire, body);
  const headers = trailHeaders(served.failures);
  if (served.lastRefusal !== null) {
    const status = served.lastRefusal.status >= 400 && served.lastRefusal.status < 600 ? served.lastRefusal.status : 502;
    const message = `no free model could serve this request (last upstream status ${served.lastRefusal.status}${served.lastRefusal.code === '' ? '' : `, code ${served.lastRefusal.code}`})`;
    if (wire === 'anthropic') sendJson(res, status, anthropicErrorBody(message, 'api_error'), headers);
    else sendJson(res, status, openAIErrorBody(message, 'upstream_error', served.lastRefusal.code || null), headers);
    return;
  }
  const stream = body.stream === true;
  if (wire === 'openai') {
    if (stream) sendStream(res, openAIStreamAnswer(served.answer, served.model.id, body), headers);
    else sendJson(res, 200, openAIJsonAnswer(served.answer, served.model.id, body), headers);
  } else {
    if (stream) sendStream(res, anthropicStreamAnswer(served.answer, served.model.id, body), headers);
    else sendJson(res, 200, anthropicJsonAnswer(served.answer, served.model.id, body), headers);
  }
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await completions('openai', req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await completions('anthropic', req, res);
    return;
  }
  if (url.pathname === '/v1/messages') sendJson(res, 404, anthropicErrorBody(`no route ${req.method ?? ''} ${url.pathname}`, 'not_found_error'));
  else sendJson(res, 404, openAIErrorBody(`no route ${req.method ?? ''} ${url.pathname}`, 'invalid_request_error', 'not_found'));
}

// ---------------------------------------------------------------------------
// Trial lifecycle: listen, register, poll /work, exit 0
// ---------------------------------------------------------------------------

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    // IPv4 explicitly: a bare listen() binds the IPv6 wildcard, which the trial pod
    // does not deliver IPv4 traffic to.
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the catalog; retried while the env is still starting. */
async function register(port: number): Promise<void> {
  let lastError: unknown = null;
  for (let i = 0; i < 40; i++) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`POST /register answered HTTP ${res.status}: ${(await res.text()).slice(0, 300)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
      return;
    } catch (err) {
      lastError = err;
      await new Promise((resolve) => setTimeout(resolve, 500));
    }
  }
  throw lastError instanceof Error ? lastError : new Error('register never answered');
}

/** Poll GET /work; exit 0 the moment it says done. */
function pollWork(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status === 200) {
          const work: unknown = await res.json();
          if (isRecord(work) && work.done === true) {
            clearInterval(timer);
            server.close();
            process.exit(0);
          }
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte, and nothing else.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendJson(res, 500, openAIErrorBody('the server failed to answer', 'internal_error', null));
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  if (address.port === 8080) throw new Error('the OS picked 8080, which the env owns: restart');
  log(`listening on 0.0.0.0:${address.port} with ${brief.models.filter((m) => m.free).length} free models`);
  await register(address.port);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  pollWork(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
