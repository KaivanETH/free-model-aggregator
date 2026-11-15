// The router: which free model serves a request, and what a provider's refusal teaches us.
//
// Ranking: every free model that meets the request's requirements (tools, vision, JSON mode, context
// length) on a provider that is not known to be spent, best quality-table rating first. A provider's
// refusal is read for what it is:
//
//   - a spent budget, a payment wall or an outage (the refusal's message says so) ends that provider for
//     the trial: it is never dialled again;
//   - any other 402/429 (a rate-limit window, a queue) is transient: the provider stays in the ranking,
//     but two refusals in a row inside one request, or three across requests, end it too;
//   - 401/403 ends the provider (its key is wrong), 400/404 ends the model, a 5xx or an unreachable
//     provider is skipped for this request.
//
// Ties on rating go to the provider with more evidence of headroom (fewer refusals so far), then the
// brief's order. Everything is a function of the brief and the answers seen so far: no clock, no
// randomness, so a replay routes identically.

import type { ClientApi, ProviderEntry, Requirement } from './brief.ts';
import type { ModelInfo } from './catalog.ts';

export interface ProviderState {
  calls: number;
  served: number;
  refusals: number;
  consecutive: number;
  dead: boolean;
  deadModels: Set<string>;
}

export interface Candidate {
  provider: ProviderEntry;
  model: ModelInfo;
}

export type Outcome = 'served' | 'terminal' | 'transient' | 'bad_model' | 'bad_key' | 'skip';

const TERMINAL = /budget|quota[^.]*(spent|exhaust|exceed|used)|spent|exhausted|credits?|payment|billing|\bis out\b|\bout of\b|outage|unavailable|disabled|suspended|deactivated|daily limit|per[- ]day|limit reached/i;

export class Router {
  readonly providers: ProviderEntry[];
  readonly models: ModelInfo[];
  readonly state = new Map<string, ProviderState>();

  constructor(providers: ProviderEntry[], models: ModelInfo[]) {
    this.providers = providers;
    this.models = models;
    for (const p of providers) this.state.set(p.id, { calls: 0, served: 0, refusals: 0, consecutive: 0, dead: false, deadModels: new Set() });
  }

  #order(id: string): number {
    return this.providers.findIndex((p) => p.id === id);
  }

  /** Whether a model can carry this request at all (capabilities and the wire's limits). */
  fits(m: ModelInfo, provider: ProviderEntry, req: Requirement, api: ClientApi, jsonMode: boolean): boolean {
    if (!m.free) return false;
    if (req.tools === true && !m.tools) return false;
    if (req.vision === true && !m.vision) return false;
    if (req.json === true && !m.json) return false;
    if (req.min_context !== undefined && (m.context_length === null || m.context_length < req.min_context)) return false;
    // The Messages wire has no JSON mode: an OpenAI JSON-mode request needs a wire that can carry it.
    if (api === 'openai' && jsonMode && provider.wire === 'anthropic') return false;
    return true;
  }

  /** Every live candidate for a request, best first. `skip` holds the providers this request has ruled out. */
  candidates(req: Requirement, api: ClientApi, jsonMode: boolean, skip: ReadonlySet<string>, only: string | null): Candidate[] {
    const out: Candidate[] = [];
    for (const m of this.models) {
      const provider = this.providers.find((p) => p.id === m.provider);
      if (provider === undefined) continue;
      const st = this.state.get(provider.id) as ProviderState;
      if (st.dead || st.deadModels.has(m.id) || skip.has(provider.id)) continue;
      if (only !== null && m.id !== only && `${m.provider}/${m.id}` !== only) continue;
      if (!this.fits(m, provider, req, api, jsonMode)) continue;
      out.push({ provider, model: m });
    }
    out.sort((a, b) => {
      if (a.model.rated !== b.model.rated) return a.model.rated ? -1 : 1;
      if (a.model.rating !== b.model.rating) return b.model.rating - a.model.rating;
      // Equal ratings: the provider with fewer calls so far (a fresh free tier is the safest call, and
      // spreading the load puts off every provider's exhaustion), then fewer refusals.
      const sa = this.state.get(a.provider.id) as ProviderState;
      const sb = this.state.get(b.provider.id) as ProviderState;
      if (sa.calls !== sb.calls) return sa.calls - sb.calls;
      if (sa.refusals !== sb.refusals) return sa.refusals - sb.refusals;
      const oa = this.#order(a.provider.id);
      const ob = this.#order(b.provider.id);
      if (oa !== ob) return oa - ob;
      return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
    });
    return out;
  }

  /** Classify one provider answer and update that provider's state. */
  record(providerId: string, modelId: string, status: number, message: string): Outcome {
    const st = this.state.get(providerId) as ProviderState;
    if (status >= 200 && status < 300) {
      st.calls += 1;
      st.served += 1;
      st.consecutive = 0;
      return 'served';
    }
    if (status === 401 || status === 403) {
      st.dead = true;
      return 'bad_key';
    }
    if (status === 400 || status === 404 || status === 422) {
      st.deadModels.add(modelId);
      return 'bad_model';
    }
    if (status === 402 || status === 429) {
      if (/concurrent/i.test(message)) return 'skip';
      st.calls += 1;
      st.refusals += 1;
      st.consecutive += 1;
      if (TERMINAL.test(message) || st.consecutive >= 3) {
        st.dead = true;
        return 'terminal';
      }
      return 'transient';
    }
    return 'skip';
  }
}
