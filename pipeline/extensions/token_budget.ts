/** Local-only OpenRouter accounting boundary, pinned to pi 1.5.2.
 * The native provider retains message/tool conversion. Both stream entry points
 * (including compaction) share this ledger; terminal events wait for accounting.
 */
import { writeFileSync, renameSync } from "node:fs";
import {
	createAssistantMessageEventStream,
	type AssistantMessage,
	type Provider,
} from "@earendil-works/pi-ai";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";

type Cost = { kind: "unknown" | "provider_total" | "estimate"; value: number | null; currency: "USD" };
const unknownCost = (): Cost => ({ kind: "unknown", value: null, currency: "USD" });
const tokenCount = (value: unknown): value is number => Number.isSafeInteger(value) && (value as number) >= 0;
const money = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value) && value >= 0;

type RawUsage = { prompt_tokens: number; completion_tokens: number; cost?: unknown };
function validUsage(value: unknown): value is RawUsage {
	return value !== null && typeof value === "object"
		&& "prompt_tokens" in value && "completion_tokens" in value
		&& tokenCount(value.prompt_tokens) && tokenCount(value.completion_tokens)
		&& Number.isSafeInteger(value.prompt_tokens + value.completion_tokens);
}
export class BudgetLedger {
	readonly usage;
	queue = Promise.resolve();
	constructor(readonly path: string, limit: number,
		readonly publish: (event: string, usage: unknown) => void) {
		if (!Number.isSafeInteger(limit) || limit <= 0) throw new Error("invalid token budget");
		this.usage = {
			schema: "evaluator-pipeline/usage/v1", ready: false, token_budget: limit,
			total_tokens: 0, prompt_tokens: 0, completion_tokens: 0, requests: 0,
			usage_reliable: true, stop_reason: null as null | "token_limit" | "usage_unknown",
			cost: unknownCost(),
		};
	}
	persist(event: string) {
		writeFileSync(this.path + ".tmp", JSON.stringify(this.usage) + "\n", { mode: 0o600 });
		renameSync(this.path + ".tmp", this.path);
		this.publish(event, this.usage);
	}
}

/** Wrap native transport, not session totals: cache/reasoning are never added twice. */
export function budgetProvider(provider: Provider, ledger: BudgetLedger): Provider {
	const wrap = (native: Provider["streamSimple"]): Provider["streamSimple"] => (model, context, options = {}) => {
		const result = createAssistantMessageEventStream();
		const previous = ledger.queue;
		const { promise, resolve: release } = Promise.withResolvers<void>();
		ledger.queue = promise;
		const empty = (): AssistantMessage => ({
			role: "assistant", api: model.api, provider: model.provider, model: model.id,
			content: [], timestamp: Date.now(), stopReason: "error",
			usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
		});
		(async () => {
			await previous;
			let message = empty();
			let started = false;
			const previousCost = ledger.usage.cost;
			try {
				if (ledger.usage.stop_reason) throw new Error(ledger.usage.stop_reason);
				if (model.provider !== "openrouter" || model.api !== "openai-completions") {
					throw new Error("local budget provider requires OpenRouter chat completions");
				}
				ledger.usage.requests++;
				ledger.usage.usage_reliable = false; // A killed/incomplete request stays unknown on disk.
				ledger.usage.cost = unknownCost();
				started = true;
				ledger.persist("request_started");
				let rawUsage: RawUsage | undefined;
				let invalidUsage = false;
				let sawDone = false;
				let fetchCalls = 0;
				const transport = options.fetch ?? globalThis.fetch;
				const upstream = native(model, context, {
					...options,
					maxRetries: 0,
					fetch: async (input, init) => {
						if (++fetchCalls !== 1) throw new Error("hidden transport retry forbidden");
						const response = await transport(input, init);
						if (!response.body) return response;
						const decoder = new TextDecoder();
						let pending = "";
						const inspect = (text: string) => {
							pending += text;
							const lines = pending.split(/\r?\n/);
							pending = lines.pop()!;
							for (const line of lines) if (/^data:\s*\[DONE\]\s*$/.test(line)) sawDone = true;
						};
						const body = response.body.pipeThrough(new TransformStream<Uint8Array, Uint8Array>({
							transform(chunk, controller) { inspect(decoder.decode(chunk, { stream: true })); controller.enqueue(chunk); },
							flush() { inspect(decoder.decode() + "\n"); },
						}));
						return new Response(body, { status: response.status, statusText: response.statusText, headers: response.headers });
					},
					onPayload: async (payload, selectedModel) => {
						const changed = await options.onPayload?.(payload, selectedModel);
						const outgoing = (changed ?? payload) as Record<string, unknown>;
						return { ...outgoing, stream_options: { ...(outgoing.stream_options as object), include_usage: true } };
					},
					onProviderStreamEvent: async (event, selectedModel) => {
						if (event !== null && typeof event === "object" && "usage" in event && event.usage != null) {
							if (!validUsage(event.usage)) invalidUsage = true;
							else rawUsage = event.usage;
						}
						await options.onProviderStreamEvent?.(event, selectedModel);
					},
				});
				let terminal: "done" | "error" | undefined;
				for await (const event of upstream) {
					if (event.type === "done" || event.type === "error") {
						message = event.type === "done" ? event.message : event.error;
						terminal = event.type;
					} else result.push(event);
				}
				if (terminal !== "done" || !sawDone || invalidUsage || !rawUsage) {
					throw new Error(message.errorMessage || "missing, malformed or interrupted provider usage");
				}
				ledger.usage.prompt_tokens += rawUsage.prompt_tokens;
				ledger.usage.completion_tokens += rawUsage.completion_tokens;
				ledger.usage.total_tokens = ledger.usage.prompt_tokens + ledger.usage.completion_tokens;
				if (!Number.isSafeInteger(ledger.usage.total_tokens)) throw new Error("token total overflow");
				ledger.usage.usage_reliable = true;
				let cost = unknownCost();
				if (money(rawUsage.cost)) cost = { kind: "provider_total", value: rawUsage.cost, currency: "USD" };
				// Catalog zeros can mean missing prices. Do not manufacture a free request.
				else if (money(message.usage.cost.total) && message.usage.cost.total > 0) {
					cost = { kind: "estimate", value: message.usage.cost.total, currency: "USD" };
				}
				const old = previousCost;
				ledger.usage.cost = ledger.usage.requests === 1 ? cost
					: old.value === null || cost.value === null ? unknownCost()
					: { kind: old.kind === "provider_total" && cost.kind === "provider_total" ? "provider_total" : "estimate",
						value: old.value + cost.value, currency: "USD" };
				if (ledger.usage.total_tokens >= ledger.usage.token_budget) ledger.usage.stop_reason = "token_limit";
				ledger.persist("request_complete");
				result.push({ type: "done", reason: message.stopReason as "stop" | "length" | "toolUse", message });
				result.end();
			} catch (error) {
				if (started) {
					ledger.usage.usage_reliable = false;
					ledger.usage.stop_reason = "usage_unknown";
					ledger.usage.cost = unknownCost();
				}
				try {
					ledger.persist(started ? "usage_unknown" : "request_blocked");
				} catch {
					// Persistence failure cannot permit another request or a successful completion.
					ledger.usage.usage_reliable = false;
					ledger.usage.stop_reason = "usage_unknown";
					ledger.publish("usage_unknown", ledger.usage);
				}
				message.stopReason = "error";
				message.errorMessage = `pipeline budget: ${ledger.usage.stop_reason ?? String(error)}`;
				result.push({ type: "error", reason: "error", error: message });
				result.end();
			} finally { release(); }
		})();
		return result;
	};
	return {
		...provider,
		stream: wrap(provider.stream.bind(provider)) as Provider["stream"],
		streamSimple: wrap(provider.streamSimple.bind(provider)),
	};
}

export default function (pi: ExtensionAPI) {
	if (process.env.PIPELINE_LOCAL !== "1") throw new Error("token budget extension is local-only");
	const path = process.env.PIPELINE_USAGE_PATH;
	if (!path) throw new Error("PIPELINE_USAGE_PATH is required");
	let sessionContext: ExtensionContext | undefined;
	const ledger = new BudgetLedger(path, Number(process.env.PIPELINE_TOKEN_BUDGET), (event, usage) => {
		if (!sessionContext) throw new Error("budget RPC context unavailable");
		// pi redirects extension stdout to stderr; notify uses the supported RPC channel.
		sessionContext.ui.notify("PIPELINE_BUDGET:" + JSON.stringify({ event, usage }), "info");
	});
	pi.on("session_start", (_event, ctx) => {
		sessionContext = ctx;
		if (ctx.model?.provider !== "openrouter" || ctx.model.api !== "openai-completions") {
			throw new Error("local token accounting requires OpenRouter chat completions");
		}
		const provider = ctx.modelRegistry.getProvider("openrouter");
		if (!provider) throw new Error("OpenRouter provider unavailable");
		pi.registerProvider(budgetProvider(provider, ledger));
		ledger.usage.ready = true;
		ledger.persist("ready");
	});
}
