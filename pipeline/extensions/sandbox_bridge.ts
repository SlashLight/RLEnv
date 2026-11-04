/**
 * Pipeline sandbox bridge.
 *
 * Replaces every file and shell tool with an HTTP call into the trusted runner,
 * which executes the operation inside the one-shot tool container. There is no
 * local fallback: without gateway configuration the extension refuses to load.
 */

import type {
	BashOperations,
	EditOperations,
	ExtensionAPI,
	ReadOperations,
	WriteOperations,
} from "@earendil-works/pi-coding-agent";
import {
	createBashTool,
	createEditTool,
	createReadTool,
	createWriteTool,
} from "@earendil-works/pi-coding-agent";

const WORKSPACE = "/workspace";
const gatewayUrl = process.env.PIPELINE_GATEWAY_URL;
const gatewayToken = process.env.PIPELINE_GATEWAY_TOKEN;

if (!gatewayUrl || !gatewayToken) {
	throw new Error("pipeline sandbox bridge: PIPELINE_GATEWAY_URL and PIPELINE_GATEWAY_TOKEN are required");
}

const hostCwd = process.cwd();

/** pi resolves tool paths against its own cwd; the agent's cwd is /workspace. */
function toContainerPath(path: string): string {
	if (path === hostCwd) return WORKSPACE;
	if (path.startsWith(`${hostCwd}/`)) return `${WORKSPACE}${path.slice(hostCwd.length)}`;
	return path;
}

async function callGateway(
	op: string,
	body: Record<string, unknown>,
	timeoutMs: number,
	signal?: AbortSignal,
): Promise<Record<string, unknown>> {
	const response = await fetch(`${gatewayUrl}/tool/${op}`, {
		method: "POST",
		headers: { "Content-Type": "application/json", "X-Pipeline-Token": gatewayToken },
		body: JSON.stringify(body),
		signal,
		timeout: timeoutMs,
	});
	const payload = (await response.json()) as Record<string, unknown>;
	if (!response.ok) {
		throw new Error(String(payload.error ?? `tool gateway error ${response.status}`));
	}
	return payload;
}

function readOps(): ReadOperations {
	return {
		readFile: async (path) => {
			const payload = await callGateway("read", { path: toContainerPath(path) }, 30_000);
			return Buffer.from(String(payload.data), "base64");
		},
		access: async (path) => {
			await callGateway("access", { path: toContainerPath(path) }, 30_000);
		},
		detectImageMimeType: async (path) => {
			const payload = await callGateway("mimetype", { path: toContainerPath(path) }, 30_000);
			return (payload.mime as string | null) ?? null;
		},
	};
}

function writeOps(): WriteOperations {
	return {
		writeFile: async (path, content) => {
			const data = Buffer.from(content).toString("base64");
			await callGateway("write", { path: toContainerPath(path), data }, 60_000);
		},
		mkdir: async (path) => {
			await callGateway("mkdir", { path: toContainerPath(path) }, 30_000);
		},
	};
}

function editOps(): EditOperations {
	const read = readOps();
	const write = writeOps();
	return { readFile: read.readFile, access: read.access, writeFile: write.writeFile };
}

function bashOps(): BashOperations {
	return {
		exec: async (command, cwd, { onData, signal, timeout }) => {
			const budget = timeout ?? 60;
			const target = toContainerPath(cwd);
			const containerCwd = target === WORKSPACE || target.startsWith(`${WORKSPACE}/`)
				? target
				: WORKSPACE;
			const payload = await callGateway(
				"bash",
				{ command, cwd: containerCwd, timeout: budget },
				(budget + 20) * 1000,
				signal,
			);
			const output = String(payload.output ?? "");
			if (output.length > 0) {
				onData(Buffer.from(output));
			}
			if (payload.timed_out) {
				throw new Error(`command timed out after ${budget}s\n${output}`);
			}
			return { exitCode: Number(payload.exit_code ?? 1) };
		},
	};
}

export default function (pi: ExtensionAPI) {
	pi.registerTool(createReadTool(WORKSPACE, { operations: readOps() }));
	pi.registerTool(createWriteTool(WORKSPACE, { operations: writeOps() }));
	pi.registerTool(createEditTool(WORKSPACE, { operations: editOps() }));
	pi.registerTool(createBashTool(WORKSPACE, { operations: bashOps() }));
}
