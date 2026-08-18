# ADR-0014: Integrate MCP Tools Through a Governed Client Boundary

- Status: Proposed
- Date: 2026-08-18
- Extends: [ADR-0002](0002-explicit-tool-boundary-and-bounded-loop.md),
  [ADR-0003](0003-observable-runtime-and-cooperative-cancellation.md), and
  [ADR-0011](0011-govern-tool-actions-before-side-effects.md)
- Clarifies: [ADR-0012](0012-separate-workspace-containment-from-process-isolation.md) and
  [ADR-0013](0013-load-repository-resources-through-context.md)
- Detailed design: [Phase 10 Detailed Design](../phase-10-detailed-design.md)

## Context

Phase 10 must let DQAgent discover and invoke external capabilities through MCP without creating a
second agent loop, trusting remote metadata as policy, or bypassing the Phase 9 governed action
boundary.

MCP introduces two separate trust transitions:

1. A remote server supplies tool names, descriptions, annotations, and JSON Schemas that may become
   model-visible capabilities.
2. A model-selected invocation sends data to a process outside DQAgent's local tool implementation
   and may cause an external effect.

Neither transition is covered by JSON Schema validation alone. A valid schema can still be
oversized, hostile, misleading, or incompatible with the model provider. A valid tool call can
disclose data or mutate remote state. A transport timeout or disconnect after request transmission
cannot prove that the server performed no effect.

The current governed action implementation is intentionally workspace-oriented. Its fixed guards,
prepared action, workspace identity, containment checks, and effect classes were sufficient when
workspace actions were the only concrete effect domain. MCP is now a second real domain, so a
capability-neutral governed orchestration boundary is justified. Workspace-specific checks must
remain intact, while remote actions receive their own trusted configuration, guards, effect
classification, revalidation, and evidence.

The current MCP specification revision is `2026-07-28`. It uses stateless JSON-RPC requests:
each request carries the protocol version and the caller's capabilities, and modern servers expose
`server/discover` instead of the legacy `initialize` session handshake. The protocol is stateless,
but a stdio server process and its transport streams still have an application-owned lifecycle.

DQAgent should not hand-roll the complete JSON-RPC and stdio protocol implementation. The official
Python SDK v2 supports the current revision and remains a transport/protocol dependency rather than
an application architecture.

## Decision

DQAgent will add an application-owned MCP client boundary that discovers a bounded, immutable tool
catalog during application composition and adapts allowed remote tools into the existing
`ToolRegistry`. Model-selected invocations will pass through the same semantic governance order as
local effectful actions before the MCP client sends a request.

The first implementation supports only:

- MCP client behavior, not an MCP server;
- the `2026-07-28` protocol revision;
- one explicitly configured stdio server for one application composition;
- `server/discover`, paginated `tools/list`, and `tools/call`;
- sequential calls through the existing sequential agent loop;
- text and bounded JSON-compatible structured tool results.

Streamable HTTP, OAuth, legacy revision fallback, resources, prompts, subscriptions,
server-initiated sampling or elicitation, roots, tasks, multi-round-trip `input_required`, binary
media, and MCP server implementation remain outside this initial contract.

### Dependency direction

The intended dependency direction is:

```text
trusted application composition
    -> MCPServerConfig
    -> MCPClient / MCPTransport application ports
    -> official MCP Python SDK v2 adapter
    -> managed stdio server process

MCPClient
    -> bounded MCPToolCatalog
    -> MCP tool adapters
    -> ToolRegistry
    -> existing AgentRuntime

model ToolCall
    -> ToolRegistry
    -> common governed orchestration
    -> MCP-specific preparation, guards, policy, approval, and revalidation
    -> MCPClient.call_tool
    -> bounded ToolResult + structured action evidence
```

`AgentRuntime` remains unaware of MCP, transport types, server configuration, and SDK values. It
continues to receive provider-neutral `ToolDefinition`, `ToolCall`, and `ToolResult` values.
Official SDK types remain inside the MCP adapter package and cannot appear in application ports,
runtime models, policy contracts, events, or evaluation case schemas.

The official `mcp` Python SDK v2 will be added as a bounded dependency during implementation.
DQAgent owns protocol revision acceptance, catalog limits, naming, authority, retries, result
projection, and failure mapping even when the SDK owns wire framing and transport mechanics.
SDK adoption is contingent on enforceable inbound and outbound raw-message limits. If the SDK does
not expose a pre-parse bound, the adapter must add a bounded stream layer rather than accepting an
unbounded frame before DQAgent validation.

### Trusted server configuration

An MCP server is available only through an immutable `MCPServerConfig` supplied by the trusted
composition root. The configuration owns:

- a stable, provider-compatible `server_id`;
- a direct command and argument vector with no shell interpretation;
- a minimal allowlisted environment and secret references;
- the required protocol revision;
- an explicit allowlist of remote tool names;
- trusted effect classifications and optional policy overrides for those tools;
- connection, discovery, call, output, pagination, schema, and cleanup limits.

Model input, repository instructions, memory, retrieval content, MCP metadata, and MCP results
cannot select the server command, change its environment, add credentials, broaden the tool
allowlist, alter effect classification, relax limits, or modify policy.

The server's self-reported name, version, website, icons, title, and annotations are display
metadata only. They never replace the configured `server_id` or become an authorization identity.
Server instructions and extension metadata are not projected into model context, persisted in a
session, or treated as host policy in the initial implementation. Adding them later must reuse
ADR-0013's provenance, budget, and lower-authority projection contract.

The initial stdio profile has no protocol-level authentication. Any credential needed by a trusted
stdio server is supplied only through explicitly allowlisted host configuration and is omitted from
model-visible schemas, arguments, results, events, and diagnostics. HTTP authentication and OAuth
require a later amendment because they add endpoint trust, token storage, redirect, issuer, and
cross-origin behavior.

### Modern protocol discovery and process lifecycle

Application composition starts the configured stdio process, performs `server/discover`, and
requires an exact `2026-07-28` response before exposing any remote tool. DQAgent does not continue
with the legacy `initialize` flow when modern discovery fails or returns an unsupported revision.

Although modern requests are protocol-stateless, DQAgent owns one stdio process and transport for
the application composition. The process may serve multiple sequential tool calls and therefore
may retain implementation state outside the MCP protocol contract. DQAgent does not treat protocol
statelessness as process, tenant, or data isolation.

The stdio adapter:

- starts a direct argument vector without a shell;
- provides only the configured minimal environment;
- reserves stdout for MCP frames;
- enforces raw inbound and outbound message limits before or during protocol decoding;
- drains bounded stderr without projecting raw stderr into model context;
- serializes requests because the current runtime invokes tools sequentially;
- closes streams and performs bounded direct-child termination and reaping on shutdown or an
  unrecoverable protocol failure.

This transport cannot reuse the Phase 9 one-shot `LocalSubprocessRunner` because MCP requires a
long-lived bidirectional stream. It must, however, preserve ADR-0012's capability honesty, secret
handling, bounded output, cancellation, and cleanup principles.

The stdio server remains a normal process under the current user account. DQAgent does not claim
host filesystem, network, credential, syscall, descendant-process, or external-service isolation.
User approval cannot create those technical guarantees.

### Bounded immutable tool catalog

After discovery, DQAgent follows `tools/list` pagination under explicit limits for page count, total
tool count, total response bytes, per-tool metadata, and aggregate catalog size. Repeated cursors,
duplicate remote names, malformed pages, unsupported capability changes, or limit exhaustion fail
the entire server catalog closed. DQAgent does not expose a partial catalog whose omissions could
change model behavior invisibly.

Every admitted tool preserves content-free provenance:

- configured server ID;
- exact remote tool name;
- local tool name;
- protocol revision;
- canonical input-schema digest;
- optional output-schema digest;
- trusted effect classification;
- discovery generation identity.

Input schemas are untrusted protocol data. DQAgent requires a Draft 2020-12 object schema, validates
it structurally, bounds its serialized bytes, depth, node count, string sizes, and collection sizes,
and does not resolve external references. Unsupported or unsafe schema features reject that tool
before registration.

The catalog is frozen before `AgentRuntime` receives tool definitions. Tool-list change
notifications and automatic mid-run refresh are unsupported. A refresh requires a new application
composition or an explicit future lifecycle that produces a new catalog generation. An active run
never changes its tool definitions after the model has observed them.

### Stable provider-compatible names

The local model-visible name is derived from the trusted server ID and exact remote name rather than
from a server title:

```text
mcp_<server_id>_<normalized_remote_name>[_<digest>]
```

The projection uses only ASCII letters, digits, underscores, and hyphens and is at most 64
characters so it remains compatible with the current model-provider boundary. If normalization or
truncation can lose identity, the name includes a stable short SHA-256 suffix over the configured
server ID and exact remote name.

The exact remote name remains in internal provenance and is sent back to the server; the normalized
name is only the local model-facing identity. Any collision with a local tool, a Phase 9 reserved
tool, another MCP tool, or another normalized projection fails composition. A remote tool never
silently replaces an existing registration.

### Explicit tool admission and effect classification

Discovery does not grant model reachability. DQAgent exposes only the intersection of:

1. tools returned by the configured server;
2. exact remote names present in the trusted allowlist;
3. tools whose bounded schema and metadata pass validation.

MCP annotations such as read-only, destructive, idempotent, or open-world hints are retained only as
untrusted diagnostics. They do not select policy, approval, retries, or effect state.

Trusted composition classifies each admitted tool as:

- `external_read`: sends bounded arguments to the server but is not expected to mutate remote state;
- `external_mutation`: may change remote state;
- `external_unknown`: semantics are not sufficiently known.

Every MCP call is an external disclosure because its arguments leave DQAgent's local tool boundary.
The default MCP policy therefore requires foreground exact-action approval for all admitted tools.
A trusted composition may explicitly allow one named `external_read` tool without approval.
`external_mutation` and `external_unknown` remain approval-required in the initial implementation.
Policy denial, missing classification, or missing approval fails before request transmission.

### Common governed orchestration

DQAgent will generalize the Phase 9 governed tool implementation only at the orchestration boundary
now that workspace and MCP are two concrete effect domains.

A common governed tool contract will retain the fixed semantic order:

```text
bounded parse and schema validation
    -> immutable domain-specific prepared action
    -> non-overridable domain guards
    -> trusted policy: allow | deny | require_approval
    -> exact-action approval when required
    -> required pre-hooks
    -> effect-boundary revalidation
    -> execute at most once
    -> post-hooks and independent bounded evidence
    -> bounded model observation
```

Workspace actions keep their existing `PreparedAction`, workspace guards, policy behavior, approval
binding, and action records. MCP adds a domain-specific prepared remote action and remote guards.
The shared abstraction coordinates order and evidence; it does not replace domain-specific
invariants with one generic list of optional checks.

`ToolRegistry` will dispatch through a governed-tool protocol instead of recognizing only the
concrete workspace `ActionTool` class. Legacy local `Tool` behavior remains available for genuinely
in-process, non-governed handlers. MCP tools cannot register through that legacy path.

An immutable prepared MCP action binds:

- configured server ID and catalog generation;
- exact remote and local tool names;
- protocol revision;
- input-schema digest;
- trusted effect classification;
- canonical JSON arguments and their digest;
- effective input, output, duration, and transport limits.

Approval presentation is bounded and sanitized. Action records retain digests, classifications,
outcomes, timing, and content-free diagnostics rather than raw arguments, credentials, descriptions,
or results.

Immediately before request transmission, the adapter revalidates the active server binding,
connection health, protocol revision, catalog generation, remote name, schema digest, trusted
classification, and effective limits. Drift fails closed and requires fresh preparation and
approval.

### Retry, cancellation, and effect evidence

DQAgent sends a model-selected MCP tool invocation at most once. `AgentRuntime` and the MCP adapter
do not automatically retry `tools/call`, including calls classified as read-only. A future retry
policy requires explicit idempotency and reconciliation evidence owned by trusted composition, not
an MCP annotation.

Remote effect evidence follows these rules:

- If preparation, policy, approval, pre-hooks, or revalidation fails before request transmission,
  effect state is `none`.
- Once any request bytes may have been transmitted, timeout, cancellation, EOF, malformed response,
  transport loss, or cleanup failure yields `unknown` unless stronger direct evidence exists.
- A valid terminal MCP response yields `complete` as transport-level completion evidence only. It
  does not independently prove that an external mutation is correct, durable, or truthful.
- MCP `isError=true` is a model-visible tool failure but does not prove that no remote effect
  occurred.

`RunContext` deadline and cancellation remain the caller contract. The adapter cancels the waiting
SDK operation and uses protocol cancellation where supported. If the server does not terminate the
operation, DQAgent may close the unhealthy transport and reap the direct stdio process. None of
these actions claims rollback of remote state, child-process effects, or network operations already
performed.

### Tool-result projection

MCP tool results are untrusted external data. The initial text-only DQAgent path supports:

- bounded text content blocks;
- bounded JSON-compatible `structuredContent`, rendered deterministically;
- optional validation against a bounded admitted output schema.

Binary images, audio, embedded resource bodies, resource subscriptions, and arbitrary extension
payloads are not projected into model context in the initial implementation. Unsupported blocks are
reported through bounded omission evidence; they are not silently decoded or persisted.

Result collection has byte, character, item, nesting, and elapsed-time limits before materializing
an unbounded Python structure or model-visible string. Secret redaction is defense in depth and does
not justify sending a secret to a remote server. Authentication values never enter model-supplied
tool arguments.

An invalid output schema, malformed result, unsupported mandatory content, or failed result
sanitization becomes a typed observation or protocol failure. The model receives a stable bounded
recovery observation, while internal evidence retains the separate transport and effect state.

The client does not install sampling, elicitation, roots, or automatic `input_required` handlers.
If a server requests a multi-round-trip continuation, the call fails as an unsupported capability;
DQAgent does not let the SDK automatically ask the user, call the model, or grant more context.

### Events and evaluation

MCP adds structured stage events for server start, discovery, catalog completion/failure, remote
call start/completion/failure, transport cancellation, and cleanup. Event attributes contain only
stable server IDs or digests, local tool names, protocol revision, catalog generation, bounded
counts, durations, outcomes, truncation flags, and reason codes. They exclude commands, host paths,
environment values, credentials, raw schemas, arguments, descriptions, stderr, and tool results.

Deterministic evaluation remains above production execution under ADR-0004. Phase 10 will add a
versioned fixture suite that uses the production MCP client ports, catalog adapter, `ToolRegistry`,
governed orchestration, events, and result projection. Most cases use a scripted transport; a
controlled stdio fixture validates real framing, process lifecycle, cancellation, and cleanup.

Representative coverage includes:

- exact modern discovery and unsupported revision;
- pagination, repeated cursor, duplicate names, catalog bounds, and atomic failure;
- provider-compatible names, truncation digest, and local/remote collisions;
- invalid, hostile, oversized, or externally referenced schemas;
- explicit allowlist and untrusted-annotation non-authority;
- approval allow, reject, unavailable, and catalog drift;
- success, `isError`, malformed result, output-schema mismatch, and result truncation;
- timeout/cancellation before send and unknown effect after possible send;
- server exit, stderr bounds, cleanup failure, and no secret leakage;
- unsupported `input_required`, binary content, resources, prompts, and legacy fallback;
- preservation of existing Phase 3 through Phase 9 deterministic baselines.

## Consequences

- MCP joins the existing provider-neutral tool pool and model/tool loop instead of creating a second
  runtime.
- The official SDK reduces protocol and framing risk, while DQAgent still owns security, limits,
  naming, policy, evidence, and application failure semantics.
- Dynamic remote discovery becomes deterministic application composition: only a complete bounded
  catalog is exposed, and it remains stable for an active run.
- The governed action implementation gains a real capability-neutral orchestration abstraction only
  after two concrete domains exist. Workspace safety rules remain domain-specific and mandatory.
- Remote calls are more restrictive than local demonstrations because every call discloses data and
  remote annotations cannot establish trust.
- Exact approval binds the known request and catalog identity but cannot prove the server's actual
  implementation or external-state result.
- Stdio provides a useful credential-free local integration substrate, but the server process is not
  a sandbox and may retain hidden state across calls.
- The synchronous DQAgent tool boundary requires a bounded bridge to the SDK's asynchronous client.
  That bridge owns one event loop/process lifecycle and must not call nested `asyncio.run` from tool
  handlers.
- Modern-only revision support reduces compatibility but keeps the first implementation aligned with
  the current stateless protocol. Legacy support requires separate tests and an explicit amendment.
- Resources, prompts, HTTP, OAuth, subscriptions, tasks, binary content, and an MCP server remain
  unsupported until a measured use case justifies their additional trust and lifecycle contracts.

## Acceptance Conditions

This ADR may move from Proposed to Accepted only when:

- the official SDK is isolated behind provider-neutral MCP ports;
- one controlled stdio server can be discovered, namespaced, admitted, invoked, cancelled, and
  cleaned up through the production tool path;
- all admitted MCP tools use the common governed orchestration and no MCP tool can register through
  the legacy direct-handler path;
- trusted configuration, schema/catalog/result bounds, exact approval, no automatic retry, and
  none/complete/unknown effect semantics have focused regressions;
- the Phase 10 deterministic evaluation suite and accepted baseline exercise the production path;
- Ruff, strict mypy, full pytest, compile checks, existing deterministic evaluations, credential
  scans, and documentation consistency gates pass;
- README, architecture, roadmap, evaluation documentation, and a source-reading comparison describe
  the implemented boundary and residual risks without claiming HTTP, legacy, or isolation support.

## Alternatives Considered

### Hand-roll JSON-RPC, stdio framing, and MCP types

Rejected because protocol mechanics are not DQAgent's differentiating architecture and would add
framing, cancellation, compatibility, and parser risk. The official SDK is kept behind a narrow
adapter so its API does not become the application boundary.

### Expose official SDK types throughout the runtime

Rejected because provider and protocol SDK values would leak into `AgentRuntime`, policy, events,
tests, and evaluation schemas. This would repeat the provider-coupling problem avoided by ADR-0001.

### Register discovered MCP tools as legacy `Tool` handlers

Rejected because remote calls disclose data and may produce effects. Direct handlers would bypass
trusted admission, exact approval, at-most-once execution evidence, unknown-effect handling, and
governed call limits.

### Trust MCP annotations for read-only or idempotent policy

Rejected because annotations are server-controlled hints. Letting them select approval or retry
would allow the remote capability provider to authorize itself.

### Put every remote tool through workspace-specific `ActionTool` fields

Rejected because synthetic workspace paths and containment checks would misrepresent the remote
effect domain. The shared abstraction is orchestration order; remote and workspace actions retain
different prepared identities and guards.

### Automatically fall back to legacy initialize-based revisions

Deferred because dual-era lifecycle, capability, cancellation, and conformance behavior would expand
the first compatibility matrix without improving the modern Phase 10 learning outcome.

### Support stdio and Streamable HTTP together

Deferred because HTTP introduces endpoint trust, origin validation, TLS, authentication, OAuth token
lifecycle, redirects, proxies, and different connection semantics. Stdio is sufficient to validate
the client, dynamic catalog, governance, and evaluation boundaries first.

### Implement resources, prompts, and an MCP server in the first increment

Rejected because tools already provide the concrete external-capability use case. Resources and
prompts require separate context authority and persistence decisions, while a server reverses the
trust direction and is explicitly later in the roadmap.

## Specification Basis

- [MCP 2026-07-28 specification](https://modelcontextprotocol.io/specification/2026-07-28)
- [MCP 2026-07-28 architecture][mcp-architecture]
- [MCP 2026-07-28 transports][mcp-transports]
- [MCP 2026-07-28 tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)
- [Official MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [MCP Python SDK versioning](https://py.sdk.modelcontextprotocol.io/versioning/)

[mcp-architecture]: https://modelcontextprotocol.io/specification/2026-07-28/architecture
[mcp-transports]: https://modelcontextprotocol.io/specification/2026-07-28/basic/transports
