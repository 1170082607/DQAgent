# Phase 10 Detailed Design: MCP Client and Governed External Tools

- Status: Proposed implementation contract; implementation has not started
- Date: 2026-08-18
- Roadmap status: Phase 10 `In Progress`
- Scope: Phase 10 bounded v1 only
- Protocol profile: MCP `2026-07-28`, one stdio server, tools only

## 1. Purpose and Authority

This document turns the Phase 10 roadmap outcome into the smallest coherent implementation
contract. The roadmap remains the source of truth, and
[ADR-0014](adr/0014-integrate-mcp-tools-through-governed-client.md) is the normative architecture
decision.

The design must prove one complete capability:

```text
trusted stdio server configuration
    -> bounded server start and modern discovery
    -> complete immutable admitted tool catalog
    -> existing ToolRegistry and AgentRuntime
    -> governed remote tool invocation
    -> bounded result and content-free evidence
    -> bounded server cleanup
```

Implementation detail must not silently widen that path. A feature absent from the success boundary
or explicitly listed as a non-goal requires a later roadmap decision or ADR amendment.

### 1.1 V1 success boundary

The first complete Phase 10 path must let a caller:

1. supply one trusted, versioned local MCP server configuration;
2. start one direct-argv stdio server inside one foreground run;
3. require exact MCP `2026-07-28` discovery before exposing tools;
4. collect every `tools/list` page within explicit bounds;
5. admit only explicitly allowlisted tools with trusted effect classifications;
6. expose stable provider-compatible tool names through the existing `ToolRegistry`;
7. invoke an admitted tool through the existing governed execution order;
8. receive a bounded model observation and content-free action evidence;
9. observe cancellation, timeout, connection loss, unknown effect, and cleanup honestly;
10. exercise the same production path through deterministic evaluation and a small foreground CLI.

The first server fixture and deterministic model are controlled and credential-free. They validate
the harness boundary, not arbitrary third-party server safety or live-model quality.

### 1.2 Non-goals and deferred capabilities

Phase 10 v1 does not add:

- an MCP server;
- Streamable HTTP, URLs, TLS, OAuth, bearer tokens, redirects, proxies, or origin policy;
- legacy `initialize` negotiation or automatic protocol fallback;
- more than one configured MCP server per foreground run;
- dynamic server discovery, plugin directories, marketplaces, or auto-installation;
- resources, prompts, subscriptions, list-change notifications, or catalog refresh during a run;
- roots, sampling, elicitation, logging, tasks, or multi-round-trip `input_required`;
- images, audio, embedded resource bodies, arbitrary extension payloads, or multimodal projection;
- automatic retry, idempotency inference, reconciliation, rollback, or durable effect recovery;
- background servers, cross-run connection reuse, restart recovery, or durable connection state;
- session, retrieval, memory, workflow, coding, or multi-agent composition in the MCP application;
- persistent approvals, reusable grants, delegated approvers, or a general policy DSL;
- a public plugin API, generic transport framework, or generic remote-resource abstraction;
- host filesystem, network, credential, syscall, descendant-process, or tenant isolation;
- live-model MCP evaluation or production readiness claims.

The first CLI is a one-shot foreground composition. It is not added to the existing `dqagent`
session/RAG/memory path because that would create unrelated cross-phase integration requirements.

## 2. Ownership and Existing Architecture

Phase 10 extends existing boundaries without moving protocol ownership into `AgentRuntime`.

| Concern | Phase 10 owner | Existing contract retained |
| --- | --- | --- |
| Model/tool iteration and repeated-call rejection | `AgentRuntime` | Phase 3 bounded loop |
| Tool definitions, lookup, argument schema validation | `ToolRegistry` | Provider-neutral tools |
| Run lifecycle | `RunCoordinator` and `RunContext` | One lifecycle owner |
| Governance and record capacity | Governed execution behind `ToolRegistry` | ADR-0011 |
| Workspace actions and guards | Existing `ActionTool` path | Phase 9 behavior unchanged |
| MCP config, discovery, catalog, names, calls | Phase 10 MCP modules | No SDK leakage |
| Stdio process and async SDK lifecycle | MCP transport adapter | ADR-0012 capability honesty |
| Foreground exact approval | Neutral console approval provider | Existing approval contract |
| End-to-end MCP run and cleanup | `MCPAgentApplication` | Coordination above runtime |
| MCP evaluation | `MCPEvaluationRunner` | ADR-0004 production-path pattern |

Phase-specific reuse is limited:

- Phase 2/3 supplies provider-neutral tools, the single model loop, `RunContext`, lifecycle events,
  repeated-call rejection, and provider retry ownership.
- Phase 4 supplies versioned cases, deterministic fixtures, direct predicates, bounded reports, and
  the separation between tests and evaluations.
- Phase 6/7/8 supply lower-authority external-data and provenance lessons only. Context, retrieval,
  and memory services are not dependencies of the v1 MCP composition.
- Phase 9 supplies authorization order, exact foreground approval, no automatic side-effect retry,
  private bounded evidence collection, and honest process/isolation semantics.

MCP standardizes external capability transport. It does not gain planning, memory, context
authority, or action authority by being a protocol.

## 3. Focused Module Layout

The expected ownership is:

| Module | Responsibility |
| --- | --- |
| `mcp.py` | Immutable config, limits, catalog, protocol-neutral client ports, results, errors |
| `mcp_transport.py` | Official SDK v2 adapter, async worker, stdio lifecycle, raw message bounds |
| `mcp_tools.py` | Schema admission, names, MCP actions, guards, policy, result mapping |
| `mcp_application.py` | One complete foreground MCP run, record retention, cleanup evidence |
| `mcp_cli.py` | Strict config-file loading, provider composition, approval UX, bounded output |
| `mcp_evaluation.py` | Versioned fixtures, production-path runner, predicates, report |
| `mcp_evaluation_cli.py` | Credential-free deterministic evaluation entry point |
| `tools.py` | Common governed-tool dispatch and private execution driver |
| `tool_governance.py` | Shared approval scope and existing workspace governance contracts |
| `approval_cli.py` | Reusable foreground console approval provider |

These names define ownership, not a requirement for one class per concept. Private helpers remain
with their owner when splitting them would create a speculative abstraction.

Only these Phase 10 values are intended as public composition contracts:

- `MCPServerConfig`;
- `MCPToolGrant`;
- `MCPLimits`;
- `MCPRunRequest`;
- `MCPRunResult`;
- `MCPAgentApplication`;
- `create_mcp_agent_application`.

Catalog pages, SDK adapters, wire values, prepared actions, guards, action records, and evaluation
fixtures remain module-owned unless an implemented second consumer proves a public boundary.

Dependency direction:

```text
mcp_cli -> mcp_application -> mcp_tools -> mcp
                |                 |          ^
                |                 v          |
                +-----------> tools ---------+
                |                 |
                v                 v
          AgentRuntime      tool_governance
                |
                v
          provider-neutral LLMClient

mcp_transport -> official `mcp` SDK
       ^
       |
      mcp

mcp_evaluation -> mcp_application + scripted client / controlled stdio fixture
```

`runtime.py`, provider adapters, session, retrieval, memory, workflow, workspace, and coding modules
must not import the MCP SDK.

## 4. Cross-Boundary Invariants

1. Only trusted application composition selects the server config, process argv, environment names,
   tool allowlist, effect classifications, policy overrides, limits, approval provider, and SDK
   adapter.
2. Server metadata, tool descriptions, annotations, schemas, results, model output, repository
   content, retrieval, and memory cannot expand authority.
3. No tool is model-reachable until modern discovery and every catalog page complete successfully.
4. Catalog admission is atomic. A limit, cursor, duplicate, collision, or schema failure exposes no
   partial MCP catalog.
5. One active run sees one immutable catalog generation and one stable set of model-facing names.
6. Every MCP tool invocation is governed and reaches `tools/call` at most once.
7. An exact approval binds the run, server scope, catalog generation, tool identity, schema digest,
   effect classification, canonical arguments, and effective limits.
8. MCP annotations never select policy, approval, idempotency, retry, or effect state.
9. Configured authentication values are never accepted as model-supplied tool arguments.
10. Raw inbound and outbound message bounds apply before or during protocol decoding, not only after
    the SDK has materialized an arbitrary object.
11. Timeout, cancellation, EOF, malformed frames, and cleanup do not imply rollback.
12. Once request transmission may have begun, missing terminal evidence is `unknown` effect.
13. A valid MCP response proves protocol completion only; it does not verify external-state truth.
14. The stdio server is a normal local process, not a host, network, credential, or syscall sandbox.
15. `EventSink` remains best-effort telemetry and is never a security precondition.
16. The MCP application does not compose Session, RAG, Memory, Workflow, Coding, or Multi-Agent.

## 5. Trusted Configuration and Limits

### 5.1 Configuration file contract

The foreground CLI accepts one explicit `--config PATH`. The path is caller-supplied host
configuration, not model input or a repository instruction. The file is strict UTF-8 JSON:

```json
{
  "schema_version": 1,
  "server_id": "demo",
  "protocol_revision": "2026-07-28",
  "command": ["python", "-m", "trusted_mcp_server"],
  "environment_names": [],
  "tools": [
    {
      "name": "lookup",
      "effect": "external_read",
      "policy": "require_approval"
    }
  ]
}
```

The v1 schema uses `additionalProperties: false` at every object level.

Configuration rules:

- `schema_version` is exactly `1`.
- `protocol_revision` is exactly `2026-07-28`.
- `server_id` is lower-case ASCII and matches `[a-z][a-z0-9_-]{0,31}`.
- `command` is a bounded non-empty string array and is executed directly without a shell.
- `environment_names` contains bounded unique environment variable names, never inline values.
- every named environment variable must exist when composition requires it;
- `tools` is a bounded non-empty list with unique exact remote names;
- effect is one of `external_read`, `external_mutation`, or `external_unknown`;
- policy is `require_approval`, except an `external_read` grant may explicitly use `allow`;
- at most one grant may use `allow` in the v1 config;
- `external_mutation` and `external_unknown` cannot use `allow` in v1;
- tool omission is denial; there is no wildcard, regular expression, or allow-all form;
- no setting can enable an out-of-scope MCP capability.

The loader bounds file bytes before full JSON materialization and rejects BOM, duplicate JSON keys,
non-finite numbers, invalid UTF-8, and non-object roots. It does not read referenced config files,
expand templates, execute substitutions, or load secrets from the JSON body.

The provider API key and explicitly selected server environment values become configured secret
values for sanitization and exact-value disclosure guards. Their values are not retained in
`MCPServerConfig.__repr__`, events, errors, reports, or approval records.

### 5.2 Owner-specific limits

`MCPLimits` contains only Phase 10 owner limits:

| Limit group | Required fields |
| --- | --- |
| Config | file bytes, argv items/characters, environment names, grants |
| Process | start seconds, close seconds, stderr bytes, direct-child cleanup seconds |
| Protocol | inbound message bytes, outbound message bytes, request seconds |
| Discovery | page count, cursor characters, total tools, aggregate catalog bytes |
| Tool metadata | name/title/description characters, annotations bytes |
| Schema | serialized bytes, depth, node count, object entries, array items, string characters |
| Call input | raw argument bytes, canonical argument characters |
| Result | content blocks, structured nodes/depth, source bytes, rendered characters |
| Evidence | action records, diagnostics, omissions, event attributes |

Limits are immutable positive finite values validated before process start. Model arguments cannot
increase them. A server response cannot negotiate larger limits.

Initial numeric defaults are implementation configuration fixed by focused tests and evaluation
fixtures, not durable architecture. No path may use `tuple(values)`, `list(values)`, unrestricted
`json.loads`, unrestricted SDK reads, or recursive traversal before enforcing its corresponding
bound.

The existing `AgentRuntime.max_iterations` and one run-wide `max_governed_calls` continue to bound
model/tool iterations. The MCP application chooses a small fixed governed-call capacity; server
discovery calls do not consume model-action capacity.

## 6. Protocol-Neutral MCP Domain

### 6.1 Configuration and grant values

The core immutable values are conceptually:

```python
class MCPToolEffect(StrEnum):
    EXTERNAL_READ = "external_read"
    EXTERNAL_MUTATION = "external_mutation"
    EXTERNAL_UNKNOWN = "external_unknown"


class MCPToolPolicy(StrEnum):
    ALLOW = "allow"
    REQUIRE_APPROVAL = "require_approval"


@dataclass(frozen=True, slots=True)
class MCPToolGrant:
    remote_name: str
    effect: MCPToolEffect
    policy: MCPToolPolicy


@dataclass(frozen=True, slots=True)
class MCPServerConfig:
    server_id: str
    protocol_revision: str
    command: tuple[str, ...]
    environment_names: tuple[str, ...]
    grants: tuple[MCPToolGrant, ...]
    limits: MCPLimits
```

These values contain no SDK types and no resolved secret values. Environment values are supplied
separately to the transport factory through a private sanitized dependency.

### 6.2 Catalog values

One discovered remote tool becomes:

```python
@dataclass(frozen=True, slots=True)
class MCPToolDescriptor:
    server_id: str
    protocol_revision: str
    remote_name: str
    local_name: str
    description: str
    input_schema: Mapping[str, object]
    input_schema_digest: str
    output_schema: Mapping[str, object] | None
    output_schema_digest: str | None
    effect: MCPToolEffect
    policy: MCPToolPolicy
```

The descriptor preserves the exact remote name and a bounded model-facing description. The
description begins with a host-owned marker identifying an untrusted external MCP tool, then appends
the bounded server description when present. A missing server description uses a neutral fallback.

Icons, websites, server instructions, raw annotations, unknown extension fields, and raw discovery
messages are not retained. Annotations are bounded during intake and then discarded; content-free
catalog evidence may record only whether annotations were present.

`MCPToolCatalog` contains:

- configured server ID;
- exact protocol revision;
- bounded content-free server info;
- immutable admitted descriptors;
- omitted allowlisted names, if any;
- a canonical catalog-generation digest.

The generation digest covers sorted descriptor identities, schema digests, effects, policies, and
the exact protocol revision. It excludes descriptions, titles, timings, process IDs, and generated
request IDs.

`MCPToolCatalogEvidence` is a content-free result projection containing generation, admitted and
omitted counts, local names, schema digests, page count, and truncation/failure flags. It never
contains raw schemas or descriptions.

### 6.3 Client port

The synchronous application port is:

```python
class MCPClient(Protocol):
    @property
    def identity(self) -> str: ...

    def start(self, context: RunContext) -> MCPDiscoveryResult: ...

    def list_tools(
        self,
        cursor: str | None,
        context: RunContext,
    ) -> MCPToolPage: ...

    def call_tool(
        self,
        request: MCPCallRequest,
        context: RunContext,
    ) -> MCPCallResult: ...

    def close(self) -> MCPClientCloseResult: ...
```

The port is synchronous because `AgentRuntime` and `ToolRegistry` are synchronous. The SDK adapter
owns the async bridge. A scripted client implements the same port for focused tests and most
evaluation cases.

`MCPCallRequest` contains the exact remote name, immutable canonical arguments, expected catalog
generation, expected schema digest, and effective duration/result limits. It contains no local
model-facing name or approval text.

`MCPCallResult` contains only protocol-neutral bounded values:

- terminal `is_error`;
- text blocks;
- optional JSON-compatible `structured_content`;
- optional validated output-schema status;
- bounded omission evidence;
- transport completion identity.

SDK content classes, JSON-RPC envelopes, headers, and process handles do not cross this port.

### 6.4 Transmission and close evidence

Transport failures carry a content-free `MCPTransmissionState`:

- `not_sent`: no request bytes entered the server stream;
- `maybe_sent`: request transmission began but no valid terminal result exists;
- `response_received`: a complete response envelope was received, even if its content is invalid.

This state is evidence, not a retry hint.

`MCPClientCloseResult` records:

- whether startup occurred;
- whether protocol streams closed;
- whether the direct child was reaped;
- whether termination or force-kill was attempted;
- whether stderr was truncated;
- bounded reason codes and elapsed time.

It contains no process ID, command, host path, environment, stderr body, or secret.

## 7. Stdio Transport and Lifecycle

### 7.1 State machine

One client instance follows:

```text
new
  -> starting
  -> discovering
  -> ready
  -> calling -> ready
  -> closing
  -> closed

starting/discovering/calling/closing -> failed
```

Only one call may be active because the existing runtime invokes tools sequentially. Calling before
`ready`, calling after `failed`, concurrent calling, double start, or restart after close is a typed
configuration or lifecycle failure.

One `MCPAgentApplication.run` creates one fresh client instance and one fresh process. The process
may serve several sequential model-selected calls within that run, then is closed in `finally`.
Connections and hidden server state are not reused across application runs.

### 7.2 Official SDK adapter and async bridge

The implementation uses the official `mcp` Python SDK v2 only inside `mcp_transport.py`.

A dedicated worker thread owns one AnyIO/asyncio event loop, the SDK client context, and the stdio
process lifecycle. Synchronous port methods submit one coroutine and wait in short bounded intervals
while checking `RunContext`.

The implementation must not:

- call nested `asyncio.run` from a tool handler;
- create one event loop per request;
- expose SDK sessions to `ToolRegistry`;
- allow the SDK to auto-fulfill sampling, elicitation, roots, or `input_required`;
- allow automatic legacy fallback;
- inherit the complete host environment;
- let SDK logging or stderr flow into model output.

Before adding the dependency, T1 verifies that the selected SDK version can be wrapped with
enforceable raw inbound/outbound message limits. If a narrow bounded stream wrapper is not possible,
implementation stops and ADR-0014 is revisited. It must not proceed with post-parse-only limits.

### 7.3 Process start

Start behavior:

1. validate config, limits, direct argv, environment names, and secret values;
2. construct a minimal environment using existing Phase 9 environment rules where applicable;
3. start one direct child without shell interpretation;
4. reserve stdout for protocol frames;
5. drain bounded stderr concurrently;
6. confirm the SDK transport is active within the start deadline;
7. emit `MCP_SERVER_STARTED`;
8. enter modern discovery.

Spawn failure, missing executable, invalid environment, start timeout, or early child exit fails
before the model receives any MCP tool.

The transport does not reuse `LocalSubprocessRunner`: that adapter is one-shot and has no
bidirectional stdin contract. Shared environment, secret, bounded stream, cleanup, and capability
principles may be extracted only where behavior is genuinely identical.

### 7.4 Modern discovery

The client sends `server/discover` with preferred version `2026-07-28` and only the client
capabilities required by this design. It does not advertise resources, prompts, roots, sampling,
elicitation, subscriptions, tasks, or multi-round-trip support.

Discovery succeeds only when:

- the response is structurally valid;
- the selected version is exactly `2026-07-28`;
- the server advertises the tools capability;
- required identity fields are bounded and valid;
- no mandatory unsupported capability is required.

Server instructions and extension metadata are ignored. A legacy method-not-found response,
unsupported revision, or missing tools capability closes the client and fails composition.

### 7.5 Cancellation and close

During start, discovery, list, or call, `RunContext` deadline and cancellation are checked while
waiting. Cancellation requests the SDK operation to stop and uses protocol cancellation where the
SDK supports it.

If a call may have been sent, cancellation never reports `none` effect. The application records
`unknown`, closes the client, and does not continue the model loop with another remote action.

Close behavior:

1. cancel any active SDK task;
2. close protocol streams;
3. wait a bounded interval for direct-child exit;
4. terminate the direct child when still active;
5. wait a second bounded interval;
6. force-kill only the direct child when supported and still active;
7. reap the direct child;
8. stop the worker loop and join the worker thread;
9. return bounded cleanup evidence.

No step claims descendant cleanup, host isolation, rollback, or termination of external work already
started by the server.

## 8. Discovery, Schema Admission, and Naming

### 8.1 Paginated tool collection

After successful discovery, the catalog builder calls `tools/list` until `nextCursor` is absent.

Before appending a page it checks:

- page response byte evidence;
- page count;
- cursor type and character bound;
- repeated cursor;
- item count;
- aggregate tool count;
- aggregate metadata/schema bytes;
- duplicate exact remote names.

The builder consumes iterables incrementally and stops at `limit + 1`. It never materializes an
untrusted generator before checking the bound.

A failure in any page invalidates the whole catalog. Previous pages are discarded and no registry is
constructed.

### 8.2 Allowlist admission

The builder creates an exact grant map from trusted configuration. For each discovered tool:

- no matching grant means the tool is ignored and never exposed;
- more than one matching grant is a configuration failure;
- an allowlisted name missing from the completed server catalog is an atomic composition failure;
- server annotations cannot modify the matched grant;
- the grant effect and policy become the trusted descriptor values.

Requiring every configured tool to exist prevents a silently degraded catalog from changing the
model's available capability set.

### 8.3 Bounded schema subset

V1 accepts only an inlined, bounded Draft 2020-12 subset sufficient for controlled object arguments.

Allowed structural keywords:

- `$schema`, only when it identifies Draft 2020-12;
- `type`;
- `properties`;
- `required`;
- `additionalProperties`;
- `items`;
- `enum` and `const`;
- numeric minimum/maximum forms;
- `minLength` and `maxLength`;
- `minItems` and `maxItems`;
- `minProperties` and `maxProperties`;
- bounded `description`, `title`, and `default`.

V1 rejects:

- `$ref`, `$dynamicRef`, recursive anchors, and external schema loading;
- `pattern`, `patternProperties`, and unbounded regular expressions;
- `allOf`, `anyOf`, `oneOf`, `not`, `if`, `then`, and `else`;
- `unevaluatedProperties` and `unevaluatedItems`;
- `contentEncoding`, `contentMediaType`, and content schemas;
- unknown non-extension keywords outside the reviewed subset;
- non-`2020-12` explicit dialects;
- a top-level schema whose type is not `object`.

The schema walker is iterative, counts every key/value node, checks depth before descent, bounds all
strings and collections, rejects non-finite numbers, and creates a plain immutable JSON-compatible
copy. `Draft202012Validator.check_schema` runs only after structural bounds pass.

This compatibility restriction is intentional. Expanding schema support requires focused
complexity and denial-of-service tests; it is not done implicitly because the SDK accepts a schema.
An admitted output schema uses the same subset and limits.

### 8.4 Canonical schemas and arguments

Canonical JSON:

- uses sorted object keys;
- uses UTF-8 and `ensure_ascii=True`;
- rejects NaN and infinities;
- uses compact separators;
- preserves array order;
- never uses Python `repr`;
- never includes SDK types.

Input and output schema digests are lower-case SHA-256 over canonical bytes.

Model arguments are bounded as raw UTF-8 bytes before JSON parsing, must decode to one object, pass
the admitted input schema, and then become an immutable canonical mapping. The prepared action holds
the canonical mapping privately with `repr=False`; records and events retain only its digest.

### 8.5 Provider-compatible local names

Remote names must match `[A-Za-z0-9_./:-]{1,128}` before local projection. Local names use:

```text
mcp_<server_id>_<normalized_remote_name>[_<digest10>]
```

Normalization:

1. map remote ASCII letters to lower case;
2. preserve digits, underscores, and hyphens;
3. replace `.`, `/`, and `:` with `_`;
4. collapse repeated underscores;
5. trim separators from the normalized remote segment;
6. use `tool` when the normalized segment becomes empty;
7. reserve space for an optional ten-hex-character digest;
8. truncate only the normalized remote segment, never `mcp_` or the server ID.

The digest suffix is required when case folding, replacement, collapsing, trimming, or truncation
loses identity. It is SHA-256 over the exact UTF-8 tuple `(server_id, remote_name)`.

The final name must match `[A-Za-z0-9_-]{1,64}`. Any collision with local tools, Phase 9 reserved
names, another MCP descriptor, or another normalized projection fails the complete catalog.

### 8.6 Catalog freeze

After admission, descriptors are sorted by local name and frozen. The builder computes the catalog
generation and constructs the complete `ToolRegistry` before the first model request.

There is no mutation API on `MCPToolCatalog`. There is no notification listener, refresh timer, or
late registration. A fresh run starts a fresh server and creates a new generation.

## 9. Governed MCP Tool Integration

### 9.1 Behavior-preserving common driver

The current `ActionTool` implementation contains workspace preparation, guards, policy, approval,
hooks, executor order, events, output sanitization, and record collection in one concrete class.

Phase 10 extracts only the invariant orchestration into a private `_GovernedExecutionDriver`.
Before extraction, focused characterization tests freeze every existing Phase 9 ordering, event,
error, approval, hook, collector, cancellation, and effect-state contract.

The driver coordinates:

```text
reserve record capacity
    -> bounded parse and schema validation
    -> domain prepare
    -> domain hard guards
    -> domain policy
    -> exact approval when required
    -> domain revalidation
    -> domain pre-hook stage
    -> final effect-boundary revalidation
    -> executor attempt at most once
    -> domain post-hook stage
    -> bounded result and record
```

It does not define a public plugin interface, generic guard registry, generic policy DSL, or durable
journal.

Existing workspace `ActionTool` becomes one adapter to the driver and retains:

- `PreparedAction`;
- workspace `GuardContext`;
- the fixed seven workspace guards;
- `DefaultActionPolicy`;
- workspace approval/revalidation;
- existing hooks;
- `ActionRecord`;
- existing events and tool errors.

No Phase 9 event, error code, action digest, or behavior may change as an incidental MCP refactor.

### 9.2 Registry dispatch and record retention

`ToolRegistry` replaces concrete `isinstance(ActionTool)` dispatch with a private governed-tool
protocol requiring:

- one `ToolDefinition`;
- bounded `execute_detailed`;
- explicit `ToolExecutionContext`;
- one bounded governed record result.

Legacy `Tool` remains unchanged. MCP names cannot register through the legacy path.

The private run collector accepts records satisfying an internal content-free record protocol.
`CodingAgentApplication` still retains only workspace `ActionRecord`; `MCPAgentApplication` retains
only `MCPActionRecord`. No mixed public record tuple or cross-domain journal is introduced.

Collection remains synchronous, run-bound, bounded by `max_governed_calls`, and cleared exactly
once. Collector failure is observation failure and never converts an unknown effect into `none`.

### 9.3 Approval view

Phase 10 does not replace the existing workspace `ApprovalRequest` or `ApprovalDecision` fields and
constructors. Instead, the common driver and foreground provider consume an internal read-only view:

```python
class ApprovalDecisionView(Protocol):
    outcome: str
    run_id: str
    scope_kind: str
    scope_id: str
    action_digest: str
    provider_identity: str


class ApprovalRequestView(Protocol):
    run_id: str
    scope_kind: str
    scope_id: str
    action_kind: str
    effect_kind: str
    action_digest: str
    policy_identity: str
    policy_reason: str
    action_display: str

    def approve(self, *, provider_identity: str) -> ApprovalDecisionView: ...

    def reject(
        self,
        *,
        reason: str,
        provider_identity: str,
    ) -> ApprovalDecisionView: ...
```

Existing workspace requests add non-breaking `scope_kind="workspace"` and
`scope_id=workspace_id` properties. `MCPApprovalRequest` uses
`scope_kind="mcp_server"` and the trusted server ID. Decisions expose the same binding view, while
each domain retains its own exact validation and concrete request/decision type.

The foreground console provider moves to `approval_cli.py` and formats only the bounded view:
scope, action/effect kind, digest, policy, limits, capabilities, and sanitized action display.
`coding.py` may re-export it temporarily to avoid an unrelated public import break.

The request owns the decision factory. The provider calls `request.approve(...)` or
`request.reject(...)` and never branches on workspace versus MCP concrete types. There is one
provider invocation abstraction, not an MCP-specific approval-provider interface. No durable
approval storage, reusable approval, or cross-domain approval conversion is added.

### 9.4 Prepared MCP action

`PreparedMCPAction` is separate from workspace `PreparedAction`:

```python
@dataclass(frozen=True, slots=True)
class PreparedMCPAction:
    server_id: str
    catalog_generation: str
    protocol_revision: str
    local_tool_name: str
    remote_tool_name: str
    input_schema_digest: str
    effect: MCPToolEffect
    policy: MCPToolPolicy
    arguments: Mapping[str, object] = field(repr=False)
    arguments_digest: str
    limits: MCPCallLimits
    display_text: str
```

Its canonical digest covers every field that can change request identity or authorization,
including canonical argument content. It excludes process ID, SDK request ID, timings, descriptions,
titles, annotations, and server-reported version strings.

Preparation:

- performs no process start, discovery, approval, hook, or remote call;
- uses the frozen descriptor selected by the local tool adapter;
- requires exact local and remote names;
- requires the current catalog generation and input-schema digest;
- validates canonical arguments against the admitted schema;
- rejects configured secret values in field names or string values;
- computes effective limits no greater than trusted configuration;
- creates a complete single-line sanitized approval display.

If approval is required, the full canonical argument projection must fit the approval-display bound.
Truncated or redacted approval arguments fail before approval because the user would not be
approving the exact outbound request.

### 9.5 MCP hard guards

MCP guards run in this fixed order:

1. `max_governed_calls`;
2. `server_scope`;
3. `catalog_binding`;
4. `tool_admission`;
5. `argument_limits`;
6. `secret_disclosure`;
7. `transport_capability`.

Guard meanings:

- `server_scope`: configured server ID and active client identity match the prepared scope.
- `catalog_binding`: generation, protocol revision, and schema digest match the frozen catalog.
- `tool_admission`: exact local/remote name, effect, and policy match the trusted grant.
- `argument_limits`: canonical bytes/characters and effective time/result limits remain bounded.
- `secret_disclosure`: configured secrets and forbidden authentication fields are absent.
- `transport_capability`: the client is ready, modern, tools-capable, sequential, and supports
  enforceable message/cancellation/cleanup behavior required by the action.

Dependency exception, malformed guard result, wrong order, missing client, or stale catalog fails
closed before transmission.

The same guards run after approval and again at the final effect boundary. The final pass occurs
immediately before `MCPClient.call_tool`.

### 9.6 Policy and hooks

`DefaultMCPActionPolicy` follows the trusted grant:

- `external_read` plus explicit `allow` returns allow;
- every `require_approval` grant returns require approval;
- `external_mutation` or `external_unknown` with `allow` is invalid configuration;
- missing or malformed grants deny.

MCP annotations do not enter the policy object.

The v1 MCP composition configures no pre-hooks or post-hooks and exposes no hook configuration.
The common driver still executes the hook stages with empty immutable sequences. Existing workspace
hook behavior is unchanged. A real MCP hook use case requires a later design rather than adding a
general remote plugin surface now.

### 9.7 Executor and effect state

The MCP executor:

1. receives one `PreparedMCPAction`;
2. constructs one `MCPCallRequest`;
3. emits the existing generic action executor-start event;
4. invokes `MCPClient.call_tool` once;
5. maps transport/result evidence;
6. emits executor completion;
7. builds one `MCPActionRecord`;
8. returns one bounded `ToolResult` or raises a typed terminal control/unknown-effect error.

Effect mapping:

- preparation, guard, policy, approval, or pre-send revalidation failure: `none`;
- transport failure proven `not_sent`: `none`;
- valid terminal MCP response: `complete` as transport completion only;
- timeout or cancellation after possible send: `unknown` and terminal run control;
- EOF, protocol loss, or connection failure after possible send: `unknown` and terminal MCP failure;
- cleanup failure after a possibly sent request: prior effect state is retained or widened to
  `unknown`, never narrowed;
- v1 does not synthesize `partial` without direct protocol evidence.

`isError=true` produces an error `ToolResult` with `complete` transport effect. The loop may recover
because the connection remains valid, but the result does not claim that the server made no change.

No call is automatically retried, including `external_read`.

### 9.8 MCP action record

`MCPActionRecord` is immutable, bounded, and content-free:

- run ID;
- action and arguments digests;
- server ID and catalog generation;
- local tool name and remote-name digest;
- protocol revision and schema digest;
- trusted effect and policy;
- ordered guard outcomes;
- approval outcome and provider identity;
- executor attempts (`0` or `1`);
- transmission state;
- effect state;
- terminal `is_error` when known;
- output/result truncation and omission flags;
- cleanup relevance;
- bounded reason codes and diagnostics.

It excludes raw remote name when unsafe, arguments, argument values, descriptions, annotations,
schemas, tool content, structured content, stderr, command argv, environment, secrets, and SDK
objects.

## 10. Tool Result Projection

### 10.1 Supported content

V1 accepts:

- MCP text content blocks;
- JSON-compatible `structuredContent`;
- optional output-schema validation when an admitted output schema exists.

V1 rejects or omits:

- images and audio;
- embedded resource bodies and resource links;
- unknown mandatory content block types;
- extension payloads not explicitly admitted;
- multi-round-trip `input_required`.

Unsupported optional blocks produce bounded omission evidence. A result containing no supported
content after omission is an observation failure.

### 10.2 Collection and validation order

Result handling order:

```text
raw message bound
    -> SDK decode
    -> result envelope/type validation
    -> bounded content-block intake
    -> bounded structured-content traversal
    -> optional output-schema validation
    -> secret sanitization
    -> deterministic rendering
    -> final character bound
```

The adapter never converts an arbitrary iterable to a tuple before its item bound. Structured JSON
uses the same iterative depth/node/string/collection limits as schema admission.

If both text and structured content exist, both are preserved under separate fields. Text is not
parsed as JSON, and structured content is not flattened into prose.

### 10.3 Model-visible rendering

The model receives compact deterministic JSON:

```json
{
  "status": "ok",
  "text": ["bounded text"],
  "structured": {"key": "value"},
  "omissions": []
}
```

`status` is `ok` or `error` from MCP `isError`. The rendering does not include server metadata,
schema, transport diagnostics, approval details, or action evidence.

MCP `isError=true` maps to `ToolOutcome.ERROR` with the existing `EXECUTION_ERROR` code. Result
shape/sanitization failures map to `OBSERVATION_FAILURE`; limit failures use `RESOURCE_LIMIT`;
approval, policy, call capacity, timeout, and cancellation retain existing stable distinctions.

No new model-visible error code is added unless an implementation case proves the existing
distinctions cannot support recovery.

## 11. MCP Application and Foreground CLI

### 11.1 Request and result

```python
@dataclass(frozen=True, slots=True)
class MCPRunRequest:
    user_message: str


@dataclass(frozen=True, slots=True)
class MCPRunResult:
    request: MCPRunRequest
    agent: AgentRunResult
    catalog: MCPToolCatalogEvidence
    action_records: tuple[MCPActionRecord, ...]
    cleanup: MCPClientCloseResult
    observation_limitations: tuple[str, ...]
```

`MCPRunResult` has no workspace diff, validator verdict, session snapshot, retrieval result, memory
recall, workflow checkpoint, or planning state.

`evidence_complete` is derived from catalog completeness, action-record collection, result
observation, and direct-child cleanup. It is not a claim that the external service state is correct.

### 11.2 One-run orchestration

`MCPAgentApplication.run` serializes calls with an in-process lock and owns one `RunCoordinator`.

```text
validate MCPRunRequest
    -> create/derive RunContext
    -> start fresh MCP client/process
    -> modern discovery
    -> complete catalog collection/admission
    -> construct MCP governed tools and ToolRegistry
    -> construct AgentRuntime with the caller's LLMClient
    -> create bounded run record collector
    -> AgentRuntime.execute with the same RunScope and ToolExecutionContext
    -> close record collector
    -> close MCP client/process in finally
    -> produce MCPRunResult or attach MCPFailureEvidence to the original error
```

Discovery, model execution, remote calls, and cleanup occur under one end-to-end run identity.
`AgentRuntime.execute`, not `AgentRuntime.run`, is used so `MCPAgentApplication` remains the sole
terminal lifecycle owner.

The registry is constructed only after catalog freeze and before the first model request. Built-in
demo tools are not included in the first MCP composition; the tool pool contains only admitted MCP
tools. This avoids unrelated name and evaluation combinations.

The application is reusable only sequentially. Every `run` starts a fresh server and catalog. It
does not preserve conversation history between runs.

### 11.3 Failure evidence

`MCPFailureEvidence` may be attached to a typed run error and contains:

- run ID;
- catalog evidence if discovery completed;
- retained MCP action records;
- cleanup evidence;
- observation limitations;
- whether any request may have been sent.

It does not contain the user prompt, tool arguments, results, config file, command, environment,
stderr, secret values, or raw exceptions.

Failure evidence never changes the original error category. Cancellation and deadline exhaustion
remain control errors; configuration, protocol, transport, and unexpected failures retain distinct
typed causes.

### 11.4 Foreground CLI

The new entry point is:

```text
dqagent-mcp --config PATH --message TEXT [provider overrides] [--non-interactive]
```

The CLI:

- loads `.env` only for existing LLM settings and referenced environment values;
- loads one strict MCP config file;
- creates the configured LLM client;
- chooses foreground or fail-closed non-interactive approval;
- runs one `MCPAgentApplication`;
- prints the final answer and bounded content-free catalog/action/cleanup summary;
- prints sanitized failures to stderr;
- returns nonzero when the run fails, cleanup is incomplete, evidence collection is incomplete, or
  an effect is unknown.

It does not provide an interactive multi-turn loop, session ID, retrieval index, memory database,
workflow, workspace, coding target, multiple config files, background mode, or approval bypass.

## 12. Events, Errors, and Failure Semantics

### 12.1 Events

New transport/catalog events are limited to:

- `MCP_SERVER_STARTED`;
- `MCP_DISCOVERY_COMPLETED`;
- `MCP_DISCOVERY_FAILED`;
- `MCP_SERVER_CLOSED`.

Remote actions reuse existing `TOOL_CALL_*` and `ACTION_*` event types. Generic action events add
only a bounded `action_domain="mcp"` attribute where needed.

MCP event attributes may contain:

- trusted server ID;
- protocol revision;
- catalog-generation digest;
- bounded page/tool/admitted/omitted counts;
- local tool name;
- effect/policy/outcome enums;
- transmission/effect/cleanup state;
- elapsed time, truncation flags, and reason codes.

They exclude config paths, command argv, process IDs, environment names/values, secret values,
server instructions, raw names when unsafe, descriptions, schemas, arguments, results, stderr, and
SDK exceptions.

### 12.2 Error taxonomy

Application-level errors:

- `MCPConfigurationError`: invalid config, unsupported revision, missing grants, collisions;
- `MCPProtocolError`: malformed discovery/page/result or unsupported mandatory capability;
- `MCPTransportError`: spawn, EOF, framing, connection, worker, or cleanup failure;
- `MCPUnknownEffectError`: request may have been sent and terminal effect evidence is absent.

They derive from one `MCPError`. `MCPConfigurationError` uses `CONFIGURATION`.
`MCPProtocolError`, `MCPTransportError`, and `MCPUnknownEffectError` use `UNAVAILABLE` and are
non-retryable in v1.

Model-visible tool errors reuse existing `ToolErrorCode` values as described in section 10.3.

### 12.3 Failure matrix

| Failure | Model-visible behavior | Effect | Run behavior |
| --- | --- | --- | --- |
| Config invalid | No model request | `none` | Configuration failure |
| Spawn/start failure | No model request | `none` | Transport failure |
| Discover unsupported/malformed | No MCP tools exposed | `none` | Protocol failure + cleanup |
| Page/catalog/schema/collision failure | No partial catalog | `none` | Config/protocol failure |
| Argument parse/schema/preparation failure | Tool error | `none` | Loop may recover |
| MCP guard or policy deny | Tool error | `none` | Loop may recover |
| Approval reject/unavailable/mismatch | Tool error | `none` | Loop may recover |
| Pre-send transport failure | No model continuation | `none` | Terminal transport failure |
| MCP `isError=true` response | Tool error | `complete` transport | Loop may recover |
| Supported successful result | Tool success | `complete` transport | Loop continues |
| Result observation failure after response | Tool error | `complete` transport | Loop may recover |
| Timeout/cancel after possible send | Control error | `unknown` | Terminal |
| Connection loss after possible send | No further model action | `unknown` | Terminal MCP failure |
| Unknown-effect record retention failure | Bounded failure evidence | `unknown` | Terminal |
| Cleanup failure before any call | Application failure | `none` | Terminal |
| Cleanup failure after a call | Application failure | prior or `unknown` | Terminal |
| Post-run evidence incomplete | Not a clean success | unchanged | CLI nonzero |

No row implies external rollback or correctness.

## 13. Security and Residual Risk

### 13.1 Trust matrix

| Input | Trust | Allowed influence |
| --- | --- | --- |
| MCP config file | Trusted after strict validation | Server process, grants, limits, policy |
| Host environment values | Trusted secret dependency | Selected server env only |
| Server identity/metadata | Untrusted | Bounded diagnostics only |
| Tool name/description/schema | Untrusted | Candidate catalog after validation |
| MCP annotations | Untrusted hint | No authorization influence |
| Model arguments | Untrusted | Prepared request after schema/secret checks |
| Approval response | Authorization input | Exact current action only |
| Tool result | Untrusted external data | Bounded model observation |
| SDK/process errors | Untrusted diagnostic | Typed bounded reason only |

### 13.2 Secret boundary

Prevention precedes redaction:

- the JSON config cannot contain environment values;
- only named environment variables enter the server process;
- model arguments containing configured exact secret values fail before send;
- approval, events, action records, reports, and errors never retain raw arguments or results;
- stderr is bounded and never projected to the model;
- output sanitization scans configured secret values before rendering.

This does not prove that the server process cannot read other host files, environment inherited by
the operating system, credential stores, network services, or user-account resources. The minimal
environment reduces accidental exposure but is not process isolation.

### 13.3 Prompt and context authority

Server instructions, descriptions, schemas, and results can influence model behavior only as
bounded tool metadata or tool-result data. They cannot:

- become system instructions;
- select another tool grant;
- change approval requirements;
- modify policy or limits;
- add a tool during a run;
- request resources, prompts, roots, sampling, or elicitation;
- persist in session, retrieval, or memory state.

Tool descriptions remain necessary for model selection but are untrusted. Enforcement remains
outside model context.

### 13.4 Residual risks

V1 explicitly retains:

- malicious or buggy server behavior behind a valid protocol response;
- hidden server-process state within one run;
- host filesystem/network access available to the local user account;
- server child processes not terminated by direct-child cleanup;
- external effects that cannot be independently verified;
- unknown effects after possible transmission;
- synchronous console approval that cannot be force-interrupted;
- provider-specific variation in how models use dynamic tool schemas;
- official SDK defects or unsupported pre-parse bounds;
- no tenancy, durable audit, replay, or recovery.

These limits are not Phase 10 failures when they are enforced, tested, and reported honestly.

## 14. Deterministic MCP Evaluation

### 14.1 Production-path boundary

`MCPEvaluationRunner` calls the production `MCPAgentApplication`. It does not implement a second
catalog builder, name mapper, governance path, result renderer, or agent loop.

Deterministic mode replaces only:

- model completions;
- approval decisions;
- the scripted MCP client for most cases;
- purpose-built transport/collector failure fixtures.

Schema admission, naming, catalog freeze, registry adaptation, governance, result projection,
application coordination, events, reports, and cleanup evidence remain production code.

A small controlled stdio fixture validates the official SDK adapter, modern discovery, framing,
process lifecycle, cancellation, stderr bounds, and cleanup. It uses no network and no credentials.
It is test infrastructure only and does not become a supported DQAgent MCP server product.

### 14.2 Versioned suites

T9 first adds:

- `evaluations/cases/phase-10-mcp-smoke-v1.json`;
- a three-case substrate suite;
- report schema and CLI.

The smoke cases prove:

1. one allowlisted read tool is discovered, approved, called, and rendered;
2. one unallowlisted or colliding tool never reaches the model;
3. one possible-send transport failure records unknown effect and terminal cleanup.

T10 adds `evaluations/cases/phase-10-mcp-baseline-v1.json` with approximately 8-10 representative
cases and an accepted credential-free baseline.

Representative coverage:

1. exact modern discovery and complete pagination;
2. unsupported revision or missing tools capability fails before model request;
3. allowlist admission, missing grant, duplicate name, repeated cursor, and atomic catalog failure;
4. provider-compatible name normalization, digest suffix, and local collision;
5. unsupported/oversized schema and secret-shaped argument denial;
6. explicit read allow, required approval success, rejection, and stale catalog binding;
7. successful text/structured result and output-schema validation;
8. server `isError`, unsupported content, result truncation, and observation failure;
9. pre-send failure versus possible-send timeout/EOF unknown effect;
10. real stdio lifecycle, bounded stderr, cancellation, and direct-child cleanup.

Focused tests cover combinatorial boundaries rather than expanding the end-to-end case count.

### 14.3 Report

The report includes:

- suite/case/config digests;
- protocol revision and client identity;
- catalog generation, counts, and local names;
- tool calls and stable outcomes;
- governance trajectory and effect state;
- event subsequence;
- result truncation/omission evidence;
- cleanup evidence;
- direct predicates and pass/fail reasons;
- bounded observation limitations.

It excludes config paths, commands, environment, secrets, raw schemas, descriptions, arguments,
results, stderr, process IDs, and absolute temporary paths.

The deterministic fingerprint excludes generated run/request IDs, timestamps, process IDs, output
text, and wall-clock timing. Timing is observed and bounded, not golden.

There is no live mode in Phase 10 v1.

## 15. Implementation Dependency Graph

```text
T0 detailed design/readiness
 |
 +--> T1 SDK/raw-bound feasibility and dependency pin
 |          |
 |          +--> T3 client port + scripted client ----+
 |                                                   |
 +--> T2 config/catalog/schema/naming ----------------+----> T4 catalog discovery/admission
 |                                                   |             |
 +--> T5 behavior-preserving governed driver --------+             |
             |                                                     |
             +--> T6 MCP action/guards/policy/approval/registry <--+
                              |
 T1 + T3 + T4 + T6 ----------+--> T7 stdio SDK transport + result/events/errors
                                                    |
                                               T8 application/CLI
                                                    |
                                               T9 eval substrate
                                                    |
                                               T10 cases/baseline/docs
                                                    |
                                               T11 audit/closure
```

T1 is a stop/go compatibility gate. T2, T3, and the behavior-preserving part of T5 may progress
independently after T0. T4 requires the pure catalog types. T6 requires both the frozen catalog and
the common governed driver. The first real stdio path is T7; no earlier task may claim MCP
integration from scripted objects alone.

T8 is the first complete user-facing path. T9/T10 must call T8's production application.

## 16. Checkpoints

### T0: Detailed design and implementation readiness

Persist this design, verify ADR/roadmap consistency, record current gates, and confirm no unresolved
scope decision. Do not add a dependency or source module.

### T1: SDK and raw-bound feasibility

Inspect and pin official SDK v2, prove modern-only mode, disable automatic callbacks/fallback, and
prove bounded raw stdio messages plus direct-child cleanup can be enforced. Stop and revisit the ADR
if those guarantees are unavailable. Do not implement tools or application composition.

### T2: Config, limits, schema subset, and names

Implement strict JSON config loading, immutable config/grants/limits, bounded schema traversal,
canonical digests, local-name projection, collision handling, and pure catalog values. No process,
SDK, model, registry, approval, or remote call.

### T3: Protocol-neutral client port and scripted client

Implement request/result/transmission/close values, the synchronous `MCPClient` port, and a bounded
scripted client. No official SDK, process, `ToolRegistry`, or model loop.

### T4: Discovery, pagination, admission, and catalog freeze

Implement modern discovery validation, page collection, exact grant admission, missing-grant
failure, atomic catalog generation, and content-free catalog evidence against the scripted client.
No real stdio process or tool execution.

### T5: Behavior-preserving governed execution driver

Add characterization regressions, extract the private common driver, add the non-breaking approval
view and private record protocol, and keep all Phase 9 behavior unchanged. Do not add MCP domain
behavior until the workspace path is green.

### T6: MCP prepared action and governed registry adapter

Implement `PreparedMCPAction`, fixed MCP guards, trusted policy, exact approval binding, no-op hook
stages, `MCPActionRecord`, local/remote name binding, one-call executor against the scripted client,
and governed registry dispatch. No official SDK or public CLI.

### T7: Stdio SDK transport, result projection, events, and errors

Implement the bounded async bridge, direct-child stdio lifecycle, modern discovery calls, raw
message bounds, cancellation, close evidence, text/structured result projection, event attributes,
and typed failures. No HTTP, legacy, resources, prompts, callbacks, or binary content.

### T8: MCP application and foreground CLI

Compose one run with fresh process, catalog, registry, runtime, record collector, cleanup, failure
evidence, exact foreground approval, strict config, bounded stdout/stderr, and non-interactive
fail-closed behavior. Keep all other application features disabled.

### T9: Deterministic evaluation substrate

Add versioned cases, fixture digests, scripted and controlled stdio clients, production application
runner, direct predicates, bounded report, cleanup checks, CLI, and the three-case smoke suite.

### T10: Representative cases, baseline, and documentation

Add the 8-10 case suite, accepted credential-free baseline, CI/local wiring, README, architecture,
roadmap, evaluation documentation, and one source-reading comparison. Do not add live mode.

### T11: Final audit and closure

Run focused/full/static/evaluation gates, credential/artifact scans, a fresh security/integration/
compatibility/evaluation-validity review, accepted finding remediation with red-to-green
regressions, and final closure. Only then mark ADR-0014 Accepted and Phase 10 Complete.

## 17. Acceptance Criteria

Phase 10 is complete only when:

- one strict trusted config can start one stdio MCP server for one foreground run;
- exact modern `2026-07-28` discovery completes before model exposure;
- a complete bounded catalog is atomically admitted, namespaced, collision-checked, and frozen;
- only exact allowlisted tools reach the model;
- untrusted annotations, schemas, descriptions, instructions, and results cannot authorize action;
- every MCP invocation uses the common governed order and reaches `tools/call` at most once;
- exact approval binds server, catalog, tool, schema, arguments, effect, policy, and limits;
- pre-send failure is distinguishable from possible-send unknown effect;
- cancellation, timeout, EOF, protocol loss, and cleanup preserve honest effect evidence;
- text and structured results are bounded, sanitized, deterministic, and independently evidenced;
- one production `MCPAgentApplication` and `dqagent-mcp` CLI exercise the path;
- deterministic scripted and real-stdio cases call the production application;
- existing Phase 3 through Phase 9 regressions remain passing;
- Ruff, strict mypy, full pytest, compile checks, deterministic evaluations, credential scans, and
  documentation consistency gates pass;
- ADR-0014 is accepted only after implementation evidence and fresh closure;
- HTTP, OAuth, legacy, resources, prompts, callbacks, binary content, multiple servers, persistent
  connections, Session/RAG/Memory/Workflow/Coding composition, and production isolation remain
  absent.

The scope rule is: implement one complete governed MCP tool path, not a general MCP platform.

## 18. Specification Basis

- [MCP 2026-07-28 specification](https://modelcontextprotocol.io/specification/2026-07-28)
- [MCP 2026-07-28 architecture][mcp-architecture]
- [MCP 2026-07-28 transports][mcp-transports]
- [MCP 2026-07-28 tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)
- [Official MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [MCP Python SDK versioning](https://py.sdk.modelcontextprotocol.io/versioning/)

[mcp-architecture]: https://modelcontextprotocol.io/specification/2026-07-28/architecture
[mcp-transports]: https://modelcontextprotocol.io/specification/2026-07-28/basic/transports
