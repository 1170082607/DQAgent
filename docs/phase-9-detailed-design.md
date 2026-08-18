# Phase 9 Detailed Design: Coding Agent Harness and Safety

- Status: Proposed
- Date: 2026-08-13
- Revised: 2026-08-13 after pre-implementation scope review
- Roadmap status: Phase 9 remains `Planned`
- Scope: Phase 9 only

## 1. Purpose and Authority

This document turns the seven Phase 9 requirements in [the roadmap](roadmap.md) into the smallest
coherent v1 implementation contract. The roadmap remains the source of truth. This design must not
remove a roadmap requirement, and implementation detail must not silently widen one.

Normative architectural inputs are:

- [ADR-0011](adr/0011-govern-tool-actions-before-side-effects.md): govern a prepared action before
  side effects, with exact foreground approval, ordered hooks, and independent evidence;
- [ADR-0012](adr/0012-separate-workspace-containment-from-process-isolation.md): workspace
  containment and subprocess isolation are different guarantees;
- [ADR-0013](adr/0013-load-repository-resources-through-context.md): repository instructions and
  skills are request-scoped, provenance-bound context resources.

Phase 9 preserves the Phase 3 bounded single-model loop and validates it in one realistic coding
composition. The target is a useful initial coding agent with a strong harness, not a general coding
platform.

### 1.1 V1 success boundary

The first complete Phase 9 path must let a caller:

1. choose one trusted workspace and bounded harness configuration;
2. submit a coding request with explicit target paths and optional skill keys;
3. run one existing `AgentRuntime` loop with governed read, literal search, single-file patch, and
   direct-argument command tools;
4. approve an exact side-effecting action in the foreground when policy requires it;
5. receive a bounded final workspace diff and structured validator results;
6. exercise the same path through disposable deterministic fixture repositories;
7. use the path through a public `CodingAgentApplication` and a small foreground CLI.

The implementation should prove these properties before adding optional combinations or richer
syntax.

### 1.2 Non-goals and deferred capabilities

Phase 9 v1 does not add:

- planning, reflection, goal continuation, or a durable task graph;
- multi-agent delegation, mailboxes, or worktree coordination;
- MCP clients, servers, resources, prompts, or remote tools;
- background commands, pending approvals, durable grants, audit recovery, or cross-restart resume;
- session-wide, path-wide, command-prefix, or reusable approval grants;
- automatic retry or generic reconciliation of side-effecting actions;
- multi-file patch actions, workspace transactions, or rollback;
- semantic skill selection, transitive skill references, plugin discovery, or background indexing;
- complete Session/RAG/Memory composition in the coding application;
- a generic validator language, evaluation DSL, policy framework, or hook platform;
- container, remote-worker, or production hostile-repository isolation.

Phase 3, 6, 7, and 8 disabled-path and authority regressions remain required. They do not make their
services dependencies of the Phase 9 coding path.

## 2. Ownership and Existing Architecture

Phase 9 extends existing boundaries rather than moving domain ownership into `AgentRuntime`.

| Concern | Phase 9 owner | Existing contract retained |
| --- | --- | --- |
| Model/tool iteration, provider retry, repeated-call rejection | `AgentRuntime` | Phase 3 bounded loop |
| Tool lookup and JSON Schema validation | `ToolRegistry` | Provider-neutral calls and results |
| Action preparation, authorization, hooks, effect attempt | Governed executor behind `ToolRegistry` | One runtime tool port |
| Run identity, deadline, cancellation, terminal lifecycle | `RunContext` and `RunCoordinator` | One lifecycle owner |
| DQAgent-owned filesystem authority | `Workspace` from trusted `WorkspaceScope` | Composition chooses scope |
| Process execution and declared lifecycle/isolation capabilities | `SubprocessRunner` | Adapter owns spawned process |
| Repository instructions and skills | Repository loader plus `ContextBuilder` | Active context ownership |
| Task baseline, final diff, validators, task verdict | `CodingAgentApplication` | Coordination above runtime |
| Disposable coding evaluation | `CodingEvaluationRunner` | ADR-0004 production-path pattern |

Phase-specific reuse is deliberately limited:

- Phase 3 supplies `AgentRuntime`, `ToolRegistry`, provider-neutral models, `RunContext`, lifecycle,
  events, retry ownership, and the bounded loop.
- Phase 5 supplies design lessons about explicit state and no false rollback, but `WorkflowRunner`
  is not on the Phase 9 critical path.
- Phase 6 supplies `ContextBuilder` ownership, budgets, request-scoped projections, and transcript
  separation. The coding v1 does not require session persistence.
- Phase 7 supplies untrusted-content, provenance, and production-path evaluation principles. Vector
  retrieval does not select instructions or skills.
- Phase 8 supplies composition-owned scope, policy-before-mutation, exact digest confirmation,
  fail-closed dependency behavior, and lower-authority context patterns. Memory domain types do not
  authorize actions and Memory is disabled in the v1 coding composition.

## 3. Focused Module Layout

The expected ownership is:

| Module | Responsibility |
| --- | --- |
| `workspace.py` | Scope, limits, path authority, protected/secret rules, snapshots, diffs |
| `tool_governance.py` | Prepared actions, guards, policy, approval, hooks, governed execution records |
| `coding_tools.py` | Read, search, single-file patch, command schemas and adapters |
| `subprocesses.py` | Bounded local process execution and capability declaration |
| `repository_context.py` | Instruction hierarchy, skill catalog/body, provenance evidence |
| `coding.py` | Coding request/result, application orchestration, foreground composition |
| `coding_evaluation.py` | Disposable cases, runner, predicates, reports |

These names define ownership, not a requirement for one class per concept. Tightly coupled private
helpers may stay together. Public exports are limited to caller-facing composition contracts.

Dependency direction:

```text
AgentRuntime -> ToolRegistry -> coding_tools -> tool_governance
                                           |          |
                                           v          v
                                      workspace   subprocesses

repository_context -> ContextBuilder -> CodingAgentApplication
                                           |        |
                                           v        v
                                      AgentRuntime  workspace/subprocesses
                                                    |
                                                    v
                                          CodingEvaluationRunner
```

`workspace`, `tool_governance`, and `subprocesses` must not import runtime, workflow, session,
retrieval, memory, or evaluation modules.

## 4. Cross-Boundary Invariants

1. Only trusted composition chooses workspace root, limits, protected/secret rules, policy,
   approval provider, hooks, executable policy, subprocess backend, and validators.
2. Model output, repository resources, retrieval passages, memory, tool output, and the user's
   approval response cannot expand a technical capability.
3. Every model-reachable coding action is canonical before policy and reaches its executor at most
   once per tool call.
4. Hard guards run before side effects and cannot be overridden by policy, approval, hooks, or
   prompt text.
5. Exact approval becomes invalid when its run, workspace, action digest, precondition, or required
   backend capability changes.
6. Every DQAgent-owned filesystem operation uses the same current workspace resolver semantics at
   the operation boundary.
7. A local process is never described as host-filesystem, network, credential, syscall, or
   workspace-only isolated without an enforcing backend.
8. Secret prevention precedes redaction. Limits and sanitization apply before content reaches the
   model, events, diagnostics, reports, or retained result objects.
9. Timeout, cancellation, and failure never imply rollback. Effect and observation uncertainty stay
   explicit.
10. Task changes and validation are judged from snapshots, diffs, and validator results, not model
    or tool success prose.
11. Repository instructions and skills remain request-scoped guidance and never become policy.
12. `EventSink` remains best-effort telemetry and cannot authorize or block execution.

## 5. Trusted Configuration and Limits

`WorkspaceScope` is immutable and issued by trusted application composition:

```python
@dataclass(frozen=True, slots=True)
class WorkspaceScope:
    workspace_id: str
    root: Path
    limits: WorkspaceLimits
    protected_paths: tuple[PurePosixPath, ...]
    secret_paths: tuple[PurePosixPath, ...]
```

The root is an existing canonical directory. `workspace_id` is opaque and does not expose the host
path. One `CodingAgentApplication` serializes mutation runs for its workspace; this is an in-process
composition invariant, not a distributed lease.

Limit ownership follows the module that enforces it rather than one cross-module configuration
object:

- `WorkspaceLimits`: logical paths, snapshot entries/bytes/time, and rendered diff;
- coding-tool configuration: raw arguments and read/search/single-file-patch collection/output;
- governance configuration: one run-wide governed-call/retained-record count;
- subprocess and validator definitions: argv, wall time, streams, and cleanup;
- repository-context configuration: catalog entries, instruction/body bytes, aggregate characters,
  and omissions;
- evaluation/report configuration: retained report content and event-attribute projection.

Trusted application composition assembles these immutable owner-specific values. It does not add a
generic configuration framework, and T1 does not predeclare limits for behavior owned by later tasks.

Initial numeric defaults are implementation configuration fixed by tests and evaluation fixtures,
not durable architecture. Model arguments may tighten a configured limit and may never enlarge it.
A single `max_governed_calls` ceiling bounds both governed-call admission and retained action records;
an excess call is rejected before policy or effect execution with a bounded resource-limit result.
No separate reservation system for model actions, command starts, and validator starts is required in
v1. `AgentRuntime.max_iterations`, the unified governed-call ceiling, schema/per-operation bounds,
and the fixed validator list provide the initial run bounds.

## 6. Workspace and Secret Boundary

### 6.1 Path authority

All model-facing paths are non-empty workspace-relative logical paths using POSIX separators. The
shared resolver rejects NUL, absolute paths, drive or UNC forms, parent traversal, ambiguous empty
segments, and any resolved target outside the root.

For existing paths, resolution checks components without treating a string prefix as containment.
Symlink, junction, or reparse-point escape is denied. For a missing patch-create target, the resolver
establishes authority through its nearest existing contained parent and revalidates that parent at
the effect boundary. Purpose-specific resolution distinguishes read, search, patch, instruction,
skill, snapshot, and command-cwd rules without changing the root.

The local resolver prevents straightforward path escape but does not claim kernel-level race-free
containment under hostile concurrent filesystem mutation. Phase 9 evaluation uses disposable,
exclusively owned fixture roots.

### 6.2 Protected paths, secrets, and outward text

`.git` and descendants are protected by default. Trusted composition supplies additional protected
and secret exact paths or subtrees; conservative defaults include `.env`, `.env.*`, private keys,
and configured credential files.

DQAgent-owned read, search, patch, instruction/skill loading, snapshot rendering, and diff rendering
deny or omit those resources before content is read. Denied errors contain stable reason codes and a
safe logical identity only when disclosing that identity is allowed. They never include denied
content, absolute host paths, or raw exceptions.

Subprocess environments are constructed from an explicit allowlist. They do not inherit arbitrary
host variables, and configured secret names/values are excluded. A shared sanitizer redacts known
literal secrets and host paths before final output truncation. Redaction is defense in depth, not a
substitute for preventing access.

An arbitrary local subprocess may still discover or modify host and excluded workspace resources.
That residual risk is visible in the backend capability profile and workspace observation
completeness; policy and approval do not turn the process into a sandbox.

### 6.3 Baseline, final snapshot, and diff

`WorkspaceObserver` captures one bounded task baseline before the model loop and one bounded final
snapshot after it. It walks only the contained workspace and does not follow links. Included regular
files record normalized path, kind, size, and a content digest when hashing completes within limits.
Bounded UTF-8 text may be retained for diff rendering; binary or oversized content is represented by
safe metadata and a digest when available.

Protected and secret content is never captured or fingerprinted in v1. Excluded or unobservable
regions are explicit observation blind spots. A conclusion that depends on proving those regions
unchanged is `indeterminate`, not a false success.

`WorkspaceDiff` reports deterministic create, modify, delete, and type-change records, including
untracked files. It separately represents inventory/content completeness and omission reasons. It
does not depend on Git or infer truth from a patch/command return string. Phase 9 does not require
per-action snapshots; patch operation records and process results are corroborating action evidence,
while the task baseline-to-final diff is the workspace fact.

`CodingRequest.target_paths` defines the normal task-evidence scope. Observation can be complete for
those targets even when excluded secret/protected regions remain explicit global blind spots. A
normal run may therefore be validated for its declared targets without claiming that an arbitrary
local subprocess left every unobservable path unchanged. Evaluation cases separately declare
forbidden paths or require global absence of changes; any blind spot overlapping such a predicate
makes that predicate indeterminate.

## 7. Governed Action Boundary

### 7.1 Prepared action

After bounded JSON parsing and Draft 2020-12 schema validation, each coding adapter constructs an
immutable `PreparedAction`. It contains only effect-determining normalized fields:

- action and effect kind;
- opaque workspace identity and logical targets or cwd;
- read/search windows, one patch operation, or direct argv;
- executable identity and non-secret environment identity where applicable;
- effect preconditions and required backend capabilities;
- effective limits;
- a versioned canonical SHA-256 digest.

Canonical encoding uses sorted-key UTF-8 JSON and never hashes Python representation, an absolute
workspace root, or a secret value. Preparation may inspect a target but must not mutate, start a
process, request approval, or invoke a hook.

### 7.2 Fixed execution order

Every governed coding tool follows:

```text
lookup -> bounded parse -> schema validation -> prepare immutable action
       -> hard guards -> policy: allow | deny | require_approval
       -> exact foreground approval when required
       -> revalidate action/preconditions/capabilities
       -> ordered pre-hooks
       -> final effect-boundary revalidation
       -> execute at most once
       -> ordered post-hooks
       -> bounded ToolResult + structured ActionRecord
```

Hard guards check workspace identity, current containment, protected/secret denial, configured
limits, preconditions, and required subprocess capabilities. Dependency exception or malformed guard
or policy output fails closed.

The initial policy may allow contained read/search and require approval for patch/command. Policy
rules are trusted composition data. A model or repository file cannot select a weaker policy.

### 7.3 Approval

Approval is foreground, exact-action, and non-persistent. `ApprovalRequest` includes run ID,
workspace ID, action/effect kind, canonical digest, policy reason, and a bounded sanitized action
display. Approval succeeds only when the provider returns `APPROVE` for the same identities and all
effect-relevant preconditions and capabilities still hold immediately before execution.

Rejection, unavailable provider, EOF, malformed response, or identity mismatch fails closed and is a
recoverable model-visible tool result. Cancellation remains a control error. A provider that can
enforce its own deadline may report `TIMED_OUT`; Phase 9 does not claim that every synchronous console
provider can forcibly interrupt blocking input. It never caches or broadens an approval.

### 7.4 Hooks

The governed boundary owns ordered `PreActionHook` and `PostActionHook` protocols. Hook inputs are
immutable, bounded, and sanitized. Hooks receive no workspace or subprocess capability and cannot
return a modified action, policy, approval, scope, or executor.

- required pre-hook failure stops before the effect;
- optional pre-hook failure is recorded and execution may continue;
- configured post-hooks are attempted in order after an execution attempt;
- post-hook failure cannot roll back, conceal, or reclassify an effect that already occurred.

Hooks run synchronously in process and are trusted application extensions. Elapsed time may be
recorded, but v1 does not promise hard termination of arbitrary Python hook code. A blocking or
side-effecting hook is a composition defect. External hook workers and a general plugin system are
out of scope. `EventSink` is not a hook.

### 7.5 Action outcome and evidence transport

Tool protocol outcome and effect state are separate. A rejected approval has an error tool result and
`none` effect; a nonzero command has an error tool result but may have completed workspace effects; a
timeout may have `partial` or `unknown` effects.

A bounded immutable `ActionRecord` contains action identity, guard/policy/approval trajectory, hook
results, executor attempt count (`0` or `1`), effect state, backend capability identity, and sanitized
diagnostics. It does not contain a second full workspace snapshot.

`ToolResult` remains the bounded model observation. The coding composition retains action records
through a private synchronous run-scoped collector so `CodingRunResult` and evaluation do not depend
on event delivery or parse natural-language tool output. Phase 9 does not expose a generic journal,
persistence, recovery, replay, or cross-run audit API. The collector is bounded by the maximum calls
admitted by `max_governed_calls`, binds every append to the exact active run ID, and is cleared at run
completion. A mismatched append is an observation failure. An excess call is rejected before an
executor can run and is visible through its bounded tool result and event; validator starts do not
consume or reserve model-call capacity.

### 7.6 Registry and runtime bridge

`ToolRegistry` remains the only tool dependency called by `AgentRuntime`. It supports legacy tools
and governed action tools. Governed tool raw argument bytes are bounded before JSON allocation, then
use the existing schema validator before entering preparation.

The coding composition allowlists its four coding tool names and rejects registering them through the
legacy handler path. Existing Phase 2/3 tools retain their handler and worker-thread timeout behavior
outside this composition. `AgentRuntime` may pass a small provider-neutral execution context with
`RunContext` and stage-event emission, but it must not import workspace, policy, approval, hook,
subprocess, or repository-resource types. Direct governed registry dispatch requires the explicit
run-scoped `ToolExecutionContext`; if it is omitted, the registry fails closed rather than creating
a per-call collector.

No side-effecting executor is automatically retried. Provider retry remains limited to model
requests made before a completion is accepted.

## 8. Coding Tool Contracts

All schemas set `additionalProperties: false`, use bounded fields, and return stable structured
headers plus bounded text. Truncation states which ceiling was reached.

### 8.1 `workspace_read`

Arguments are `path`, optional one-based `start_line`, and bounded `line_count`. The tool reads only a
contained regular non-secret UTF-8 file, with optional BOM support. It distinguishes missing, empty,
EOF, binary/invalid text, source limit, line limit, and output truncation. Returned lines are numbered
and no unbounded full-file allocation is required.

### 8.2 `workspace_search`

Arguments are a non-empty literal `query`, optional contained file/directory `path`, optional bounded
relative glob, `case_sensitive`, and bounded `max_matches`. V1 does not support regex. A Python walker
uses shared workspace semantics and deterministic path/line/column ordering. It bounds visited files,
source bytes, matches, displayed line bytes, elapsed work, and output. Zero matches is a successful
explicit observation. Denied, binary, link, or oversized resources produce safe omission counts.

### 8.3 `workspace_patch`

One call addresses exactly one file and one operation: `create`, `update`, or `delete`.

- create supplies complete bounded UTF-8 content and requires nonexistence;
- update supplies the expected SHA-256 digest and ordered exact old/new replacements with expected
  occurrence counts;
- delete supplies the expected SHA-256 digest.

Preparation validates syntax, containment, file kind, digest, occurrence counts, and limits before
approval. Execution revalidates immediately before mutation. Update computes the complete new
content in memory and uses a same-directory temporary file plus one atomic replacement where
supported. Create opens the final target exclusively after revalidation and never clobbers a target
that appears concurrently; a write/close failure may leave a partial created file and must be
reported as such. Delete checks the exact digest immediately before removal. Rename, chmod, links,
binary patches, wildcards, fuzzy hunks, and arbitrary scripts are absent.

The model may modify several files through several separately governed calls. This keeps approval,
TOCTOU, and failure semantics exact without inventing a cross-file transaction. Multi-file patch
actions require later evaluation evidence.

### 8.4 `workspace_command`

Arguments are a non-empty bounded array of non-empty strings `argv`, optional contained existing
`cwd`, and optional timeout no greater than the configured ceiling. The model cannot provide
environment variables. Trusted
composition resolves and allowlists the executable. Invocation is direct and never joins arguments
into an implicit shell string. A shell interpreter is available only if trusted policy explicitly
enables that higher-risk executable and governs its complete argv.

The result reports exit/signal, duration, timeout/cancellation, stdout/stderr truncation, spawn and
cleanup status, and backend capabilities. Nonzero exit is a model-visible execution error, not an
escaped registry exception. Command effects are observed by the task final diff; the local process
is not represented as restricted to workspace paths or denied secrets.

## 9. Subprocess and Validator Boundary

### 9.1 Local subprocess runner

Commands and validators share one `SubprocessRunner`. A request fixes direct argv, canonical
contained cwd, a constructed minimal environment, no stdin, timeout, stdout/stderr ceilings, and
required capabilities. The adapter drains both streams within bounds and returns a structured result
without retaining raw unbounded buffers.

Every supported backend must own, terminate on timeout/cancellation, and reap its direct child within
a bounded cleanup attempt. Process-group or descendant-tree cleanup is claimed only when implemented
and tested on that platform. Missing required lifecycle or isolation capability is denied before
spawn. A timeout or cleanup failure never claims rollback.

The initial local backend explicitly does not provide host filesystem, network, credential, syscall,
or workspace-only process isolation. Phase 13 may add a container or remote backend without changing
the ADR-0011 authorization order.

### 9.2 Validators

Validators are trusted application configuration and are never model tools. `ValidatorDefinition`
contains stable ID, direct argv, logical cwd, timeout/output ceilings, accepted exit codes, and
required backend capabilities. `ValidatorResult` contains status, safe argv identity, exit code,
duration, bounded stdout/stderr, truncation/decoding evidence, backend identity, and safe diagnostics.

Validators run after the model loop and final task snapshot. They must not modify evaluated source
files. Expected caches/build artifacts use trusted ignored paths or environment controls; v1 does not
capture a post-validator snapshot. A validator that violates this contract is a composition defect.

A normal coding composition may configure no validators. That result is explicitly `not_validated`,
never `passed`. Every disposable evaluation case that claims task success must configure at least one
case-relevant validator. All configured validators participate in the v1 verdict; there is no
required/optional validator mode. Timeout, cancellation, unavailable executable, or missing
capability remains visible and cannot be overwritten by the model's final answer.

## 10. Repository Instructions and Skills

### 10.1 Common resource contract

Repository resources carry stable kind/key, normalized source identity, content digest, applicable
path, selection reason, authority classification, character count, and content. Selected and omitted
evidence is bounded and excludes denied content and host paths.

Mutable repository and skill text is projected as clearly delimited repository/skill guidance, not
as host-owned mandatory policy. It cannot change workspace, guards, policy, approval, executable
rules, backend capabilities, or validators.

### 10.2 Repository instructions

V1 loads `AGENTS.md` by deterministic root-to-target applicability. `CodingRequest` supplies explicit
task target paths; the loader resolves them through `Workspace`, examines only their ancestor chains,
deduplicates shared instruction files, and orders resources from root to deeper applicable scope.
Free-form user text does not select paths.

An intended create target may be absent. The loader derives its ancestor chain from the nearest
existing contained directory without creating the target. A missing parent, path escape, or
ambiguous target fails context preparation.

Absent optional instructions are an empty success. Containment denial, invalid text, ambiguity, or a
mandatory resource failure stops context assembly; optional oversize produces typed omission
evidence. Files are admitted atomically and never silently truncated.

### 10.3 Skills

Trusted composition supplies contained workspace or read-only skill roots. The loader constructs a
bounded catalog with stable key, name, and description. Duplicate active keys are configuration
errors. A complete `SKILL.md` body is loaded only when `CodingRequest.skill_keys` explicitly selects
an unambiguous key.

Phase 9 does not infer a skill from user prose, expose a model skill-loading tool, or load references,
assets, scripts, or nested files named by the body. Missing/unknown/duplicate/oversized body behavior
is typed and bounded. The catalog and one selected body are sufficient to validate on-demand reusable
skills without defining a plugin or resource graph.

### 10.4 Context projection and lifetime

`ContextBuilder.build` accepts an optional immutable `RepositoryContext`. Repository resources have
an independent budget plus the overall context budget and selected/omitted projection evidence. They
do not reuse generic host knowledge's system-policy projection.

The application loads one immutable projection before the loop and uses it for all model requests in
that run. Editing an instruction or skill file appears in the final diff but does not self-modify the
active prompt. A later coding request reloads current content. Resource bodies are not written to a
session transcript, summary, retrieval index, or memory record.

## 11. Coding Application and Foreground CLI

### 11.1 Request and trusted composition

`CodingRequest` contains the user message, explicit task target paths, and optional skill keys. It
does not contain workspace root, policy, approval mode, secret rules, executable policy, validators,
or output ceilings; those are trusted composition dependencies.

`CodingAgentApplication` is composed with one workspace, observer, repository loader, context
builder, governed coding registry/runtime, optional validators, and run coordinator. V1 does not
compose session, retrieval, or memory services. Their existing disabled behavior and authority
invariants are covered by regression tests.

### 11.2 Execution path

```text
validate request and target paths
  -> capture bounded task baseline
  -> load applicable instructions and explicitly selected skills
  -> build bounded active context
  -> execute one AgentRuntime loop with governed coding tools
  -> capture bounded final snapshot and task diff
  -> run configured trusted validators
  -> return CodingRunResult
```

Validators do not start after run cancellation/deadline. On an escaped runtime failure, the
application attempts only bounded final observation allowed by the still-active cleanup contract,
then preserves the original control/failure category. It does not turn failures into successful
values or claim workspace rollback.

### 11.3 Result and task verdict

`CodingRunResult` contains the normal agent result, repository-context evidence, ordered action
records, baseline/final identities, task diff, validator results, and a task verdict:

Verdict rules are evaluated in this order after a normally completed loop:

1. `failed` when any configured validator definitively failed;
2. `indeterminate` when required target/forbidden-path observation is incomplete, an effect remains
   unknown where it matters, or a configured validator is unavailable, timed out, or could not run;
3. `not_validated` when required target-scoped workspace evidence is complete but no validator was
   configured;
4. `passed` when required target-scoped workspace evidence is complete, at least one validator was
   configured, and every configured validator passed.

This verdict describes harness evidence, not generic semantic correctness. Disposable evaluation
also compares expected and forbidden diffs. The model's final answer cannot rewrite the verdict.
A `passed` verdict never asserts that protected, secret, or backend-unisolated blind spots were
unchanged; those limitations remain in the result.

Failure exceptions may carry bounded `CodingFailureEvidence` with records and final observation
available at failure time. No action record, full diff, or repository body is persisted as a durable
audit or session transaction.

### 11.4 Foreground CLI

Phase 9 exposes a small `dqagent-code` entry point using the same production application. It accepts
an explicit workspace, request, target paths, and skill keys; provider and trusted safety settings
follow existing configuration patterns. The CLI presents exact approval summaries and displays a
bounded final diff, validator statuses, verdict, and observation limitations.

The CLI does not add background operation, reusable approvals, dynamic policy editing, arbitrary
environment injection, or a second execution loop. Non-interactive use fails closed when policy
requires approval and no explicit provider is configured.

## 12. Events, Errors, and Failure Semantics

Existing lifecycle events remain owned by `RunCoordinator`. Phase 9 may add non-terminal events for
prepared action, policy, approval, hooks, executor, observation, repository context, and validators.
Events contain stable IDs, kinds, outcomes, counts, durations, reason codes, and truncation flags;
they exclude raw prompts, file/process output, patch content, secrets, absolute paths, and free-form
approval text.

New stable tool errors are added only when externally actionable distinctions exist: containment or
protected-resource denial, policy denial, approval rejection/unavailable/mismatch, precondition
conflict, missing capability, resource/output limit, process failure, and observation failure.

| Failure | Model-visible behavior | Effect statement | Run behavior |
| --- | --- | --- | --- |
| Parse/schema/preparation error | Tool error | `none` | Loop may recover |
| Hard guard or policy deny | Tool error | `none` | Loop may recover |
| Approval reject/unavailable/mismatch | Tool error | `none` | Loop may recover |
| Required pre-hook failure | Tool error | `none` | Loop may recover |
| Patch conflict before write | Tool error | `none` | Loop may recover |
| Executor/process nonzero failure | Tool error | executor-specific | Loop may recover |
| Timeout or cleanup uncertainty | Tool/control result | `partial` or `unknown` where appropriate | Deadline/cancel may terminate |
| Post-hook failure | Visible separate hook result | does not change prior effect | Loop may recover |
| Incomplete final observation | Application evidence | unknown where proof is absent | Verdict `indeterminate` |
| No validator configured | No fabricated result | observed workspace effect retained | Verdict `not_validated` |
| Configured validator fail | Application result | workspace already observed | Verdict `failed` |

## 13. Disposable Coding Evaluation

`CodingEvaluationRunner` follows ADR-0004 and calls the production `CodingAgentApplication`. Each
case materializes a fresh temporary repository from a reviewed fixture source, uses a controlled
minimal environment, and owns its model script, approval fixture, policy, targets, skills,
validators, expected diff, and forbidden paths. Cases share no workspace, approval state, mutable
cache, credentials, child process, session, retrieval, or memory state.

Deterministic mode replaces only model completions, approval decisions, and purpose-built failure
fixtures. It uses real workspace resolution, governance, tools, subprocess adapter, context
projection, application orchestration, diff, and validators. Reports are bounded and credential-free.
A deterministic baseline proves regression behavior in controlled fixtures, not safety against an
arbitrary hostile repository or live-model coding quality.

The representative production-path suite should contain approximately 8-10 cases:

1. bounded read/search, including explicit no-result;
2. approved single-file edit with expected diff and passing validator;
3. traversal, protected path, and secret denial;
4. rejected or stale exact approval fails before effect;
5. required pre-hook block and post-hook failure after a visible effect;
6. representative command nonzero/output-limit/timeout cleanup behavior;
7. failing validator overrides a model success claim;
8. nested instruction applicability and hostile guidance cannot authorize action;
9. explicit skill body load plus bounded omission/unknown-key behavior;
10. incomplete observation cannot produce false success.

Focused unit and integration tests, rather than combinatorial E2E cases, cover path forms, schema
edges, every approval outcome, hook modes, individual limits, process races, duplicate resources,
and legacy Phase 3 behavior. The evaluation case format remains domain-specific and does not become a
general validator or policy DSL.

The T13 implementation intentionally stops at a three-case smoke/negative suite. It proves the
substrate's production-path composition and negative meta-properties without claiming representative
coding coverage or a baseline. Expanding to the cases above is the T14 checkpoint.

## 14. Implementation Dependency Graph

```text
T0 design/readiness
 |
 +--> T1 workspace authority/limits/secrets
            |
            +--> T2 baseline/final diff --------------------------------------+
            +--> T3 action/guards/policy + capability vocabulary              |
            |          +--> T4 approval/hooks --> T5 executor/registry -------+
            |          +--> T8 local subprocess -------> T9 command/validators|
            +--> T6 read/search --> T7 single-file patch ---------------------+
            +--> T10 instructions --> T11 skills/context --------------------+
                                                                              |
 Phase 3 AgentRuntime + ContextBuilder + T1-T11 --------------------------> T12 application/CLI
                                                                              |
 ADR-0004 + T12 ----------------------------------------------------------> T13 eval substrate
                                                                              |
                                                                         T14 cases/baseline/docs
                                                                              |
                                                                         T15 audit/closure
```

T1 is the first implementation dependency after T0. T2, T3, and T10 may then progress independently;
T3 establishes the shared capability vocabulary before T8 implements a backend. T5 integrates
governance with the existing runtime port. T12 is the first complete useful coding path; T13/T14 must
call it rather than build an evaluation-only loop.

## 15. Checkpoints

### T0: Implementation readiness

Confirm roadmap mapping, ADR consistency, dependency direction, platform constraints, existing
regression commands, and that no unresolved durable decision remains.

### T1: Workspace authority, limits, and secrets

Implement scope, workspace observation limits, resolver, protected/secret denial, and outward
sanitization. Prove traversal/link/missing-target behavior without exposing a mutating model tool.

### T2: Baseline, final observation, and diff

Implement bounded snapshots, deterministic diffs, blind spots, and completeness. Do not add keyed
secret fingerprints or per-action snapshot machinery.

### T3: Prepared action, hard guards, and policy

Implement the shared subprocess capability vocabulary, exact canonical action identity, effect
preconditions, non-overridable guards, tri-state policy, and minimal structured records without
executing effects.

### T4: Exact approval and synchronous hooks

Implement foreground exact-action approval and ordered trusted hooks with fail-closed pre-hook and
non-concealing post-hook semantics. Do not promise hard hook timeout or durable grants.

### T5: Governed execution and runtime bridge

Fix the at-most-once execution order, bounded private action-record collection, governed registry
dispatch, raw argument limit, events, and legacy compatibility.

### T6: Bounded read and literal search

Deliver deterministic contained read/search with source, collection, elapsed, and output limits.

### T7: Single-file structured patch

Deliver create/update/delete with exact digest/occurrence preconditions, approval-before-write,
effect-boundary revalidation, and independent final diff evidence.

### T8: Local bounded subprocess

Extend the T3 capability contract with direct argv requests/results, minimal allowlisted environment
construction,
stream/time limits, direct-child cleanup, and honest platform-specific backend declarations.

### T9: Governed command and validators

Compose the command tool and trusted optional validators on the subprocess boundary. Prove missing
capabilities deny before spawn and no-validator never means passed.

### T10: Repository instructions

Deliver contained target-aware `AGENTS.md` hierarchy, provenance, atomic budgets, and hostile-content
authority tests.

### T11: Skills and context projection

Deliver bounded catalog, explicit `SKILL.md` body selection, `ContextBuilder` projection, and
transcript/retrieval/memory exclusion. Do not implement references.

### T12: Coding application and foreground CLI

Compose one production path with baseline, context, runtime, final diff, validators, action records,
verdict, exact approval UX, and bounded final display. Keep Session/RAG/Memory disabled.

### T13: Disposable evaluation substrate

Add versioned fixture definitions, temporary materialization, production-path runner, direct
predicates, safe reports, cleanup, and CLI integration without yet expanding a large case matrix.
Implemented as `coding_evaluation.py`, `coding_evaluation_cli.py`, and the three-case
`phase-9-coding-smoke-v1.json` substrate suite.

### T14: Representative cases, baseline, and documentation

Add the 8-10 representative cases, credential-free baseline/CI command, architecture/evaluation
documentation, and evidence-to-roadmap mapping.

### T15: Final audit and closure

Run Ruff, strict mypy, full pytest/coverage, all existing deterministic evaluations, Phase 9 suite,
credential/artifact scan, specialized reviews, and a fresh final audit. Update roadmap/version and
closure records only after findings are dispositioned and the user accepts closure evidence.

## 16. Acceptance Criteria

Phase 9 is complete only when:

- the existing bounded loop can use workspace-scoped read, literal search, single-file patch, and
  command tools with explicit enforced limits;
- canonical actions pass hard guards and allow/deny/require-approval policy before effects;
- exact foreground approval fails closed and side-effecting calls are never automatically retried;
- synchronous pre/post hooks have deterministic ownership and failure behavior outside telemetry;
- path containment, secret prevention, process limits, and the local isolation ceiling are tested
  and represented honestly;
- applicable instructions and explicitly selected skill bodies enter active context on demand with
  provenance, budgets, authority labels, and omission evidence;
- task changes are observed through bounded complete-or-explicitly-incomplete diffs and structured
  optional validator results;
- a caller can use the production `CodingAgentApplication`/CLI and see approval, diff, validation,
  verdict, and limitations;
- disposable fixture evaluation calls that production path and covers useful coding plus the major
  safety failures;
- Phase 3/6/7/8 regressions and authority invariants remain passing;
- Planning, Multi-Agent, MCP, background work, durable approval/recovery, transitive skill
  references, multi-file patch, and Phase 13 isolation remain absent.

The core scope rule is: reduce the number of mechanisms v1 must implement, not the safety properties
v1 must prove.
