# ADR-0011: Govern Tool Actions Before Side Effects

- Status: Accepted
- Date: 2026-08-13
- Amends: [ADR-0002](0002-explicit-tool-boundary-and-bounded-loop.md)

## Context

ADR-0002 established a provider-neutral tool boundary in which `ToolRegistry` resolves a tool,
validates model-controlled JSON against its schema, and invokes a handler. That is sufficient for
the read-only demonstration tools implemented through Phase 8, but it does not establish who may
authorize filesystem or subprocess effects.

Phase 9 adds workspace-scoped read, search, patch, and command actions. These actions introduce a
trust transition from a model-proposed request to an application-authorized capability use and,
for mutating actions, an external effect. They also need policy decisions, foreground user approval,
pre/post hooks, and evidence that describes actual workspace changes and validator results.
Implementing those concerns independently inside each handler would make ordering and failure
semantics inconsistent and would prevent Phase 10 tools from reusing one governance path.

The core model/tool loop is not the owner of action authority. Repository instructions, skills,
retrieved data, memory, model output, and tool results may suggest an action, but none may expand
the capabilities supplied by the application composition root.

## Decision

DQAgent will introduce an application-owned governed action boundary between schema validation and
effectful execution. `AgentRuntime` will continue to call one provider-neutral tool execution port;
it will not own workspace policy, approval UI, hooks, or executor-specific behavior.

A model-reachable workspace action follows this fixed semantic order:

```text
lookup -> parse and schema validation -> immutable prepared action
       -> non-overridable guards
       -> policy: allow | deny | require_approval
       -> exact-action approval when required
       -> required pre-hooks
       -> execute at most once
       -> post-hooks and independent evidence collection
       -> bounded model observation plus structured execution evidence
```

### Prepared actions and authority

A tool adapter converts validated arguments into an immutable prepared action before authorization.
The prepared action contains the normalized fields that determine its effects, including its action
kind and effect class, workspace identity, targets or command, working directory, relevant
permissions, limits, and preconditions. It has a canonical digest suitable for binding an approval
to that exact action.

The composition root supplies workspace scope, hard guards, policy, approval provider, hooks,
executors, and validators. Model input and context resources cannot select a wider workspace,
disable a guard, replace a validator, or grant themselves permission.

Hard guards protect invariants such as workspace containment, protected or secret resources, and
required isolation capabilities. A policy `allow` decision or user approval cannot override a hard
guard. A policy evaluates only a successfully prepared action and returns one of three explicit
outcomes: allow, deny, or require user approval, with a stable reason.

### Approval semantics

Phase 9 approval is foreground, exact-action, and non-persistent. An approval is valid only for the
canonical prepared-action digest, workspace scope, and effect-relevant preconditions shown to the
approval provider. If any of those values drift before execution, the action must be prepared,
guarded, evaluated, and approved again.

The digest proves only that the action presented for approval matches the action submitted for
execution. It does not prove that a human approved it; that guarantee belongs to the configured
approval provider and its caller-facing interaction.

An unavailable non-interactive approval provider or end-of-input fails closed. A provider that
declares enforceable deadline support must also fail closed on timeout; a synchronous console
provider must not claim that capability when it cannot interrupt blocking input. User rejection is a
model-visible recovery observation; run cancellation or abort remains a distinct control result.
Phase 9 does not introduce session-wide grants, reusable command-prefix approvals, pending approval
storage, policy amendments, background waiting, or cross-restart recovery.

### Hooks and events

The governed action boundary owns ordered pre/post tool hooks. Hooks are typed by stage and receive
bounded, sanitized data. They do not own permission policy and cannot expand authority or mutate an
already approved action. A rule that changes allow, deny, or approval behavior belongs in policy
composition rather than a general-purpose hook.

A required pre-hook failure occurs before the effect and fails closed. A post-hook runs after the
execution attempt; its failure cannot roll back or conceal an effect that already occurred. Results
therefore keep action outcome separate from hook or observation outcome.

Existing `EventSink` implementations remain best-effort telemetry under ADR-0003. They may observe
the governed path but are not security hooks and cannot become fail-closed preconditions.

### Execution and evidence

Side-effecting actions are executed at most once by the governed boundary and are not automatically
retried by `AgentRuntime`. A timeout, cancellation, executor failure, or patch failure does not imply
that no effect occurred. When the boundary cannot prove the result, effect state is reported as
partial or unknown rather than as rollback or a clean failure.

`ToolResult` remains the bounded model-visible observation used by the existing loop. It is not the
only record of reality. Bounded action records separately retain the policy and approval trajectory,
effect state, and sanitized diagnostics; application-owned task evidence retains workspace-diff
completeness and validator results. This decision does not require a public or durable evidence
journal. Workspace changes and task validity are judged from those structured facts, not inferred
from a handler's natural-language success string or the model's final answer.

Disposable coding-task evaluation remains above production execution under ADR-0004. It uses the
same governed action path and judges expected and forbidden changes, validator results, policy and
approval outcomes, limits, and runtime events. It does not implement a second agent loop.

## Consequences

- ADR-0002's tool lookup, JSON Schema validation, provider-neutral calls/results, bounded loop, and
  recovery observations remain in force, but a validated effectful request no longer invokes a
  handler directly.
- `AgentRuntime` retains model/tool iteration, repeated-call protection, and provider retry
  ownership. Governance is composed behind its existing tool execution dependency.
- Authorization decisions become deterministic and testable independently of prompts and executor
  implementations.
- Approval cannot silently become a durable or blanket capability. Introducing persistence,
  recovery, delegated approvers, expiry, or revocation will require a later architectural decision.
- Hooks can extend the execution path without coupling extensions to the core loop, but their
  ordering and inability to elevate authority are contractual constraints.
- Failed actions may still produce change evidence. Callers and evaluations must handle none,
  partial, complete, and unknown effect states.
- Session rollback and workflow checkpoint rollback still cannot undo workspace effects. Phase 9
  does not add a workspace transaction manager.
- Phase 10 MCP tools must adapt their proposed actions to this governance path and cannot bypass it
  through transport-specific handlers. MCP schemas, instructions, and results remain untrusted
  inputs rather than authorization.
- Planning, multi-agent coordination, MCP implementation, background approval, and automatic
  mutating-tool retry remain outside Phase 9.

## Implementation Evidence

The Phase 9 implementation supports this decision through
`src/dqagent/tool_governance.py`, `src/dqagent/tools.py`, `src/dqagent/coding_tools.py`, and
`src/dqagent/coding.py`. The governed path performs bounded parsing, immutable preparation,
hard guards, tri-state policy, exact approval, revalidation, ordered hooks, one executor attempt,
post-effect evidence, and bounded private action-record retention.

Evidence includes `tests/test_tool_governance.py`, `tests/test_tool_governance_t4.py`,
`tests/test_tool_governance_t5.py`, `tests/test_coding_tools_t7.py`,
`tests/test_coding_application_t12.py`, and the Phase 9 deterministic coding cases. The current
full suite and Phase 3/6/7/8/9 deterministic gates pass. The implementation does not add a
durable action journal, rollback, reusable approval, automatic side-effect retry, or host/process
isolation.

## Alternatives Considered

### Describe safety rules only in the prompt

Rejected because model output is a request for authority, not an enforcement boundary. Prompt text
cannot reliably contain filesystem or subprocess effects.

### Put policy, approval, and hooks in each handler

Rejected because handlers would disagree on ordering, action identity, failure semantics, and
evidence. It would also make a shared Phase 10 governance path impractical.

### Add permission logic directly to `AgentRuntime`

Rejected because the runtime owns the provider-neutral model/tool state machine, not workspace or
transport policy. Coupling them would reverse the existing dependency direction and encourage
Planning or MCP-specific behavior in the core loop.

### Use `EventSink` as the hook mechanism

Rejected because ADR-0003 explicitly makes event sinks best effort and non-semantic. Required
preconditions must be able to fail closed, while telemetry failures must not alter the run.

### Represent every tool call as a Phase 5 workflow node

Rejected because foreground approval and final validators do not require durable checkpoint or
resume semantics. It would add a second control-flow system without a Phase 9 requirement.

### Cache approval for a session, path, or command prefix

Rejected for Phase 9 because broader grants require expiry, revocation, audit, conflict, and recovery
semantics. Exact-action approval is the smallest defensible initial contract.

### Automatically retry an action after an execution or isolation failure

Rejected because the first attempt may have produced partial effects. Retry requires action-specific
idempotency and reconciliation evidence that the Phase 9 boundary cannot infer generically.
