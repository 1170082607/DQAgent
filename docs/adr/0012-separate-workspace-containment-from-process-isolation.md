# ADR-0012: Separate Workspace Containment from Process Isolation

- Status: Accepted
- Date: 2026-08-13
- Extends: [ADR-0002](0002-explicit-tool-boundary-and-bounded-loop.md), [ADR-0003](0003-observable-runtime-and-cooperative-cancellation.md)

## Context

Phase 9 must expose filesystem and command capabilities without overstating their safety. A canonical
path check can constrain DQAgent-owned file operations, but it cannot constrain an arbitrary child
process. Likewise, the current tool timeout runs a synchronous handler in a Python worker thread;
ADR-0002 and ADR-0003 already record that the thread may continue after the caller reports timeout.

The project therefore needs explicit answers to two different questions:

1. Which workspace resources may DQAgent-owned operations address?
2. Which effects can the selected execution backend technically prevent or terminate?

Calling both answers a sandbox would create a false trust boundary. It would also obscure the Phase
13 roadmap decision to move untrusted execution behind a process, container, or remote-worker
boundary.

## Decision

DQAgent will model workspace containment and subprocess isolation as separate guarantees that are
composed for governed tool actions.

### Workspace containment

The application composition root fixes one canonical workspace root and immutable workspace limits.
A shared workspace resolver is the only path authority for DQAgent-owned read, search, patch,
repository-resource loading, and change observation.

The resolver accepts workspace-relative logical paths and rejects absolute or parent traversal,
resolved paths outside the root, and escapes through symlinks, junctions, or reparse points. For a
nonexistent mutation target, containment is established from an existing contained ancestor before
creation and revalidated at the effect boundary. Protected repository metadata and configured
secret paths are denied independently of ordinary path containment.

Path checks are performed at the boundary that performs or observes the operation. A previous check
or user approval is not proof against path or target drift. Read, search, patch, resource loading,
snapshotting, and diff generation must share the same path semantics rather than implement local
string checks.

### Secret and output handling

Secret handling is prevention first. DQAgent-owned file tools do not expose configured secret paths,
secret environment variables are omitted from constructed subprocess environments, and structured
results contain only the fields required by the caller. Output is bounded and sanitized before it
reaches model context, events, diagnostics, or evaluation reports.

Redaction is defense in depth, not a complete exfiltration boundary. It cannot make an action safe
when a secret was unnecessarily read or passed to a child process, and it does not justify broader
filesystem or network access. Without a filesystem-isolating backend, DQAgent cannot guarantee that
an arbitrary command will not discover other host secrets; policy and capability checks must expose
that limitation rather than relying on redaction.

### Subprocess lifecycle

Command actions and harness-owned validators use one bounded subprocess boundary rather than the
generic worker-thread timeout. A subprocess request fixes an executable argument vector, a contained
working directory, a minimal allowlisted environment, wall-clock and output limits, and any
supported process resource limits. The adapter continuously drains bounded stdout and stderr and
returns a structured exit, timeout, cancellation, truncation, cleanup, and diagnostic result.

Timeout or cancellation invokes the backend's declared termination procedure and waits for bounded
cleanup. Every backend must own and reap the direct process it starts. It may claim process-group or
descendant-tree termination only on platforms where that mechanism is implemented and tested. An
action that requires a stronger lifecycle capability is denied when the selected backend does not
provide it. Termination does not claim rollback of filesystem, network, or external-service effects
already performed. When effect completeness cannot be established by workspace observation, the
result remains partial or unknown.

The initial command contract prefers explicit argument vectors. A shell interpreter is a distinct,
higher-risk action requiring explicit policy; a free-form shell string is not the default identity
used for policy or approval.

### Isolation capability

Every subprocess backend declares the isolation capabilities it actually enforces. The initial local
backend guarantees only the direct-process lifecycle and the working-directory, environment, and
output controls it implements on every supported platform. Process-group or descendant cleanup is a
separate declared capability, not an implied cross-platform guarantee. A normal host process does
not thereby isolate host filesystem access, network access, credentials, system calls, or other user
processes.

Workspace containment applies only to DQAgent-owned filesystem operations. It must not be presented
as containment of arbitrary commands. If an action requires an isolation capability unavailable on
the selected backend, the hard guard denies it; policy or user approval cannot manufacture the
missing technical guarantee.

Phase 9 may use the local backend only for actions whose declared isolation requirements are met;
conservative policy and explicit approval still apply where required. Deterministic evaluation uses
controlled disposable fixture repositories so cases do not share mutable state or credentials. The
project will not claim that local execution makes a general untrusted repository production-safe.
Phase 13 may add container or remote-worker backends and amend or supersede the local isolation
ceiling without changing ADR-0011's authorization order.

## Consequences

- One workspace scope becomes a shared capability for local tools, context resources, and change
  observation rather than a collection of handler-specific path checks.
- Approval can authorize a risk within an existing capability but cannot authorize path escape,
  secret exposure, or a missing isolation capability.
- Existing `RunContext` deadline and cancellation signals remain the run-level control contract;
  the subprocess adapter enforces them against the process scope its capability profile declares.
- The generic Python thread timeout remains suitable only for handlers whose continued execution is
  understood and acceptable. It is not used as the safety boundary for Phase 9 commands.
- Command and validator outputs have explicit limits and truncation evidence. A large output cannot
  rely on later context compaction as its first bound.
- Local execution remains useful for controlled coding fixtures but carries an explicit residual
  risk for host filesystem and network access.
- Phase 10 remote tools must declare their own transport, authentication, cancellation, and effect
  guarantees. Local workspace containment does not transfer to an MCP server.
- Phase 12 worktree separation, if later added, will address mutable-workspace ownership and not be
  mistaken for process security isolation.
- Container orchestration, remote workers, tenancy, durable audit, and production secret management
  remain Phase 13 concerns.

## Alternatives Considered

### Rely only on resolved-path containment

Rejected because an arbitrary child process can independently address paths outside the workspace or
use the network. A safe file helper is not a process sandbox.

### Reuse the current worker-thread timeout for commands

Rejected because Python cannot safely terminate the worker or its descendants. Reporting a timeout
while effects continue would violate the required subprocess limit.

### Treat user approval as an isolation substitute

Rejected because approval answers whether an action is authorized; it does not create filesystem,
network, process, or system-call enforcement.

### Call every local subprocess a sandbox

Rejected because a process boundary alone provides none of the stronger guarantees normally implied
by that term. Capabilities must be declared and tested individually.

### Require a container or remote worker in Phase 9

Rejected because the roadmap places the production untrusted-execution migration in Phase 13.
Implementing that platform now would expand Phase 9 beyond validating a bounded coding harness.

### Remove the command tool

Rejected because command execution and validator feedback are explicit Phase 9 requirements and are
necessary to evaluate realistic coding tasks.
