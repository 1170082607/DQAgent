# DQAgent Roadmap

## Vision

Build and understand production-oriented AI Agent systems incrementally, from a model/tool loop to
an evaluated, safe, durable agent harness that can support long-running and collaborative work.

The roadmap is the source of truth for project direction. A phase describes a learning outcome and
an observable engineering capability, not a commitment to reproduce every feature of an existing
framework.

## Target Outcome

The project is intended to develop mid-to-senior AI Agent engineering judgment. On completion, a
contributor should be able to:

- Explain where model behavior ends and harness responsibility begins.
- Choose deliberately between an agent loop, a deterministic workflow, and multiple agents.
- Design provider, tool, context, persistence, and protocol boundaries with explicit failure
  behavior.
- Evaluate probabilistic outcomes and execution trajectories instead of relying on demos.
- Build safe action environments with permissions, approvals, isolation, and auditability.
- Operate an agent system with cost, latency, reliability, and security constraints.
- Read mature frameworks as design references without treating their APIs as architecture.

This is primarily an agent application and harness engineering path. Model training, Agentic RL,
GUI agents, and low-code platforms are optional study tracks rather than prerequisites.

## Guiding Principles

- Build the smallest complete capability for the active phase.
- Preserve the simple model/tool loop; add harness mechanisms around clear boundaries.
- Establish evaluation before adding features whose quality cannot be proven deterministically.
- Treat model output, retrieved content, tool metadata, and external protocol data as untrusted.
- Keep current architecture separate from future plans.
- Compare with mature projects after establishing a working local baseline.
- Add abstractions for real boundaries or multiple implementations, not hypothetical variation.
- Do not assume that more orchestration, more tools, or more agents produce better outcomes.

## Phase Completion Standard

Starting with Phase 4, a phase is complete only when it includes:

- A minimal end-to-end capability with explicit ownership and dependency direction.
- Tests for externally observable behavior and important failure paths.
- Evaluation cases or regression evidence appropriate to probabilistic behavior.
- Structured events and stable error behavior for new execution paths.
- Updated README, architecture, roadmap, and an ADR when a durable decision is introduced.
- A source-reading comparison that separates reusable ideas from framework-specific mechanics.
- Passing Ruff, strict mypy, and pytest checks.

## Retrospective After Phase 3

Phases 0-3 established a sound harness kernel:

- Provider SDK types are isolated behind a neutral `LLMClient` boundary.
- Model-requested tools are validated and executed through an explicit registry.
- The model/tool/observation loop is bounded and has defined recovery observations.
- Each run has identity, deadlines, cooperative cancellation, classified errors, retries, and
  ordered events.
- Conversation state commits only after successful execution.

This is stronger than a typical tutorial loop in failure semantics, testability, and provider
isolation. It is not yet evidence of an effective agent product. The current project cannot measure
model behavior, manage a finite context over long sessions, persist or resume work, safely expose
mutating environment tools, or prove that planning and multi-agent designs outperform a simpler
loop.

The original roadmap placed evaluation, security, and operational concerns in a final hardening
phase. That sequencing is no longer acceptable: every later feature is probabilistic or expands
the trust boundary. Evaluation moves to Phase 4, while security and operability become continuous
tracks.

## Completed Foundation

### Phase 0: Repository Foundation

**Status:** Complete

- [x] Standard Python `src` package layout.
- [x] Project metadata and dependency management through `pyproject.toml`.
- [x] Ruff, mypy, pytest, and test coverage reporting.
- [x] GitHub Actions verification workflow.
- [x] Open-source documentation, contribution guide, and Apache-2.0 license.

### Phase 1: LLM Client and Chat

**Status:** Complete

- [x] Provider-neutral message and completion models.
- [x] `LLMClient` application boundary.
- [x] OpenAI Responses API adapter.
- [x] Interactive and one-shot CLI.
- [x] In-memory conversation history and reset behavior.
- [x] Configuration validation and provider error translation.
- [x] Unit tests for application, configuration, and provider mapping.

Deferred from this phase: streaming output, persistent sessions, retries, rate limiting, and usage
accounting. Retry ownership was added in Phase 3; the remaining concerns are scheduled below.

### Phase 2: Tools and Agent Loop

**Status:** Complete

- [x] Define tool metadata, input schemas, and execution results.
- [x] Build an explicit tool registry.
- [x] Parse provider tool calls without leaking provider types into the application layer.
- [x] Implement a bounded agent loop: model request, tool execution, observation, and next request.
- [x] Define failure behavior for invalid arguments, unknown tools, timeouts, and repeated calls.
- [x] Compare the minimal implementation with OpenAI Agents SDK and EINO.

### Phase 3: Runtime

**Status:** Complete

- [x] Execution context with run identifiers, deadlines, cancellation, and metadata.
- [x] Structured events and explicit completed, failed, cancelled, and timed-out lifecycle states.
- [x] Stable error categories and bounded retries for retryable model-provider failures.
- [x] Event sinks for tracing, metrics, and audit adapters.

Cancellation is cooperative. A deadline bounds how long the caller waits and is propagated to model
and tool boundaries, but Python cannot force-stop an already-running thread. Hard execution
isolation, durable event delivery, tool retry/idempotency policies, and concurrent tool execution
remain later production concerns.

## Planned Development

### Phase 4: Evaluation Foundation

**Status:** Complete

**Outcome:** Make agent behavior measurable before adding more behavior.

- [x] Define versioned evaluation cases with inputs, fixtures, expected outcomes, and trace constraints.
- [x] Evaluate final-answer properties, tool selection and arguments, trajectory invariants, latency,
  attempts, and token usage where the provider exposes it.
- [x] Keep deterministic runtime tests separate from probabilistic model evaluations.
- [x] Add a deterministic evaluation mode for CI and an explicit, credentialed live-model mode for
  local runs.
- [x] Produce a baseline report for the Phase 3 agent and make regressions visible.
- [x] Study BFCL and GAIA evaluation semantics without making a large benchmark suite a runtime
  dependency.

The first evaluators should be deterministic predicates and structured trace checks. LLM-as-judge
may be added only for qualities that cannot be evaluated directly, with calibration examples and
known limitations documented.

### Phase 5: Workflow and Durable Execution

**Status:** Complete

**Outcome:** Orchestrate deterministic multi-step work with explicit state and recovery.

- [x] Define workflow state, nodes, transitions, terminal states, and validation rules.
- [x] Implement sequential and conditional execution before bounded parallel branches.
- [x] Specify branch merge, sibling cancellation, partial failure, and result ordering semantics.
- [x] Add checkpoint storage, interruption, resume, and replay/idempotency boundaries.
- [x] Reuse `RunContext` and runtime events instead of creating a second lifecycle model.
- [x] Compare the implementation with LangGraph persistence/interrupts and EINO graph execution.

A workflow is a deterministic orchestration mechanism, not a substitute for model agency. Model
decisions remain inside agent nodes; the graph owns known control flow, durable progress, and human
or system interrupts.

### Phase 6: Context Engineering and Sessions

**Status:** Complete

**Outcome:** Keep long-running model context relevant, bounded, and recoverable.

- [x] Separate durable session transcripts from the active model context.
- [x] Add session identity, persistence, resume, and explicit conversation concurrency behavior.
- [x] Assemble prompts from owned sections instead of one hard-coded system string.
- [x] Define context budgets and preserve tool-call/tool-result pairing during trimming.
- [x] Add cheap structural compaction before model-generated summaries and retain summary provenance.
- [x] Load project knowledge on demand rather than injecting all available instructions up front.
- [x] Evaluate long-session constraint retention, context overflow recovery, and compaction loss.

Session storage answers "what happened"; context construction answers "what should the model see
now." They must remain separate responsibilities.

### Phase 7: Retrieval-Augmented Generation

**Status:** Complete

**Outcome:** Ground answers in external knowledge with measurable retrieval quality.

- [x] Build an ingestion pipeline with document identity, chunking, metadata, and update/delete
  behavior.
- [x] Define provider-neutral embedding and retrieval boundaries only when concrete implementations
  exist.
- [x] Start with a small local store and explicit indexing lifecycle.
- [x] Return provenance with retrieved content and preserve citations through the answer path.
- [x] Evaluate retrieval independently with recall-oriented metrics before evaluating generated
  answers.
- [x] Address stale data, duplicate chunks, prompt injection in retrieved content, and empty
  retrieval.

RAG is an external knowledge service. It is not conversation history or long-term user memory.

The first embedding implementation is deterministic feature hashing for a credential-free baseline.
It validates the provider boundary and index lifecycle but is lexical rather than semantic. A real
embedding provider and scalable vector store require a concrete corpus and a new measured baseline.

### Phase 8: Long-Term Memory

**Status:** Complete

**Outcome:** Retain useful experience across sessions without turning the transcript into memory.

- [x] Define policy-governed memory records, explicit user/project scope, provenance, confidence, and lifecycle.
- [x] Separate extraction, admission, selection, retrieval, consolidation, correction, and forgetting.
- [x] Provide transactional SQLite persistence with scope revision concurrency control and logical forgetting.
- [x] Deliver explicit management, bounded cross-session recall, lower-authority context projection, and safe failure semantics.
- [x] Evaluate false memories, stale preferences, cross-session recall, scope isolation, and harmful over-retrieval through a deterministic production-path suite.
- [x] Record implementation evidence, source comparison, ADR acceptance, final audit closure, and release quality gates.

Memory is selected state with policy. Saving every message or embedding the full transcript does not
meet this phase's objective.

Detailed T0-T13 evidence, including the retrospective reconstruction of the missing T0-T4 record,
is preserved in the [Phase 8 Closure Record](learning/phase-8-closure.md). The architecture and
acceptance contract are in [ADR-0009](adr/0009-policy-governed-long-term-memory.md), the implemented
dependency direction is in [architecture.md](architecture.md), and the deterministic behavioral
evidence is in [the Phase 8 evaluation suite](../evaluations/README.md#phase-8-memory).

Deferred from this phase: encrypted sensitive-memory storage, forensic erasure, complete PII
classification, persistent or managed memory vector indexes, unconfirmed or automatic writes,
distributed tenancy or leases, background consolidation, bulk deletion, durable audit delivery, and
live-model memory-quality evaluation.

### Phase 9: Coding Agent Harness and Safety

**Status:** Complete

**Outcome:** Validate the accumulated runtime in a realistic, bounded action environment.

- [x] Add workspace-scoped read, search, patch, and command tools with explicit output limits.
- [x] Introduce policy decisions for allow, deny, and user approval before side-effecting actions.
- [x] Add pre/post tool hooks without coupling policy extensions to the core loop.
- [x] Define secret handling, path containment, subprocess limits, and hard isolation boundaries.
- [x] Load repository instructions and reusable skills on demand through the context layer.
- [x] Observe changes through diffs and validator results, not only tool return strings.
- [x] Compose the production `CodingAgentApplication` and foreground `dqagent-code` path with
      bounded action/context/diff/validator evidence and evidence-derived verdicts.
- [x] Add a versioned disposable coding-evaluation substrate with fresh repositories, controlled
      deterministic fixtures, direct predicates, bounded reports, and cleanup evidence.

T12 is the first complete production application path: `CodingAgentApplication` coordinates one
foreground run from target validation through final observation and trusted validators, while
`dqagent-code` exposes the same composition. T13 now provides the evaluator substrate and a
three-case smoke/negative suite. T14 adds representative cases and an accepted baseline; its finding
remediation, task-local fresh closure, and T15 final audit/closure evidence are complete. The
bounded v1 completion does not claim Phase 13 host/process isolation or live-model quality.

The durable completion evidence is preserved in the
[Phase 9 Closure Record](learning/phase-9-closure.md).

The initial coding agent should be useful with one model loop and a strong harness. Planning and
multiple agents are added only after evaluation shows where the simpler design fails.

### Phase 10: MCP and External Integration

**Status:** In Progress

**Outcome:** Discover and invoke external capabilities through a governed protocol boundary.

The initial client, transport, trust, governance, and compatibility contract is proposed in
[ADR-0014](adr/0014-integrate-mcp-tools-through-governed-client.md). Implementation evidence is
required before that decision becomes Accepted.
The bounded v1 task and acceptance contract is in the
[Phase 10 Detailed Design](phase-10-detailed-design.md).

- Implement an MCP client for tool discovery and invocation before building a server.
- Translate schemas into the existing tool boundary with stable namespacing and collision behavior.
- Define transport lifecycle, cancellation, timeouts, authentication, and connection failures.
- Treat server instructions, schemas, tool results, resources, and prompts as untrusted data.
- Integrate MCP tools with the same permission, event, and evaluation paths as local tools.
- Add resources, prompts, remote transports, or an MCP server only when a concrete use case requires
  them.

MCP standardizes integration; it does not add planning or reasoning ability.

### Phase 11: Planning and Long-Running Tasks

**Status:** Planned

**Outcome:** Make complex work explicit, recoverable, and budgeted.

- Distinguish an in-context checklist from a durable task graph.
- Implement task dependencies, ownership, status transitions, and stale-work recovery.
- Compare ReAct, plan-and-execute, and reflection using the Phase 4 evaluation harness.
- Add background operation handles and completion notifications without blocking the model loop.
- Resume interrupted work from durable task and workflow state.
- Bound replanning, reflection, time, model calls, and cost.

Planning is a policy choice, not a mandatory wrapper around every request. A plan must improve an
observed outcome enough to justify extra latency and tokens.

### Phase 12: Multi-Agent Coordination

**Status:** Planned

**Outcome:** Delegate only work that benefits from isolation or parallel ownership.

- Start with a subagent call that has an explicit input contract and isolated context.
- Add durable task ownership, result envelopes, asynchronous mailboxes, and cancellation
  propagation.
- Define concurrency limits, duplicate claims, partial failure, and orphan recovery.
- Isolate mutable workspaces for parallel coding tasks.
- Compare local delegation with cross-process agent protocols before adopting A2A or equivalent.
- Evaluate quality, latency, and cost against the best single-agent baseline.

Multi-agent coordination is a distributed system with nondeterministic workers. It should not be
used as a role-playing abstraction or as a substitute for a function, tool, or workflow node.

### Phase 13: Production Readiness and Capstone

**Status:** Planned

**Outcome:** Demonstrate and operate one coherent agent product under production constraints.

- Integrate the accumulated capabilities into a repository-maintenance capstone or another domain
  chosen through an ADR before this phase begins.
- Add streaming and user-visible progress without weakening terminal-state guarantees.
- Add durable telemetry delivery, cost budgets, rate limiting, backpressure, and overload behavior.
- Define authentication, tenancy, secret management, data retention, and audit requirements.
- Move untrusted execution behind a process, container, or remote-worker isolation boundary.
- Run fault-injection, load, recovery, security, and end-to-end evaluation suites.
- Publish an architecture narrative, benchmark report, operations runbook, and recorded demo.

This phase closes production gaps; it does not postpone all production thinking. The cross-cutting
tracks below apply from Phase 4 onward.

## Continuous Engineering Tracks

- **Evaluation:** Every new capability adds cases to the shared regression corpus and records its
  quality, latency, and cost impact.
- **Security:** Every new action or data source defines trust, validation, permission, and leakage
  behavior when introduced.
- **Observability:** New states and boundaries emit correlated events before concrete telemetry
  backends are added.
- **Durability:** Side effects, retries, checkpoints, and replay document idempotency expectations.
- **Source reading:** Each phase compares a small local implementation with one or two mature
  systems and records versioned evidence under `docs/learning/`.

## Optional Study Tracks

These topics are valuable but should not interrupt the main path without a concrete role or project
need:

- Agentic RL, supervised fine-tuning, and trajectory-based model improvement.
- Browser, GUI, voice, and multimodal agents.
- Low-code agent platforms.
- Additional model providers beyond the implementation needed to validate the neutral boundary.
- Large-scale distributed serving beyond the capstone's operational requirements.

## Reassessment Sources

The Phase 3 reassessment used the following projects as references, not templates:

- [Hello-Agents](https://github.com/datawhalechina/hello-agents): broad curriculum covering agent
  paradigms, framework construction, memory/RAG, context engineering, protocols, evaluation, and
  capstone applications.
- [learn-claude-code](https://github.com/shareAI-lab/learn-claude-code): incremental harness design
  covering permissions, hooks, context compaction, memory, task systems, background work, teams,
  isolation, and MCP around a stable agent loop.
- [OpenAI Agents SDK and EINO comparison](learning/phase-2-framework-comparison.md): runtime, tool,
  graph, tracing, checkpoint, and collaboration reference points already studied in Phase 2.

See [Roadmap Reassessment After Phase 3](learning/roadmap-reassessment-after-phase-3.md) for the
evidence, trade-offs, and rejected sequencing alternatives.
