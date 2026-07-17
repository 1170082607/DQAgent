# DQAgent Roadmap

## Vision

Build and understand production-oriented AI Agent systems incrementally, starting from direct model
interaction and progressing toward reliable agent runtimes and collaboration.

The roadmap is the source of truth for project direction. A phase describes learning and engineering
outcomes, not a commitment to reproduce every feature of an existing framework.

## Guiding Principles

- Build the smallest complete capability for the active phase.
- Understand the execution model before adopting a framework.
- Compare with mature projects after establishing a working baseline.
- Keep implemented architecture separate from future plans.
- Add production concerns when the capability they protect exists.

## Phase 0: Repository Foundation

**Status:** Complete

- [x] Standard Python `src` package layout.
- [x] Project metadata and dependency management through `pyproject.toml`.
- [x] Ruff, mypy, pytest, and test coverage reporting.
- [x] GitHub Actions verification workflow.
- [x] Open-source documentation, contribution guide, and Apache-2.0 license.

## Phase 1: LLM Client and Chat

**Status:** Complete

- [x] Provider-neutral message and completion models.
- [x] `LLMClient` application boundary.
- [x] OpenAI Responses API adapter.
- [x] Interactive and one-shot CLI.
- [x] In-memory conversation history and reset behavior.
- [x] Configuration validation and provider error translation.
- [x] Unit tests for application, configuration, and provider mapping.

Deferred from this phase: streaming output, persistent sessions, retries, rate limiting, and usage
accounting.

## Phase 2: Tools and Agent Loop

**Status:** Next

Planned outcomes:

- Define tool metadata, input schemas, and execution results.
- Build an explicit tool registry.
- Parse provider tool calls without leaking provider types into the application layer.
- Implement a bounded agent loop: model request, tool execution, observation, and next model request.
- Define failure behavior for invalid arguments, unknown tools, timeouts, and repeated calls.
- Compare the minimal implementation with OpenAI Agents SDK and EINO.

## Phase 3: Runtime

**Status:** Planned

- Execution context and run identifiers.
- Structured events and lifecycle states.
- Cancellation, deadlines, retries, and error classification.
- Hooks for tracing, metrics, and audit logs.

Runtime precedes workflow because workflows require a stable execution lifecycle and event model.

## Phase 4: Workflow Engine

**Status:** Planned

- Explicit state and transitions.
- Sequential, conditional, and parallel execution.
- Checkpointing and resumability boundaries.
- Comparison with LangGraph's graph and state model.

## Later Phases

5. Memory and session persistence.
6. Retrieval-augmented generation.
7. MCP client and server integration.
8. Planning and task decomposition.
9. Coding-agent environment and action loop.
10. Multi-agent coordination.
11. Production hardening: evaluation, security, tenancy, scaling, and operations.

Later phases remain directional. Their module boundaries should not be created until the preceding
phases reveal concrete requirements.
