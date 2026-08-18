# Architecture Decision Records

Architecture Decision Records capture decisions that constrain future implementation or explain why
the project chose one approach over credible alternatives.

Use an ADR when a decision affects module boundaries, dependency direction, persistence, execution
semantics, public interfaces, or operational behavior. Do not use ADRs for routine refactoring or
temporary experiments.

## Naming

Use `NNNN-short-title.md`, for example `0001-provider-neutral-llm-boundary.md`.

## Lifecycle

- `Proposed`: under discussion.
- `Accepted`: current decision.
- `Superseded`: replaced by another ADR.
- `Deprecated`: retained for history but no longer recommended.

Start from [0000-template.md](0000-template.md).

## Records

- [ADR-0001: Provider-Neutral LLM Boundary](0001-provider-neutral-llm-boundary.md)
- [ADR-0002: Explicit Tool Boundary and Bounded Agent Loop](0002-explicit-tool-boundary-and-bounded-loop.md)
- [ADR-0003: Observable Runtime and Cooperative Cancellation](0003-observable-runtime-and-cooperative-cancellation.md)
- [ADR-0004: Separate Evaluation Harness from Agent Execution](0004-evaluation-harness-boundary.md)
- [ADR-0005: Checkpoint Deterministic Workflow Progress at Node Boundaries](0005-checkpointed-deterministic-workflow.md)
- [ADR-0006: Separate Durable Session Transcripts from Active Model Context](0006-separate-session-transcript-from-active-context.md)
- [ADR-0007: Use an Explicit Retrieval Index and Citation Boundary](0007-explicit-retrieval-index-and-citation-boundary.md)
- [ADR-0008: Coordinate the End-to-End Run Above AgentRuntime](0008-coordinate-end-to-end-run-above-agent-runtime.md)
- [ADR-0009: Policy-Governed Long-Term Memory](0009-policy-governed-long-term-memory.md)
- [ADR-0010: Keep Memory Extraction Before Deterministic Admission](0010-transient-memory-extraction-boundary.md)
- [ADR-0011: Govern Tool Actions Before Side Effects](0011-govern-tool-actions-before-side-effects.md)
- [ADR-0012: Separate Workspace Containment from Process Isolation](0012-separate-workspace-containment-from-process-isolation.md)
- [ADR-0013: Load Repository Instructions and Skills Through Context](0013-load-repository-resources-through-context.md)
- [ADR-0014: Integrate MCP Tools Through a Governed Client Boundary](0014-integrate-mcp-tools-through-governed-client.md)
