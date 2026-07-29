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
