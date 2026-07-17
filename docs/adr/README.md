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
