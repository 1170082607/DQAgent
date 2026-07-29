# ADR-0004: Separate Evaluation Harness from Agent Execution

- Status: Accepted
- Date: 2026-07-28

## Context

After Phase 3, DQAgent could execute and observe a bounded model/tool loop but could not express or
measure behavioral expectations. Later roadmap phases add probabilistic behavior, so ordinary unit
tests alone cannot show whether those changes improve answers or trajectories.

Evaluation also has a trust problem: a second evaluation-only agent loop would drift from production
semantics, while running real models in CI would introduce credentials, cost, latency, and
nondeterministic failures.

## Decision

DQAgent will keep evaluation above the production runtime. `EvaluationRunner` creates an isolated
`AgentRuntime` per case and judges only provider-neutral outputs already owned by the harness:
conversation items, structured events, terminal state, elapsed time, attempts, and token usage.

Evaluation suites are JSON documents with an explicit schema version. Deterministic mode replaces
only the `LLMClient` with scripted completions and remains the CI gate. Live mode must be selected
explicitly and supplies the real provider adapter; credentials are never required by the default
evaluation path.

Version 1 uses deterministic predicates and structural trace checks. LLM-as-judge is excluded until
a quality cannot be checked directly and calibrated examples justify its additional variability.

## Consequences

- CI exercises the real runtime, tool registry, event stream, and evaluators without network access.
- The same semantic case can run against fixtures and a live model, while mode-specific cases remain
  explicit.
- Provider token usage crosses the neutral `Completion` boundary and is recorded in model-completed
  events; missing usage stays unknown rather than becoming zero.
- Deterministic success proves harness behavior, not model quality. Live reports require repeated
  sampling and environmental context before comparison.
- The initial schema intentionally supports a small predicate set rather than a general benchmark
  DSL or evaluator plugin system.

## Alternatives Considered

Pytest-only cases were rejected because they do not produce a stable behavioral report or provide an
explicit live-model path. Live-model CI was rejected because nondeterminism, credentials, network,
and cost make it a poor correctness gate. A separate evaluation loop and a benchmark-framework
dependency were rejected because both would obscure DQAgent's runtime semantics at this stage.
