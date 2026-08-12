# Behavioral Evaluations

This directory contains versioned behavioral, context, and retrieval cases with committed baseline
reports.
Evaluations are kept separate from `tests/`: tests verify deterministic implementation contracts,
while evaluations measure agent outcomes, bounded model-context projections, or retrieval ranking.

## Case Contract

Each suite declares `schema_version`, `suite_id`, and isolated cases. A version 1 case owns:

- `input`: a user message and optional system prompt.
- `modes`: whether the case is meaningful in deterministic, live, or both modes.
- `fixture`: scripted provider-neutral completions required by deterministic mode.
- `expected.answer`: exact, non-empty, or case-insensitive containment predicates.
- `expected.tool_calls`: the exact ordered tool names, structural JSON arguments, outcomes, and
  optional error codes.
- `expected.trace`: an ordered event subsequence, an optional attempt ceiling, and mode-specific
  latency/token resource limits.

Changing the meaning of existing fields requires a new schema version. Adding or revising behavior
should normally create a new suite file instead of rewriting evidence from an earlier baseline.

## Modes

The default is credential-free and deterministic:

```bash
dqagent-eval --mode deterministic
```

It runs the real `AgentRuntime` and real tools against scripted model completions. This makes runtime,
tool-boundary, event, and evaluator regressions suitable for CI. It does not measure model quality.

Live mode is explicit and uses the same cases and predicates with the configured model provider:

```bash
dqagent-eval --mode live --output .local/evaluations/live-report.json
```

Live mode requires `DQAGENT_MODEL`; OpenAI additionally requires `OPENAI_API_KEY`, while the local
llama.cpp provider requires a running `llama-server`. Reports are samples affected by model version,
provider behavior, prompt nondeterminism, network, latency, and cost. Run repeated samples before
treating a change in pass rate as a model regression.

## Baselines

`baselines/phase-3-deterministic-v1.json` is the Phase 3 harness baseline generated from
`cases/phase-3-baseline-v1.json`. Wall-clock values are observations from one run, not golden values;
the case-level ceilings are the regression gates.

Trajectory constraints are shared across modes because they describe agent semantics. Latency and
token ceilings live under `resource_limits.<mode>` because deterministic fixtures, hosted models,
and local inference have materially different operational envelopes.

`cases/phase-6-context-v1.json` is a separate component-level suite run by
`dqagent-context-eval`. It uses the production `ContextBuilder` without invoking a model and checks:

- Retention of an old explicit constraint after whole-turn compaction.
- Preservation of the current request while oversized history is omitted within budget.
- Visibility of complete-turn loss when structural summary input is deliberately too small.

`baselines/phase-6-context-deterministic-v1.json` records the Phase 6 result. The loss case passes only
when the omitted marker is absent and `structural_omitted_turns` reports the dropped complete record;
it does not claim compaction is lossless.

`cases/phase-7-retrieval-v1.json` indexes a discriminating fixture corpus through the production
ingestion pipeline, then measures `Recall@k`, reciprocal rank, and explicit no-result behavior
through the production retriever. Cases include multi-chunk documents, lexical distractors,
paraphrased queries, multiple relevant documents, adversarial passage content, and a no-answer query.
Every case uses the suite-level score threshold. Recall and reciprocal rank are not applicable to
no-result cases, so their per-case values are `null` and ranking means exclude them:

```bash
dqagent-retrieval-eval --output .local/evaluations/retrieval-report.json
```

`baselines/phase-7-retrieval-deterministic-v1.json` records the credential-free hashing-embedding
baseline. Its score proves deterministic regression behavior, not general semantic retrieval quality.
Replacing the embedding implementation or corpus requires a new suite/baseline rather than silently
rewriting this evidence.

`cases/phase-7-rag-answer-v1.json` is a separate live-only answer suite. Run it with the configured
provider:

```bash
dqagent-rag-answer-eval --output .local/evaluations/rag-answer-report.json
```

It checks expected factual claim fragments, claim-level citation coverage, insufficient-evidence
behavior, and forbidden outputs from adversarial retrieved instructions. Each claim requires a
same-sentence citation to one allowed source whose retrieved chunk contains the lexical claim.
Allowed source IDs are alternatives, not a requirement to cite every listed document. These are
explicit lexical predicates, not semantic entailment or an LLM judge. Citation coverage is `null`
for no-answer cases and excluded from the coverage mean; those cases use the separate
insufficient-evidence check. The suite remains separate from the credential-free retrieval gate.

## Phase 8 Memory

`cases/phase-8-memory-v1.json` and `baselines/phase-8-memory-deterministic-v1.json` are the
credential-free long-term memory regression suite and baseline. Run them with:

```bash
dqagent-memory-eval --output .local/evaluations/memory-report.json
```

The v1 suite has 13 cases covering confirmed cross-session preference, assistant false inference,
current-request precedence, correction, expiry, forgetting, exact scope isolation, sensitive/secret
denial, irrelevant no-result, harmful over-retrieval, instruction-shaped memory, RAG/citation
separation, and the memory-disabled regression. Each case composes the production `MemoryService`,
`DefaultMemoryPolicy`, SQLite memory store, `MemorySelector`, `ContextBuilder`, and
`SessionAgentApplication`. Only the extractor and answer LLM are scripted fixtures. Write
admission, recall ranking, context selection, and answer utilization are reported as separate
stages, so a passing answer cannot hide an upstream failure.

The report includes false admission rate, `Recall@k`, `Precision@k`, scope leakage, stale/forgotten
recall, harmful over-retrieval, correction compliance, memory context character/record counts,
direct answer predicate pass rate, and explicit no-result correctness. `null` means a metric is not
applicable because its denominator is zero. No-result correctness covers enabled-memory cases with
both expected-result and expected-no-result outcomes; disabled-memory controls are excluded from
that denominator. Direct answer and citation checks are exact lexical predicates; there is no
LLM-as-judge.

The committed v1 baseline reports 13/13 passed, false admission `0.0` (0/3), mean `Recall@k`
`1.0` (7 applicable cases), mean `Precision@k` `1.0` (7 applicable cases), scope leakage `0.0`
(0/1), stale/forgotten recall `0.0` (0/3), harmful over-retrieval `0.0` (0/2), correction
compliance `1.0` (1/1), no-result correctness `1.0` (12 applicable cases), direct answer predicate
pass rate `1.0` (13/13), and mean projected memory context of 366.17 characters across 12 enabled
cases. These are fixture-corpus and architecture-regression measurements, not general memory
quality claims.

The deterministic/live boundary is intentional. The current Phase 8 gate always uses the real
production memory/session path with a temporary SQLite store, deterministic hashing embeddings,
scripted extraction fixtures, and scripted answers. It needs no credentials or network and proves
schema, policy, transaction, ranking, context, event, and evaluator regression behavior. It does not
prove an LLM can extract true facts or use memory well. There is currently no live Phase 8 mode;
adding one would require a separate non-CI report with model, prompt, extraction, policy, selector,
store, context, session, and answer identities, repeated samples, and calibrated expectations.
The current gate does not require credentials.
