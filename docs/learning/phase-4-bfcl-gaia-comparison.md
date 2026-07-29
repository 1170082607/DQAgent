# Phase 4 Comparison: DQAgent, BFCL, and GAIA

## Scope and Evidence

This note studies evaluation semantics, not leaderboard scores or benchmark implementation details.
It uses the following primary references:

- Berkeley Function Calling Leaderboard repository at commit
  [`6ea5797`](https://github.com/ShishirPatil/gorilla/tree/6ea57973c7a6097fd7c5915698c54c17c5b1b6c8/berkeley-function-call-leaderboard)
  and the [BFCL paper](https://arxiv.org/abs/2402.18808).
- [GAIA paper, arXiv:2311.12983](https://arxiv.org/abs/2311.12983) and the
  [GAIA dataset card](https://huggingface.co/datasets/gaia-benchmark/GAIA).

Evidence was recorded on 2026-07-28. The environment could resolve the BFCL repository HEAD but
could not download the upstream pages or resolve a GAIA dataset revision. Conclusions below are
therefore limited to the stable benchmark semantics documented by the cited papers and entry points;
they do not claim coverage of the latest leaderboard categories or evaluator code.

## What Each Benchmark Measures

BFCL narrows the problem to function calling. Its central question is whether a model selects the
right function and constructs arguments that are structurally and semantically acceptable across
different calling patterns. This makes tool choice and argument structure first-class outputs rather
than incidental text in a transcript.

GAIA widens the problem to real-world assistant tasks requiring combinations of reasoning, tool use,
web research, multimodal inputs, and multi-step execution. It emphasizes a final answer that can be
checked exactly and stratifies tasks by difficulty. A correct final answer is valuable, but it does
not by itself diagnose where an agent trajectory failed.

DQAgent Phase 4 sits below both in scope. It is a regression harness for one small agent, not a public
benchmark. It combines BFCL-like structural checks for tool calls with GAIA-like final-answer
predicates, then adds runtime-owned trajectory and operational constraints.

## Reusable Design Lessons

### Prefer structured comparison when structure exists

Tool names and JSON arguments should be compared as typed structures. Comparing serialized strings
would make whitespace and key order affect correctness while missing the actual behavior. DQAgent
therefore parses arguments and checks ordered calls, outcomes, and stable error codes.

### Separate capability dimensions

Final-answer correctness, tool selection, arguments, recovery trajectory, latency, attempts, and
token usage answer different questions. Combining them into one score would hide whether a change
improved task quality by spending more calls or whether a correct answer came from the wrong path.
The Phase 4 report retains each check and raw metric per case.

### Make task fixtures and hidden environment explicit

Benchmarks need reproducible task data and controlled resources. DQAgent version 1 has only local
scripted completions and built-in tools, so it records fixtures in each versioned case. It does not
claim to reproduce GAIA's web, file, or multimodal environment.

### Deterministic and probabilistic evidence serve different owners

The deterministic suite detects harness regressions in CI. Live mode samples model behavior but is
sensitive to model revisions and provider conditions. Treating either as a substitute for the other
would resemble using mocked integration tests as an SLO, or using production traffic as a unit test.

## Deliberate Non-Adoptions

- No BFCL or GAIA runtime dependency: the local case set is too small to justify benchmark tooling.
- No aggregate leaderboard score: four cases cannot support a meaningful general capability claim.
- No LLM-as-judge: current qualities are directly testable, so a probabilistic judge would add noise
  without information.
- No hidden test split: repository regression cases must be reviewable. Hidden tasks matter for public
  leaderboard contamination, which is not DQAgent's current problem.
- No broad web or multimodal tasks: the Phase 3 harness does not own those environments yet.

## Next Evaluation Steps

Each later phase should add cases that isolate its new behavior and retain the Phase 3 suite as a
regression corpus. Before comparing models or architectures, live evaluation needs repeated trials,
model and prompt identity, environment fixture versions, and aggregate statistics with failure
inspection rather than a single pass rate.
