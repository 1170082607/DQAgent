# Phase 9 Coding Harness Comparison

This note records the source-reading comparison used for the Phase 9 T14
representative evaluation. It compares the local implementation with the
mechanism order documented by `learn-claude-code`; it is not a feature or
security certification of either project.

## Source Evidence

- `learn-claude-code` README and mechanism track at commit
  `a9cafe953aa714f9cb1171f217d96bd2734bbcc7`.
- Permission and hook discussion:
  `https://github.com/shareAI-lab/learn-claude-code/blob/a9cafe953aa714f9cb1171f217d96bd2734bbcc7/s03_permission/README.en.md`
- Context compaction discussion:
  `https://github.com/shareAI-lab/learn-claude-code/blob/a9cafe953aa714f9cb1171f217d96bd2734bbcc7/s08_context_compact/README.en.md`
- Local design: `docs/phase-9-detailed-design.md`, ADR-0011, ADR-0012, ADR-0013.
- Local implementation: `src/dqagent/coding.py`, `coding_tools.py`,
  `tool_governance.py`, `repository_context.py`, `coding_evaluation.py`.
- T14 evidence: `evaluations/cases/phase-9-coding-baseline-v1.json`,
  `evaluations/baselines/phase-9-coding-deterministic-v1.json`, and
  `tests/test_coding_evaluation_t14.py`.

## Comparison

| Concern | Source-reading lesson | DQAgent Phase 9 choice |
| --- | --- | --- |
| Loop ownership | Keep one model-controlled loop while adding harness mechanisms around it. | `CodingAgentApplication` coordinates one existing `AgentRuntime` loop; T14 calls that path rather than an evaluation-only loop. |
| Permission boundary | Permission checks and hooks belong around effects, not inside model prose. | Hard guards, default policy, exact approval, synchronous hooks, and at-most-once execution live in the governed action boundary. |
| Context growth | Add context-management mechanisms before assuming an unbounded prompt is safe. | Repository instructions and explicit skill bodies are selected through bounded, provenance-bearing context projection. |
| Evaluation order | Mechanism additions need observable regression cases, not only a successful demo. | Cases assert diff, validator, governance, context, event, limit, and cleanup evidence through direct predicates. |
| Failure semantics | A denied or interrupted action must remain distinguishable from a completed effect. | Reject, stale approval, required pre-hook failure, command nonzero/timeout, unknown effect, and incomplete observation retain separate evidence and verdicts. |
| Isolation claim | Local command execution should not be described as a sandbox without an actual isolation boundary. | The local backend declares direct-child and bounded-stream capabilities while reporting missing host, network, credential, syscall, descendant, and workspace-only isolation. |

## Reusable Ideas

The useful shared principle is mechanism order: preserve the simple loop, then
add explicit permission, hook, context, and recovery boundaries around effects.
This keeps model behavior separate from harness authority and makes failure
semantics testable.

The local implementation also adopts progressive disclosure for repository
skills: catalog metadata is bounded, a body is loaded only for an explicit
unambiguous key, and invalid or oversized resources become omission evidence.
This is intentionally narrower than a general plugin or skill runtime.

## Deliberate Differences

DQAgent makes exact approval identity, workspace authority, path containment,
secret handling, direct argv, subprocess capabilities, final diff evidence,
and trusted validators explicit because those are the Phase 9 learning
boundaries. The T14 evaluator uses fresh disposable repositories and scripted
model/approval fixtures so the production path can be exercised without
credentials or network access.

The suite does not infer live-model quality from scripted completions. It also
does not prove safety against arbitrary hostile repositories, process-tree or
host-resource isolation, or recovery of effects after timeout or failure. Those
limits remain part of the documented contract and are deferred to later
production-readiness work.
