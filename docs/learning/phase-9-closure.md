# Phase 9 Closure Record

## Scope

Phase 9, `Coding Agent Harness and Safety`, is complete within its bounded v1 scope.
The phase delivers one production `CodingAgentApplication`/`dqagent-code` path above the
existing `AgentRuntime`, governed workspace actions, bounded observation, trusted validators,
request-scoped repository context, and a credential-free disposable coding evaluation.

This record is durable project evidence. It does not claim live-model coding quality, semantic
patch correctness, hostile-repository safety, host/process/network/credential/syscall isolation,
or any Phase 13 capability.

## T0-T15 Evidence

| Task | Evidence | Result |
| --- | --- | --- |
| T0 | `.local/reviews/phase9/T0/2026-08-18-readiness-01-read-only.md` | Current-worktree readiness backfill; historical record not fabricated |
| T1 | `.local/reviews/phase9/T1/2026-08-18-fresh-01-read-only.md`, disposition, `fresh-02-closure.md` | Current-worktree evidence chain closed |
| T2-T4 | `.local/reviews/phase9/T2` through `T4` review/disposition/closure chains | Frozen task findings closed |
| T5-T9 | Integration, security, concurrency/partial-effect, and isolation artifacts | Frozen task findings closed |
| T10-T11 | Repository instruction, skill, context authority and omission-budget artifacts | Frozen task findings closed |
| T12 | Comprehensive integration/security/context/approval/CLI/compatibility review chain | Frozen task findings closed |
| T13 | Disposable evaluator substrate review and regression closure | Frozen task findings closed |
| T14 | Representative 10-case suite, baseline, evaluation-validity review and closure | Frozen task findings closed |
| T15 | `T15/2026-08-18-final-audit-02-closure.md` and this record | Complete within v1 scope |

## Final Findings

The final-audit findings `P9-FA-F001` through `P9-FA-F005` were dispositioned as `accept`.
Each has current owner-scoped remediation, a persisted regression or evidence chain, current
focused/full/static/evaluation gates, and fresh closure evidence:

- T0/T1 evidence was backfilled for the current worktree without fabricating historical review.
- The deterministic coding evaluator now verifies active request, coding tools, repository
  guidance, instruction provenance, and selected skill markers at the model boundary.
- Workspace rules, sanitizer inputs, prepared actions, hooks, coding-tool secrets, capabilities,
  preconditions, validator argv, validator paths, environments, and exit-code collections are
  bounded before materialization.
- Lifecycle wording distinguishes T14 task-local closure from Phase 9 completion.

## Quality and Evaluation Gates

- Full pytest: `817 passed, 9 skipped`; coverage `85.47%`.
- Dependency/focused Phase 9 suite: `393 passed, 9 skipped`.
- Ruff: passed.
- Strict mypy: 53 source files, no issues.
- Compile check: passed.
- `git diff --check`: no whitespace errors; only existing LF/CRLF conversion warnings.
- Phase 3 deterministic: `4/4`.
- Phase 6 context deterministic: `3/3`.
- Phase 7 retrieval deterministic: `7/7`.
- Phase 8 memory deterministic: `13/13`.
- Phase 9 T14 coding evaluation: `10/10`, cleanup failures `0`.
- T14 deterministic fingerprint:
  `c0d36e802a025be6bb7dee82b7eeec0af917a1a66b42299ed69cb84a3ba40f5d`.
- Credential/artifact scan: no tracked credentials, local reports, caches, databases, absolute
  workspace paths, or temporary repository roots in the changed production/docs/evaluation
  boundary.

The nine skips are platform evidence boundaries: Windows symlink/developer privilege cases and
the POSIX-specific SIGTERM case. They do not become cross-platform isolation claims.

## ADR Acceptance Evidence

- ADR-0011 is implemented by the governed action boundary, exact approval, synchronous hooks,
  at-most-once execution, partial/unknown effect records, and private bounded collection.
- ADR-0012 is implemented by shared workspace authority and the local subprocess capability
  profile. The local backend does not claim host/process isolation.
- ADR-0013 is implemented by target-applicable instruction loading, explicit skill selection,
  provenance, lower-authority context projection, bounded omissions, and non-persistence.

## Deferred and Residual Scope

Phase 9 intentionally does not add planning, reflection, durable tasks, background work,
multi-agent coordination, MCP, reusable approvals, rollback, multi-file transactions, durable
action journals, transitive skill graphs, semantic skill selection, container/remote-worker
isolation, or live-model evaluation.

The runtime remains synchronous and cooperative. Foreground approval and Python hooks can block.
The coding application serializes one workspace only within one process. Local subprocesses can
still access host resources and descendants. Workspace containment is not a kernel-level
race-free hostile-filesystem guarantee.

## Completion Decision

Phase 9 is **Complete within the documented v1 boundary**. The package version is `0.9.0`;
the roadmap status is `Complete`; formal architecture, README, evaluation documentation, and
ADR implementation evidence point to this record and preserve all deferred limitations.
