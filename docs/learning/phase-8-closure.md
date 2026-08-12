# Phase 8 Closure Record

## Scope and Evidence Boundary

Phase 8 is complete as the bounded v1 long-term-memory capability described by
[ADR-0009](../adr/0009-policy-governed-long-term-memory.md). This note preserves the detailed
delivery record that is intentionally omitted from the roadmap.

The T0-T4 entries below are a retrospective reconstruction from repository history, current source,
tests, evaluation artifacts, and the retained Phase 8 execution prompts. The worktree does not
contain independent checkpoint review/disposition records for T0-T4, so this note does not claim that
those records existed or recreate findings that cannot be evidenced. T5-T13 are summarized from the
retained execution evidence and implementation history. T13 is represented by the current uncommitted
worktree delta after `39996de`; it is not a commit made by Codex.

The detailed task definitions remain in
`.local/learning-notes/2026-08-07-phase-8-batch-execution-prompts.md`. Files under `.local/` are
execution material, not normative project documentation.

## T0-T4 Retrospective Reconstruction

| Task | Reconstructed result | Evidence | Record status |
| --- | --- | --- | --- |
| T0 | Proposed architecture contract for scoped, policy-governed memory | `a4bb6fc`; `docs/adr/0009-policy-governed-long-term-memory.md`; `docs/adr/README.md` | Retrospective; no independent checkpoint review record found |
| T1 | Domain values and structural invariants for scopes, candidates, records, tombstones, provenance, and confidence | `ce44239`, later correction `d40b77a`; `src/dqagent/memory.py`; `tests/test_memory.py` | Retrospective; no independent checkpoint review/disposition record found |
| T2 | Deterministic admission and recall-eligibility policy with stable decision/reason codes | `1edd5c8`, later correction `d40b77a`; `src/dqagent/memory_policy.py`; `tests/test_memory_policy.py` | Retrospective; no independent checkpoint review/disposition record found |
| T3 | Provider-neutral transactional store contract, in-memory adapter, scope revision CAS, and atomic change sets | `20bd2c9`; `src/dqagent/memory_store.py`; `tests/test_memory_store.py` | Retrospective; no independent checkpoint review/disposition record found |
| T4 | Standard-library SQLite adapter with schema checks, transactions, rollback, cross-connection CAS, and logical forgetting | `6429c38`; final `src/dqagent/memory_store.py`; final `tests/test_memory_store.py` | Retrospective; no independent checkpoint review/disposition record found |

The history also contains `6d57e14` and merge commit `bea60fa` around the early domain/policy work.
They are retained as history evidence, not counted as additional T1/T2 checkpoints. The later
`d40b77a` correction is included because the current behavior and tests, rather than an early
intermediate implementation, are the evidence for the closed phase.

## T5-T13 Execution Summary

| Task | Delivered boundary | Primary evidence |
| --- | --- | --- |
| T5 | Explicit `MemoryService` proposal, preview, exact-digest confirmation, inspection, correction, expiry, and forgetting; deterministic consolidation remains outside storage | `7bac1a0`; `src/dqagent/memory_service.py`; `src/dqagent/memory_consolidation.py`; `tests/test_memory_service.py` |
| T6 | Separate `dqagent-memory` CLI with explicit scope, interactive confirmation, safe defaults, and stable lifecycle/error output | `36b337f`; `src/dqagent/memory_cli.py`; `tests/test_memory_cli.py` |
| T7 | Exact-scope filter-before-rank recall with request-time hashing embeddings, deterministic ordering, bounded limits, and no persistent memory vector index | `ec7a6a3`; `src/dqagent/memory_recall.py`; `tests/test_memory_recall.py` |
| T8 | Lower-authority, untrusted memory projection with an independent budget, atomic record admission, RAG separation, and disabled-path regression | `b7b8217`; `src/dqagent/context.py`; `tests/test_context_memory.py` |
| T9 | Optional read-only cross-session recall in the required retrieval -> memory -> context -> runtime -> session CAS order, with narrow best-effort failure handling | `2d94f13`; `src/dqagent/application.py`; `src/dqagent/execution.py`; `tests/test_session_memory_recall.py` |
| T10 | Bounded committed-turn extraction to transient candidates, strict model output validation, no tools/store access, and preview-only persistence boundary | `54c0067`; ADR-0010; `src/dqagent/memory_extraction.py`; `tests/test_memory_extraction.py` |
| T11 | Versioned credential-free deterministic evaluation through the production memory/session path with separate write, recall, context, and answer metrics | `02ba3d1`; `evaluations/cases/phase-8-memory-v1.json`; `evaluations/baselines/phase-8-memory-deterministic-v1.json`; `src/dqagent/memory_evaluation.py`; `tests/test_memory_evaluation.py` |
| T12 | Evidence-based README, architecture, evaluation documentation, ADR update, and pinned LangGraph Store/Letta comparison; final status intentionally left for T13 | `39996de`; `README.md`; `docs/architecture.md`; `evaluations/README.md`; `docs/learning/phase-8-memory-framework-comparison.md` |
| T13 | Final audit closure, CI memory evaluator/artifact, scope-ID digests in memory metadata, finite obvious-PII denial, ADR acceptance, version `0.8.0`, and Phase 8 completion | Current worktree delta after `39996de`; `.github/workflows/ci.yml`; `src/dqagent/errors.py`; `src/dqagent/memory_policy.py`; `src/dqagent/memory_service.py`; `tests/test_ci_workflow.py`, `tests/test_memory_policy.py`, `tests/test_memory_service.py` |

## Final Quality Evidence

The recorded T13 release run passed:

- `ruff check .`.
- `mypy src` in strict mode.
- Full pytest with `--basetemp .local/pytest-phase8-t13`: 424 passed, 89.06% coverage.
- Focused memory boundary and closure tests: 175 passed.
- Phase 3, Phase 6, Phase 7, and Phase 8 deterministic evaluations.
- Documentation/ADR consistency checks and `git diff --check`.
- Credential and tracked-artifact scans; no credentials, SQLite databases, local reports, pytest
  temporary output, editor state, or generated artifacts are part of the intended release diff.

The Phase 8 deterministic baseline reports 13/13 cases passed, false admission `0/3`, mean
`Recall@k` `1.0` over seven applicable cases, mean `Precision@k` `1.0` over seven applicable cases,
scope leakage `0/1`, stale/forgotten recall `0/3`, harmful over-retrieval `0/2`, correction
compliance `1/1`, no-result correctness `12/12`, direct answer predicates `13/13`, and mean
projected memory context of 366.17 characters across 12 enabled cases.

These are deterministic fixture and architecture-regression measurements. They do not establish
live model extraction truth, general semantic retrieval quality, compliance-grade authorization, or
PII-classification completeness.

## ADR-0009 and Completion Decision

ADR-0009 is `Accepted` because its v1 ownership, scope, authority, lifecycle, transaction, failure,
context, and deletion statements are exercised by the implementation and tests listed above. The
acceptance records the bounded contract; it does not promote deferred production capabilities into
guarantees. ADR-0010 remains the detailed extraction-boundary decision.

Phase 8 is `Complete` because it has an end-to-end user/project-scoped memory capability, explicit
write authority and lifecycle controls, cross-session read behavior, probabilistic extraction
containment, deterministic evaluation evidence, documented dependency direction, and passing release
quality gates. Completion is limited to the v1 boundary below.

## Deferred Scope and Residual Limitations

The current implementation intentionally does not claim:

- encrypted sensitive-memory storage or compliance-grade authorization;
- forensic erasure from database files, backups, snapshots, or other storage layers;
- complete PII classification beyond the finite deterministic defense-in-depth patterns;
- persistent or managed memory vector indexes;
- unconfirmed or automatic durable writes;
- distributed tenancy, leases, or multi-worker ownership;
- background consolidation, bulk deletion, or durable audit delivery;
- live-model memory-quality evaluation or semantic/LLM-based judging.

The local store is unencrypted. Direct service callers must provide their own human-authorization UX,
and corrected superseded history remains an inspection-only lifecycle record. These are explicit
limitations, not hidden acceptance criteria for Phase 8.

## Review and Git Boundary

This note records evidence; it is not a substitute for missing T0-T4 checkpoint artifacts. No Git
metadata operation was used to create this record. The current T13 and documentation changes remain
available for user review before any manual commit. The suggested commit subject for the complete
release is:

```text
docs(memory): complete phase 8 long-term memory
```
