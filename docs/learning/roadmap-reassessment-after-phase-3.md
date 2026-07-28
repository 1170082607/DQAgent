# Roadmap Reassessment After Phase 3

## Scope

This note reassesses DQAgent after the runtime phase against its target: developing the judgment
needed to design, evaluate, and operate production-oriented AI Agent systems. It records why the
post-Phase-3 roadmap changed and separates source evidence from project-specific conclusions.

Sources were read on 2026-07-28 at:

- Hello-Agents commit `6c616938c521c89bc4b2bf001bf237d259f1726b`.
- learn-claude-code commit `a9cafe953aa714f9cb1171f217d96bd2734bbcc7`.
- The OpenAI Agents SDK and EINO commits recorded in the existing
  [Phase 2 comparison](phase-2-framework-comparison.md).

No source code was copied from the reference projects.

## What Phases 0-3 Established

DQAgent already has the essential harness kernel: a provider-neutral model boundary, explicit tool
contracts, validation of untrusted arguments, a bounded model/tool loop, per-run context, classified
errors, controlled model retry, cooperative cancellation, and ordered events. Conversation state and
run lifecycle also have separate ownership.

These decisions create a better base for later experiments than a feature-rich tutorial framework
whose error and state semantics are implicit. The main gap is no longer basic execution. It is the
ability to prove behavioral quality while adding state, context, external knowledge, and broader
action authority.

## Evidence From Reference Projects

### Hello-Agents

Hello-Agents provides a broad curriculum. After classic paradigms and framework construction, it
covers memory and retrieval, context engineering, MCP/A2A/ANP protocols, agent evaluation, and
integrated applications. Its evaluation chapter distinguishes tool-call evaluation (BFCL), general
assistant benchmarks (GAIA), LLM judges, pairwise win rate, and human verification.

Reusable lesson: an Agent engineer needs more than an execution loop. Context, retrieval,
protocols, evaluation, and a complete application are separate competencies.

Project-specific conclusion: DQAgent should not copy the chapter order. Waiting until after memory,
RAG, and protocols to add evaluation would leave several probabilistic phases without a regression
baseline.

### learn-claude-code

learn-claude-code keeps one model-controlled loop and incrementally adds harness mechanisms around
it. Its current track introduces permissions and hooks before broad autonomy, compaction before
cross-session memory, durable tasks before background work and teams, and workspace isolation before
comprehensive multi-agent operation. MCP tools join the same tool pool rather than creating a second
execution path.

The project also distinguishes two forms of planning state:

- An in-session todo list guides the current execution.
- A persisted task graph carries dependencies, ownership, and recovery across sessions.

Reusable lesson: capabilities such as permission checks, context compaction, memory, durable tasks,
and teams are harness layers. They should preserve a small loop and reuse common execution,
permission, and observation paths.

Project-specific conclusion: DQAgent should build a safe coding harness before adding planning and
multiple agents. A capable model with well-designed tools is the baseline; orchestration is added
only when evaluation identifies a real limitation.

## Gap Analysis

| Competency | State after Phase 3 | Roadmap response |
| --- | --- | --- |
| Harness correctness | Strong unit coverage for deterministic runtime behavior | Preserve and extend per phase |
| Behavioral quality | No versioned cases or model evaluation baseline | New Phase 4 |
| Durable orchestration | No workflow state, checkpoint, or replay contract | Phase 5 |
| Context engineering | Unbounded in-memory messages and one system string | Phase 6 |
| External knowledge | No ingestion, retrieval, grounding, or citations | Phase 7 |
| Long-term memory | No selection, consolidation, correction, or deletion policy | Phase 8 |
| Protocol integration | Local tools only | Phase 10 |
| Safe action environment | No file, shell, permission, approval, or hard isolation boundary | Phase 9 |
| Long-running autonomy | No durable tasks, background work, or bounded replanning | Phase 11 |
| Collaboration | No delegation, ownership, mailbox, or workspace isolation | Phase 12 |
| Operations | Best-effort events, no cost/SLO/tenancy/deployment story | Continuous track and Phase 13 |

## Sequencing Decisions

### Move evaluation to Phase 4

Tests answer whether the harness follows deterministic contracts. Evaluations answer whether the
model and harness produce acceptable outcomes and trajectories. RAG, memory, planning, and
multi-agent coordination cannot be assessed honestly with unit tests alone, so a small evaluation
harness must precede them.

### Keep workflow, but narrow its claim

A workflow engine remains valuable for known control flow, durable checkpoints, interrupts, and
recovery. It should not encode every possible model decision or be presented as the source of
agency. Agent nodes may use a model-controlled loop; the graph owns deterministic orchestration.

### Put context before retrieval and memory

Retrieval and memory only help when the system can decide what enters a finite model context.
Session transcripts, active context, retrieved knowledge, and selected memories are different data
products. Building context ownership first prevents them from collapsing into one message list.

### Put RAG before long-term memory

RAG establishes ingestion, indexing, retrieval, provenance, and retrieval evaluation against
external documents. Long-term memory can reuse storage and retrieval mechanics, but adds policy:
what to extract, retain, consolidate, correct, forget, and expose to a user. Treating memory as a
specialized RAG index would hide those policy decisions.

### Build the coding harness before adding MCP tools

Workspace-scoped file and command tools provide a concrete reason to design allow, deny, approval,
hook, containment, and isolation semantics. MCP can then add dynamically discovered external tools
to an existing permission, event, and evaluation path. Introducing MCP first would either expose
external side effects without governance or create a permission abstraction without a realistic
local use case. MCP remains an integration protocol, not an agent capability by itself.

### Build one safe coding agent before planning and teams

A coding harness provides a realistic vertical test for tools, context, permissions, observation,
and evaluation. Planning, background work, and multi-agent coordination should improve measured
failures in that baseline. This prevents architecture driven by imagined scale.

### Make production concerns continuous

The old final "production hardening" phase was too late for trust boundaries and evaluation.
Security, observability, durability, cost, and regression evidence now grow with each feature. The
final phase integrates and stress-tests them rather than introducing them for the first time.

## Rejected Alternatives

### Continue directly to the original Workflow Engine phase

Rejected because the project would gain more execution paths without a way to compare outcomes or
detect model-behavior regressions.

### Follow Hello-Agents chapter order exactly

Rejected because it is a broad course for varied backgrounds. DQAgent already has backend
engineering foundations and benefits from introducing evaluation earlier and low-code/model
training topics only when role requirements justify them.

### Follow learn-claude-code mechanism order exactly

Rejected because it is a coding-harness tutorial, while DQAgent also aims to teach durable
workflow, RAG, memory, and general protocol boundaries. Its "stable loop, layered harness"
principle is reused; its product-specific sequence is not copied.

### Add multi-agent coordination soon after workflow

Rejected because delegation adds distributed ownership, cancellation, isolation, and cost without
a single-agent evaluation baseline. A function, tool, or workflow node is usually the cheaper
boundary.

## Sources

- [Hello-Agents README](https://github.com/datawhalechina/hello-agents/blob/6c616938c521c89bc4b2bf001bf237d259f1726b/README_EN.md)
- [Hello-Agents: Building Your Agent Framework](https://github.com/datawhalechina/hello-agents/blob/6c616938c521c89bc4b2bf001bf237d259f1726b/docs/chapter7/Chapter7-Building-Your-Agent-Framework.md)
- [Hello-Agents: Memory and Retrieval](https://github.com/datawhalechina/hello-agents/blob/6c616938c521c89bc4b2bf001bf237d259f1726b/docs/chapter8/Chapter8-Memory-and-Retrieval.md)
- [Hello-Agents: Context Engineering](https://github.com/datawhalechina/hello-agents/blob/6c616938c521c89bc4b2bf001bf237d259f1726b/docs/chapter9/Chapter9-Context-Engineering.md)
- [Hello-Agents: Agent Communication Protocols](https://github.com/datawhalechina/hello-agents/blob/6c616938c521c89bc4b2bf001bf237d259f1726b/docs/chapter10/Chapter10-Agent-Communication-Protocols.md)
- [Hello-Agents: Agent Performance Evaluation](https://github.com/datawhalechina/hello-agents/blob/6c616938c521c89bc4b2bf001bf237d259f1726b/docs/chapter12/Chapter12-Agent-Performance-Evaluation.md)
- [learn-claude-code README](https://github.com/shareAI-lab/learn-claude-code/blob/a9cafe953aa714f9cb1171f217d96bd2734bbcc7/README.md)
- [learn-claude-code: Permission](https://github.com/shareAI-lab/learn-claude-code/blob/a9cafe953aa714f9cb1171f217d96bd2734bbcc7/s03_permission/README.en.md)
- [learn-claude-code: Context Compact](https://github.com/shareAI-lab/learn-claude-code/blob/a9cafe953aa714f9cb1171f217d96bd2734bbcc7/s08_context_compact/README.en.md)
- [learn-claude-code: Task System](https://github.com/shareAI-lab/learn-claude-code/blob/a9cafe953aa714f9cb1171f217d96bd2734bbcc7/s12_task_system/README.en.md)
- [learn-claude-code: MCP Tools](https://github.com/shareAI-lab/learn-claude-code/blob/a9cafe953aa714f9cb1171f217d96bd2734bbcc7/s19_mcp_plugin/README.en.md)
