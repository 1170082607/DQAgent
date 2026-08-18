# ADR-0013: Load Repository Instructions and Skills Through Context

- Status: Accepted
- Date: 2026-08-13
- Clarifies: [ADR-0006](0006-separate-session-transcript-from-active-context.md)

## Context

ADR-0006 makes `PromptAssembler` and `ContextBuilder` responsible for explicitly requested project
knowledge and bounded active context. The implemented allowlisted knowledge path renders documents
as system messages, which is appropriate for host-configured knowledge but is unsafe as the default
authority for mutable repository content.

Phase 9 adds repository instructions and reusable skills loaded on demand. Those files may be
modified by repository authors, dependencies, or the coding agent itself. If they receive the same
authority as host-owned mandatory policy, repository content could appear to authorize tool actions,
expand workspace scope, suppress validators, or override security rules. Loading every file eagerly
would also waste context and expand the prompt-injection surface.

Instruction and skill selection is primarily a deterministic applicability and explicit-selection
problem. It is not the semantic evidence-ranking problem owned by Phase 7 retrieval, and the
resources are not durable conversation state or Phase 8 memory.

## Decision

Repository instructions and reusable skills will be loaded as provenance-bound, request-scoped
context resources through the Phase 6 context layer. They will not be loaded by `AgentRuntime`, the
tool registry, the retrieval index, the memory service, or the session store.

### Selection and loading

Repository instructions are selected by deterministic path applicability within the contained
workspace. The loader begins from a composition-selected repository root and applies the documented
root-to-target hierarchy without scanning outside that root. Nested instructions apply only to their
declared path scope.

Skills use progressive disclosure. The active context may contain a bounded catalog with stable key,
name, and description. A complete skill body is loaded only after an explicit caller or harness
selection of an unambiguous stable key. Phase 9 does not perform semantic auto-selection, implicit
model inference over arbitrary skill names, plugin discovery, or referenced-resource loading.

The Phase 9 skill contract ends at one selected `SKILL.md` body. It does not define a reference
syntax or recursively load assets, scripts, or supporting documents named by that body. A later
measured use case may extend progressive disclosure with contained references, but must first define
selection, cycle, depth, count, budget, and failure semantics. Missing, unreadable, duplicate,
ambiguous, or oversized catalog entries or skill bodies produce typed omission or failure evidence;
they do not silently expand the search scope.

### Provenance, budget, and projection

Every selected resource preserves its kind, stable key, source path or source identity, content
digest, applicability, selection reason, and authority classification. Instruction hierarchy and
skill catalog/body have explicit independent limits. A body is admitted atomically or omitted; it is
not silently truncated into an ambiguous instruction. Selected and omitted resources are exposed in
context-assembly evidence without leaking denied content.

Host-owned mandatory policy remains distinct from repository-origin and skill-origin instructions.
Mutable resources can guide repository conventions and task execution but cannot grant action
authority. They cannot change workspace scope, hard guards, policy decisions, approval requirements,
subprocess capabilities, or configured validators. The governed action boundary enforces those
rules outside the model context.

Repository and skill resources are context projections, not conversation history. Their bodies are
not appended to the durable session transcript, embedded in generated summaries, or promoted in
authority when the model repeats them. A later turn rebuilds the applicable projection from current
sources and records fresh provenance.

### Relationship to retrieval and memory

Repository resource selection does not use the Phase 7 vector index by default. Path applicability
and explicit skill keys remain deterministic. If a future measured use case needs semantic skill
discovery, it must preserve this authority and provenance contract and separately justify retrieval
indexing, ranking, and evaluation.

Repository instructions and skills are not Phase 8 memory records. They do not use memory consent,
consolidation, correction, forgetting, or recall-ranking semantics. Retrieval passages and recalled
memory remain lower-authority data and cannot select or authorize a skill-driven action.

## Consequences

- `PromptAssembler` and `ContextBuilder` retain ownership of what the model sees; Phase 9 extends
  their resource model rather than creating a second prompt or retrieval subsystem.
- ADR-0006's current system-role projection for host-configured knowledge is not reused blindly for
  mutable repository instructions. Resource origin and authority become explicit context concerns.
- Repository instructions can be path-aware and skills can be loaded progressively without placing
  all repository guidance in every model request.
- Limiting the initial skill contract to the catalog and one explicitly selected body keeps
  progressive disclosure testable without prematurely defining a plugin or transitive resource
  graph.
- Resource provenance and selected/omitted evidence make context construction inspectable and allow
  deterministic evaluation of hierarchy, conflicts, missing resources, and budget behavior.
- Context delimiters and authority labels reduce accidental instruction elevation but are not a hard
  prompt-injection sandbox. Safety still depends on ADR-0011 enforcement outside the model.
- Session persistence remains lossless conversation history and does not retain transient resource
  bodies. A session can therefore observe updated instructions on a later turn without rewriting its
  transcript.
- Phase 10 MCP instructions, prompts, and resources can reuse the provenance, budget, and authority
  concepts, but their remote source and transport require their own trust handling.
- Phase 11 plans and Phase 12 subagents may consume explicitly projected resources, but neither a
  plan nor delegated context acquires action authority from them.
- A general plugin runtime, semantic skill retrieval, background indexing, and remote skill
  distribution remain outside Phase 9.

## Implementation Evidence

`src/dqagent/repository_context.py` implements contained target-aware `AGENTS.md` hierarchy
loading, bounded skill catalog discovery, explicit single-body selection, provenance, omission
evidence, and source revalidation. `src/dqagent/context.py` projects these immutable resources
as lower-authority request-scoped user data; `src/dqagent/coding.py` loads one projection before
the single `AgentRuntime` loop and does not reload it during that run.

Evidence includes `tests/test_repository_context.py`,
`tests/test_repository_context_skills.py`, `tests/test_coding_application_t12.py`, the T14
hostile-guidance/skill case, and the current Phase 9 deterministic evaluation. The projection
remains a model-facing authority convention, not a hard prompt-injection sandbox; resources do
not become policy, approval, validator, transcript, retrieval, or memory state.

## Alternatives Considered

### Render repository instructions as existing system knowledge

Rejected because an allowlisted path proves that the host permits reading a file, not that mutable
file content owns system policy or action authorization.

### Load all repository instructions and skills at startup

Rejected because it wastes bounded context, ignores path applicability, expands prompt-injection
surface, and conflicts with the roadmap's on-demand requirement.

### Use Phase 7 retrieval to select instructions and skills

Rejected for Phase 9 because deterministic hierarchy and explicit keys solve the current selection
problem. Semantic relevance does not establish instruction authority and would add unnecessary index
lifecycle and ranking behavior.

### Let tools or `AgentRuntime` load resource bodies

Rejected because selection and prompt projection belong to context ownership. Moving them into the
runtime would couple the core loop to repository conventions, while treating loading only as a tool
would make mandatory instruction applicability depend on model choice.

### Persist loaded resources in the session transcript

Rejected because the transcript records successful conversation protocol, not reproducible context
projections. Persistence would retain stale instructions and blur source provenance.

### Allow instructions or skills to modify permission policy

Rejected because mutable context is model input, not an authorization source. Policy extensions
must be supplied explicitly by trusted application composition and pass ADR-0011's ownership rules.

### Load transitive skill references in Phase 9

Deferred because the roadmap requires reusable skills loaded on demand, which a bounded catalog and
one explicitly selected `SKILL.md` body already satisfy. A reference graph would add durable syntax,
cycle, depth, atomic-admission, and partial-failure semantics without a demonstrated Phase 9 task
that needs them.
