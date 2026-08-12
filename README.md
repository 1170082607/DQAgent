# DQAgent

DQAgent is an engineering-first learning project for building AI Agent capabilities incrementally
from foundational components. Its purpose is to understand the design of production-oriented agent
systems instead of treating frameworks as black boxes.

**Status:** Pre-alpha. Phase 8 T11 is implemented: DQAgent supports durable bounded sessions,
explicit local RAG, optional exact-scope read-only memory recall, and explicit source-to-transient-
candidate extraction with provider-neutral boundaries, citation/memory provenance, and deterministic
regression evaluations.

## Goals

- Learn the core abstractions and execution models behind modern AI Agents.
- Build small, testable capabilities before comparing them with mature frameworks.
- Evolve from model invocation to tools and runtime, then through evaluation, durable workflow,
  context engineering, retrieval, memory, safe action harnesses, integrations, planning, and
  multi-agent collaboration.
- Apply backend engineering principles such as explicit boundaries, dependency inversion,
  observability, failure handling, and incremental delivery.
- Learn when an agent loop, deterministic workflow, or multi-agent design is justified by evidence.

## Non-goals

- Reimplement LangGraph, OpenHands, AutoGen, or the OpenAI Agents SDK feature for feature.
- Present experimental code as a production-ready Agent platform.
- Add abstractions for roadmap stages that have not been implemented.
- Treat workflow graphs, prompt chains, or additional agents as substitutes for model capability.

## Phase 7 Capabilities

- Interactive and one-shot command-line chat.
- In-memory conversation history with optional system prompts.
- A provider-neutral `LLMClient` application boundary.
- An OpenAI Responses API adapter.
- A llama.cpp `llama-server` adapter using its OpenAI-compatible Chat Completions endpoint.
- JSON Schema tool definitions and an explicit tool registry.
- A bounded model/tool/observation loop with repeated-call protection.
- Structured recovery observations for invalid arguments, unknown tools, timeouts, and tool errors.
- Built-in `current_time` and deterministic `get_weather` demonstration tools.
- Per-run IDs, metadata, deadlines, and cooperative cancellation.
- Ordered lifecycle, model-attempt, retry, and tool-call events.
- Classified provider failures with bounded retry of transient model requests.
- Event sinks for tracing, metrics, and audit adapters.
- Versioned evaluation cases with inputs, scripted fixtures, expected outcomes, and trace constraints.
- Deterministic predicates for final answers, tool names and arguments, tool outcomes, and event order.
- Per-case latency, model-attempt, tool-call, and provider-reported token metrics.
- Credential-free deterministic evaluation for CI and explicit, credentialed live-model evaluation.
- A committed Phase 3 behavioral baseline and BFCL/GAIA evaluation comparison.
- Validated acyclic workflow definitions with sequential and conditional transitions.
- Bounded parallel leaf branches with disjoint-key merge and cooperative sibling cancellation.
- Explicit running, interrupted, completed, failed, cancelled, and timed-out workflow states.
- Compare-and-swap checkpoints backed by memory or atomic local JSON files.
- Resume from the last uncompleted node and replay from original input under a new workflow ID.
- Stable node idempotency keys for external side-effect deduplication.
- Shared `RunContext`, lifecycle events, deadlines, cancellation, and event sinks across agent and
  workflow runtimes.
- Durable session IDs, complete provider-neutral transcripts, resume, and CAS conflict detection.
- In-memory and atomic local JSON session stores with explicit process-local concurrency semantics.
- Named prompt sections and allowlisted project knowledge loaded only by requested key.
- Character-estimated context budgets with a reserve for tool schemas, output, and tokenizer error.
- Whole-turn trimming and JSONL structural compaction that never emit partial tool records.
- Optional structural/model summaries with source, size, and compaction-loss provenance.
- `CONTEXT_ASSEMBLED` events with budget, retained/omitted turn, knowledge, and summary metadata.
- A deterministic context evaluation suite for constraint retention, overflow recovery, and known
  compaction loss.
- Document identity, metadata, whitespace-aware chunking, exact duplicate folding, and explicit
  replace/delete indexing behavior.
- Provider-neutral document/query embedding, vector-store, and retriever contracts with deterministic
  local implementations.
- Citation-labelled untrusted retrieval data in lower-authority user context, with trusted retrieval
  policy retained in system context and no retrieved content in durable session history.
- One coordinator-owned run across retrieval, context, and model/tool work, with stage scopes that
  cannot start or terminate the lifecycle.
- Retrieval start, completion, and failure events with a committed `Recall@k`/MRR/no-result suite.
- A separate live answer-level RAG suite for lexical claim/citation linkage, insufficient evidence,
  and adversarial retrieved instructions.
- Phase 8 memory domain values, deterministic admission policy, and transactional in-memory/SQLite
  memory stores.
- A model-free `MemoryService` with transient proposal/preview, exact digest confirmation, list/show,
  atomic correction, forgetting tombstones, expiry materialization, and fail-closed errors.
- An independent `dqagent-memory` CLI for remember, list, show, correct, and forget without model
  credentials; it defaults to `.local/memory.sqlite3` and accepts an explicit `--database` path.
- Content-free memory operation metadata for event-ready audit attributes; memory content remains in
  explicit result payloads only.
- Request-time memory recall projected into bounded context as lower-authority untrusted user data,
  with atomic record omission, independent budgeting, and content-free projection evidence.
- Optional read-only memory recall in durable `dqagent` sessions with an explicit SQLite database
  and exact user/project scope; typed recall dependency failures fall back without memory.
- A pure `MemoryExtractor` boundary for one committed, bounded session turn, with deterministic
  fixtures and a model-assisted strict JSON path that has no mutation tools or store access.
- Source-derived extraction provenance, independent extraction run IDs, and an explicit pipeline that
  sends every candidate through policy preview and exact digest confirmation.
- A credential-free Phase 8 memory evaluation suite and deterministic baseline that separately reports
  admission, ranking, context selection, answer utilization, no-result semantics, and direct predicates.
- Environment-based configuration with explicit validation.
- Unit tests, Ruff linting, mypy strict type checking, and GitHub Actions CI.

Automatic memory writes, streaming, hard execution isolation, approval policy, distributed
session/workflow leases, durable telemetry delivery, LLM-as-judge, and repeated live-model sampling
remain deferred. Evaluation, security, durability, and observability remain continuous constraints
for later phases.

## Installation

DQAgent requires Python 3.11 or newer.

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
```

Activate the virtual environment using the command appropriate for your shell, then configure the
required environment variables. The `dqagent` CLI and live evaluation mode load a local `.env` file
when present; `.env.example` documents the supported variables and remains safe to commit.

PowerShell:

```powershell
$env:OPENAI_API_KEY = "your-api-key"
$env:DQAGENT_MODEL = "your-model-id"
```

Bash:

```bash
export OPENAI_API_KEY="your-api-key"
export DQAGENT_MODEL="your-model-id"
```

For an OpenAI-compatible endpoint, optionally set `OPENAI_BASE_URL`. Request timeout defaults to 60
seconds and can be changed with `DQAGENT_TIMEOUT_SECONDS`. A complete agent run defaults to 120
seconds and three model attempts; configure these with `DQAGENT_RUN_TIMEOUT_SECONDS` and
`DQAGENT_MAX_MODEL_ATTEMPTS`.

## Local llama.cpp Provider

Start `llama-server` with a GGUF model. Tool use requires a model and chat template that support
function calling; enable Jinja template processing when required by the selected llama.cpp build and
model:

```powershell
llama-server -m C:\models\tool-capable-model.gguf --host 127.0.0.1 --port 8080 --jinja
```

Read the model identifier from `http://127.0.0.1:8080/v1/models`, then configure DQAgent:

```powershell
$env:DQAGENT_PROVIDER = "llama_cpp"
$env:DQAGENT_MODEL = "your-local-model-id"
$env:LLAMA_CPP_BASE_URL = "http://127.0.0.1:8080/v1"
dqagent --message "What time is it in UTC+8?"
```

`LLAMA_CPP_BASE_URL` defaults to `http://127.0.0.1:8080/v1`. No API key is required by default. If
the server was started with authentication, set `LLAMA_CPP_API_KEY`. `DQAGENT_BASE_URL` or the CLI
`--base-url` option overrides the provider-specific URL.

The adapter uses Chat Completions because that is llama-server's compatible conversation/tool
endpoint. OpenAI continues to use the Responses API; both remain behind the same `LLMClient`
boundary. The compatibility reference is the
[llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/tree/6ba5ef247034cd57201360aed246d98f5a404d92/tools/server)
resolved on 2026-07-28.

## Usage

Start an interactive conversation:

```bash
dqagent
```

Send one message and exit:

```bash
dqagent --message "Explain the difference between an agent loop and a workflow."
```

Create a durable session, then resume it from a later process using the same ID:

```bash
dqagent --session-id learning-1 --message "Remember that the project constraint is ALPHA."
dqagent --session-id learning-1 --message "Which project constraint applies?"
```

Session files default to `.local/sessions`; override the location with `--session-dir`. The active
context defaults to a 32,000-character estimate with 4,000 characters reserved for tool schemas,
output, and tokenizer error. `--context-max-characters` changes the total estimate. This budget is
provider-neutral and does not claim exact token counting.

The model can call `current_time` when a request needs current time information and `get_weather`
when a demonstration needs a weather lookup for a non-empty city and a `YYYY-MM-DD` date.
`get_weather` always returns structured, deterministic sunny data marked with `is_demo: true` and
does not contact a weather service; its output must not be treated as a real forecast. Tool use is
automatic; tool calls and observations stay inside the agent loop and only the final answer is
printed.

Run the credential-free behavioral baseline used by CI:

```bash
dqagent-eval --mode deterministic
```

Run the live-capable subset against the configured provider only when explicitly intended:

```bash
dqagent-eval --mode live --output .local/evaluations/live-report.json
```

Deterministic evaluation proves the harness and evaluators are stable; it does not measure model
quality. Live reports are model samples and should be repeated before drawing regression conclusions.
See [evaluations/README.md](evaluations/README.md) for the case contract and report semantics.

Run the credential-free Phase 6 context regression suite:

```bash
dqagent-context-eval --output .local/evaluations/context-report.json
```

Build a small local retrieval index, inspect it without an LLM, and use it to ground a durable
session:

```bash
dqagent-index upsert project-readme README.md --source README.md
dqagent-index query "How are durable sessions resumed?"
dqagent --session-id grounded-1 --retrieval-index .local/retrieval/index.json \
  --message "How are durable sessions resumed? Cite the source."
```

`upsert` replaces every previous chunk for the same document ID; stale chunks from an older version
do not remain searchable. Remove a document with `dqagent-index delete project-readme`. The local
hashing embedding is deterministic and credential-free, but relies on lexical overlap and is not a
production semantic embedding model. Retrieved passages enter only the active request, are labelled
as untrusted external data, and are not written to the durable transcript. Answers can cite rank-local
IDs such as `[R1]`; the application result retains the source, offsets, digest, score, and metadata,
and separates valid cited IDs from retrieved-but-uncited and unknown answer IDs.

Run the independent retrieval baseline:

```bash
dqagent-retrieval-eval --output .local/evaluations/retrieval-report.json
```

Manage explicit memory without starting an agent or configuring a provider:

```bash
dqagent-memory remember --scope-kind user --scope-id user-7 \
  --kind preference --topic response.language \
  --content "The user prefers concise Chinese answers."
dqagent-memory list --scope-kind user --scope-id user-7
dqagent-memory show --scope-kind user --scope-id user-7 --memory-id MEMORY_ID
dqagent-memory correct --scope-kind user --scope-id user-7 --memory-id MEMORY_ID \
  --kind preference --topic response.language \
  --content "The user prefers concise English answers."
dqagent-memory forget --scope-kind user --scope-id user-7 --memory-id MEMORY_ID
```

`remember` and `correct` print the exact transient candidate, provenance summary, expiry, and
digest before asking for `yes` or `confirm`. `forget` prints the exact target before the same
confirmation. A rejection or EOF does not create a pending queue or mutate the database; there is
no bulk-clear command and no `--yes` bypass. Successful output is stdout, sanitized failures are
stderr, and the CLI never reads `DQAGENT_MODEL`, provider credentials, or a session ID to choose
the memory scope.

Run the separate answer-level RAG suite with the configured live provider. Repeat it before drawing
model-quality conclusions:

```bash
dqagent-rag-answer-eval --output .local/evaluations/rag-answer-report.json
```

Run the durable workflow example. The first process checkpoints and interrupts after `prepare`; the
second claims the checkpoint and continues from `apply`:

```bash
python examples/workflow_resume.py start demo-1
python examples/workflow_resume.py resume demo-1
```

Workflow checkpoints provide at-least-once recovery. A node that performs external side effects must
pass `context.metadata["idempotency_key"]` to a system that can deduplicate retries. Checkpointing
cannot guarantee exactly-once effects across a process crash.

Useful interactive commands:

- `/reset`: clear in-memory history while preserving the system prompt. Durable sessions reject
  reset; use a new session ID so the existing transcript remains recoverable.
- `/exit` or `/quit`: end the session.

## Repository Layout

```text
src/dqagent/       Application and provider code
src/dqagent/runtime.py Observable agent runtime and retry policy
src/dqagent/execution.py Run identity, deadline, and cancellation context
src/dqagent/events.py Shared agent/workflow lifecycle events
src/dqagent/evaluation.py Behavioral case loader, runner, checks, and reports
src/dqagent/context.py Prompt assembly, knowledge loading, budgets, and compaction
src/dqagent/session.py Durable transcript model and session stores
src/dqagent/context_evaluation.py Deterministic context regression runner
src/dqagent/retrieval.py Ingestion, embedding, local index, retrieval, and provenance
src/dqagent/retrieval_evaluation.py Recall-oriented retrieval regression runner
src/dqagent/memory.py Domain values and policy contracts for selected memory
src/dqagent/memory_consolidation.py Store-neutral deterministic consolidation
src/dqagent/memory_service.py Explicit memory management application service
src/dqagent/memory_store.py Exact-scope transactional memory stores
src/dqagent/memory_extraction.py Source-to-transient-candidate extraction boundary
src/dqagent/memory_cli.py Independent model-free memory management CLI
src/dqagent/memory_evaluation.py Production-path Phase 8 memory evaluation runner and metrics
src/dqagent/memory_evaluation_cli.py Credential-free Phase 8 memory evaluation CLI
src/dqagent/workflow.py Deterministic workflow definition and runner
src/dqagent/checkpoint.py Workflow checkpoint contract and stores
evaluations/        Versioned cases and committed baseline reports
tests/             Automated tests
docs/roadmap.md    Authoritative development plan
docs/architecture.md Current implemented architecture
docs/adr/          Durable architectural decisions
docs/learning/     Experiments, research, and source-reading notes
examples/          Runnable usage examples
```

## Development

```bash
ruff check .
mypy src
pytest
dqagent-eval --mode deterministic
dqagent-context-eval
dqagent-retrieval-eval
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for change guidelines.

## Documentation

- [Roadmap](docs/roadmap.md)
- [Architecture](docs/architecture.md)
- [Architecture Decision Records](docs/adr/README.md)
- [Learning Notes](docs/learning/README.md)
- [Behavioral Evaluations](evaluations/README.md)

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE).
