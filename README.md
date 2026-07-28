# DQAgent

DQAgent is an engineering-first learning project for building AI Agent capabilities incrementally
from foundational components. Its purpose is to understand the design of production-oriented agent
systems instead of treating frameworks as black boxes.

**Status:** Pre-alpha. Phase 3 is implemented: a provider-neutral tool-using agent with an
observable, bounded runtime backed by the OpenAI Responses API. Phase 4 will establish behavioral
evaluation before the project adds workflow, persistence, or broader autonomy.

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

## Phase 3 Capabilities

- Interactive and one-shot command-line chat.
- In-memory conversation history with optional system prompts.
- A provider-neutral `LLMClient` application boundary.
- An OpenAI Responses API adapter.
- JSON Schema tool definitions and an explicit tool registry.
- A bounded model/tool/observation loop with repeated-call protection.
- Structured recovery observations for invalid arguments, unknown tools, timeouts, and tool errors.
- A built-in `current_time` tool that accepts a numeric UTC offset.
- Per-run IDs, metadata, deadlines, and cooperative cancellation.
- Ordered lifecycle, model-attempt, retry, and tool-call events.
- Classified provider failures with bounded retry of transient model requests.
- Event sinks for tracing, metrics, and audit adapters.
- Environment-based configuration with explicit validation.
- Unit tests, Ruff linting, mypy strict type checking, and GitHub Actions CI.

Behavioral evaluation, streaming, persistence, hard execution isolation, approval gates, durable
telemetry delivery, and workflow orchestration are intentionally deferred to later phases. From
Phase 4 onward, evaluation, security, and observability are continuous constraints rather than one
final hardening step.

## Installation

DQAgent requires Python 3.11 or newer.

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
```

Activate the virtual environment using the command appropriate for your shell, then configure the
required environment variables. `.env.example` is a reference file; DQAgent does not automatically
load `.env` files.

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

## Usage

Start an interactive conversation:

```bash
dqagent
```

Send one message and exit:

```bash
dqagent --message "Explain the difference between an agent loop and a workflow."
```

The model can call `current_time` when a request needs current time information. Tool use is
automatic; tool calls and observations stay inside the agent loop and only the final answer is
printed.

Useful interactive commands:

- `/reset`: clear conversation history while preserving the system prompt.
- `/exit` or `/quit`: end the session.

## Repository Layout

```text
src/dqagent/       Application and provider code
src/dqagent/runtime.py Observable agent runtime and retry policy
src/dqagent/execution.py Run identity, deadline, and cancellation context
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
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for change guidelines.

## Documentation

- [Roadmap](docs/roadmap.md)
- [Architecture](docs/architecture.md)
- [Architecture Decision Records](docs/adr/README.md)
- [Learning Notes](docs/learning/README.md)

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE).
