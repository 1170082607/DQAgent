# DQAgent

DQAgent is an engineering-first learning project for building AI Agent capabilities incrementally
from foundational components. Its purpose is to understand the design of production-oriented agent
systems instead of treating frameworks as black boxes.

**Status:** Pre-alpha. Phase 4 is implemented: the provider-neutral, observable Phase 3 agent now has
a versioned behavioral evaluation harness with deterministic CI and explicit live-model modes.
Phase 5 will add deterministic workflow and durable execution semantics.

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

## Phase 4 Capabilities

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
- Environment-based configuration with explicit validation.
- Unit tests, Ruff linting, mypy strict type checking, and GitHub Actions CI.

Streaming, persistence, hard execution isolation, approval gates, durable telemetry delivery,
LLM-as-judge, repeated live-model sampling, and workflow orchestration are intentionally deferred.
Evaluation, security, and observability now remain continuous constraints for later phases.

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

Useful interactive commands:

- `/reset`: clear conversation history while preserving the system prompt.
- `/exit` or `/quit`: end the session.

## Repository Layout

```text
src/dqagent/       Application and provider code
src/dqagent/runtime.py Observable agent runtime and retry policy
src/dqagent/execution.py Run identity, deadline, and cancellation context
src/dqagent/evaluation.py Behavioral case loader, runner, checks, and reports
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
