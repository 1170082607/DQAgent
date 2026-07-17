# DQAgent Architecture

## Status

This document describes the implemented Phase 1 architecture. Planned capabilities belong in the
[roadmap](roadmap.md) until code and accepted decisions make them part of the system.

## System Context

DQAgent is currently a local command-line application. It accepts user input, maintains conversation
history in memory, invokes an LLM provider, and prints the assistant response.

```text
User
  |
  v
CLI
  |
  v
ChatApplication
  |
  v
LLMClient protocol
  |
  v
OpenAIResponsesClient
  |
  v
OpenAI Responses API
```

## Module Responsibilities

### CLI

`dqagent.cli` parses command-line arguments, loads configuration, selects the concrete provider, and
drives one-shot or interactive chat. It contains presentation concerns but no provider request
mapping.

### Application

`dqagent.application.ChatApplication` owns the chat use case and in-memory conversation. It sends an
ordered message history through `LLMClient` and commits user and assistant messages only after a
valid provider response is returned.

The application is intentionally single-session and synchronous in Phase 1.

### Models and LLM Boundary

`dqagent.models` defines provider-neutral `Message`, `Role`, and `Completion` values.
`dqagent.llm.LLMClient` is a structural protocol comparable to a backend service port: the
application depends on required behavior rather than an SDK implementation.

### OpenAI Provider

`dqagent.providers.openai.OpenAIResponsesClient` translates neutral messages into OpenAI Responses
API input and translates the SDK response back into a neutral completion. Provider SDK types do not
cross this adapter boundary.

### Configuration

`dqagent.config.Settings` reads and validates environment variables. API keys are required at
runtime and must not be stored in repository files.

## Dependency Rules

```text
CLI -> Application -> LLM protocol and domain models
CLI -> Configuration -> Errors
OpenAI adapter -> LLM models + OpenAI SDK
```

- The application layer must not import `openai` or provider-specific types.
- Provider adapters may depend on external SDKs and translate their failures.
- The CLI is the composition root and is responsible for wiring concrete implementations.
- Planned modules must not be introduced solely to reserve future directory names.

## Conversation Consistency

`ChatApplication.send` builds a pending history, calls the provider, validates the response, and then
commits both sides of the exchange. If the provider fails, the user message is not retained. This is
similar to committing a small transaction only after the external operation succeeds.

## Error Handling

- Missing or invalid environment configuration becomes `ConfigurationError`.
- OpenAI SDK failures become `LLMProviderError`.
- Empty provider text is rejected as an invalid response.
- The CLI converts expected DQAgent errors into a concise message and non-zero exit code.

Retries are intentionally deferred until the runtime phase, where retry policy can use explicit
error classification and lifecycle events.

## Testing Strategy

- Application tests use an in-memory fake implementing the `LLMClient` protocol.
- Provider tests verify request and response translation without network access.
- Configuration tests cover missing and invalid environment values.
- CI runs Ruff, mypy strict mode, and pytest on Python 3.11.

No live provider test is run in CI because it would require credentials, incur cost, and introduce
network nondeterminism.

## Current Limitations

- Conversation history is process-local and unbounded.
- Requests and output are non-streaming.
- Only the OpenAI Responses API adapter is implemented.
- There are no tools, agent loop, runtime events, persistence, or observability pipeline.

## Related Decisions

- [ADR-0001: Provider-neutral LLM boundary](adr/0001-provider-neutral-llm-boundary.md)
