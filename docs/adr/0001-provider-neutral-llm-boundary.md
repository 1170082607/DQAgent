# ADR-0001: Provider-Neutral LLM Boundary

- Status: Accepted
- Date: 2026-07-16

## Context

Phase 1 requires one concrete model provider, but later learning phases need to study tool calling,
runtime behavior, and agent loops without coupling those concepts to one SDK's request and response
types.

Calling the OpenAI SDK directly from the chat application would be initially shorter, but provider
details would spread into conversation state and later tool-loop logic.

## Decision

The application depends on a small `LLMClient` protocol that accepts provider-neutral conversation
items and optional tool definitions, then returns a provider-neutral completion. Phase 2 expanded
the original text-only values with tool calls and tool results without changing the dependency
direction.

The first adapter uses the OpenAI Python SDK and the Responses API. Request mapping, SDK exceptions,
and response extraction remain inside `OpenAIResponsesClient`. The CLI acts as the composition root
that wires the adapter into the application.

## Consequences

- Application tests can use small fakes without mocking the OpenAI SDK.
- Provider-specific types cannot become application-layer contracts accidentally.
- A second provider can be evaluated against an existing behavioral boundary.
- The abstraction must remain small; provider-specific capabilities may require explicit extension
  instead of forcing all providers into a lowest-common-denominator interface.
- Phase 1 contains one interface with one implementation. It is justified by an external system
  boundary, not by speculative provider count.

## Alternatives Considered

### Call the OpenAI SDK directly from `ChatApplication`

Rejected because it couples the primary use case and its tests to one provider and makes the future
agent loop responsible for SDK-specific translation.

### Build a generic HTTP client instead of using the SDK

Rejected because authentication, transport errors, response schemas, and API evolution are not the
learning objective of Phase 1. The SDK is used as an infrastructure dependency, while the Agent
abstractions remain project-owned.
