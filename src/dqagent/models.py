"""Provider-neutral chat models."""

from dataclasses import dataclass
from enum import StrEnum


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ValueError("message content must not be empty")


@dataclass(frozen=True, slots=True)
class Completion:
    content: str
    response_id: str | None = None
    model: str | None = None

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ValueError("completion content must not be empty")
