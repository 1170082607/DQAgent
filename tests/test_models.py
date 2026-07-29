import pytest

from dqagent.models import Completion, Message, Role, TokenUsage


def test_token_usage_requires_non_negative_integers() -> None:
    assert TokenUsage(1, 2, 3).total_tokens == 3

    with pytest.raises(ValueError, match="input tokens"):
        TokenUsage(-1, 2, 1)

    with pytest.raises(ValueError, match="output tokens"):
        TokenUsage(1, True, 2)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Message(Role.USER, "  "),
        lambda: Completion("\n"),
    ],
)
def test_text_models_reject_blank_content(factory: object) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        factory()  # type: ignore[operator]
