import pytest

from dqagent.models import Completion, Message, Role


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
