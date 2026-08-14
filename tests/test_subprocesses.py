import pytest

from dqagent.subprocesses import (
    IsolationCapability,
    normalize_isolation_capabilities,
    validate_isolation_capabilities,
)


def test_isolation_capabilities_are_technical_and_immutable() -> None:
    capabilities = normalize_isolation_capabilities(
        (
            IsolationCapability.DIRECT_ARGV,
            IsolationCapability.BOUNDED_OUTPUT,
            IsolationCapability.DIRECT_ARGV,
        )
    )

    assert capabilities == frozenset(
        {IsolationCapability.DIRECT_ARGV, IsolationCapability.BOUNDED_OUTPUT}
    )
    assert validate_isolation_capabilities(capabilities) == capabilities
    assert all(
        term not in capability.value
        for capability in IsolationCapability
        for term in ("approval", "deny", "policy")
    )


@pytest.mark.parametrize("value", ["direct_argv", ("direct_argv",), (None,)])
def test_isolation_capability_validation_rejects_malformed_values(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        normalize_isolation_capabilities(value)  # type: ignore[arg-type]
