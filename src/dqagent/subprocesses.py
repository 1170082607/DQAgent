"""Provider-neutral technical capabilities for subprocess backends.

This module deliberately contains no subprocess request, result, or execution
code.  A capability is a tested backend guarantee, not a policy decision or a
user approval.
"""

from collections.abc import Iterable
from enum import StrEnum

__all__ = [
    "IsolationCapability",
    "normalize_isolation_capabilities",
    "validate_isolation_capabilities",
]


class IsolationCapability(StrEnum):
    """A technical subprocess guarantee that a backend can declare and test."""

    DIRECT_ARGV = "direct_argv"
    WORKING_DIRECTORY_CONTROL = "working_directory_control"
    ALLOWLISTED_ENVIRONMENT = "allowlisted_environment"
    NO_STDIN = "no_stdin"
    WALL_TIME_LIMIT = "wall_time_limit"
    BOUNDED_OUTPUT = "bounded_output"
    DIRECT_CHILD_TERMINATION = "direct_child_termination"
    DIRECT_CHILD_REAP = "direct_child_reap"
    PROCESS_GROUP_TERMINATION = "process_group_termination"
    DESCENDANT_TREE_TERMINATION = "descendant_tree_termination"


def normalize_isolation_capabilities(
    capabilities: Iterable[IsolationCapability],
) -> frozenset[IsolationCapability]:
    """Validate and freeze a backend capability declaration."""

    if isinstance(capabilities, (str, bytes)):
        raise TypeError("isolation capabilities must be an iterable of IsolationCapability")
    try:
        values = tuple(capabilities)
    except TypeError as error:
        raise TypeError(
            "isolation capabilities must be an iterable of IsolationCapability"
        ) from error
    if any(not isinstance(value, IsolationCapability) for value in values):
        raise ValueError("isolation capabilities must contain only IsolationCapability values")
    return frozenset(values)


def validate_isolation_capabilities(
    capabilities: Iterable[IsolationCapability],
) -> frozenset[IsolationCapability]:
    """Compatibility-named validator used by action and backend contracts."""

    return normalize_isolation_capabilities(capabilities)
