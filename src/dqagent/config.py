"""Environment-backed runtime configuration."""

import os
from collections.abc import Mapping
from dataclasses import dataclass

from dqagent.errors import ConfigurationError

DEFAULT_TIMEOUT_SECONDS = 60.0


@dataclass(frozen=True, slots=True)
class Settings:
    api_key: str
    model: str
    base_url: str | None = None
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Settings":
        source = os.environ if environ is None else environ
        api_key = source.get("OPENAI_API_KEY", "").strip()
        model = source.get("DQAGENT_MODEL", "").strip()
        base_url = source.get("OPENAI_BASE_URL", "").strip() or None

        missing = [
            name
            for name, value in (("OPENAI_API_KEY", api_key), ("DQAGENT_MODEL", model))
            if not value
        ]
        if missing:
            raise ConfigurationError(
                "missing required environment variables: " + ", ".join(missing)
            )

        raw_timeout = source.get(
            "DQAGENT_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS)
        ).strip()
        try:
            timeout_seconds = float(raw_timeout)
        except ValueError as exc:
            raise ConfigurationError("DQAGENT_TIMEOUT_SECONDS must be a number") from exc

        if timeout_seconds <= 0:
            raise ConfigurationError("DQAGENT_TIMEOUT_SECONDS must be greater than zero")

        return cls(
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
        )
