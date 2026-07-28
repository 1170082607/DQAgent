"""Environment-backed runtime configuration."""

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass

from dqagent.errors import ConfigurationError

DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_RUN_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_MODEL_ATTEMPTS = 3


@dataclass(frozen=True, slots=True)
class Settings:
    api_key: str
    model: str
    base_url: str | None = None
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    run_timeout_seconds: float = DEFAULT_RUN_TIMEOUT_SECONDS
    max_model_attempts: int = DEFAULT_MAX_MODEL_ATTEMPTS

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

        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ConfigurationError(
                "DQAGENT_TIMEOUT_SECONDS must be a finite number greater than zero"
            )

        raw_run_timeout = source.get(
            "DQAGENT_RUN_TIMEOUT_SECONDS", str(DEFAULT_RUN_TIMEOUT_SECONDS)
        ).strip()
        try:
            run_timeout_seconds = float(raw_run_timeout)
        except ValueError as exc:
            raise ConfigurationError("DQAGENT_RUN_TIMEOUT_SECONDS must be a number") from exc
        if not math.isfinite(run_timeout_seconds) or run_timeout_seconds <= 0:
            raise ConfigurationError(
                "DQAGENT_RUN_TIMEOUT_SECONDS must be a finite number greater than zero"
            )

        raw_attempts = source.get(
            "DQAGENT_MAX_MODEL_ATTEMPTS", str(DEFAULT_MAX_MODEL_ATTEMPTS)
        ).strip()
        try:
            max_model_attempts = int(raw_attempts)
        except ValueError as exc:
            raise ConfigurationError("DQAGENT_MAX_MODEL_ATTEMPTS must be an integer") from exc
        if max_model_attempts < 1:
            raise ConfigurationError("DQAGENT_MAX_MODEL_ATTEMPTS must be at least one")

        return cls(
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            run_timeout_seconds=run_timeout_seconds,
            max_model_attempts=max_model_attempts,
        )
