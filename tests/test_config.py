import pytest

from dqagent.config import (
    DEFAULT_MAX_MODEL_ATTEMPTS,
    DEFAULT_RUN_TIMEOUT_SECONDS,
    DEFAULT_TIMEOUT_SECONDS,
    Settings,
)
from dqagent.errors import ConfigurationError


def test_settings_load_required_and_optional_values() -> None:
    settings = Settings.from_env(
        {
            "OPENAI_API_KEY": "secret",
            "DQAGENT_MODEL": "model-name",
            "OPENAI_BASE_URL": "https://example.test/v1",
            "DQAGENT_TIMEOUT_SECONDS": "12.5",
            "DQAGENT_RUN_TIMEOUT_SECONDS": "30",
            "DQAGENT_MAX_MODEL_ATTEMPTS": "4",
        }
    )

    assert settings == Settings(
        api_key="secret",
        model="model-name",
        base_url="https://example.test/v1",
        timeout_seconds=12.5,
        run_timeout_seconds=30,
        max_model_attempts=4,
    )


def test_settings_use_default_timeout() -> None:
    settings = Settings.from_env(
        {"OPENAI_API_KEY": "secret", "DQAGENT_MODEL": "model-name"}
    )

    assert settings.timeout_seconds == DEFAULT_TIMEOUT_SECONDS
    assert settings.run_timeout_seconds == DEFAULT_RUN_TIMEOUT_SECONDS
    assert settings.max_model_attempts == DEFAULT_MAX_MODEL_ATTEMPTS


def test_settings_report_all_missing_required_values() -> None:
    with pytest.raises(ConfigurationError) as error:
        Settings.from_env({})

    assert "OPENAI_API_KEY" in str(error.value)
    assert "DQAGENT_MODEL" in str(error.value)


@pytest.mark.parametrize("timeout", ["not-a-number", "0", "-1", "nan", "inf"])
def test_settings_reject_invalid_timeout(timeout: str) -> None:
    with pytest.raises(ConfigurationError, match="DQAGENT_TIMEOUT_SECONDS"):
        Settings.from_env(
            {
                "OPENAI_API_KEY": "secret",
                "DQAGENT_MODEL": "model-name",
                "DQAGENT_TIMEOUT_SECONDS": timeout,
            }
        )


@pytest.mark.parametrize("timeout", ["not-a-number", "0", "nan", "inf"])
def test_settings_reject_invalid_run_timeout(timeout: str) -> None:
    with pytest.raises(ConfigurationError, match="DQAGENT_RUN_TIMEOUT_SECONDS"):
        Settings.from_env(
            {
                "OPENAI_API_KEY": "secret",
                "DQAGENT_MODEL": "model-name",
                "DQAGENT_RUN_TIMEOUT_SECONDS": timeout,
            }
        )


@pytest.mark.parametrize("attempts", ["not-an-integer", "0", "-1"])
def test_settings_reject_invalid_model_attempts(attempts: str) -> None:
    with pytest.raises(ConfigurationError, match="DQAGENT_MAX_MODEL_ATTEMPTS"):
        Settings.from_env(
            {
                "OPENAI_API_KEY": "secret",
                "DQAGENT_MODEL": "model-name",
                "DQAGENT_MAX_MODEL_ATTEMPTS": attempts,
            }
        )
