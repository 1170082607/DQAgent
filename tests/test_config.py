import pytest

from dqagent.config import DEFAULT_TIMEOUT_SECONDS, Settings
from dqagent.errors import ConfigurationError


def test_settings_load_required_and_optional_values() -> None:
    settings = Settings.from_env(
        {
            "OPENAI_API_KEY": "secret",
            "DQAGENT_MODEL": "model-name",
            "OPENAI_BASE_URL": "https://example.test/v1",
            "DQAGENT_TIMEOUT_SECONDS": "12.5",
        }
    )

    assert settings == Settings(
        api_key="secret",
        model="model-name",
        base_url="https://example.test/v1",
        timeout_seconds=12.5,
    )


def test_settings_use_default_timeout() -> None:
    settings = Settings.from_env(
        {"OPENAI_API_KEY": "secret", "DQAGENT_MODEL": "model-name"}
    )

    assert settings.timeout_seconds == DEFAULT_TIMEOUT_SECONDS


def test_settings_report_all_missing_required_values() -> None:
    with pytest.raises(ConfigurationError) as error:
        Settings.from_env({})

    assert "OPENAI_API_KEY" in str(error.value)
    assert "DQAGENT_MODEL" in str(error.value)


@pytest.mark.parametrize("timeout", ["not-a-number", "0", "-1"])
def test_settings_reject_invalid_timeout(timeout: str) -> None:
    with pytest.raises(ConfigurationError, match="DQAGENT_TIMEOUT_SECONDS"):
        Settings.from_env(
            {
                "OPENAI_API_KEY": "secret",
                "DQAGENT_MODEL": "model-name",
                "DQAGENT_TIMEOUT_SECONDS": timeout,
            }
        )
