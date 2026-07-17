# Contributing

DQAgent is developed incrementally. Changes should improve the active roadmap phase without
introducing abstractions for unimplemented future phases.

## Development setup

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Verification

Run all checks before submitting a change:

```bash
ruff check .
mypy src
pytest
```

## Change guidelines

- Keep changes focused on one capability or decision.
- Add tests for externally observable behavior.
- Update README or `docs/` when behavior or architecture changes.
- Add an ADR for decisions that constrain future implementation choices.
- Never commit API keys, `.env` files, editor state, or local agent configuration.
