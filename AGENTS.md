# DQAgent Project Instructions

## Purpose

DQAgent is an incremental learning project for understanding and building
production-oriented AI Agent systems from first principles.

The authoritative development plan is maintained in `docs/roadmap.md`.

## Scope Discipline

- Implement only the active roadmap phase unless a later capability is required to validate a
  current architectural decision.
- Introduce frameworks or dependencies only for a documented engineering reason.
- Keep changes focused. Discuss broad refactors before implementing them.
- Do not create modules for planned capabilities before they have real behavior.

## Architecture

- Prefer explicit responsibilities and dependency directions.
- Keep provider-specific SDK code behind provider-neutral application interfaces.
- Add abstractions for existing boundaries or multiple real implementations, not hypothetical
  future requirements.
- Record durable architectural decisions in `docs/adr/`.

## Engineering Requirements

- Use idiomatic Python and type hints for public interfaces.
- Keep secrets and local development configuration out of source control.
- Test externally observable behavior and important failure paths.
- Run Ruff, mypy, and pytest before considering a change complete.

## Version Control Workflow

- Codex has read-only access to Git metadata and may run inspection commands such as `git status`,
  `git diff`, and `git log`.
- Codex must not attempt Git operations that write repository metadata or history, including
  `git add`, `git commit`, branch or tag changes, rebases, resets, and remote configuration changes.
- When the user asks Codex to stage, commit, or otherwise update Git state, provide the exact commands
  for the user to review and run manually instead of attempting them.
- Codex must not run `git push` or otherwise modify a remote repository. Leave pushing to the user
  after they review the local changes and history.

## Documentation

- Keep README, roadmap, and architecture consistent with implemented behavior.
- Update documentation together with behavior or architectural changes.
- Store experiments, source-reading notes, and framework comparisons in `docs/learning/`.

## Local Learning Notes

- When the user explicitly asks for an explanation, analysis, walkthrough, or review for learning,
  provide the answer in the conversation and save the substantive explanation as a new Markdown
  file under `.local/learning-notes/`.
- Name files `YYYY-MM-DD-short-topic.md`. If that name already exists, add a short numeric suffix
  instead of overwriting an earlier note.
- Make each note self-contained: include the topic, relevant context, the explanation, important
  trade-offs, and links or paths to the code being discussed when useful.
- Write the note in the conversation's primary language unless the user requests another language.
- Do not save routine progress updates, terse factual answers, or implementation handoff summaries
  unless the user asks to preserve them for learning.
- Treat `.local/learning-notes/` as private study material excluded from Git. Continue using
  `docs/learning/` for durable project knowledge that should be reviewed and version controlled.

## Definition of Done

- Relevant tests pass.
- Static checks pass.
- Documentation reflects the implementation.
- No credentials, generated artifacts, editor state, or local agent configuration are included.
