# Examples

The current runtime is exposed through the CLI:

```bash
dqagent --message "What is an agent loop?"
dqagent --system "Answer as a backend architecture reviewer."
dqagent --run-timeout 30 --max-model-attempts 2 --message "What time is UTC+8?"
```

Start a workflow that checkpoints after `prepare` and intentionally interrupts before `apply`, then
resume it in a second process:

```bash
python examples/workflow_resume.py start demo-1
python examples/workflow_resume.py resume demo-1
```

The example writes private state under `.local/checkpoints/`. Examples are runnable, focused on one
capability, and use public package interfaces. Conceptual notes belong in `docs/learning/` instead.
