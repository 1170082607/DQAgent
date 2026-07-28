# Examples

The current runtime is exposed through the CLI:

```bash
dqagent --message "What is an agent loop?"
dqagent --system "Answer as a backend architecture reviewer."
dqagent --run-timeout 30 --max-model-attempts 2 --message "What time is UTC+8?"
```

Future examples should be runnable, focused on one capability, and use public package interfaces.
Conceptual notes belong in `docs/learning/` instead.
