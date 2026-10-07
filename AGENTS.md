# Project agent rules

## GPT-5.6 Sol delegation

When the main/root chat is running `gpt-5.6-sol`, regardless of its reasoning level:

- Use the main `gpt-5.6-sol` agent only for planning, task decomposition, coordination, review, and final synthesis.
- Assign every bounded implementation, investigation, testing, or other execution subtask to a subagent using `gpt-5.6-luna` with `reasoning_effort: max`.
- Do not spawn `gpt-5.6-sol` subagents for those subtasks.
- If a request has no meaningful bounded subtask to delegate, the main agent may handle it directly without spawning a subagent.
