# Fundamentals Notes
- Agent = (Policy, Memory, Tools, Goals, Environment Interface)
- Loop: observe → plan → act → reflect → update memory
- Memory: short-term (context), long-term (vector store), episodic (logs)
- Planner: task decomposition → tool selection → subgoals
- Executor: safe tool invocation + error handling + retries
- Reflection: self-critique → patch plan → next iteration
