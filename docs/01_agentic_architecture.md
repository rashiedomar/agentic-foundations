# Agentic Architecture (Draft)
- `src/agent_loop.py`: orchestrates perceive–plan–act–reflect
- `src/memory/`: vector DB + episodic logs + key-value cache
- `src/tools/`: wrappers for search, calculator, python, browser, APIs
- `src/planner/`: reasoning chain, schemas, guardrails
- `src/executor/`: tool routing, retries, observability
