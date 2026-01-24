```markdown
# 🤖 Agentic Foundations

> My completed journey learning to build autonomous AI agents from scratch

This repository documents my full learning path into **agentic AI systems** — systems that can reason, plan, use tools, and act autonomously to achieve goals.

## ✅ Course Completed

This repo started as a **learning-in-public** project. The course is now **fully completed**, and the repository serves as:
- A structured reference for agentic AI fundamentals → advanced topics
- A collection of working agent prototypes (planning, tools, multi-agent workflows)
- Notes, lessons, and experiments you can reuse for your own learning or projects

It’s still not a polished “framework” — but it *is* a complete, end-to-end learning + implementation repo.

---

## 🎯 What This Repo Covers

Through this repo, I learned and implemented:

- What makes an agent different from a model or tool
- The agent loop (Perceive → Plan → Act → Reflect)
- Memory systems (short-term, long-term, episodic)
- Planning strategies (CoT, ToT, ReAct)
- Tool integration (registries, safe execution, tool selection)
- Robust execution (error handling, retries, guardrails)
- Multi-agent collaboration (delegation, coordination, communication protocols)
- Evaluation basics (measuring agent quality and reliability)

---

## 📚 Lessons

Each lesson builds on the previous ones.

### Phase 1: Foundations ✅ (Complete)
- ✅ **[Lesson 1: What is an AI Agent?](lessons/01-what-is-an-agent.md)** — Agents vs models vs tools
- ✅ **[Lesson 2: The Agent Loop](lessons/02-agent-loop.md)** — Perceive → Plan → Act → Reflect
- ✅ **[Lesson 3: Memory Systems](lessons/03-memory-systems.md)** — Short/long-term + episodic memory
- ✅ **[Lesson 4: First Agent](lessons/04-first-agent.md)** — Simple reflection agent

### Phase 2: Core Components ✅ (Complete)
- ✅ **[Lesson 5: Planning & Reasoning](lessons/05-planning-reasoning.md)** — Task decomposition, CoT/ToT/ReAct
  - 📓 [Notebook: Interactive Planning Experiments](notebooks/lesson_05_planning_experiments.py)
  - 🏗️ [Prototype: Planning Agent](prototypes/planning_agent.py)
- ✅ **[Lesson 6: Tool Integration](lessons/06-tool-integration.md)** — Giving agents superpowers
  - 🛠️ [Tool Registry & Execution](src/tools/)
  - 🔧 [Examples: Tool Patterns](notebooks/lesson_06_tool_examples.py)
  - 🤖 [Integrated Agent: Planning + Tools](prototypes/integrated_agent.py)

### Phase 3: Advanced Topics ✅ (Complete)
- ✅ **Lesson 7: Error Handling** — Making agents robust and reliable
- ✅ **Lesson 8: Multi-Agent Systems** — Coordination, delegation, communication
- ✅ **Lesson 9: Environment Interfaces** — Connecting agents to external systems
- ✅ **Lesson 10: Evaluation** — Testing, metrics, and reliability

---

## 🛠️ What's In This Repo

```

agentic-foundations/
├── lessons/           # Markdown lessons with concepts & examples
├── src/              # Reusable agent components
│   └── tools/        # Tool registry and execution system
├── prototypes/       # Production-ready agent implementations
│   ├── planning_agent.py
│   └── integrated_agent.py
├── notebooks/        # Experiments & runnable examples
├── docs/             # Architecture notes
└── logs/             # Reflections and run logs

````

---

## 🚀 Quick Start

1. **Clone the repo**
   ```bash
   git clone https://github.com/rashiedomar/agentic-foundations.git
   cd agentic-foundations
````

2. **Read lessons in order**

   ```bash
   cat lessons/01-what-is-an-agent.md
   ```

3. **Run experiments**

   ```bash
   python notebooks/lesson_05_planning_experiments.py
   python notebooks/lesson_06_tool_examples.py
   ```

4. **Try the integrated agent**

   ```python
   from prototypes.integrated_agent import IntegratedAgent

   agent = IntegratedAgent(llm=your_llm)
   result = agent.execute_goal(
       goal="Research the latest AI developments and create a summary",
       use_tools=True
   )
   print(result)
   ```

---

## 📖 Resources Referenced

* **Frameworks:** LangGraph, CrewAI, AutoGen
* **Papers:** ReAct, Chain-of-Thought, Tree-of-Thoughts, Toolformer
* **Docs:** OpenAI Agents SDK, Anthropic tool use / function calling
* **Community:** GitHub discussions, Discord communities

---

## 🗺️ Progress Tracker (Final)

**Status:** ✅ Course Completed
**Lessons Completed:** 10/10
**Phases Completed:** Phase 1 ✅ | Phase 2 ✅ | Phase 3 ✅
**Last Updated:** January 2026

---

## 🎯 What’s Next

Even though the course is complete, I may keep extending this repo with:

* More real-world agent projects (research, coding, ops automation)
* Better evaluation harnesses and benchmarks
* Additional multi-agent patterns and templates

---

## 🤝 Contributing

This began as my personal learning repository, but contributions are welcome:

* Open issues to discuss ideas or improvements
* PRs for fixes, resources, or examples
* Fork and build your own version

---

**License:** MIT
**Status:** 🟢 Completed + Maintained

```
```
