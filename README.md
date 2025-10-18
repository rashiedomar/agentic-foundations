# 🤖 Agentic Foundations

> My journey learning to build autonomous AI agents from scratch

This repository documents everything I'm learning about **agentic AI systems** - AI that can reason, plan, use tools, and act autonomously to achieve goals.

## 🎯 What This Repo Is About

This is a **learning-in-public** repository where I:
- Document core concepts as I learn them
- Build progressively complex agents
- Share code examples and experiments
- Track my progress and reflections

**Not a polished framework** - this is raw learning, experiments, and notes as I figure things out.

## 📚 Lessons

Each lesson builds on the previous ones. Start from Lesson 1 if you're new to agentic AI.

### Phase 1: Foundations ✅
- ✅ **[Lesson 1: What is an AI Agent?](lessons/01-what-is-an-agent.md)** - Understanding agents vs models vs tools
- ✅ **[Lesson 2: The Agent Loop](lessons/02-agent-loop.md)** - Perceive → Plan → Act → Reflect cycle
- ✅ **[Lesson 3: Memory Systems](lessons/03-memory-systems.md)** - Short-term, long-term, and episodic memory
- ✅ **[Lesson 4: First Agent](lessons/04-first-agent.md)** - Building a simple reflection agent

### Phase 2: Core Components 🚧
- ✅ **[Lesson 5: Planning & Reasoning](lessons/05-planning-reasoning.md)** - Task decomposition, CoT, ToT, ReAct strategies
  - 📓 [Notebook: Interactive Planning Experiments](notebooks/lesson_05_planning_experiments.py)
  - 🏗️ [Prototype: Production Planning Agent](prototypes/planning_agent.py)
- 🔄 **Lesson 6: Tool Integration** - Giving agents superpowers (In Progress)
- 🔜 **Lesson 7: Error Handling** - Making agents robust

### Phase 3: Advanced Topics
- 🔜 **Lesson 8: Multi-Agent Systems** - Agents working together
- 🔜 **Lesson 9: Environment Interfaces** - Connecting agents to the real world
- 🔜 **Lesson 10: Evaluation** - Measuring agent performance

*More lessons added as I learn...*

## 🛠️ What's In This Repo

```
agentic-foundations/
├── lessons/           # Markdown lessons with concepts & examples
│   ├── 01-what-is-an-agent.md
│   ├── 02-agent-loop.md
│   ├── 03-memory-systems.md
│   ├── 04-first-agent.md
│   └── 05-planning-reasoning.md
├── src/              # Reusable agent components I'm building
├── prototypes/       # Production-ready agent implementations
│   └── planning_agent.py      # Full planning agent with multiple strategies
├── notebooks/        # Interactive learning & experiments
│   └── lesson_05_planning_experiments.py
├── docs/            # Technical notes and architecture
└── logs/            # Run logs and reflections
```

## 🚀 Quick Start

Want to follow along? Here's how:

1. **Clone the repo**
   ```bash
   git clone https://github.com/rashiedomar/agentic-foundations.git
   cd agentic-foundations
   ```

2. **Start with Lesson 1**
   ```bash
   # Read through lessons in order
   cat lessons/01-what-is-an-agent.md
   ```

3. **Try the interactive notebooks**
   ```bash
   # Run the planning experiments
   python notebooks/lesson_05_planning_experiments.py
   ```

4. **Experiment with prototypes**
   ```python
   # Use the production planning agent
   from prototypes.planning_agent import PlanningAgent, PlanningStrategy
   
   agent = PlanningAgent(llm=your_llm)
   result = agent.execute_goal(
       goal="Your goal here",
       strategy=PlanningStrategy.CHAIN_OF_THOUGHT
   )
   ```

5. **Fork and experiment yourself**
   Add your own learnings and experiments!

## 📖 Resources I'm Learning From

- **Frameworks:** LangGraph, CrewAI, AutoGen
- **Research Papers:** ReAct, Chain-of-Thought, Tree-of-Thoughts
- **Documentation:** OpenAI Agents SDK, Anthropic Claude
- **Community:** GitHub discussions, Discord communities

## 🎓 My Learning Goals

By the end of this journey, I want to:
- [x] Understand what makes an agent different from a model
- [x] Grasp the core agent loop and planning strategies
- [ ] Build agents that can plan and execute multi-step tasks
- [ ] Integrate tools to give agents real capabilities
- [ ] Create agents for real-world use cases (research, coding, data analysis)
- [ ] Contribute to the agentic AI community

## 🗺️ Progress Tracker

**Current Phase:** Phase 2 - Core Components  
**Lessons Completed:** 5/10  
**Current Focus:** Tool Integration (Lesson 6)  
**Next Milestone:** Build complete agent with planning + tools  
**Last Updated:** October 2025

Check [ROADMAP.md](ROADMAP.md) for my detailed learning plan and progress.

## 🎯 Coming Up Next

After completing **Lesson 6: Tool Integration**, I'll build a **real integrated example** that combines:
- Planning & reasoning (Lesson 5) ✅
- Tool integration (Lesson 6) 🔄
- Complete working agent that can research, analyze, and report

## 🤝 Contributing

This is my personal learning repo, but if you're learning too:
- Open issues to discuss concepts
- Share resources via PRs
- Fork and create your own learning journey!

## 📬 Connect

Learning AI agents too? Let's connect!
- Open an issue to discuss
- Follow the repo for updates
- Share your own learning journey

---

**License:** MIT  
**Status:** 🟢 Actively Learning  
**Progress:** Phase 2 - Core Components (50% complete)