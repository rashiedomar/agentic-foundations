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

### Phase 2: Core Components ✅
- ✅ **[Lesson 5: Planning & Reasoning](lessons/05-planning-reasoning.md)** - Task decomposition, CoT, ToT, ReAct strategies
  - 📓 [Notebook: Interactive Planning Experiments](notebooks/lesson_05_planning_experiments.py)
  - 🏗️ [Prototype: Production Planning Agent](prototypes/planning_agent.py)
- ✅ **[Lesson 6: Tool Integration](lessons/06-tool-integration.md)** - Giving agents superpowers
  - 🛠️ [Prototype: Tool Registry & Execution](src/tools/)
  - 🔧 [Examples: Tool Integration Patterns](notebooks/lesson_06_tool_examples.py)
  - 🤖 [Complete Agent: Planning + Tools](prototypes/integrated_agent.py)
- 🔜 **Lesson 7: Error Handling** - Making agents robust

### Phase 3: Advanced Topics 🚧
- 🔄 **Lesson 8: Multi-Agent Systems** - Agents working together (In Progress)
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
│   ├── 05-planning-reasoning.md
│   └── 06-tool-integration.md
├── src/              # Reusable agent components I'm building
│   └── tools/        # Tool registry and execution system
├── prototypes/       # Production-ready agent implementations
│   ├── planning_agent.py      # Full planning agent with multiple strategies
│   └── integrated_agent.py    # Complete agent with planning + tools
├── notebooks/        # Interactive learning & experiments
│   ├── lesson_05_planning_experiments.py
│   └── lesson_06_tool_examples.py
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
   
   # Explore tool integration
   python notebooks/lesson_06_tool_examples.py
   ```

4. **Experiment with prototypes**
   ```python
   # Use the integrated agent (planning + tools)
   from prototypes.integrated_agent import IntegratedAgent
   
   agent = IntegratedAgent(llm=your_llm)
   result = agent.execute_goal(
       goal="Research the latest AI developments and create a summary",
       use_tools=True
   )
   ```

5. **Fork and experiment yourself**
   Add your own learnings and experiments!

## 📖 Resources I'm Learning From

- **Frameworks:** LangGraph, CrewAI, AutoGen
- **Research Papers:** ReAct, Chain-of-Thought, Tree-of-Thoughts, Toolformer
- **Documentation:** OpenAI Agents SDK, Anthropic Claude, Function Calling
- **Community:** GitHub discussions, Discord communities

## 🎓 My Learning Goals

By the end of this journey, I want to:
- [x] Understand what makes an agent different from a model
- [x] Grasp the core agent loop and planning strategies
- [x] Build agents that can plan and execute multi-step tasks
- [x] Integrate tools to give agents real capabilities
- [ ] Create agents for real-world use cases (research, coding, data analysis)
- [ ] Build multi-agent systems that collaborate
- [ ] Contribute to the agentic AI community

## 🗺️ Progress Tracker

**Current Phase:** Phase 3 - Advanced Topics  
**Lessons Completed:** 6/10  
**Phase 1:** ✅ Complete (4/4 lessons)  
**Phase 2:** ✅ Complete (2/2 lessons)  
**Current Focus:** Multi-Agent Systems (Lesson 8)  
**Next Milestone:** Build collaborative multi-agent system  
**Last Updated:** December 24, 2025

Check [ROADMAP.md](ROADMAP.md) for my detailed learning plan and progress.

## 🎯 Recent Achievements

✅ **Completed Phase 2: Core Components**
- Built complete planning system with multiple strategies (CoT, ToT, ReAct)
- Implemented comprehensive tool integration system
- Created production-ready integrated agent combining planning + tools
- Developed tool registry, safe execution, and error handling
- Built working examples: research assistant, data analyst, code helper

## 🔥 Coming Up Next

Now working on **Lesson 8: Multi-Agent Systems**:
- Agent communication protocols
- Task delegation and coordination
- Building agent teams that collaborate
- Real example: Research team with specialized agents

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
**Progress:** Phase 2 Complete | Phase 3 In Progress (60% complete)
