# 🧭 Agentic AI Systems — Roadmap

> Status legend: ✅ done · 🔜 next · ⏳ in-progress · 🧪 experiment · 🧩 idea

## Phase 1 — Foundations (Weeks 1–3)
**Goal:** Understand agentic loops and rebuild tiny agents.
- Study: agents vs models vs tools; Perceive→Plan→Act→Reflect
- Try frameworks: LangGraph, CrewAI, AutoGen, MetaGPT, AutoGPT (skim)
- Build: Reflection Agent (self-critique), Task-Solver CLI (planner + executor)
**Deliverables:** `docs/00_fundamentals.md`, `notebooks/Agentic_Fundamentals.ipynb`, demo logs in `logs/`

## Phase 2 — Infrastructure (Weeks 4–6)
**Goal:** Modular core for reusable agents.
- Modules: `src/memory/`, `src/tools/`, `src/planner/`, `src/executor/`
- Vector memory (FAISS/Chroma), run loop, config schema (YAML/JSON)
**Deliverables:** `src/agent_loop.py`, minimal tests

## Phase 3 — Prototypes (Weeks 7–10)
**Goal:** Narrow-scope agents that work.
- Research Assistant (PDF → summary → next-steps plan)
- Dev Agent (write + run + debug Python)
- Pose/JEPA Vision Agent (hooks to your motion pipeline)
**Deliverables:** `prototypes/research_agent/`, `dev_agent/`, `vision_agent/`

## Phase 4 — Multi-Agent Orchestration (Weeks 11–14)
**Goal:** Team-of-agents with roles & messaging.
- Planner assigns subtasks → Workers execute → Reviewer critiques
- JSON task schema; experiment with CrewAI/AutoGen orchestration
**Deliverables:** `multi_agent_demo.py`, comms diagram in `docs/`

## Phase 5 — Environment Interfaces (Weeks 15–18)
**Goal:** Perception–action loops.
- Browser (Playwright), CLI tools, optional robotics sim
- CCTV/IoT hooks for Crosswalk project
**Deliverables:** `src/env/` adapters, run traces

## Phase 6 — Domain Specialization (Weeks 19–24)
**Goal:** Concrete value in your domains.
- SomaliGPT Research Agent · Crosswalk Safety Decision Agent
- Mining Ops Optimizer · Education Mentor Agent
**Deliverables:** local demo (Streamlit/React+FastAPI), report with diagrams

## Phase 7 — Evaluation & Reliability (Weeks 25–28)
**Goal:** Measure autonomy & quality.
- Metrics: task success, planning depth, reflection usefulness
- Dashboard: run histories, error taxonomy, ablations
**Deliverables:** `notebooks/Eval_Dashboard.ipynb`, technical report draft

## Phase 8 — Long-Term (Month 7+)
- Integrate SomaliGPT as base model; add VLM/JEPA hooks
- Build **AgenticOS**: a memory graph where multiple domain agents collaborate

---

### Working Agreements
- Small daily commits. Log each run in `logs/DATE_title.md`
- Use GitHub Projects for tasks (Learning · Building · Testing · Reflecting)
- Tag releases per phase: `v0.1 Foundations`, `v0.2 Prototypes`, `v1.0 Framework`
