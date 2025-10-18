# Lesson 5: Planning & Reasoning

> **What you'll learn:** How agents break down complex goals, choose the right tools, and reason through multi-step problems to achieve sophisticated outcomes.

---

## 🎯 Learning Objectives

By the end of this lesson, you'll understand:
- How agents decompose complex tasks into manageable steps
- Different planning strategies (Chain-of-Thought, Tree-of-Thoughts, ReAct)
- Tool selection and orchestration patterns
- How to implement planning in your agents
- Common pitfalls and how to avoid them

---

## 🧠 The Problem: Complexity Requires Planning

Imagine asking an agent: **"Research the top 5 AI startups from 2024, analyze their business models, and create a competitive analysis report."**

Without planning, an agent might:
- Get overwhelmed and produce shallow results
- Miss important steps
- Use the wrong tools
- Get stuck in loops
- Produce incoherent outputs

**Planning is what transforms a simple reactor into an intelligent problem-solver.**

---

## 📊 Core Concepts

### 1. Task Decomposition

Breaking big problems into smaller, solvable pieces.

```
BIG GOAL: "Create a competitive analysis report"
    ↓
SUB-TASKS:
1. Define evaluation criteria
2. Search for top AI startups
3. Gather data on each startup
4. Analyze business models
5. Compare and contrast
6. Generate report
```

### 2. Planning Strategies

#### **Chain-of-Thought (CoT)**
Linear, step-by-step reasoning:

```python
def chain_of_thought(goal):
    prompt = f"""
    Goal: {goal}
    
    Let's think step by step:
    1. First, I need to...
    2. Then, I should...
    3. Next, I will...
    4. Finally, I'll...
    """
    return llm.complete(prompt)
```

**Pros:** Simple, interpretable  
**Cons:** Can't backtrack, single path

#### **Tree-of-Thoughts (ToT)**
Explores multiple reasoning paths:

```
Goal
├── Path A
│   ├── Step A1 ✓
│   └── Step A2 ✗ (backtrack)
└── Path B
    ├── Step B1 ✓
    └── Step B2 ✓ (continue)
```

**Pros:** Can explore alternatives  
**Cons:** Computationally expensive

#### **ReAct (Reasoning + Acting)**
Interleaves thinking and action:

```python
def react_loop(goal):
    while not goal_achieved:
        # THINK
        thought = think_about_next_step(goal, context)
        
        # ACT
        action = decide_action(thought)
        observation = execute_action(action)
        
        # OBSERVE & UPDATE
        context.update(observation)
```

**Pros:** Grounded in reality, adaptive  
**Cons:** Can get stuck in local optima

### 3. Tool Selection Logic

How agents decide which tool to use:

```python
class ToolSelector:
    def __init__(self, tools):
        self.tools = tools
    
    def select_tool(self, task, context):
        # Generate tool descriptions
        tool_descriptions = [t.description for t in self.tools]
        
        prompt = f"""
        Task: {task}
        Context: {context}
        
        Available tools:
        {tool_descriptions}
        
        Which tool is best for this task? Why?
        Output: {{"tool": "name", "reasoning": "..."}}
        """
        
        response = llm.complete(prompt)
        return json.loads(response)
```

---

## 🛠️ Implementation: Building a Planning Agent

Let's build an agent that can plan and execute multi-step tasks:

```python
from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Task:
    id: str
    description: str
    dependencies: List[str] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None

class PlanningAgent:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
        self.plan = []
        self.memory = []
    
    def decompose_goal(self, goal: str) -> List[Task]:
        """Break down a goal into subtasks"""
        prompt = f"""
        Goal: {goal}
        
        Break this down into specific, actionable subtasks.
        Consider dependencies between tasks.
        
        Output JSON:
        {{
            "tasks": [
                {{
                    "id": "task_1",
                    "description": "...",
                    "dependencies": []
                }},
                ...
            ]
        }}
        """
        
        response = self.llm.complete(prompt)
        tasks_data = json.loads(response)
        
        return [Task(**task) for task in tasks_data["tasks"]]
    
    def prioritize_tasks(self, tasks: List[Task]) -> List[Task]:
        """Order tasks based on dependencies"""
        # Topological sort for dependency resolution
        sorted_tasks = []
        completed = set()
        
        while len(sorted_tasks) < len(tasks):
            for task in tasks:
                if task.id in completed:
                    continue
                    
                # Check if all dependencies are completed
                deps_met = all(
                    dep in completed 
                    for dep in (task.dependencies or [])
                )
                
                if deps_met:
                    sorted_tasks.append(task)
                    completed.add(task.id)
        
        return sorted_tasks
    
    def execute_plan(self, goal: str):
        """Main planning and execution loop"""
        
        # STEP 1: Decompose
        print(f"🎯 Goal: {goal}")
        tasks = self.decompose_goal(goal)
        print(f"📝 Decomposed into {len(tasks)} tasks")
        
        # STEP 2: Prioritize
        ordered_tasks = self.prioritize_tasks(tasks)
        
        # STEP 3: Execute
        for task in ordered_tasks:
            print(f"\n▶️ Executing: {task.description}")
            task.status = TaskStatus.IN_PROGRESS
            
            try:
                # Select appropriate tool
                tool_selection = self.select_tool_for_task(task)
                tool = self.tools[tool_selection["tool"]]
                
                # Execute with tool
                result = tool.execute(task.description)
                
                # Update task
                task.result = result
                task.status = TaskStatus.COMPLETED
                
                # Store in memory
                self.memory.append({
                    "task": task.description,
                    "result": result
                })
                
                print(f"✅ Completed: {task.id}")
                
            except Exception as e:
                task.status = TaskStatus.FAILED
                print(f"❌ Failed: {task.id} - {str(e)}")
                
                # Decide whether to retry or replan
                if self.should_replan(task, e):
                    self.replan(goal, tasks, task)
    
    def should_replan(self, failed_task: Task, error: Exception) -> bool:
        """Determine if we need to adjust the plan"""
        prompt = f"""
        Task failed: {failed_task.description}
        Error: {str(error)}
        
        Should we:
        1. Retry the same approach
        2. Replan with a different approach
        3. Skip this task
        4. Abort the entire plan
        
        Respond with the number only.
        """
        
        response = self.llm.complete(prompt)
        return "2" in response
    
    def replan(self, original_goal: str, tasks: List[Task], failed_task: Task):
        """Generate alternative plan when something fails"""
        completed_tasks = [t for t in tasks if t.status == TaskStatus.COMPLETED]
        
        prompt = f"""
        Original goal: {original_goal}
        Completed so far: {completed_tasks}
        Failed task: {failed_task.description}
        
        Generate an alternative approach to achieve the goal.
        """
        
        new_plan = self.decompose_goal(prompt)
        self.execute_plan(original_goal)  # Recursive execution with new plan
```

---

## 🎭 Advanced Planning Patterns

### 1. **Hierarchical Planning**
Plans within plans:

```python
class HierarchicalPlanner:
    def create_plan(self, goal):
        # High-level plan
        major_milestones = self.create_high_level_plan(goal)
        
        # Detailed sub-plans for each milestone
        detailed_plan = []
        for milestone in major_milestones:
            sub_plan = self.create_detailed_plan(milestone)
            detailed_plan.append({
                "milestone": milestone,
                "steps": sub_plan
            })
        
        return detailed_plan
```

### 2. **Parallel Execution**
When tasks can run simultaneously:

```python
import asyncio

async def execute_parallel_tasks(tasks):
    # Group independent tasks
    independent_groups = group_by_dependencies(tasks)
    
    for group in independent_groups:
        # Execute all tasks in group simultaneously
        results = await asyncio.gather(*[
            execute_task(task) for task in group
        ])
```

### 3. **Adaptive Planning**
Plans that evolve based on feedback:

```python
class AdaptivePlanner:
    def __init__(self):
        self.plan_history = []
        self.success_patterns = []
    
    def learn_from_execution(self, plan, outcome):
        """Learn what works and what doesn't"""
        if outcome.success:
            self.success_patterns.append({
                "goal_type": plan.goal_type,
                "approach": plan.approach,
                "success_rate": outcome.metrics
            })
    
    def generate_plan(self, goal):
        # Check if we've seen similar goals
        similar_successes = self.find_similar_patterns(goal)
        
        if similar_successes:
            # Adapt successful pattern
            return self.adapt_pattern(similar_successes[0], goal)
        else:
            # Generate new plan
            return self.create_new_plan(goal)
```

---

## 💡 Best Practices

### ✅ DO:
- **Start simple**: Begin with linear planning before trying complex strategies
- **Make plans explicit**: Always output the plan before executing
- **Handle failures gracefully**: Build in retry and replan logic
- **Track dependencies**: Ensure tasks execute in the right order
- **Set clear success criteria**: Know when a task is truly complete

### ❌ DON'T:
- **Over-decompose**: Don't break tasks down too granularly
- **Ignore context**: Each step should know what came before
- **Assume perfection**: Plans will fail; build for resilience
- **Forget the goal**: Keep the original objective in focus

---

## 🚨 Common Pitfalls

### 1. **The Planning Paralysis**
Agent spends too much time planning, never executes.

**Solution:**
```python
MAX_PLANNING_TIME = 30  # seconds
start_time = time.time()

while not plan_complete:
    if time.time() - start_time > MAX_PLANNING_TIME:
        # Use best plan so far
        break
    # Continue planning...
```

### 2. **The Infinite Decomposition**
Tasks keep breaking into smaller and smaller pieces.

**Solution:**
```python
def decompose_with_limit(task, max_depth=3, current_depth=0):
    if current_depth >= max_depth:
        return [task]  # Stop decomposing
    # Continue decomposition...
```

### 3. **The Brittle Plan**
Plan fails completely if one step fails.

**Solution:** Build in checkpoints and alternative paths.

---

## 🔬 Real-World Example

Let's trace through a real planning scenario:

**Goal:** "Research quantum computing startups and create an investment memo"

```
PLANNING PHASE:
==============
1. Decomposition:
   - Define research criteria
   - Search for quantum computing startups
   - Collect data on each startup
   - Analyze market position
   - Evaluate technical advantages
   - Assess investment potential
   - Write investment memo

2. Tool Selection:
   - Web search → Finding startups
   - Data extraction → Gathering info
   - Analysis tools → Market analysis
   - Writing tools → Memo generation

3. Dependency Graph:
   Define criteria → Search → Collect data
                                    ↓
                            Analyze & Evaluate
                                    ↓
                              Write memo

EXECUTION TRACE:
===============
▶️ Task 1: Define research criteria
   Tool: reasoning_engine
   Output: Focus on Series A/B, quantum hardware/software, 
           strong technical team

▶️ Task 2: Search for quantum computing startups
   Tool: web_search
   Output: Found 15 relevant companies

▶️ Task 3: Collect data (parallel execution)
   Tool: web_scraper + data_extractor
   Output: Gathered funding, team, technology data

[... continues through all tasks ...]

✅ FINAL OUTPUT: 5-page investment memo with recommendations
```

---

## 🎮 Try It Yourself

### Challenge: Build a Meta-Planner

Create an agent that can plan how to plan! It should:

1. Analyze the goal complexity
2. Choose appropriate planning strategy (CoT, ToT, or ReAct)
3. Execute the chosen strategy
4. Evaluate plan quality
5. Learn from outcomes

```python
class MetaPlanner:
    def __init__(self):
        self.strategies = {
            "simple": ChainOfThoughtPlanner(),
            "complex": TreeOfThoughtsPlanner(),
            "interactive": ReActPlanner()
        }
    
    def analyze_complexity(self, goal):
        # Your implementation here
        pass
    
    def select_strategy(self, complexity_score):
        # Your implementation here
        pass
    
    def execute(self, goal):
        complexity = self.analyze_complexity(goal)
        strategy = self.select_strategy(complexity)
        return strategy.plan_and_execute(goal)
```

---

## 📚 Deep Dive Resources

### Essential Papers
- [Chain-of-Thought Prompting (Wei et al., 2022)](https://arxiv.org/abs/2201.11903)
- [Tree of Thoughts (Yao et al., 2023)](https://arxiv.org/abs/2305.10601)
- [ReAct: Synergizing Reasoning and Acting (Yao et al., 2022)](https://arxiv.org/abs/2210.03629)
- [Voyager: An Open-Ended Embodied Agent (Wang et al., 2023)](https://arxiv.org/abs/2305.16291)

### Implementations to Study
- **LangChain Planning**: Chain/Tree implementations
- **AutoGPT**: Hierarchical goal management
- **BabyAGI**: Task creation and prioritization
- **CAMEL**: Role-playing for task decomposition

---

## 🎯 Key Takeaways

1. **Planning transforms reactive agents into proactive problem-solvers**
2. **Different strategies suit different problems** - know when to use each
3. **Task decomposition is an art** - not too big, not too small
4. **Dependencies matter** - respect the order of operations
5. **Plans should be flexible** - build in adaptation and recovery
6. **Tool selection is crucial** - right tool for the right job

---

## ➡️ What's Next?

Now that you understand how agents plan and reason, the next step is giving them the right tools to execute those plans.

**Next Lesson:** [Tool Integration - Giving Agents Superpowers](06-tool-integration.md)

We'll explore how to build, integrate, and orchestrate tools that let your agents interact with the world!

---

## 🤔 Reflection Questions

1. **When would you use Tree-of-Thoughts over Chain-of-Thought?**
2. **How might you combine multiple planning strategies in one agent?**
3. **What happens when an agent's plan conflicts with user expectations?**
4. **How could you make planning more efficient for time-critical tasks?**
5. **What role should human feedback play in agent planning?**

---

## 📝 Your Notes

*Space for your thoughts and experiments:*

**What clicked:**
- 

**Still confused about:**
- 

**Ideas to try:**
- 

**Real-world applications I'm thinking about:**
- 

---

**Lesson Status:** 📝 In Progress  
**Difficulty:** ⭐⭐⭐ (Intermediate)  
**Time to Complete:** 45-60 minutes  
**Prerequisites:** Lessons 1-4
