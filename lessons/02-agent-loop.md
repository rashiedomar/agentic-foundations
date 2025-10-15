# Lesson 2: The Agent Loop

> **What you'll learn:** The four-step cycle that powers every autonomous agent - Perceive, Plan, Act, Reflect - and how to build your first agent loop.

---

## 🎯 Learning Objectives

By the end of this lesson, you'll understand:
- The four core steps every agent follows
- How agents make decisions autonomously
- Different loop patterns and when to use them
- How to implement a basic agent loop
- Common pitfalls and how to avoid them

---

## 🔄 Quick Recap

In Lesson 1, you learned that agents are different from language models because they:
- Have goals
- Work autonomously
- Use tools
- Have memory

**But HOW do they actually work?**

The answer: **The Agent Loop** - a continuous cycle that runs until the goal is achieved.

---

## 📖 The Core Concept

Every agent, regardless of complexity, follows this basic pattern:

```
┌─────────────────────────────────────────────┐
│                                             │
│   PERCEIVE → PLAN → ACT → REFLECT          │
│       ↑                          ↓          │
│       └──────────────────────────┘          │
│            (Loop continues)                 │
│                                             │
└─────────────────────────────────────────────┘
```

This is called the **agent loop**, and it's the secret sauce that makes agents autonomous.

Let's break down each step.

---

## 1️⃣ PERCEIVE: Gather Context

**Question:** "What do I know right now?"

In this step, the agent gathers all relevant information:
- **Current state** - Where am I? What's happening?
- **Memory** - What have I learned so far?
- **Available tools** - What can I do?
- **Goal** - What am I trying to achieve?

### Example: Research Agent

```
Goal: "Summarize recent AI agent research"

PERCEIVE:
- I have access to: web_search, pdf_reader, text_writer
- My memory shows: I haven't started yet
- Current context: Empty, need to begin research
- Available information: None yet
```

### In Code:

```python
def perceive(self):
    """Gather all relevant context"""
    context = {
        "goal": self.goal,
        "memory": self.memory.retrieve_relevant(),
        "available_tools": self.tools,
        "current_state": self.state,
        "past_actions": self.history[-5:]  # Last 5 actions
    }
    return context
```

**Key Point:** The agent needs a complete picture before deciding what to do next.

---

## 2️⃣ PLAN: Reasoning & Decision Making

**Question:** "What should I do next?"

This is where the agent's "intelligence" shines. It analyzes the context and decides:
- **What action to take**
- **Which tool to use**
- **What parameters to pass**
- **Why this is the best next step**

### Planning Strategies

Different agents use different planning approaches:

#### A. Chain of Thought (CoT)
The agent reasons step-by-step:
```
"To summarize AI agent research, I need to:
1. First, search for recent papers
2. Then, read the top papers
3. Then, extract key findings
4. Finally, synthesize into a summary

Let me start with step 1: search"
```

#### B. ReAct (Reasoning + Acting)
Combines reasoning with action in a loop:
```
Thought: I need to find recent AI agent papers
Action: web_search("AI agents research 2024")
Observation: Found 10 papers from arxiv
Thought: Good, now I should read the top 3
Action: pdf_reader("paper1.pdf")
```

#### C. Task Decomposition
Break big goal into smaller subgoals:
```
Main Goal: Summarize AI agent research
├─ Subgoal 1: Find papers (search)
├─ Subgoal 2: Read papers (pdf_reader)
├─ Subgoal 3: Extract insights (analyze)
└─ Subgoal 4: Write summary (text_writer)
```

### Example: Research Agent Planning

```
Context: Need to research AI agents, have web_search tool available

PLAN:
- Reasoning: "I should start by searching for recent papers"
- Action: Use web_search
- Parameters: "AI agents 2024 research papers"
- Expected outcome: List of relevant papers
```

### In Code:

```python
def plan(self, context):
    """Decide the next action based on context"""
    
    # Agent reasons about what to do
    reasoning_prompt = f"""
    Goal: {context['goal']}
    Current state: {context['current_state']}
    Available tools: {context['available_tools']}
    Past actions: {context['past_actions']}
    
    What should I do next to make progress toward the goal?
    Think step-by-step.
    """
    
    # LLM generates a plan
    plan = self.llm.complete(reasoning_prompt)
    
    # Parse into structured action
    next_action = self.parse_plan(plan)
    
    return next_action
```

**Key Point:** Good planning = progress. Bad planning = wasted actions.

---

## 3️⃣ ACT: Execute the Plan

**Question:** "Let me do it!"

Now the agent executes the planned action:
- **Call the tool** with the right parameters
- **Handle errors** if something goes wrong
- **Capture the result** for the next step

### Example: Research Agent Acting

```
PLAN: Search for "AI agents 2024 research papers"

ACT:
- Tool: web_search
- Parameters: {"query": "AI agents 2024 research papers"}
- Execution: [Calling search API...]
- Result: Found 10 papers:
  1. "ReAct: Synergizing Reasoning and Acting"
  2. "AutoGPT: Building Autonomous Agents"
  ...
```

### In Code:

```python
def act(self, action):
    """Execute the planned action"""
    
    try:
        # Get the tool
        tool = self.tools[action.tool_name]
        
        # Execute with parameters
        result = tool.run(**action.parameters)
        
        # Log the action
        self.history.append({
            "action": action,
            "result": result,
            "timestamp": now()
        })
        
        return {
            "success": True,
            "result": result
        }
        
    except Exception as e:
        # Handle errors gracefully
        return {
            "success": False,
            "error": str(e)
        }
```

**Key Point:** Actions change the world. This is where the agent actually does something.

---

## 4️⃣ REFLECT: Evaluate & Learn

**Question:** "Did it work? What did I learn?"

After acting, the agent reflects on the results:
- **Evaluate** - Was the action successful?
- **Learn** - What did I discover?
- **Decide** - Should I continue or adjust my plan?
- **Update memory** - Store important findings

### Example: Research Agent Reflecting

```
ACTION RESULT: Found 10 papers about AI agents

REFLECT:
- Success? ✅ Yes, got relevant results
- Quality? Good - papers are recent and relevant
- What I learned: There are 3 main approaches (ReAct, AutoGPT, Tool use)
- Next step: Should I read these papers now? Yes
- Progress toward goal: 20% complete
- Update memory: Store paper list for next iteration
```

### Reflection Types

#### Simple Validation
```python
if result.success:
    return "continue"
else:
    return "retry with different approach"
```

#### Deep Self-Critique
```python
critique_prompt = f"""
I just performed this action: {action}
The result was: {result}

Critically evaluate:
1. Did this help achieve my goal?
2. What did I learn?
3. What should I do differently next time?
4. Should I continue this approach or change strategy?
"""
```

#### Progress Tracking
```python
progress = {
    "steps_completed": len(self.history),
    "goal_progress": self.estimate_progress(),
    "remaining_work": self.estimate_remaining(),
    "confidence": self.assess_confidence()
}
```

### In Code:

```python
def reflect(self, action, result):
    """Evaluate the action and learn from it"""
    
    reflection = {
        "action_successful": result.get("success", False),
        "learned": self.extract_learnings(result),
        "progress": self.estimate_progress(),
        "should_continue": self.check_goal_achieved(),
        "adjustments": self.suggest_adjustments(result)
    }
    
    # Update memory with learnings
    self.memory.store(reflection["learned"])
    
    # Update state
    self.state.update(result)
    
    return reflection
```

**Key Point:** Reflection turns experience into learning. Without it, agents repeat mistakes.

---

## 🔄 The Complete Loop in Action

Let's see a full cycle with our Research Agent:

```python
class SimpleResearchAgent:
    def __init__(self, goal, tools):
        self.goal = goal
        self.tools = tools
        self.memory = []
        self.state = "not_started"
        self.history = []
        
    def run(self):
        """Main agent loop"""
        max_iterations = 10
        iteration = 0
        
        while not self.is_goal_achieved() and iteration < max_iterations:
            print(f"\n--- Iteration {iteration + 1} ---")
            
            # 1. PERCEIVE
            context = self.perceive()
            print(f"Context: {context['current_state']}")
            
            # 2. PLAN
            action = self.plan(context)
            print(f"Plan: {action.description}")
            
            # 3. ACT
            result = self.act(action)
            print(f"Result: {result.get('summary', result)}")
            
            # 4. REFLECT
            reflection = self.reflect(action, result)
            print(f"Reflection: Progress {reflection['progress']}%")
            
            # Update for next iteration
            iteration += 1
            
            if reflection['should_stop']:
                print("\n✅ Goal achieved!")
                break
                
        return self.get_final_output()
```

### Real Execution Example:

```
Goal: "Summarize the top 3 AI agent frameworks"

--- Iteration 1 ---
PERCEIVE: Need to start research, have web_search available
PLAN: Search for "top AI agent frameworks 2024"
ACT: web_search() → Found LangGraph, CrewAI, AutoGen
REFLECT: Good results, now need details. Progress: 25%

--- Iteration 2 ---
PERCEIVE: Have framework names, need more info
PLAN: Search for "LangGraph documentation"
ACT: web_search() → Found official docs
REFLECT: Got good info on LangGraph. Progress: 50%

--- Iteration 3 ---
PERCEIVE: Need info on other frameworks
PLAN: Search for "CrewAI vs AutoGen comparison"
ACT: web_search() → Found comparison article
REFLECT: Have enough info now. Progress: 75%

--- Iteration 4 ---
PERCEIVE: Have all info, need to synthesize
PLAN: Write summary comparing the 3 frameworks
ACT: text_writer() → Created summary document
REFLECT: Summary complete and accurate. Progress: 100%

✅ Goal achieved!
```

---

## 🔀 Different Loop Patterns

### 1. Fixed Iteration Loop
```python
for i in range(max_iterations):
    perceive() → plan() → act() → reflect()
```
**Use when:** You know roughly how many steps are needed

### 2. Goal-Based Loop
```python
while not goal_achieved():
    perceive() → plan() → act() → reflect()
```
**Use when:** The agent should work until successful

### 3. Reflective Loop with Branching
```python
while not done:
    context = perceive()
    plan = plan(context)
    result = act(plan)
    reflection = reflect(result)
    
    if reflection.needs_replanning:
        continue  # Re-plan with new info
    elif reflection.goal_achieved:
        break
    elif reflection.stuck:
        ask_for_help()
```
**Use when:** Complex tasks that might need course correction

---

## ⚠️ Common Pitfalls

### 1. Infinite Loops
**Problem:** Agent keeps repeating the same action
```python
# BAD: No progress tracking
while not done:
    search_papers()  # Does this forever!
```

**Solution:** Track progress and set max iterations
```python
# GOOD: Safety limits
max_iterations = 10
while not done and iterations < max_iterations:
    # ...
```

### 2. Poor Reflection
**Problem:** Agent doesn't learn from mistakes
```python
# BAD: No evaluation
result = act()
# Just continues without checking
```

**Solution:** Always evaluate results
```python
# GOOD: Check and adjust
result = act()
if not result.success:
    adjust_strategy()
```

### 3. No Exit Condition
**Problem:** Agent doesn't know when to stop
```python
# BAD: Vague stopping condition
while "not perfect":  # When is it perfect?
    improve()
```

**Solution:** Clear success criteria
```python
# GOOD: Specific goal
while accuracy < 0.95 and iterations < 20:
    improve()
```

---

## 💡 Key Takeaways

1. **The loop is everything** - Perceive → Plan → Act → Reflect is the core pattern
2. **Each step has a purpose:**
   - Perceive = Understanding
   - Plan = Decision making
   - Act = Execution
   - Reflect = Learning
3. **Reflection is what makes agents smart** - without it, they're just scripts
4. **Always have exit conditions** - both success and failure cases
5. **The loop enables autonomy** - the agent can keep going without you

**Remember:** The loop isn't just a code pattern - it's how autonomous intelligence works.

---

## 🛠️ Hands-On Challenge

Let's build your first agent loop!

### Task: Build a "Reflection Agent"

Create an agent that:
1. Takes a question as input
2. Generates an initial answer
3. Critiques its own answer
4. Improves the answer based on critique
5. Repeats until satisfied (max 3 iterations)

### Starter Code:

```python
class ReflectionAgent:
    def __init__(self, llm):
        self.llm = llm
        
    def run(self, question):
        answer = None
        
        for iteration in range(3):
            print(f"\n--- Iteration {iteration + 1} ---")
            
            # 1. PERCEIVE
            context = {
                "question": question,
                "current_answer": answer
            }
            
            # 2. PLAN & ACT (combined for simplicity)
            if answer is None:
                # First attempt
                answer = self.llm.complete(f"Answer this: {question}")
            else:
                # Improve based on critique
                answer = self.llm.complete(f"""
                Question: {question}
                Current answer: {answer}
                Improve this answer based on the critique.
                """)
            
            print(f"Answer: {answer[:100]}...")
            
            # 4. REFLECT
            critique = self.llm.complete(f"""
            Question: {question}
            Answer: {answer}
            
            Critique this answer:
            - Is it accurate?
            - Is it complete?
            - What's missing?
            Rate quality 1-10.
            """)
            
            print(f"Critique: {critique[:100]}...")
            
            # Check if good enough
            if "9" in critique or "10" in critique:
                print("✅ Satisfied with answer!")
                break
                
        return answer

# Try it:
# agent = ReflectionAgent(your_llm)
# final_answer = agent.run("Explain how transformers work")
```

### Your Task:

1. **Implement the reflection agent** using the starter code
2. **Run it with different questions** and observe how it improves
3. **Modify it** to track improvement metrics
4. **Share your results** in the repo!

---

## 🤔 Reflection Questions

1. **Why is the loop necessary?** Why can't an agent just plan everything upfront?

2. **What happens if you skip the Reflect step?** How does the agent's behavior change?

3. **How would you implement a "stuck detector"?** How do you know if the agent is making progress?

4. **When should an agent stop?** What are good exit conditions?

---

## 📚 Learn More

### Essential Reading
- **ReAct Paper** - [Reasoning and Acting](https://arxiv.org/abs/2210.03629) - The foundation of modern agent loops
- **Chain of Thought** - [How agents think step-by-step](https://arxiv.org/abs/2201.11903)
- **Reflexion** - [Agents that learn from mistakes](https://arxiv.org/abs/2303.11366)

### Videos
- [LangChain: Agent Loop Explained](https://www.youtube.com/langchain)
- [Building Your First Agent](https://www.deeplearning.ai/short-courses/)

### Code Examples
- LangGraph's agent executor
- CrewAI's task loop
- AutoGen's conversation patterns

---

## ➡️ What's Next?

You now understand HOW agents work internally. But agents are useless without memory!

**Next Lesson:** [Memory Systems: How Agents Remember](03-memory-systems.md)

We'll explore:
- Short-term vs long-term memory
- Vector databases for semantic memory
- Episodic memory (remembering what happened)
- Building a memory module for your agents

---

## 📝 My Personal Notes

*Fill this in as you learn:*

**What clicked for me:**
- 

**What confused me:**
- 

**Ideas for my own agents:**
- 

**Questions I still have:**
- 

---

**Lesson Status:** ✅ Complete  
**Estimated Time:** 30-40 minutes  
**Next Step:** Build the reflection agent, then move to Lesson 3!
