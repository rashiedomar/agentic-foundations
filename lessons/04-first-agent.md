# Lesson 4: Building Your First Complete Agent

> **What you'll learn:** How to combine the agent loop and memory systems into a working autonomous agent that can accomplish real tasks.

---

## 🎯 Learning Objectives

By the end of this lesson, you'll be able to:
- Combine the agent loop with memory systems
- Build a functional agent from scratch
- Handle errors and edge cases
- Test and debug your agent
- Understand the complete agent architecture

---

## 🎬 What We're Building

Today, you're building your first **real agent** - a **Reflection Agent** that:
- Takes a question
- Generates an initial answer
- Critiques its own answer
- Improves the answer iteratively
- Stops when satisfied (or max iterations reached)

**Why this agent?**
- Simple enough to understand
- Complex enough to be useful
- Teaches all the core concepts
- You can actually run and test it!

---

## 🏗️ The Architecture

Here's what we're building:

```
┌─────────────────────────────────────────────────┐
│         REFLECTION AGENT                        │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌──────────────┐      ┌──────────────┐       │
│  │  SHORT-TERM  │      │  EPISODIC    │       │
│  │   MEMORY     │      │   MEMORY     │       │
│  │              │      │              │       │
│  │ - Current Q  │      │ - All loops  │       │
│  │ - Curr Ans   │      │ - Critiques  │       │
│  │ - Last Crit  │      │ - History    │       │
│  └──────┬───────┘      └──────┬───────┘       │
│         │                     │                │
│         └──────────┬──────────┘                │
│                    │                           │
│         ┌──────────▼──────────┐                │
│         │   AGENT LOOP        │                │
│         │                     │                │
│         │  1. PERCEIVE        │                │
│         │  2. PLAN            │                │
│         │  3. ACT             │                │
│         │  4. REFLECT         │                │
│         └─────────────────────┘                │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

## 📝 Step-by-Step Implementation

### Step 1: Set Up the Project

First, create the folder structure:

```bash
cd agentic-foundations/prototypes
mkdir my_first_agent
cd my_first_agent
touch agent.py
touch test.py
```

### Step 2: Memory Components

Let's build our memory systems:

```python
# agent.py

from datetime import datetime
from typing import List, Dict, Optional

class ShortTermMemory:
    """Working memory for current task"""
    
    def __init__(self, max_size: int = 5):
        self.max_size = max_size
        self.context = []
    
    def add(self, item: Dict):
        """Add item to working memory"""
        self.context.append(item)
        
        # Keep only recent items
        if len(self.context) > self.max_size:
            self.context = self.context[-self.max_size:]
    
    def get(self) -> List[Dict]:
        """Get current context"""
        return self.context
    
    def clear(self):
        """Clear working memory"""
        self.context = []


class EpisodicMemory:
    """Log of all actions and results"""
    
    def __init__(self):
        self.episodes = []
    
    def log(self, action_type: str, details: Dict):
        """Log an action"""
        episode = {
            "timestamp": datetime.now().isoformat(),
            "action": action_type,
            "details": details
        }
        self.episodes.append(episode)
    
    def get_history(self) -> List[Dict]:
        """Get all episodes"""
        return self.episodes
    
    def get_summary(self) -> str:
        """Summarize the session"""
        summary = f"Total actions: {len(self.episodes)}\n"
        
        action_counts = {}
        for ep in self.episodes:
            action = ep["action"]
            action_counts[action] = action_counts.get(action, 0) + 1
        
        for action, count in action_counts.items():
            summary += f"  - {action}: {count}\n"
        
        return summary
```

### Step 3: The LLM Interface

We need a way to call the language model:

```python
# agent.py (continued)

class SimpleLLM:
    """Simple LLM interface - replace with real API"""
    
    def __init__(self, model_name: str = "gpt-4"):
        self.model_name = model_name
        # In practice, initialize OpenAI/Anthropic client here
    
    def complete(self, prompt: str) -> str:
        """
        Generate completion from prompt
        
        In practice, this would call:
        - OpenAI API: openai.chat.completions.create()
        - Anthropic API: anthropic.messages.create()
        - Local model: ollama, etc.
        """
        # TODO: Replace with actual API call
        # For now, this is a placeholder
        print(f"\n[LLM CALL] {prompt[:100]}...")
        
        # Simulated response (you'll replace this)
        return "This is where the LLM response would be"
```

### Step 4: The Core Agent

Now, let's build the agent with the full loop:

```python
# agent.py (continued)

class ReflectionAgent:
    """
    Agent that iteratively improves its answers through self-reflection
    """
    
    def __init__(self, llm: SimpleLLM, max_iterations: int = 3):
        self.llm = llm
        self.max_iterations = max_iterations
        
        # Memory systems
        self.short_term = ShortTermMemory()
        self.episodic = EpisodicMemory()
        
        # Agent state
        self.current_question = None
        self.current_answer = None
        self.iteration = 0
        self.satisfied = False
    
    def run(self, question: str) -> str:
        """
        Main agent loop
        
        Args:
            question: The question to answer
            
        Returns:
            Final answer after reflection iterations
        """
        self.current_question = question
        self.iteration = 0
        self.satisfied = False
        
        print(f"\n{'='*60}")
        print(f"QUESTION: {question}")
        print(f"{'='*60}")
        
        while not self.satisfied and self.iteration < self.max_iterations:
            self.iteration += 1
            print(f"\n--- ITERATION {self.iteration} ---")
            
            # THE AGENT LOOP
            context = self.perceive()
            action = self.plan(context)
            result = self.act(action)
            self.reflect(result)
        
        print(f"\n{'='*60}")
        print(f"FINAL ANSWER:\n{self.current_answer}")
        print(f"{'='*60}")
        
        return self.current_answer
    
    # ============================================
    # STEP 1: PERCEIVE
    # ============================================
    
    def perceive(self) -> Dict:
        """Gather current context"""
        context = {
            "question": self.current_question,
            "current_answer": self.current_answer,
            "iteration": self.iteration,
            "recent_history": self.short_term.get(),
            "max_iterations": self.max_iterations
        }
        
        # Log perception
        self.episodic.log("perceive", {
            "iteration": self.iteration,
            "has_answer": self.current_answer is not None
        })
        
        print(f"[PERCEIVE] Iteration {self.iteration}/{self.max_iterations}")
        if self.current_answer:
            print(f"[PERCEIVE] Current answer exists: {len(self.current_answer)} chars")
        
        return context
    
    # ============================================
    # STEP 2: PLAN
    # ============================================
    
    def plan(self, context: Dict) -> Dict:
        """Decide what to do next"""
        
        if context["current_answer"] is None:
            # First iteration: generate initial answer
            action = {
                "type": "generate_answer",
                "reason": "No answer exists yet, need to create initial response"
            }
        else:
            # Later iterations: improve based on critique
            action = {
                "type": "improve_answer",
                "reason": "Answer exists, need to improve based on reflection"
            }
        
        # Log plan
        self.episodic.log("plan", action)
        
        print(f"[PLAN] Action: {action['type']}")
        print(f"[PLAN] Reason: {action['reason']}")
        
        return action
    
    # ============================================
    # STEP 3: ACT
    # ============================================
    
    def act(self, action: Dict) -> Dict:
        """Execute the planned action"""
        
        if action["type"] == "generate_answer":
            result = self._generate_initial_answer()
        else:
            result = self._improve_answer()
        
        # Update short-term memory
        self.short_term.add({
            "action": action["type"],
            "result": result["success"]
        })
        
        # Log action
        self.episodic.log("act", {
            "action_type": action["type"],
            "success": result["success"]
        })
        
        return result
    
    def _generate_initial_answer(self) -> Dict:
        """Generate the first answer"""
        print(f"[ACT] Generating initial answer...")
        
        prompt = f"""
        Question: {self.current_question}
        
        Provide a clear, accurate, and comprehensive answer to this question.
        Think step-by-step if needed.
        """
        
        answer = self.llm.complete(prompt)
        self.current_answer = answer
        
        print(f"[ACT] Generated answer ({len(answer)} chars)")
        
        return {
            "success": True,
            "answer": answer
        }
    
    def _improve_answer(self) -> Dict:
        """Improve the current answer based on previous critique"""
        print(f"[ACT] Improving answer based on critique...")
        
        # Get the last critique from short-term memory
        recent = self.short_term.get()
        last_critique = None
        for item in reversed(recent):
            if "critique" in item:
                last_critique = item["critique"]
                break
        
        prompt = f"""
        Question: {self.current_question}
        
        Current Answer: {self.current_answer}
        
        Critique: {last_critique}
        
        Improve the answer by addressing the issues mentioned in the critique.
        Make it more accurate, complete, and clear.
        """
        
        improved_answer = self.llm.complete(prompt)
        self.current_answer = improved_answer
        
        print(f"[ACT] Improved answer ({len(improved_answer)} chars)")
        
        return {
            "success": True,
            "answer": improved_answer
        }
    
    # ============================================
    # STEP 4: REFLECT
    # ============================================
    
    def reflect(self, result: Dict):
        """Evaluate the result and decide next steps"""
        print(f"[REFLECT] Evaluating quality...")
        
        # Generate critique of current answer
        critique_prompt = f"""
        Question: {self.current_question}
        Answer: {self.current_answer}
        
        Critically evaluate this answer:
        1. Is it accurate?
        2. Is it complete?
        3. Is it clear and well-explained?
        4. What could be improved?
        
        Rate the quality from 1-10 where:
        - 1-5: Poor, needs significant improvement
        - 6-7: Decent, but could be better
        - 8-9: Good, minor improvements possible
        - 10: Excellent, no improvements needed
        
        Format: Quality: [score]/10
        Then explain your rating.
        """
        
        critique = self.llm.complete(critique_prompt)
        
        # Store critique in memory
        self.short_term.add({
            "critique": critique
        })
        
        # Log reflection
        self.episodic.log("reflect", {
            "iteration": self.iteration,
            "critique_length": len(critique)
        })
        
        print(f"[REFLECT] Critique generated")
        
        # Check if we should stop
        if self._is_satisfied(critique):
            self.satisfied = True
            print(f"[REFLECT] ✅ Satisfied with answer quality!")
        elif self.iteration >= self.max_iterations:
            print(f"[REFLECT] ⏱️ Max iterations reached")
        else:
            print(f"[REFLECT] 🔄 Will improve in next iteration")
    
    def _is_satisfied(self, critique: str) -> bool:
        """Check if the answer is good enough"""
        # Simple heuristic: look for high quality score
        critique_lower = critique.lower()
        
        # Check for quality indicators
        if "quality: 9" in critique_lower or "quality: 10" in critique_lower:
            return True
        if "excellent" in critique_lower and "no improvements needed" in critique_lower:
            return True
        
        return False
    
    # ============================================
    # UTILITY METHODS
    # ============================================
    
    def get_session_summary(self) -> str:
        """Get summary of the session"""
        return self.episodic.get_summary()
    
    def get_full_history(self) -> List[Dict]:
        """Get complete episode history"""
        return self.episodic.get_history()
```

### Step 5: Test It!

Create a test file to run your agent:

```python
# test.py

from agent import ReflectionAgent, SimpleLLM

def test_reflection_agent():
    """Test the reflection agent with a sample question"""
    
    # Initialize LLM (you'll need to set up API keys for real use)
    llm = SimpleLLM(model_name="gpt-4")
    
    # Create agent
    agent = ReflectionAgent(llm, max_iterations=3)
    
    # Test question
    question = "What is the difference between a language model and an AI agent?"
    
    # Run agent
    final_answer = agent.run(question)
    
    # Print summary
    print("\n" + "="*60)
    print("SESSION SUMMARY")
    print("="*60)
    print(agent.get_session_summary())
    
    return final_answer

if __name__ == "__main__":
    test_reflection_agent()
```

---

## 🔧 Making It Work with Real APIs

To use a real LLM, replace the `SimpleLLM` class:

### Using OpenAI:

```python
from openai import OpenAI

class SimpleLLM:
    def __init__(self, model_name="gpt-4o-mini", api_key=None):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
    
    def complete(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content
```

### Using Anthropic (Claude):

```python
from anthropic import Anthropic

class SimpleLLM:
    def __init__(self, model_name="claude-3-5-sonnet-20241022", api_key=None):
        self.client = Anthropic(api_key=api_key)
        self.model_name = model_name
    
    def complete(self, prompt: str) -> str:
        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=1000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return message.content[0].text
```

---

## 🐛 Debugging Tips

### 1. Add Verbose Logging

```python
class ReflectionAgent:
    def __init__(self, llm, max_iterations=3, verbose=True):
        self.verbose = verbose
        # ... rest of init
    
    def _log(self, message):
        if self.verbose:
            print(f"[DEBUG] {message}")
```

### 2. Save Sessions to File

```python
import json

def save_session(agent, filename="session_log.json"):
    """Save session history to file"""
    history = agent.get_full_history()
    with open(filename, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Session saved to {filename}")
```

### 3. Test with Simple Questions First

```python
# Start simple
easy_questions = [
    "What is 2+2?",
    "What is Python?",
    "Name three colors"
]

for q in easy_questions:
    print(f"\nTesting: {q}")
    agent.run(q)
```

---

## 📊 Example Output

When you run the agent, you should see something like:

```
============================================================
QUESTION: What is the difference between a language model and an AI agent?
============================================================

--- ITERATION 1 ---
[PERCEIVE] Iteration 1/3
[PLAN] Action: generate_answer
[PLAN] Reason: No answer exists yet, need to create initial response
[ACT] Generating initial answer...
[ACT] Generated answer (523 chars)
[REFLECT] Evaluating quality...
[REFLECT] Critique generated
[REFLECT] 🔄 Will improve in next iteration

--- ITERATION 2 ---
[PERCEIVE] Iteration 2/3
[PERCEIVE] Current answer exists: 523 chars
[PLAN] Action: improve_answer
[PLAN] Reason: Answer exists, need to improve based on reflection
[ACT] Improving answer based on critique...
[ACT] Improved answer (687 chars)
[REFLECT] Evaluating quality...
[REFLECT] Critique generated
[REFLECT] ✅ Satisfied with answer quality!

============================================================
FINAL ANSWER:
[The improved answer appears here]
============================================================

============================================================
SESSION SUMMARY
============================================================
Total actions: 8
  - perceive: 2
  - plan: 2
  - act: 2
  - reflect: 2
```

---

## 🎯 Challenges & Exercises

### Challenge 1: Add Iteration Limit Check
Make the agent stop if it's not improving after 2 iterations

### Challenge 2: Add Cost Tracking
Count how many LLM calls are made and estimate the cost

### Challenge 3: Multi-Question Agent
Modify the agent to handle multiple questions in sequence

### Challenge 4: Add Long-Term Memory
Store successful answer patterns for future use

### Challenge 5: Compare Answers
Show the difference between iteration 1 and final answer

---

## ⚠️ Common Issues

### Issue 1: Infinite Loop
**Problem:** Agent keeps iterating without stopping

**Solution:**
```python
# Add safety check
if self.iteration >= self.max_iterations:
    print("WARNING: Max iterations reached!")
    break
```

### Issue 2: API Errors
**Problem:** API calls failing

**Solution:**
```python
def complete(self, prompt: str) -> str:
    try:
        response = self.client.chat.completions.create(...)
        return response.choices[0].message.content
    except Exception as e:
        print(f"API Error: {e}")
        return "Error generating response"
```

### Issue 3: Empty Responses
**Problem:** LLM returns empty string

**Solution:**
```python
def act(self, action):
    result = self._generate_answer()
    
    if not result.get("answer"):
        print("WARNING: Empty response, retrying...")
        result = self._generate_answer()
    
    return result
```

---

## 💡 Key Takeaways

1. **Combining components is powerful** - Loop + Memory = Intelligence
2. **Start simple** - This agent does one thing well
3. **Logging is crucial** - You can't debug what you can't see
4. **Iteration enables improvement** - Reflection makes agents smarter
5. **Real agents need error handling** - Always expect things to fail

**Remember:** This is your foundation. Every complex agent follows this same pattern!

---

## 📚 What You Built

✅ Complete agent loop (Perceive → Plan → Act → Reflect)  
✅ Short-term memory for context  
✅ Episodic memory for logging  
✅ Self-reflection capability  
✅ Iterative improvement  
✅ Error handling basics  

**You now have a working agent! 🎉**

---

## ➡️ What's Next?

Now that you have a basic agent, you can:
- Add tools (web search, calculator, etc.)
- Build different types of agents (research, coding, data analysis)
- Implement multi-agent systems
- Add long-term memory with vector databases

**Next Lesson:** [Adding Tools to Your Agent](05-tools-integration.md) *(Coming soon)*

---

## 📝 My Personal Notes

*Your reflections after building:*

**What worked well:**
- 

**What was hard:**
- 

**Ideas for improvements:**
- 

**Next agent I want to build:**
- 

---

**Lesson Status:** ✅ Complete  
**Estimated Time:** 1-2 hours (reading + implementing)  
**Your Mission:** Build and test this agent!
