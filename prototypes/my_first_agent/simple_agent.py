"""
Simple Reflection Agent - Your First AI Agent!

This agent answers questions and improves through self-reflection.
Run this file to see it in action!
"""

from datetime import datetime
from typing import List, Dict

# ============================================
# PART 1: MEMORY SYSTEMS
# ============================================

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


# ============================================
# PART 2: LLM INTERFACE (MOCK VERSION)
# ============================================

class MockLLM:
    """Mock LLM for testing without API keys"""
    
    def __init__(self):
        self.call_count = 0
    
    def complete(self, prompt: str) -> str:
        """Generate a response (simulated)"""
        self.call_count += 1
        print(f"\n💭 [LLM CALL #{self.call_count}]")
        
        # Simulate responses based on prompt
        if "critique" in prompt.lower() or "evaluate" in prompt.lower():
            # Return a critique
            if self.call_count <= 3:
                return "Quality: 7/10. The answer is decent but could be more detailed and include examples."
            else:
                return "Quality: 9/10. Excellent answer! Clear, comprehensive, and well-explained."
        else:
            # Return an answer
            if "difference between" in prompt.lower() and "agent" in prompt.lower():
                if self.call_count == 1:
                    # First answer - intentionally simple
                    return """A language model responds to prompts and generates text. 
An AI agent can autonomously pursue goals using tools and planning."""
                else:
                    # Improved answer
                    return """A language model (like GPT-4) is a one-shot system that responds to prompts but doesn't take actions or persist memory. It's reactive.

An AI agent, on the other hand, is an autonomous system that:
1. Has goals it works toward
2. Can plan multi-step solutions
3. Uses tools to interact with the world (search, code execution, APIs)
4. Maintains memory across interactions
5. Learns from experience through reflection

Example: ChatGPT is a language model. An agent that searches the web, reads papers, synthesizes findings, and writes a report autonomously is an AI agent."""
            else:
                return f"This is a simulated answer to: {prompt[:100]}..."


# ============================================
# PART 3: THE REFLECTION AGENT
# ============================================

class ReflectionAgent:
    """Agent that improves answers through self-reflection"""
    
    def __init__(self, llm, max_iterations: int = 3):
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
        """Main agent loop"""
        self.current_question = question
        self.iteration = 0
        self.satisfied = False
        
        print(f"\n{'='*60}")
        print(f"🎯 QUESTION: {question}")
        print(f"{'='*60}")
        
        while not self.satisfied and self.iteration < self.max_iterations:
            self.iteration += 1
            print(f"\n{'─'*60}")
            print(f"🔄 ITERATION {self.iteration}/{self.max_iterations}")
            print(f"{'─'*60}")
            
            # THE AGENT LOOP: Perceive → Plan → Act → Reflect
            context = self.perceive()
            action = self.plan(context)
            result = self.act(action)
            self.reflect(result)
        
        print(f"\n{'='*60}")
        print(f"✅ FINAL ANSWER:")
        print(f"{'='*60}")
        print(self.current_answer)
        print(f"{'='*60}\n")
        
        return self.current_answer
    
    # ====== STEP 1: PERCEIVE ======
    
    def perceive(self) -> Dict:
        """Gather current context"""
        context = {
            "question": self.current_question,
            "current_answer": self.current_answer,
            "iteration": self.iteration,
            "recent_history": self.short_term.get()
        }
        
        self.episodic.log("perceive", {"iteration": self.iteration})
        print(f"👁️  [PERCEIVE] Gathering context...")
        
        return context
    
    # ====== STEP 2: PLAN ======
    
    def plan(self, context: Dict) -> Dict:
        """Decide what to do next"""
        if context["current_answer"] is None:
            action = {
                "type": "generate_answer",
                "reason": "No answer exists yet"
            }
        else:
            action = {
                "type": "improve_answer",
                "reason": "Improving based on critique"
            }
        
        self.episodic.log("plan", action)
        print(f"🧠 [PLAN] Action: {action['type']}")
        
        return action
    
    # ====== STEP 3: ACT ======
    
    def act(self, action: Dict) -> Dict:
        """Execute the planned action"""
        if action["type"] == "generate_answer":
            result = self._generate_initial_answer()
        else:
            result = self._improve_answer()
        
        self.short_term.add({
            "action": action["type"],
            "success": result["success"]
        })
        
        self.episodic.log("act", {"action_type": action["type"]})
        
        return result
    
    def _generate_initial_answer(self) -> Dict:
        """Generate the first answer"""
        print(f"⚡ [ACT] Generating initial answer...")
        
        prompt = f"Question: {self.current_question}\n\nProvide a clear and accurate answer."
        answer = self.llm.complete(prompt)
        self.current_answer = answer
        
        print(f"✓ Generated answer ({len(answer)} characters)")
        
        return {"success": True, "answer": answer}
    
    def _improve_answer(self) -> Dict:
        """Improve based on critique"""
        print(f"⚡ [ACT] Improving answer based on critique...")
        
        # Get last critique from memory
        recent = self.short_term.get()
        last_critique = None
        for item in reversed(recent):
            if "critique" in item:
                last_critique = item["critique"]
                break
        
        prompt = f"""Question: {self.current_question}
        
Current Answer: {self.current_answer}

Critique: {last_critique}

Improve the answer by addressing the critique."""
        
        improved = self.llm.complete(prompt)
        self.current_answer = improved
        
        print(f"✓ Improved answer ({len(improved)} characters)")
        
        return {"success": True, "answer": improved}
    
    # ====== STEP 4: REFLECT ======
    
    def reflect(self, result: Dict):
        """Evaluate and decide next steps"""
        print(f"🤔 [REFLECT] Evaluating quality...")
        
        critique_prompt = f"""Question: {self.current_question}
Answer: {self.current_answer}

Critically evaluate this answer:
1. Is it accurate?
2. Is it complete?
3. What could be improved?

Rate quality 1-10. Format: Quality: [score]/10"""
        
        critique = self.llm.complete(critique_prompt)
        
        # Store critique
        self.short_term.add({"critique": critique})
        self.episodic.log("reflect", {"iteration": self.iteration})
        
        print(f"📊 Critique: {critique[:100]}...")
        
        # Check if satisfied
        if "9" in critique or "10" in critique:
            self.satisfied = True
            print(f"✅ [REFLECT] Satisfied with quality!")
        elif self.iteration >= self.max_iterations:
            print(f"⏱️  [REFLECT] Max iterations reached")
        else:
            print(f"🔄 [REFLECT] Will improve in next iteration")
    
    # ====== UTILITY METHODS ======
    
    def get_session_summary(self) -> str:
        """Get summary of the session"""
        return self.episodic.get_summary()


# ============================================
# PART 4: TEST IT!
# ============================================

def main():
    """Test the reflection agent"""
    
    print("\n" + "="*60)
    print("🤖 REFLECTION AGENT - DEMO")
    print("="*60)
    
    # Create LLM (mock version - no API needed)
    llm = MockLLM()
    
    # Create agent
    agent = ReflectionAgent(llm, max_iterations=3)
    
    # Test question
    question = "What is the difference between a language model and an AI agent?"
    
    # Run agent
    final_answer = agent.run(question)
    
    # Print summary
    print("\n" + "="*60)
    print("📊 SESSION SUMMARY")
    print("="*60)
    print(agent.get_session_summary())
    print(f"Total LLM calls: {llm.call_count}")
    
    print("\n✨ Done! Try modifying the question or max_iterations to experiment!\n")


if __name__ == "__main__":
    main()