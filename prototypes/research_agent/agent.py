"""
Research Assistant Agent
Implements: Complete agent loop (Lessons 1-7)
- Perceive, Plan, Act, Reflect (Lesson 2)
- Memory systems (Lesson 3)
- Tool integration (Lesson 6)
- Error handling & recovery (Lesson 7)
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ResearchAgent:
    """
    Autonomous research agent that can:
    - Search multiple sources
    - Synthesize information
    - Create structured reports
    - Handle errors gracefully
    - Learn from experience
    """
    
    def __init__(self, llm, tools, memory_manager, max_iterations: int = 10):
        self.llm = llm
        self.tools = tools
        self.memory = memory_manager
        self.max_iterations = max_iterations
        
        # Agent state
        self.current_goal = None
        self.iteration = 0
        self.satisfied = False
        
        logger.info("ResearchAgent initialized")
    
    def research(self, topic: str, depth: str = "medium") -> Dict[str, Any]:
        """
        Main research method - complete agent loop
        
        Args:
            topic: Research topic
            depth: Research depth (quick, medium, deep)
        
        Returns:
            Research results and report
        """
        print(f"\n{'='*60}")
        print(f"🔍 RESEARCH TOPIC: {topic}")
        print(f"📊 DEPTH: {depth}")
        print(f"{'='*60}\n")
        
        self.current_goal = topic
        self.iteration = 0
        self.satisfied = False
        
        # Clear working memory for new task
        self.memory.clear_working_memory()
        
        # Main agent loop (Lesson 2)
        while not self.satisfied and self.iteration < self.max_iterations:
            self.iteration += 1
            print(f"\n--- ITERATION {self.iteration} ---")
            
            try:
                # THE AGENT LOOP
                context = self.perceive()
                plan = self.plan(context, depth)
                results = self.act(plan)
                self.reflect(results, plan)
                
            except Exception as e:
                logger.error(f"Error in iteration {self.iteration}: {e}")
                # Error recovery (Lesson 7)
                if not self._handle_error(e):
                    break
        
        # Generate final report
        report = self._generate_report()
        
        print(f"\n{'='*60}")
        print("✅ RESEARCH COMPLETE")
        print(f"{'='*60}\n")
        
        return report
    
    # ============================================
    # STEP 1: PERCEIVE (Lesson 2)
    # ============================================
    
    def perceive(self) -> Dict[str, Any]:
        """Gather current context"""
        print("[PERCEIVE] Gathering context...")
        
        context = {
            "goal": self.current_goal,
            "iteration": self.iteration,
            "max_iterations": self.max_iterations,
            "memory": self.memory.get_context(),
            "available_tools": self.tools.list_tools(),
            "previous_results": self._get_previous_results()
        }
        
        # Log to episodic memory
        self.memory.episodic.log_action("perceive", {
            "iteration": self.iteration,
            "context_size": len(str(context))
        })
        
        print(f"[PERCEIVE] Context ready (iteration {self.iteration}/{self.max_iterations})")
        
        return context
    
    # ============================================
    # STEP 2: PLAN (Lesson 2 & 5)
    # ============================================
    
    def plan(self, context: Dict, depth: str) -> Dict[str, Any]:
        """Create research plan based on context"""
        print("[PLAN] Creating research plan...")
        
        # Determine number of sources based on depth
        num_sources = {"quick": 2, "medium": 4, "deep": 6}.get(depth, 4)
        
        # Check what we already know
        known_facts = self._check_existing_knowledge(self.current_goal)
        
        if self.iteration == 1:
            # First iteration - broad search
            plan = {
                "strategy": "broad_search",
                "steps": [
                    {
                        "action": "search_wikipedia",
                        "tool": "wikipedia",
                        "params": {"query": self.current_goal, "limit": 2}
                    },
                    {
                        "action": "search_web",
                        "tool": "web_search",
                        "params": {"query": self.current_goal, "limit": 3}
                    }
                ],
                "reason": "First iteration - gathering general information"
            }
        else:
            # Later iterations - targeted search based on gaps
            gaps = self._identify_knowledge_gaps(context)
            
            if gaps:
                plan = {
                    "strategy": "fill_gaps",
                    "steps": [
                        {
                            "action": f"search_{gap_type}",
                            "tool": self._select_tool_for_gap(gap),
                            "params": {"query": gap, "limit": 2}
                        }
                        for gap in gaps[:2]  # Focus on top 2 gaps
                    ],
                    "reason": f"Filling knowledge gaps: {gaps}"
                }
            else:
                # We have enough info, synthesize
                plan = {
                    "strategy": "synthesize",
                    "steps": [
                        {
                            "action": "create_report",
                            "tool": "file_writer",
                            "params": {
                                "filename": f"research_{self.current_goal.replace(' ', '_')}",
                                "content": "to_be_generated"
                            }
                        }
                    ],
                    "reason": "Sufficient information gathered, creating report"
                }
                self.satisfied = True
        
        # Log plan
        self.memory.episodic.log_action("plan", {
            "iteration": self.iteration,
            "strategy": plan["strategy"],
            "steps_count": len(plan["steps"])
        })
        
        print(f"[PLAN] Strategy: {plan['strategy']}")
        print(f"[PLAN] Steps: {len(plan['steps'])}")
        print(f"[PLAN] Reason: {plan['reason']}")
        
        return plan
    
    # ============================================
    # STEP 3: ACT (Lesson 2 & 6)
    # ============================================
    
    def act(self, plan: Dict) -> List[Dict]:
        """Execute the plan using tools"""
        print("[ACT] Executing plan...")
        
        results = []
        
        for i, step in enumerate(plan["steps"]):
            print(f"[ACT] Step {i+1}/{len(plan['steps'])}: {step['action']}")
            
            try:
                # Get the tool
                tool = self.tools.get_tool(step["tool"])
                if not tool:
                    raise ValueError(f"Tool not found: {step['tool']}")
                
                # Execute tool with retry logic (Lesson 7)
                result = self._execute_with_retry(tool, step["params"])
                
                if result.success:
                    print(f"[ACT] ✅ Step {i+1} succeeded")
                    results.append({
                        "step": step["action"],
                        "success": True,
                        "data": result.data
                    })
                    
                    # Store in short-term memory
                    self.memory.short_term.add({
                        "type": "tool_result",
                        "tool": step["tool"],
                        "action": step["action"],
                        "data": result.data
                    })
                    
                else:
                    print(f"[ACT] ⚠️ Step {i+1} failed: {result.error}")
                    results.append({
                        "step": step["action"],
                        "success": False,
                        "error": result.error
                    })
                
            except Exception as e:
                logger.error(f"Error executing step {i+1}: {e}")
                results.append({
                    "step": step["action"],
                    "success": False,
                    "error": str(e)
                })
        
        # Log results
        self.memory.episodic.log_action("act", {
            "iteration": self.iteration,
            "steps_executed": len(results),
            "successful_steps": sum(1 for r in results if r["success"])
        })
        
        success_rate = sum(1 for r in results if r["success"]) / len(results) if results else 0
        print(f"[ACT] Completed {len(results)} steps (Success rate: {success_rate:.0%})")
        
        return results
    
    # ============================================
    # STEP 4: REFLECT (Lesson 2)
    # ============================================
    
    def reflect(self, results: List[Dict], plan: Dict):
        """Evaluate results and decide next steps"""
        print("[REFLECT] Evaluating results...")
        
        # Count successes
        successful = sum(1 for r in results if r.get("success"))
        total = len(results)
        
        # Extract learnings
        learnings = []
        for result in results:
            if result.get("success") and result.get("data"):
                data = result["data"]
                
                # Store interesting facts
                if "results" in data:
                    for item in data["results"][:2]:  # Top 2 results
                        fact = item.get("snippet") or item.get("summary", "")
                        if fact:
                            self.memory.remember_fact(fact[:200], self.current_goal)
                            learnings.append(fact[:100])
        
        # Determine if we should continue
        progress = self._estimate_progress(results)
        
        reflection = {
            "success_rate": successful / total if total > 0 else 0,
            "learnings_count": len(learnings),
            "progress": progress,
            "should_continue": progress < 0.8 and self.iteration < self.max_iterations
        }
        
        # Log reflection
        self.memory.episodic.log_action("reflect", reflection)
        
        print(f"[REFLECT] Success rate: {reflection['success_rate']:.0%}")
        print(f"[REFLECT] Progress: {progress:.0%}")
        print(f"[REFLECT] Learnings: {len(learnings)} facts stored")
        
        if progress >= 0.8:
            self.satisfied = True
            print("[REFLECT] ✅ Research goal achieved!")
        elif self.iteration >= self.max_iterations:
            print("[REFLECT] ⏱️ Max iterations reached")
            self.satisfied = True
        else:
            print("[REFLECT] 🔄 Continuing research...")
    
    # ============================================
    # HELPER METHODS
    # ============================================
    
    def _execute_with_retry(self, tool, params, max_retries=3):
        """Execute tool with retry logic (Lesson 7)"""
        for attempt in range(max_retries):
            result = tool.execute(**params)
            
            if result.success:
                return result
            
            if attempt < max_retries - 1:
                print(f"[RETRY] Attempt {attempt + 1} failed, retrying...")
                import time
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return result
    
    def _handle_error(self, error: Exception) -> bool:
        """Handle errors during execution (Lesson 7)"""
        print(f"[ERROR RECOVERY] Handling error: {error}")
        
        self.memory.episodic.log_action("error", {
            "iteration": self.iteration,
            "error": str(error),
            "error_type": type(error).__name__
        })
        
        # For now, continue to next iteration
        return self.iteration < self.max_iterations
    
    def _get_previous_results(self) -> List[Dict]:
        """Get results from previous iterations"""
        recent = self.memory.short_term.get_latest(5)
        return [r for r in recent if r.get("type") == "tool_result"]
    
    def _check_existing_knowledge(self, topic: str) -> List[str]:
        """Check what we already know about the topic"""
        return self.memory.long_term.get_topic_facts(topic)
    
    def _identify_knowledge_gaps(self, context: Dict) -> List[str]:
        """Identify what information is still missing"""
        # Simple heuristic: check if we have enough sources
        results_count = sum(
            1 for item in context["memory"]["recent"]
            if item.get("type") == "tool_result" and item.get("data")
        )
        
        if results_count < 3:
            return [f"{self.current_goal} details", f"{self.current_goal} recent"]
        
        return []
    
    def _select_tool_for_gap(self, gap: str) -> str:
        """Select appropriate tool to fill knowledge gap"""
        if "recent" in gap.lower() or "news" in gap.lower():
            return "news"
        elif "detail" in gap.lower():
            return "wikipedia"
        else:
            return "web_search"
    
    def _estimate_progress(self, results: List[Dict]) -> float:
        """Estimate how much of the research is complete"""
        # Simple heuristic based on:
        # - Number of successful tool calls
        # - Facts stored in memory
        
        successful_tools = sum(1 for r in results if r.get("success"))
        facts_count = len(self.memory.long_term.get_topic_facts(self.current_goal))
        
        # Progress = weighted combination
        tool_progress = min(successful_tools / 5, 1.0)  # Target: 5 sources
        knowledge_progress = min(facts_count / 10, 1.0)  # Target: 10 facts
        
        return (tool_progress * 0.6) + (knowledge_progress * 0.4)
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate final research report using LLM"""
        print("\n[REPORT] Generating final report...")
        
        # Gather all information
        facts = self.memory.long_term.get_topic_facts(self.current_goal)
        recent_results = self.memory.short_term.get()
        
        # Create report using LLM
        prompt = f"""
        Create a comprehensive research report on: {self.current_goal}
        
        Information gathered:
        
        Facts ({len(facts)}):
        {chr(10).join(f"- {fact}" for fact in facts[:10])}
        
        Recent research results:
        {json.dumps(recent_results[-5:], indent=2)}
        
        Create a well-structured report with:
        1. Executive Summary
        2. Key Findings (3-5 points)
        3. Detailed Analysis
        4. Sources
        5. Conclusion
        
        Format: Markdown
        """
        
        report_content = self.llm.generate(prompt)
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"research_{self.current_goal.replace(' ', '_')}_{timestamp}"
        
        file_tool = self.tools.get_tool("file_writer")
        save_result = file_tool.execute(
            filename=filename,
            content=report_content,
            format="md"
        )
        
        print(f"[REPORT] ✅ Report saved: {save_result.data.get('filename')}")
        
        return {
            "topic": self.current_goal,
            "report": report_content,
            "file": save_result.data if save_result.success else None,
            "stats": {
                "iterations": self.iteration,
                "facts_found": len(facts),
                "sources_used": len(self.memory.short_term.get()),
                "success_rate": self.memory.episodic.get_success_rate()
            },
            "session_summary": self.memory.episodic.get_summary()
        }