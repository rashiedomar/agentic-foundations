"""
Planning + Tools Integration Example
=====================================
Save as: prototypes/planning_with_tools_example.py

This demonstrates how to integrate the planning agent (Lesson 5)
with the tool system (Lesson 6) to create a complete working agent!

This is the REAL THING - an agent that can plan AND execute!
"""

import json
from typing import List, Dict, Any

# Import from your prototypes
# from planning_agent import PlanningAgent, Task, TaskStatus
# from tool_system import ToolRegistry, WebSearchTool, FileOperationsTool, CalculatorTool

# For this example, we'll include simplified versions

# ============================================================================
# Simplified Planning Agent (from Lesson 5)
# ============================================================================

class Task:
    def __init__(self, id: str, description: str, tool: str = None, dependencies: List[str] = None):
        self.id = id
        self.description = description
        self.tool = tool
        self.dependencies = dependencies or []
        self.status = "pending"
        self.result = None

class SimplePlanner:
    """Simplified planner for demonstration"""
    
    def create_plan(self, goal: str, available_tools: List[str]) -> List[Task]:
        """Create a plan to achieve the goal"""
        
        print(f"\n📋 Planning for goal: {goal}")
        print(f"Available tools: {available_tools}\n")
        
        # Simple keyword-based planning (in real life, LLM does this)
        tasks = []
        
        if "research" in goal.lower():
            tasks = [
                Task("task_1", "Search for information", tool="websearch", dependencies=[]),
                Task("task_2", "Analyze search results", tool=None, dependencies=["task_1"]),
                Task("task_3", "Write report", tool="fileoperations", dependencies=["task_2"])
            ]
        
        elif "calculate" in goal.lower():
            tasks = [
                Task("task_1", "Extract numbers from request", tool=None, dependencies=[]),
                Task("task_2", "Perform calculation", tool="calculator", dependencies=["task_1"]),
                Task("task_3", "Format result", tool=None, dependencies=["task_2"])
            ]
        
        else:
            # Generic plan
            tasks = [
                Task("task_1", "Understand the goal", tool=None, dependencies=[]),
                Task("task_2", "Execute appropriate action", tool=None, dependencies=["task_1"]),
                Task("task_3", "Verify completion", tool=None, dependencies=["task_2"])
            ]
        
        print("🗺️  Plan created:")
        for task in tasks:
            deps = f" (depends on: {task.dependencies})" if task.dependencies else ""
            tool_str = f" [uses: {task.tool}]" if task.tool else ""
            print(f"  {task.id}: {task.description}{tool_str}{deps}")
        
        return tasks

# ============================================================================
# Complete Agent with Planning + Tools
# ============================================================================

class CompleteAgent:
    """
    An agent that can both plan AND execute using tools
    This is what you've been building toward!
    """
    
    def __init__(self, name: str, tools, planner):
        self.name = name
        self.tools = tools
        self.planner = planner
        self.memory = []
    
    def execute_goal(self, goal: str) -> Dict[str, Any]:
        """
        Complete workflow: Plan → Execute → Report
        """
        print("\n" + "=" * 60)
        print(f"🤖 {self.name} Starting Mission")
        print("=" * 60)
        print(f"Goal: {goal}\n")
        
        # STEP 1: Create plan
        print("STEP 1: Planning")
        print("-" * 40)
        
        available_tools = self.tools.list_tools()
        tasks = self.planner.create_plan(goal, available_tools)
        
        # STEP 2: Execute plan
        print("\nSTEP 2: Execution")
        print("-" * 40)
        
        completed = set()
        results = []
        
        for task in tasks:
            # Wait for dependencies
            if not all(dep in completed for dep in task.dependencies):
                print(f"\n⏸️  {task.id} waiting for dependencies...")
                continue
            
            print(f"\n▶️  Executing {task.id}: {task.description}")
            
            if task.tool:
                # Execute using tool
                result = self._execute_task_with_tool(task)
            else:
                # Execute using reasoning (simulated)
                result = self._execute_task_with_reasoning(task)
            
            task.result = result
            task.status = "completed"
            completed.add(task.id)
            results.append({
                "task": task.description,
                "result": result
            })
            
            print(f"   ✅ Completed")
        
        # STEP 3: Summarize
        print("\nSTEP 3: Summary")
        print("-" * 40)
        
        summary = {
            "goal": goal,
            "status": "completed",
            "tasks_completed": len(completed),
            "total_tasks": len(tasks),
            "results": results
        }
        
        print(f"✅ Mission complete!")
        print(f"   Completed {len(completed)}/{len(tasks)} tasks")
        
        return summary
    
    def _execute_task_with_tool(self, task: Task) -> Any:
        """Execute a task using a tool"""
        print(f"   🛠️  Using tool: {task.tool}")
        
        # Extract parameters from task description (simplified)
        params = self._extract_params(task)
        
        # Execute tool
        result = self.tools.execute_tool(task.tool, **params)
        
        if result.success:
            print(f"   📊 Tool returned: {self._summarize_result(result.data)}")
            return result.data
        else:
            print(f"   ❌ Tool failed: {result.error}")
            return None
    
    def _execute_task_with_reasoning(self, task: Task) -> str:
        """Execute a task using reasoning (simulated LLM)"""
        print(f"   💭 Using reasoning...")
        
        # In real implementation, this calls the LLM
        # For now, simulate with simple logic
        
        if "analyze" in task.description.lower():
            return "Analysis: Found 3 relevant sources with key insights about AI agents"
        elif "understand" in task.description.lower():
            return "Understanding: User wants information about AI agents"
        elif "format" in task.description.lower():
            return "Formatted result ready"
        elif "verify" in task.description.lower():
            return "Verification: Task completed successfully"
        else:
            return f"Completed: {task.description}"
    
    def _extract_params(self, task: Task) -> dict:
        """Extract parameters from task description (simplified)"""
        # In real implementation, LLM extracts these
        
        if task.tool == "websearch":
            # Extract query from task
            return {
                "query": "AI agents tutorial",
                "max_results": 5
            }
        elif task.tool == "calculator":
            return {
                "expression": "2 + 2"
            }
        elif task.tool == "fileoperations":
            return {
                "operation": "write",
                "path": "agent_report.txt",
                "content": "Agent research report generated successfully!"
            }
        else:
            return {}
    
    def _summarize_result(self, data: Any) -> str:
        """Summarize tool result for display"""
        if isinstance(data, list):
            return f"{len(data)} items"
        elif isinstance(data, dict):
            return f"{len(data)} fields"
        elif isinstance(data, str):
            return data[:50] + "..." if len(data) > 50 else data
        else:
            return str(data)

# ============================================================================
# Example Scenarios
# ============================================================================

def scenario_1_research():
    """Scenario: Research and create a report"""
    
    print("\n" + "🎬" * 20)
    print("SCENARIO 1: Research Agent")
    print("🎬" * 20)
    
    # Set up (in real code, import from your prototypes)
    from tool_system import ToolRegistry, WebSearchTool, FileOperationsTool
    
    tools = ToolRegistry()
    tools.register(WebSearchTool())
    tools.register(FileOperationsTool())
    
    planner = SimplePlanner()
    agent = CompleteAgent("ResearchBot", tools, planner)
    
    # Execute goal
    result = agent.execute_goal(
        "Research the latest developments in AI agents and create a summary report"
    )
    
    print("\n📄 Final Result:")
    print(json.dumps(result, indent=2))

def scenario_2_calculation():
    """Scenario: Mathematical calculation"""
    
    print("\n" + "🎬" * 20)
    print("SCENARIO 2: Calculator Agent")
    print("🎬" * 20)
    
    from tool_system import ToolRegistry, CalculatorTool
    
    tools = ToolRegistry()
    tools.register(CalculatorTool())
    
    planner = SimplePlanner()
    agent = CompleteAgent("MathBot", tools, planner)
    
    result = agent.execute_goal(
        "Calculate the compound interest on $10,000 at 5% for 3 years"
    )
    
    print("\n📄 Final Result:")
    print(json.dumps(result, indent=2))

def scenario_3_multi_tool():
    """Scenario: Using multiple tools together"""
    
    print("\n" + "🎬" * 20)
    print("SCENARIO 3: Multi-Tool Agent")
    print("🎬" * 20)
    
    from tool_system import (
        ToolRegistry,
        WebSearchTool,
        CalculatorTool,
        WeatherTool,
        FileOperationsTool
    )
    
    # Create full-featured agent
    tools = ToolRegistry()
    tools.register(WebSearchTool())
    tools.register(CalculatorTool())
    tools.register(WeatherTool())
    tools.register(FileOperationsTool())
    
    planner = SimplePlanner()
    agent = CompleteAgent("SuperAgent", tools, planner)
    
    # Complex goal requiring multiple tools
    result = agent.execute_goal(
        "Research AI agents, calculate adoption rates, and save a report"
    )
    
    print("\n📊 Tool Usage Stats:")
    stats = tools.get_all_stats()
    for tool_name, tool_stats in stats.items():
        if tool_stats['executions'] > 0:
            print(f"\n{tool_name}:")
            print(f"  Executions: {tool_stats['executions']}")
            print(f"  Success rate: {tool_stats['success_rate']:.1%}")
            print(f"  Total time: {tool_stats['total_time']:.2f}s")

# ============================================================================
# Integration Patterns
# ============================================================================

def pattern_1_sequential():
    """Pattern: Sequential tool execution"""
    
    print("\n📚 PATTERN 1: Sequential Execution")
    print("Task 1 → Task 2 → Task 3")
    print("-" * 40)
    
    # Each task waits for previous to complete
    print("""
    Plan:
    1. Search for information (uses: websearch)
    2. Analyze results (uses: reasoning)
    3. Save report (uses: fileoperations)
    
    Dependencies ensure correct order!
    """)

def pattern_2_parallel():
    """Pattern: Parallel tool execution"""
    
    print("\n📚 PATTERN 2: Parallel Execution")
    print("Task 1 ─┐")
    print("Task 2 ─┼→ Task 4")
    print("Task 3 ─┘")
    print("-" * 40)
    
    print("""
    Plan:
    1. Search source A (no dependencies)
    2. Search source B (no dependencies)
    3. Search source C (no dependencies)
    4. Combine results (depends on 1, 2, 3)
    
    Tasks 1-3 can run in parallel!
    """)

def pattern_3_conditional():
    """Pattern: Conditional tool execution"""
    
    print("\n📚 PATTERN 3: Conditional Execution")
    print("Task 1 → [condition] → Task 2a OR Task 2b")
    print("-" * 40)
    
    print("""
    Plan:
    1. Try to find cached data
    2a. If found: Use cached data
    2b. If not found: Fetch fresh data
    3. Process data
    
    Different paths based on results!
    """)

# ============================================================================
# Main Demo
# ============================================================================

def main():
    """Run complete demonstration"""
    
    print("\n")
    print("*" * 60)
    print("  LESSON 5 + 6 INTEGRATION: COMPLETE AGENT!")
    print("*" * 60)
    print("""
This demonstrates the power of combining:
- Planning (Lesson 5): Breaking down goals into tasks
- Tools (Lesson 6): Actual capabilities to DO things

Together = A real AI agent! 🤖
""")
    
    # Run scenarios
    input("Press Enter to run Scenario 1 (Research Agent)...")
    scenario_1_research()
    
    input("\nPress Enter to run Scenario 2 (Calculator Agent)...")
    scenario_2_calculation()
    
    input("\nPress Enter to run Scenario 3 (Multi-Tool Agent)...")
    scenario_3_multi_tool()
    
    print("\n" + "=" * 60)
    print("🎉 INTEGRATION COMPLETE!")
    print("=" * 60)
    print("""
You now have:
✅ Planning Agent (breaks down goals)
✅ Tool System (executes actions)
✅ Complete Agent (plans + acts)

Next steps:
1. Connect to a real LLM (OpenAI, Anthropic)
2. Add more tools for your use case
3. Improve planning with better strategies
4. Add error handling and recovery (Lesson 7)
5. Build your specific agent!

🚀 You're ready to build real agents!
""")

if __name__ == "__main__":
    # Show integration patterns
    pattern_1_sequential()
    pattern_2_parallel()
    pattern_3_conditional()
    
    # Run full demo
    main()