"""
Planning Agent - Production Implementation
==========================================
Save as: prototypes/planning_agent.py

A production-ready planning agent with multiple strategies,
error handling, and tool orchestration.

Usage:
    from planning_agent import PlanningAgent, PlanningStrategy
    
    agent = PlanningAgent(llm_client=your_llm)
    result = agent.execute_goal(
        goal="Research top AI startups",
        strategy=PlanningStrategy.REACT
    )
"""

import json
import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Core Data Models
# ============================================================================

class TaskStatus(Enum):
    """Task execution states"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"

class PlanningStrategy(Enum):
    """Available planning strategies"""
    CHAIN_OF_THOUGHT = "cot"
    TREE_OF_THOUGHTS = "tot"
    REACT = "react"
    AUTO = "auto"  # Agent chooses strategy

@dataclass
class Task:
    """Represents a single task in a plan"""
    id: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    tool: Optional[str] = None
    result: Any = None
    error: Optional[str] = None
    attempts: int = 0
    max_attempts: int = 3
    created_at: datetime = field(default_factory=datetime.now)
    
    def can_execute(self, completed_tasks: set) -> bool:
        """Check if all dependencies are satisfied"""
        return all(dep in completed_tasks for dep in self.dependencies)
    
    def mark_completed(self, result: Any):
        """Mark task as successfully completed"""
        self.status = TaskStatus.COMPLETED
        self.result = result
        logger.info(f"✅ Task {self.id} completed")
    
    def mark_failed(self, error: str):
        """Mark task as failed"""
        self.status = TaskStatus.FAILED
        self.error = error
        self.attempts += 1
        logger.error(f"❌ Task {self.id} failed: {error}")
    
    def can_retry(self) -> bool:
        """Check if task can be retried"""
        return self.attempts < self.max_attempts

@dataclass
class Plan:
    """Represents a complete execution plan"""
    goal: str
    tasks: List[Task]
    strategy: PlanningStrategy
    created_at: datetime = field(default_factory=datetime.now)
    completed_tasks: set = field(default_factory=set)
    
    def get_next_executable_tasks(self) -> List[Task]:
        """Get all tasks that can be executed now"""
        return [
            task for task in self.tasks
            if task.status == TaskStatus.PENDING
            and task.can_execute(self.completed_tasks)
        ]
    
    def is_complete(self) -> bool:
        """Check if all tasks are completed"""
        return len(self.completed_tasks) == len(self.tasks)
    
    def get_progress(self) -> Dict[str, Any]:
        """Get plan execution progress"""
        return {
            "total_tasks": len(self.tasks),
            "completed": len(self.completed_tasks),
            "failed": len([t for t in self.tasks if t.status == TaskStatus.FAILED]),
            "pending": len([t for t in self.tasks if t.status == TaskStatus.PENDING]),
            "progress_percent": (len(self.completed_tasks) / len(self.tasks) * 100) if self.tasks else 0
        }

# ============================================================================
# Tool System
# ============================================================================

@dataclass
class Tool:
    """Represents an available tool"""
    name: str
    description: str
    function: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def execute(self, **kwargs) -> Any:
        """Execute the tool with given parameters"""
        try:
            logger.info(f"🔧 Executing tool: {self.name}")
            return self.function(**kwargs)
        except Exception as e:
            logger.error(f"Tool {self.name} failed: {str(e)}")
            raise

class ToolRegistry:
    """Manages available tools"""
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool):
        """Register a new tool"""
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name"""
        return self.tools.get(name)
    
    def list_tools(self) -> List[str]:
        """List all available tool names"""
        return list(self.tools.keys())
    
    def get_tool_descriptions(self) -> str:
        """Get formatted descriptions of all tools"""
        descriptions = []
        for tool in self.tools.values():
            descriptions.append(f"- {tool.name}: {tool.description}")
        return "\n".join(descriptions)

# ============================================================================
# Planning Strategies
# ============================================================================

class ChainOfThoughtPlanner:
    """Implements Chain-of-Thought planning"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def create_plan(self, goal: str, tools: ToolRegistry) -> List[Task]:
        """Create a linear plan using CoT reasoning"""
        prompt = f"""
Goal: {goal}

Available tools:
{tools.get_tool_descriptions()}

Create a step-by-step plan to achieve this goal.
Think through each step carefully and consider tool usage.

Output JSON format:
{{
    "reasoning": "Your step-by-step thinking...",
    "tasks": [
        {{
            "id": "task_1",
            "description": "specific action to take",
            "tool": "tool_name or null",
            "dependencies": []
        }},
        ...
    ]
}}
"""
        try:
            response = self.llm.complete(prompt)
            plan_data = json.loads(response)
            
            logger.info(f"CoT Reasoning: {plan_data.get('reasoning', 'N/A')}")
            
            tasks = []
            for task_data in plan_data["tasks"]:
                task = Task(
                    id=task_data["id"],
                    description=task_data["description"],
                    tool=task_data.get("tool"),
                    dependencies=task_data.get("dependencies", [])
                )
                tasks.append(task)
            
            return tasks
        except Exception as e:
            logger.error(f"CoT planning failed: {str(e)}")
            raise

class TreeOfThoughtsPlanner:
    """Implements Tree-of-Thoughts planning"""
    
    def __init__(self, llm, num_branches: int = 3):
        self.llm = llm
        self.num_branches = num_branches
    
    def create_plan(self, goal: str, tools: ToolRegistry) -> List[Task]:
        """Create plan by exploring multiple approaches"""
        # Generate multiple plan candidates
        candidates = self._generate_candidates(goal, tools)
        
        # Evaluate each candidate
        evaluated = self._evaluate_candidates(candidates, goal)
        
        # Select best plan
        best_plan = max(evaluated, key=lambda x: x["score"])
        
        logger.info(f"ToT selected plan with score: {best_plan['score']}")
        
        return best_plan["tasks"]
    
    def _generate_candidates(self, goal: str, tools: ToolRegistry) -> List[Dict]:
        """Generate multiple plan candidates"""
        prompt = f"""
Goal: {goal}

Available tools: {tools.get_tool_descriptions()}

Generate {self.num_branches} different approaches to achieve this goal.
Each approach should be valid but emphasize different strategies.

Output JSON with {self.num_branches} different plans.
"""
        # Implementation would generate multiple plans
        # Simplified for prototype
        candidates = []
        for i in range(self.num_branches):
            candidates.append({
                "approach": f"Approach {i+1}",
                "tasks": []  # Would be populated from LLM
            })
        return candidates
    
    def _evaluate_candidates(self, candidates: List[Dict], goal: str) -> List[Dict]:
        """Evaluate and score each candidate plan"""
        for candidate in candidates:
            # Simplified scoring - real implementation would be more sophisticated
            candidate["score"] = len(candidate["tasks"]) * 0.5  # Placeholder
        return candidates

class ReActPlanner:
    """Implements ReAct (Reason + Act) planning"""
    
    def __init__(self, llm, tools: ToolRegistry, max_iterations: int = 10):
        self.llm = llm
        self.tools = tools
        self.max_iterations = max_iterations
        self.memory = []
    
    def execute(self, goal: str) -> Any:
        """Execute goal using ReAct loop"""
        logger.info(f"🔄 Starting ReAct loop for goal: {goal}")
        
        for iteration in range(self.max_iterations):
            # Think
            thought = self._think(goal)
            logger.info(f"💭 Thought: {thought['reasoning']}")
            
            # Check if done
            if thought.get("is_complete"):
                logger.info("✅ ReAct loop completed")
                return thought.get("final_answer")
            
            # Act
            action = thought.get("action")
            if action:
                observation = self._act(action)
                logger.info(f"👁️  Observation: {observation}")
                
                # Add to memory
                self.memory.append({
                    "thought": thought["reasoning"],
                    "action": action,
                    "observation": observation
                })
        
        logger.warning("⚠️  ReAct loop reached max iterations")
        return "Max iterations reached without completion"
    
    def _think(self, goal: str) -> Dict[str, Any]:
        """Reasoning step"""
        memory_str = "\n".join([
            f"Step {i+1}: {m['thought']} -> {m['action']} -> {m['observation']}"
            for i, m in enumerate(self.memory)
        ])
        
        prompt = f"""
Goal: {goal}

Previous steps:
{memory_str or 'None yet'}

Available tools: {self.tools.get_tool_descriptions()}

What should I do next?

Output JSON:
{{
    "reasoning": "your thinking...",
    "is_complete": false,
    "action": {{"tool": "tool_name", "params": {{...}}}} or null
}}
"""
        response = self.llm.complete(prompt)
        return json.loads(response)
    
    def _act(self, action: Dict[str, Any]) -> str:
        """Action execution step"""
        tool_name = action.get("tool")
        params = action.get("params", {})
        
        tool = self.tools.get(tool_name)
        if not tool:
            return f"Error: Tool {tool_name} not found"
        
        try:
            result = tool.execute(**params)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"

# ============================================================================
# Main Planning Agent
# ============================================================================

class PlanningAgent:
    """
    Main planning agent with multiple strategies
    """
    
    def __init__(self, llm, tools: Optional[ToolRegistry] = None):
        self.llm = llm
        self.tools = tools or ToolRegistry()
        self.current_plan: Optional[Plan] = None
        
        # Initialize strategy implementations
        self.strategies = {
            PlanningStrategy.CHAIN_OF_THOUGHT: ChainOfThoughtPlanner(llm),
            PlanningStrategy.TREE_OF_THOUGHTS: TreeOfThoughtsPlanner(llm),
            PlanningStrategy.REACT: ReActPlanner(llm, self.tools)
        }
    
    def execute_goal(
        self,
        goal: str,
        strategy: PlanningStrategy = PlanningStrategy.CHAIN_OF_THOUGHT,
        parallel: bool = False
    ) -> Dict[str, Any]:
        """
        Execute a goal using specified planning strategy
        
        Args:
            goal: The goal to achieve
            strategy: Planning strategy to use
            parallel: Whether to execute independent tasks in parallel
            
        Returns:
            Execution results
        """
        logger.info(f"🎯 Executing goal: {goal}")
        logger.info(f"📋 Strategy: {strategy.value}")
        
        try:
            # Auto-select strategy if requested
            if strategy == PlanningStrategy.AUTO:
                strategy = self._select_strategy(goal)
                logger.info(f"🤖 Auto-selected strategy: {strategy.value}")
            
            # Special handling for ReAct (doesn't use task-based planning)
            if strategy == PlanningStrategy.REACT:
                result = self.strategies[strategy].execute(goal)
                return {
                    "status": "completed",
                    "strategy": strategy.value,
                    "result": result
                }
            
            # Create plan
            planner = self.strategies[strategy]
            tasks = planner.create_plan(goal, self.tools)
            self.current_plan = Plan(goal=goal, tasks=tasks, strategy=strategy)
            
            logger.info(f"📝 Plan created with {len(tasks)} tasks")
            
            # Execute plan
            self._execute_plan(parallel=parallel)
            
            # Return results
            return {
                "status": "completed" if self.current_plan.is_complete() else "partial",
                "strategy": strategy.value,
                "progress": self.current_plan.get_progress(),
                "results": [
                    {"task": t.description, "result": t.result}
                    for t in self.current_plan.tasks
                    if t.status == TaskStatus.COMPLETED
                ]
            }
            
        except Exception as e:
            logger.error(f"❌ Goal execution failed: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def _execute_plan(self, parallel: bool = False):
        """Execute the current plan"""
        if not self.current_plan:
            raise ValueError("No plan to execute")
        
        max_iterations = len(self.current_plan.tasks) * 2
        iterations = 0
        
        while not self.current_plan.is_complete() and iterations < max_iterations:
            iterations += 1
            
            # Get executable tasks
            executable_tasks = self.current_plan.get_next_executable_tasks()
            
            if not executable_tasks:
                logger.warning("No executable tasks found - possible circular dependency")
                break
            
            # Execute tasks
            for task in executable_tasks:
                self._execute_task(task)
        
        progress = self.current_plan.get_progress()
        logger.info(f"📊 Plan execution finished: {progress['completed']}/{progress['total_tasks']} tasks completed")
    
    def _execute_task(self, task: Task):
        """Execute a single task"""
        logger.info(f"▶️  Executing task: {task.description}")
        task.status = TaskStatus.IN_PROGRESS
        
        try:
            # Get tool if specified
            if task.tool:
                tool = self.tools.get(task.tool)
                if not tool:
                    raise ValueError(f"Tool {task.tool} not found")
                
                # Execute tool (would need proper parameter extraction)
                result = tool.execute()
            else:
                # Use LLM to execute task
                result = self._llm_execute_task(task)
            
            task.mark_completed(result)
            self.current_plan.completed_tasks.add(task.id)
            
        except Exception as e:
            task.mark_failed(str(e))
            
            # Retry logic
            if task.can_retry():
                logger.info(f"🔄 Retrying task {task.id} (attempt {task.attempts + 1})")
                task.status = TaskStatus.PENDING
            else:
                logger.error(f"Task {task.id} failed after {task.max_attempts} attempts")
    
    def _llm_execute_task(self, task: Task) -> Any:
        """Execute task using LLM"""
        prompt = f"""
Task: {task.description}
Goal context: {self.current_plan.goal}

Execute this task and provide the result.
"""
        return self.llm.complete(prompt)
    
    def _select_strategy(self, goal: str) -> PlanningStrategy:
        """Auto-select best strategy for goal"""
        prompt = f"""
Goal: {goal}

Which planning strategy is best?
- CoT: Simple, linear problems
- ToT: Complex decisions with tradeoffs
- ReAct: Need tools and iteration

Output: {{"strategy": "cot|tot|react", "reasoning": "..."}}
"""
        response = self.llm.complete(prompt)
        data = json.loads(response)
        
        strategy_map = {
            "cot": PlanningStrategy.CHAIN_OF_THOUGHT,
            "tot": PlanningStrategy.TREE_OF_THOUGHTS,
            "react": PlanningStrategy.REACT
        }
        
        return strategy_map.get(data["strategy"], PlanningStrategy.CHAIN_OF_THOUGHT)

# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Mock LLM for testing
    class MockLLM:
        def complete(self, prompt: str) -> str:
            # Return mock responses
            if "Create a step-by-step plan" in prompt:
                return json.dumps({
                    "reasoning": "Breaking down the goal into sequential steps",
                    "tasks": [
                        {"id": "task_1", "description": "Step 1", "tool": None, "dependencies": []},
                        {"id": "task_2", "description": "Step 2", "tool": None, "dependencies": ["task_1"]}
                    ]
                })
            return "Mock result"
    
    # Create agent
    agent = PlanningAgent(llm=MockLLM())
    
    # Register a test tool
    def mock_search(query: str) -> str:
        return f"Search results for: {query}"
    
    agent.tools.register(Tool(
        name="search",
        description="Search the web",
        function=mock_search
    ))
    
    # Execute goal
    result = agent.execute_goal(
        goal="Research the top 3 AI startups",
        strategy=PlanningStrategy.CHAIN_OF_THOUGHT
    )
    
    print(json.dumps(result, indent=2))