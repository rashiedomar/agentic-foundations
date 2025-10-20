"""
Tool System - Production Implementation
========================================
Save as: prototypes/tool_system.py

A complete tool system that integrates with planning_agent.py
Includes base classes, concrete tools, and integration patterns.

Usage:
    from tool_system import ToolRegistry, WebSearchTool, FileOperationsTool
    from planning_agent import PlanningAgent
    
    # Create tools
    registry = ToolRegistry()
    registry.register(WebSearchTool())
    registry.register(FileOperationsTool())
    
    # Create agent with tools
    agent = PlanningAgent(llm=your_llm, tools=registry)
    result = agent.execute_goal("Research AI agents and create report")
"""

import json
import logging
import time
import os
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from functools import wraps
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Core Tool System
# ============================================================================

@dataclass
class ToolResult:
    """Standardized tool execution result"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata,
            "execution_time": self.execution_time
        }

class BaseTool(ABC):
    """
    Base class for all tools
    
    Every tool must implement:
    - description property
    - parameters property
    - _execute method
    """
    
    def __init__(self):
        self.name = self.__class__.__name__.replace("Tool", "").lower()
        self.execution_count = 0
        self.total_execution_time = 0.0
        self.success_count = 0
        self.failure_count = 0
        self.last_error = None
    
    @property
    @abstractmethod
    def description(self) -> str:
        """
        Tool description for LLM to understand when to use it
        Should include: what it does, when to use it, what it returns
        """
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        """
        Parameter schema for the tool
        Format: {
            "param_name": {
                "type": "string|integer|boolean|object|array",
                "description": "what this parameter does",
                "required": True|False,
                "default": value (optional)
            }
        }
        """
        pass
    
    @abstractmethod
    def _execute(self, **kwargs) -> Any:
        """
        Internal execution logic - override this in subclasses
        Should return the actual data, not ToolResult
        """
        pass
    
    def execute(self, **kwargs) -> ToolResult:
        """
        Execute tool with error handling and metrics
        This method should not be overridden
        """
        self.execution_count += 1
        start_time = time.time()
        
        try:
            logger.info(f"🔧 Executing {self.name} with params: {kwargs}")
            
            # Validate parameters
            self._validate_parameters(kwargs)
            
            # Execute
            result = self._execute(**kwargs)
            
            # Record success
            execution_time = time.time() - start_time
            self.total_execution_time += execution_time
            self.success_count += 1
            
            logger.info(f"✅ {self.name} completed in {execution_time:.2f}s")
            
            return ToolResult(
                success=True,
                data=result,
                metadata={
                    "tool": self.name,
                    "execution_count": self.execution_count,
                    "timestamp": datetime.now().isoformat()
                },
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.total_execution_time += execution_time
            self.failure_count += 1
            self.last_error = str(e)
            
            logger.error(f"❌ {self.name} failed: {e}")
            
            return ToolResult(
                success=False,
                error=str(e),
                metadata={
                    "tool": self.name,
                    "execution_count": self.execution_count,
                    "timestamp": datetime.now().isoformat()
                },
                execution_time=execution_time
            )
    
    def _validate_parameters(self, params: dict):
        """Validate parameters against schema"""
        # Check required parameters
        for param_name, param_schema in self.parameters.items():
            if param_schema.get("required", False):
                if param_name not in params:
                    raise ValueError(f"Missing required parameter: {param_name}")
        
        # Check for unexpected parameters
        allowed_params = set(self.parameters.keys())
        provided_params = set(params.keys())
        unexpected = provided_params - allowed_params
        
        if unexpected:
            logger.warning(f"Unexpected parameters ignored: {unexpected}")
    
    def get_stats(self) -> dict:
        """Get tool usage statistics"""
        return {
            "name": self.name,
            "executions": self.execution_count,
            "successes": self.success_count,
            "failures": self.failure_count,
            "success_rate": self.success_count / self.execution_count if self.execution_count > 0 else 0,
            "total_time": self.total_execution_time,
            "avg_time": self.total_execution_time / self.execution_count if self.execution_count > 0 else 0,
            "last_error": self.last_error
        }

class ToolRegistry:
    """
    Central registry for managing tools
    Integrates with planning agents
    """
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self._tool_stats: Dict[str, Dict] = {}
    
    def register(self, tool: BaseTool):
        """Register a new tool"""
        self.tools[tool.name] = tool
        logger.info(f"✅ Registered tool: {tool.name}")
    
    def unregister(self, tool_name: str):
        """Remove a tool from registry"""
        if tool_name in self.tools:
            del self.tools[tool_name]
            logger.info(f"❌ Unregistered tool: {tool_name}")
    
    def get(self, name: str) -> Optional[BaseTool]:
        """Get tool by name"""
        return self.tools.get(name)
    
    def list_tools(self) -> List[str]:
        """List all tool names"""
        return list(self.tools.keys())
    
    def get_tool_descriptions(self) -> str:
        """
        Get formatted descriptions for LLM
        Used by planning agents to decide which tools to use
        """
        descriptions = []
        for tool in self.tools.values():
            desc = f"""
Tool: {tool.name}
Description: {tool.description}
Parameters: {json.dumps(tool.parameters, indent=2)}
"""
            descriptions.append(desc.strip())
        return "\n\n".join(descriptions)
    
    def get_all_stats(self) -> Dict[str, dict]:
        """Get statistics for all tools"""
        return {name: tool.get_stats() for name, tool in self.tools.items()}
    
    def execute_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool by name"""
        tool = self.get(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                error=f"Tool '{tool_name}' not found"
            )
        
        return tool.execute(**kwargs)

# ============================================================================
# Rate Limiting Decorator
# ============================================================================

class RateLimiter:
    """Rate limiter for tools"""
    
    def __init__(self, max_calls: int, period: int):
        """
        Args:
            max_calls: Maximum number of calls allowed
            period: Time period in seconds
        """
        self.max_calls = max_calls
        self.period = period
        self.calls: List[float] = []
    
    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            
            # Remove old calls outside the time window
            self.calls = [c for c in self.calls if now - c < self.period]
            
            # Check if rate limit exceeded
            if len(self.calls) >= self.max_calls:
                wait_time = self.period - (now - self.calls[0])
                raise Exception(
                    f"Rate limit exceeded. Wait {wait_time:.1f}s before calling again. "
                    f"(Limit: {self.max_calls} calls per {self.period}s)"
                )
            
            # Record this call
            self.calls.append(now)
            
            return func(*args, **kwargs)
        return wrapper

# ============================================================================
# Concrete Tool Implementations
# ============================================================================

class WebSearchTool(BaseTool):
    """
    Web search tool using a search API
    """
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        self.api_key = api_key or os.getenv("SEARCH_API_KEY")
    
    @property
    def description(self) -> str:
        return """
Search the web for information using a search engine.
Use this when you need current information, want to find websites,
or need to research a topic online.

Returns a list of search results with titles, URLs, and snippets.
"""
    
    @property
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "query": {
                "type": "string",
                "description": "The search query",
                "required": True
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return",
                "required": False,
                "default": 10
            }
        }
    
    @RateLimiter(max_calls=10, period=60)  # 10 calls per minute
    def _execute(self, query: str, max_results: int = 10) -> List[dict]:
        """Execute web search"""
        
        # Mock implementation - replace with actual API
        # Example: Brave Search API, Serper, or SerpAPI
        
        logger.info(f"Searching for: {query}")
        
        # In production, you would call a real search API:
        # response = requests.get(
        #     "https://api.search.brave.com/res/v1/web/search",
        #     params={"q": query, "count": max_results},
        #     headers={"X-Subscription-Token": self.api_key}
        # )
        # return response.json()["web"]["results"]
        
        # Mock results for demonstration
        mock_results = [
            {
                "title": f"Result {i+1} for '{query}'",
                "url": f"https://example.com/result{i+1}",
                "snippet": f"This is a relevant result about {query}..."
            }
            for i in range(min(3, max_results))
        ]
        
        return mock_results

class WebFetchTool(BaseTool):
    """
    Fetch content from a URL
    """
    
    @property
    def description(self) -> str:
        return """
Fetch the full content of a webpage given its URL.
Use this after searching to read the actual content of interesting pages.

Returns the text content of the webpage.
"""
    
    @property
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "url": {
                "type": "string",
                "description": "The URL to fetch",
                "required": True
            }
        }
    
    def _execute(self, url: str) -> str:
        """Fetch webpage content"""
        
        try:
            response = requests.get(
                url,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10
            )
            response.raise_for_status()
            
            # In production, you'd parse HTML properly
            # from bs4 import BeautifulSoup
            # soup = BeautifulSoup(response.text, 'html.parser')
            # return soup.get_text()
            
            return response.text[:1000]  # Return first 1000 chars
            
        except requests.RequestException as e:
            raise Exception(f"Failed to fetch {url}: {str(e)}")

class FileOperationsTool(BaseTool):
    """
    Read, write, or list files
    """
    
    def __init__(self, workspace_dir: str = "."):
        super().__init__()
        self.workspace_dir = Path(workspace_dir).resolve()
    
    @property
    def description(self) -> str:
        return """
Perform file operations: read, write, or list files.
Use this to save reports, read data files, or explore the workspace.

Operations:
- read: Read contents of a file
- write: Write content to a file
- list: List files in a directory
"""
    
    @property
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "operation": {
                "type": "string",
                "description": "Operation to perform: read, write, or list",
                "required": True
            },
            "path": {
                "type": "string",
                "description": "File or directory path",
                "required": True
            },
            "content": {
                "type": "string",
                "description": "Content to write (only for write operation)",
                "required": False
            }
        }
    
    def _execute(self, operation: str, path: str, content: str = None) -> Any:
        """Execute file operation"""
        
        # Security: Ensure path is within workspace
        full_path = (self.workspace_dir / path).resolve()
        if not full_path.is_relative_to(self.workspace_dir):
            raise ValueError("Path outside workspace not allowed")
        
        if operation == "read":
            if not full_path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            return full_path.read_text()
        
        elif operation == "write":
            if content is None:
                raise ValueError("Content required for write operation")
            
            # Create parent directories if needed
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            full_path.write_text(content)
            return f"Wrote {len(content)} characters to {path}"
        
        elif operation == "list":
            if not full_path.exists():
                raise FileNotFoundError(f"Directory not found: {path}")
            
            if full_path.is_file():
                return [full_path.name]
            
            return [f.name for f in full_path.iterdir()]
        
        else:
            raise ValueError(f"Unknown operation: {operation}")

class CalculatorTool(BaseTool):
    """
    Perform mathematical calculations
    """
    
    @property
    def description(self) -> str:
        return """
Perform mathematical calculations.
Use this for arithmetic operations or when you need precise numeric results.

Supports: addition, subtraction, multiplication, division, power, sqrt
"""
    
    @property
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "expression": {
                "type": "string",
                "description": "Mathematical expression to evaluate (e.g., '2 + 2', 'sqrt(16)')",
                "required": True
            }
        }
    
    def _execute(self, expression: str) -> float:
        """Evaluate mathematical expression"""
        
        # Safe evaluation using ast
        import ast
        import operator
        import math
        
        # Allowed operations
        operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
        }
        
        # Allowed functions
        functions = {
            'sqrt': math.sqrt,
            'abs': abs,
            'round': round,
        }
        
        def eval_expr(node):
            if isinstance(node, ast.Num):
                return node.n
            elif isinstance(node, ast.BinOp):
                return operators[type(node.op)](
                    eval_expr(node.left),
                    eval_expr(node.right)
                )
            elif isinstance(node, ast.UnaryOp):
                return operators[type(node.op)](eval_expr(node.operand))
            elif isinstance(node, ast.Call):
                func_name = node.func.id
                if func_name not in functions:
                    raise ValueError(f"Function {func_name} not allowed")
                args = [eval_expr(arg) for arg in node.args]
                return functions[func_name](*args)
            else:
                raise ValueError(f"Unsupported operation: {type(node)}")
        
        try:
            tree = ast.parse(expression, mode='eval')
            result = eval_expr(tree.body)
            return result
        except Exception as e:
            raise ValueError(f"Invalid expression: {expression}. Error: {str(e)}")

class WeatherTool(BaseTool):
    """
    Get current weather information
    """
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        self.api_key = api_key or os.getenv("WEATHER_API_KEY")
    
    @property
    def description(self) -> str:
        return """
Get current weather information for a location.
Use this when you need weather data, temperature, or conditions.

Returns weather data including temperature, conditions, humidity.
"""
    
    @property
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "location": {
                "type": "string",
                "description": "City name or location (e.g., 'London', 'New York')",
                "required": True
            }
        }
    
    def _execute(self, location: str) -> dict:
        """Get weather data"""
        
        # Mock implementation - replace with actual weather API
        # Example: OpenWeatherMap, WeatherAPI, etc.
        
        # In production:
        # response = requests.get(
        #     "https://api.openweathermap.org/data/2.5/weather",
        #     params={"q": location, "appid": self.api_key, "units": "metric"}
        # )
        # return response.json()
        
        # Mock data
        return {
            "location": location,
            "temperature": 22,
            "condition": "Partly Cloudy",
            "humidity": 65,
            "wind_speed": 15
        }

# ============================================================================
# Example Usage & Integration
# ============================================================================

def example_usage():
    """Example of using the tool system"""
    
    print("=" * 60)
    print("TOOL SYSTEM EXAMPLE")
    print("=" * 60)
    
    # Create tool registry
    registry = ToolRegistry()
    
    # Register tools
    registry.register(WebSearchTool())
    registry.register(WebFetchTool())
    registry.register(FileOperationsTool())
    registry.register(CalculatorTool())
    registry.register(WeatherTool())
    
    print(f"\n✅ Registered {len(registry.list_tools())} tools")
    print(f"Tools: {registry.list_tools()}\n")
    
    # Example 1: Search the web
    print("Example 1: Web Search")
    print("-" * 40)
    result = registry.execute_tool("websearch", query="AI agents", max_results=3)
    if result.success:
        print(f"Found {len(result.data)} results:")
        for r in result.data:
            print(f"  - {r['title']}")
    print()
    
    # Example 2: Calculator
    print("Example 2: Calculator")
    print("-" * 40)
    result = registry.execute_tool("calculator", expression="2 + 2 * 3")
    if result.success:
        print(f"Result: {result.data}")
    print()
    
    # Example 3: Weather
    print("Example 3: Weather")
    print("-" * 40)
    result = registry.execute_tool("weather", location="Tokyo")
    if result.success:
        print(f"Weather: {json.dumps(result.data, indent=2)}")
    print()
    
    # Example 4: File operations
    print("Example 4: File Operations")
    print("-" * 40)
    result = registry.execute_tool(
        "fileoperations",
        operation="write",
        path="test_output.txt",
        content="This is a test file created by the agent!"
    )
    if result.success:
        print(f"✅ {result.data}")
    print()
    
    # Show tool statistics
    print("Tool Statistics:")
    print("-" * 40)
    stats = registry.get_all_stats()
    for tool_name, tool_stats in stats.items():
        print(f"\n{tool_name}:")
        print(f"  Executions: {tool_stats['executions']}")
        print(f"  Success rate: {tool_stats['success_rate']:.1%}")
        print(f"  Avg time: {tool_stats['avg_time']:.3f}s")

if __name__ == "__main__":
    example_usage()