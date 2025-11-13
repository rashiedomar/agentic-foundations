# Lesson 6: Tool Integration

> **What you'll learn:** How to give agents real capabilities by designing, building, and integrating tools that let them interact with the world.

---

## 🎯 Learning Objectives

By the end of this lesson, you'll understand:
- What tools are and why agents need them
- How to design effective tools for agents
- Different types of tools and their use cases
- Safe tool execution and error handling
- Tool composition and chaining
- Integrating tools with planning systems

---

## 🤔 The Problem: Planning Without Action

Remember your planning agent from Lesson 5? It could:
- ✅ Break down complex goals
- ✅ Create multi-step plans
- ✅ Reason through problems

But it **couldn't actually DO anything**! 

```
Agent's Plan:
1. Search the web for AI startups ❌ (can't search)
2. Analyze their business models ❌ (can't analyze)
3. Create a report ❌ (can't create files)

Result: Just text describing what SHOULD happen
```

**Tools are what transform plans into actions.**

---

## 🛠️ What Are Tools?

**Tools = Functions that agents can call to interact with the world**

Think of tools as the agent's "hands" - they let the agent:
- 🔍 Search the web
- 📁 Read/write files
- 💻 Execute code
- 📊 Query databases
- 📧 Send emails
- 🌐 Make API calls
- And much more!

### The Tool Abstraction

```python
class Tool:
    """
    A tool is a function the agent can use
    """
    name: str              # What it's called
    description: str       # What it does (for LLM to understand)
    function: Callable     # The actual function to execute
    parameters: dict       # What inputs it needs
    
    def execute(self, **kwargs):
        """Run the tool with given parameters"""
        return self.function(**kwargs)
```

---

## 🎨 Tool Design Principles

### 1. **Single Responsibility**

Each tool should do ONE thing well.

```python
# ❌ BAD: Tool that does too much
class SuperTool:
    def execute(self, action, url, query, code):
        if action == "search":
            return search_web(query)
        elif action == "fetch":
            return fetch_url(url)
        elif action == "code":
            return run_code(code)
        # Too many responsibilities!

# ✅ GOOD: Focused tools
class WebSearchTool:
    """Only searches the web"""
    def execute(self, query: str):
        return search_web(query)

class WebFetchTool:
    """Only fetches URLs"""
    def execute(self, url: str):
        return fetch_url(url)

class CodeExecutorTool:
    """Only runs code"""
    def execute(self, code: str, language: str):
        return run_code(code, language)
```

### 2. **Clear, Descriptive Names**

The LLM needs to understand what the tool does from its name and description.

```python
# ❌ BAD: Vague names
class Tool1:  # What does this do?
    description = "Does stuff"

# ✅ GOOD: Clear names
class WebSearchTool:
    description = """
    Searches the web for information using a search engine.
    Use this when you need current information or want to find websites.
    
    Parameters:
    - query (str): The search query
    
    Returns: List of search results with titles, URLs, and snippets
    """
```

### 3. **Robust Error Handling**

Tools operate in the real world - things WILL go wrong.

```python
class WebSearchTool:
    def execute(self, query: str) -> dict:
        try:
            results = search_api.search(query)
            return {
                "success": True,
                "results": results
            }
        except RateLimitError as e:
            return {
                "success": False,
                "error": "Rate limit exceeded. Try again in 60 seconds.",
                "retry_after": 60
            }
        except NetworkError as e:
            return {
                "success": False,
                "error": "Network error. Check internet connection.",
                "retryable": True
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "retryable": False
            }
```

### 4. **Type Safety**

Use type hints to make parameters clear.

```python
from typing import List, Dict, Optional
from pydantic import BaseModel, Field

class SearchParams(BaseModel):
    """Validated search parameters"""
    query: str = Field(..., description="The search query", min_length=1)
    max_results: int = Field(10, description="Max results to return", ge=1, le=100)
    language: Optional[str] = Field("en", description="Language code")

class WebSearchTool:
    def execute(self, params: SearchParams) -> Dict:
        # Parameters are automatically validated!
        return search_api.search(
            query=params.query,
            limit=params.max_results,
            lang=params.language
        )
```

### 5. **Idempotency (When Possible)**

Running the same tool with same inputs should give consistent results.

```python
# ✅ GOOD: Idempotent
class WebSearchTool:
    def execute(self, query: str):
        return search(query)  # Same query = same results

# ⚠️ NOT Idempotent (but that's OK for some tools)
class SendEmailTool:
    def execute(self, to: str, subject: str):
        send_email(to, subject)  # Sends email every time!
```

---

## 📦 Types of Tools

### 1. **Information Retrieval Tools**

Tools that GET information from external sources.

```python
class WebSearchTool:
    """Search the internet"""
    def execute(self, query: str) -> List[dict]:
        return search_engine.search(query)

class WebFetchTool:
    """Fetch webpage content"""
    def execute(self, url: str) -> str:
        return requests.get(url).text

class DatabaseQueryTool:
    """Query a database"""
    def execute(self, sql: str) -> List[dict]:
        return database.execute(sql)

class WeatherTool:
    """Get current weather"""
    def execute(self, location: str) -> dict:
        return weather_api.get_weather(location)
```

### 2. **Action/Mutation Tools**

Tools that CHANGE state or perform actions.

```python
class FileWriteTool:
    """Write content to a file"""
    def execute(self, path: str, content: str) -> dict:
        with open(path, 'w') as f:
            f.write(content)
        return {"success": True, "path": path}

class SendEmailTool:
    """Send an email"""
    def execute(self, to: str, subject: str, body: str):
        email_client.send(to=to, subject=subject, body=body)
        return {"success": True}

class CreateIssueTool:
    """Create a GitHub issue"""
    def execute(self, title: str, body: str):
        github.create_issue(title=title, body=body)
```

### 3. **Computation Tools**

Tools that process or transform data.

```python
class CodeExecutorTool:
    """Execute Python code safely"""
    def execute(self, code: str) -> dict:
        # Run in sandbox
        result = sandbox.run_python(code, timeout=10)
        return result

class DataAnalysisTool:
    """Analyze CSV data"""
    def execute(self, csv_path: str, analysis: str):
        df = pd.read_csv(csv_path)
        # Perform analysis based on instruction
        return results

class ImageGeneratorTool:
    """Generate images from text"""
    def execute(self, prompt: str):
        return dalle.generate(prompt)
```

### 4. **Compound/Meta Tools**

Tools that use other tools.

```python
class ResearchTool:
    """Deep research using multiple tools"""
    def __init__(self, search_tool, fetch_tool, summarize_tool):
        self.search = search_tool
        self.fetch = fetch_tool
        self.summarize = summarize_tool
    
    def execute(self, topic: str):
        # 1. Search for sources
        results = self.search.execute(topic)
        
        # 2. Fetch top sources
        contents = [
            self.fetch.execute(r['url']) 
            for r in results[:3]
        ]
        
        # 3. Summarize
        summary = self.summarize.execute(contents)
        
        return summary
```

---

## 🔐 Safe Tool Execution

**Tools can be dangerous!** They interact with the real world.

### Safety Principles

#### 1. **Sandboxing**

Run potentially dangerous code in isolated environments.

```python
class SafeCodeExecutor:
    def execute(self, code: str):
        # Create isolated sandbox
        sandbox = Docker.create_container(
            image="python:3.11-slim",
            network="none",  # No internet access
            memory_limit="128m",
            cpu_limit=0.5,
            timeout=10
        )
        
        try:
            result = sandbox.run(code)
            return result
        finally:
            sandbox.cleanup()
```

#### 2. **Input Validation**

Never trust tool inputs!

```python
class FileReadTool:
    ALLOWED_EXTENSIONS = {'.txt', '.md', '.json', '.csv'}
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    
    def execute(self, path: str) -> str:
        # Validate extension
        ext = Path(path).suffix
        if ext not in self.ALLOWED_EXTENSIONS:
            raise ValueError(f"File type {ext} not allowed")
        
        # Prevent path traversal
        abs_path = Path(path).resolve()
        if not abs_path.is_relative_to(SAFE_DIR):
            raise ValueError("Path outside allowed directory")
        
        # Check file size
        if abs_path.stat().st_size > self.MAX_FILE_SIZE:
            raise ValueError("File too large")
        
        # Now safe to read
        return abs_path.read_text()
```

#### 3. **Rate Limiting**

Prevent abuse and respect API limits.

```python
from functools import wraps
import time

class RateLimiter:
    def __init__(self, max_calls: int, period: int):
        self.max_calls = max_calls
        self.period = period  # seconds
        self.calls = []
    
    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            
            # Remove old calls
            self.calls = [c for c in self.calls if now - c < self.period]
            
            # Check limit
            if len(self.calls) >= self.max_calls:
                wait = self.period - (now - self.calls[0])
                raise RateLimitError(f"Rate limit exceeded. Wait {wait:.1f}s")
            
            # Record this call
            self.calls.append(now)
            
            return func(*args, **kwargs)
        return wrapper

class WebSearchTool:
    @RateLimiter(max_calls=10, period=60)  # 10 calls per minute
    def execute(self, query: str):
        return search_api.search(query)
```

#### 4. **Confirmation for Destructive Actions**

Some tools should require explicit confirmation.

```python
class DeleteFileTool:
    def execute(self, path: str, confirm: bool = False):
        if not confirm:
            return {
                "success": False,
                "error": "Destructive action requires confirmation",
                "confirmation_required": True,
                "action": f"Delete file: {path}"
            }
        
        # Actually delete
        os.remove(path)
        return {"success": True, "deleted": path}
```

---

## 🔗 Tool Composition & Chaining

**Powerful agents combine simple tools to solve complex problems.**

### Sequential Chaining

```python
# Research workflow: search → fetch → summarize
def research_topic(topic: str):
    # Tool 1: Search
    search_results = web_search_tool.execute(topic)
    
    # Tool 2: Fetch top results
    pages = []
    for result in search_results[:5]:
        content = web_fetch_tool.execute(result['url'])
        pages.append(content)
    
    # Tool 3: Summarize
    summary = summarize_tool.execute(pages)
    
    return summary
```

### Parallel Execution

```python
import asyncio

async def research_multiple_topics(topics: List[str]):
    """Search multiple topics in parallel"""
    tasks = [
        web_search_tool.execute_async(topic)
        for topic in topics
    ]
    
    results = await asyncio.gather(*tasks)
    return results
```

### Conditional Chaining

```python
def smart_research(topic: str):
    # Try cache first
    cached = cache_tool.execute(topic)
    if cached:
        return cached
    
    # Not cached - do research
    results = web_search_tool.execute(topic)
    
    # If not enough results, try alternative search
    if len(results) < 3:
        results = alternative_search_tool.execute(topic)
    
    # Fetch and summarize
    summary = process_results(results)
    
    # Cache for next time
    cache_tool.execute(topic, summary)
    
    return summary
```

---

## 🔌 Integrating Tools with Planning

**This is where it all comes together!**

### The Integration Pattern

```python
class AgentWithTools:
    def __init__(self, llm, planner, tools):
        self.llm = llm
        self.planner = planner
        self.tools = {tool.name: tool for tool in tools}
    
    def execute_goal(self, goal: str):
        # 1. Create plan
        plan = self.planner.create_plan(
            goal=goal,
            available_tools=list(self.tools.keys())
        )
        
        # 2. Execute plan with tools
        for task in plan.tasks:
            if task.tool:
                # Task requires a tool
                tool = self.tools[task.tool]
                
                # Get parameters from LLM
                params = self.extract_parameters(task, tool)
                
                # Execute tool
                result = tool.execute(**params)
                
                # Update task with result
                task.result = result
            else:
                # Task uses LLM reasoning only
                result = self.llm.complete(task.description)
                task.result = result
        
        return plan
```

### Tool Selection by LLM

```python
def select_tool(self, task: str) -> Tool:
    """Let LLM choose the right tool"""
    
    tool_descriptions = "\n".join([
        f"- {name}: {tool.description}"
        for name, tool in self.tools.items()
    ])
    
    prompt = f"""
    Task: {task}
    
    Available tools:
    {tool_descriptions}
    
    Which tool is best for this task?
    Output JSON: {{"tool": "tool_name", "reasoning": "why this tool"}}
    """
    
    response = self.llm.complete(prompt)
    selection = json.loads(response)
    
    return self.tools[selection['tool']]
```

### Parameter Extraction

```python
def extract_parameters(self, task: str, tool: Tool) -> dict:
    """Extract tool parameters from task description"""
    
    prompt = f"""
    Task: {task}
    Tool: {tool.name}
    
    Tool parameters:
    {json.dumps(tool.parameters, indent=2)}
    
    Extract the parameter values from the task.
    Output JSON with parameter values.
    """
    
    response = self.llm.complete(prompt)
    return json.loads(response)
```

---

## 💻 Implementation: Complete Tool System

Here's a production-ready tool system:

```python
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class ToolResult(BaseModel):
    """Standardized tool result"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class BaseTool(ABC):
    """Base class for all tools"""
    
    def __init__(self):
        self.name = self.__class__.__name__
        self.execution_count = 0
        self.last_error = None
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for LLM"""
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> Dict:
        """Tool parameter schema"""
        pass
    
    @abstractmethod
    def _execute(self, **kwargs) -> Any:
        """Internal execution logic"""
        pass
    
    def execute(self, **kwargs) -> ToolResult:
        """Execute tool with error handling"""
        self.execution_count += 1
        
        try:
            logger.info(f"Executing {self.name} with {kwargs}")
            
            # Validate parameters
            self._validate_params(kwargs)
            
            # Execute
            result = self._execute(**kwargs)
            
            logger.info(f"{self.name} completed successfully")
            
            return ToolResult(
                success=True,
                data=result,
                metadata={
                    "tool": self.name,
                    "execution_count": self.execution_count
                }
            )
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"{self.name} failed: {e}")
            
            return ToolResult(
                success=False,
                error=str(e),
                metadata={
                    "tool": self.name,
                    "execution_count": self.execution_count
                }
            )
    
    def _validate_params(self, params: dict):
        """Validate parameters against schema"""
        required = [
            k for k, v in self.parameters.items()
            if v.get('required', False)
        ]
        
        missing = [p for p in required if p not in params]
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")

class ToolRegistry:
    """Manages available tools"""
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
    
    def register(self, tool: BaseTool):
        """Register a new tool"""
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def get(self, name: str) -> Optional[BaseTool]:
        """Get tool by name"""
        return self.tools.get(name)
    
    def list_tools(self) -> List[str]:
        """List all tool names"""
        return list(self.tools.keys())
    
    def get_descriptions(self) -> str:
        """Get formatted tool descriptions for LLM"""
        descriptions = []
        for tool in self.tools.values():
            desc = f"""
Tool: {tool.name}
Description: {tool.description}
Parameters: {json.dumps(tool.parameters, indent=2)}
"""
            descriptions.append(desc)
        return "\n---\n".join(descriptions)
```

---

## 🎯 Example: Building Real Tools

### Web Search Tool

```python
class WebSearchTool(BaseTool):
    @property
    def description(self) -> str:
        return "Search the web for information using a search engine"
    
    @property
    def parameters(self) -> Dict:
        return {
            "query": {
                "type": "string",
                "description": "The search query",
                "required": True
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results",
                "default": 10
            }
        }
    
    def _execute(self, query: str, max_results: int = 10) -> List[dict]:
        # Use actual search API (Brave, Serper, etc.)
        results = search_api.search(query, limit=max_results)
        
        return [
            {
                "title": r.title,
                "url": r.url,
                "snippet": r.snippet
            }
            for r in results
        ]
```

### File Operations Tool

```python
class FileOperationsTool(BaseTool):
    @property
    def description(self) -> str:
        return "Read, write, or list files in the workspace"
    
    @property
    def parameters(self) -> Dict:
        return {
            "operation": {
                "type": "string",
                "enum": ["read", "write", "list"],
                "required": True
            },
            "path": {
                "type": "string",
                "required": True
            },
            "content": {
                "type": "string",
                "required": False
            }
        }
    
    def _execute(self, operation: str, path: str, content: str = None):
        if operation == "read":
            with open(path, 'r') as f:
                return f.read()
        
        elif operation == "write":
            with open(path, 'w') as f:
                f.write(content)
            return f"Wrote {len(content)} characters to {path}"
        
        elif operation == "list":
            return os.listdir(path)
```

### Code Executor Tool

```python
class CodeExecutorTool(BaseTool):
    @property
    def description(self) -> str:
        return "Execute Python code in a safe sandbox"
    
    @property
    def parameters(self) -> Dict:
        return {
            "code": {
                "type": "string",
                "description": "Python code to execute",
                "required": True
            }
        }
    
    def _execute(self, code: str) -> dict:
        # Execute in sandbox
        stdout, stderr = execute_in_sandbox(code, timeout=10)
        
        return {
            "stdout": stdout,
            "stderr": stderr,
            "success": stderr == ""
        }
```

---

## 🏆 Best Practices

### 1. **Make Tools Discoverable**

```python
# ✅ GOOD: Clear, searchable tool catalog
tools = [
    WebSearchTool(),          # "search" or "web" or "google"
    FileReadTool(),           # "read" or "file"
    CalculatorTool(),         # "calculate" or "math"
    WeatherTool(),            # "weather"
]
```

### 2. **Provide Examples in Descriptions**

```python
class WebSearchTool(BaseTool):
    @property
    def description(self) -> str:
        return """
        Search the web for information.
        
        Examples:
        - query: "latest AI research papers"
        - query: "weather in San Francisco"
        - query: "Python asyncio tutorial"
        
        Returns: List of search results with titles, URLs, snippets
        """
```

### 3. **Return Structured Data**

```python
# ❌ BAD: Unstructured string
def search(query):
    return "Found 3 results: ..."

# ✅ GOOD: Structured data
def search(query):
    return [
        {
            "title": "...",
            "url": "...",
            "snippet": "...",
            "relevance": 0.95
        }
    ]
```

### 4. **Log Everything**

```python
class WebSearchTool(BaseTool):
    def _execute(self, query: str):
        logger.info(f"Searching for: {query}")
        
        start_time = time.time()
        results = search_api.search(query)
        elapsed = time.time() - start_time
        
        logger.info(f"Found {len(results)} results in {elapsed:.2f}s")
        
        return results
```

---

## 🎓 Advanced Topics

### Dynamic Tool Generation

```python
def create_api_tool(api_spec: dict) -> BaseTool:
    """Create a tool from OpenAPI spec"""
    
    class DynamicAPITool(BaseTool):
        @property
        def description(self):
            return api_spec['description']
        
        @property
        def parameters(self):
            return api_spec['parameters']
        
        def _execute(self, **kwargs):
            return call_api(api_spec['endpoint'], kwargs)
    
    return DynamicAPITool()
```

### Tool Learning

```python
class AdaptiveTool(BaseTool):
    """Tool that learns from usage"""
    
    def __init__(self):
        super().__init__()
        self.success_patterns = []
        self.failure_patterns = []
    
    def execute(self, **kwargs):
        result = super().execute(**kwargs)
        
        # Learn from result
        if result.success:
            self.success_patterns.append(kwargs)
        else:
            self.failure_patterns.append(kwargs)
        
        return result
    
    def get_suggestions(self, task: str) -> dict:
        """Suggest parameters based on past successes"""
        # Analyze success patterns and suggest similar parameters
        pass
```

---

## 🚀 What's Next?

Now you can:
- ✅ Design effective tools
- ✅ Implement safe tool execution
- ✅ Compose tools into workflows
- ✅ Integrate tools with planning

**Next Steps:**
1. Build your tool library
2. Connect tools to your planning agent (from Lesson 5)
3. Create a complete agent that plans AND acts!

**In the next section**, we'll build practical tools and see them work with your planning agent.

---

## 🤔 Reflection Questions

1. **What tools would be most useful for YOUR use cases?**
2. **How would you make a dangerous tool (like DeleteFile) safe?**
3. **When should tools be composed vs kept separate?**
4. **How do you balance tool complexity vs simplicity?**
5. **What happens when a tool fails mid-execution?**

---

## 📝 Your Notes

*Space for your thoughts and experiments:*

**Tools I want to build:**
- 

**Safety concerns:**
- 

**Integration ideas:**
- 

**Questions:**
- 

---

**Lesson Status:** 📝 Ready for Implementation  
**Difficulty:** ⭐⭐⭐ (Intermediate)  
**Time to Complete:** 60-90 minutes  
**Prerequisites:** Lessons 1-5

**Next Lesson:** [Error Handling & Recovery](07-error-handling.md)
