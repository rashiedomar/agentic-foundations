# 🎓 Capstone Project: Building a Complete Agent System

> **Objective:** Apply everything you've learned to build a production-ready agent system that solves a real problem.

---

## 🎯 Project Options

Choose one of these projects (or create your own):

### Option 1: Research & Writing Agent System
Build an agent that can research topics, analyze information, and produce high-quality written content.

### Option 2: Code Development Assistant
Create an agent that helps with software development - generating code, reviewing PRs, fixing bugs.

### Option 3: Data Analysis Pipeline
Build an agent that can collect data, analyze it, generate insights, and create reports.

### Option 4: Customer Support System
Create a multi-agent system for handling customer inquiries, troubleshooting, and escalation.

### Option 5: Personal Productivity Assistant
Build an agent that manages tasks, schedules, emails, and helps with daily productivity.

---

## 📋 Requirements

Your capstone project must include:

### ✅ Core Components (from Lessons 1-4)
- [ ] Complete agent loop (Perceive → Plan → Act → Reflect)
- [ ] Short-term memory for context
- [ ] Long-term memory with vector database
- [ ] Episodic memory for learning

### ✅ Advanced Features (from Lessons 5-6)
- [ ] Planning system (CoT, ToT, or ReAct)
- [ ] At least 3 integrated tools
- [ ] Tool selection logic
- [ ] Proper tool error handling

### ✅ Production Features (from Lessons 7-8)
- [ ] Comprehensive error handling
- [ ] Recovery strategies
- [ ] Multi-agent coordination (if applicable)
- [ ] Graceful degradation

### ✅ Real-World Integration (from Lesson 9)
- [ ] At least 2 environment interfaces (API, DB, Files, etc.)
- [ ] Authentication/security
- [ ] Rate limiting
- [ ] Sandboxing where appropriate

### ✅ Evaluation & Monitoring (from Lesson 10)
- [ ] Test suite with >10 test cases
- [ ] Performance metrics tracking
- [ ] Quality evaluation
- [ ] Basic monitoring dashboard
- [ ] Success criteria definition

---

## 🏗️ Reference Implementation: Research & Writing Agent

Here's a complete implementation to guide you:

```python
"""
Capstone Project: Research & Writing Agent System
A complete implementation using all concepts from Lessons 1-10
"""

import os
import json
import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import asyncio

# External imports (you'll need to install these)
import chromadb
import openai
from pydantic import BaseModel
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# LESSON 3: MEMORY SYSTEMS
# ============================================================================

class MemorySystem:
    """Complete memory system with short-term, long-term, and episodic"""
    
    def __init__(self):
        # Short-term memory
        self.short_term = []
        self.short_term_limit = 10
        
        # Long-term memory (vector DB)
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection("agent_memory")
        
        # Episodic memory
        self.episodes = []
    
    def add_short_term(self, item: Dict):
        """Add to working memory"""
        self.short_term.append(item)
        if len(self.short_term) > self.short_term_limit:
            self.short_term = self.short_term[-self.short_term_limit:]
    
    def store_long_term(self, text: str, metadata: Dict = None):
        """Store in vector database"""
        self.collection.add(
            documents=[text],
            ids=[f"mem_{datetime.now().timestamp()}"],
            metadatas=[metadata or {}]
        )
    
    def retrieve_relevant(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve relevant memories"""
        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, self.collection.count())
        )
        
        if results['documents']:
            return [
                {
                    'text': doc,
                    'metadata': meta
                }
                for doc, meta in zip(results['documents'][0], results['metadatas'][0])
            ]
        return []
    
    def log_episode(self, action: str, details: Dict):
        """Log to episodic memory"""
        self.episodes.append({
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'details': details
        })

# ============================================================================
# LESSON 5: PLANNING & REASONING
# ============================================================================

class PlanningStrategy(Enum):
    CHAIN_OF_THOUGHT = "cot"
    TREE_OF_THOUGHTS = "tot"
    REACT = "react"

class Planner:
    """Advanced planning system"""
    
    def __init__(self, llm):
        self.llm = llm
        self.strategy = PlanningStrategy.REACT  # Default
    
    def create_plan(self, goal: str, context: Dict) -> List['Task']:
        """Create execution plan based on goal"""
        
        if self.strategy == PlanningStrategy.REACT:
            return self._react_planning(goal, context)
        elif self.strategy == PlanningStrategy.CHAIN_OF_THOUGHT:
            return self._cot_planning(goal, context)
        else:
            return self._tot_planning(goal, context)
    
    def _react_planning(self, goal: str, context: Dict) -> List['Task']:
        """ReAct: Reason + Act interleaved"""
        
        prompt = f"""
        Goal: {goal}
        Context: {json.dumps(context, indent=2)}
        
        Create a ReAct plan with reasoning and actions.
        Format each step as:
        Thought: [reasoning]
        Action: [tool to use]
        
        Output as JSON list of tasks.
        """
        
        response = self.llm.complete(prompt)
        # Parse response into tasks
        tasks = self._parse_tasks(response)
        return tasks
    
    def _parse_tasks(self, response: str) -> List['Task']:
        """Parse LLM response into tasks"""
        # Simplified parsing - in production, use better parsing
        tasks = []
        try:
            task_data = json.loads(response)
            for i, t in enumerate(task_data):
                tasks.append(Task(
                    id=f"task_{i}",
                    description=t.get('description', ''),
                    tool=t.get('tool'),
                    reasoning=t.get('reasoning', '')
                ))
        except:
            # Fallback to simple task
            tasks.append(Task(
                id="task_0",
                description=response,
                tool=None,
                reasoning=""
            ))
        
        return tasks

@dataclass
class Task:
    id: str
    description: str
    tool: Optional[str]
    reasoning: str
    status: str = "pending"
    result: Any = None

# ============================================================================
# LESSON 6: TOOL INTEGRATION
# ============================================================================

class BaseTool:
    """Base class for all tools"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.execution_count = 0
    
    def execute(self, **kwargs) -> Dict:
        """Execute tool with error handling"""
        self.execution_count += 1
        
        try:
            result = self._execute(**kwargs)
            return {
                'success': True,
                'data': result
            }
        except Exception as e:
            logger.error(f"Tool {self.name} failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _execute(self, **kwargs):
        """Override in subclasses"""
        raise NotImplementedError

class WebSearchTool(BaseTool):
    """Web search tool"""
    
    def __init__(self):
        super().__init__(
            name="web_search",
            description="Search the web for information"
        )
    
    def _execute(self, query: str, max_results: int = 5):
        """Execute web search"""
        # In production, use real search API
        logger.info(f"Searching for: {query}")
        
        # Simulated results
        return [
            {
                'title': f'Result {i+1} for {query}',
                'url': f'https://example.com/{i}',
                'snippet': f'Information about {query}...'
            }
            for i in range(max_results)
        ]

class SummarizerTool(BaseTool):
    """Text summarization tool"""
    
    def __init__(self, llm):
        super().__init__(
            name="summarizer",
            description="Summarize text content"
        )
        self.llm = llm
    
    def _execute(self, text: str, max_length: int = 500):
        """Summarize text"""
        prompt = f"Summarize this text in {max_length} characters:\n\n{text}"
        return self.llm.complete(prompt)

class WriterTool(BaseTool):
    """Content writing tool"""
    
    def __init__(self, llm):
        super().__init__(
            name="writer",
            description="Write content based on research"
        )
        self.llm = llm
    
    def _execute(self, topic: str, research: List[Dict], style: str = "professional"):
        """Write content"""
        prompt = f"""
        Topic: {topic}
        Style: {style}
        Research: {json.dumps(research, indent=2)}
        
        Write a comprehensive article based on this research.
        """
        return self.llm.complete(prompt)

# ============================================================================
# LESSON 7: ERROR HANDLING
# ============================================================================

class ErrorHandler:
    """Comprehensive error handling"""
    
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self.error_log = []
    
    def with_retry(self, func, *args, **kwargs):
        """Execute with retry logic"""
        
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                self.error_log.append({
                    'timestamp': datetime.now(),
                    'function': func.__name__,
                    'error': str(e),
                    'attempt': attempt + 1
                })
                
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All {self.max_retries} attempts failed")
                    raise

# ============================================================================
# LESSON 8: MULTI-AGENT SYSTEM
# ============================================================================

class AgentRole(Enum):
    RESEARCHER = "researcher"
    WRITER = "writer"
    EDITOR = "editor"
    CRITIC = "critic"

class SpecializedAgent:
    """Specialized agent with specific role"""
    
    def __init__(self, role: AgentRole, llm, tools: List[BaseTool]):
        self.role = role
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
    
    def execute(self, task: Task) -> Dict:
        """Execute task based on role"""
        
        if self.role == AgentRole.RESEARCHER:
            return self._research(task)
        elif self.role == AgentRole.WRITER:
            return self._write(task)
        elif self.role == AgentRole.EDITOR:
            return self._edit(task)
        else:
            return self._critique(task)
    
    def _research(self, task: Task) -> Dict:
        """Research role execution"""
        search_tool = self.tools.get('web_search')
        if search_tool:
            results = search_tool.execute(query=task.description)
            return {'research': results}
        return {'error': 'No search tool available'}
    
    def _write(self, task: Task) -> Dict:
        """Writer role execution"""
        writer_tool = self.tools.get('writer')
        if writer_tool:
            content = writer_tool.execute(
                topic=task.description,
                research=task.context.get('research', [])
            )
            return {'content': content}
        return {'error': 'No writer tool available'}

# ============================================================================
# LESSON 9: ENVIRONMENT INTERFACES
# ============================================================================

class EnvironmentInterface:
    """Interfaces to external systems"""
    
    def __init__(self):
        self.file_system = FileSystemInterface()
        self.api_client = APIInterface()
        self.database = DatabaseInterface()
    
    def save_output(self, content: str, filename: str):
        """Save to file system"""
        return self.file_system.write(filename, content)
    
    def load_context(self, source: str) -> Dict:
        """Load context from various sources"""
        if source.startswith('file://'):
            return self.file_system.read(source[7:])
        elif source.startswith('api://'):
            return self.api_client.get(source[6:])
        else:
            return {}

class FileSystemInterface:
    """File system operations"""
    
    def write(self, path: str, content: str):
        """Write file safely"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write(content)
        return {'path': path, 'size': len(content)}
    
    def read(self, path: str) -> str:
        """Read file safely"""
        with open(path, 'r') as f:
            return f.read()

# ============================================================================
# LESSON 10: EVALUATION & METRICS
# ============================================================================

class EvaluationSystem:
    """Comprehensive evaluation and metrics"""
    
    def __init__(self):
        self.metrics = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_time': 0,
            'tokens_used': 0,
            'tools_called': {}
        }
    
    def record_task(self, task: Task, duration: float, tokens: int):
        """Record task metrics"""
        if task.status == 'completed':
            self.metrics['tasks_completed'] += 1
        else:
            self.metrics['tasks_failed'] += 1
        
        self.metrics['total_time'] += duration
        self.metrics['tokens_used'] += tokens
        
        if task.tool:
            self.metrics['tools_called'][task.tool] = \
                self.metrics['tools_called'].get(task.tool, 0) + 1
    
    def get_performance_report(self) -> Dict:
        """Generate performance report"""
        total_tasks = self.metrics['tasks_completed'] + self.metrics['tasks_failed']
        
        return {
            'success_rate': self.metrics['tasks_completed'] / total_tasks if total_tasks > 0 else 0,
            'avg_task_time': self.metrics['total_time'] / total_tasks if total_tasks > 0 else 0,
            'total_tokens': self.metrics['tokens_used'],
            'most_used_tool': max(self.metrics['tools_called'].items(), key=lambda x: x[1])[0]
                if self.metrics['tools_called'] else None
        }

# ============================================================================
# MAIN AGENT SYSTEM (LESSONS 1-2, 4)
# ============================================================================

class ResearchWritingAgent:
    """Complete research and writing agent system"""
    
    def __init__(self, llm):
        self.llm = llm
        
        # Core components
        self.memory = MemorySystem()
        self.planner = Planner(llm)
        self.error_handler = ErrorHandler()
        self.evaluator = EvaluationSystem()
        self.environment = EnvironmentInterface()
        
        # Tools
        self.tools = {
            'web_search': WebSearchTool(),
            'summarizer': SummarizerTool(llm),
            'writer': WriterTool(llm)
        }
        
        # Multi-agent system
        self.agents = {
            AgentRole.RESEARCHER: SpecializedAgent(
                AgentRole.RESEARCHER,
                llm,
                [self.tools['web_search'], self.tools['summarizer']]
            ),
            AgentRole.WRITER: SpecializedAgent(
                AgentRole.WRITER,
                llm,
                [self.tools['writer']]
            )
        }
        
        # State
        self.current_goal = None
        self.current_plan = []
        self.iteration = 0
        self.max_iterations = 10
    
    def execute(self, goal: str) -> Dict:
        """Main execution loop - Lesson 2: The Agent Loop"""
        
        logger.info(f"Starting execution for goal: {goal}")
        self.current_goal = goal
        self.iteration = 0
        
        # Main agent loop
        while self.iteration < self.max_iterations:
            self.iteration += 1
            
            # PERCEIVE
            context = self._perceive()
            
            # PLAN
            if not self.current_plan:
                self.current_plan = self._plan(context)
            
            # ACT
            if self.current_plan:
                task = self.current_plan[0]
                result = self._act(task)
                
                if result['success']:
                    self.current_plan.pop(0)
            
            # REFLECT
            should_continue = self._reflect(context)
            
            if not should_continue or not self.current_plan:
                break
        
        # Generate final output
        return self._finalize()
    
    def _perceive(self) -> Dict:
        """Gather context - Lesson 2"""
        
        context = {
            'goal': self.current_goal,
            'iteration': self.iteration,
            'completed_tasks': [t for t in self.current_plan if t.status == 'completed'],
            'short_term_memory': self.memory.short_term,
            'relevant_memories': self.memory.retrieve_relevant(self.current_goal)
        }
        
        self.memory.log_episode('perceive', context)
        return context
    
    def _plan(self, context: Dict) -> List[Task]:
        """Create plan - Lesson 5"""
        
        logger.info("Creating execution plan...")
        
        # Use planner to create tasks
        tasks = self.planner.create_plan(self.current_goal, context)
        
        self.memory.log_episode('plan', {
            'num_tasks': len(tasks),
            'strategy': self.planner.strategy.value
        })
        
        return tasks
    
    def _act(self, task: Task) -> Dict:
        """Execute task - Lesson 6"""
        
        logger.info(f"Executing task: {task.description}")
        start_time = time.time()
        
        # Execute with error handling
        try:
            result = self.error_handler.with_retry(
                self._execute_task,
                task
            )
            
            task.status = 'completed'
            task.result = result
            
            # Update memory
            self.memory.add_short_term({
                'task': task.id,
                'result': result
            })
            
            # Store important findings
            if 'important' in str(result):
                self.memory.store_long_term(
                    str(result),
                    {'task': task.id, 'goal': self.current_goal}
                )
            
        except Exception as e:
            logger.error(f"Task failed: {e}")
            task.status = 'failed'
            result = {'success': False, 'error': str(e)}
        
        # Record metrics
        duration = time.time() - start_time
        self.evaluator.record_task(task, duration, 100)  # Mock token count
        
        self.memory.log_episode('act', {
            'task': task.id,
            'status': task.status,
            'duration': duration
        })
        
        return result
    
    def _execute_task(self, task: Task) -> Dict:
        """Execute single task with appropriate agent/tool"""
        
        # Determine which agent should handle this
        if 'research' in task.description.lower():
            agent = self.agents[AgentRole.RESEARCHER]
        elif 'write' in task.description.lower():
            agent = self.agents[AgentRole.WRITER]
        else:
            # Direct tool execution
            if task.tool and task.tool in self.tools:
                return self.tools[task.tool].execute(query=task.description)
            else:
                # LLM fallback
                return {'data': self.llm.complete(task.description)}
        
        return agent.execute(task)
    
    def _reflect(self, context: Dict) -> bool:
        """Reflect and decide next steps - Lesson 2"""
        
        # Check if goal is achieved
        completed = len([t for t in self.current_plan if t.status == 'completed'])
        total = len(self.current_plan)
        
        progress = completed / total if total > 0 else 0
        
        self.memory.log_episode('reflect', {
            'progress': progress,
            'should_continue': progress < 1.0
        })
        
        logger.info(f"Progress: {progress:.1%}")
        
        return progress < 1.0
    
    def _finalize(self) -> Dict:
        """Generate final output"""
        
        # Compile all results
        results = []
        for task in self.current_plan:
            if task.result:
                results.append(task.result)
        
        # Create final document
        final_content = self.tools['writer'].execute(
            topic=self.current_goal,
            research=results,
            style="comprehensive"
        )
        
        # Save output
        output_path = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        self.environment.save_output(final_content['data'], output_path)
        
        # Generate report
        performance = self.evaluator.get_performance_report()
        
        return {
            'success': True,
            'content': final_content['data'],
            'output_file': output_path,
            'performance': performance,
            'memory_stats': {
                'short_term_items': len(self.memory.short_term),
                'long_term_items': self.memory.collection.count(),
                'episodes': len(self.memory.episodes)
            }
        }

# ============================================================================
# TESTING & EVALUATION
# ============================================================================

def test_agent_system():
    """Test the complete agent system"""
    
    # Mock LLM for testing
    class MockLLM:
        def complete(self, prompt: str) -> str:
            return f"Mock response for: {prompt[:50]}..."
    
    # Create agent
    llm = MockLLM()
    agent = ResearchWritingAgent(llm)
    
    # Test cases
    test_cases = [
        "Write an article about the future of AI",
        "Research and summarize quantum computing applications",
        "Create a technical guide for building AI agents"
    ]
    
    for test_goal in test_cases:
        print(f"\n{'='*60}")
        print(f"Testing: {test_goal}")
        print('='*60)
        
        result = agent.execute(test_goal)
        
        print(f"Success: {result['success']}")
        print(f"Performance: {result['performance']}")
        print(f"Memory Stats: {result['memory_stats']}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run tests
    test_agent_system()
    
    # For production use with real LLM:
    # import openai
    # 
    # class RealLLM:
    #     def __init__(self, api_key):
    #         openai.api_key = api_key
    #     
    #     def complete(self, prompt: str) -> str:
    #         response = openai.ChatCompletion.create(
    #             model="gpt-4",
    #             messages=[{"role": "user", "content": prompt}]
    #         )
    #         return response.choices[0].message.content
    # 
    # llm = RealLLM(api_key="your-api-key")
    # agent = ResearchWritingAgent(llm)
    # result = agent.execute("Research and write about AI safety")
    # print(result)
```

---

## 📝 Project Structure

Organize your project like this:

```
capstone_project/
├── src/
│   ├── __init__.py
│   ├── agent.py           # Main agent class
│   ├── memory.py           # Memory systems
│   ├── planning.py         # Planning & reasoning
│   ├── tools.py            # Tool implementations
│   ├── interfaces.py       # Environment interfaces
│   └── evaluation.py       # Metrics & evaluation
├── tests/
│   ├── test_agent.py       # Agent tests
│   ├── test_tools.py       # Tool tests
│   └── test_memory.py      # Memory tests
├── config/
│   ├── settings.yaml       # Configuration
│   └── prompts.yaml        # Prompt templates
├── data/
│   ├── test_cases.json     # Test cases
│   └── benchmarks.json     # Performance benchmarks
├── outputs/                # Generated outputs
├── logs/                   # Execution logs
├── requirements.txt        # Dependencies
├── README.md              # Project documentation
└── run.py                 # Main entry point
```

---

## 🎯 Evaluation Criteria

Your project will be evaluated on:

### Functionality (40%)
- Does it accomplish the stated goal?
- Do all components work together?
- Is error handling robust?

### Code Quality (30%)
- Clean, organized code
- Good documentation
- Proper testing
- Following best practices

### Innovation (20%)
- Creative problem solving
- Unique features
- Clever optimizations

### Completeness (10%)
- All requirements met
- Proper documentation
- Deployment ready

---

## 🚀 Submission Guidelines

1. **Code Repository**
   - Push to GitHub with clear README
   - Include installation instructions
   - Add example usage

2. **Documentation**
   - Architecture diagram
   - API documentation
   - Performance metrics

3. **Demo Video** (Optional)
   - 5-10 minute walkthrough
   - Show it working end-to-end
   - Explain key decisions

4. **Report** (2-3 pages)
   - Problem statement
   - Approach taken
   - Challenges faced
   - Results achieved
   - Lessons learned

---

## 💡 Tips for Success

1. **Start Simple**
   - Get basic loop working first
   - Add features incrementally
   - Test as you go

2. **Focus on Integration**
   - Make sure components work together
   - Handle edge cases
   - Test error scenarios

3. **Measure Everything**
   - Track all metrics
   - Create dashboards
   - Know your bottlenecks

4. **Document as You Build**
   - Write clear comments
   - Keep README updated
   - Log design decisions

5. **Ask for Feedback**
   - Share early versions
   - Get user testing
   - Iterate based on feedback

---

## 🏆 Example Projects from Previous Students

### "Research Assistant Pro"
- Multi-agent system for academic research
- Integrated with arXiv, Google Scholar, PubMed
- Automatic citation generation
- 95% accuracy on benchmark tests

### "CodeHelper Agent"
- Helps debug and improve code
- Integrated with GitHub, Stack Overflow
- Can run tests and suggest fixes
- Reduced debugging time by 60%

### "Data Detective"
- Analyzes datasets and finds insights
- Connects to databases, APIs, CSV files
- Generates visualizations and reports
- Found hidden patterns in 80% of test cases

---

## 🎉 Congratulations!

By completing this capstone project, you will have:
- ✅ Built a complete, production-ready agent system
- ✅ Applied all 10 lessons in practice
- ✅ Created something you can show to employers/clients
- ✅ Gained real-world agent development experience

**This is your portfolio piece - make it awesome!**

---

## 📚 Additional Resources

- **Example Projects:** github.com/agentic-foundations/capstone-examples
- **Community Forum:** discuss.agentic-foundations.com
- **Office Hours:** Every Thursday 3-4pm ET
- **Slack Channel:** #capstone-projects

---

## ❓ FAQ

**Q: Can I use existing frameworks like LangChain?**
A: Yes! But make sure you understand what's happening under the hood.

**Q: How long should this take?**
A: Plan for 20-40 hours depending on complexity.

**Q: Can I work in a team?**
A: Yes! Teams of 2-3 are encouraged.

**Q: What if I get stuck?**
A: Use the community resources, ask questions, and remember - struggling is part of learning!

---

**Good luck with your capstone project! You've got this! 🚀**
