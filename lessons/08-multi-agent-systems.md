# Lesson 8: Multi-Agent Systems

> **What you'll learn:** How to orchestrate multiple specialized agents working together, coordinate their actions, and build systems where agents collaborate to solve complex problems beyond what any single agent could achieve.

---

## 🎯 Learning Objectives

By the end of this lesson, you'll understand:
- When and why to use multiple agents vs a single agent
- Different multi-agent architectures (hierarchical, flat, hybrid)
- Communication patterns and protocols between agents
- Role specialization and delegation strategies
- Conflict resolution and consensus mechanisms
- Building and coordinating agent teams
- Real-world multi-agent frameworks

---

## 🤔 The Single Agent Problem

Your monolithic agent is getting overwhelmed:

```python
class GodAgent:
    """One agent trying to do EVERYTHING"""
    
    def execute(self, task: str):
        if "research" in task:
            self.do_research()
        if "code" in task:
            self.write_code()
        if "design" in task:
            self.create_designs()
        if "analyze" in task:
            self.analyze_data()
        if "write" in task:
            self.write_content()
        # ... 50 more conditions ...
```

**Problems:**
- ❌ Overwhelmed with context (token limits)
- ❌ Jack of all trades, master of none
- ❌ Hard to debug when things go wrong
- ❌ Can't parallelize work
- ❌ Difficult to maintain and improve
- ❌ Single point of failure

**Solution: Specialized agents working together! 🤝**

---

## 🌟 Why Multi-Agent Systems?

### The Power of Specialization

Just like a company has different departments, agents can specialize:

```
┌─────────────────────────────────────────────────┐
│           MULTI-AGENT SYSTEM                    │
├─────────────────────────────────────────────────┤
│                                                 │
│  👔 Manager Agent                               │
│     └─ Plans, delegates, coordinates            │
│                                                 │
│  🔍 Research Agent                              │
│     └─ Finds information, validates sources     │
│                                                 │
│  💻 Coding Agent                                │
│     └─ Writes, tests, debugs code               │
│                                                 │
│  ✍️ Writing Agent                               │
│     └─ Creates documentation, reports           │
│                                                 │
│  🎨 Design Agent                                │
│     └─ Creates visuals, UI/UX                   │
│                                                 │
│  ✅ QA Agent                                     │
│     └─ Reviews, validates, critiques            │
│                                                 │
└─────────────────────────────────────────────────┘
```

### Benefits

**1. Expertise**
Each agent masters a specific domain

**2. Parallelization**
Multiple agents work simultaneously

**3. Modularity**
Swap out agents without breaking the system

**4. Scalability**
Add new specialist agents as needed

**5. Fault Tolerance**
One agent failing doesn't crash everything

**6. Better Results**
Collaboration often beats individual effort

---

## 🏗️ Multi-Agent Architectures

### 1. Hierarchical (Manager-Worker)

One agent coordinates, others execute.

```
        ┌─────────────┐
        │   MANAGER   │  ← Makes decisions
        │    Agent    │     Delegates tasks
        └──────┬──────┘
               │
       ┬───────┼───────┬
       │       │       │
   ┌───▼───┐ ┌─▼─────┐ ┌──▼────┐
   │Worker1│ │Worker2│ │Worker3│  ← Execute tasks
   │Research│ │Coding │ │Writing│
   └───────┘ └───────┘ └───────┘
```

**When to use:**
- Clear task decomposition
- One agent has the "big picture"
- Workers are specialized tools

**Example:**

```python
class ManagerAgent:
    """Orchestrates worker agents"""
    
    def __init__(self, workers: Dict[str, Agent]):
        self.workers = workers
    
    def execute(self, goal: str):
        # 1. Break down the goal
        plan = self.create_plan(goal)
        
        # 2. Assign tasks to workers
        results = []
        for task in plan.tasks:
            worker = self.select_worker(task)
            result = worker.execute(task)
            results.append(result)
        
        # 3. Synthesize results
        final_output = self.synthesize(results)
        return final_output
    
    def select_worker(self, task: Task) -> Agent:
        """Choose the best worker for the task"""
        task_type = self.classify_task(task)
        return self.workers[task_type]

# Usage
manager = ManagerAgent(workers={
    "research": ResearchAgent(),
    "coding": CodingAgent(),
    "writing": WritingAgent()
})

result = manager.execute("Build a web scraper and document it")
```

### 2. Flat/Peer-to-Peer

Agents are equals, coordinate through communication.

```
   ┌─────────┐      ┌─────────┐
   │ Agent 1 │◄────►│ Agent 2 │
   │Research │      │Analysis │
   └────┬────┘      └────┬────┘
        │                │
        │    ┌───────┐   │
        └───►│ Agent │◄──┘
             │  3    │
             │Writing│
             └───────┘
```

**When to use:**
- No clear hierarchy
- Agents need to collaborate dynamically
- Consensus-based decision making

**Example:**

```python
class CollaborativeAgentSystem:
    """Agents work as equals"""
    
    def __init__(self, agents: List[Agent]):
        self.agents = agents
        self.message_board = MessageBoard()
    
    def solve_problem(self, problem: str):
        # 1. All agents see the problem
        self.message_board.post({
            "type": "problem",
            "content": problem,
            "from": "system"
        })
        
        # 2. Agents volunteer or vote on approach
        proposals = []
        for agent in self.agents:
            proposal = agent.propose_solution(problem)
            proposals.append(proposal)
        
        # 3. Consensus on best approach
        chosen_plan = self.vote_on_proposals(proposals)
        
        # 4. Collaborative execution
        results = []
        for task in chosen_plan.tasks:
            # Agents claim tasks they're good at
            assigned_agent = self.claim_task(task)
            result = assigned_agent.execute(task)
            
            # Share result with others
            self.message_board.post({
                "type": "result",
                "task": task,
                "result": result,
                "from": assigned_agent.name
            })
            results.append(result)
        
        return self.combine_results(results)
```

### 3. Hybrid (Manager + Peer Groups)

Manager coordinates, but workers can also collaborate.

```
        ┌─────────────┐
        │   MANAGER   │
        └──────┬──────┘
               │
       ┬───────┼───────┬
       │       │       │
   ┌───▼───┐ ┌─▼─────┐ ┌──▼────┐
   │Team 1 │ │Team 2 │ │Team 3 │
   └───┬───┘ └───┬───┘ └───┬───┘
       │         │         │
   ┌───▼───┬─────▼──┬──────▼───┐
   │Agent A│ Agent B│  Agent C │
   │◄─────►│◄──────►│◄────────►│
   └───────┴────────┴──────────┘
      Peer collaboration
```

**When to use:**
- Complex problems needing both coordination and collaboration
- Teams of specialists
- Real-world organizational structures

---

## 💬 Communication Patterns

### 1. Message Passing

Agents send structured messages.

```python
from dataclasses import dataclass
from typing import Any, Optional
from enum import Enum

class MessageType(Enum):
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    QUESTION = "question"

@dataclass
class Message:
    """Standard message format"""
    type: MessageType
    from_agent: str
    to_agent: Optional[str]  # None = broadcast
    content: Any
    timestamp: datetime
    reply_to: Optional[str] = None
    priority: int = 0

class MessageBus:
    """Central message routing"""
    
    def __init__(self):
        self.messages = []
        self.subscribers = {}  # agent_id -> queue
    
    def send(self, message: Message):
        """Send message to recipient(s)"""
        self.messages.append(message)
        
        if message.to_agent:
            # Direct message
            self._deliver(message.to_agent, message)
        else:
            # Broadcast to all
            for agent_id in self.subscribers:
                if agent_id != message.from_agent:
                    self._deliver(agent_id, message)
    
    def subscribe(self, agent_id: str):
        """Agent subscribes to messages"""
        self.subscribers[agent_id] = []
    
    def receive(self, agent_id: str) -> List[Message]:
        """Agent gets its messages"""
        messages = self.subscribers[agent_id]
        self.subscribers[agent_id] = []  # Clear queue
        return messages
    
    def _deliver(self, agent_id: str, message: Message):
        """Deliver message to agent's queue"""
        if agent_id in self.subscribers:
            self.subscribers[agent_id].append(message)

# Usage
bus = MessageBus()

# Agent 1 sends request to Agent 2
bus.send(Message(
    type=MessageType.REQUEST,
    from_agent="research_agent",
    to_agent="coding_agent",
    content={
        "task": "implement web scraper",
        "requirements": "..."
    },
    timestamp=datetime.now()
))

# Agent 2 receives and responds
messages = bus.receive("coding_agent")
for msg in messages:
    if msg.type == MessageType.REQUEST:
        # Process and respond
        bus.send(Message(
            type=MessageType.RESPONSE,
            from_agent="coding_agent",
            to_agent=msg.from_agent,
            content={"code": "...", "status": "complete"},
            reply_to=msg.id,
            timestamp=datetime.now()
        ))
```

### 2. Shared Memory/Blackboard

Agents read and write to shared state.

```python
class Blackboard:
    """Shared knowledge base for agents"""
    
    def __init__(self):
        self.data = {}
        self.locks = {}
        self.history = []
    
    def write(self, key: str, value: Any, agent_id: str):
        """Agent writes to blackboard"""
        # Acquire lock
        with self._lock(key):
            old_value = self.data.get(key)
            self.data[key] = value
            
            # Log the change
            self.history.append({
                "action": "write",
                "key": key,
                "old_value": old_value,
                "new_value": value,
                "agent": agent_id,
                "timestamp": datetime.now()
            })
    
    def read(self, key: str) -> Any:
        """Agent reads from blackboard"""
        return self.data.get(key)
    
    def query(self, condition: Callable) -> Dict:
        """Find items matching condition"""
        return {
            k: v for k, v in self.data.items()
            if condition(k, v)
        }
    
    def subscribe(self, key: str, callback: Callable):
        """Agent subscribes to changes"""
        # Notify agent when key changes
        pass

# Usage
blackboard = Blackboard()

# Research agent writes findings
blackboard.write(
    key="company_info",
    value={"name": "TechCorp", "revenue": "10M"},
    agent_id="research_agent"
)

# Analysis agent reads and adds analysis
company_info = blackboard.read("company_info")
blackboard.write(
    key="analysis",
    value={"growth": "20% YoY", "rating": "Strong"},
    agent_id="analysis_agent"
)

# Writing agent combines all info
all_research = blackboard.query(
    lambda k, v: k.startswith("company") or k.startswith("analysis")
)
```

### 3. Request-Response

Simple synchronous interaction.

```python
class Agent:
    """Base agent with request-response"""
    
    def request(self, other_agent: 'Agent', request: dict) -> dict:
        """Send request and wait for response"""
        return other_agent.handle_request(request)
    
    def handle_request(self, request: dict) -> dict:
        """Handle incoming request"""
        # Override in subclass
        pass

# Example
class ResearchAgent(Agent):
    def handle_request(self, request: dict) -> dict:
        if request["type"] == "find_info":
            info = self.search(request["query"])
            return {"status": "success", "data": info}
        return {"status": "unknown_request"}

class WritingAgent(Agent):
    def write_article(self, topic: str):
        # Get research
        research_data = self.request(
            research_agent,
            {"type": "find_info", "query": topic}
        )
        
        # Write using the data
        article = self.compose(research_data["data"])
        return article
```

---

## 🎭 Role Specialization

### Defining Agent Roles

```python
from abc import ABC, abstractmethod

class AgentRole(ABC):
    """Base class for agent roles"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @property
    @abstractmethod
    def capabilities(self) -> List[str]:
        pass
    
    @property
    @abstractmethod
    def expertise_level(self) -> float:
        """0.0 to 1.0"""
        pass

class ResearcherRole(AgentRole):
    @property
    def name(self) -> str:
        return "Researcher"
    
    @property
    def capabilities(self) -> List[str]:
        return [
            "web_search",
            "data_collection",
            "source_validation",
            "fact_checking"
        ]
    
    @property
    def expertise_level(self) -> float:
        return 0.9  # Highly specialized

class CriticRole(AgentRole):
    @property
    def name(self) -> str:
        return "Critic"
    
    @property
    def capabilities(self) -> List[str]:
        return [
            "quality_assessment",
            "error_detection",
            "improvement_suggestions"
        ]
    
    @property
    def expertise_level(self) -> float:
        return 0.85

class SpecializedAgent:
    """Agent with a specific role"""
    
    def __init__(self, role: AgentRole, llm, tools):
        self.role = role
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        """Create role-specific system prompt"""
        return f"""
You are a {self.role.name} agent with the following capabilities:
{', '.join(self.role.capabilities)}

Your expertise level: {self.role.expertise_level * 100}%

Your role is to {self._get_role_description()}.

Available tools: {list(self.tools.keys())}

Always stay in your role and leverage your specialized capabilities.
"""
    
    def can_handle(self, task: str) -> float:
        """Rate ability to handle task (0-1)"""
        # Check if task matches capabilities
        task_lower = task.lower()
        matches = sum(
            1 for cap in self.role.capabilities
            if cap.replace('_', ' ') in task_lower
        )
        
        return min(
            matches / len(self.role.capabilities),
            self.role.expertise_level
        )
    
    def execute(self, task: str) -> dict:
        """Execute task using role expertise"""
        prompt = f"""
{self.system_prompt}

Task: {task}

Execute this task using your role's capabilities.
"""
        response = self.llm.complete(prompt)
        return {"result": response, "role": self.role.name}
```

---

## 🤝 Coordination Strategies

### 1. Task Delegation

```python
class TaskDelegator:
    """Intelligently assign tasks to agents"""
    
    def __init__(self, agents: List[SpecializedAgent]):
        self.agents = agents
        self.workload = {agent.role.name: 0 for agent in agents}
    
    def delegate(self, task: str) -> SpecializedAgent:
        """Find best agent for task"""
        
        # Score each agent
        scores = []
        for agent in self.agents:
            capability_score = agent.can_handle(task)
            workload_score = 1 - (self.workload[agent.role.name] / 10)
            
            # Combined score
            total_score = capability_score * 0.7 + workload_score * 0.3
            scores.append((total_score, agent))
        
        # Pick highest scoring agent
        scores.sort(reverse=True)
        best_agent = scores[0][1]
        
        # Update workload
        self.workload[best_agent.role.name] += 1
        
        return best_agent
    
    def complete_task(self, agent: SpecializedAgent):
        """Mark task as complete"""
        self.workload[agent.role.name] -= 1

# Usage
delegator = TaskDelegator([
    SpecializedAgent(ResearcherRole(), llm, research_tools),
    SpecializedAgent(CriticRole(), llm, critic_tools),
    SpecializedAgent(WriterRole(), llm, writing_tools)
])

task = "Find information about quantum computing"
agent = delegator.delegate(task)
result = agent.execute(task)
delegator.complete_task(agent)
```

### 2. Auction-Based Assignment

Agents "bid" on tasks.

```python
class TaskAuction:
    """Agents bid on tasks they want"""
    
    def auction_task(self, task: str, agents: List[Agent]) -> Agent:
        """Run auction for task assignment"""
        
        # Collect bids
        bids = []
        for agent in agents:
            bid = agent.bid_on_task(task)
            bids.append((bid, agent))
        
        # Winner is highest bidder
        bids.sort(reverse=True)
        winner = bids[0][1]
        
        print(f"Task '{task}' won by {winner.role.name} "
              f"with bid {bids[0][0]}")
        
        return winner

class BiddingAgent(SpecializedAgent):
    """Agent that bids on tasks"""
    
    def bid_on_task(self, task: str) -> float:
        """Calculate bid for task (0-1)"""
        
        # Base bid on capability match
        base_bid = self.can_handle(task)
        
        # Adjust based on current workload
        if self.current_workload > 5:
            base_bid *= 0.5  # Busy, lower bid
        
        # Adjust based on recent success rate
        base_bid *= self.recent_success_rate()
        
        return base_bid
```

### 3. Voting & Consensus

Agents vote on decisions.

```python
class ConsensusSystem:
    """Agents reach consensus through voting"""
    
    def __init__(self, agents: List[Agent]):
        self.agents = agents
    
    def propose_and_vote(self, proposal: dict) -> bool:
        """Propose action and vote"""
        
        print(f"\n📋 Proposal: {proposal['description']}")
        
        votes = []
        for agent in self.agents:
            vote = agent.vote_on_proposal(proposal)
            votes.append((agent.role.name, vote))
            print(f"  {agent.role.name}: {'✅ Yes' if vote else '❌ No'}")
        
        # Simple majority
        yes_votes = sum(1 for _, vote in votes if vote)
        threshold = len(self.agents) / 2
        
        approved = yes_votes > threshold
        print(f"\n{'✅ APPROVED' if approved else '❌ REJECTED'} "
              f"({yes_votes}/{len(self.agents)} votes)")
        
        return approved
    
    def debate_until_consensus(
        self,
        topic: str,
        max_rounds: int = 3
    ) -> dict:
        """Discuss until consensus"""
        
        for round in range(max_rounds):
            print(f"\n🗣️ Round {round + 1}")
            
            # Each agent shares perspective
            perspectives = []
            for agent in self.agents:
                view = agent.give_opinion(topic)
                perspectives.append({
                    "agent": agent.role.name,
                    "opinion": view
                })
            
            # Try to find consensus
            proposal = self.synthesize_proposal(perspectives)
            
            if self.propose_and_vote(proposal):
                return proposal
        
        # No consensus reached
        return {"status": "no_consensus", "perspectives": perspectives}
```

---

## 🔄 Multi-Agent Workflow Patterns

### Pattern 1: Sequential Pipeline

```python
class AgentPipeline:
    """Agents process in sequence"""
    
    def __init__(self, agents: List[Agent]):
        self.agents = agents
    
    def execute(self, initial_input: Any) -> Any:
        """Pass output through agent chain"""
        
        current_output = initial_input
        
        for i, agent in enumerate(self.agents):
            print(f"\n📍 Stage {i+1}: {agent.role.name}")
            
            try:
                current_output = agent.process(current_output)
                print(f"✅ Output: {str(current_output)[:100]}...")
            except Exception as e:
                print(f"❌ Failed at {agent.role.name}: {e}")
                raise
        
        return current_output

# Example: Content creation pipeline
pipeline = AgentPipeline([
    ResearchAgent(),    # Gather information
    WritingAgent(),     # Draft content
    EditorAgent(),      # Refine and improve
    CriticAgent(),      # Final review
    PublishAgent()      # Publish
])

final_content = pipeline.execute("Write about AI agents")
```

### Pattern 2: Parallel + Merge

```python
import asyncio

class ParallelAgentSystem:
    """Agents work in parallel, results merged"""
    
    def __init__(self, agents: List[Agent], merger: Agent):
        self.agents = agents
        self.merger = merger
    
    async def execute_parallel(self, task: str) -> dict:
        """Execute task with all agents in parallel"""
        
        print(f"🚀 Launching {len(self.agents)} agents in parallel...")
        
        # Launch all agents
        tasks = [
            agent.execute_async(task)
            for agent in self.agents
        ]
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle results
        successful = []
        failed = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed.append((self.agents[i].role.name, result))
            else:
                successful.append({
                    "agent": self.agents[i].role.name,
                    "result": result
                })
        
        print(f"✅ {len(successful)} succeeded, ❌ {len(failed)} failed")
        
        # Merge successful results
        merged = self.merger.merge_results(successful)
        
        return {
            "merged_result": merged,
            "individual_results": successful,
            "failures": failed
        }

# Example: Multi-source research
research_system = ParallelAgentSystem(
    agents=[
        WebResearchAgent(),
        AcademicResearchAgent(),
        NewsResearchAgent()
    ],
    merger=SynthesisAgent()
)

comprehensive_research = await research_system.execute_parallel(
    "Latest developments in quantum computing"
)
```

### Pattern 3: Debate & Refine

```python
class DebateSystem:
    """Agents debate to improve output"""
    
    def __init__(self, proposer: Agent, critics: List[Agent]):
        self.proposer = proposer
        self.critics = critics
    
    def debate(self, topic: str, rounds: int = 3) -> dict:
        """Iterative debate to refine solution"""
        
        # Initial proposal
        proposal = self.proposer.generate_proposal(topic)
        history = [{"round": 0, "proposal": proposal, "critiques": []}]
        
        for round_num in range(1, rounds + 1):
            print(f"\n🔄 Round {round_num}")
            
            # Critics review
            critiques = []
            for critic in self.critics:
                critique = critic.critique(proposal, topic)
                critiques.append({
                    "critic": critic.role.name,
                    "critique": critique
                })
                print(f"  💬 {critic.role.name}: {critique['summary']}")
            
            # Proposer refines based on feedback
            proposal = self.proposer.refine_proposal(
                proposal,
                critiques
            )
            
            history.append({
                "round": round_num,
                "proposal": proposal,
                "critiques": critiques
            })
            
            # Check if critics are satisfied
            if all(c['satisfied'] for c in critiques):
                print("\n✅ Consensus reached!")
                break
        
        return {
            "final_proposal": proposal,
            "history": history,
            "rounds": round_num
        }

# Example: Code review debate
debate = DebateSystem(
    proposer=CodingAgent(),
    critics=[
        SecurityCriticAgent(),
        PerformanceCriticAgent(),
        ReadabilityCriticAgent()
    ]
)

refined_code = debate.debate("Implement user authentication system")
```

---

## 🛠️ Complete Implementation: Multi-Agent System

```python
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import asyncio
import logging

logger = logging.getLogger(__name__)

@dataclass
class AgentConfig:
    """Configuration for an agent"""
    name: str
    role: str
    capabilities: List[str]
    max_concurrent_tasks: int = 1
    timeout: int = 300

class BaseMultiAgent(ABC):
    """Base class for agents in multi-agent system"""
    
    def __init__(self, config: AgentConfig, llm, tools):
        self.config = config
        self.llm = llm
        self.tools = tools
        self.current_tasks = 0
    
    @abstractmethod
    def execute(self, task: str, context: dict) -> dict:
        """Execute a task"""
        pass
    
    def can_handle(self, task: str) -> float:
        """Assess ability to handle task (0-1)"""
        score = 0.0
        task_lower = task.lower()
        
        for capability in self.config.capabilities:
            if capability.lower() in task_lower:
                score += 0.3
        
        return min(score, 1.0)
    
    def is_available(self) -> bool:
        """Check if agent can take more tasks"""
        return self.current_tasks < self.config.max_concurrent_tasks

class MultiAgentOrchestrator:
    """Coordinates multiple agents"""
    
    def __init__(self, agents: List[BaseMultiAgent]):
        self.agents = {agent.config.name: agent for agent in agents}
        self.message_bus = MessageBus()
        self.blackboard = Blackboard()
        self.task_queue = []
        
        # Initialize communication
        for agent_name in self.agents:
            self.message_bus.subscribe(agent_name)
    
    def execute_goal(self, goal: str) -> dict:
        """Execute a complex goal using multiple agents"""
        
        logger.info(f"Starting multi-agent execution: {goal}")
        
        # 1. Plan: Decompose goal into tasks
        plan = self._create_plan(goal)
        logger.info(f"Created plan with {len(plan.tasks)} tasks")
        
        # 2. Delegate: Assign tasks to agents
        assignments = self._assign_tasks(plan.tasks)
        
        # 3. Execute: Run tasks (respecting dependencies)
        results = self._execute_plan(assignments)
        
        # 4. Synthesize: Combine results
        final_output = self._synthesize_results(results, goal)
        
        return final_output
    
    def _create_plan(self, goal: str) -> 'Plan':
        """Create execution plan"""
        
        # Use a planner agent or LLM
        manager = self.agents.get('manager')
        if manager:
            return manager.create_plan(goal, list(self.agents.keys()))
        
        # Fallback: simple decomposition
        return self._simple_decomposition(goal)
    
    def _assign_tasks(self, tasks: List['Task']) -> Dict[str, List['Task']]:
        """Assign tasks to best agents"""
        
        assignments = {name: [] for name in self.agents}
        
        for task in tasks:
            # Score each agent
            scores = []
            for agent_name, agent in self.agents.items():
                if agent.is_available():
                    score = agent.can_handle(task.description)
                    scores.append((score, agent_name))
            
            if scores:
                # Assign to highest scoring available agent
                scores.sort(reverse=True)
                best_agent = scores[0][1]
                assignments[best_agent].append(task)
                logger.info(f"Assigned '{task.description}' to {best_agent}")
            else:
                logger.warning(f"No available agent for task: {task.description}")
                self.task_queue.append(task)
        
        return assignments
    
    def _execute_plan(self, assignments: Dict) -> List[dict]:
        """Execute all assignments"""
        
        results = []
        
        # Execute tasks for each agent
        for agent_name, tasks in assignments.items():
            if not tasks:
                continue
            
            agent = self.agents[agent_name]
            
            for task in tasks:
                try:
                    # Check dependencies
                    if not self._dependencies_met(task, results):
                        logger.info(f"Waiting for dependencies: {task.description}")
                        continue
                    
                    # Get context from blackboard
                    context = self._get_task_context(task)
                    
                    # Execute
                    agent.current_tasks += 1
                    result = agent.execute(task.description, context)
                    agent.current_tasks -= 1
                    
                    # Store result
                    results.append({
                        "task": task,
                        "agent": agent_name,
                        "result": result,
                        "success": True
                    })
                    
                    # Update blackboard
                    self.blackboard.write(
                        key=f"result_{task.id}",
                        value=result,
                        agent_id=agent_name
                    )
                    
                    logger.info(f"✅ {agent_name} completed: {task.description}")
                    
                except Exception as e:
                    logger.error(f"❌ {agent_name} failed: {e}")
                    results.append({
                        "task": task,
                        "agent": agent_name,
                        "error": str(e),
                        "success": False
                    })
        
        return results
    
    def _dependencies_met(self, task: 'Task', results: List[dict]) -> bool:
        """Check if task dependencies are complete"""
        if not task.dependencies:
            return True
        
        completed_task_ids = {
            r['task'].id for r in results if r.get('success')
        }
        
        return all(dep in completed_task_ids for dep in task.dependencies)
    
    def _get_task_context(self, task: 'Task') -> dict:
        """Get relevant context for task"""
        
        # Get dependency results
        dep_results = {}
        for dep_id in (task.dependencies or []):
            result = self.blackboard.read(f"result_{dep_id}")
            if result:
                dep_results[dep_id] = result
        
        return {
            "dependencies": dep_results,
            "shared_knowledge": self.blackboard.query(
                lambda k, v: "knowledge" in k
            )
        }
    
    def _synthesize_results(self, results: List[dict], goal: str) -> dict:
        """Combine agent results into final output"""
        
        # Check for failures
        failures = [r for r in results if not r.get('success')]
        if failures:
            logger.warning(f"⚠️ {len(failures)} tasks failed")
        
        successful = [r for r in results if r.get('success')]
        
        # Get synthesis agent
        synthesizer = self.agents.get('synthesizer')
        if synthesizer:
            return synthesizer.synthesize(successful, goal)
        
        # Simple combination
        return {
            "goal": goal,
            "success": len(failures) == 0,
            "results": successful,
            "failures": failures
        }

# Example specialized agents
class ResearchMultiAgent(BaseMultiAgent):
    """Research specialist"""
    
    def execute(self, task: str, context: dict) -> dict:
        # Research implementation
        search_results = self.tools['web_search'].execute(task)
        
        return {
            "findings": search_results,
            "sources": [r['url'] for r in search_results],
            "summary": self._summarize(search_results)
        }

class CodingMultiAgent(BaseMultiAgent):
    """Coding specialist"""
    
    def execute(self, task: str, context: dict) -> dict:
        # Code generation implementation
        code = self.llm.complete(f"Write code for: {task}")
        
        # Test the code
        test_result = self.tools['code_executor'].execute(code)
        
        return {
            "code": code,
            "tests_passed": test_result['success'],
            "output": test_result['output']
        }

class QAMultiAgent(BaseMultiAgent):
    """Quality assurance specialist"""
    
    def execute(self, task: str, context: dict) -> dict:
        # Review previous agent's work
        artifact = context.get('dependencies', {}).get('artifact')
        
        issues = self._find_issues(artifact)
        
        return {
            "quality_score": self._calculate_score(issues),
            "issues_found": issues,
            "approved": len(issues) == 0
        }
```

---

## 🎯 Real-World Example: Software Development Team

```python
# Create specialized agents
manager = BaseMultiAgent(
    AgentConfig(
        name="manager",
        role="project_manager",
        capabilities=["planning", "coordination", "synthesis"]
    ),
    llm=llm,
    tools=[]
)

researcher = ResearchMultiAgent(
    AgentConfig(
        name="researcher",
        role="researcher",
        capabilities=["research", "data_gathering", "analysis"]
    ),
    llm=llm,
    tools=research_tools
)

coder = CodingMultiAgent(
    AgentConfig(
        name="coder",
        role="developer",
        capabilities=["coding", "testing", "debugging"]
    ),
    llm=llm,
    tools=coding_tools
)

qa = QAMultiAgent(
    AgentConfig(
        name="qa",
        role="quality_assurance",
        capabilities=["testing", "review", "validation"]
    ),
    llm=llm,
    tools=qa_tools
)

# Create orchestrator
orchestrator = MultiAgentOrchestrator([
    manager,
    researcher,
    coder,
    qa
])

# Execute complex goal
result = orchestrator.execute_goal(
    "Build a web scraper for news articles with error handling"
)

print(result)
```

---

## 🌐 Popular Multi-Agent Frameworks

### 1. CrewAI

```python
from crewai import Agent, Task, Crew

# Define agents
researcher = Agent(
    role="Research Analyst",
    goal="Find accurate information",
    backstory="Expert at finding reliable sources"
)

writer = Agent(
    role="Content Writer",
    goal="Create engaging content",
    backstory="Skilled at storytelling"
)

# Define tasks
research_task = Task(
    description="Research AI agents",
    agent=researcher
)

writing_task = Task(
    description="Write article based on research",
    agent=writer
)

# Create crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    verbose=True
)

# Execute
result = crew.kickoff()
```

### 2. AutoGen

```python
from autogen import AssistantAgent, UserProxyAgent

# Create agents
assistant = AssistantAgent(
    name="assistant",
    llm_config={"model": "gpt-4"}
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    code_execution_config={"work_dir": "coding"}
)

# Start conversation
user_proxy.initiate_chat(
    assistant,
    message="Build a calculator app"
)
```

### 3. LangGraph

```python
from langgraph.graph import StateGraph

# Define agent workflow
workflow = StateGraph()

workflow.add_node("research", research_agent)
workflow.add_node("code", coding_agent)
workflow.add_node("review", review_agent)

workflow.add_edge("research", "code")
workflow.add_edge("code", "review")

# Compile and run
app = workflow.compile()
result = app.invoke({"task": "Build API"})
```

---

## ⚠️ Common Pitfalls

### 1. Over-Communication Overhead

**Problem:** Agents spend more time talking than working

**Solution:**
```python
# Batch messages
def batch_communicate(agents, interval=5):
    """Batch messages to reduce overhead"""
    message_buffer = []
    
    while True:
        # Collect messages for interval
        time.sleep(interval)
        
        # Process batch
        for msg in message_buffer:
            deliver(msg)
        
        message_buffer = []
```

### 2. Circular Dependencies

**Problem:** Agent A waits for B, B waits for A

**Solution:** Dependency graph validation
```python
def validate_dependencies(tasks):
    """Check for circular dependencies"""
    visited = set()
    path = set()
    
    def has_cycle(task_id):
        if task_id in path:
            return True
        if task_id in visited:
            return False
        
        visited.add(task_id)
        path.add(task_id)
        
        task = get_task(task_id)
        for dep in task.dependencies:
            if has_cycle(dep):
                return True
        
        path.remove(task_id)
        return False
    
    return any(has_cycle(t.id) for t in tasks)
```

### 3. Agent Conflicts

**Problem:** Agents disagree and block progress

**Solution:** Conflict resolution protocol
```python
class ConflictResolver:
    def resolve(self, conflicting_agents, issue):
        # 1. Try consensus
        if self.try_consensus(conflicting_agents, issue):
            return "consensus"
        
        # 2. Escalate to manager
        if self.manager_available():
            return self.manager_decides(issue)
        
        # 3. Majority vote
        return self.vote(conflicting_agents, issue)
```

---

## 💡 Key Takeaways

1. **Specialization beats generalization** - Focused agents outperform jack-of-all-trades
2. **Communication is critical** - Choose the right pattern for your use case
3. **Coordination overhead is real** - More agents ≠ always better
4. **Start simple** - Begin with 2-3 agents, scale as needed
5. **Design for failure** - Agents will conflict and fail
6. **Clear roles prevent chaos** - Well-defined responsibilities are essential
7. **Different architectures for different problems** - Hierarchical vs flat vs hybrid

---

## 🚀 Hands-On Challenge

### Build a Multi-Agent Content Creation System

Create a system with:

1. **Research Agent** - Gathers information
2. **Writing Agent** - Creates draft
3. **Editor Agent** - Improves quality
4. **Fact-Checker Agent** - Verifies claims
5. **Manager Agent** - Coordinates everyone

**Requirements:**
- Agents communicate via message bus
- Sequential pipeline execution
- Error handling and recovery
- Quality metrics at each stage

**Starter code:**
```python
# Your implementation here
class ContentCreationSystem:
    def __init__(self):
        self.agents = [
            ResearchAgent(),
            WritingAgent(),
            EditorAgent(),
            FactCheckerAgent()
        ]
        self.manager = ManagerAgent(self.agents)
    
    def create_content(self, topic: str) -> dict:
        return self.manager.orchestrate(topic)

# Test it
system = ContentCreationSystem()
article = system.create_content("Future of AI Agents")
```

---

## 📚 Further Learning

### Essential Reading
- **Multi-Agent Systems** - Wooldridge textbook
- **CrewAI Documentation** - Practical multi-agent patterns
- **AutoGen Papers** - Multi-agent conversations

### Videos
- [Multi-Agent Systems Explained](https://youtube.com)
- [Building Agent Teams](https://deeplearning.ai)

### Frameworks to Explore
- **CrewAI** - Role-playing agents
- **AutoGen** - Conversational agents
- **LangGraph** - Workflow orchestration
- **Semantic Kernel** - Multi-agent planning

---

## ➡️ What's Next?

You now understand how to build systems where multiple specialized agents collaborate!

**Next Lesson:** [Production Deployment](09-production-deployment.md)

Learn how to take your multi-agent systems to production with monitoring, scaling, and reliability!

---

## 🤔 Reflection Questions

1. **When would a single agent be better than multiple agents?**
2. **How do you prevent agents from talking in circles?**
3. **What's the right granularity for agent specialization?**
4. **How do you handle one agent holding up the entire system?**
5. **What metrics would you track in a multi-agent system?**

---

## 📝 Your Notes

*Your insights on multi-agent systems:*

**Use cases for my projects:**
- 

**Agent team I want to build:**
- 

**Communication patterns that make sense:**
- 

**Concerns about coordination:**
- 

**Questions:**
- 

---

**Lesson Status:** ✅ Complete  
**Estimated Time:** 90-120 minutes  
**Difficulty:** ⭐⭐⭐⭐ (Advanced)  
**Prerequisites:** Lessons 1-7  
**Next:** Production Deployment 🚀
